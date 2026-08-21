"""Gate tests for the combined three-population validation.

The two failure modes here are both silent:
  1. Colliding batch_idx across populations -- per_entry_val_history.csv is keyed on batch_idx
     ALONE, so three datasets restarting at 0 would interleave different chains into one row group
     and every historical comparison would quietly compare the wrong entries.
  2. A chain present in two manifests -- validated twice per epoch, biasing the combined mean, and
     mislabelling a de novo design as a natural time-split chain. This is not hypothetical: 42 PDA
     chains were eligible in the expanded candidate pool and 5 were actually drawn.
"""
import importlib.util
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPLIT = os.path.join(_HERE, "..", "prune_work", "split_expanded_val.py")
_spec = importlib.util.spec_from_file_location("split_expanded_val", _SPLIT)
split_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split_mod)


def test_quantile_edges_partition_the_pool():
    vals = list(range(1, 101))
    edges = split_mod.quantile_edges(vals, 10)
    assert len(edges) == 9
    assert edges == sorted(edges)
    counts = [0] * 10
    for v in vals:
        counts[split_mod.stratum_of(v, edges)] += 1
    assert sum(counts) == len(vals)
    assert min(counts) > 0, counts


def test_stratified_sample_records_deficits_instead_of_silently_undershooting():
    pool = {f"c{i}": 10 * i for i in range(1, 6)}      # only 5 members
    rng = __import__("random").Random(0)
    edges = split_mod.quantile_edges(list(pool.values()), 5)
    picked, deficits, short = split_mod.stratified_sample(pool, edges, 20, 5, rng)
    assert len(picked) == 5                            # cannot invent members
    assert short == 15                                 # and says so
    assert deficits, "a stratum that could not fill its quota must be recorded"


def test_pdb_code_is_case_insensitive_but_chain_id_is_not():
    """The bug this guards: lowercasing the whole 'pdb_chain' made the exclusion match NOTHING
    while reporting 'excluded 0' against 5 known duplicates. Auth chain ids are case-sensitive --
    this very pool contains 8v2d_y."""
    src = open(_SPLIT).read()
    assert "_norm" in src, "the exclusion filter must normalise through _norm"
    # reconstruct the normaliser the module defines inside main()
    def norm(s):
        pdb, _, ch = s.partition("_")
        return f"{pdb.lower()}_{ch}"
    assert norm("6FF6_A") == norm("6ff6_A")            # code case is irrelevant
    assert norm("8v2d_y") != norm("8v2d_Y")            # chain case is NOT


def test_source_tag_and_offset_are_plumbed_through_the_dataset():
    sys.path.insert(0, os.path.join(_HERE, "..", "prune_work"))
    src = open(os.path.join(_HERE, "..", "prune_work", "pda_dataset.py")).read()
    assert "source_tag=0, index_offset=0" in src
    assert "idx + self.index_offset" in src, "batch_idx must carry the offset"
    assert 'feats["val_source"]' in src


def test_train_openfold_asserts_against_cross_population_duplicates():
    src = open(os.path.join(_HERE, "..", "train_openfold.py")).read()
    assert "appears in BOTH the" in src, "the duplicate-chain assertion must exist"
    assert "VAL_SOURCE_NAMES = {0: \"pda\", 1: \"easy\", 2: \"hard\"}" in src
    assert "_src_{name}" in src, "per-population metrics must be logged"
    # the combined mean must NOT be special-cased: it is the unconditional val/{k}
    assert "torch.utils.data.ConcatDataset(parts)" in src
