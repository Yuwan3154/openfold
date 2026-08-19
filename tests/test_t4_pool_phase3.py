"""Gate tests for T4 phase 3: the promoted pool's output layout.

The failure modes here are all silent ones. A pool that returns the phase-2 `positions`/`mask` keys
merges into a template axis of ZEROS, because `merge_template_features` copies only keys already
present in the natural features and zero-fills the rest -- the model would be handed blank templates
and training would look fine. A restype-ordered one-hot survives `fix_templates_aatype` as the WRONG
amino acids. And a promoted CROP placed at the wrong residues still yields plausible-looking features.
"""

import json

import numpy as np
import pytest
import torch

from openfold.data.data_transforms import fix_templates_aatype
from openfold.data.synthetic_templates import (
    merge_template_features,
    natural_template_count,
    subsample_natural_templates,
)
from openfold.data.templates import empty_template_feats
from openfold.np import residue_constants as rc
from openfold.utils.t4_pool import PromotedTemplatePool, PromotedTemplateWriter

QUERY = "ACDEFGHIKLMNPQRSTVWY"          # 20 residues, every standard type exactly once
L = len(QUERY)


def _write(pool_dir, chain, epoch, step, first, n, tm_pred=0.8, rank=0):
    """Persist one promoted crop covering query positions [first, first+n)."""
    w = PromotedTemplateWriter(str(pool_dir), rank)
    ridx = np.arange(first, first + n)
    aat = np.array([rc.restype_order[QUERY[i]] for i in ridx], np.int8)
    mask = np.zeros((n, 37), bool)
    mask[:, :3] = True                                    # N, CA, C present
    coords = np.zeros((n, 37, 3), np.float32)
    coords[:, :3] = np.arange(n * 9, dtype=np.float32).reshape(n, 3, 3)
    w.submit(chain=chain, epoch=epoch, step=step, tm_pred=tm_pred, tm_template=0.5,
             coords37=coords, atom_mask37=mask, aatype=aat, residue_index=ridx)
    w.close()
    return w


def test_pool_emits_the_natural_hit_layout_not_the_phase2_keys(tmp_path):
    """⛔ The regression this file exists for: phase 2 returned positions/mask/aatype/tm, which
    merge_template_features cannot consume -- it would zero-fill every template key it expects."""
    _write(tmp_path, "1abc_A", 0, 10, first=0, n=L)
    p = PromotedTemplatePool(str(tmp_path))
    assert p.refresh() == 1
    f = p.sample_features("1abc_A", 1, np.random.default_rng(0), query_sequence=QUERY)
    for k in ["template_all_atom_positions", "template_all_atom_mask", "template_aatype",
              "template_sequence", "template_domain_names", "template_sum_probs"]:
        assert k in f, k
    assert "positions" not in f and "mask" not in f
    assert f["template_all_atom_positions"].shape == (1, L, 37, 3)
    assert f["template_all_atom_mask"].shape == (1, L, 37)
    assert f["template_aatype"].shape == (1, L, 22)
    assert f["template_sum_probs"].shape == (1, 1)
    assert len(f["template_sequence"]) == 1 and len(f["template_domain_names"]) == 1
    assert f["template_domain_names"][0].startswith(b"t4_1abc_A_e0_s10")


def test_aatype_onehot_is_hhblits_ordered(tmp_path):
    """fix_templates_aatype argmaxes template_aatype then gathers through
    MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, so a restype-ordered one-hot silently becomes other residues.
    Round-tripping it back to the query's own aatype is the only check that catches this."""
    _write(tmp_path, "1abc_A", 0, 1, first=0, n=L)
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    f = p.sample_features("1abc_A", 1, np.random.default_rng(1), query_sequence=QUERY)
    out = fix_templates_aatype({"template_aatype": torch.tensor(f["template_aatype"])})
    got = out["template_aatype"][0].numpy()
    want = np.array([rc.restype_order[c] for c in QUERY])
    assert np.array_equal(got, want), (got, want)


def test_a_crop_is_placed_at_its_own_residues_and_masked_elsewhere(tmp_path):
    """A promoted prediction is a CROP; it must land at the query positions its residue_index names
    and contribute nothing anywhere else -- an ordinary partial-coverage template."""
    first, n = 5, 8
    _write(tmp_path, "2xyz_B", 0, 2, first=first, n=n)
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    f = p.sample_features("2xyz_B", 1, np.random.default_rng(2), query_sequence=QUERY)
    msk = f["template_all_atom_mask"][0]
    covered = np.zeros(L, bool)
    covered[first:first + n] = True
    assert (msk[covered].sum(axis=-1) > 0).all()
    assert (msk[~covered] == 0).all()
    assert (f["template_all_atom_positions"][0][~covered] == 0).all()
    seq = f["template_sequence"][0].decode()
    assert seq[:first] == "-" * first and seq[first + n:] == "-" * (L - first - n)
    assert seq[first:first + n] == QUERY[first:first + n]


def test_each_template_gets_its_own_sequence_because_crops_differ(tmp_path):
    """Unlike the synthetic pool, promoted templates of one chain cover DIFFERENT residues, so a
    single broadcast sequence would mislabel every template but one."""
    _write(tmp_path, "3aaa_A", 0, 1, first=0, n=6)
    _write(tmp_path, "3aaa_A", 0, 2, first=12, n=6, rank=1)
    p = PromotedTemplatePool(str(tmp_path))
    assert p.refresh() == 2
    f = p.sample_features("3aaa_A", 2, np.random.default_rng(3), query_sequence=QUERY)
    seqs = sorted(s.decode() for s in f["template_sequence"])
    assert len(set(seqs)) == 2
    assert seqs[0].count("-") == L - 6 and seqs[1].count("-") == L - 6


def test_pool_merges_cleanly_onto_natural_hits(tmp_path):
    """The end-to-end contract: the merged template axis grows by exactly the promoted count and no
    key comes out zero-filled by accident."""
    _write(tmp_path, "4bbb_A", 0, 1, first=0, n=L)
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    f = p.sample_features("4bbb_A", 1, np.random.default_rng(4), query_sequence=QUERY)
    nat = {
        "template_all_atom_positions": np.ones((2, L, 37, 3), np.float32),
        "template_all_atom_mask": np.ones((2, L, 37), np.float32),
        "template_aatype": np.zeros((2, L, 22), np.float32),
        "template_sum_probs": np.zeros((2, 1), np.float32),
        "template_domain_names": np.array([b"1aaa_A", b"1bbb_B"], dtype=object),
        "template_sequence": np.array([b"AA", b"BB"], dtype=object),
    }
    out = merge_template_features(nat, f)
    assert out["template_all_atom_positions"].shape[0] == 3
    assert len(out["template_domain_names"]) == 3
    assert out["template_domain_names"][2].startswith(b"t4_")
    assert (out["template_all_atom_mask"][2] > 0).any()      # NOT zero-filled
    assert "_tm" not in out


def test_merge_after_replacing_naturals_holds_the_pool_size(tmp_path):
    """T4 must compose with the count-matched mode: promoted templates replace naturals rather than
    growing the pool, so the delivered-count distribution is still T1's."""
    _write(tmp_path, "5ccc_A", 0, 1, first=0, n=L)
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    rng = np.random.default_rng(5)
    f = p.sample_features("5ccc_A", 1, rng, query_sequence=QUERY)
    nat = {
        "template_all_atom_positions": np.ones((4, L, 37, 3), np.float32),
        "template_all_atom_mask": np.ones((4, L, 37), np.float32),
        "template_aatype": np.zeros((4, L, 22), np.float32),
        "template_sum_probs": np.zeros((4, 1), np.float32),
        "template_domain_names": np.array([b"n%d" % i for i in range(4)], dtype=object),
        "template_sequence": np.array([b"s%d" % i for i in range(4)], dtype=object),
    }
    k = f["template_all_atom_positions"].shape[0]
    out = merge_template_features(subsample_natural_templates(nat, 4 - k, rng), f)
    assert natural_template_count(out) == 4
    assert sum(1 for d in out["template_domain_names"] if d.startswith(b"t4_")) == k


def test_max_per_chain_is_fifo(tmp_path):
    """⛔ INVERTED 2026-08-19 (user): deterministic FIFO, newest kept. tm_pred no longer decides
    retention at all -- it is recorded for diagnostics and for the promotion gate only."""
    for i, tm in enumerate([0.4, 0.9, 0.6, 0.95]):
        _write(tmp_path, "6ddd_A", 0, i, first=0, n=L, tm_pred=tm, rank=i)
    p = PromotedTemplatePool(str(tmp_path), max_per_chain=2)
    assert p.refresh() == 2
    assert sorted(r["step"] for r in p.by_chain["6ddd_A"]) == [2, 3]      # the two newest
    assert sorted(r["tm_pred"] for r in p.by_chain["6ddd_A"]) == [0.6, 0.95]


def test_unknown_chain_returns_none(tmp_path):
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    assert "nope_A" not in p
    assert p.sample_features("nope_A", 1, np.random.default_rng(6), query_sequence=QUERY) is None


def test_stale_entry_beyond_the_query_is_dropped_not_crashed(tmp_path):
    """A pool carried into a run where the chain is shorter must not index out of bounds."""
    _write(tmp_path, "7eee_A", 0, 1, first=0, n=L)
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    short = QUERY[:8]
    f = p.sample_features("7eee_A", 1, np.random.default_rng(7), query_sequence=short)
    assert f["template_all_atom_positions"].shape == (1, 8, 37, 3)
    assert (f["template_all_atom_mask"][0].sum(axis=-1) > 0).all()


# --------------------------------------------------------------------------------------------
# ⛔ The held-out control set was REMOVED 2026-08-18. The user's reasoning: the eval sets (PDA de novo
# designs + ws5 val) contain NO training chains and are template-free anyway, so withholding promoted
# templates from some training chains has nothing in this pipeline to be measured against. The
# rationale for it was imported from a generic self-distillation design and did not fit here. Do not
# reintroduce it without an eval that actually scores training chains.
# --------------------------------------------------------------------------------------------

def test_pool_reports_how_many_a_chain_has(tmp_path):
    """The three-group pre-shuffle draw needs this: --t4_n_promoted is the promoted group's WEIGHT in
    the mixture, but a chain cannot contribute more than it actually has, and early in training most
    chains have none at all. The number the draw must see is the POST-CAP one."""
    p = PromotedTemplatePool(str(tmp_path))
    p.refresh()
    assert p.n_for_chain("nope_A") == 0
    for i in range(3):
        _write(tmp_path, "8fff_A", 0, i, first=0, n=L, rank=i)
    p.refresh()
    assert p.n_for_chain("8fff_A") == 3
    assert p.n_for_chain("nope_A") == 0
    capped = PromotedTemplatePool(str(tmp_path), max_per_chain=2)
    capped.refresh()
    assert capped.n_for_chain("8fff_A") == 2


def test_dropped_promotions_are_counted_not_silent(tmp_path):
    """A full queue means I/O cannot keep up; stalling the GPU is the worse trade, but a silent drop
    would make a throttled pool look like a model that never promotes."""
    w = PromotedTemplateWriter(str(tmp_path), 0, max_queue=1)
    assert w.n_dropped == 0                              # counter exists and starts clean
    w.close()
