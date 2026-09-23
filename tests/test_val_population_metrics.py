"""val_pop/*: exact all-rank per-population validation means, from one all_gather_object + dedup.

⛔⛔ THE BUG THIS REPLACES (RAW Phase M §6(3)): val/{metric}_{group} were synced by Lightning 2.5.1 key by
key in each rank's dict INSERTION order, and a group key is created when a rank first sees a member, so ranks
0/1 synced nonneural where ranks 2/3 synced train_overlap: CROSS-PAIRED. The 'mean' also averages the int64
batch counter by truncating division. A group absent from one rank's shard would have skipped that key's
collectives on that rank alone: a DDP desync.

The gloo simulations reproduce those conditions on 4 CPU ranks, with the REAL DistributedSampler padding
(duplicates of the first entries on the last ranks). The known-bad control replays the removed logging
through Lightning's own _ResultCollection + _sync_ddp and must show the published bug signature.
"""

import os

import pytest
import torch
from lightning_fabric.utilities.distributed import _sync_ddp_if_available
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from torch.utils.data.distributed import DistributedSampler

from openfold.utils import allrank_metrics
from openfold.utils.allrank_metrics import (
    dedup_entries,
    gather_records,
    population_means,
    population_records,
)
from tests.gloo_harness import record_collectives, run_ranks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = 4  # the A6000's DDP world size
SOURCE_NAMES = {0: "pda", 1: "easy", 2: "hard"}  # train_openfold.VAL_SOURCE_NAMES (checked by the wiring test)
GROUPS = ("all", "train_overlap", "held_out", "nonneural", "neural_gated", "src_pda", "src_easy", "src_hard")
METRICS = ("lddt_ca", "drmsd_ca", "alignment_rmsd", "recall_2A", "gdt_ts", "gdt_ha")

# entry -> (val_source, is_train_overlap, in_nonneural_subset or None for non-PDA entries)
# A: 14 entries, so DistributedSampler pads 16 with entries 0 and 1 (onto ranks 2 and 3); rank 0 meets
#    nonneural before train_overlap, rank 2 the reverse, and rank 3 holds no nonneural entry at all.
SCENARIO_A = {0: (0, 0, 1), 1: (0, 0, 0), 2: (0, 1, 0), 3: (1, 0, None), 4: (0, 1, 0), 5: (2, 0, None),
              6: (1, 0, None), 7: (2, 1, None), 8: (0, 0, 1), 9: (0, 0, 1), 10: (0, 0, 1), 11: (1, 0, None),
              12: (2, 0, None), 13: (0, 0, 0)}
# B: 16 entries, every rank holds every group, ranks 0/1 and 2/3 meet them in different orders (Run C v2's
#    shape): the condition under which the old per-key sync cross-pairs without a count mismatch.
SCENARIO_B = {}
for _r in range(WORLD):
    _order = ([(0, 0, 1), (0, 1, 0), (1, 0, None), (2, 0, None)] if _r < 2 else
              [(0, 1, 0), (0, 0, 1), (1, 0, None), (2, int(_r == 3), None)])
    for _p, _e in enumerate(_order):
        SCENARIO_B[_r + WORLD * _p] = _e


def _value(idx, metric, rank):
    # dyadic, so float64 sums are exact in any order; a padded duplicate (off its home rank) differs
    return (idx + 1) / 16 + METRICS.index(metric) / 256 + (rank - idx % WORLD) / 1024


def _rank_entries(scenario, rank):
    return list(DistributedSampler(range(len(scenario)), num_replicas=WORLD, rank=rank, shuffle=False))


def _batch(scenario, idx, rank):
    """One validation batch (batch size 1) as `_log` sees it: recycling dim stripped."""
    src, overlap, nonneural = scenario[idx]
    batch = {"batch_idx": torch.tensor([idx]), "is_train_overlap": torch.tensor([overlap]),
             "val_source": torch.tensor([src])}
    if nonneural is not None:
        batch["in_nonneural_subset"] = torch.tensor([nonneural])
    metrics = {m: torch.tensor([_value(idx, m, rank)]) for m in METRICS}
    return batch, metrics


def _groups(entry):
    src, overlap, nonneural = entry
    g = {"all", "train_overlap" if overlap else "held_out", f"src_{SOURCE_NAMES[src]}"}
    if nonneural is not None:
        g.add("nonneural" if nonneural else "neural_gated")
    return g


def _exact(scenario):
    """Means over each entry's ORIGINAL copy, straight from the table. Unpadded, DistributedSampler puts
    entry i on rank i % WORLD; the padding repeats go to later ranks."""
    out = {"n_duplicates_dropped": float(WORLD * -(-len(scenario) // WORLD) - len(scenario))}
    for g in GROUPS:
        members = [i for i, e in scenario.items() if g in _groups(e)]
        out[f"n_{g}"] = float(len(members))
        for m in METRICS:
            if members:
                out[f"{m}_{g}"] = sum(_value(i, m, i % WORLD) for i in members) / len(members)
    return out


def _new_scheme(scenario):
    def fn(rank, world):
        records = []
        for idx in _rank_entries(scenario, rank):
            batch, metrics = _batch(scenario, idx, rank)
            records.extend(population_records(batch, metrics, SOURCE_NAMES))
        calls = record_collectives()
        out = population_means(gather_records(records, distributed=True), GROUPS)
        return {"calls": list(calls), "out": out}
    return fn


def _old_log(rc, batch, metrics):
    """The removed train_openfold._log validation branch, verbatim in its key logic."""
    def log(name, v):
        rc.log("validation_step", name, torch.mean(v), on_step=False, on_epoch=True, sync_dist=True,
               sync_dist_fn=lambda t, group=None, reduce_op=None: _sync_ddp_if_available(
                   t, group, reduce_op=reduce_op), batch_size=1)
    for k, v in metrics.items():
        log(f"val/{k}", v)
    if "is_train_overlap" in batch:
        suffix = "train_overlap" if bool(batch["is_train_overlap"].flatten()[0]) else "held_out"
        for k, v in metrics.items():
            log(f"val/{k}_{suffix}", v)
    if "in_nonneural_subset" in batch:
        suffix = "nonneural" if bool(batch["in_nonneural_subset"].flatten()[0]) else "neural_gated"
        for k, v in metrics.items():
            log(f"val/{k}_{suffix}", v)
    if "val_source" in batch:
        name = SOURCE_NAMES[int(batch["val_source"].flatten()[0])]
        for k, v in metrics.items():
            log(f"val/{k}_src_{name}", v)


def _old_scheme(scenario):
    def fn(rank, world):
        rc = _ResultCollection(training=False)
        for idx in _rank_entries(scenario, rank):
            _old_log(rc, *_batch(scenario, idx, rank))
        return {k: float(v) for k, v in rc.metrics(on_step=False)["log"].items()}
    return fn


def test_imports_are_this_checkout():
    print("allrank_metrics:", allrank_metrics.__file__)
    assert allrank_metrics.__file__.startswith(REPO_ROOT), (allrank_metrics.__file__, REPO_ROOT)


def test_records_carry_the_old_tags_group_rules():
    batch, metrics = _batch(SCENARIO_A, 4, rank=0)
    (idx, groups, values), = population_records(batch, metrics, SOURCE_NAMES)
    assert (idx, set(groups)) == (4, {"all", "train_overlap", "neural_gated", "src_pda"})
    assert set(values) == set(METRICS)
    batch, metrics = _batch(SCENARIO_A, 7, rank=3)
    assert set(population_records(batch, metrics, SOURCE_NAMES)[0][1]) == {"all", "train_overlap", "src_hard"}


def test_dedup_keeps_the_lowest_rank_copy():
    gathered = [[(0, ("all",), {"x": 1.0})], [(1, ("all",), {"x": 2.0})],
                [(2, ("all",), {"x": 3.0}), (0, ("all",), {"x": 99.0})]]
    records, n_dropped = dedup_entries(gathered)
    assert [r[0] for r in records] == [0, 1, 2] and records[0][2]["x"] == 1.0 and n_dropped == 1


def test_empty_group_logs_its_zero_count_and_no_mean():
    out = population_means([[(0, ("all", "held_out"), {"x": 0.5})]], ("all", "nonneural"))
    assert out == {"n_duplicates_dropped": 0.0, "n_all": 1.0, "x_all": 0.5, "n_nonneural": 0.0}


def test_single_process_needs_no_collective():
    # no process group exists here, so any collective would raise
    records = [(0, ("all",), {"x": 0.25})]
    assert gather_records(records, distributed=False) == [records]


@pytest.mark.parametrize("name,scenario", [("A_padding_order_missing_group", SCENARIO_A),
                                           ("B_order_only", SCENARIO_B)])
def test_gloo_4_ranks_issue_the_same_collectives_and_get_the_exact_pooled_means(tmp_path, name, scenario):
    res = run_ranks(_new_scheme(scenario), WORLD, tmp_path / "store")
    exact = _exact(scenario)
    for r in range(WORLD):
        print(name, "rank", r, "entries", _rank_entries(scenario, r), "calls", res[r]["calls"])
    assert all(res[r]["calls"] == ["all_gather_object"] for r in range(WORLD))
    assert all(res[r]["out"] == res[0]["out"] for r in range(WORLD)), "ranks disagree"
    assert res[0]["out"] == exact
    print(name, {k: v for k, v in exact.items() if k.startswith(("n_", "lddt_ca"))})


def test_known_bad_control_old_per_key_sync_misses_keys_when_a_group_is_absent_from_a_rank():
    """Scenario A under the removed logging: rank 3 would sync 6 fewer keys (24 fewer NCCL ops) -> desync.

    Counted, not run: running it would hang or mis-pair the collectives.
    """
    keys = []
    for r in range(WORLD):
        rc = _ResultCollection(training=False)
        for idx in _rank_entries(SCENARIO_A, r):
            _old_log(rc, *_batch(SCENARIO_A, idx, r))
        keys.append(len(rc))
    print("synced keys per rank (old scheme):", keys)
    assert keys[3] == keys[0] - len(METRICS)


def test_known_bad_control_old_per_key_sync_cross_pairs_the_groups(tmp_path):
    """Scenario B under the removed logging, through Lightning's real sync: the published bug signature."""
    old = run_ranks(_old_scheme(SCENARIO_B), WORLD, tmp_path / "store")[0]
    exact = _exact(SCENARIO_B)
    for g in ("train_overlap", "held_out", "nonneural", "neural_gated"):
        print(g, "old", old[f"val/lddt_ca_{g}"], "exact", exact[f"lddt_ca_{g}"])
        assert abs(old[f"val/lddt_ca_{g}"] - exact[f"lddt_ca_{g}"]) > 1e-3, g
    # the groups every rank creates at the same position, with totals divisible by 4, stay exact
    for tag, g in (("val/lddt_ca", "all"), ("val/lddt_ca_src_pda", "src_pda"), ("val/lddt_ca_src_easy", "src_easy"),
                   ("val/lddt_ca_src_hard", "src_hard")):
        assert old[tag] == pytest.approx(exact[f"lddt_ca_{g}"], abs=1e-6), tag
