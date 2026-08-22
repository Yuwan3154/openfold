"""Independent tests of the promoted-pool EVICTION mechanism (user directive 2026-08-22).

`tests/test_t4_pool.py` and `test_t4_pool_phase3.py` already assert that the cap keeps the newest
and is deterministic. These tests deliberately do NOT re-assert that. They cover what those files
leave open, and they are written against the mechanism's contract rather than its implementation:

  1. `tm_pred` has ZERO influence on retention -- the OLD keep-the-best cap must be genuinely dead,
     not merely outranked. Adversarial construction: oldest record has the BEST score.
  2. Eviction is a READ-TIME filter that deletes nothing, so it is reversible.
  3. Eviction is PER CHAIN -- a prolific chain cannot evict a quiet one.
  4. Boundaries: cap 0 (uncapped), cap 1, cap > population.
  5. The `sample` tiebreak under --t4_promote_all, where K records share (epoch, step, rank).
  6. Run C's actual configuration: K=4 promote-all against max_per_chain=64.

⚠️ (6) is the one the live-run concern maps onto: if retention silently ate the extra promote-all
volume, the pool would report more templates than it serves and the recombination signal Run C is
built around would quietly vanish.
"""

import json

import numpy as np
import pytest

from openfold.utils.t4_pool import PromotedTemplatePool, PromotedTemplateWriter

L_CROP = 12


def _promote(w, chain, epoch=0, step=0, tm=0.8, sample=0, start=5, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((L_CROP, 37), bool)
    mask[:, :5] = True
    coords = rng.normal(size=(L_CROP, 37, 3)).astype(np.float32) * mask[..., None]
    w.submit(chain, epoch, step, tm, tm - 0.2, coords, mask,
             rng.integers(0, 20, L_CROP), np.arange(start, start + L_CROP), sample=sample)


def _npzs(pool, chain):
    return [r["npz"] for r in pool.by_chain[chain]]


# ---------------------------------------------------------------- 1. tm_pred is inert

def test_the_best_scoring_record_is_evicted_when_it_is_the_oldest(tmp_path):
    """⛔⛔ The OLD mechanism ranked retention by tm_pred. Reversed 2026-08-19. This is the
    adversarial case that separates the two policies: score and recency disagree maximally."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    _promote(w, "1abc_A", epoch=0, step=0, tm=0.99, seed=1)   # oldest, BEST score
    _promote(w, "1abc_A", epoch=9, step=9, tm=0.01, seed=2)   # newest, WORST score
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=1)
    pool.refresh()
    kept = pool.by_chain["1abc_A"]
    assert len(kept) == 1
    assert kept[0]["tm_pred"] == pytest.approx(0.01), "tm_pred still drives retention"
    assert kept[0]["epoch"] == 9


def test_retention_is_bit_identical_when_only_the_scores_are_permuted(tmp_path):
    """The strongest form: build two pools identical except for tm_pred, and require the SAME
    survivors. Any residual score-sensitivity anywhere in refresh() breaks this."""
    kept_sets = []
    for scores in ([0.1, 0.5, 0.9, 0.3], [0.9, 0.3, 0.1, 0.5]):
        root = tmp_path / f"p{scores[0]}"
        w = PromotedTemplateWriter(str(root), rank=0)
        for i, s in enumerate(scores):
            _promote(w, "1abc_A", epoch=i, step=i, tm=s, seed=i)
        w.close()
        pool = PromotedTemplatePool(str(root), max_per_chain=2)
        pool.refresh()
        kept_sets.append([(r["epoch"], r["step"]) for r in pool.by_chain["1abc_A"]])
    assert kept_sets[0] == kept_sets[1] == [(3, 3), (2, 2)]


# ---------------------------------------------------------------- 2. eviction deletes nothing

def test_eviction_is_read_time_and_deletes_no_files(tmp_path):
    """⛔ Documented invariant: nothing in the codebase deletes pool files. If eviction ever became
    destructive, raising the cap later would silently return fewer templates than the index claims."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(6):
        _promote(w, "1abc_A", epoch=i, step=i, seed=i)
    w.close()
    on_disk = sorted(p.name for p in tmp_path.rglob("*.npz"))
    assert len(on_disk) == 6

    PromotedTemplatePool(str(tmp_path), max_per_chain=2).refresh()
    assert sorted(p.name for p in tmp_path.rglob("*.npz")) == on_disk
    # index untouched too
    assert len((tmp_path / "rank0/index.jsonl").read_text().strip().splitlines()) == 6


def test_raising_the_cap_recovers_previously_evicted_records(tmp_path):
    """The reversibility that (only) a read-time filter can give."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(6):
        _promote(w, "1abc_A", epoch=i, step=i, seed=i)
    w.close()
    assert PromotedTemplatePool(str(tmp_path), max_per_chain=2).refresh() == 2
    assert PromotedTemplatePool(str(tmp_path), max_per_chain=5).refresh() == 5
    assert PromotedTemplatePool(str(tmp_path), max_per_chain=99).refresh() == 6


# ---------------------------------------------------------------- 3. per-chain, not global

def test_a_prolific_chain_cannot_evict_a_quiet_one(tmp_path):
    """The cap is per chain. A global cap would let one heavily-promoted chain starve every other,
    and the effect would scale with --t4_promote_all."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(20):
        _promote(w, "loud_A", epoch=i, step=i, seed=i)
    _promote(w, "quiet_B", epoch=0, step=0, seed=99)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=3)
    assert pool.refresh() == 4                      # 3 + 1, not 3 total
    assert pool.n_for_chain("loud_A") == 3
    assert pool.n_for_chain("quiet_B") == 1


# ---------------------------------------------------------------- 4. boundaries

def test_cap_zero_means_uncapped(tmp_path):
    """`max_per_chain=0` is the "no cap" sentinel, NOT "keep nothing" -- an off-by-one here would
    empty the pool while every log still reported healthy writes."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(7):
        _promote(w, "1abc_A", epoch=i, step=i, seed=i)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=0)
    assert pool.refresh() == 7
    assert pool.n_for_chain("1abc_A") == 7


def test_cap_larger_than_the_population_keeps_everything(tmp_path):
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(3):
        _promote(w, "1abc_A", epoch=i, step=i, seed=i)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=64)
    assert pool.refresh() == 3


def test_refresh_is_idempotent(tmp_path):
    """An epoch trains on one snapshot; two dataloader workers refreshing independently must agree."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(8):
        _promote(w, "1abc_A", epoch=i // 3, step=i, seed=i)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=4)
    pool.refresh()
    first = _npzs(pool, "1abc_A")
    for _ in range(3):
        pool.refresh()
        assert _npzs(pool, "1abc_A") == first


def test_blank_and_whitespace_index_lines_are_skipped(tmp_path):
    """A crashed writer can leave a partial trailing line; refresh must not die on it."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    _promote(w, "1abc_A", epoch=0, step=0, seed=1)
    w.close()
    idx = tmp_path / "rank0/index.jsonl"
    idx.write_text(idx.read_text() + "\n   \n\n")
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=4)
    assert pool.refresh() == 1


# ---------------------------------------------------------------- 5. the promote-all sample tiebreak

def test_all_k_samples_of_one_step_are_retained_and_ordered_deterministically(tmp_path):
    """Under --t4_promote_all the K samples share (epoch, step, rank), so `sample` is the only
    separator left. They are distinct predictions and must all survive a cap that has room."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for k in range(4):
        _promote(w, "1abc_A", epoch=3, step=11, tm=0.5 + 0.05 * k, sample=k, seed=k)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=4)
    assert pool.refresh() == 4
    assert [r["sample"] for r in pool.by_chain["1abc_A"]] == [0, 1, 2, 3]


def test_a_tight_cap_slices_samples_by_index_not_by_score(tmp_path):
    """When the cap cuts into one step's K samples, the survivor is chosen by `sample` ascending --
    a deterministic rule. It must NOT be the best-scoring sample."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for k in range(4):
        _promote(w, "1abc_A", epoch=3, step=11, tm=0.9 - 0.2 * k, sample=k, seed=k)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=1)
    pool.refresh()
    kept = pool.by_chain["1abc_A"][0]
    assert kept["sample"] == 0
    # sample 0 happens to be the best here, so pin the reverse case too
    root2 = tmp_path / "rev"
    w = PromotedTemplateWriter(str(root2), rank=0)
    for k in range(4):
        _promote(w, "1abc_A", epoch=3, step=11, tm=0.1 + 0.2 * k, sample=k, seed=k)
    w.close()
    pool2 = PromotedTemplatePool(str(root2), max_per_chain=1)
    pool2.refresh()
    kept2 = pool2.by_chain["1abc_A"][0]
    assert kept2["sample"] == 0
    assert kept2["tm_pred"] == pytest.approx(0.1), "score leaked into the sample tiebreak"


def test_newer_step_outranks_a_lower_sample_index(tmp_path):
    """`sample` is the LAST separator, so it must never outrank recency."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    _promote(w, "1abc_A", epoch=1, step=1, sample=0, seed=1)
    _promote(w, "1abc_A", epoch=1, step=2, sample=3, seed=2)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=1)
    pool.refresh()
    assert pool.by_chain["1abc_A"][0]["step"] == 2


# ---------------------------------------------------------------- 6. Run C's real configuration

def test_run_c_k4_promote_all_against_cap_64_keeps_the_newest_16_steps(tmp_path):
    """Run C: --explore_k 4 --t4_promote_all --t4_max_per_chain 64 across 4 DDP ranks.

    With K=4 per step, a cap of 64 holds exactly 16 steps' worth -- and it must hold ALL FOUR
    samples of each of those 16, because serving fewer would silently shrink the recombination
    pool while the write counters still reported the full volume.
    """
    w = PromotedTemplateWriter(str(tmp_path), rank=0, max_queue=4096)
    n_steps = 40
    for s in range(n_steps):
        for k in range(4):
            _promote(w, "1abc_A", epoch=2, step=s, tm=0.5, sample=k, seed=s * 4 + k)
    w.close()
    assert len(list(tmp_path.rglob("*.npz"))) == n_steps * 4

    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=64)
    assert pool.refresh() == 64
    recs = pool.by_chain["1abc_A"]
    steps = sorted({r["step"] for r in recs})
    assert steps == list(range(n_steps - 16, n_steps)), "cap did not keep a contiguous newest block"
    for s in steps:
        assert sorted(r["sample"] for r in recs if r["step"] == s) == [0, 1, 2, 3], \
            f"step {s} lost samples to the cap"


def test_promote_all_volume_is_not_silently_eaten_across_four_ranks(tmp_path):
    """Same idea with DDP: 4 ranks x K=4 at the same step = 16 records that all tie on (epoch,
    step). The cap must resolve them deterministically and keep 64 of them, not collapse duplicates."""
    for r in range(4):
        w = PromotedTemplateWriter(str(tmp_path), rank=r, max_queue=4096)
        for s in range(10):
            for k in range(4):
                _promote(w, "1abc_A", epoch=5, step=s, sample=k, seed=r * 100 + s * 4 + k)
        w.close()
    assert len(list(tmp_path.rglob("*.npz"))) == 4 * 10 * 4

    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=64)
    assert pool.refresh() == 64
    recs = pool.by_chain["1abc_A"]
    assert len({r["npz"] for r in recs}) == 64, "duplicate paths retained"
    # newest 4 steps x 4 ranks x 4 samples = 64
    assert sorted({r["step"] for r in recs}) == [6, 7, 8, 9]
    assert sorted({r["_rank"] for r in recs}) == [0, 1, 2, 3]


def test_the_served_templates_come_from_the_retained_set_only(tmp_path):
    """The read path must draw from the post-eviction snapshot. If sample_features could reach an
    evicted record the cap would be decorative."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for i in range(10):
        _promote(w, "1abc_A", epoch=i, step=i, seed=i)
    w.close()
    pool = PromotedTemplatePool(str(tmp_path), max_per_chain=3)
    pool.refresh()
    retained = {r["_path"] for r in pool.by_chain["1abc_A"]}
    assert len(retained) == 3

    q = ("ACDEFGHIKLMNPQRSTVWY" * 3)[:40]
    rng = np.random.default_rng(0)
    # ask for more than the cap holds; must be clipped to the retained set, not backfilled
    feats = pool.sample_features("1abc_A", 8, rng, q)
    assert feats["template_all_atom_positions"].shape[0] == 3


def test_index_records_outnumbering_files_would_be_visible(tmp_path):
    """Guard against the class of bug where K records resolve to one file (caught in production
    2026-08-20): under promote-all, record count must equal distinct npz count."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0, max_queue=4096)
    for s in range(5):
        for k in range(4):
            _promote(w, "1abc_A", epoch=1, step=s, sample=k, seed=s * 4 + k)
    w.close()
    lines = [json.loads(x) for x in (tmp_path / "rank0/index.jsonl").read_text().splitlines() if x]
    assert len(lines) == 20
    assert len({l["npz"] for l in lines}) == 20
    assert len(list(tmp_path.rglob("*.npz"))) == 20
