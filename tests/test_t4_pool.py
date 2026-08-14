"""Gate tests for T4's promoted-template pool (openfold/utils/t4_pool.py).

The failure modes that matter are the ones that stay silent in a single-process run: per-process
hash sharding (which broke the production template run), DDP ranks clobbering each other, and a
crop being placed at the wrong residues on read-back.
"""

import json

import numpy as np
import pytest

from openfold.utils.t4_pool import PromotedTemplatePool, PromotedTemplateWriter

L_FULL, L_CROP = 40, 12


def _promote(w, chain, epoch=0, step=0, start=5, tm=0.8, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((L_CROP, 37), bool)
    mask[:, :5] = True
    coords = rng.normal(size=(L_CROP, 37, 3)).astype(np.float32) * mask[..., None]
    w.submit(chain, epoch, step, tm, tm - 0.2, coords, mask,
             rng.integers(0, 20, L_CROP), np.arange(start, start + L_CROP))
    return coords, mask


@pytest.fixture
def written(tmp_path):
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    coords, mask = _promote(w, "1abc_A", start=5, tm=0.80, seed=1)
    _promote(w, "1abc_A", step=1, start=20, tm=0.60, seed=2)
    _promote(w, "2xyz_B", step=2, start=0, tm=0.70, seed=3)
    w.close()
    return tmp_path, coords, mask


def test_writer_persists_every_promotion(written):
    root, _, _ = written
    lines = [json.loads(x) for x in (root / "rank0/index.jsonl").read_text().splitlines() if x]
    assert len(lines) == 3
    assert {l["chain"] for l in lines} == {"1abc_A", "2xyz_B"}
    assert all((root / "rank0" / l["npz"]).is_file() for l in lines)


def test_sharding_is_stable_across_processes(written):
    """⛔ builtin hash() is randomized per process; crc32 is not. A per-process shard would give
    every rank and every restart a different directory -- the exact bug that hit production."""
    import zlib
    root, _, _ = written
    lines = [json.loads(x) for x in (root / "rank0/index.jsonl").read_text().splitlines() if x]
    for l in lines:
        expected = f"shard{zlib.crc32(l['chain'].encode()) % 1000:04d}"
        assert l["npz"].split("/")[0] == expected


def test_pool_merges_all_ranks(tmp_path):
    """Ranks must not clobber each other -- each writes its own subtree, the reader merges."""
    for r in range(3):
        w = PromotedTemplateWriter(str(tmp_path), rank=r)
        _promote(w, "1abc_A", step=r, tm=0.5 + 0.1 * r, seed=r)
        w.close()
    pool = PromotedTemplatePool(str(tmp_path))
    assert pool.refresh() == 3
    assert len(pool.by_chain["1abc_A"]) == 3


def test_crop_is_placed_at_its_residue_index(written):
    """A promoted CROP must land on the residues it covers and be masked everywhere else."""
    root, coords, mask = written
    pool = PromotedTemplatePool(str(root))
    pool.refresh()
    # pick the tm=0.80 entry deterministically by capping the pool to the single best
    pool.max_per_chain = 1
    pool.refresh()
    f = pool.sample_features("1abc_A", 1, np.random.default_rng(0), n_res=L_FULL)
    assert f["positions"].shape == (1, L_FULL, 37, 3)
    covered = np.arange(5, 5 + L_CROP)
    assert (f["mask"][0][covered][:, :5] == 1).all()
    outside = np.setdiff1d(np.arange(L_FULL), covered)
    assert (f["mask"][0][outside] == 0).all()
    assert (f["positions"][0][outside] == 0).all()
    np.testing.assert_allclose(f["positions"][0][covered][mask], coords[mask], rtol=1e-6)


def test_cap_keeps_the_best_not_the_newest(written):
    """A cap that evicted by recency would let a late bad epoch push out good templates."""
    root, _, _ = written
    pool = PromotedTemplatePool(str(root), max_per_chain=1)
    pool.refresh()
    assert len(pool.by_chain["1abc_A"]) == 1
    assert pool.by_chain["1abc_A"][0]["tm_pred"] == pytest.approx(0.80)


def test_missing_chain_returns_none(written):
    root, _, _ = written
    pool = PromotedTemplatePool(str(root))
    pool.refresh()
    assert "nosuch_A" not in pool
    assert pool.sample_features("nosuch_A", 1, np.random.default_rng(0), n_res=L_FULL) is None


def test_stale_entry_longer_than_the_chain_is_clipped(tmp_path):
    """A pool entry whose residues run past the current chain must be clipped, not crash."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    _promote(w, "1abc_A", start=L_FULL - 4, seed=9)      # runs off the end
    w.close()
    pool = PromotedTemplatePool(str(tmp_path))
    pool.refresh()
    f = pool.sample_features("1abc_A", 1, np.random.default_rng(0), n_res=L_FULL)
    assert f["mask"].shape == (1, L_FULL, 37)
    assert (f["mask"][0][: L_FULL - 4] == 0).all()


def test_full_queue_drops_instead_of_blocking(tmp_path):
    """Stalling the GPU to persist a template is the worse trade; drops must be COUNTED."""
    w = PromotedTemplateWriter(str(tmp_path), rank=0, max_queue=1)
    w._thread.join(timeout=0)                            # writer thread is alive; flood the queue
    for i in range(200):
        _promote(w, f"chain{i}_A", step=i, seed=i)
    w.close()
    assert w.n_written + w.n_dropped == 200
    assert w.n_written > 0
