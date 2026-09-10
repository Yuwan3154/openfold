"""Gate tests for prune_work/merge_band_trees.py.

The failure mode this exists for is silent. A chain's templates live in ONE npz storing coords for
present atoms only, and the trainer reconstructs with `full[:, atom_mask] = coords`. If round 1 and
round 2 ever disagree about a chain's atom_mask / aatype / residue_index, concatenating their coords
scatters round-2 atoms onto round-1 positions -- wrong structures, no exception, and the live
sequence-agreement assert cannot see it because the sequence is unchanged.

So the load-bearing test here is not that a merge works; it is that a FRAME MISMATCH REFUSES to
merge and gets recorded.
"""

import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "prune_work", "merge_band_trees.py")
L = 30


def _shard(root, chain):
    return os.path.join(str(root), "shard%04d" % (zlib.crc32(chain.encode()) % 1000))


def _atom_mask():
    am = np.zeros((L, 37), bool)
    am[:, 1] = True                     # CA everywhere
    am[:5, 0] = True                    # plus a few N, so n_present != L
    return am


def _write(root, chain, rewinds, rng, residue_offset=1):
    am = _atom_mask()
    d = _shard(root, chain)
    os.makedirs(d, exist_ok=True)
    np.savez(
        os.path.join(d, chain + ".npz"),
        coords=rng.standard_normal((len(rewinds), int(am.sum()), 3)).astype(np.float32),
        atom_mask=am,
        aatype=np.arange(L, dtype=np.int8) % 20,
        residue_index=np.arange(residue_offset, L + residue_offset, dtype=np.int32),
        rewind_steps=np.asarray(rewinds, np.int16),
        model="cc89", schedule="tiered",
    )


def _index(path, chains, tms, rewinds):
    tm = np.tile(np.asarray(tms, np.float32), (len(chains), 1))
    rw = np.tile(np.asarray(rewinds, np.int16), (len(chains), 1))
    band = (tm > 0.3) & (tm < 0.9)
    slot = np.full(tm.shape, -1, np.int16)
    for i in range(len(chains)):
        slot[i, np.flatnonzero(band[i])] = np.arange(int(band[i].sum()), dtype=np.int16)
    np.savez(str(path), chains=np.array(chains, "<U8"), tm=tm, rewind=rw,
             length=np.full(len(chains), L, np.int32), slot=slot,
             min_tm=np.float32(0.3), max_tm=np.float32(0.9))


@pytest.fixture
def trees(tmp_path):
    """Four chains covering every branch: merged, round-1 only, frame mismatch, round-2 only."""
    rng = np.random.default_rng(0)
    a, b, dst = tmp_path / "a", tmp_path / "b", tmp_path / "m"
    r1, r2 = [375, 300, 250, 200], [340, 280, 220]

    _write(a, "1aaa_A", r1, rng)
    _write(b, "1aaa_A", r2, rng)                       # frames agree -> merge
    _write(a, "2bbb_A", r1, rng)                       # round-1 only -> copy through
    _write(a, "3ccc_A", r1, rng)
    _write(b, "3ccc_A", r2, rng, residue_offset=9)     # SHIFTED frame -> must be skipped
    _write(b, "4ddd_A", r2, rng)                       # round-2 only -> copy through

    _index(tmp_path / "ia.npz", ["1aaa_A", "2bbb_A", "3ccc_A"], [0.95, 0.7, 0.5, 0.2], r1)
    _index(tmp_path / "ib.npz", ["1aaa_A", "3ccc_A", "4ddd_A"], [0.8, 0.6, 0.4], r2)

    base = [sys.executable, SCRIPT,
            "--index-a", str(tmp_path / "ia.npz"), "--root-a", str(a),
            "--index-b", str(tmp_path / "ib.npz"), "--root-b", str(b),
            "--dst-root", str(dst), "--out-index", str(tmp_path / "im.npz")]
    for extra in ([], ["--index-only"]):
        r = subprocess.run(base + extra, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    return dst, tmp_path / "im.npz"


def _load(dst, chain):
    return np.load(os.path.join(_shard(dst, chain), chain + ".npz"), allow_pickle=False)


def test_frame_mismatch_refuses_to_merge_and_is_recorded(trees):
    """⛔ THE test. A shifted residue_index must never be concatenated."""
    dst, _ = trees
    assert not os.path.exists(os.path.join(_shard(dst, "3ccc_A"), "3ccc_A.npz")), \
        "a chain whose residue_index differs between rounds was merged anyway"
    # a count is not evidence: every skip must be individually re-checkable
    recorded = open(os.path.join(str(dst), "_skipped_0000.txt")).read()
    assert "3ccc_A:frame_differs_residue_index" in recorded, recorded


def test_matching_frames_concatenate_in_order(trees):
    dst, _ = trees
    m = _load(dst, "1aaa_A")
    assert m["coords"].shape[0] == 7
    assert int(m["n_round1"]) == 4 and int(m["n_round2"]) == 3
    assert list(m["rewind_steps"]) == [375, 300, 250, 200, 340, 280, 220]
    assert np.array_equal(m["atom_mask"], _atom_mask())


@pytest.mark.parametrize("chain,n", [("2bbb_A", 4), ("4ddd_A", 3)])
def test_one_sided_chains_copy_through_untouched(trees, chain, n):
    dst, _ = trees
    d = _load(dst, chain)
    assert d["coords"].shape[0] == n
    assert "n_round1" not in d.files, "a one-sided chain must not be labelled as merged"


def test_absent_half_is_marked_unusable_not_zero(trees):
    """A round-2-only chain has no round-1 rungs; TM 0 would sit INSIDE a band starting at 0."""
    _, out_index = trees
    z = np.load(str(out_index), allow_pickle=False)
    chains = [str(c) for c in z["chains"]]
    assert chains == ["1aaa_A", "2bbb_A", "3ccc_A", "4ddd_A"]
    assert z["tm"].shape == (4, 7)
    i = chains.index("4ddd_A")
    assert (z["tm"][i, :4] == -1.0).all()
    assert (z["slot"][i, :4] == -1).all()


def test_band_disagreement_is_fatal(tmp_path):
    """Two trees pruned to different bands must not be merged -- `eligible` would mean two things."""
    rng = np.random.default_rng(1)
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "1aaa_A", [375, 300], rng)
    _write(b, "1aaa_A", [340, 280], rng)
    _index(tmp_path / "ia.npz", ["1aaa_A"], [0.8, 0.5], [375, 300])
    z = dict(np.load(str(tmp_path / "ia.npz"), allow_pickle=False))
    z["min_tm"] = np.float32(0.4)                     # <-- a different band
    np.savez(str(tmp_path / "ib.npz"), **z)
    r = subprocess.run(
        [sys.executable, SCRIPT, "--index-a", str(tmp_path / "ia.npz"), "--root-a", str(a),
         "--index-b", str(tmp_path / "ib.npz"), "--root-b", str(b),
         "--dst-root", str(tmp_path / "m"), "--out-index", str(tmp_path / "im.npz")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "different bands" in r.stderr, r.stderr
