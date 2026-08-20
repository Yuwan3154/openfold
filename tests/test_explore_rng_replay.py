"""Gate tests for the best-of-K RNG replay -- the mechanism the whole design rests on.

⛔⛔ The claim under test: snapshotting the RNG before a scoring forward and restoring it before the
grad-carrying forward makes the two forwards draw the SAME randomness. If it does not hold, the
backward runs through a sample that was never scored. That failure is completely silent.

These are the CPU-checkable halves: that the snapshot/restore helpers cover both the CPU and CUDA
generators, that they compose correctly with the noise-ladder scale (restoring the RNG alone is NOT
enough once each rung has its own scale), and that dropout -- not just the pair init -- is replayed.
The remaining risks (nondeterministic CUDA reductions, fused kernels with private RNG, activation
checkpointing recomputing dropout in backward) can only be measured on a real GPU forward, which is
what --explore_verify_replay instruments.
"""

import math

import pytest
import torch
import torch.nn as nn

from openfold.model.contractive_recycling import sample_gaussian_pair_init

C_Z = 64
SIGMA0 = math.sqrt(2.0 / (5.0 * C_Z))


class _Helpers:
    """The two helpers from train_openfold, verbatim in behaviour, without importing the trainer
    (which pulls in deepspeed/lightning). Kept in sync by test_matches_trainer_source below."""

    def _rng_snapshot(self):
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        return (torch.get_rng_state(),
                torch.cuda.get_rng_state(dev) if dev is not None else None)

    def _rng_restore(self, snap):
        cpu_state, cuda_state = snap
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, torch.cuda.current_device())


H = _Helpers()


def test_snapshot_restore_replays_the_pair_init_draw():
    snap = H._rng_snapshot()
    a = sample_gaussian_pair_init((2, 8, 8, C_Z), C_Z)
    H._rng_restore(snap)
    b = sample_gaussian_pair_init((2, 8, 8, C_Z), C_Z)
    assert torch.equal(a, b)


def test_without_restore_the_draw_differs():
    """Negative control: if this ever passes, the test above proves nothing."""
    a = sample_gaussian_pair_init((2, 8, 8, C_Z), C_Z)
    b = sample_gaussian_pair_init((2, 8, 8, C_Z), C_Z)
    assert not torch.equal(a, b)


def test_dropout_is_replayed_too_not_just_the_pair_init():
    """⛔ The reason the FULL rng state is saved rather than a generator for the pair init: the
    Evoformer runs dropout, and a replay that only reproduced z_0 would backprop through a different
    dropout mask than the one that was scored."""
    drop = nn.Dropout(0.5)
    drop.train()
    x = torch.ones(256)
    snap = H._rng_snapshot()
    a = drop(x)
    H._rng_restore(snap)
    b = drop(x)
    assert torch.equal(a, b)
    c = drop(x)                                  # no restore -> must differ
    assert not torch.equal(a, c)


def test_pair_init_and_dropout_replay_together_in_sequence():
    """The real forward interleaves them; replaying must reproduce the whole sequence, not just the
    first consumer of the generator."""
    drop = nn.Dropout(0.3)
    drop.train()

    def _seq():
        z = sample_gaussian_pair_init((1, 6, 6, C_Z), C_Z)
        return z, drop(torch.ones(128)), sample_gaussian_pair_init((1, 6, 6, C_Z), C_Z)

    snap = H._rng_snapshot()
    a = _seq()
    H._rng_restore(snap)
    b = _seq()
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_restoring_rng_alone_is_NOT_enough_under_a_noise_ladder():
    """⛔⛔ THE REGRESSION THIS FILE EXISTS FOR. With a per-sample noise ladder, the scale is part of
    what defines the sample. Restoring only the RNG replays the same random FIELD but at whatever
    scale the last rung left behind -- a different z_0, hence a different sample, hence a gradient
    aimed at something that was never scored."""
    ladder = [0.0, 1.0, 2.0, 4.0]
    snaps, draws = [], []
    for tau in ladder:
        snaps.append(H._rng_snapshot())
        draws.append(sample_gaussian_pair_init((1, 6, 6, C_Z), C_Z, scale=tau))

    pick = 1                                     # winner drawn at tau=1.0
    # WRONG: restore the RNG but leave the scale at the last rung's value (4.0)
    H._rng_restore(snaps[pick])
    wrong = sample_gaussian_pair_init((1, 6, 6, C_Z), C_Z, scale=ladder[-1])
    assert not torch.equal(wrong, draws[pick])
    # RIGHT: restore the RNG *and* the scale
    H._rng_restore(snaps[pick])
    right = sample_gaussian_pair_init((1, 6, 6, C_Z), C_Z, scale=ladder[pick])
    assert torch.equal(right, draws[pick])
    # and the wrong one is off by exactly the scale ratio, confirming the mechanism
    assert torch.allclose(wrong, draws[pick] * (ladder[-1] / ladder[pick]), atol=1e-6)


def test_zero_rung_replays_trivially_and_consumes_no_randomness():
    """tau=0 short-circuits before touching the generator, so it must NOT advance the RNG -- otherwise
    a ladder containing 0 would desynchronise every later rung's replay."""
    before = torch.get_rng_state()
    z = sample_gaussian_pair_init((1, 6, 6, C_Z), C_Z, scale=0.0)
    after = torch.get_rng_state()
    assert torch.equal(z, torch.zeros_like(z))
    assert torch.equal(before, after), "scale=0 advanced the RNG; a ladder replay would desync"


def test_each_rank_snapshots_independently():
    """Under DDP each rank runs its own training_step on its own batch and snapshots its own state.
    Simulated here by two independent generator states: restoring one must not perturb the other's
    stream, which is what makes the per-rank replay sound."""
    torch.manual_seed(11)
    snap_a = H._rng_snapshot()
    a1 = torch.randn(64)
    torch.manual_seed(22)
    snap_b = H._rng_snapshot()
    b1 = torch.randn(64)
    H._rng_restore(snap_a)
    assert torch.equal(torch.randn(64), a1)
    H._rng_restore(snap_b)
    assert torch.equal(torch.randn(64), b1)


def test_matches_trainer_source():
    """Keeps the local copies honest: the trainer's helpers must still be the two-element
    (cpu, cuda) snapshot these tests model. A drift here silently invalidates this whole file."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "train_openfold.py"
    txt = src.read_text()
    assert "def _rng_snapshot(self):" in txt
    assert "torch.cuda.get_rng_state(dev) if dev is not None else None" in txt
    assert "def _rng_restore(self, snap):" in txt
    assert "torch.cuda.set_rng_state(cuda_state, torch.cuda.current_device())" in txt
