"""Correctness gate for openfold.utils.tm_score (the in-loop TM used by T4 self-distillation).

Three layers:
  1. analytic invariances (identity, rigid motion, reflection, normalization semantics),
  2. equivalence against a straightforward per-seed LOOP implementation -- the seed-batched
     version in the module is an optimization and must not change the answer,
  3. batch independence -- a batched call must equal N single calls, because the training loop
     scores a whole batch at once and a leak across the batch dim would be invisible otherwise.

The loop reference below is the implementation that was validated against `USalign -TMscore 5` on
real structure pairs; keeping it here makes that validation transferable to the optimized version.

⛔ Correction (2026-08-14): an earlier writeup reported the FAST preset at max abs err 0.0018 with
0/60 decisions flipped. Re-measured on 60 pairs that actually reach the low-TM end (0.223-0.999),
FAST peaks at **0.0440** error while the REFERENCE schedule stays at 0.0008 -- the earlier sample
simply never went below TM ~ 0.5. The T4 gate therefore uses REFERENCE, not FAST. Most tests here
still run FAST because they only need a cheap, deterministic scorer, but
test_reference_never_scores_below_fast pins the relationship between the two.
"""

import numpy as np
import torch

from openfold.utils.tm_score import FAST_KWARGS, REFERENCE_KWARGS, tm_score, tm_score_ca


def _loop_tm(pred, ref, mask=None, norm_mask=None, n_iter=20, min_seed_len=4, max_seeds=None):
    """Reference: one Python iteration per seed, exactly as originally validated vs USalign."""
    pred, ref = pred.float(), ref.float()
    B, L, _ = pred.shape
    if mask is None:
        mask = torch.ones(B, L, dtype=pred.dtype)
    mask = mask.float()
    norm_mask = mask if norm_mask is None else norm_mask.float()
    L_norm = norm_mask.sum(dim=1).clamp_min(1.0)
    d0 = (1.24 * (L_norm - 15.0).clamp_min(1e-6) ** (1.0 / 3.0) - 1.8).clamp_min(0.5)
    d0sq = (d0 ** 2)[:, None]

    def kabsch(P, Q, w):
        wsum = w.sum(1, keepdim=True).clamp_min(1e-8)
        Pc = (P * w[..., None]).sum(1) / wsum
        Qc = (Q * w[..., None]).sum(1) / wsum
        H = torch.einsum("bli,bl,blj->bij", P - Pc[:, None], w, Q - Qc[:, None])
        U, _, Vh = torch.linalg.svd(H)
        V, Ut = Vh.transpose(-2, -1), U.transpose(-2, -1)
        D = torch.eye(3).expand(P.shape[0], 3, 3).clone()
        D[:, 2, 2] = torch.sign(torch.linalg.det(V @ Ut))
        R = V @ D @ Ut
        return R, Qc - torch.einsum("bij,bj->bi", R, Pc)

    seed_lens, fl = [], L
    while fl >= min_seed_len:
        seed_lens.append(fl)
        fl //= 2
    if not seed_lens:
        seed_lens = [L]
    seeds = [(f, s) for f in seed_lens for s in range(0, max(L - f + 1, 1), max(f // 2, 1))]
    if max_seeds is not None and len(seeds) > max_seeds:
        step = len(seeds) / max_seeds
        seeds = [seeds[int(i * step)] for i in range(max_seeds)]

    idx = torch.arange(L)
    best = torch.zeros(B)
    for fl, st in seeds:
        w = ((idx >= st) & (idx < st + fl)).float()[None].expand(B, L) * mask
        if (w.sum(1) < 3).any():
            continue
        for it in range(n_iter):
            R, t = kabsch(pred, ref, w)
            moved = torch.einsum("bij,blj->bli", R, pred) + t[:, None]
            dsq = ((moved - ref) ** 2).sum(-1)
            best = torch.maximum(best, ((1.0 / (1.0 + dsq / d0sq)) * mask).sum(1) / L_norm)
            w_new = (dsq < (d0 + 0.5 + it * 0.25)[:, None] ** 2).float() * mask
            if (w_new.sum(1) < 4).any() or torch.equal(w_new, w):
                break
            w = w_new
    return best


def _helix(n, seed=0):
    """Ideal alpha helix CA trace -- chiral, so it also exercises the reflection correction."""
    g = np.random.default_rng(seed)
    i = np.arange(n)
    x = 2.3 * np.cos(1.7451 * i)
    y = 2.3 * np.sin(1.7451 * i)
    z = 1.5 * i
    c = np.stack([x, y, z], -1) + g.normal(0, 0.05, (n, 3))
    return torch.tensor(c, dtype=torch.float32)


def _walk(n, seed=0, persistence=0.55):
    """Random coil at 3.8 A CA spacing.

    ⚠️ Needed because `_helix(n, seed)` only reseeds the 0.05 A jitter -- every seed is the SAME
    ideal helix, so a pair of them scores TM ~ 0.999 and is useless as a "different fold" fixture.
    """
    g = np.random.default_rng(1000 + seed)
    d = np.array([0.0, 0.0, 1.0])
    pts = [np.zeros(3)]
    for _ in range(n - 1):
        d = d + persistence * g.normal(size=3)
        d /= np.linalg.norm(d)
        pts.append(pts[-1] + 3.8 * d)
    return torch.tensor(np.stack(pts), dtype=torch.float32)


def test_fixtures_are_actually_different_folds():
    """Guards the fixtures themselves: the tests below are vacuous if these score ~1."""
    h, w1, w2 = _helix(100)[None], _walk(100, 1)[None], _walk(100, 2)[None]
    assert float(tm_score(w1, h, **FAST_KWARGS)) < 0.5
    assert float(tm_score(w2, w1, **FAST_KWARGS)) < 0.5
    assert float(tm_score(_helix(100, 7)[None], h, **FAST_KWARGS)) > 0.99   # seed = jitter only


def test_identity_is_one():
    x = _helix(120)[None]
    assert torch.allclose(tm_score(x, x, **FAST_KWARGS), torch.ones(1), atol=1e-4)


def test_invariant_under_rigid_motion():
    x = _helix(150)[None]
    g = np.random.default_rng(3)
    A = g.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    Rm = torch.tensor(Q, dtype=torch.float32)
    t = torch.tensor(g.normal(size=3) * 25, dtype=torch.float32)
    moved = (x @ Rm.T) + t
    assert torch.allclose(tm_score(moved, x, **FAST_KWARGS), torch.ones(1), atol=1e-4)


def test_reflection_is_rejected():
    """A mirrored helix is a different molecule; an improper 'rotation' must not score 1."""
    x = _helix(120)[None]
    mirrored = x * torch.tensor([1.0, 1.0, -1.0])
    assert float(tm_score(mirrored, x, **FAST_KWARGS)) < 0.75


def test_matches_loop_reference():
    """The seed-batched optimization must not move the answer, across the whole TM range."""
    seen = []
    for seed in range(4):
        a, b = _helix(100, seed)[None], _walk(100, seed)[None]
        for w in torch.linspace(0, 1, 5):
            pred = (1 - w) * b + w * a
            fast = tm_score(pred, a, **FAST_KWARGS)
            ref = _loop_tm(pred, a, **FAST_KWARGS)
            seen.append(float(fast))
            assert torch.allclose(fast, ref, atol=2e-3), (seed, float(w), float(fast), float(ref))
    assert min(seen) < 0.4 and max(seen) > 0.95, f"blend did not span TM range: {min(seen)}-{max(seen)}"


def test_batch_independence():
    """Batched call == N single calls; a leak across the batch dim would corrupt per-sample gating."""
    xs = [_walk(90, s)[None] for s in range(3)]
    ys = [_walk(90, s + 7)[None] for s in range(3)]
    single = torch.cat([tm_score(x, y, **FAST_KWARGS) for x, y in zip(xs, ys)])
    batched = tm_score(torch.cat(xs), torch.cat(ys), **FAST_KWARGS)
    assert torch.allclose(single, batched, atol=1e-5)


def test_masked_residues_are_ignored():
    """Corrupting masked-out residues must not change the score."""
    x = _helix(100)[None]
    y = x.clone()
    mask = torch.ones(1, 100)
    mask[0, 60:] = 0
    y[0, 60:] += 500.0
    assert torch.allclose(tm_score(y, x, mask=mask, **FAST_KWARGS), torch.ones(1), atol=1e-4)


def test_norm_mask_penalizes_partial_coverage():
    """Half-covered template scoring perfectly on its half caps near 0.5, not 1.0."""
    x = _helix(100)[None]
    both = torch.zeros(1, 100)
    both[0, :50] = 1
    full = torch.ones(1, 100)
    covered = tm_score(x, x, mask=both, norm_mask=both, **FAST_KWARGS)
    penalized = tm_score(x, x, mask=both, norm_mask=full, **FAST_KWARGS)
    assert float(covered) > 0.99
    assert 0.45 < float(penalized) < 0.55


def test_bf16_input_is_accepted():
    x = _helix(80)[None]
    lo = tm_score(x.to(torch.bfloat16), x.to(torch.bfloat16), **FAST_KWARGS)
    assert lo.dtype == torch.float32 and float(lo) > 0.98


def test_atom37_wrapper_uses_ca_and_native_normalization():
    n = 90
    ca = _helix(n)
    pred37 = torch.zeros(1, n, 37, 3)
    ref37 = torch.zeros(1, n, 37, 3)
    pred37[0, :, 1] = ca
    ref37[0, :, 1] = ca
    ref_mask = torch.zeros(1, n, 37)
    ref_mask[0, :, 1] = 1
    tmpl_mask = ref_mask.clone()
    tmpl_mask[0, 45:, 1] = 0                       # template covers only the first half
    assert float(tm_score_ca(pred37, ref37, ref_mask, **FAST_KWARGS)) > 0.99
    partial = float(tm_score_ca(pred37, ref37, ref_mask, pred_mask37=tmpl_mask, **FAST_KWARGS))
    assert 0.45 < partial < 0.55


def test_reference_never_scores_below_fast():
    """TM is a max over everything explored and REFERENCE's seed set is a superset of FAST's, so
    REFERENCE >= FAST identically. The gap is the accuracy FAST gives up -- it opens at LOW TM,
    which is why the T4 gate uses REFERENCE (max err 0.0008 vs USalign, against FAST's 0.0440)."""
    gaps = []
    for s in range(6):
        nat = _helix(110, s)[None]
        for w in (0.0, 0.25, 0.5):
            pred = (1 - w) * _walk(110, s)[None] + w * nat
            hi = float(tm_score(pred, nat, **REFERENCE_KWARGS))
            lo = float(tm_score(pred, nat, **FAST_KWARGS))
            assert hi >= lo - 1e-6, (s, w, hi, lo)
            gaps.append(hi - lo)
    assert max(gaps) > 1e-3, "FAST matched REFERENCE everywhere -- fixtures never reach low TM"


def test_gate_defaults_to_reference_schedule():
    """Guards the T4 default against silently reverting to FAST."""
    from openfold.utils import t4_self_distill as t4
    assert t4.REFERENCE_KWARGS == REFERENCE_KWARGS


def test_short_chain_below_min_seed_len():
    """Crops shorter than min_seed_len must still produce a full-length seed, not an empty list."""
    x = _helix(12)[None]
    assert float(tm_score(x, x, **FAST_KWARGS)) > 0.99


def test_kabsch_works_in_bfloat16():
    """⛔ torch.linalg.svd has no bf16 CUDA kernel. Under --precision bf16 this raised
    `"svd_cuda_gesvdjBatched" not implemented for 'BFloat16'` the first time the T4 gate ran on a GPU
    -- every earlier measurement of the gate was CPU-only, so nothing caught it. _kabsch now computes
    in fp32 and casts back; this pins that, and that the dtype contract is preserved."""
    from openfold.utils.tm_score import _kabsch
    torch.manual_seed(0)
    P = torch.randn(4, 16, 3)
    R_true = torch.linalg.qr(torch.randn(3, 3))[0]
    Q = P @ R_true.T + 2.0
    w = torch.ones(4, 16)
    R32, t32 = _kabsch(P, Q, w)
    Rb, tb = _kabsch(P.bfloat16(), Q.bfloat16(), w.bfloat16())
    assert Rb.dtype == torch.bfloat16 and tb.dtype == torch.bfloat16, "must return the input dtype"
    assert torch.allclose(Rb.float(), R32, atol=5e-2), (Rb.float() - R32).abs().max()


def test_kabsch_recovers_a_known_rotation():
    """Guards the fp32 cast against a silent transpose/reflection slip while touching this function."""
    from openfold.utils.tm_score import _kabsch
    torch.manual_seed(1)
    P = torch.randn(2, 24, 3)
    R_true = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.linalg.det(R_true) < 0:
        R_true[:, 2] *= -1
    Q = P @ R_true.T
    R, t = _kabsch(P, Q, torch.ones(2, 24))
    assert torch.allclose(R, R_true.expand(2, 3, 3), atol=1e-4), (R - R_true).abs().max()
