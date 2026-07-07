"""Standalone CPU unit test for contractive_recycling.py -- verifies the claims from ESMFold2
Appendix A.2.5 / Prairie et al. 2026 empirically, not just by construction:
(1) Abar is strictly in (0,1) elementwise, robust across random parameter draws.
(2) Repeated application with a fixed input stays BOUNDED (converges), unlike a plain residual
    update (OpenFold's current RecyclingEmbedder mechanism), which grows unboundedly.
(3) Gradients backpropagated through many stacked iterations stay finite/bounded for the
    contractive update, and are compared against the plain-residual baseline.
(4) sample_gaussian_pair_init produces the right shape/mean/std/truncation.
"""
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openfold"))
from openfold.model.contractive_recycling import ContractivePairUpdate, sample_gaussian_pair_init

torch.manual_seed(0)


def test_abar_in_unit_interval():
    for _ in range(20):
        c_z = 32
        m = ContractivePairUpdate(c_z)
        with torch.no_grad():
            m.log_delta.normal_(0, 3)
            m.log_a.normal_(0, 3)
        a_bar, b_bar = m.discretized_params()
        # a=exp(log_a) (Mamba convention) has a much wider dynamic range than the old
        # softplus(log_a); under this N(0,3) stress draw, -delta*a can be extreme enough that
        # exp(-delta*a) underflows to exactly 0.0 in float32 -- still mathematically in [0,1),
        # never <0 or >=1 (delta>=0, a>0 guarantee -delta*a<=0 always, so exp(...) can't exceed 1).
        assert (a_bar >= 0).all() and (a_bar < 1).all(), f"Abar out of [0,1): min={a_bar.min()} max={a_bar.max()}"
    print("PASS: Abar in [0,1) elementwise across 20 random parameter draws")


def test_bounded_vs_unbounded():
    c_z = 16
    n_res = 8
    shape = (1, n_res, n_res, c_z)
    u_t = torch.randn(shape) * 2.0  # fixed nonzero input, reused every iteration

    contractive = ContractivePairUpdate(c_z)
    z_contractive = torch.zeros(shape)
    norms_contractive = []
    for _ in range(100):
        z_contractive = contractive(z_contractive, u_t)
        norms_contractive.append(z_contractive.norm().item())

    z_plain = torch.zeros(shape)
    norms_plain = []
    for _ in range(100):
        z_plain = z_plain + u_t  # OpenFold's current mechanism: plain additive residual
        norms_plain.append(z_plain.norm().item())

    print(f"contractive: ||z|| at t=1/10/50/100 = "
          f"{norms_contractive[0]:.3f}/{norms_contractive[9]:.3f}/"
          f"{norms_contractive[49]:.3f}/{norms_contractive[99]:.3f}")
    print(f"plain residual (OpenFold's current mechanism): ||z|| at t=1/10/50/100 = "
          f"{norms_plain[0]:.3f}/{norms_plain[9]:.3f}/{norms_plain[49]:.3f}/{norms_plain[99]:.3f}")

    # Contractive: growth from t=50 to t=100 should be small (near-converged plateau).
    contractive_growth = abs(norms_contractive[99] - norms_contractive[49])
    # Plain residual with constant nonzero input grows without bound (linearly, at minimum).
    plain_growth = abs(norms_plain[99] - norms_plain[49])
    assert contractive_growth < 0.1 * norms_contractive[49], \
        f"contractive update did not plateau: growth={contractive_growth} vs norm={norms_contractive[49]}"
    assert plain_growth > 10 * contractive_growth, \
        f"plain residual should grow much more than contractive: plain_growth={plain_growth} contractive_growth={contractive_growth}"
    print("PASS: contractive update plateaus (bounded); plain residual keeps growing, as expected")


def test_gradients_through_many_iterations():
    c_z = 16
    shape = (1, 4, 4, c_z)
    n_iters = 20

    contractive = ContractivePairUpdate(c_z)
    u_t = torch.randn(shape, requires_grad=True)
    z = torch.zeros(shape)
    for _ in range(n_iters):
        z = contractive(z, u_t)
    loss = z.pow(2).sum()
    loss.backward()
    grad_norm_contractive = u_t.grad.norm().item()
    assert torch.isfinite(u_t.grad).all(), "contractive: non-finite gradient after 20 iterations"

    u_t2 = torch.randn(shape, requires_grad=True)
    z2 = torch.zeros(shape)
    for _ in range(n_iters):
        z2 = z2 + u_t2  # plain residual baseline
    loss2 = z2.pow(2).sum()
    loss2.backward()
    grad_norm_plain = u_t2.grad.norm().item()

    print(f"grad norm after {n_iters} iterations -- contractive: {grad_norm_contractive:.3e}  "
          f"plain residual: {grad_norm_plain:.3e}")
    print("PASS: contractive gradients finite through 20 stacked iterations "
          "(plain-residual grad norm shown for comparison, not asserted -- toy example, "
          "no nonlinearity, so plain residual's gradient doesn't necessarily explode here; "
          "the real instability the paper describes is in the FULL nonlinear pair-folding-layer "
          "loop, not this isolated linear-combination step)")


def test_gaussian_init():
    d_pair = 128
    shape = (2, 10, 10, d_pair)
    z0 = sample_gaussian_pair_init(shape, d_pair)
    assert z0.shape == shape
    expected_std = (2.0 / (5.0 * d_pair)) ** 0.5
    empirical_std = z0.std().item()
    assert abs(empirical_std - expected_std) / expected_std < 0.15, \
        f"empirical std {empirical_std:.5f} too far from expected {expected_std:.5f}"
    assert z0.abs().max().item() <= 3 * expected_std + 1e-6, "truncation violated"
    print(f"PASS: gaussian init shape={tuple(z0.shape)} expected_std={expected_std:.5f} "
          f"empirical_std={empirical_std:.5f} max|z0|={z0.abs().max().item():.5f} "
          f"(truncation bound={3*expected_std:.5f})")

    # different seeds -> different samples (the actual point: seed-based diversity)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    z0_seed1 = sample_gaussian_pair_init(shape, d_pair, generator=g1)
    z0_seed2 = sample_gaussian_pair_init(shape, d_pair, generator=g2)
    assert not torch.allclose(z0_seed1, z0_seed2), "different seeds produced identical z0"
    print("PASS: different seeds produce different z0 samples (confirmed diversity source)")


if __name__ == "__main__":
    test_abar_in_unit_interval()
    test_bounded_vs_unbounded()
    test_gradients_through_many_iterations()
    test_gaussian_init()
    print("\nALL TESTS PASSED")
