"""Gate tests for the pair-init noise scale (the replica-exchange "temperature" knob).

Two failure modes are silent and both are covered here:
  1. scale=1.0 not being bit-identical to the unscaled call would perturb every existing run that
     enables --gaussian_pair_init, including the live one, with no error anywhere.
  2. Scaling the std but NOT the truncation would clip the hot rungs of a ladder back toward the
     cold ones, so a ladder would silently compress at the top.
"""

import math

import pytest
import torch

from openfold.model.contractive_recycling import sample_gaussian_pair_init

C_Z = 128
SIGMA0 = math.sqrt(2.0 / (5.0 * C_Z))
SHAPE = (1, 16, 16, C_Z)


def _draw(scale=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    kw = {} if scale is None else {"scale": scale}
    return sample_gaussian_pair_init(SHAPE, C_Z, generator=g, **kw)


def test_scale_one_is_bit_identical_to_no_scale():
    """⛔ The live run passes no scale; the default must not move a single bit."""
    assert torch.equal(_draw(None), _draw(1.0))


def test_std_is_linear_in_scale():
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
        got = float(_draw(scale).std())
        want = scale * SIGMA0
        # trunc_normal_ at +/-3 sigma shaves the tails, so the realized std runs a few % under sigma
        assert want * 0.9 < got < want * 1.02, (scale, got, want)


def test_truncation_scales_with_the_std():
    """A fixed +/-3*sigma0 bound would clip a hot rung back toward the cold ones."""
    for scale in [0.5, 2.0, 8.0]:
        m = float(_draw(scale).abs().max())
        bound = 3.0 * scale * SIGMA0
        assert m <= bound + 1e-6, (scale, m, bound)
        assert m > 0.5 * bound, (scale, m, bound)      # actually reaches out toward the new bound


def test_scale_is_a_pure_multiple_of_the_same_draw():
    """Same generator seed => the scaled sample is exactly `scale` times the unscaled one, so a
    ladder's rungs are the SAME random field at different widths, not unrelated fields."""
    a = _draw(1.0, seed=7)
    b = _draw(4.0, seed=7)
    assert torch.allclose(b, 4.0 * a, atol=1e-6, rtol=1e-5)


def test_layernorm_makes_the_scale_a_noop_on_the_plain_path():
    """⛔⛔ The reason --gaussian_pair_init_scale is documented as contractive-only: LayerNorm is
    scale-invariant, so on the plain-additive recycling path the knob does nothing real. The tell is
    SATURATION -- scale=4 and scale=100 deviate from scale=1 by the same amount, which is LayerNorm's
    eps, not a signal change."""
    ln = torch.nn.LayerNorm(C_Z)
    z = _draw(1.0, seed=3)
    base = ln(z)
    d4 = float((ln(4.0 * z) - base).abs().max())
    d100 = float((ln(100.0 * z) - base).abs().max())
    assert d4 < 1e-2 and d100 < 1e-2
    assert d100 == pytest.approx(d4, rel=0.1)


def test_contractive_path_does_see_the_scale():
    """The other half of the same point: with use_contractive the pair state is used RAW."""
    from openfold.model.contractive_recycling import ContractivePairUpdate

    cpu = ContractivePairUpdate(C_Z)
    z = _draw(1.0, seed=5)
    u = torch.randn(SHAPE, generator=torch.Generator().manual_seed(11))
    ref = cpu(z, u)
    d1 = float((cpu(2.0 * z, u) - ref).abs().max())
    d2 = float((cpu(8.0 * z, u) - ref).abs().max())
    assert d1 > 1e-3
    assert d2 > 3.0 * d1          # grows with the scale instead of saturating


def test_config_carries_the_default():
    from openfold.config import model_config

    cfg = model_config("finetuning_ptm", train=True)
    assert cfg.model.recycling_embedder.gaussian_pair_init_scale == 1.0


def test_scale_zero_is_exact_zeros_not_a_crash():
    """⛔⛔ scale=0 is the deterministic 'T=0' rung of a temperature ladder, not an invalid input.
    torch's trunc_normal_ computes norm_cdf((a-mean)/std) and raises ZeroDivisionError on std=0, so
    the sampler must short-circuit. A ladder containing 0 would otherwise kill training at step 1."""
    z = _draw(0.0)
    assert torch.equal(z, torch.zeros_like(z))
    assert z.shape == SHAPE


def test_scale_zero_matches_the_stock_all_zero_pair_init():
    """The point of the rung: it must be bit-identical to what use_gaussian_pair_init=False does."""
    assert torch.equal(_draw(0.0), torch.zeros(SHAPE))
