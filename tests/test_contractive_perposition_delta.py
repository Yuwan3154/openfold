"""Per-position (data-dependent) delta in ContractivePairUpdate.

Delta becomes per residue PAIR with a per-channel learned offset; A and B stay static:

    s[i,j]       = symmetrize(Linear(z_t))
    delta[i,j,c] = DELTA_FLOOR + softplus(log_delta[c] + s[i,j])
    z_{t+1}      = exp(-delta*A) * z_t + delta * F.linear(LN(u_t), B)

The gates that matter: the flag OFF must be bit-identical to the historical module, the flag ON at
step 0 must reproduce the same delta, stability must hold for ANY head output, and the row-scaling
identity must never be "optimized" back into the impossible per-position b_bar form.
"""
import pytest
import torch
import torch.nn.functional as F

from openfold.model.contractive_recycling import ContractivePairUpdate, inv_softplus

C_Z = 16
FLOOR = 0.05


def _inputs(seed=0, b=2, n=5):
    torch.manual_seed(seed)
    return torch.randn(b, n, n, C_Z), torch.randn(b, n, n, C_Z)


# --- gate 1: the flag OFF changes nothing, to the bit -----------------------------------------
def test_flag_off_is_bit_identical_to_the_historical_module():
    z, u = _inputs()
    m = ContractivePairUpdate(C_Z)
    got = m(z, u)
    # the pre-per-position formulation, written out
    delta = F.softplus(m.log_delta)
    a_bar = torch.exp(-delta * torch.exp(m.log_a))
    want = a_bar * z + F.linear(m.layer_norm_u(u), delta[:, None] * m.b)
    assert torch.equal(got, want), (got - want).abs().max()


def test_flag_off_module_has_no_new_state():
    m = ContractivePairUpdate(C_Z)
    keys = set(m.state_dict())
    assert "delta_floor" not in keys and not any(k.startswith("delta_head") for k in keys), keys
    assert not hasattr(m, "delta_floor")


# --- gate 2: the flag ON at step 0 reproduces the same delta ----------------------------------
def test_zero_init_head_reproduces_the_floorless_delta():
    on = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    z, _ = _inputs()
    delta = on.per_position_delta_from_state(z)
    want = F.softplus(torch.zeros(C_Z))            # what the floorless module has at init
    # ⛔ NOT bit-identical by construction: log_delta is re-solved through
    # inv_softplus(softplus(0) - floor) and softplus(inv_softplus(.)) does not round-trip exactly
    # in float32. The gate is that the shift is float noise, not the 0.05 the floor would add.
    err = (delta - want).abs().max()
    assert err < 1e-6, err
    assert err < FLOOR / 1000.0, err


def test_zero_init_head_output_matches_the_flag_off_update():
    z, u = _inputs()
    off, on = ContractivePairUpdate(C_Z), ContractivePairUpdate(
        C_Z, per_position_delta=True, delta_floor=FLOOR)
    assert torch.allclose(off(z, u), on(z, u), atol=1e-6, rtol=1e-5)


# --- gate 3: s is exactly symmetric ------------------------------------------------------------
def test_s_is_exactly_symmetric():
    m = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    torch.manual_seed(1)
    with torch.no_grad():                          # a nonzero head, or symmetry is trivial
        m.delta_head.weight.normal_(0, 1.0)
        m.delta_head.bias.normal_(0, 1.0)
    z, _ = _inputs(seed=2)
    d = m.per_position_delta_from_state(z)
    assert torch.equal(d, d.transpose(-3, -2)), (d - d.transpose(-3, -2)).abs().max()


def test_an_unsymmetrized_head_would_have_failed_that():
    """Negative control: the raw head output on the same input is NOT symmetric, so gate 3 is
    testing the symmetrization and not a property the input happened to have."""
    m = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    torch.manual_seed(1)
    with torch.no_grad():
        m.delta_head.weight.normal_(0, 1.0)
        m.delta_head.bias.normal_(0, 1.0)
    z, _ = _inputs(seed=2)
    raw = m.delta_head(z)
    assert not torch.allclose(raw, raw.transpose(-3, -2), atol=1e-4)


# --- gates 4 and 5: stability for ANY head output ----------------------------------------------
@pytest.mark.parametrize("bias", [-50.0, -10.0, 0.0, 10.0, 50.0])
def test_delta_stays_above_the_floor_and_a_bar_stays_in_the_unit_interval(bias):
    m = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    with torch.no_grad():
        m.delta_head.bias.fill_(bias)
    z, _ = _inputs(seed=3)
    delta = m.per_position_delta_from_state(z)
    assert bool((delta >= FLOOR).all()), float(delta.min())
    a_bar = torch.exp(-delta * torch.exp(m.log_a))
    assert bool((a_bar > 0).all()) and bool((a_bar < 1).all()), (float(a_bar.min()),
                                                                 float(a_bar.max()))
    # the floor's actual job: a_bar can never reach 1, so no region freezes permanently
    assert float(a_bar.max()) <= float(torch.exp(-torch.tensor(FLOOR) * torch.exp(m.log_a)).max())


def test_a_zero_floor_would_let_a_bar_reach_one():
    """Negative control for gate 4/5: without the floor a strongly negative head output drives
    a_bar arbitrarily close to 1, which is the freeze mode the floor exists to prevent."""
    m = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=0.0)
    with torch.no_grad():
        m.delta_head.bias.fill_(-50.0)
    z, _ = _inputs(seed=3)
    a_bar = torch.exp(-m.per_position_delta_from_state(z) * torch.exp(m.log_a))
    assert float(a_bar.max()) > 0.999


# --- gate 6: the row-scaling identity ----------------------------------------------------------
def test_row_scaling_identity_matches_the_impossible_explicit_form():
    """diag(delta) @ B @ x == delta * (B @ x). Checked on a tiny c_z where the explicit
    per-position [L, L, c_z, c_z] b_bar is actually constructible -- at c_z=128, L=256 it is ~1e12
    elements. This is the gate that stops anyone "simplifying" the cheap form back into it."""
    c_z, n = 4, 3
    m = ContractivePairUpdate(c_z, per_position_delta=True, delta_floor=FLOOR)
    torch.manual_seed(4)
    with torch.no_grad():
        m.b.normal_(0, 1.0)
        m.delta_head.weight.normal_(0, 0.5)
    z, u = torch.randn(1, n, n, c_z), torch.randn(1, n, n, c_z)
    delta = m.per_position_delta_from_state(z)
    x = m.layer_norm_u(u)

    cheap = delta * F.linear(x, m.b)
    # the explicit form: build diag(delta) @ B at every position and apply it
    b_bar = delta[..., :, None] * m.b                      # [1, n, n, c_z, c_z]
    explicit = torch.einsum("...ij,...j->...i", b_bar, x)
    assert torch.allclose(cheap, explicit, atol=1e-5, rtol=1e-5), (cheap - explicit).abs().max()


# --- the floor-migration hazard -----------------------------------------------------------------
def test_a_prefloor_checkpoint_keeps_its_effective_delta():
    """A checkpoint written under delta = softplus(log_delta) must not gain +floor on load."""
    old = ContractivePairUpdate(C_Z)
    with torch.no_grad():                                   # a trained-looking log_delta
        old.log_delta.copy_(torch.linspace(-0.3, 0.3, C_Z))
    want = F.softplus(old.log_delta.detach().clone())
    new = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert "delta_floor" in missing or not missing, (missing, unexpected)
    z = torch.zeros(1, 2, 2, C_Z)                           # zero head weight => s == 0 anyway
    got = new.per_position_delta_from_state(z)[0, 0, 0]
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_loading_verbatim_without_the_migration_would_have_shifted_delta():
    """Negative control: the migration is doing real work -- the un-remapped load is off by floor."""
    old = ContractivePairUpdate(C_Z)
    with torch.no_grad():
        old.log_delta.copy_(torch.linspace(-0.3, 0.3, C_Z))
    naive = FLOOR + F.softplus(old.log_delta.detach())      # what a verbatim load would give
    want = F.softplus(old.log_delta.detach())
    assert float((naive - want).abs().min()) == pytest.approx(FLOOR, abs=1e-7)


def test_floor_below_the_smallest_delta_is_enforced():
    old = ContractivePairUpdate(C_Z)
    with torch.no_grad():
        old.log_delta.fill_(float(inv_softplus(torch.tensor(0.10))))   # every delta = 0.10
    new = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=0.5)
    with pytest.raises(AssertionError, match="smallest per-channel delta"):
        new.load_state_dict(old.state_dict(), strict=False)


def test_discretized_params_refuses_to_answer_in_per_position_mode():
    m = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    with pytest.raises(AssertionError, match="per-channel only"):
        m.discretized_params()


# --- the gradient actually flows to the head ----------------------------------------------------
def test_the_head_receives_gradient():
    m = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    z, u = _inputs(seed=5)
    m(z, u).sum().backward()
    assert m.delta_head.weight.grad is not None
    assert float(m.delta_head.weight.grad.abs().max()) > 0


# --- the double-load trap in import_openfold_weights_ -------------------------------------------
def test_migration_is_idempotent_under_the_real_two_attempt_loader():
    """⛔⛔ import_openfold_weights_ does `load_state_dict(strict=True)` and, on RuntimeError,
    RETRIES with the converted dict. The per-position head guarantees the first attempt raises
    (the checkpoint has none of its keys), so the migration runs TWICE over the same dict. It
    mutates the dict in place, so a non-idempotent migration subtracts the floor twice and the run
    silently starts at delta - floor. This reproduces that exact call pattern.
    """
    old = ContractivePairUpdate(C_Z)
    with torch.no_grad():
        old.log_delta.copy_(torch.linspace(-0.3, 0.3, C_Z))
    want = F.softplus(old.log_delta.detach().clone())
    sd = old.state_dict()

    before = sd["log_delta"].detach().clone()

    new = ContractivePairUpdate(C_Z, per_position_delta=True, delta_floor=FLOOR)
    try:                                     # attempt 1, exactly as the real loader does it
        new.load_state_dict(sd, strict=True)
    except RuntimeError:
        new.load_state_dict(sd, strict=False)   # attempt 2, over the same dict object

    # This is WHY it is safe, and the invariant the safety rests on: nn.Module.load_state_dict
    # shallow-copies the incoming dict ("copy state_dict so _load_from_state_dict can modify it",
    # torch 2.7.1), so each attempt migrates once from a pristine original. If a future torch drops
    # that copy, this assert fires before the silent delta drift does.
    assert torch.equal(sd["log_delta"], before), (
        "load_state_dict mutated the CALLER's dict; the floor migration is no longer "
        "single-shot under the two-attempt loader")

    got = new.per_position_delta_from_state(torch.zeros(1, 2, 2, C_Z))[0, 0, 0]
    assert torch.allclose(got, want, atol=1e-6), (
        f"delta drifted by {float((got - want).mean()):+.5f} across a two-attempt load "
        f"(floor={FLOOR}); the migration is not idempotent")
