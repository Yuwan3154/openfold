"""B is a full [c_z, c_z] matrix, matching esm/models/esmfold2/model.py's `parcae_b_cont`.

The gate that matters: identity init must reproduce the OLD per-channel behaviour exactly, so the
change is a capacity extension and not a silent numerical change to any existing run.
"""
import math

import pytest
import torch

from openfold.model.contractive_recycling import ContractivePairUpdate

C_Z = 16


def test_b_is_a_full_matrix_initialised_to_identity():
    m = ContractivePairUpdate(C_Z)
    assert m.b.shape == (C_Z, C_Z)
    assert torch.equal(m.b, torch.eye(C_Z))


def test_b_bar_is_diag_delta_at_init():
    m = ContractivePairUpdate(C_Z)
    _, b_bar = m.discretized_params()
    delta = torch.nn.functional.softplus(torch.zeros(C_Z))
    assert torch.allclose(b_bar, torch.diag(delta))


def test_identity_init_matches_the_old_elementwise_update_exactly():
    m = ContractivePairUpdate(C_Z)
    torch.manual_seed(0)
    z = torch.randn(2, 5, 5, C_Z)
    u = torch.randn(2, 5, 5, C_Z)
    got = m(z, u)
    # the pre-matrix formulation, written out
    a_bar = torch.exp(-torch.nn.functional.softplus(m.log_delta) * torch.exp(m.log_a))
    b_bar_vec = torch.nn.functional.softplus(m.log_delta) * torch.ones(C_Z)
    want = a_bar * z + b_bar_vec * m.layer_norm_u(u)
    assert torch.allclose(got, want, atol=0, rtol=0), (got - want).abs().max()


def test_off_diagonal_b_actually_mixes_channels():
    m = ContractivePairUpdate(C_Z)
    with torch.no_grad():
        m.b.copy_(torch.eye(C_Z).roll(1, dims=0))      # pure channel permutation
    z = torch.zeros(1, 3, 3, C_Z)
    u = torch.zeros(1, 3, 3, C_Z)
    u[..., 0] = 5.0
    out = m(z, u)
    # a permuted B must move the response off channel 0; the vector form never could
    assert out[..., 0].abs().max() < out.abs().max()


def test_vector_checkpoint_migrates_to_diag_losslessly():
    old = ContractivePairUpdate(C_Z)
    sd = old.state_dict()
    sd["b"] = torch.linspace(0.5, 2.0, C_Z)            # a pre-matrix checkpoint
    new = ContractivePairUpdate(C_Z)
    missing, unexpected = new.load_state_dict(sd, strict=True), None
    assert new.b.shape == (C_Z, C_Z)
    assert torch.allclose(new.b, torch.diag(sd["b"]))
    # and it reproduces what the old code would have computed with that vector
    torch.manual_seed(1)
    z, u = torch.randn(1, 4, 4, C_Z), torch.randn(1, 4, 4, C_Z)
    a_bar = torch.exp(-torch.nn.functional.softplus(new.log_delta) * torch.exp(new.log_a))
    delta = torch.nn.functional.softplus(new.log_delta)
    want = a_bar * z + (delta * torch.diag(new.b)) * new.layer_norm_u(u)
    assert torch.allclose(new(z, u), want, atol=1e-6)


def test_contraction_still_holds_for_any_learned_b():
    """a_bar in (0,1) is what bounds the state, and it does not involve B at all."""
    m = ContractivePairUpdate(C_Z)
    with torch.no_grad():
        m.log_delta.uniform_(-5, 5)
        m.log_a.uniform_(-5, 5)
        m.b.normal_(0, 3)                              # B far from identity
    a_bar, _ = m.discretized_params()
    assert (a_bar > 0).all() and (a_bar < 1).all()
