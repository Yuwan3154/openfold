"""Gate tests for --resume_from_ema (train_openfold.select_ema_warmstart_weights).

⛔⛔ Validation swaps the EMA weights in for the duration of every val epoch, so a `best-*`
checkpoint's score describes its EMA weights -- the live `state_dict` at that step was never
evaluated. Warm-starting a new run from `state_dict` therefore silently starts from an unmeasured
model, and the resulting epoch-0 val cannot be compared to the score that picked the checkpoint.

The failure mode these tests exist to prevent is the SILENT one: falling back to the live weights
when the EMA is absent, which would look like a successful warm start.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_openfold import select_ema_warmstart_weights  # noqa: E402


def _ckpt(live=1.0, ema=2.0, with_ema=True):
    ck = {"state_dict": {"model.evoformer.w": torch.full((3,), live),
                         "model.recycling_embedder.contractive_pair_update.b": torch.full((4,), live)}}
    if with_ema:
        ck["ema"] = {"params": {"evoformer.w": torch.full((3,), ema),
                                "recycling_embedder.contractive_pair_update.b": torch.full((4,), ema)}}
    return ck


def test_flag_off_returns_none_so_the_legacy_path_is_untouched():
    """Default OFF must be a no-op: every existing launcher has to keep loading state_dict."""
    assert select_ema_warmstart_weights(_ckpt(), resume_from_ema=False) is None


def test_flag_off_is_a_noop_even_when_no_ema_exists():
    assert select_ema_warmstart_weights(_ckpt(with_ema=False), resume_from_ema=False) is None


def test_ema_weights_are_returned_not_the_live_ones():
    sd = select_ema_warmstart_weights(_ckpt(live=1.0, ema=2.0), resume_from_ema=True)
    assert torch.allclose(sd["model.evoformer.w"], torch.full((3,), 2.0))
    # negative control: the live value must NOT be what came back
    assert not torch.allclose(sd["model.evoformer.w"], torch.full((3,), 1.0))


def test_keys_get_the_wrapper_prefix():
    """EMA params are stored unprefixed; OpenFoldWrapper holds the model at `model.`, so an
    unprefixed load would match nothing and (with strict=False) silently train from scratch."""
    sd = select_ema_warmstart_weights(_ckpt(), resume_from_ema=True)
    assert all(k.startswith("model.") for k in sd)
    assert set(sd) == {"model.evoformer.w",
                       "model.recycling_embedder.contractive_pair_update.b"}


def test_missing_ema_raises_instead_of_falling_back_silently():
    """⛔ The whole point of the flag. A fall-back would look like a successful warm start."""
    with pytest.raises(ValueError, match="no ema/params"):
        select_ema_warmstart_weights(_ckpt(with_ema=False), resume_from_ema=True, ckpt_path="x.ckpt")


def test_empty_ema_params_also_raises():
    ck = _ckpt()
    ck["ema"] = {"params": {}}
    with pytest.raises(ValueError, match="no ema/params"):
        select_ema_warmstart_weights(ck, resume_from_ema=True)


def test_ema_present_but_none_raises():
    ck = _ckpt()
    ck["ema"] = None
    with pytest.raises(ValueError, match="no ema/params"):
        select_ema_warmstart_weights(ck, resume_from_ema=True)


def test_selection_does_not_mutate_the_checkpoint():
    """The caller's `sd` is reused by the surrounding branches; mutating it would corrupt them."""
    ck = _ckpt(live=1.0, ema=2.0)
    before_live = ck["state_dict"]["model.evoformer.w"].clone()
    before_keys = set(ck["ema"]["params"])
    select_ema_warmstart_weights(ck, resume_from_ema=True)
    assert torch.allclose(ck["state_dict"]["model.evoformer.w"], before_live)
    assert set(ck["ema"]["params"]) == before_keys


def test_vector_b_survives_selection_for_the_migration_hook():
    """Run C loads a ckpt whose `b` is a per-channel VECTOR into a full-matrix parameter. The
    expansion is done by ContractivePairUpdate._load_from_state_dict, so selection must hand the
    1-D tensor through untouched -- pre-expanding here would bypass that hook."""
    sd = select_ema_warmstart_weights(_ckpt(), resume_from_ema=True)
    assert sd["model.recycling_embedder.contractive_pair_update.b"].dim() == 1
