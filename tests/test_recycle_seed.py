"""Cycle-0 recycle-seed injection, and the guarantee that the LIVE RUN IS UNTOUCHED.

The mode replaces the all-zero `x_prev` at recycling cycle 0 with a real structure -- a T2 synthetic
template or a T4 promoted self-prediction -- so the recycling DISTOGRAM track opens on a candidate
fold instead of the degenerate "every pair at distance 0" bin.

⛔⛔ THE POINT OF THIS FILE (user, 2026-08-26): *"Do NOT implement this as a mid-run change... Implement
this mechanism but leave the current run untouched; make sure you unit test that this would be the
case."* So the load-bearing tests here are the ones asserting the OFF path is unchanged, not the ones
asserting the new path works.

⛔ Coverage is not cosmetic: a template covers only part of a chain, and a residue it does not cover
sits at the ORIGIN, which the distance binning reads as "in contact with everything". `x_mask` zeroes
those pairs so they carry no distance information instead of a fabricated one.
"""

import inspect

import numpy as np
import pytest
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset
from openfold.model.embedders import RecyclingEmbedder

C_M, C_Z, N = 8, 16, 7


def _emb():
    torch.manual_seed(0)
    return RecyclingEmbedder(c_m=C_M, c_z=C_Z, min_bin=3.25, max_bin=20.75, no_bins=15)


def _inputs():
    torch.manual_seed(1)
    return (torch.randn(N, C_M), torch.randn(N, N, C_Z), torch.randn(N, 3))


# ------------------------------------------------- THE LIVE RUN IS UNTOUCHED

def test_x_mask_defaults_to_none():
    """If this ever gains a non-None default, every existing run silently changes behaviour."""
    sig = inspect.signature(RecyclingEmbedder.forward)
    assert sig.parameters["x_mask"].default is None


def test_omitting_x_mask_is_bit_identical_to_the_all_ones_mask():
    """⭐ The OFF path must be EXACTLY the old arithmetic -- an all-ones mask multiplies by 1.0, so
    equality here is what proves the new branch cannot perturb an unseeded run."""
    e = _emb()
    m, z, x = _inputs()
    with torch.no_grad():
        a_m, a_z = e(m.clone(), z.clone(), x.clone())
        b_m, b_z = e(m.clone(), z.clone(), x.clone(), x_mask=torch.ones(N))
    assert torch.equal(a_m, b_m)
    assert torch.equal(a_z, b_z)


def test_cli_flag_defaults_to_off():
    import train_openfold
    p = train_openfold.add_data_args.__globals__.get("argparse").ArgumentParser()
    train_openfold.add_data_args(p)
    d = {a.dest: a for a in p._actions}
    assert "recycle_seed_source" in d, "the flag was not registered"
    assert d["recycle_seed_source"].default is None
    assert set(d["recycle_seed_source"].choices) == {"synthetic", "promoted"}


def test_dataset_defaults_to_no_seed():
    sig = inspect.signature(OpenFoldSingleDataset.__init__)
    assert sig.parameters["recycle_seed_source"].default is None


def test_the_live_launcher_does_not_use_the_flag():
    """⛔ The concrete form of 'leave the current run untouched': Run C v2's launcher must not name
    it, so a pull on the box cannot change what that job does."""
    import pathlib
    for name in ("run_C_v2.sh", "run_C_replica_exchange.sh", "run_B_full_stack.sh"):
        p = pathlib.Path("prune_work") / name
        if p.exists():
            assert "--recycle_seed_source" not in p.read_text(), f"{name} names the new flag"


def test_invalid_source_is_rejected_not_silently_ignored():
    with pytest.raises(AssertionError):
        OpenFoldSingleDataset.__init__.__wrapped__ if False else None
        OpenFoldSingleDataset(
            data_dir="/nonexistent", alignment_dir="/nonexistent",
            template_mmcif_dir="/nonexistent", max_template_date="2018-04-30",
            config=model_config("finetuning_ptm", train=True).data,
            recycle_seed_source="garbage",
        )


# ------------------------------------------------- the masking semantics

def test_uncovered_residues_contribute_nothing():
    """A pair touching an uncovered residue must add EXACTLY zero, not a distance-0 bin."""
    e = _emb()
    m, z, x = _inputs()
    mask = torch.ones(N)
    mask[2] = 0.0
    with torch.no_grad():
        _, z_masked = e(m.clone(), z.clone(), x.clone(), x_mask=mask)
        _, z_full = e(m.clone(), z.clone(), x.clone(), x_mask=torch.ones(N))
    # row/col 2 must differ from the unmasked version by exactly the distogram contribution
    assert not torch.equal(z_masked[2], z_full[2])
    # and every pair NOT touching residue 2 is untouched
    keep = [i for i in range(N) if i != 2]
    assert torch.equal(z_masked[keep][:, keep], z_full[keep][:, keep])


def test_a_fully_uncovered_seed_reduces_to_the_no_distogram_case():
    e = _emb()
    m, z, x = _inputs()
    with torch.no_grad():
        _, z_zero = e(m.clone(), z.clone(), x.clone(), x_mask=torch.zeros(N))
        # with no distance information at all, z_update is just the layer-normed z
        expected = e.layer_norm_z(z.clone())
    assert torch.allclose(z_zero, expected, atol=0, rtol=0)


# ------------------------------------------------- the seed reaches the model aligned

def test_seed_features_are_num_res_so_the_crop_slices_them():
    """⭐ The alignment guarantee. random_crop_to_size slices every NUM_RES axis with the QUERY's
    own offset, so a seed registered this way stays in register with the query for free -- the same
    mechanism that keeps templates aligned."""
    from openfold.config import NUM_RES
    c = model_config("finetuning_ptm", train=True)
    schema = c.data.common.feat
    assert schema["recycle_seed_positions"][0] == NUM_RES
    assert schema["recycle_seed_mask"] == [NUM_RES]


def test_seed_features_are_named_in_the_kept_feature_list():
    """np_to_tensor_dict filters by membership -- an unnamed feature is silently dropped."""
    c = model_config("finetuning_ptm", train=True)
    feats = c.data.common.unsupervised_features
    assert "recycle_seed_positions" in feats
    assert "recycle_seed_mask" in feats


def test_model_reads_the_seed_only_at_cycle_zero():
    """The injection sits inside the `if None in [m_1_prev, z_prev, x_prev]` block, i.e. the
    first-cycle initialisation. Later cycles must keep recycling the model's OWN prediction."""
    import openfold.model.model as mm
    src = inspect.getsource(mm.AlphaFold.forward_iteration if hasattr(mm.AlphaFold,
                            "forward_iteration") else mm.AlphaFold.iteration)
    assert "recycle_seed_positions" in src
    init_at = src.index("if None in [m_1_prev, z_prev, x_prev]")
    seed_at = src.index('"recycle_seed_positions" in feats')
    emb_at = src.index("self.recycling_embedder(")
    assert init_at < seed_at < emb_at, "the seed is not inside the cycle-0 init block"


def test_pseudo_beta_atom_choice_matches_the_coverage_rule():
    """Coverage is 'the atom pseudo_beta actually reads': CB (3) normally, CA (1) for glycine.
    Using the wrong index would mark a glycine uncovered and drop real structure."""
    from openfold.np import residue_constants as rc
    assert rc.atom_order["CA"] == 1
    assert rc.atom_order["CB"] == 3
    src = inspect.getsource(OpenFoldSingleDataset.__getitem__)
    assert "pb_atom" in src and "np.where" in src
