import numpy as np

from diagnose_indel import diag_item, native_backbone, pair_geoms, dssp
from indel_edit import edit


def helix_bb(L, noise=0.0, seed=0):
    # ideal alpha-helix CA trace with N/C/O placed deterministically; enough for DSSP-free geometry tests
    t = np.arange(L) * np.deg2rad(100)
    ca = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(L)], 1)
    bb = np.zeros((L, 4, 3))
    bb[:, 1] = ca
    bb[:, 0] = ca + np.array([-0.9, -0.5, -0.4])
    bb[:, 2] = ca + np.array([0.9, 0.5, 0.4])
    bb[:, 3] = ca + np.array([1.2, 0.8, 0.9])
    return bb + np.random.default_rng(seed).normal(scale=noise, size=bb.shape)


def test_unedited_has_no_seams_and_full_identity():
    bb = helix_bb(40)
    orig = np.arange(40)
    ss = dssp(bb)
    band = {"cn": (0.0, 99.0), "caca": (0.0, 99.0)}
    rec = diag_item(bb, orig, ss, band, 0.0)
    assert rec["n_seam"] == 0 and rec["n_bg"] == 39 and rec["q3_surv"] == 1.0 and rec["n_surv"] == 40


def test_seam_pairs_counted_for_deletion_and_insertion():
    bb = helix_bb(40)
    new, orig, _ = edit(bb, [("del", 10, 14), ("ins", 25, 3)])
    ss = dssp(bb)
    band = {"cn": (0.0, 99.0), "caca": (0.0, 99.0)}
    rec = diag_item(new, orig, ss, band, 0.0)
    # deletion seam: 1 pair; insertion block of 3 between two survivors: 4 pairs
    assert rec["n_seam"] == 5 and rec["n_seam"] + rec["n_bg"] == len(orig) - 1
    assert rec["n_surv"] == 40 - 5


def test_band_flags_a_stretched_seam():
    bb = helix_bb(30)
    cn, caca = pair_geoms(bb)
    band = {"cn": (cn.min() - 1e-6, cn.max() + 1e-6), "caca": (caca.min() - 1e-6, caca.max() + 1e-6)}
    new, orig, _ = edit(bb, [("del", 10, 19)])   # 10 deleted residues -> a long gap
    rec = diag_item(new, orig, dssp(bb), band, 0.0)
    assert rec["seam_caca_out"] == 1 and rec["bg_caca_out"] == 0 and rec["seam_cn_out"] == 1
