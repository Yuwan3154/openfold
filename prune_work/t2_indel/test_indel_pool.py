import json

import numpy as np

import indel_pool as ip


def fake_item(L, arm, draw, rng):
    mask = np.zeros((L, 37), bool)
    mask[:, :5] = True
    mask[::3, 5:8] = True
    return dict(arm=arm, model="cc89", rewind=250, draw=draw, coords=rng.normal(size=(int(mask.sum()), 3)),
                atom_mask=mask, aatype=rng.integers(0, 20, L), orig_idx=np.where(np.arange(L) % 5 == 0, -1, np.arange(L)),
                ops=[["ins", 3, 2]], tm_native=0.5 + 0.01 * draw, tm_template=0.4, L_native=60)


def test_roundtrip_ragged(tmp_path):
    rng = np.random.default_rng(0)
    items = [fake_item(L, a, d, rng) for d, (L, a) in enumerate([(50, "comp"), (71, "comp"), (63, "raygun1dir")])]
    path = tmp_path / "x_A.npz"
    np.savez(path, **ip.pack_chain("x_A", items))
    for i, it in enumerate(items):
        t = ip.read_template(path, i)
        assert np.array_equal(t["aatype"], it["aatype"]) and np.array_equal(t["atom_mask"], it["atom_mask"])
        assert np.array_equal(t["coords"], it["coords"].astype(np.float32))
        assert np.array_equal(t["orig_idx"], it["orig_idx"]) and t["arm"] == it["arm"] and t["draw"] == it["draw"]
        assert np.array_equal(t["residue_index"], np.arange(1, len(it["aatype"]) + 1))
        assert t["ops"] == it["ops"] and abs(t["tm_native"] - it["tm_native"]) < 1e-6
        assert ip.atom37_coords(t).shape == (len(it["aatype"]), 37, 3)
    z = np.load(path)
    assert z["res_offsets"].tolist() == [0, 50, 121, 184] and int(z["n_templates"]) == 3


def test_mismatched_mask_rejected(tmp_path):
    import pytest
    it = fake_item(40, "comp", 0, np.random.default_rng(1))
    it["coords"] = it["coords"][:-1]
    with pytest.raises(AssertionError):
        ip.pack_chain("y", [it])
