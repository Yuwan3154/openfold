import numpy as np

from make_indel_inputs import ATOM37, SLOT, apply_plan, write_pdb


def fake_chain(L=12):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(L, 37, 3)) * 5
    mask = np.zeros((L, 37), bool)
    mask[:, [SLOT[n] for n in ("N", "CA", "C", "O", "CB")]] = True
    return ["ALA"] * L, xyz, mask


def test_survivors_keep_all_atoms_and_inserted_are_gly_backbone_only():
    names, xyz, mask = fake_chain()
    nn, nx, nm, orig, n2n = apply_plan(names, xyz, mask, [("ins", 3, 2), ("del", 7, 8)])
    assert len(nn) == 12 + 2 - 2
    for i, o in enumerate(orig):
        if o >= 0:
            assert np.array_equal(nx[i], xyz[o]) and np.array_equal(nm[i], mask[o]) and nn[i] == "ALA"
        else:
            assert nn[i] == "GLY" and nm[i].sum() == 4 and not nm[i][SLOT["CB"]]


def test_pdb_has_contiguous_numbering_and_only_present_atoms(tmp_path):
    names, xyz, mask = fake_chain()
    nn, nx, nm, orig, _ = apply_plan(names, xyz, mask, [("del", 0, 1), ("ins", 5, 3)])
    p = tmp_path / "e.pdb"
    write_pdb(p, nn, nx, nm)
    rows = [ln for ln in open(p) if ln.startswith("ATOM")]
    assert len(rows) == nm.sum()
    assert sorted({int(r[22:26]) for r in rows}) == list(range(1, len(nn) + 1))
    assert {r[17:20] for r in rows} == {"ALA", "GLY"}
    assert all(ATOM37[SLOT[r[12:16].strip()]] == r[12:16].strip() for r in rows)
