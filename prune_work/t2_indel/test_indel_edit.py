import numpy as np
import pytest

from indel_edit import edit, validate, write_pdb


def chain(L, seed=0):
    return np.random.default_rng(seed).normal(size=(L, 4, 3)) * 5


def test_no_ops_is_identity():
    c = chain(10)
    x, orig, n2n = edit(c, [])
    assert np.array_equal(x, c) and orig.tolist() == list(range(10)) and n2n.tolist() == list(range(10))


def test_interior_insertion_exact_interpolation():
    c = chain(8)
    x, orig, n2n = edit(c, [("ins", 3, 4)])
    assert x.shape == (12, 4, 3)
    assert orig.tolist() == [0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7]
    for j in range(1, 5):
        assert np.allclose(x[3 + j], c[3] + j / 5 * (c[4] - c[3]), atol=1e-9)
    assert np.array_equal(x[8], c[4]) and n2n[4] == 8 and n2n[3] == 3


def test_deletion_mapping():
    c = chain(10)
    x, orig, n2n = edit(c, [("del", 2, 4)])
    assert x.shape == (7, 4, 3)
    assert orig.tolist() == [0, 1, 5, 6, 7, 8, 9]
    assert n2n.tolist() == [0, 1, -1, -1, -1, 2, 3, 4, 5, 6]
    assert np.array_equal(x[2], c[5])


def test_terminal_insertions_extrapolate():
    c = chain(6)
    x, orig, _ = edit(c, [("ins", -1, 3), ("ins", 5, 2)])
    assert orig.tolist() == [-1, -1, -1, 0, 1, 2, 3, 4, 5, -1, -1]
    for j in (1, 2, 3):  # residue nearest the chain is j=1
        assert np.allclose(x[3 - j], c[0] - j * (c[1] - c[0]), atol=1e-9)
    for j in (1, 2):
        assert np.allclose(x[8 + j], c[5] + j * (c[5] - c[4]), atol=1e-9)


def test_insert_and_delete_together_use_surviving_neighbours():
    c = chain(10)
    x, orig, _ = edit(c, [("del", 4, 5), ("ins", 3, 2)])
    assert orig.tolist() == [0, 1, 2, 3, -1, -1, 6, 7, 8, 9]
    for j in (1, 2):
        assert np.allclose(x[3 + j], c[3] + j / 3 * (c[6] - c[3]), atol=1e-9)


def test_insertion_after_deleted_boundary_residue():
    c = chain(10)
    x, orig, _ = edit(c, [("del", 4, 5), ("ins", 5, 1)])  # site 5: residue 5 deleted, 6 survives
    assert orig.tolist() == [0, 1, 2, 3, -1, 6, 7, 8, 9]
    assert np.allclose(x[4], c[3] + 0.5 * (c[6] - c[3]), atol=1e-9)


def test_terminal_deletion_then_terminal_insertion():
    c = chain(8)
    x, orig, _ = edit(c, [("del", 0, 1), ("ins", -1, 2)])
    assert orig.tolist() == [-1, -1, 2, 3, 4, 5, 6, 7]
    assert np.allclose(x[1], c[2] - (c[3] - c[2]), atol=1e-9)


def test_validation_rejects_bad_ops():
    for ops in ([("del", 2, 4), ("del", 4, 6)], [("ins", 3, 1), ("ins", 3, 2)],
                [("del", 3, 6), ("ins", 4, 2)], [("del", 0, 9)], [("ins", 3, 0)], [("foo", 1, 2)]):
        with pytest.raises((AssertionError, ValueError)):
            validate(10, ops)


def test_pdb_write_roundtrip(tmp_path):
    c = chain(5)
    p = tmp_path / "a.pdb"
    write_pdb(p, c, ["GLY"] * 5)
    rows = [ln for ln in open(p) if ln.startswith("ATOM")]
    assert len(rows) == 20
    xyz = np.array([[float(r[30:38]), float(r[38:46]), float(r[46:54])] for r in rows]).reshape(5, 4, 3)
    assert np.allclose(xyz, c, atol=1e-3)
    assert [int(r[22:26]) for r in rows][::4] == [1, 2, 3, 4, 5]
