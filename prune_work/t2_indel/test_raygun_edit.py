import numpy as np

from indel_edit import edit
from raygun_edit import alignment_pairs, ops_from_pairs


def run(native, gen):
    L = len(native)
    pairs = alignment_pairs(native, gen)
    ops = ops_from_pairs(L, len(gen), pairs)
    coords = np.zeros((L, 4, 3)) + np.arange(L)[:, None, None]
    new, orig, _ = edit(coords, ops)
    return pairs, ops, orig


def check_consistent(native, gen):
    pairs, ops, orig = run(native, gen)
    assert len(orig) == len(gen)
    expect = {g: n for n, g in pairs}
    assert [expect.get(j, -1) for j in range(len(gen))] == orig.tolist()
    return ops


def test_identical_has_no_ops():
    assert check_consistent("ACDEFGHIKL", "ACDEFGHIKL") == []


def test_mismatch_is_aligned_not_indel():
    assert check_consistent("ACDEFGHIKL", "ACDEWGHIKL") == []


def test_single_internal_deletion_and_insertion():
    ops = check_consistent("ACDEFGHIKLMNPQ", "ACDEFHIKLMNPQ")
    assert [o[0] for o in ops] == ["del"]
    ops = check_consistent("ACDEFGHIKLMNPQ", "ACDEFWWGHIKLMNPQ")
    assert [o[0] for o in ops] == ["ins"] and ops[0][2] == 2


def test_terminal_indels():
    native = "MKTAYIAKQRQISFVKSHFSRQ"
    check_consistent(native, "GG" + native)
    check_consistent(native, native + "GGG")
    check_consistent(native, native[3:])
    check_consistent(native, native[:-4])


def test_many_scattered_gaps_stay_consistent():
    rng = np.random.default_rng(0)
    for _ in range(200):
        native = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=int(rng.integers(30, 90))))
        gen = list(native)
        for _ in range(int(rng.integers(1, 6))):
            p = int(rng.integers(0, len(gen)))
            if rng.random() < 0.5 and len(gen) > 12:
                del gen[p:p + int(rng.integers(1, 4))]
            else:
                gen[p:p] = list(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=int(rng.integers(1, 4))))
        for p in rng.integers(0, len(gen), size=3):
            gen[int(p)] = "W"
        check_consistent(native, "".join(gen))
