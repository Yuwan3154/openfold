import collections

import numpy as np

from indel_edit import edit, validate
from sample_indels import FRAC_HI, FRAC_LO, draw_plan, place_deletions, split_total


def test_split_total_sums_and_positive():
    rng = np.random.default_rng(1)
    for T in (1, 2, 5, 17, 100):
        for k in (1, 2, 5):
            segs = split_total(T, k, rng)
            assert sum(segs) == T and all(s >= 1 for s in segs)
            assert len(segs) == (k if T >= k else T)


def test_cap_when_total_below_k():
    rng = np.random.default_rng(0)
    assert split_total(3, 5, rng) == [1, 1, 1]
    assert split_total(1, 4, rng) == [1]


def test_split_total_uniform_over_compositions():
    rng = np.random.default_rng(2)
    c = collections.Counter(tuple(split_total(4, 2, rng)) for _ in range(30000))
    assert set(c) == {(1, 3), (2, 2), (3, 1)}
    assert all(abs(v / 30000 - 1 / 3) < 0.02 for v in c.values())


def test_place_deletions_valid_and_uniform_start():
    rng = np.random.default_rng(3)
    L, starts = 12, collections.Counter()
    for _ in range(20000):
        ops = place_deletions(L, [2], rng)
        starts[ops[0][1]] += 1
        validate(L, ops)
    assert set(starts) == set(range(0, L - 2 + 1))
    assert all(abs(v / 20000 - 1 / 11) < 0.012 for v in starts.values())


def test_place_deletions_multi_separated():
    rng = np.random.default_rng(4)
    for _ in range(2000):
        ops = place_deletions(40, [3, 2, 1, 4], rng)
        validate(40, ops)
        for a, b in zip(ops, ops[1:]):
            assert b[1] - a[2] >= 2                      # >= 1 retained residue between segments
        assert [o[2] - o[1] + 1 for o in ops] == [3, 2, 1, 4]


def test_plan_bounds_and_validity_across_lengths():
    for L in (50, 51, 86, 137, 209, 350, 384):
        for d in range(300):
            r = draw_plan(L, "1abc_A", d)
            for name in ("ins", "del"):
                x = r[name]
                assert FRAC_LO <= x["frac"] <= FRAC_HI
                assert x["T"] == int(round(x["frac"] * L))
                assert sum(x["segments"]) == x["T"]
                assert 1 <= x["k_drawn"] <= 5 and len(x["segments"]) <= x["k_drawn"]
            ops = [tuple(o) for o in r["ops"]]
            validate(L, ops)
            n_ins = sum(o[2] for o in ops if o[0] == "ins")
            n_del = sum(o[2] - o[1] + 1 for o in ops if o[0] == "del")
            assert n_ins == r["ins"]["T"] and n_del == r["del"]["T"]
            coords = np.zeros((L, 4, 3)) + np.arange(L)[:, None, None]
            new, orig, n2n = edit(coords, ops)
            assert len(new) == L + n_ins - n_del
            assert (orig == -1).sum() == n_ins and (n2n == -1).sum() == n_del
            assert L + n_ins <= 1.3 * L + 1                   # span stays under cc89's 484 for L <= 370


def test_deterministic_and_draw_dependent():
    a, b = draw_plan(100, "x_A", 5), draw_plan(100, "x_A", 5)
    assert a == b
    assert draw_plan(100, "x_A", 6) != a and draw_plan(100, "y_A", 5) != a
    assert draw_plan(100, "x_A", 5, global_seed=1) != a


def test_ins_del_fractions_independent():
    fr = np.array([(draw_plan(200, "c_A", d)["ins"]["frac"], draw_plan(200, "c_A", d)["del"]["frac"])
                   for d in range(2000)])
    assert abs(np.corrcoef(fr.T)[0, 1]) < 0.06
    assert abs(fr.mean(0) - 0.2).max() < 0.01


def test_insertion_site_coverage_includes_terminals():
    sites = collections.Counter()
    for d in range(3000):
        for o in draw_plan(60, "z_A", d)["ops"]:
            if o[0] == "ins":
                sites[o[1]] += 1
    assert -1 in sites and 59 in sites and len(sites) > 55


def test_custom_fraction_range_is_respected_and_default_unchanged():
    base = draw_plan(200, "c_A", 3)
    assert draw_plan(200, "c_A", 3, 0, FRAC_LO, FRAC_HI) == base
    for d in range(30):
        r = draw_plan(300, "c_A", d, 0, 0.01, 0.05)
        assert 0.01 <= r["ins"]["frac"] <= 0.05 and 0.01 <= r["del"]["frac"] <= 0.05
        assert r["ins"]["T"] == sum(r["ins"]["segments"]) and r["del"]["T"] == sum(r["del"]["segments"])
