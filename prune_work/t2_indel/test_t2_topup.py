"""Unit test of the top-up status / round membership on synthetic data (numpy + pandas). Run: python test_t2_topup.py"""
import os
import tempfile

import numpy as np
import pandas as pd

import t2_topup as T


def write_round(top, rd, chain, runs, passed):
    d = os.path.join(top, rd, "templates", chain[1:3])
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(dict(chain=chain, pass_all=[True] * passed + [False] * (runs - passed))).to_csv(os.path.join(d, chain + ".metrics.csv"), index=False)


def main():
    tmp = tempfile.mkdtemp()
    pack, top = os.path.join(tmp, "pack"), os.path.join(tmp, "topup")
    os.makedirs(pack)
    rows = []
    def add(chain, L, passed, src="main", runs=64):
        rows.extend(dict(chain=chain, L_native=L, pass_all=i < passed, source=src) for i in range(runs))
    add("1aaa_A", 100, 64)            # already 64
    add("2bbb_A", 150, 50)            # short, production reachable
    add("3ccc_A", 200, 0)             # zero survivors -> rewind-100 track
    add("4ddd_A", 250, 1)             # expected extra > cap (p ~ 3 %) -> rewind-100 track
    add("5eee_A", 80, 0); add("5eee_A", 80, 5, src="rescue", runs=256)   # rescued chain (5/256 survivors): rewind-100 track
    pd.DataFrame(rows).to_csv(os.path.join(pack, "metrics_all.csv.gz"), index=False)
    d = T.status(pack, top)
    assert d.have.to_dict() == {"1aaa_A": 64, "2bbb_A": 50, "3ccc_A": 0, "4ddd_A": 1, "5eee_A": 5}, d.have.to_dict()
    assert list(T.members(d, "P1").index) == ["2bbb_A"], list(T.members(d, "P1").index)
    assert sorted(T.members(d, "R1").index) == ["3ccc_A", "4ddd_A", "5eee_A"], sorted(T.members(d, "R1").index)
    # P1 ran for 2bbb_A: 64 draws, 3 passed -> still short (53): goes to P2; then P2/P3/P4 with no gain: after P4 the production cap is exhausted -> R1
    write_round(top, "P1", "2bbb_A", 63, 3)   # stage B dropped one draw (span cutoff / changed sequence): the chain must still move on to P2
    d = T.status(pack, top)
    assert d.loc["2bbb_A", "have"] == 53 and list(T.members(d, "P1").index) == [] and list(T.members(d, "P2").index) == ["2bbb_A"]
    write_round(top, "P2", "2bbb_A", 120, 3); write_round(top, "P3", "2bbb_A", 256, 3)
    d = T.status(pack, top)
    assert list(T.members(d, "P4").index) == ["2bbb_A"] and "2bbb_A" not in set(T.members(d, "R1").index)
    write_round(top, "P4", "2bbb_A", 570, 1)
    d = T.status(pack, top)
    assert d.loc["2bbb_A", "extra_prod"] == 63 + 120 + 256 + 570 and d.loc["2bbb_A", "have"] == 60
    assert "2bbb_A" in set(T.members(d, "R1").index) and list(T.members(d, "P2").index) == []
    # R1 for 3ccc_A gains 2 survivors -> goes to R2; 4ddd_A gains nothing -> accepted as failed (not in R2)
    write_round(top, "R1", "3ccc_A", 256, 2); write_round(top, "R1", "4ddd_A", 256, 0)
    d = T.status(pack, top)
    assert sorted(T.members(d, "R2").index) == ["3ccc_A"] and "4ddd_A" not in set(T.members(d, "R1").index)
    sel = T.plan(pack, top, "R2", 2)
    assert open(os.path.join(top, "R2", "ids_shard0.txt")).read().split() == ["3ccc_A"]
    assert T.ROUNDS["P1"]["n_draws"] + T.ROUNDS["P2"]["n_draws"] + T.ROUNDS["P3"]["n_draws"] + T.ROUNDS["P4"]["n_draws"] == T.CAP == T.ROUNDS["R1"]["n_draws"] + T.ROUNDS["R2"]["n_draws"] + T.ROUNDS["R3"]["n_draws"]
    print("test_t2_topup: all checks passed")


if __name__ == "__main__":
    main()
