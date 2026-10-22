"""T8b regeneration round table (t2_topup.T8B_ROUNDS): same rounds / sizes / rewinds as ROUNDS, draw ids fresh (>= 4048, past every id of the T2 product), no two ranges overlap
(base 4048..4111 included), and `plan --table t8b` writes those ids and seed offsets into round.json. Run: python test_t8b_table.py"""
import json
import os
import tempfile

import pandas as pd

import t2_topup as T

BASE = (4048, 64, 21)   # t2_prod.sbatch DRAW_START / n-draws / SEED_OFFSET of the T8b base run


def main():
    assert sorted(T.T8B_ROUNDS) == sorted(T.ROUNDS)
    for rd, t in T.T8B_ROUNDS.items():
        o = T.ROUNDS[rd]
        assert (t["proto"], t["n_draws"], t["rewind"]) == (o["proto"], o["n_draws"], o["rewind"]), rd
    old_max = max(o["draw_start"] + o["n_draws"] for o in T.ROUNDS.values())
    assert old_max == BASE[0], old_max
    spans = sorted([(BASE[0], BASE[0] + BASE[1], "base")] + [(t["draw_start"], t["draw_start"] + t["n_draws"], rd) for rd, t in T.T8B_ROUNDS.items()])
    for (a0, a1, ra), (b0, b1, rb) in zip(spans, spans[1:]):
        assert a1 == b0, (ra, a1, rb, b0)   # contiguous and disjoint
    offs = [BASE[2]] + [t["seed_offset"] for t in T.T8B_ROUNDS.values()]
    assert sorted(offs) == list(range(21, 21 + len(offs))), offs
    assert not set(offs) & ({0} | {o["seed_offset"] for o in T.ROUNDS.values()})   # never an offset of the T2 product (0 = its base)
    print(f"PASS table: {len(spans)} disjoint contiguous id ranges {spans[0][0]}..{spans[-1][1] - 1}, seed offsets {min(offs)}..{max(offs)}")

    tmp = tempfile.mkdtemp()
    pack, top = os.path.join(tmp, "pack"), os.path.join(tmp, "topup")
    os.makedirs(pack)
    pd.DataFrame([dict(chain="2bbb_A", L_native=150, pass_all=i < 50, source="main") for i in range(64)]).to_csv(os.path.join(pack, "metrics_all.csv.gz"), index=False)
    for table, rounds in (("t8b", T.T8B_ROUNDS), ("t2", T.ROUNDS)):
        T.plan(pack, os.path.join(top, table), "P1", 2, table)
        cfg = json.load(open(os.path.join(top, table, "P1", "round.json")))
        assert cfg["draw_start"] == rounds["P1"]["draw_start"] and cfg["table"] == table, cfg
        assert cfg["stage_b_args"] == f"--num-steps 250 --rewind 125 --seed-offset {rounds['P1']['seed_offset']} --store passing", cfg["stage_b_args"]
    print("PASS plan --table t8b / t2: round.json carries the table's draw_start and seed offset")
    print("ALL PASS")


if __name__ == "__main__":
    main()
