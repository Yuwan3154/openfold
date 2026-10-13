"""Top-up driver (T2_TOPUP_DIVERSITY_PLAN.md; user 10-13): status of every chain and the chain lists / settings of the next generation round.
Status (rows = chain): the stored survivors of the first production (main run, or the rescue run for rescued chains; from <pack>/metrics_all.csv.gz) + the passing templates of every finished top-up round
(<topup>/<round>/templates/<id[1:3]>/<id>.metrics.csv). Rounds (cap 1,024 extra draws per chain and protocol):
  P1..P4  production protocol (half-step: --num-steps 250 --rewind 125): +64, +128, +256, +576 draws (draw ids 64.., cumulative 1,024), noise seed offsets 11..14
  R1..R3  rewind-100 attempt (--num-steps 250 --rewind 100): +256, +256, +512 draws (draw ids 2000.., cumulative 1,024), seed offsets 3001..3003
Round membership is keyed on COMPLETED rounds (a metrics.csv of the round exists), never on run counts (stage B can write fewer rows than n_draws).
Who goes into which round:
  P1: chains with 0 < base survivors < 64 whose expected extra draws (at the chain's own smoothed pass rate) are <= 1,024; P2..P4: chains of the previous P round still < 64 candidates.
  R1: chains still < 64 that need rescue = zero base survivors, or expected extra draws > 1,024, or production cap exhausted (P4 done); R2, R3: chains of R1 that gained >= 1 survivor and are still < 64 (a chain that gains nothing in R1 is accepted as failed).
Usage: python t2_topup.py status --pack P --topup T ;  python t2_topup.py plan --pack P --topup T --round P1 --n-shards 2
`plan` writes <topup>/<round>/ids_shard<k>.txt (length-sorted, chain i -> shard i % n) and <topup>/<round>/round.json (n_draws, draw_start, stage-B arguments).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROUNDS = {
    "P1": dict(proto="prod", n_draws=64, draw_start=64, rewind=125, seed_offset=11),
    "P2": dict(proto="prod", n_draws=128, draw_start=128, rewind=125, seed_offset=12),
    "P3": dict(proto="prod", n_draws=256, draw_start=256, rewind=125, seed_offset=13),
    "P4": dict(proto="prod", n_draws=576, draw_start=512, rewind=125, seed_offset=14),
    "R1": dict(proto="r100", n_draws=256, draw_start=2000, rewind=100, seed_offset=3001),
    "R2": dict(proto="r100", n_draws=256, draw_start=2256, rewind=100, seed_offset=3002),
    "R3": dict(proto="r100", n_draws=512, draw_start=2512, rewind=100, seed_offset=3003),
}
CAP = 1024


def base_status(pack):
    m = pd.read_csv(os.path.join(pack, "metrics_all.csv.gz"), usecols=["chain", "L_native", "pass_all", "source"])
    main = m[m.source == "main"].groupby("chain").agg(L=("L_native", "first"), n_run=("pass_all", "size"), n_pass=("pass_all", "sum"))
    resc = m[m.source == "rescue"].groupby("chain").agg(n_run_r=("pass_all", "size"), n_pass_r=("pass_all", "sum"))
    d = main.join(resc, how="left")
    use_r = d.n_pass_r.notna() & (d.n_pass_r > 0)
    d["have_base"] = np.where(use_r, d.n_pass_r, d.n_pass).astype(int)
    d["runs_base"] = np.where(use_r, d.n_run_r, d.n_run).astype(int)
    d["rescued"] = d.n_pass_r.notna()   # the first rescue (rewind 100) was already tried for this chain: its further draws go to the rewind-100 track
    return d[["L", "have_base", "runs_base", "rescued"]]


def round_counts(topup):
    """rows = chain x round: run variants and passing templates of every finished top-up round (a round counts only chains whose metrics.csv exists)"""
    rows = []
    for rd in sorted(ROUNDS):
        for f in glob.glob(os.path.join(topup, rd, "templates", "*", "*.metrics.csv")):
            m = pd.read_csv(f, usecols=["chain", "pass_all"])
            rows.append((m.chain.iloc[0], rd, len(m), int(m.pass_all.sum())))
    return pd.DataFrame(rows, columns=["chain", "round", "runs", "passed"])


def done_flags(rc, index):
    """done_<round> = a metrics.csv of that round exists for the chain (stage B may write fewer rows than n_draws: draws above the span cutoff or with a changed sequence are dropped, so run counts are NOT used for membership)"""
    out = pd.DataFrame(index=index)
    for rd in ROUNDS:
        out["done_" + rd] = out.index.isin(rc[rc["round"] == rd].chain) if len(rc) else False
    return out


def status(pack, topup):
    d = base_status(pack)
    rc = round_counts(topup)
    for proto, rounds in (("prod", "P"), ("r100", "R")):
        s = rc[rc["round"].str.startswith(rounds)].groupby("chain").agg(**{f"runs_{proto}": ("runs", "sum"), f"pass_{proto}": ("passed", "sum")}) if len(rc) else pd.DataFrame()
        d = d.join(s, how="left") if len(s) else d.assign(**{f"runs_{proto}": 0, f"pass_{proto}": 0})
    d = d.fillna({"runs_prod": 0, "pass_prod": 0, "runs_r100": 0, "pass_r100": 0}).astype({"runs_prod": int, "pass_prod": int, "runs_r100": int, "pass_r100": int})
    d["have"] = d.have_base + d.pass_prod + d.pass_r100
    d["p_base"] = (d.have_base + 1) / (d.runs_base + 2)
    d["exp_extra"] = np.ceil(np.maximum(64 - d.have_base, 0) / d.p_base).astype(int)
    d["extra_prod"] = d.runs_prod
    d["extra_r100"] = d.runs_r100
    return d.join(done_flags(rc, d.index))


def members(d, rd):
    short = d.have < 64
    prod_ok = (d.have_base > 0) & (d.exp_extra <= CAP) & ~d.rescued   # chains the production protocol can finish within the cap
    if rd == "P1":
        return d[short & prod_ok & ~d.done_P1 & ~d.done_R1]
    if rd in ("P2", "P3", "P4"):
        prev = {"P2": "done_P1", "P3": "done_P2", "P4": "done_P3"}[rd]
        return d[short & prod_ok & d[prev] & ~d["done_" + rd]]
    if rd == "R1":
        need = ~prod_ok | d.done_P4
        return d[short & need & ~d.done_R1]
    prev = {"R2": "done_R1", "R3": "done_R2"}[rd]
    return d[short & d[prev] & ~d["done_" + rd] & (d.pass_r100 >= 1)]


def plan(pack, topup, rd, n_shards):
    d = status(pack, topup)
    sel = members(d, rd).sort_values(["L"])
    out = os.path.join(topup, rd)
    os.makedirs(out, exist_ok=True)
    ids = list(sel.index)
    for k in range(n_shards):
        with open(os.path.join(out, f"ids_shard{k}.txt"), "w") as fh:
            fh.write("\n".join(ids[k::n_shards]) + "\n")
    cfg = dict(ROUNDS[rd], round=rd, n_chains=len(ids), stage_b_args=f"--num-steps 250 --rewind {ROUNDS[rd]['rewind']} --seed-offset {ROUNDS[rd]['seed_offset']} --store passing")
    with open(os.path.join(out, "round.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(f"round {rd}: {len(ids)} chains, {cfg['n_draws']} draws each (ids {cfg['draw_start']}..{cfg['draw_start'] + cfg['n_draws'] - 1}), expected deficit {int((64 - sel.have).sum())} templates -> {out}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "plan"])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--topup", required=True)
    ap.add_argument("--round", choices=sorted(ROUNDS))
    ap.add_argument("--n-shards", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.pack, a.topup, a.round, a.n_shards)
        return
    d = status(a.pack, a.topup)
    print(f"rows = chain (n = {len(d)}): candidates passing (base + top-up): zero {int((d.have == 0).sum())}, 1-63 {int(((d.have > 0) & (d.have < 64)).sum())}, >= 64 {int((d.have >= 64).sum())}; extra draws used: production {int(d.extra_prod.sum())}, rewind-100 {int(d.extra_r100.sum())}")


if __name__ == "__main__":
    main()
