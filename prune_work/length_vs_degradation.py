"""Is Run C's epoch-0 PDA regression concentrated on SHORT chains?

Hypothesis: gaussian_pair_init injects noise per PAIR element at fixed sigma, so a short chain has
far fewer elements to average it out and the same tau is relatively more destructive. The selector
picked tau=16/32 on ~35% of steps, and tau=32 is past the measured tau~24.8 crossover.

Baseline = Run B ep10 per-entry (its aggregate 0.7699 sits within 0.007 of the validate_only check
on the same weights, so it is a sound proxy). Test = Run C ep0 per-entry.
"""
import csv
import json
import statistics as st
from collections import defaultdict

E = "/home/jupyter-chenxi/prune_work/eval_out"
full = json.load(open(f"{E}/pda_cluster_representatives.json"))
mdl = json.load(open(f"{E}/pda_cluster_representatives_modelable.json"))


def k(e):
    return (e["pdb"].lower(), e["chain_id"])


full_key = [k(e) for e in full]          # Run B batch_idx 0..424
mdl_key = [k(e) for e in mdl]            # Run C pda batch_idx 0..305
length = {k(e): len(e["seq"]) for e in mdl}


def load(path, epoch, keys, lo=0, hi=10**9):
    """first row per (epoch, batch_idx); DDP pads the sampler so a few repeat"""
    seen, out = set(), {}
    for r in csv.DictReader(open(path)):
        if int(r["epoch"]) != epoch:
            continue
        bi = int(r["batch_idx"])
        if bi in seen or not (lo <= bi < hi):
            continue
        seen.add(bi)
        if bi - lo < len(keys):
            out[keys[bi - lo]] = float(r["lddt_ca"])
    return out


base = load("/home/jupyter-chenxi/runs/runB_full_stack_pda_eval/lightning_logs/version_1/"
            "per_entry_val_history.csv", 10, full_key)
new = load("/home/jupyter-chenxi/runs/runC_replica_exchange/lightning_logs/version_0/"
           "per_entry_val_history.csv", 0, mdl_key, 0, 306)
common = [key for key in mdl_key if key in base and key in new]
print(f"Run B ep10 entries {len(base)}  |  Run C ep0 pda entries {len(new)}  |  joined {len(common)}")
print(f"baseline mean {st.fmean(base[c] for c in common):.4f}  "
      f"runC ep0 mean {st.fmean(new[c] for c in common):.4f}  "
      f"delta {st.fmean(new[c] - base[c] for c in common):+.4f}")

BINS = [(0, 30), (31, 50), (51, 80), (81, 130), (131, 220), (221, 10**9)]
print(f"\n{'length bin':>14} {'n':>5} {'baseline':>10} {'runC ep0':>10} {'delta':>9} {'% worse':>9}")
for lo, hi in BINS:
    g = [c for c in common if lo <= length[c] <= hi]
    if not g:
        continue
    b = st.fmean(base[c] for c in g)
    n_ = st.fmean(new[c] for c in g)
    worse = 100.0 * sum(1 for c in g if new[c] < base[c]) / len(g)
    label = f"{lo}-{hi if hi < 10**9 else '+'}"
    print(f"{label:>14} {len(g):5d} {b:10.4f} {n_:10.4f} {n_ - b:+9.4f} {worse:8.1f}%")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


L = [length[c] for c in common]
D = [new[c] - base[c] for c in common]
rho = spearman(L, D)
print(f"\nSpearman(length, delta_lddt) over {len(common)} entries = {rho:+.4f}")
print("  positive => SHORT chains degraded MORE (the hypothesis)")
print("  ~0       => degradation is length-independent (hypothesis dead)")

short = [d for c, d in zip(common, D) if length[c] <= 50]
long_ = [d for c, d in zip(common, D) if length[c] > 50]
print(f"\n  <=50 res : n={len(short):3d}  mean delta {st.fmean(short):+.4f}")
print(f"  > 50 res : n={len(long_):3d}  mean delta {st.fmean(long_):+.4f}")
print(f"  gap      : {st.fmean(short) - st.fmean(long_):+.4f}")
if len(short) > 1 and len(long_) > 1:
    se = (st.variance(short) / len(short) + st.variance(long_) / len(long_)) ** 0.5
    print(f"  Welch t  : {(st.fmean(short) - st.fmean(long_)) / se:+.2f}" if se else "")
