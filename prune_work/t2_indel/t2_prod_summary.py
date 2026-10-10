"""Per-chain survivor table of a production run (env: any with pandas). Reads <templates>/<id[1:3]>/<id>.metrics.csv of every chain and the chain list, writes survivors.tsv
(rows = chain of the list; columns: n_run = variants that went through diffusion, n_pass = variants passing ALL gates (TM window, bond envelope, loop, break), and the number failing each
gate) and prints the coverage table (chains with >= k survivors) plus the chains with no metrics file (never run / skipped at stage A or B: see the skipped.jsonl files).
Run: python t2_prod_summary.py --list template_chains.txt --templates T --out survivors.tsv
"""
import argparse
import os

import numpy as np
import pandas as pd

GATES = ["pass_tm", "pass_bond", "pass_loop", "pass_break"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [ln.strip() for ln in open(a.list) if ln.strip()]
    rows, tms, lens, passes = [], [], [], []
    for cid in ids:
        f = os.path.join(a.templates, cid[1:3], cid + ".metrics.csv")
        if not os.path.isfile(f):
            rows.append(dict(chain=cid, has_metrics=False))
            continue
        m = pd.read_csv(f)
        tms.append(m.tm_native.values)
        lens.append(np.full(len(m), int(m.L_native.iloc[0])))
        passes.append(m.pass_all.values)
        rows.append(dict(chain=cid, has_metrics=True, n_run=len(m), n_pass=int(m.pass_all.sum()), **{"fail_" + g[5:]: int((~m[g]).sum()) for g in GATES}))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    have = df[df.has_metrics]
    print(f"rows = chain of the list (n = {len(df)}); {len(have)} have metrics, {len(df) - len(have)} have none")
    for k in (1, 8, 16, 32, 64):
        print(f"chains with >= {k:2d} surviving templates: {int((have.n_pass >= k).sum())} ({100 * (have.n_pass >= k).sum() / len(df):.1f} % of the list)")
    print(f"mean survivors per chain (over chains with metrics): {have.n_pass.mean():.1f} of {have.n_run.mean():.1f} run")


    tm, nat_len, ok = np.concatenate(tms), np.concatenate(lens), np.concatenate(passes).astype(bool)
    edges = np.round(np.arange(0.4, 0.95, 0.1), 1)
    print("\nTM-score of each template vs its native (sequence-independent USalign TM, normalised by the native length); rows = template, bins of 0.1 in 0.4-0.9 (last bin includes 0.9)")
    print(f"{'bin':>9} {'all run variants':>18} {'share':>7} {'passing all gates':>18} {'share':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tm >= lo) & ((tm < hi) if hi < 0.9 else (tm <= hi))
        print(f"{lo:.1f}-{hi:.1f}   {int(sel.sum()):>18d} {sel.sum() / len(tm):>7.3f} {int((sel & ok).sum()):>18d} {(sel & ok).sum() / max(ok.sum(), 1):>7.3f}")
    print(f"outside the window: tm < 0.4: {int((tm < 0.4).sum())} ({(tm < 0.4).mean():.3f}), tm > 0.9: {int((tm > 0.9).sum())} ({(tm > 0.9).mean():.3f}); total variants {len(tm)}, passing {int(ok.sum())}")
    print("\nsame, by native length (rows = passing templates):")
    print(f"{'L bin':>9} " + " ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])) + "   n")
    for a, b in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 10000)]:
        sel = ok & (nat_len >= a) & (nat_len < b)
        if sel.sum():
            h = [((tm >= lo) & ((tm < hi) if hi < 0.9 else (tm <= hi)) & sel).sum() / sel.sum() for lo, hi in zip(edges[:-1], edges[1:])]
            print(f"{a:>4}-{b if b < 10000 else 'max':<4} " + " ".join(f"{x:>7.3f}" for x in h) + f"   {int(sel.sum())}")


if __name__ == "__main__":
    main()