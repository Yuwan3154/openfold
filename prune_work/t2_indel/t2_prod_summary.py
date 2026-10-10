"""Per-chain survivor table of a production run (env: any with pandas). Reads <templates>/<id[1:3]>/<id>.metrics.csv of every chain and the chain list, writes survivors.tsv
(rows = chain of the list; columns: n_run = variants that went through diffusion, n_pass = variants passing ALL gates (TM window, bond envelope, loop, break), and the number failing each
gate) and prints the coverage table (chains with >= k survivors) plus the chains with no metrics file (never run / skipped at stage A or B: see the skipped.jsonl files).
Run: python t2_prod_summary.py --list template_chains.txt --templates T --out survivors.tsv
"""
import argparse
import os

import pandas as pd

GATES = ["pass_tm", "pass_bond", "pass_loop", "pass_break"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [ln.strip() for ln in open(a.list) if ln.strip()]
    rows = []
    for cid in ids:
        f = os.path.join(a.templates, cid[1:3], cid + ".metrics.csv")
        if not os.path.isfile(f):
            rows.append(dict(chain=cid, has_metrics=False))
            continue
        m = pd.read_csv(f)
        rows.append(dict(chain=cid, has_metrics=True, n_run=len(m), n_pass=int(m.pass_all.sum()), **{"fail_" + g[5:]: int((~m[g]).sum()) for g in GATES}))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    have = df[df.has_metrics]
    print(f"rows = chain of the list (n = {len(df)}); {len(have)} have metrics, {len(df) - len(have)} have none")
    for k in (1, 8, 16, 32, 64):
        print(f"chains with >= {k:2d} surviving templates: {int((have.n_pass >= k).sum())} ({100 * (have.n_pass >= k).sum() / len(df):.1f} % of the list)")
    print(f"mean survivors per chain (over chains with metrics): {have.n_pass.mean():.1f} of {have.n_run.mean():.1f} run")


if __name__ == "__main__":
    main()
