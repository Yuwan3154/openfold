"""Exclusion table of a production run: every chain of the list WITHOUT a <id>.metrics.csv and WHY. Two modes (env pp1c / pandas not needed):
  --mode ids  : write <work>/ids_no_natives.txt = list chains without a natives npz (input of t2_extract_natives.py, whose skipped.tsv in a NEW dir gives the extraction reasons; the extractor overwrites skipped.tsv per run, so the original reasons were lost)
  --mode build: join the list with metrics files, stage A / stage B skipped.jsonl (deduped by chain) and the extractor's skipped.tsv -> <out>/exclusion.tsv (chain, stage, reason, detail) + a count per reason.
Every chain without metrics must get a reason: chains with none are printed as UNEXPLAINED.
Run: python t2_exclusion.py --mode ids|build --list L --templates T --work W [--extract-skips S.tsv] --out O
"""
import argparse
import glob
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ids", "build"])
    ap.add_argument("--list", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--extract-skips", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [ln.strip() for ln in open(a.list) if ln.strip()]
    have_nat = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.work, "natives", "*", "*.npz"))}
    if a.mode == "ids":
        miss = [c for c in ids if c not in have_nat]
        open(a.out, "w").write("\n".join(miss) + "\n")
        print(f"list {len(ids)} chains, natives npz present {len(ids) - len(miss)}, without {len(miss)} -> {a.out}")
        return
    have = {os.path.basename(p)[:-len(".metrics.csv")] for p in glob.glob(os.path.join(a.templates, "*", "*.metrics.csv")) if not p.endswith(".rescue.metrics.csv")}
    why = {}
    if a.extract_skips:
        for ln in open(a.extract_skips):
            p = ln.rstrip("\n").split("\t")
            why[p[0]] = ("extraction", p[1], p[2] if len(p) > 2 else "")
    for stage, f in (("stage A", os.path.join(a.work, "stageA", "skipped.jsonl")), ("stage B", os.path.join(a.templates, "skipped.jsonl"))):
        if os.path.isfile(f):
            for ln in open(f):
                r = json.loads(ln)
                if "draw" not in r:   # per-draw records of a chain that still has other draws are not exclusions
                    why.setdefault(r["chain"], (stage, r["why"], ""))
    rows = []
    for c in ids:
        if c in have:
            continue
        s, r, d = why.get(c, ("UNEXPLAINED", "no metrics, no skip record", ""))
        rows.append((c, s, r, d))
    with open(os.path.join(a.out, "exclusion.tsv"), "w") as fh:
        fh.write("chain\tstage\treason\tdetail\n")
        fh.writelines("\t".join(r) + "\n" for r in rows)
    print(f"list {len(ids)} chains, with metrics {len(have & set(ids))}, excluded {len(rows)}")
    for (s, r), n in Counter((r[1], r[2]) for r in rows).most_common():
        print(f"  {n:5d}  {s}: {r}")
    unexplained = [r[0] for r in rows if r[1] == "UNEXPLAINED"]
    print(f"UNEXPLAINED chains: {len(unexplained)} {unexplained[:10]}")


if __name__ == "__main__":
    main()
