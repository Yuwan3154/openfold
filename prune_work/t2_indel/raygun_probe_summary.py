"""Mutation / insertion / deletion budget of Raygun outputs vs the native, from raygun_probe*.json rows.

Per output (global alignment, see raygun_probe.py): mutations = mismatches among aligned pairs; insertions =
generated residues with no native partner; deletions = native residues with no generated partner. Reported as
per-output means, as a fraction of the native length, and the 'excess indel' = (insertions + deletions) - |net
length change|, i.e. gap residues beyond what the length change alone requires.
Run: python raygun_probe_summary.py raygun_probe.json raygun_probe_hi.json
"""
import json
import sys

import numpy as np
import pandas as pd


def main():
    rows = [r for f in sys.argv[1:] for r in json.load(open(f))["rows"]]
    d = pd.DataFrame(rows)
    d["mut"] = ((1 - d.identity) * d.n_pairs).round().astype(int)
    d["ins"], d["dele"] = d.n_extra_in_generated, d.n_native_unmatched
    d["net"] = (d.L_target - d.L).abs()
    d["excess_indel"] = d.ins + d.dele - d.net
    d["kind"] = np.where(d.target == "same", "same length", "edited length")
    for c in ("mut", "ins", "dele"):
        d[c + "_pct"] = 100 * d[c] / d.L
    g = d.groupby(["kind", "noise"])
    out = g.agg(n=("mut", "size"), mut=("mut", "mean"), mut_pct=("mut_pct", "mean"), ins=("ins", "mean"),
                ins_pct=("ins_pct", "mean"), dele=("dele", "mean"), dele_pct=("dele_pct", "mean"),
                net=("net", "mean"), excess_indel=("excess_indel", "mean"), identity_min=("identity", "min"))
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
    e = d[d.kind == "edited length"]
    print("\nedited length, mutations per output by |net length change| bin:")
    e = e.assign(netbin=pd.cut(e.net, [-1, 5, 15, 30, 100]))
    print(e.groupby(["netbin", "noise"], observed=True).agg(n=("mut", "size"), mut=("mut", "mean"), mut_pct=("mut_pct", "mean"),
                                                            ins=("ins", "mean"), dele=("dele", "mean")).round(2).to_string())
    print("\nbetween-repeat diversity (noise > 0, same chain+target): mean pairwise identity of the repeats' sequences")
    div = []
    for (_, _, nz), grp in d[d.noise > 0].groupby(["chain", "target", "noise"]):
        seqs = grp.seq.tolist()
        if len(seqs) > 1 and len({len(s) for s in seqs}) == 1:
            ids = [np.mean([a == b for a, b in zip(x, y)]) for i, x in enumerate(seqs) for y in seqs[i + 1:]]
            div.append((nz, float(np.mean(ids))))
    print(pd.DataFrame(div, columns=["noise", "pairwise_identity"]).groupby("noise").pairwise_identity.mean().round(3).to_string())


if __name__ == "__main__":
    main()
