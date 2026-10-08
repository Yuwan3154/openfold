"""One comparison table over the refold-score csvs of the pilot arms (refold_score.py output): per arm, designs (seq_kind > 0) and the generation
sequence (seq_kind 0) separately: number of templates, mean/median paired TM, counts above 0.5 and 0.7 (TM > 0.5 is the user's bar), mean pLDDT,
mean MPNN score. Optional --arm-filter restricts a csv to one arm label. Env: any with pandas.
Run: python refold_compare.py --csv LABEL=path[:ARM] ...  (e.g. --csv control=refold_cpool_scores.csv indel_comp=refold_scores_esm.csv:comp)
"""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True, metavar="LABEL=PATH[:ARM]")
    a = p.parse_args()
    rows = []
    for spec in a.csv:
        label, rest = spec.split("=", 1)
        path, _, arm = rest.partition(":")
        d = pd.read_csv(path)
        if arm:
            d = d[d.arm == arm]
        assert len(d) > 0, f"{label}: no rows ({path} arm={arm!r})"
        assert d.arm.nunique() == 1, f"{label}: {path} holds arms {sorted(d.arm.unique())}; pick one with :ARM"
        assert "temp" not in d or d.temp.isna().all(), f"{label}: {path} mixes lower-temperature rows; split by temp first"  # older score csvs have no temp column
        assert d.tm_paired.notna().all(), f"{label}: {int(d.tm_paired.isna().sum())} rows without a paired TM"
        for kind, sub in (("designs", d[d.seq_kind > 0]), ("generation_seq", d[d.seq_kind == 0])):
            if len(sub) == 0:
                continue
            rows.append(dict(arm=label, kind=kind, templates=sub.groupby(["chain", "i"]).ngroups, n=len(sub), mean_tm=sub.tm_paired.mean(),
                             median_tm=sub.tm_paired.median(), gt05=int((sub.tm_paired > 0.5).sum()), gt07=int((sub.tm_paired > 0.7).sum()),
                             frac_gt05=(sub.tm_paired > 0.5).mean(), frac_gt07=(sub.tm_paired > 0.7).mean(),
                             plddt=sub.plddt.mean(), mpnn=sub.mpnn_score.mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
