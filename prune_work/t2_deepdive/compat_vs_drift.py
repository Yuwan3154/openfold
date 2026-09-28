"""Does the query sequence's fit to a synthetic template track HOW the template departed from its native?

Joins the per-template drift decomposition (sse_drift.py) with sequence-structure compatibility:
ProteinMPNN per-residue NLL of the query (native) sequence (designability.csv, full-backbone
score_only) and, when present, ProteinEBM energy / pTM under the query sequence (ebm_score.py).
Compatibility is reported relative to the chain's own native (delta), since both scales are
chain-dependent. Correlations are Spearman WITHIN each (model, chain) and then summarized as the
median over the 5 chains; the partial correlation removes TM (rank-residualized) to ask what each
drift component explains beyond global similarity.

Run: python compat_vs_drift.py --dir <deepdive_dir> [--ebm-csv ebm_all.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

DRIFT = {"tm": "TM to native", "q3": "Q3 vs native", "elem_local_rmsd": "element local RMSD",
         "elem_global_rmsd": "element global RMSD", "q_contacts": "native contacts kept",
         "epairs_new": "new element pairs", "helix_excess": "helix fraction / native"}


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--ebm-csv")
    a = p.parse_args()
    d = pd.read_csv(os.path.join(a.dir, "out", "bakeoff_sse_drift.csv"))
    d["helix_excess"] = d.fH / d.fH_native

    des = pd.read_csv(os.path.join(a.dir, "designability.csv"))
    gen = des[des.kind == "generated"][["model", "chain", "rewind_steps", "sample", "query_seq_nll"]]
    nat = des[des.kind != "generated"].groupby("chain").query_seq_nll.first()
    gen = gen.rename(columns={"rewind_steps": "rewind"})
    d = d.merge(gen, on=["model", "chain", "rewind", "sample"], how="left", validate="one_to_one")
    assert d.query_seq_nll.notna().all(), "designability rows missing for some templates"
    d["d_mpnn_nll"] = d.query_seq_nll - d.chain.map(nat)
    comp = {"d_mpnn_nll": "MPNN NLL(query) - native"}

    if a.ebm_csv:
        e = pd.read_csv(a.ebm_csv)
        e["base"] = e.pdb_file.str.split("/").str[-1]
        is_nat = ~e.base.str.startswith("sample_")
        en = e[is_nat].assign(chain=e.base.str.replace(".pdb", "", regex=False)).set_index("chain")
        tm = pd.read_csv(os.path.join(a.dir, "pp1c_tm.csv"))
        tm["chain_id"] = tm.pdb_id + "_" + tm.chain
        es = e[~is_nat].merge(tm[["pdb_file", "model", "chain_id", "rewind_steps", "sample"]], on="pdb_file",
                              validate="one_to_one")
        es = es.rename(columns={"chain_id": "chain", "rewind_steps": "rewind"})
        d = d.merge(es[["model", "chain", "rewind", "sample", "energy", "ptm"]],
                    on=["model", "chain", "rewind", "sample"], how="left", validate="one_to_one")
        assert d.energy.notna().all(), "EBM rows missing for some templates"
        d["d_ebm_energy_per_res"] = (d.energy - d.chain.map(en.energy)) / d.L
        d["d_ebm_ptm"] = d.ptm - d.chain.map(en.ptm)
        comp.update({"d_ebm_energy_per_res": "EBM energy/res - native", "d_ebm_ptm": "EBM pTM - native"})

    rows = []
    for (m, c), g in d.groupby(["model", "chain"]):
        for ck in comp:
            for dk in DRIFT:
                rho = spearmanr(g[ck], g[dk]).statistic
                prt = partial_spearman(g[ck].values, g[dk].values, g.tm.values) if dk != "tm" else np.nan
                rows.append(dict(model=m, chain=c, compat=ck, drift=dk, rho=rho, partial_rho_given_tm=prt))
    r = pd.DataFrame(rows)
    summ = r.groupby(["compat", "drift", "model"])[["rho", "partial_rho_given_tm"]].median().round(2)
    summ["rho_min_max"] = r.groupby(["compat", "drift", "model"]).rho.agg(
        lambda s: f"{s.min():.2f}..{s.max():.2f}")
    pd.set_option("display.width", 200)
    print(summ.unstack("model").to_string())
    d.to_csv(os.path.join(a.dir, "out", "compat_drift_joined.csv"), index=False)
    r.to_csv(os.path.join(a.dir, "out", "compat_drift_rho_per_chain.csv"), index=False)
    bins = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    d["tmbin"] = pd.cut(d.tm, bins)
    print(d.groupby(["model", "tmbin"], observed=True)[list(comp)].mean().round(3).unstack("model").to_string())


if __name__ == "__main__":
    main()
