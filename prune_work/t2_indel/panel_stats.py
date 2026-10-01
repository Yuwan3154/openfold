"""Per-chain native statistics for the 600 validation chains (val_300_easy + val_300_hard), used to
stratify the indel pilot panel by length bin x native SSE content x easy/hard.

Chain ids are AUTH ids (the val lists' convention). Residues are standard amino acids (MSE -> MET) with
all of N/CA/C/O; every chain gets a status row (nothing is dropped silently).
Env: cue_openfold_gated (Biopython + mdtraj); t2_deepdive on sys.path via this file's location.
Run: python panel_stats.py --val val_300_easy.json val_300_hard.json --mmcif-dir <dir> --out panel_stats.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from Bio.PDB import MMCIFParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "t2_deepdive"))
from pool_sse import RESTYPES3, backbone_top, ss_of  # noqa: E402

BB = ("N", "CA", "C", "O")
THREE = {r: i for i, r in enumerate(RESTYPES3[:20])}
THREE["MSE"] = THREE["MET"]


def chain_backbone(model, chain_id):
    if chain_id not in [c.id for c in model]:
        return None
    resnames, resseq, xyz = [], [], []
    for res in model[chain_id]:
        if res.id[0] != " " and res.resname != "MSE":
            continue
        if res.resname not in THREE or not all(a in res for a in BB):
            continue
        resnames.append(THREE[res.resname])
        resseq.append(res.id[1])
        xyz.append([res[a].coord for a in BB])
    return np.array(resnames), np.array(resseq), np.array(xyz, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--mmcif-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    recs = [r for f in a.val for r in json.load(open(f))]
    parser = MMCIFParser(QUIET=True)
    rows = []
    for k, r in enumerate(recs):
        row = {"pdb": r["pdb"], "chain": r["chain_id"], "source": r["val_source"], "length_json": r["length"],
               "best_tm_to_train": r["best_tm_to_train"], "status": "ok"}
        path = f"{a.mmcif_dir}/{r['pdb']}.cif"
        if not os.path.exists(path):
            row["status"] = "cif_missing"
            rows.append(row)
            continue
        model = parser.get_structure(r["pdb"], path)[0]
        bb = chain_backbone(model, r["chain_id"])
        if bb is None or len(bb[0]) < 2:
            row["status"] = "chain_absent_or_empty"
            rows.append(row)
            continue
        aat, resseq, xyz = bb
        top = backbone_top(aat, np.zeros(len(aat), bool))
        ss = ss_of(top, xyz.reshape(-1, 3))
        ca = xyz[:, 1]
        d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
        row.update(n_parsed=len(aat), span=int(resseq.max() - resseq.min() + 1),
                   helix_frac=float((ss == "H").mean()), strand_frac=float((ss == "E").mean()),
                   n_ca_breaks=int((d > 4.5).sum()))
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(recs)}", flush=True)
    keys = ["pdb", "chain", "source", "length_json", "best_tm_to_train", "status", "n_parsed", "span",
            "helix_frac", "strand_frac", "n_ca_breaks"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print("status:", st, "rows:", len(rows))


if __name__ == "__main__":
    main()
