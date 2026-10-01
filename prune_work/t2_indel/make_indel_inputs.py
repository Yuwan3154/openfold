"""Build the edited Protpardelle inputs for the panel: native.pdb + 64 edited PDBs + provenance per chain.

Per chain (out/<chain>/): native.pdb (contiguously renumbered, all atoms), d00.pdb..d63.pdb, plans.json.
Survivors keep their full native atoms; inserted residues are GLY with backbone atoms only (interpolated or
extrapolated by indel_edit). Residue numbering is contiguous 1..L' (Q1). plans.json holds, per draw, the
sampled plan plus orig_idx (-1 = inserted) and native_to_new (-1 = deleted).
Env: cue_openfold_gated (Biopython). Run: python make_indel_inputs.py --panel panel.csv --mmcif-dir <dir> --out-dir <dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

from indel_edit import edit
from sample_indels import draw_plan

ATOM37 = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2',
          'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2',
          'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']
SLOT = {n: i for i, n in enumerate(ATOM37)}
AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
       "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
BB4 = ("N", "CA", "C", "O")                      # indel_edit's backbone order
BB4_SLOT = [SLOT[n] for n in BB4]


def native_atom37(model, chain_id):
    resnames, xyz, mask = [], [], []
    for res in model[chain_id]:
        name = "MET" if res.resname == "MSE" else res.resname
        if (res.id[0] != " " and res.resname != "MSE") or name not in AA3 or not all(a in res for a in BB4):
            continue
        c = np.zeros((37, 3))
        m = np.zeros(37, bool)
        for atom in res:
            if atom.get_id() in SLOT:
                c[SLOT[atom.get_id()]] = atom.coord
                m[SLOT[atom.get_id()]] = True
        resnames.append(name)
        xyz.append(c)
        mask.append(m)
    return resnames, np.array(xyz), np.array(mask)


def write_pdb(path, resnames, xyz, mask):
    lines, n = [], 0
    for i, (rn, c, m) in enumerate(zip(resnames, xyz, mask), start=1):
        for s in np.flatnonzero(m):
            n += 1
            nm = ATOM37[s]
            lines.append(f"ATOM  {n:5d} {nm:<4s} {rn:>3s} A{i:4d}    {c[s][0]:8.3f}{c[s][1]:8.3f}{c[s][2]:8.3f}"
                         f"  1.00  0.00           {nm[0]:>2s}")
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")


def apply_plan(resnames, xyz, mask, ops):
    bb = xyz[:, BB4_SLOT]
    new_bb, orig, n2n = edit(bb, ops)
    Lp = len(orig)
    new_xyz = np.zeros((Lp, 37, 3))
    new_mask = np.zeros((Lp, 37), bool)
    new_names = []
    for i, o in enumerate(orig):
        if o >= 0:
            new_xyz[i], new_mask[i] = xyz[o], mask[o]
            new_names.append(resnames[o])
        else:
            new_xyz[i, BB4_SLOT] = new_bb[i]
            new_mask[i, BB4_SLOT] = True
            new_names.append("GLY")
    return new_names, new_xyz, new_mask, orig, n2n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--mmcif-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-draws", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    panel = pd.read_csv(a.panel)
    parser = MMCIFParser(QUIET=True)
    for r in panel.itertuples():
        d = os.path.join(a.out_dir, r.key)
        os.makedirs(d, exist_ok=True)
        model = parser.get_structure(r.pdb, f"{a.mmcif_dir}/{r.pdb}.cif")[0]
        names, xyz, mask = native_atom37(model, r.chain)
        L = len(names)
        assert L == r.n_parsed, (r.key, L, r.n_parsed)
        write_pdb(os.path.join(d, "native.pdb"), names, xyz, mask)
        plans = []
        for k in range(a.n_draws):
            rec = draw_plan(L, r.key, k, a.seed)
            ops = [tuple(o) for o in rec["ops"]]
            nn, nx, nm, orig, n2n = apply_plan(names, xyz, mask, ops)
            write_pdb(os.path.join(d, f"d{k:02d}.pdb"), nn, nx, nm)
            rec["orig_idx"], rec["native_to_new"], rec["L_new"] = orig.tolist(), n2n.tolist(), len(orig)
            plans.append(rec)
        json.dump({"key": r.key, "L": L, "native_resnames": names, "plans": plans}, open(os.path.join(d, "plans.json"), "w"))
        print(f"{r.key}: L={L}  L_new range {min(p['L_new'] for p in plans)}-{max(p['L_new'] for p in plans)}", flush=True)


if __name__ == "__main__":
    main()
