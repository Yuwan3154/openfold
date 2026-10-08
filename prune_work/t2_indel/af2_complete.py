"""AF2 completion arm (user 10-07): the template is the SURVIVOR-ONLY native structure, the query is the post-edit synthetic sequence of an arm,
and AF2 predicts the whole chain, so inserted / terminal-extended positions get coordinates from AF2 instead of interpolation or partial diffusion.

Input = an arm's input PDB d<k>.pdb (renumbered 1..L'; survivors keep their native atoms, inserted residues carry placeholder atoms) plus
plans.json orig_idx (-1 = inserted). Inserted positions are removed from the template (ColabDesign rm_template, contig string "A<i>-<j>");
the ColabDesign defaults rm_template_seq=True / rm_template_sc=True stay (template = survivor backbone geometry only, query sequence carries identity).
Query sequence = the residue names of the PDB (Gly arm: GLY at inserts; composition arm: drawn types; Raygun arm: the generated sequence).
Settings (CHOSEN, ledger, same as refold_af2.py): ColabDesign fixbb + use_templates, model_1_ptm only, num_recycles 3, fp32 (V100), default seed.
Writes <out>/<chain>_d<k>.npz: pred_atom37 (L',37,3), plddt (L',), ptm, seq, orig_idx, rm (bool L'). Env: colabdesign on a GPU node.
Run: python af2_complete.py --inputs-dir D --chains c.. --draws k.. --out OUT
"""
import argparse
import json
import os
import tempfile
import time

import numpy as np
from colabdesign import mk_afdesign_model

THREE2ONE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K",
             "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
             "TRP": "W", "TYR": "Y"}


def pdb_sequence(path):
    seq = {}
    for ln in open(path):
        if ln.startswith("ATOM"):
            seq[int(ln[22:26])] = THREE2ONE[ln[17:20].strip()]
    assert sorted(seq) == list(range(1, len(seq) + 1)), path
    return "".join(seq[i] for i in sorted(seq))


def clean_pdb(src, dst):
    """Heavy atoms only, standard columns, chain A, no MODEL: the Raygun-arm inputs are cg2all (CHARMM) files with hydrogens (HT1, HA, ...),
    a MODEL record and OT1/OT2 terminal oxygens, which Biopython/ColabDesign rejects ('Empty file'). OT1 -> O, OT2/OXT dropped."""
    out, n, names = [], 0, {}
    for ln in open(src):
        if not ln.startswith("ATOM"):
            continue
        name = ln[12:16].strip()
        if name[0] == "H" or name[0].isdigit() or name in ("OT2", "OXT"):
            continue
        name = "O" if name == "OT1" else name
        assert ln[17:20] in THREE2ONE, (src, ln[17:20])
        names.setdefault(int(ln[22:26]), set()).add(name)
        n += 1
        out.append(f"ATOM  {n:5d} {name:<4s} {ln[17:20]} A{int(ln[22:26]):4d}    {ln[30:54]}  1.00  0.00           {name[0]:>2s}")
    assert n > 0 and sorted(names) == list(range(1, len(names) + 1)), src
    assert all({"N", "CA", "C", "O"} <= v for v in names.values()), (src, [i for i, v in names.items() if not {"N", "CA", "C", "O"} <= v][:5])
    open(dst, "w").write("\n".join(out) + "\nEND\n")


def runs(mask):
    """Contig string of the True runs of a bool array, 1-based: [F,T,T,F,T] -> 'A2-3,A5-5'."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append(f"A{i + 1}-{j + 1}")
            i = j + 1
        else:
            i += 1
    return ",".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=os.path.expanduser("~"))
    p.add_argument("--num-recycles", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # V100 (sm_70) has no bfloat16 units: the default use_bfloat16=True segfaulted (refold_af2.py shakedown)
    model = mk_afdesign_model(protocol="fixbb", use_templates=True, data_dir=a.data_dir, model_names=["model_1_ptm"], use_bfloat16=False)
    print("model built", flush=True)
    for c in a.chains:
        plans = json.load(open(os.path.join(a.inputs_dir, c, "plans.json")))["plans"]
        for k in a.draws:
            out = os.path.join(a.out, f"{c}_d{k:02d}.npz")
            if os.path.isfile(out):
                continue
            raw = os.path.join(a.inputs_dir, c, f"d{k:02d}.pdb")
            td = tempfile.TemporaryDirectory()
            pdb = os.path.join(td.name, "t.pdb")
            clean_pdb(raw, pdb)
            seq = pdb_sequence(pdb)
            orig = np.array(plans[k]["orig_idx"])
            rm = orig < 0
            assert len(seq) == len(orig), (c, k, len(seq), len(orig))
            kw = {"rm_template": runs(rm)} if rm.any() else {}
            model.prep_inputs(pdb_filename=pdb, chain="A", **kw)
            td.cleanup()
            t0 = time.perf_counter()
            model.predict(seq=seq, models=[0], num_recycles=a.num_recycles, verbose=False)
            aux = model.aux
            tmp = out + ".tmp.npz"
            np.savez(tmp, pred_atom37=np.asarray(aux["atom_positions"], np.float32), plddt=np.asarray(aux["plddt"], np.float32),
                     ptm=np.float32(aux["log"]["ptm"]), seq=np.array(seq), orig_idx=orig, rm=rm, num_recycles=np.int32(a.num_recycles))
            os.replace(tmp, out)
            print(f"{c} d{k:02d}: L'={len(seq)} inserted={int(rm.sum())} {time.perf_counter() - t0:.0f}s "
                  f"pLDDT {100 * np.asarray(aux['plddt']).mean():.1f} ptm {float(aux['log']['ptm']):.2f}", flush=True)


if __name__ == "__main__":
    main()
