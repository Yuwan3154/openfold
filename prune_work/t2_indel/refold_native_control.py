"""KNOWN-GOOD CONTROL for the refold test: predict the NATIVE sequence of a panel chain and compare with its NATIVE backbone.
If a predictor cannot refold its own native here (single sequence, no MSA, same settings as the pilot), then low design
refold TM says nothing about the designs. Output npz is in the refold_* format (template_ca = native CA), so
refold_score.py scores it with --control (no pool lookup). Run: python refold_native_control.py --method af2|esm
--inputs-dir inputs --chains 7du7_A ... --out out
"""
import argparse
import json
import os

import numpy as np

from atomic_io import atomic_savez

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def native(inputs_dir, chain):
    plans = json.load(open(os.path.join(inputs_dir, chain, "plans.json")))
    seq = "".join(AA3[r] for r in plans["native_resnames"])
    pdb = os.path.join(inputs_dir, chain, "native.pdb")
    ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb)
                   if l.startswith("ATOM") and l[12:16].strip() == "CA"])
    assert len(ca) == len(seq)
    return seq, ca, pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["af2", "esm"])
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.method == "af2":
        from colabdesign import mk_afdesign_model
        model = mk_afdesign_model(protocol="fixbb", use_templates=False, data_dir=os.path.expanduser("~"),
                                  model_names=["model_1_ptm"], use_bfloat16=False)
    else:
        import torch
        from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
        model = EsmFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16", device="cuda").eval()
        builder = ESMFold2InputBuilder()
    for chain in a.chains:
        seq, ca, pdb = native(a.inputs_dir, chain)
        if a.method == "af2":
            model.prep_inputs(pdb_filename=pdb, chain="A")
            model.predict(seq=seq, models=[0], num_recycles=3, verbose=False)
            pos, pl, pt = np.asarray(model.aux["atom_positions"], np.float16), np.asarray(model.aux["plddt"], np.float32), float(model.aux["log"]["ptm"])
        else:
            with torch.no_grad():
                res = builder.fold(model, StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)]), seed=0)
            pos = np.asarray(res.complex.to_protein_complex().atom37_positions, np.float16)
            pl, pt = np.asarray(res.plddt.float().cpu() if torch.is_tensor(res.plddt) else res.plddt, np.float32).reshape(-1), float(res.ptm)
        atomic_savez(os.path.join(a.out, f"{chain}_t000.npz"), pred_atom37=pos[None], plddt=pl[None], ptm=np.array([pt], np.float32),
                 template_ca=ca.astype(np.float32), seqs=np.array([seq]), seed=np.int32(0))
        print(f"{chain}: native L={len(seq)} mean pLDDT {pl.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
