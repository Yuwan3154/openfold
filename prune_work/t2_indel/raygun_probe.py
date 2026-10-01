"""How does Raygun treat a sequence when asked for a different length? (answers the user's 10-01 questions)

For each panel chain: native sequence (resolved residues) -> Raygun at target length L (identity check) and at
the edited lengths L_new of the first --n-plans draws, over a noise grid, argmax decoding (temperature None).
Each output is globally aligned to the native (Biopython PairwiseAligner, BLOSUM62, open -10, extend -0.5; these
are standard protein-alignment defaults, not tuned). Per output: identity over aligned pairs, number of gaps in
each sequence, and where the gaps fall (positions normalised to 0..1 of the NATIVE axis, so uniform spreading vs
clustering is visible in the pooled histogram). Sequences are saved so the alignment can be redone.
Env: raygun. Run: python raygun_probe.py --inputs-dir <inputs> --out probe.json [--n-plans 4] [--noise 0 0.05 0.1 0.2]
"""
import argparse
import json
import os

import numpy as np
import torch
from Bio import Align
from Bio.Align import substitution_matrices
from esm.pretrained import esm2_t33_650M_UR50D

from raygun.pretrained import raygun_8_8mil_800M

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
             "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
             "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def aligner():
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score, al.mode = -10.0, -0.5, "global"
    return al


def compare(al, native, gen):
    aln = al.align(native, gen)[0]
    n_blocks, g_blocks = aln.aligned
    matches = pairs = 0
    for (n0, n1), (g0, g1) in zip(n_blocks, g_blocks):
        for a, b in zip(native[n0:n1], gen[g0:g1]):
            pairs += 1
            matches += a == b
    # gaps in the generated sequence = native residues with no partner (deletions); gaps in the native = extra generated residues
    covered_native = np.zeros(len(native), bool)
    covered_gen = np.zeros(len(gen), bool)
    for (n0, n1), (g0, g1) in zip(n_blocks, g_blocks):
        covered_native[n0:n1] = True
        covered_gen[g0:g1] = True
    ins_pos = (np.flatnonzero(~covered_gen) / max(len(gen) - 1, 1)).round(3).tolist()
    del_pos = (np.flatnonzero(~covered_native) / max(len(native) - 1, 1)).round(3).tolist()
    return dict(identity=matches / max(pairs, 1), n_pairs=pairs, n_extra_in_generated=int((~covered_gen).sum()),
                n_native_unmatched=int((~covered_native).sum()), extra_pos=ins_pos, unmatched_pos=del_pos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-plans", type=int, default=4)
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--repeats", type=int, default=3, help="independent draws per noise > 0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--chains", nargs="*", default=None)
    a = p.parse_args()
    dev = torch.device(a.device)
    raymodel = raygun_8_8mil_800M().to(dev).eval()
    esm, alph = esm2_t33_650M_UR50D()
    esm = esm.to(dev).eval()
    bc = alph.get_batch_converter()
    al = aligner()
    torch.manual_seed(0)
    rows = []
    keys = a.chains if a.chains else sorted(os.listdir(a.inputs_dir))
    for key in keys:
        plans = json.load(open(os.path.join(a.inputs_dir, key, "plans.json")))
        seq = "".join(THREE2ONE[r] for r in plans["native_resnames"])
        with torch.no_grad():
            _, _, tok = bc([(key, seq)])
            emb = esm(tok.to(dev), repr_layers=[33], return_contacts=False)["representations"][33][:, 1:-1]
            targets = [("same", len(seq))] + [(f"d{k:02d}", plans["plans"][k]["L_new"]) for k in range(a.n_plans)]
            for tag, tl in targets:
                for nz in a.noise:
                    for rep in range(1 if nz == 0 else a.repeats):
                        out = raymodel(emb, target_lengths=torch.tensor([tl], dtype=int), noise=nz,
                                       return_logits_and_seqs=True)["generated-sequences"][0]
                        assert len(out) == tl
                        rec = compare(al, seq, out)
                        rec.update(chain=key, target=tag, L=len(seq), L_target=tl, noise=nz, rep=rep, seq=out)
                        rows.append(rec)
        sub = [r for r in rows if r["chain"] == key and r["target"] == "same" and r["noise"] == 0]
        print(f"{key}: L={len(seq)} identity at same length, noise 0: {sub[0]['identity']:.3f}", flush=True)
        json.dump({"rows": rows}, open(a.out, "w"))
    json.dump({"rows": rows}, open(a.out, "w"))
    print(f"wrote {len(rows)} rows to {a.out}", flush=True)


if __name__ == "__main__":
    main()
