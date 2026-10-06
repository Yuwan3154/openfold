"""Raygun two-step arm (user 10-06: match the Gly arm's insertion AND deletion proportions).

A single Raygun length change can only be a net stretch/squeeze; the Gly arm applies an independent deletion
fraction and insertion fraction per template. So per (chain, draw), using the SAME fractions the Gly plan drew:
  step 1  native (L) -> L1 = L - T_del   (contract; Raygun's uniform squeeze = the deletions)
  step 2  ESM-2 of the step-1 sequence -> L_new = L1 + T_ins (expand; the insertions)
(L_new equals the Gly plan's L_new by construction.) noise = --noise at BOTH steps (user chose 0.5), argmax decoding,
seed per item (crc32(chain) + draw). Output JSON in raygun_generate.py's format (+ step-1 sequences).
Env: raygun. Run: python raygun_generate2.py --inputs-dir <inputs> --out raygun_seqs2.json [--noise 0.5]
"""
import argparse
import json
import os
import zlib

import torch
from esm.pretrained import esm2_t33_650M_UR50D

from raygun.pretrained import raygun_8_8mil_800M
from raygun_probe import THREE2ONE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chains", nargs="*", default=None)
    a = p.parse_args()
    dev = torch.device(a.device)
    raymodel = raygun_8_8mil_800M().to(dev).eval()
    esm, alph = esm2_t33_650M_UR50D()
    esm = esm.to(dev).eval()
    bc = alph.get_batch_converter()

    def embed(key, s):
        _, _, tok = bc([(key, s)])
        return esm(tok.to(dev), repr_layers=[33], return_contacts=False)["representations"][33][:, 1:-1]

    def gen(emb, n):
        return raymodel(emb, target_lengths=torch.tensor([n], dtype=int), noise=a.noise,
                        return_logits_and_seqs=True)["generated-sequences"][0]

    out = {}
    for key in (a.chains if a.chains else sorted(os.listdir(a.inputs_dir))):
        plans = json.load(open(os.path.join(a.inputs_dir, key, "plans.json")))
        seq = "".join(THREE2ONE[r] for r in plans["native_resnames"])
        L = len(seq)
        mids, finals, tl = [], [], []
        with torch.no_grad():
            emb0 = embed(key, seq)
            for k, pl in enumerate(plans["plans"]):
                n_del, n_ins = pl["del"]["T"], pl["ins"]["T"]
                L1, L2 = L - n_del, L - n_del + n_ins
                assert L2 == pl["L_new"] and L1 >= 2
                torch.manual_seed((zlib.crc32(key.encode()) + k) % 2**31)
                s1 = gen(emb0, L1)
                s2 = gen(embed(key, s1), L2)
                assert len(s1) == L1 and len(s2) == L2
                mids.append(s1)
                finals.append(s2)
                tl.append(L2)
        out[key] = {"native_seq": seq, "noise": a.noise, "target_lengths": tl, "seqs": finals, "step1": mids}
        json.dump(out, open(a.out, "w"))
        print(f"{key}: {len(finals)} sequences", flush=True)
    print(f"wrote {sum(len(v['seqs']) for v in out.values())} sequences to {a.out}", flush=True)


if __name__ == "__main__":
    main()
