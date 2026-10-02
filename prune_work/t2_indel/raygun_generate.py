"""Raygun sequences for the indel panel (Option 1, user 10-02): one sequence per (chain, draw) at the draw's
L_new (net edit size matches the Gly arm), noise fixed (default 0.5), decoding argmax. Seed per item =
(crc32(chain) + draw) % 2**31. Output JSON {chain: {native_seq, noise, target_lengths, seqs}}.
Env: raygun (CPU is enough). Run: python raygun_generate.py --inputs-dir <inputs> --out raygun_seqs.json [--noise 0.5]
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
    out = {}
    for key in (a.chains if a.chains else sorted(os.listdir(a.inputs_dir))):
        plans = json.load(open(os.path.join(a.inputs_dir, key, "plans.json")))
        seq = "".join(THREE2ONE[r] for r in plans["native_resnames"])
        tl = [pl["L_new"] for pl in plans["plans"]]
        seqs = []
        with torch.no_grad():
            _, _, tok = bc([(key, seq)])
            emb = esm(tok.to(dev), repr_layers=[33], return_contacts=False)["representations"][33][:, 1:-1]
            for k, n in enumerate(tl):
                torch.manual_seed((zlib.crc32(key.encode()) + k) % 2**31)
                s = raymodel(emb, target_lengths=torch.tensor([n], dtype=int), noise=a.noise,
                             return_logits_and_seqs=True)["generated-sequences"][0]
                assert len(s) == n
                seqs.append(s)
        out[key] = {"native_seq": seq, "noise": a.noise, "target_lengths": tl, "seqs": seqs}
        json.dump(out, open(a.out, "w"))
        print(f"{key}: {len(seqs)} sequences", flush=True)
    print(f"wrote {sum(len(v['seqs']) for v in out.values())} sequences to {a.out}", flush=True)


if __name__ == "__main__":
    main()
