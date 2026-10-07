"""Fold the sequences of a mpnn_temp_make.py JSON with ESMFold2 (same settings as refold_esmfold2.py) -> refold-format npz per
(template, T): <out>/<chain>_t<i>_T<T>.npz (seqs = the folded designs; template_ca from the pool). Env: esmfold2."""
import argparse
import json
import os
import time

import numpy as np
import torch

import indel_pool as ip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--seqs-json", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
    os.makedirs(a.out, exist_ok=True)
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16", device="cuda").eval()
    builder = ESMFold2InputBuilder()
    for name, rec in json.load(open(a.seqs_json)).items():
        out = os.path.join(a.out, name + ".npz")
        if os.path.isfile(out):
            continue
        t = ip.read_template(ip.shard_path(a.pool_root, rec["chain"]), rec["i"])
        pos, pl, pt = [], [], []
        t0 = time.perf_counter()
        for s in rec["seqs"]:
            with torch.no_grad():
                r = builder.fold(model, StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=s)]), seed=0)
            pos.append(np.asarray(r.complex.to_protein_complex().atom37_positions, np.float16))
            pl.append(np.asarray(r.plddt.float().cpu() if torch.is_tensor(r.plddt) else r.plddt, np.float32).reshape(-1))
            pt.append(float(r.ptm))
        np.savez(out, pred_atom37=np.stack(pos), plddt=np.stack(pl), ptm=np.array(pt, np.float32),
                 template_ca=ip.atom37_coords(t)[:, 1], seqs=np.array(rec["seqs"]), seed=np.int32(0),
                 temp=np.float32(rec["T"]), n_unique=np.int32(rec["n_unique"]))
        print(f"{name}: {len(pos)} folds, {time.perf_counter() - t0:.0f}s, mean pLDDT {np.mean([x.mean() for x in pl]):.3f}", flush=True)


if __name__ == "__main__":
    main()
