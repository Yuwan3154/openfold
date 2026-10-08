"""Refold test, ESMFold2 (esm 3.4.1.post1, env `esmfold2` on SuperCloud): same I/O as refold_af2.py.

Settings (CHOSEN, ledger): the installed fold() defaults (num_loops 20, num_sampling_steps 200, 1 diffusion sample), seed 0,
NO outer autocast (the usage guide's rule), model loaded once. Writes <out>/<chain>_t<i>.npz with pred_atom37 (S,L,37,3 f16),
plddt (S,L) [0-1 scale, the model's own], ptm (S,), template_ca, seqs. Env: esmfold2 on a GPU node (weights in the HF cache).
Run: python refold_esmfold2.py --pool-root pool --select chain:i ... --out out --tmp-dir t [--max-seqs 32] [--esmc-precision bf16]
"""
import argparse
import os
import time

import numpy as np
import torch

import indel_pool as ip
from atomic_io import atomic_savez


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--select", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument("--esmc-precision", default="bf16")
    p.add_argument("--tmp-dir", default=None)
    a = p.parse_args()
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
    os.makedirs(a.out, exist_ok=True)
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision=a.esmc_precision, device="cuda").eval()
    print("model loaded", flush=True)
    builder = ESMFold2InputBuilder()
    for sel in a.select:
        chain, i = sel.split(":")
        i = int(i)
        out = os.path.join(a.out, f"{chain}_t{i:03d}.npz")
        if os.path.isfile(out):
            continue
        t = ip.read_template(ip.shard_path(a.pool_root, chain), i)
        seqs = ["".join(ip.AA_ORDER[int(x)] for x in t["aatype"])]
        seqs += ["".join(ip.AA_ORDER[int(x)] for x in row) for row in t["design_aatype"][: a.max_seqs]]
        pos, pl, pt = [], [], []
        t0 = time.perf_counter()
        for s in seqs:
            spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=s)])
            with torch.no_grad():
                res = builder.fold(model, spi, seed=0)
            pc = res.complex.to_protein_complex()
            assert pc.atom37_positions.shape[0] == len(s), (pc.atom37_positions.shape, len(s))
            pos.append(np.asarray(pc.atom37_positions, np.float16))
            pl.append(np.asarray(res.plddt.float().cpu() if torch.is_tensor(res.plddt) else res.plddt, np.float32).reshape(-1))
            pt.append(float(res.ptm))
        atomic_savez(out, pred_atom37=np.stack(pos), plddt=np.stack(pl), ptm=np.array(pt, np.float32),
                 template_ca=ip.atom37_coords(t)[:, 1], seqs=np.array(seqs), seed=np.int32(0))
        print(f"{chain} t{i}: {len(seqs)} predictions, L={len(seqs[0])}, {time.perf_counter() - t0:.0f}s, "
              f"mean pLDDT first={pl[0].mean():.3f} designs={np.mean([x.mean() for x in pl[1:]]):.3f}", flush=True)


if __name__ == "__main__":
    main()
