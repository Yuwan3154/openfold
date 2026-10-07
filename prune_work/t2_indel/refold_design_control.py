"""Stage 2 of the design control: fold the NATIVE sequence + the 32 ProteinMPNN sequences of each NATIVE backbone with ESMFold2.
If these refold (paired TM ~ high) while the redesigned TEMPLATE backbones do not, the template backbones are the limit.
Output: refold-format npz (template_ca = native CA). Env: esmfold2. Run: python refold_design_control.py --work-dir w --inputs-dir inputs --chains ... --out out
"""
import argparse
import os

import numpy as np
import torch

from refold_native_control import native


def read_fa(path):
    seqs, cur = [], None
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith(">"):
            cur = []
            seqs.append(cur)
        elif ln:
            cur.append(ln)
    return ["".join(s) for s in seqs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
    os.makedirs(a.out, exist_ok=True)
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16", device="cuda").eval()
    builder = ESMFold2InputBuilder()
    for chain in a.chains:
        seq, ca, _ = native(a.inputs_dir, chain)
        recs = read_fa(os.path.join(a.work_dir, "seqs", f"{chain}.fa"))
        assert recs[0] == seq, f"{chain}: MPNN native sequence != the native sequence"
        seqs = recs[:1] + recs[1:]
        pos, pl, pt = [], [], []
        for s in seqs:
            with torch.no_grad():
                res = builder.fold(model, StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=s)]), seed=0)
            pos.append(np.asarray(res.complex.to_protein_complex().atom37_positions, np.float16))
            pl.append(np.asarray(res.plddt.float().cpu() if torch.is_tensor(res.plddt) else res.plddt, np.float32).reshape(-1))
            pt.append(float(res.ptm))
        np.savez(os.path.join(a.out, f"{chain}_t000.npz"), pred_atom37=np.stack(pos), plddt=np.stack(pl), ptm=np.array(pt, np.float32),
                 template_ca=ca.astype(np.float32), seqs=np.array(seqs), seed=np.int32(0))
        print(f"{chain}: {len(seqs)} predictions, native pLDDT {pl[0].mean():.3f}, designs {np.mean([x.mean() for x in pl[1:]]):.3f}", flush=True)


if __name__ == "__main__":
    main()
