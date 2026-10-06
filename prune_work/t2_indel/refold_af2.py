"""Refold test, AF2 (ColabDesign, single sequence, NO MSA, NO templates): does a redesigned sequence fold back to its backbone?

For each selected pool template: the generation sequence plus its 32 ProteinMPNN sequences are predicted and compared
(USalign, later, on CPU) with the template backbone. Settings (CHOSEN, ledger): AF2 model_1_ptm only (models=[0]),
num_recycles 3 (the AF2 standard), seed 0. Params come from <data-dir>/params (SuperCloud ~/params -> openfold resources).
Writes <out>/<chain>_t<i>.npz: pred_atom37 (S,L,37,3 float16), plddt (S,L), ptm (S,), seq_kind (S,) 0 = generation seq,
1.. = design j, plus the template CA for scoring. Env: colabdesign (jax[cuda12]) on a GPU node.
Run: python refold_af2.py --pool-root pool --select chain:i chain:i ... --out out [--max-seqs 32]
"""
import argparse
import os
import time

import numpy as np

import indel_pool as ip
from mpnn_design_pool import write_backbone_pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--select", nargs="+", required=True, help="chain:template_index")
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=os.path.expanduser("~"))
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument("--num-recycles", type=int, default=3)
    p.add_argument("--tmp-dir", required=True)
    a = p.parse_args()
    from colabdesign import mk_afdesign_model
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp_dir, exist_ok=True)
    # V100 (sm_70) has no bfloat16 units and the default use_bfloat16=True segfaulted the first shakedown (rc 139): run fp32
    model = mk_afdesign_model(protocol="fixbb", use_templates=False, data_dir=a.data_dir, model_names=["model_1_ptm"],
                              use_bfloat16=False)
    print("model built", flush=True)
    for sel in a.select:
        chain, i = sel.split(":")
        i = int(i)
        out = os.path.join(a.out, f"{chain}_t{i:03d}.npz")
        if os.path.isfile(out):
            continue
        path = ip.shard_path(a.pool_root, chain)
        t = ip.read_template(path, i)
        pdb = os.path.join(a.tmp_dir, f"{chain}_t{i:03d}.pdb")
        write_backbone_pdb(pdb, t["coords"], t["atom_mask"], t["aatype"])
        model.prep_inputs(pdb_filename=pdb, chain="A")
        print(f"{chain} t{i}: inputs prepared, L={len(t['aatype'])}", flush=True)
        seqs = ["".join(ip.AA_ORDER[int(x)] for x in t["aatype"])]
        seqs += ["".join(ip.AA_ORDER[int(x)] for x in row) for row in t["design_aatype"][: a.max_seqs]]
        pos, pl, pt = [], [], []
        t0 = time.perf_counter()
        for s in seqs:
            model.predict(seq=s, models=[0], num_recycles=a.num_recycles, verbose=False)
            aux = model.aux
            pos.append(np.asarray(aux["atom_positions"], np.float16))
            pl.append(np.asarray(aux["plddt"], np.float32))
            pt.append(float(aux["log"]["ptm"]))
        np.savez(out, pred_atom37=np.stack(pos), plddt=np.stack(pl), ptm=np.array(pt, np.float32),
                 template_ca=ip.atom37_coords(t)[:, 1], seqs=np.array(seqs), seed=np.int32(0), num_recycles=np.int32(a.num_recycles))
        print(f"{chain} t{i}: {len(seqs)} predictions, L={len(seqs[0])}, {time.perf_counter() - t0:.0f}s, "
              f"mean pLDDT first={pl[0].mean():.1f} designs={np.mean([x.mean() for x in pl[1:]]):.1f}", flush=True)


if __name__ == "__main__":
    main()
