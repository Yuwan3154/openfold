"""AF2 sequence-structure COMPATIBILITY test of the synthetic templates (user 10-08; AF2Rank-style).

Template = the PARTIALLY DIFFUSED pool template backbone (N, CA, C, O; nothing removed), query = (seq_kind 0) the template's own INPUT sequence
(the sequence the partial diffusion was run with: native for controls, the edited arm sequence otherwise) and (seq_kind j) its first --n-designs
ProteinMPNN designs. AF2 (ColabDesign fixbb + use_templates, model_1_ptm, 3 recycles, fp32, no MSA, same settings as af2_complete.py; ColabDesign defaults
rm_template_seq/sc True so the template carries backbone geometry only) predicts each query; high pTM/pLDDT and a prediction that stays on the template
mean AF2 finds the sequence compatible with the structure. Writes <out>/<chain>_t<i:03d>.npz: pred_atom37 (S,L,37,3 f16), plddt (S,L), ptm (S,),
seq_kind (S,), seqs (S,), template_ca (L,3), num_recycles. Env: colabdesign on a GPU node.
Run: python af2_compat.py --pool-root POOL --select chain:i ... --out OUT [--n-designs 4]
"""
import argparse
import os
import tempfile
import time

import numpy as np
from colabdesign import mk_afdesign_model

import indel_pool as ip
from atomic_io import atomic_savez
from mpnn_design_pool import write_backbone_pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--select", nargs="+", required=True, help="chain:template_index")
    p.add_argument("--out", required=True)
    p.add_argument("--n-designs", type=int, default=4)
    p.add_argument("--num-recycles", type=int, default=3)
    p.add_argument("--data-dir", default=os.path.expanduser("~"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # V100 (sm_70) has no bfloat16 units: fp32 (the default use_bfloat16=True segfaulted, see refold_af2.py)
    model = mk_afdesign_model(protocol="fixbb", use_templates=True, data_dir=a.data_dir, model_names=["model_1_ptm"], use_bfloat16=False)
    for sel in a.select:
        chain, i = sel.split(":")
        i = int(i)
        out = os.path.join(a.out, f"{chain}_t{i:03d}.npz")
        if os.path.isfile(out):
            continue
        t = ip.read_template(ip.shard_path(a.pool_root, chain), i)
        assert t["design_aatype"].shape[0] >= a.n_designs, (chain, i)
        seqs = ["".join(ip.AA_ORDER[int(x)] for x in t["aatype"])]
        seqs += ["".join(ip.AA_ORDER[int(x)] for x in row) for row in t["design_aatype"][: a.n_designs]]
        with tempfile.TemporaryDirectory() as td:
            pdb = os.path.join(td, "t.pdb")
            write_backbone_pdb(pdb, t["coords"], t["atom_mask"], t["aatype"])
            model.prep_inputs(pdb_filename=pdb, chain="A")
        assert all(len(s) == len(seqs[0]) for s in seqs), (chain, i, "query lengths differ")
        pos, pl, pt = [], [], []
        t0 = time.perf_counter()
        for s in seqs:
            model.predict(seq=s, models=[0], num_recycles=a.num_recycles, verbose=False)
            aux = model.aux
            pos.append(np.asarray(aux["atom_positions"], np.float16))
            pl.append(np.asarray(aux["plddt"], np.float32))
            pt.append(float(aux["log"]["ptm"]))
        atomic_savez(out, pred_atom37=np.stack(pos), plddt=np.stack(pl), ptm=np.array(pt, np.float32), seq_kind=np.arange(len(seqs)),
                     seqs=np.array(seqs), template_ca=ip.atom37_coords(t)[:, 1].astype(np.float32), num_recycles=np.int32(a.num_recycles))
        print(f"{chain} t{i}: {len(seqs)} predictions, L={len(seqs[0])}, {time.perf_counter() - t0:.0f}s, "
              f"input-seq pTM {pt[0]:.2f} pLDDT {100 * pl[0].mean():.1f} | designs pTM {np.mean(pt[1:]):.2f}", flush=True)


if __name__ == "__main__":
    main()
