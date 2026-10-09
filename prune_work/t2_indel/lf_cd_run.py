"""ColabDesign fp32 AF2 on the lf_ctl.py controlled cases, run on the SAME node as localfold (user 10-08). Same sequence and the SAME template PDB file as localfold.
Settings as refold_af2.py / af2_complete.py: fixbb, model_1_ptm only, 3 recycles, fp32, no MSA, default seed; template cases use_templates=True (ColabDesign defaults rm_template_seq/sc).
Writes <out>/<case>.npz: ca (L,3), pred_atom37 (L,37,3), plddt (L,), ptm. Env: ~/.conda/envs/colabdesign. Run: python lf_cd_run.py cases.json OUT_DIR PARAMS_DIR_PARENT
"""
import json
import os
import sys

import jax
import numpy as np
from colabdesign import mk_afdesign_model

cases, out, data_dir = sys.argv[1:4]
int5 = os.environ.get("CD_INT5") == "1"   # emulate localfold's int5 bundle: per-32-block asymmetric 32-level quantisation of every weight tensor with ndim >= 2


def quant5(x):
    x = np.asarray(x)
    if x.ndim < 2 or x.size % 32:
        return x
    b = x.astype(np.float32).reshape(-1, 32)
    lo, hi = b.min(1, keepdims=True), b.max(1, keepdims=True)
    sc = np.where(hi > lo, (hi - lo) / 31, 1.0)
    return (np.round((b - lo) / sc) * sc + lo).reshape(x.shape).astype(x.dtype)


keep_seq = os.environ.get("CD_KEEP_TEMPLATE_SEQ") == "1"   # rm_template_seq/sc False: the template carries its residue types and side-chain atoms, as localfold's does
os.makedirs(out, exist_ok=True)
models = {}
for c in json.load(open(cases)):
    use_t = c["template"] is not None
    if use_t not in models:
        models[use_t] = mk_afdesign_model(protocol="fixbb", use_templates=use_t, data_dir=data_dir, model_names=["model_1_ptm"], use_bfloat16=False)
    m = models[use_t]
    if int5 and not getattr(m, "_q5", False):
        m._model_params = [jax.tree_util.tree_map(quant5, p) for p in m._model_params]
        m._q5 = True
    pdb = os.path.expandvars(c["template"]) if use_t else os.path.expandvars(f"$HOME/lf_parity/native/{c['name'].split('_', 1)[1]}.pdb")
    m.prep_inputs(pdb_filename=pdb, chain="A", rm_template_seq=not keep_seq, rm_template_sc=not keep_seq)
    m.predict(seq=c["seq"], models=[0], num_recycles=3, verbose=False)
    a = m.aux
    pos = np.asarray(a["atom_positions"], np.float32)
    np.savez(os.path.join(out, c["name"] + ".npz"), ca=pos[:, 1], pred_atom37=pos, plddt=100 * np.asarray(a["plddt"], np.float32), ptm=np.float32(a["log"]["ptm"]))
    print(c["name"], len(c["seq"]), "pTM %.3f pLDDT %.1f" % (float(a["log"]["ptm"]), 100 * float(np.asarray(a["plddt"]).mean())), flush=True)
