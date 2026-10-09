"""Controlled localfold-vs-ColabDesign parity set (user 10-08: CA RMSD < 1 A and |dpTM| < 0.01). Cases (identical sequence + template for both implementations):
t1 = template-free native sequence; t2 = native sequence + native backbone/CB template (named native); t3 = control-pool template (named = query) with its k-th query
(input sequence k0 / MPNN design k1). Subcommands:
  make  NATIVE T3_DIR OUT.sh REMOTE_OUT LF_CMD [ENV...]  write the run script (templates already on the node: ~/lf_seqdep2/tpl, ~/lf_t3/tpl)
  cmp   NATIVE T3_DIR REF_T1 REF_T2 LF_OUT_DIR          compare every case present in LF_OUT_DIR with the ColabDesign fp32 references
"""
import json
import os
import sys

import numpy as np

from lf_parity import native_seq, read_ca, rmsd

T1T2_CHAINS = ["7du7_A", "6x61_B", "6kyf_A"]
T3_CASES = [("control", "7du7_A", 2, 0), ("control", "7du7_A", 2, 1), ("control", "6kyf_A", 2, 0), ("control", "6kyf_A", 2, 1)]


def make(native, t3, out, rout, lf, env):
    lines = ["#!/bin/bash", "export PATH=/usr/local/cuda-12.6/bin:$PATH", f"mkdir -p {rout}"] + [f"export {e}" for e in env]
    for c in T1T2_CHAINS:
        seq = native_seq(os.path.join(native, c + ".pdb"))
        lines.append(f"{lf} --sequence={seq} --recycles=3 --out={rout}/t1_{c}.pdb 2>&1 | grep '^mean'")
        lines.append(f"{lf} --sequence={seq} --recycles=3 --template=$HOME/lf_seqdep2/tpl/{c}_nat.pdb:A --out={rout}/t2_{c}.pdb 2>&1 | grep '^mean'")
    for s, c, t, k in T3_CASES:
        q = str(np.load(os.path.join(t3, f"af2compat_{s}", f"{c}_t{t:03d}.npz"))["seqs"][k])
        lines.append(f"{lf} --sequence={q} --recycles=3 --template=$HOME/lf_t3/tpl/{s}_{c}_t{t:03d}_k{k}_q.pdb:A --out={rout}/t3_{s}_{c}_t{t:03d}_k{k}.pdb 2>&1 | grep '^mean'")
    open(out, "w").write("\n".join(lines) + "\n")


def cases(native, t3, out):
    """cases.json for lf_cd_run.py: name, sequence, template pdb on the node (None = template-free)."""
    cs = []
    for c in T1T2_CHAINS:
        seq = native_seq(os.path.join(native, c + ".pdb"))
        cs += [dict(name=f"t1_{c}", seq=seq, template=None), dict(name=f"t2_{c}", seq=seq, template=f"$HOME/lf_seqdep2/tpl/{c}_nat.pdb")]
    for s, c, t, k in T3_CASES:
        q = str(np.load(os.path.join(t3, f"af2compat_{s}", f"{c}_t{t:03d}.npz"))["seqs"][k])
        cs.append(dict(name=f"t3_{s}_{c}_t{t:03d}_k{k}", seq=q, template=f"$HOME/lf_t3/tpl/{s}_{c}_t{t:03d}_k{k}_q.pdb"))
    json.dump(cs, open(out, "w"), indent=1)


def ref(name, t3, ref1, ref2):
    if ref1 == ref2:   # same-hardware refs from lf_cd_run.py
        assert os.path.isfile(os.path.join(ref1, name + ".npz")), name
        z = np.load(os.path.join(ref1, name + ".npz"))
        return z["ca"].astype(np.float64), float(z["ptm"]), float(z["plddt"].mean())
    if name.startswith("t1_"):
        z, k = np.load(os.path.join(ref1, name[3:] + "_t000.npz")), 0
    elif name.startswith("t2_"):
        z, k = np.load(os.path.join(ref2, name[3:] + "_native.npz")), 0
    else:
        s, c, t, k = name[3:].split("_")[0], "_".join(name[3:].split("_")[1:3]), int(name.split("_t")[-1].split("_")[0]), int(name.split("_k")[-1])
        z = np.load(os.path.join(t3, f"af2compat_{s}", f"{c}_t{t:03d}.npz"))
    ca = z["pred_atom37"].reshape(-1, z["pred_atom37"].shape[-3], 37, 3)[k][:, 1].astype(np.float64)
    return ca, float(np.asarray(z["ptm"]).reshape(-1)[k]), 100 * float(np.asarray(z["plddt"]).reshape(-1, len(ca))[k].mean())


def cmp(native, t3, ref1, ref2, lfdir):
    print(f"{'case':28s} {'rmsd':>6s} {'dpTM':>7s} {'ptm_lf':>7s} {'ptm_cd':>7s} {'dpLDDT':>7s}  pass(rmsd<1 & |dpTM|<0.01)")
    n_pass = n = 0
    for f in sorted(os.listdir(lfdir)):
        if not f.endswith(".pdb"):
            continue
        name = f[:-4]
        ca, pl = read_ca(os.path.join(lfdir, f))
        rca, rptm, rpl = ref(name, t3, ref1, ref2)
        ptm = float(json.load(open(os.path.join(lfdir, name + "_summary_confidences.json")))["ptm"])
        r, d = rmsd(ca, rca), ptm - rptm
        ok = r < 1 and abs(d) < 0.01
        n, n_pass = n + 1, n_pass + ok
        print(f"{name:28s} {r:6.2f} {d:+7.3f} {ptm:7.3f} {rptm:7.3f} {pl.mean() - rpl:+7.1f}  {'PASS' if ok else 'fail'}")
    assert n == 2 * len(T1T2_CHAINS) + len(T3_CASES), f"{n} cases in {lfdir}, expected {2 * len(T1T2_CHAINS) + len(T3_CASES)}"
    print(f"{n_pass} of {n} cases pass")


if __name__ == "__main__":
    if sys.argv[1] == "cases":
        cases(*sys.argv[2:5])
    elif sys.argv[1] == "make":
        make(*sys.argv[2:6], sys.argv[6], sys.argv[7:])
    else:
        cmp(*sys.argv[2:7])
