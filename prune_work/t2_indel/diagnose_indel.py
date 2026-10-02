"""Seam-geometry and secondary-structure diagnostics for generated indel templates (non-gating; user 10-02).

Per output item and t* rung, against the chain's NATIVE and the edit sidecar (plans.json: orig_idx, -1 = inserted):
  geometry  peptide-bond C(i)-N(i+1) and CA(i)-CA(i+1) at SEAM pairs (consecutive template residues where either is
            inserted, or both survive but are not consecutive in the native = a deletion seam) vs at BACKGROUND pairs;
            'out of band' = outside the native chain's own 1st-99th percentile of the same quantity (derived from the
            same chain, not typed in); plus non-local CA clashes (|i-j| >= 3, closer than the native's minimum).
  SSE       mdtraj DSSP (3-state, backbone only) of template vs native; over SURVIVORS Q3 and native->template
            transition counts, Q3 split by template-index distance to the nearest edit event (bins 0-2, 3-5, 6-10, 11+;
            descriptive bins), the DSSP composition of the INSERTED residues, and template helix/strand fractions.
Controls (no-indel outputs, kind 'control') use the identity mapping and have no seams.
Env: any with numpy + mdtraj (+ t2_deepdive/pool_sse.py importable). Run:
  python diagnose_indel.py --inputs-dir <inputs> --out-root <sweep> --models cc89 cc91 --kinds indel control --diag-dir <dir> --workers 32
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "t2_deepdive"))
from pool_sse import backbone_top, ss_of  # noqa: E402

N, CA, C, O = 0, 1, 2, 4                  # atom37 slots
BINS = [(0, 2), (3, 5), (6, 10), (11, 10**9)]
LAB = "HEC"


def native_backbone(pdb):
    rows = {}
    for ln in open(pdb):
        nm = ln[12:16].strip()
        if ln.startswith("ATOM") and nm in ("N", "CA", "C", "O"):
            rows.setdefault(int(ln[22:26]), {})[nm] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return np.array([[r["N"], r["CA"], r["C"], r["O"]] for _, r in sorted(rows.items())])


def dssp(bb):
    top = backbone_top(np.zeros(len(bb), int), np.zeros(len(bb), bool))
    return ss_of(top, bb.reshape(-1, 3).astype(np.float32))


def pair_geoms(bb):
    cn = np.linalg.norm(bb[1:, 0] - bb[:-1, 2], axis=1)
    caca = np.linalg.norm(bb[1:, 1] - bb[:-1, 1], axis=1)
    return cn, caca


def nonlocal_min(ca):
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    iu = np.triu_indices(len(ca), k=3)
    return d, iu


def unpack_bb(z):
    mask = z["atom_mask"]
    full = np.zeros((z["coords"].shape[0], mask.size, 3), np.float32)
    full[:, mask.reshape(-1)] = z["coords"]
    full = full.reshape(z["coords"].shape[0], mask.shape[0], 37, 3)
    return full[:, :, [N, CA, C, O]]


def diag_item(bb, orig, ss_nat, band, nat_min_nl):
    """bb (L',4,3); orig (L',) native index or -1."""
    Lp = len(orig)
    cn, caca = pair_geoms(bb)
    ins = orig < 0
    seam = ins[:-1] | ins[1:] | ((orig[1:] - orig[:-1]) != 1)
    rec = {}
    for tag, sel in (("seam", seam), ("bg", ~seam)):
        rec[f"n_{tag}"] = int(sel.sum())
        for nm, v, (lo, hi) in (("cn", cn, band["cn"]), ("caca", caca, band["caca"])):
            rec[f"{tag}_{nm}_out"] = int(((v[sel] < lo) | (v[sel] > hi)).sum())
            rec[f"{tag}_{nm}_mean"] = float(v[sel].mean()) if sel.any() else np.nan
    d, iu = nonlocal_min(bb[:, 1])
    rec["n_clash"] = int((d[iu] < nat_min_nl).sum())
    ss = dssp(bb)
    surv = np.flatnonzero(~ins)
    nat_ss = ss_nat[orig[surv]]
    rec["n_surv"] = len(surv)
    rec["q3_surv"] = float((ss[surv] == nat_ss).mean())
    for s in LAB:
        for t in LAB:
            rec[f"n_{s}to{t}"] = int(((nat_ss == s) & (ss[surv] == t)).sum())
    ev = np.flatnonzero(ins)
    seam_pos = np.flatnonzero((~ins[:-1]) & (~ins[1:]) & ((orig[1:] - orig[:-1]) != 1))
    ev_pos = np.concatenate([ev, seam_pos, seam_pos + 1]) if (len(ev) + len(seam_pos)) else np.array([], int)
    dist = np.min(np.abs(surv[:, None] - ev_pos[None]), axis=1) if len(ev_pos) else np.full(len(surv), 10**9)
    for lo, hi in BINS:
        sel = (dist >= lo) & (dist <= hi)
        rec[f"q3_d{lo}"] = float((ss[surv][sel] == nat_ss[sel]).mean()) if sel.any() else np.nan
        rec[f"nq3_d{lo}"] = int(sel.sum())
    for s in LAB:
        rec[f"ins_{s}"] = int((ss[ins] == s).sum())
    rec["frac_H"], rec["frac_E"] = float((ss == "H").mean()), float((ss == "E").mean())
    return rec


def do_chain(job):
    inputs_dir, out_root, model, key, kinds, rewinds = job
    d = os.path.join(inputs_dir, key)
    plans = json.load(open(os.path.join(d, "plans.json")))
    nat = native_backbone(os.path.join(d, "native.pdb"))
    ss_nat = dssp(nat)
    cn, caca = pair_geoms(nat)
    band = {"cn": tuple(np.percentile(cn, [1, 99])), "caca": tuple(np.percentile(caca, [1, 99]))}
    dn, iu = nonlocal_min(nat[:, 1])
    nat_min_nl = float(dn[iu].min())
    rows = []
    for fn in sorted(os.listdir(os.path.join(out_root, model, key))):
        kind = "indel" if fn.startswith("d") else "control"
        if kind not in kinds:
            continue
        k = int(fn[1:3])
        z = np.load(os.path.join(out_root, model, key, fn))
        bbs = unpack_bb(z)
        orig = np.array(plans["plans"][k]["orig_idx"]) if kind == "indel" else np.arange(plans["L"])
        for r, bb in zip(z["rewind_steps"].tolist(), bbs):
            rec = diag_item(bb, orig, ss_nat, band, nat_min_nl)
            rec.update(model=model, chain=key, kind=kind, draw=k, rewind=int(r))
            rows.append(rec)
    return key, model, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--kinds", nargs="+", default=["indel", "control"])
    p.add_argument("--diag-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chains", nargs="*", default=None)
    a = p.parse_args()
    os.makedirs(a.diag_dir, exist_ok=True)
    keys = a.chains if a.chains else sorted(os.listdir(a.inputs_dir))
    jobs = [(a.inputs_dir, a.out_root, m, k, a.kinds, None) for m in a.models for k in keys
            if not os.path.isfile(os.path.join(a.diag_dir, f"{m}_{k}.csv"))]
    with Pool(a.workers) as pool:
        for key, model, rows in pool.imap_unordered(do_chain, jobs):
            with open(os.path.join(a.diag_dir, f"{model}_{key}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"{model} {key}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
