"""sse_drift.py's decomposition over the PRODUCTION synthetic-template pool (npz tree + index).

Per chain: the native comes from natives_train/<shard>/<chain>.pdb restricted to the residues the npz
holds (matched by residue number), the templates from the npz's in-band rows. Both go through the same
backbone(+CB) topology, so DSSP sees identical atom sets. The generating model is recovered with
generate_templates.py's own rule (resid_span > --span-cutoff -> cc91, else cc89).
Writes one csv.gz per worker; --validate instead checks the backbone-only DSSP path against the
full-atom one on the bake-off PDBs (same result required).

Run: <proteinebm env>/bin/python pool_sse.py --index index_band.npz --templates-root templates_band \
     --manifest natives_train/manifest.csv --out-dir out --workers 32
"""
import argparse
import csv
import gzip
import os
import sys
import zlib
from multiprocessing import Pool

import mdtraj as md
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sse_drift import LONG_SEP, contacts, element_pairs, elements, kabsch_rmsd, superpose  # noqa: E402

ATOM37 = ["N", "CA", "C", "CB", "O"]          # first five atom37 slots: N CA C CB O
IDX = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4}
RESTYPES3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
             "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK"]


def backbone_top(aatype, has_cb):
    top = md.Topology()
    ch = top.add_chain()
    for i, a in enumerate(aatype):
        r = top.add_residue(RESTYPES3[min(int(a), 20)], ch, resSeq=i + 1)
        for n in ("N", "CA", "C", "O") + (("CB",) if has_cb[i] else ()):
            top.add_atom(n, md.element.get_by_symbol(n[0]), r)
    return top


def pack(xyz37, has_cb):
    """(L,37,3) A -> flat backbone(+CB) array in backbone_top's atom order."""
    rows = []
    for i in range(xyz37.shape[0]):
        rows += [xyz37[i, 0], xyz37[i, 1], xyz37[i, 2], xyz37[i, 4]]
        if has_cb[i]:
            rows.append(xyz37[i, 3])
    return np.asarray(rows, np.float32)


def ss_of(top, flat):
    t = md.Trajectory(flat[None] / 10.0, top)
    return np.array(md.compute_dssp(t, simplified=True)[0])


def metrics(ss_n, ca_n, cb_n, cm_n, sep, els, ep_n, ss, ca, cb):
    cm, _ = contacts(cb)
    glob = superpose(ca, ca_n)
    rec = {"q3": float(np.mean(ss == ss_n))}
    for s in "HEC":
        rec[f"f{s}"] = float(np.mean(ss == s))
        for t in "HEC":
            rec[f"n_{s}to{t}"] = int(((ss_n == s) & (ss == t)).sum())
    rec["q_contacts"] = float((cm & cm_n).sum() / max(cm_n.sum(), 1))
    lr = cm_n & (sep >= LONG_SEP)
    rec["q_contacts_long"] = float((cm & lr).sum() / lr.sum()) if lr.any() else np.nan
    ep = element_pairs(cm, els)
    rec["epairs_kept"] = float(len(ep & ep_n) / max(len(ep_n), 1))
    rec["epairs_new"] = len(ep - ep_n)
    wl = wg = wn = 0.0
    ret = []
    for lab, i0, i1 in els:
        m = i1 - i0
        ret.append(float(np.mean(ss[i0:i1] == lab)))
        if m >= 3:  # Kabsch needs 3 points
            wl += m * kabsch_rmsd(ca[i0:i1], ca_n[i0:i1])[0] ** 2
            wg += m * float(((glob[i0:i1] - ca_n[i0:i1]) ** 2).sum(1).mean())
            wn += m
    rec["elem_local_rmsd"] = float(np.sqrt(wl / wn)) if wn else np.nan
    rec["elem_global_rmsd"] = float(np.sqrt(wg / wn)) if wn else np.nan
    rec["elem_retention"] = float(np.mean(ret)) if ret else np.nan
    return rec


def native_xyz37(pdb, residue_index, aatype):
    """Native N/CA/C/CB/O by residue number; returns (xyz, n_oxt_as_o).

    A C-terminal residue deposited with OXT but no O has its O slot set in the generator's mask (1vol_A),
    so that residue's template O is compared against the native's OXT: the same carboxyl oxygen position.
    """
    t = md.load_pdb(pdb)
    by_num = {r.resSeq: r for r in t.topology.residues if r.is_protein}
    xyz = np.full((len(residue_index), 37, 3), np.nan, np.float32)
    n_oxt = 0
    for i, n in enumerate(residue_index):
        r = by_num[int(n)]
        assert RESTYPES3[min(int(aatype[i]), 20)] == r.name or r.name not in RESTYPES3, (pdb, n, r.name)
        names = {a.name: a for a in r.atoms}
        for nm in IDX:
            if nm in names:
                xyz[i, IDX[nm]] = t.xyz[0, names[nm].index] * 10.0
        if "O" not in names and "OXT" in names:
            xyz[i, IDX["O"]] = t.xyz[0, names["OXT"].index] * 10.0
            n_oxt += 1
    return xyz, n_oxt


def precheck_chain(job):
    """Everything do_chain asserts about native-vs-mask agreement, counted instead of asserted."""
    chain, npz, pdb, *_ = job
    d = np.load(npz, allow_pickle=False)
    mask, aat, resi = d["atom_mask"], d["aatype"], d["residue_index"]
    t = md.load_pdb(pdb)
    nums = {r.resSeq for r in t.topology.residues if r.is_protein}
    absent = [int(n) for n in resi if int(n) not in nums]
    if absent:
        return dict(chain=chain, kind="residue_absent", n=len(absent), oxt=0)
    nat, n_oxt = native_xyz37(pdb, resi, aat)
    bb = [0, 1, 2, 4]
    miss = int(np.isnan(nat[:, bb][mask[:, bb]]).any(-1).sum())
    return dict(chain=chain, kind="backbone_missing" if miss else "ok", n=miss, oxt=n_oxt)


def do_chain(job):
    chain, npz, pdb, span, cutoff, tms, rewinds = job
    d = np.load(npz, allow_pickle=False)
    mask, aat, resi = d["atom_mask"], d["aatype"], d["residue_index"]
    assert d["coords"].shape[0] == len(tms), (chain, d["coords"].shape, len(tms))
    has_cb = mask[:, 3].copy()
    nat, _ = native_xyz37(pdb, resi, aat)
    assert not np.isnan(nat[:, [0, 1, 2, 4]][mask[:, [0, 1, 2, 4]]]).any(), chain
    top = backbone_top(aat, has_cb)
    ss_n = ss_of(top, pack(nat, has_cb))
    ca_n = nat[:, 1]
    cb_n = np.where(has_cb[:, None], nat[:, 3], ca_n)
    cm_n, sep = contacts(cb_n)
    els = elements(ss_n)
    ep_n = element_pairs(cm_n, els)
    model = "cc91" if span > cutoff else "cc89"
    rows = []
    for k in range(len(tms)):
        x = np.zeros((len(aat), 37, 3), np.float32)
        x[mask] = d["coords"][k]
        ss = ss_of(top, pack(x, has_cb))
        ca = x[:, 1]
        cb = np.where(has_cb[:, None], x[:, 3], ca)
        rec = dict(chain=chain, model=model, rewind=int(rewinds[k]), tm=float(tms[k]), L=len(aat),
                   fH_native=float(np.mean(ss_n == "H")), fE_native=float(np.mean(ss_n == "E")),
                   n_elements=len(els))
        rec.update(metrics(ss_n, ca_n, cb_n, cm_n, sep, els, ep_n, ss, ca, cb))
        rows.append(rec)
    return rows


def validate(tm_csv, natives):
    """Backbone-only topology must reproduce the full-atom DSSP call on the same coordinates."""
    import pandas as pd
    df = pd.read_csv(tm_csv)
    agree, n = [], 0
    for pdb in list(df.pdb_file) + [os.path.join(natives, f) for f in os.listdir(natives) if f.endswith(".pdb")]:
        t = md.load_pdb(pdb)
        res = [r for r in t.topology.residues if r.is_protein]
        full = np.array(md.compute_dssp(t, simplified=True)[0])[[r.index for r in res]]
        xyz = np.full((len(res), 37, 3), np.nan, np.float32)
        for i, r in enumerate(res):
            for a in r.atoms:
                if a.name in IDX:
                    xyz[i, IDX[a.name]] = t.xyz[0, a.index] * 10.0
        has_cb = ~np.isnan(xyz[:, 3, 0])
        aat = [RESTYPES3.index(r.name) if r.name in RESTYPES3 else 20 for r in res]
        bb = ss_of(backbone_top(aat, has_cb), pack(xyz, has_cb))
        agree.append(float(np.mean(bb == full)))
        n += 1
    agree = np.array(agree)
    print(f"validate: {n} structures, backbone-vs-full DSSP agreement min {agree.min():.4f} "
          f"mean {agree.mean():.4f}; {int((agree < 1).sum())} of {n} not identical")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index")
    p.add_argument("--templates-root")
    p.add_argument("--manifest")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--span-cutoff", type=int, default=484)  # generate_templates.py's default, the run used it
    p.add_argument("--limit", type=int, default=0, help="first N chains only (smoke test)")
    p.add_argument("--validate-tm-csv")
    p.add_argument("--validate-natives")
    p.add_argument("--precheck", action="store_true", help="count native-vs-mask mismatches over every chain")
    a = p.parse_args()
    if a.validate_tm_csv:
        validate(a.validate_tm_csv, a.validate_natives)
        return

    z = np.load(a.index, allow_pickle=False)
    # an NpzFile re-reads an array on EVERY z[key] access, so pull each out once
    chains = [str(c) for c in z["chains"]]
    tm_all, rw_all, slot = z["tm"], z["rewind"], z["slot"]
    lo, hi = float(z["min_tm"]), float(z["max_tm"])
    band = (tm_all > lo) & (tm_all < hi)
    man = {}
    with open(a.manifest) as fh:
        for r in csv.DictReader(fh):
            man[r["chain"]] = r
    root = a.templates_root
    jobs = []
    for i, c in enumerate(chains):
        keep = np.flatnonzero(band[i])
        if len(keep) == 0:
            continue
        assert (slot[i, keep] == np.arange(len(keep))).all(), c  # npz rows = in-band rungs in order
        m = man[c]
        jobs.append((c, os.path.join(root, f"shard{zlib.crc32(c.encode()) % 1000:04d}", f"{c}.npz"), m["pdb"],
                     int(m["resid_span"]), a.span_cutoff, tm_all[i, keep], rw_all[i, keep]))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} chains, {sum(len(j[5]) for j in jobs)} templates", flush=True)
    if a.precheck:
        from collections import Counter
        with Pool(a.workers) as pool:
            res = list(pool.imap_unordered(precheck_chain, jobs, chunksize=64))
        kinds = Counter(r["kind"] for r in res)
        print(f"precheck {len(res)} of {len(jobs)} chains: {dict(kinds)}; chains using OXT as O: "
              f"{sum(r['oxt'] > 0 for r in res)} ({sum(r['oxt'] for r in res)} residues)")
        for r in res:
            if r["kind"] != "ok":
                print("  ", r)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    fields = None
    outs = {}
    done = 0
    with Pool(a.workers) as pool:
        for rows in pool.imap_unordered(do_chain, jobs, chunksize=8):
            if fields is None:
                fields = list(rows[0].keys())
            w = done % a.workers
            if w not in outs:
                fh = gzip.open(os.path.join(a.out_dir, f"pool_sse_{w:02d}.csv.gz"), "wt")
                wr = csv.DictWriter(fh, fieldnames=fields)
                wr.writeheader()
                outs[w] = (fh, wr)
            outs[w][1].writerows(rows)
            done += 1
            if done % 2000 == 0:
                print(f"{done}/{len(jobs)} chains", flush=True)
    for fh, _ in outs.values():
        fh.close()
    print(f"DONE {done} chains", flush=True)


if __name__ == "__main__":
    main()
