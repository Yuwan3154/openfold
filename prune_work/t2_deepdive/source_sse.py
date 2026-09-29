"""SSE agreement with the native for the NATURAL and PROMOTED template sources (synthetic: pool_sse.py).

natural : each delivered hit's full template chain (from the mmCIF mirror, author chain id as OpenFold's
          mmcif_parsing uses) is aligned to the query's native with USalign in its default,
          sequence-INDEPENDENT mode; DSSP 3-state labels are compared over the aligned residue pairs.
          TM is normalized by the native (USalign's chain-2 score).
promoted: Run C v2 T4 records are on the query frame; qmap maps each native residue to its query
          position, so pairs are fixed by index (no alignment). The promoted structure exists only for a
          <=256-residue crop, so the native is DSSP'd on the SAME crop (a strand whose partner lies
          outside the crop cannot be E in either). TM = USalign -TMscore 5 on the paired crop.
Env: proteinebm (mdtraj, Biopython); USalign at ~/.local/bin/USalign.

Run: python source_sse.py --hits natural_hits.json --out-dir out --workers 16
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import zlib
from multiprocessing import Pool

import mdtraj as md
import numpy as np
from Bio.PDB import MMCIFParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_sse import IDX, RESTYPES3, backbone_top, native_xyz37, pack, ss_of  # noqa: E402

HOME = os.path.expanduser("~")
USALIGN = f"{HOME}/.local/bin/USalign"
MMCIF = f"{HOME}/data/pdb_mmcif/mmcif_files"
MANIFEST = f"{HOME}/pp1c_work/natives_train/manifest.csv"
TEMPLATES = f"{HOME}/pp1c_work/templates_band"
QMAP = f"{HOME}/pp1c_work/qmap_all_v2.npz"
T4_POOL = f"{HOME}/runs/runC_v2/t4_pool"
LABELS = "HEC"


def dssp_file(pdb):
    """(labels, resSeq list) for residues that carry a CA, in file order (USalign's residue order)."""
    t = md.load_pdb(pdb)
    ss = md.compute_dssp(t, simplified=True)[0]
    keep = [(r.index, r.resSeq) for r in t.topology.residues if r.is_protein and any(a.name == "CA" for a in r.atoms)]
    return np.array([ss[i] for i, _ in keep]), [n for _, n in keep]


def write_chain_pdb(cif, chain_id, out):
    """Protein residues of one author chain; MSE written as MET (SE -> SD) so mdtraj/USalign treat it as protein."""
    s = MMCIFParser(QUIET=True).get_structure("t", cif)
    if chain_id not in s[0]:
        return -1
    ch = s[0][chain_id]
    lines, k, n = [], 0, 0
    for r in ch:
        name = r.get_resname()
        if name == "MSE":
            name = "MET"
        if name not in RESTYPES3[:20] or "CA" not in r:
            continue
        n += 1
        for a in r:
            an = "SD" if (a.get_name() == "SE" and r.get_resname() == "MSE") else a.get_name()
            if a.element == "H":
                continue
            k += 1
            x, y, z = a.get_coord()
            lines.append(f"ATOM  {k:5d} {an:<4s} {name:3s} A{n:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                         f"          {a.element:>2s}")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")
    return n


def usalign_pairs(tpl, nat, extra=()):
    out = subprocess.run([USALIGN, tpl, nat, *extra], capture_output=True, text=True, check=True).stdout
    tm2 = next(float(l.split()[1]) for l in out.splitlines() if l.startswith("TM-score=") and "Structure_2" in l)
    lines = [l for l in out.splitlines() if l.strip()]
    i = next(j for j, l in enumerate(lines) if l.startswith('(":" denotes'))
    a1, mk, a2 = lines[i + 1], lines[i + 2], lines[i + 3]
    pairs, i1, i2 = [], 0, 0
    for c1, m, c2 in zip(a1, mk, a2):
        if c1 != "-" and c2 != "-":
            pairs.append((i1, i2, m == ":"))
        i1 += c1 != "-"
        i2 += c2 != "-"
    return tm2, pairs, i1, i2


def compare(ss_nat, ss_tpl, pairs):
    rec = {}
    for tag, sub in (("ali", pairs), ("close", [p for p in pairs if p[2]])):
        n = len(sub)
        rec[f"n_{tag}"] = n
        rec[f"q3_{tag}"] = float(np.mean([ss_nat[j] == ss_tpl[i] for i, j, _ in sub])) if n else np.nan
    for s in LABELS:
        for t in LABELS:
            rec[f"n_{s}to{t}"] = int(sum(ss_nat[j] == s and ss_tpl[i] == t for i, j, _ in pairs))
    return rec


def do_natural(job):
    chain, hits, nat_pdb = job
    ss_nat, _ = dssp_file(nat_pdb)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for rank, h in enumerate(hits):
            if not h:  # the featurizer's empty placeholder template (chain with no usable hit)
                rows.append(dict(source="natural", chain=chain, hit=h, rank=rank, status="empty_placeholder"))
                continue
            pdb_id, ch = h.split("_", 1)
            tpl = os.path.join(td, f"{h}.pdb")
            n_res = write_chain_pdb(os.path.join(MMCIF, f"{pdb_id.lower()}.cif"), ch, tpl)
            if n_res < 3:  # -1 = author chain id absent from the mmCIF
                rows.append(dict(source="natural", chain=chain, hit=h, rank=rank,
                                 status="chain_absent" if n_res < 0 else "too_short"))
                continue
            ss_tpl, _ = dssp_file(tpl)
            tm, pairs, n1, n2 = usalign_pairs(tpl, nat_pdb)
            assert (n1, n2) == (len(ss_tpl), len(ss_nat)), (h, chain, n1, len(ss_tpl), n2, len(ss_nat))
            rec = dict(source="natural", chain=chain, hit=h, rank=rank, status="ok", tm=tm, L_native=len(ss_nat),
                       L_template=len(ss_tpl), coverage=len(pairs) / len(ss_nat),
                       fH_native=float(np.mean(ss_nat == "H")), fE_native=float(np.mean(ss_nat == "E")))
            rec.update(compare(ss_nat, ss_tpl, pairs))
            rows.append(rec)
    return rows


def do_promoted(job):
    chain, recs, nat_pdb, qmap = job
    d = np.load(os.path.join(TEMPLATES, f"shard{zlib.crc32(chain.encode()) % 1000:04d}", f"{chain}.npz"))
    resi, aat_n, mask_n = d["residue_index"], d["aatype"], d["atom_mask"]
    nat37, _ = native_xyz37(nat_pdb, resi, aat_n)
    q_to_row = {int(q): j for j, q in enumerate(qmap)}
    rows = []
    for r in recs:
        p = np.load(os.path.join(T4_POOL, r["_rank"], r["npz"]))
        am, aat_p, ri = p["atom_mask"], p["aatype"], p["residue_index"]
        x = np.zeros(am.shape + (3,), np.float32)
        x[am] = p["coords"]
        real = am[:, 1]  # make_fixed_size pads with atom-less rows at residue_index 0: CA mask, not residue_index
        pr, nr = [], []
        for i in np.flatnonzero(real):
            j = q_to_row.get(int(ri[i]))
            if j is not None and mask_n[j, [0, 1, 2, 4]].all():
                pr.append(i)
                nr.append(j)
        pr, nr = np.array(pr), np.array(nr)
        assert (aat_p[pr] == aat_n[nr]).all(), (chain, r["npz"], "promoted/native residue types disagree")
        has_cb = mask_n[nr, 3] & am[pr, 3] & ~np.isnan(nat37[nr, 3, 0])
        top = backbone_top(aat_n[nr], has_cb)
        ss_n = ss_of(top, pack(nat37[nr], has_cb))
        ss_p = ss_of(top, pack(x[pr], has_cb))
        with tempfile.TemporaryDirectory() as td:
            fa, fb = os.path.join(td, "p.pdb"), os.path.join(td, "n.pdb")
            for f, xyz in ((fa, x[pr]), (fb, nat37[nr])):
                t = md.Trajectory(pack(xyz, has_cb)[None] / 10.0, top)
                t.save_pdb(f)
            tm, _, _, _ = usalign_pairs(fa, fb, ("-TMscore", "5"))
        pairs = [(k, k, True) for k in range(len(pr))]
        rec = dict(source="promoted", chain=chain, hit=r["npz"], rank=int(r["epoch"]), status="ok", tm=tm,
                   L_native=len(resi), L_template=int(real.sum()), coverage=len(pr) / len(resi),
                   fH_native=float(np.mean(ss_n == "H")), fE_native=float(np.mean(ss_n == "E")),
                   tm_pred=r["tm_pred"])
        rec.update(compare(ss_n, ss_p, pairs))
        rows.append(rec)
    return rows


def selftest(hits, man):
    """A native aligned to itself must give TM 1, full coverage, Q3 1."""
    c = hits[0]["chain"]
    nat = man[c]["pdb"]
    ss, _ = dssp_file(nat)
    tm, pairs, _, _ = usalign_pairs(nat, nat)
    r = compare(ss, ss, pairs)
    assert abs(tm - 1) < 1e-3 and len(pairs) == len(ss) and r["q3_ali"] == 1.0, (tm, len(pairs), len(ss), r)
    print(f"selftest {c}: TM {tm} pairs {len(pairs)}/{len(ss)} q3 {r['q3_ali']}  PASS", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hits", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    hits = json.load(open(a.hits))
    man = {r["chain"]: r for r in csv.DictReader(open(MANIFEST))}
    selftest(hits, man)
    chains = {h["chain"] for h in hits}
    missing = sorted(c for c in chains if c not in man)
    print(f"{len(chains)} chains; not in natives_train manifest: {len(missing)} {missing[:10]}", flush=True)

    zq = np.load(QMAP, allow_pickle=False)
    offs = np.concatenate([[0], np.cumsum(zq["qmap_len"])])
    qrow = {str(c): j for j, c in enumerate(zq["chains"])}
    t4 = {c: [] for c in chains}
    for rk in sorted(os.listdir(T4_POOL)):
        with open(os.path.join(T4_POOL, rk, "index.jsonl")) as fh:
            for line in fh:
                r = json.loads(line)
                if r["chain"] in t4:
                    r["_rank"] = rk
                    t4[r["chain"]].append(r)
    print(f"promoted records for these chains: {sum(len(v) for v in t4.values())} "
          f"({sum(1 for v in t4.values() if v)} chains have any)", flush=True)

    nat_jobs = [(h["chain"], h["hits"], man[h["chain"]]["pdb"]) for h in hits if h["hits"] and h["chain"] in man]
    amb = {c for c in t4 if c in qrow and bool(zq["ambiguous"][qrow[c]])}
    print(f"promoted: {len(amb)} chains with an ambiguous qmap excluded: {sorted(amb)[:10]}", flush=True)
    pro_jobs = [(c, v, man[c]["pdb"], zq["qmap"][offs[qrow[c]]:offs[qrow[c] + 1]])
                for c, v in t4.items() if v and c in man and c in qrow and c not in amb]
    os.makedirs(a.out_dir, exist_ok=True)
    for name, fn, jobs in (("natural", do_natural, nat_jobs), ("promoted", do_promoted, pro_jobs)):
        rows = []
        with Pool(a.workers) as pool:
            for rr in pool.imap_unordered(fn, jobs):
                rows += rr
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(a.out_dir, f"source_sse_{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(jobs)} chains -> {len(rows)} rows; status "
              f"{ {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}} }", flush=True)


if __name__ == "__main__":
    main()
