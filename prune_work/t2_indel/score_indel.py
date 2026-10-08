"""Sequence-independent TM-score of every generated template vs its chain's native (user metric, Q4).

USalign default mode (NO -TMscore flag), template = Structure_1, native = Structure_2 (native.pdb of the same
chain, all resolved residues). Headline `tm_native` = TM normalised by the NATIVE length (USalign's Structure_2
score); `tm_template` = normalised by the template's own length. Rows per (model, chain, item, rung):
  kind 'indel'   d<k> at every t* rung        kind 'control' c<k> (no-indel repack, same seed) at every rung
  kind 'edited'  the edited input d<k>.pdb itself (the t* -> 0 limit; no GPU)  [model column = 'none']
Resumable per (model, chain): an existing scores/<model>_<chain>.csv is skipped.
Run: python score_indel.py --inputs-dir <inputs> --out-root <gen out> --score-dir <dir> --models cc89 cc91 [--workers N]
     python score_indel.py --inputs-dir <inputs> --score-dir <dir> --edited-only [--chains ...]
"""
import argparse
import os
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np

from atomic_io import atomic_csv

USALIGN = os.path.expanduser("~/.local/bin/USalign")
COLS = ["model", "chain", "kind", "draw", "rewind", "L_native", "L_template", "tm_native", "tm_template",
        "rmsd", "n_aligned"]


def write_ca_pdb(path, ca):
    with open(path, "w") as f:
        for i, c in enumerate(ca, start=1):
            f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")


def ca_from_pdb(path):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in open(path) if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def unpack_ca(npz):
    mask = npz["atom_mask"]
    packed = npz["coords"]
    full = np.zeros((packed.shape[0], mask.size, 3), np.float32)
    full[:, mask.reshape(-1)] = packed
    return full.reshape(packed.shape[0], mask.shape[0], 37, 3)[:, :, 1]


def usalign(tpl_pdb, nat_pdb):
    out = subprocess.run([USALIGN, tpl_pdb, nat_pdb], capture_output=True, text=True, check=True).stdout
    tm1 = next(float(l.split()[1]) for l in out.splitlines() if l.startswith("TM-score=") and "Structure_1" in l)
    tm2 = next(float(l.split()[1]) for l in out.splitlines() if l.startswith("TM-score=") and "Structure_2" in l)
    al = next(l for l in out.splitlines() if l.startswith("Aligned length="))
    n_al = int(al.split("Aligned length=")[1].split(",")[0])
    rmsd = float(al.split("RMSD=")[1].split(",")[0])
    return tm1, tm2, rmsd, n_al


def score_one(task):
    """task = (model, chain, kind, draw, rewind, ca (n,3), nat_pdb, L_native)"""
    model, chain, kind, draw, rewind, ca, nat_pdb, L_nat = task
    with tempfile.TemporaryDirectory() as td:
        tpl = os.path.join(td, "t.pdb")
        write_ca_pdb(tpl, ca)
        tm1, tm2, rmsd, n_al = usalign(tpl, nat_pdb)
    return dict(model=model, chain=chain, kind=kind, draw=draw, rewind=rewind, L_native=L_nat,
                L_template=len(ca), tm_native=tm2, tm_template=tm1, rmsd=rmsd, n_aligned=n_al)


def tasks_for_chain(inputs_dir, out_root, key, models, edited_only):
    d = os.path.join(inputs_dir, key)
    nat_pdb = os.path.join(d, "native.pdb")
    L_nat = len(ca_from_pdb(nat_pdb))
    tasks = []
    n_draws = len([f for f in os.listdir(d) if f.startswith("d") and f.endswith(".pdb")])
    if edited_only:
        for k in range(n_draws):
            tasks.append(("none", key, "edited", k, 0, ca_from_pdb(os.path.join(d, f"d{k:02d}.pdb")), nat_pdb, L_nat))
        return tasks
    for m in models:
        for fn in sorted(f for f in os.listdir(os.path.join(out_root, m, key)) if f.endswith(".npz") and ".tmp" not in f):
            z = np.load(os.path.join(out_root, m, key, fn))
            kind = "indel" if fn.startswith("d") else "control"
            cas = unpack_ca(z)
            for r, ca in zip(z["rewind_steps"].tolist(), cas):
                tasks.append((m, key, kind, int(fn[1:3]), int(r), ca, nat_pdb, L_nat))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-root", default=None)
    p.add_argument("--score-dir", required=True)
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--chains", nargs="*", default=None)
    p.add_argument("--edited-only", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    assert a.edited_only or (a.out_root and a.models), "need --out-root and --models unless --edited-only"
    os.makedirs(a.score_dir, exist_ok=True)
    keys = a.chains if a.chains else sorted(os.listdir(a.inputs_dir))
    with Pool(a.workers) as pool:
        for key in keys:
            tag = "edited" if a.edited_only else "_".join(a.models)
            out = os.path.join(a.score_dir, f"{tag}_{key}.csv")
            if os.path.isfile(out):
                continue
            rows = pool.map(score_one, tasks_for_chain(a.inputs_dir, a.out_root, key, a.models, a.edited_only),
                            chunksize=8)
            atomic_csv(out, COLS, rows)
            print(f"{key}: {len(rows)} rows, median tm_native {np.median([r['tm_native'] for r in rows]):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
