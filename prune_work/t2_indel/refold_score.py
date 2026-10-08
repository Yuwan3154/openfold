"""Score refold predictions against their template backbones (CPU).

Per prediction: paired TM (USalign -TMscore 5, residue i <-> residue i, normalised by the template = Structure_2),
sequence-independent TM (default USalign mode), paired RMSD, mean pLDDT, pTM, backbone C-N bond median/broken fraction
(a corruption check on the predictor), plus the pool's ProteinMPNN score/recovery of that sequence (seq_kind 0 = the generation
sequence has no MPNN score). Env: any with numpy + pandas; USalign at ~/.local/bin/USalign.
Run: python refold_score.py --pool-root pool --in-dir refold_af2 --method af2 --out scores.csv
"""
import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

import indel_pool as ip

USALIGN = os.path.expanduser("~/.local/bin/USalign")


def write_ca(path, ca):
    with open(path, "w") as f:
        for i, c in enumerate(ca, start=1):
            f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")


def run(pred, tpl, extra=()):
    out = subprocess.run([USALIGN, pred, tpl, *extra], capture_output=True, text=True, check=True).stdout
    tm = next(float(l.split()[1]) for l in out.splitlines() if l.startswith("TM-score=") and "Structure_2" in l)
    rmsd = float(next(l for l in out.splitlines() if l.startswith("Aligned length=")).split("RMSD=")[1].split(",")[0])
    return tm, rmsd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--control", action="store_true", help="native-sequence control files: no pool lookup, seq_kind 0 only")
    a = p.parse_args()
    rows = []
    for f in sorted(f for f in glob.glob(os.path.join(a.in_dir, "*.npz")) if ".tmp" not in os.path.basename(f)):
        chain, ti = os.path.basename(f)[:-4].rsplit("_t", 1)
        temp = float(ti.split("_T")[1]) if "_T" in ti else np.nan   # lower-temperature test files: <chain>_t<i>_T<T>.npz
        ti = int(ti.split("_T")[0])
        z = np.load(f)
        assert a.control or np.isnan(temp), f"{f}: lower-temperature files hold only designs (no generation sequence), score them with --control"
        t = None if a.control else ip.read_template(ip.shard_path(a.pool_root, chain), ti)
        S = len(z["seqs"])
        with tempfile.TemporaryDirectory() as td:
            tpl = os.path.join(td, "t.pdb")
            write_ca(tpl, z["template_ca"])
            for s in range(S):
                pred = os.path.join(td, "p.pdb")
                write_ca(pred, z["pred_atom37"][s][:, 1].astype(np.float32))
                tm_p, rm_p = run(pred, tpl, ("-TMscore", "5"))
                tm_s, _ = run(pred, tpl)
                n, c = z["pred_atom37"][s][:, 0].astype(np.float32), z["pred_atom37"][s][:, 2].astype(np.float32)
                cn = np.linalg.norm(n[1:] - c[:-1], axis=1)
                rows.append(dict(method=a.method, chain=chain, i=ti, temp=temp, arm=("control" if a.control else t["arm"]), L=len(z["seqs"][0]), seq_kind=(s + 1 if np.isfinite(temp) else s),
                                 tm_paired=tm_p, rmsd_paired=rm_p, tm_seqind=tm_s, plddt=float(z["plddt"][s].mean()),
                                 ptm=float(z["ptm"][s]), cn_median=float(np.median(cn)), cn_broken=float((cn > 2.0).mean()),
                                 mpnn_score=float(t["design_score"][s - 1]) if (s > 0 and not a.control) else np.nan,
                                 recovery=float(t["design_recovery"][s - 1]) if (s > 0 and not a.control) else np.nan))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} predictions -> {a.out}")


if __name__ == "__main__":
    main()
