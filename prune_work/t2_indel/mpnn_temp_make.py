"""Lower-temperature ProteinMPNN test (user 10-07: 'try lower MPNN temperature first'). CPU/GPU, env protpardelle.

For each selected pool template and each temperature T: run protein_mpnn_run.py (v_48_020, seed 37, 32 samples; ONLY T differs
from the pool's settings), parse the .fa, drop duplicate sequences, keep the first --n-fold unique designs. Writes <out>.json:
{"<chain>_t<i>_T<T>": {"chain","i","T","seqs":[...], "n_unique": k, "mpnn_score": [...]}}. Env: protpardelle.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import indel_pool as ip
from mpnn_design_pool import write_backbone_pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--select", nargs="+", required=True)
    p.add_argument("--temps", nargs="+", required=True)
    p.add_argument("--mpnn-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--n-fold", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {}
    for sel in a.select:
        chain, i = sel.split(":")
        i = int(i)
        t = ip.read_template(ip.shard_path(a.pool_root, chain), i)
        for T in a.temps:
            wd = os.path.join(a.work_dir, f"{chain}_t{i:03d}_T{T}")
            os.makedirs(os.path.join(wd, "pdb"), exist_ok=True)
            write_backbone_pdb(os.path.join(wd, "pdb", "x.pdb"), t["coords"], t["atom_mask"], t["aatype"])
            subprocess.run([sys.executable, os.path.join(a.mpnn_dir, "helper_scripts", "parse_multiple_chains.py"),
                            "--input_path", os.path.join(wd, "pdb"), "--output_path", os.path.join(wd, "parsed.jsonl")], check=True)
            subprocess.run([sys.executable, os.path.join(a.mpnn_dir, "protein_mpnn_run.py"), "--jsonl_path", os.path.join(wd, "parsed.jsonl"),
                            "--out_folder", wd, "--num_seq_per_target", "32", "--sampling_temp", T, "--seed", "37",
                            "--batch_size", "32", "--model_name", "v_48_020"], check=True)
            recs, cur = [], None
            for ln in open(os.path.join(wd, "seqs", "x.fa")):
                ln = ln.strip()
                if ln.startswith(">"):
                    cur = [ln, ""]
                    recs.append(cur)
                elif ln:
                    cur[1] += ln
            designs = recs[1:]
            seen, seqs, sc = set(), [], []
            for h, s in designs:
                if s in seen:
                    continue
                seen.add(s)
                seqs.append(s)
                sc.append(float(re.search(r"(?<![_a-z])score=([-0-9.eE]+)", h).group(1)))
            res[f"{chain}_t{i:03d}_T{T}"] = dict(chain=chain, i=i, T=float(T), n_unique=len(seqs), seqs=seqs[: a.n_fold],
                                                  mpnn_score=sc[: a.n_fold])
            print(f"{chain} t{i} T={T}: {len(seqs)} unique of 32; folding the first {min(a.n_fold, len(seqs))}", flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
