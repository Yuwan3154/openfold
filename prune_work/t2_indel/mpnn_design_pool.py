"""Stage B: redesign every pool backbone with ProteinMPNN at BUILD time and merge into the pool npz.

Official settings (read from the repo's protein_mpnn_run.py defaults/README, not typed from memory): model v_48_020
(vanilla weights), sampling_temp 0.1 (default; 'suggested 0.1, 0.15, 0.2, 0.25, 0.3'), omit_AAs 'X' (default),
backbone_noise 0.0 (default), seed 37 (the repo's example scripts), 32 sequences per backbone (user 10-06).
Per chain: write each template's N/CA/C/O backbone as a PDB, parse_multiple_chains.py -> jsonl, ONE protein_mpnn_run.py
call (num_seq_per_target 32, batch_size 32), parse the .fa files, add design_* keys to the chain's pool npz.
Env: protpardelle (torch). Run: python mpnn_design_pool.py --pool-root pool --mpnn-dir ~/ProteinMPNN --work-dir w --shard 0 --num-shards 4
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

import indel_pool as ip

TEMP, SEED, NSEQ, MODEL = "0.1", 37, 32, "v_48_020"


def write_backbone_pdb(path, coords, atom_mask, aatype):
    names = ip.ATOM37.split()
    three = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS",
             "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
             "Y": "TYR", "V": "VAL"}
    full = np.zeros((len(aatype), 37, 3), np.float64)
    full[atom_mask] = coords
    lines, n = [], 0
    for i, a in enumerate(aatype, start=1):
        for nm in ("N", "CA", "C", "O"):
            j = names.index(nm)
            assert atom_mask[i - 1, j], "backbone atom missing"
            n += 1
            x = full[i - 1, j]
            lines.append(f"ATOM  {n:5d} {nm:<4s} {three[ip.AA_ORDER[int(a)]]:>3s} A{i:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}"
                         f"  1.00  0.00           {nm[0]:>2s}")
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")


def parse_fa(path, n_expected):
    recs = []
    cur = None
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith(">"):
            cur = [ln[1:], ""]
            recs.append(cur)
        elif ln:
            cur[1] += ln
    assert len(recs) == n_expected + 1, (path, len(recs))
    native, designs = recs[0], recs[1:]
    f = lambda h, k: float(re.search(rf"{k}=([-0-9.eE]+)", h).group(1))
    return (native[1], [d[1] for d in designs], np.array([f(d[0], "score") for d in designs]),
            np.array([f(d[0], "global_score") for d in designs]), np.array([f(d[0], "seq_recovery") for d in designs]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--mpnn-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chains", nargs="*", default=None)
    a = p.parse_args()
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.pool_root) for f in fs if f.endswith(".npz"))
    if a.chains:
        files = [f for f in files if os.path.basename(f)[:-4] in a.chains]
    for path in files[a.shard::a.num_shards]:
        chain = os.path.basename(path)[:-4]
        z = dict(np.load(path))
        if "design_aatype" in z:
            print(f"{chain}: already designed, skipped", flush=True)
            continue
        N = int(z["n_templates"])
        wd = os.path.join(a.work_dir, chain)
        os.makedirs(os.path.join(wd, "pdb"), exist_ok=True)
        for i in range(N):
            t = ip.read_template(path, i)
            write_backbone_pdb(os.path.join(wd, "pdb", f"t{i:03d}.pdb"), t["coords"], t["atom_mask"], t["aatype"])
        jsonl = os.path.join(wd, "parsed.jsonl")
        subprocess.run([sys.executable, os.path.join(a.mpnn_dir, "helper_scripts", "parse_multiple_chains.py"),
                        "--input_path", os.path.join(wd, "pdb"), "--output_path", jsonl], check=True)
        subprocess.run([sys.executable, os.path.join(a.mpnn_dir, "protein_mpnn_run.py"), "--jsonl_path", jsonl,
                        "--out_folder", wd, "--num_seq_per_target", str(NSEQ), "--sampling_temp", TEMP, "--seed", str(SEED),
                        "--batch_size", str(NSEQ), "--model_name", MODEL], check=True)
        R = int(z["res_offsets"][-1])
        design = np.zeros((NSEQ, R), np.int8)
        score, gscore, rec = np.zeros((N, NSEQ)), np.zeros((N, NSEQ)), np.zeros((N, NSEQ))
        for i in range(N):
            r0, r1 = int(z["res_offsets"][i]), int(z["res_offsets"][i + 1])
            nat, seqs, s, g, rc = parse_fa(os.path.join(wd, "seqs", f"t{i:03d}.fa"), NSEQ)
            assert nat == "".join(ip.AA_ORDER[int(x)] for x in z["aatype"][r0:r1]), f"{chain} t{i}: MPNN native != template sequence"
            for j, sq in enumerate(seqs):
                assert len(sq) == r1 - r0 and "X" not in sq
                design[j, r0:r1] = [ip.AA_ORDER.index(c) for c in sq]
            score[i], gscore[i], rec[i] = s, g, rc
        z.update(design_aatype=design, design_score=score.astype(np.float32), design_global_score=gscore.astype(np.float32),
                 design_recovery=rec.astype(np.float32),
                 design_meta_json=np.array(json.dumps(dict(model=MODEL, sampling_temp=float(TEMP), seed=SEED, n_seq=NSEQ,
                                                           omit_AAs="X", backbone_noise=0.0, weights="vanilla"))))
        np.savez(path, **z)
        print(f"{chain}: designed {N} backbones x {NSEQ} sequences", flush=True)


if __name__ == "__main__":
    main()
