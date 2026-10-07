"""Refold test, AF3 (google-deepmind/alphafold3 repo, env ~/alphafold3_sc/.venv) -- two subcommands, same refold npz format.

make:    write one AF3 input JSON per (template, sequence): single protein chain, EMPTY MSA (unpairedMsa/pairedMsa "") and NO
         templates, so the data pipeline is skipped (--norun_data_pipeline). Names are <chain>_t<i>_s<k> (k = 0 generation seq).
collect: read <out>/<name>/<name>_model.cif + summary_confidences.json -> <out2>/<chain>_t<i>.npz with pred_atom37 (only N,CA,C
         filled), plddt (0-1, mean of the residue's atom B-factors / 100), ptm, template_ca, seqs. Env: protpardelle (biopython).
Settings (CHOSEN, ledger): num_diffusion_samples 1, num_recycles = repo default (10), model seed 1,
--flash_attention_implementation=xla (V100 / compute 7.0).
"""
import argparse
import json
import os

import numpy as np

import indel_pool as ip


def make(a):
    os.makedirs(a.json_dir, exist_ok=True)
    n = 0
    for sel in a.select:
        chain, i = sel.split(":")
        i = int(i)
        t = ip.read_template(ip.shard_path(a.pool_root, chain), i)
        seqs = ["".join(ip.AA_ORDER[int(x)] for x in t["aatype"])]
        seqs += ["".join(ip.AA_ORDER[int(x)] for x in row) for row in t["design_aatype"][: a.max_seqs]]
        for k, s in enumerate(seqs):
            name = f"{chain}_t{i:03d}_s{k:02d}"
            js = {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 4,
                  "sequences": [{"protein": {"id": "A", "sequence": s, "unpairedMsa": "", "pairedMsa": "", "templates": []}}]}
            json.dump(js, open(os.path.join(a.json_dir, name + ".json"), "w"))
            n += 1
    print(f"wrote {n} input JSONs to {a.json_dir}")


def collect(a):
    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    os.makedirs(a.out2, exist_ok=True)
    for sel in a.select:
        chain, i = sel.split(":")
        i = int(i)
        t = ip.read_template(ip.shard_path(a.pool_root, chain), i)
        L = len(t["aatype"])
        pos, pl, pt, seqs = [], [], [], []
        for k in range(a.max_seqs + 1):
            name = f"{chain}_t{i:03d}_s{k:02d}"
            d = os.path.join(a.out, name)
            cif = os.path.join(d, f"{name}_model.cif")
            if not os.path.isfile(cif):
                continue
            res = [r for r in parser.get_structure(name, cif)[0]["A"]]
            assert len(res) == L, (name, len(res), L)
            arr = np.zeros((L, 37, 3), np.float16)
            for j, r in enumerate(res):
                for slot, an in ((0, "N"), (1, "CA"), (2, "C")):
                    arr[j, slot] = r[an].coord
            pos.append(arr)
            pl.append(np.array([np.mean([at.get_bfactor() for at in r]) / 100.0 for r in res], np.float32))
            pt.append(float(json.load(open(os.path.join(d, f"{name}_summary_confidences.json")))["ptm"]))
            seqs.append(json.load(open(os.path.join(a.json_dir, name + ".json")))["sequences"][0]["protein"]["sequence"])
        np.savez(os.path.join(a.out2, f"{chain}_t{i:03d}.npz"), pred_atom37=np.stack(pos), plddt=np.stack(pl),
                 ptm=np.array(pt, np.float32), template_ca=ip.atom37_coords(t)[:, 1], seqs=np.array(seqs), seed=np.int32(1))
        print(f"{chain} t{i}: collected {len(pos)} predictions")


def make_ctl(a):
    from refold_native_control import native
    os.makedirs(a.json_dir, exist_ok=True)
    for chain in a.chains:
        seq, _, _ = native(a.inputs_dir, chain)
        js = {"name": f"ctl_{chain}", "modelSeeds": [1], "dialect": "alphafold3", "version": 4,
              "sequences": [{"protein": {"id": "A", "sequence": seq, "unpairedMsa": "", "pairedMsa": "", "templates": []}}]}
        json.dump(js, open(os.path.join(a.json_dir, f"ctl_{chain}.json"), "w"))
    print(f"wrote {len(a.chains)} control JSONs")


def collect_ctl(a):
    from Bio.PDB import MMCIFParser
    from refold_native_control import native
    parser = MMCIFParser(QUIET=True)
    os.makedirs(a.out2, exist_ok=True)
    for chain in a.chains:
        seq, ca, _ = native(a.inputs_dir, chain)
        name = f"ctl_{chain}"
        res = [r for r in parser.get_structure(name, os.path.join(a.out, name, f"{name}_model.cif"))[0]["A"]]
        assert len(res) == len(seq)
        arr = np.zeros((1, len(seq), 37, 3), np.float16)
        for j, r in enumerate(res):
            for slot, an in ((0, "N"), (1, "CA"), (2, "C")):
                arr[0, j, slot] = r[an].coord
        pl = np.array([[np.mean([at.get_bfactor() for at in r]) / 100.0 for r in res]], np.float32)
        pt = float(json.load(open(os.path.join(a.out, name, f"{name}_summary_confidences.json")))["ptm"])
        np.savez(os.path.join(a.out2, f"{chain}_t000.npz"), pred_atom37=arr, plddt=pl, ptm=np.array([pt], np.float32),
                 template_ca=ca.astype(np.float32), seqs=np.array([seq]), seed=np.int32(1))
        print(f"{chain}: native pLDDT {pl.mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["make", "collect", "make_ctl", "collect_ctl"])
    p.add_argument("--pool-root", default=None)
    p.add_argument("--select", nargs="+", default=None)
    p.add_argument("--inputs-dir", default=None)
    p.add_argument("--chains", nargs="+", default=None)
    p.add_argument("--json-dir", required=True)
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument("--out", default=None, help="AF3 output dir (collect)")
    p.add_argument("--out2", default=None, help="refold npz dir (collect)")
    a = p.parse_args()
    {"make": make, "collect": collect, "make_ctl": make_ctl, "collect_ctl": collect_ctl}[a.cmd](a)


if __name__ == "__main__":
    main()
