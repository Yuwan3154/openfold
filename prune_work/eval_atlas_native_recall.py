"""Phase 3 (native-recall half only -- see module note): replicate the CANONICAL self-consistency
sanity check from "Rethinking Self-Consistency in Protein Generative Models" (Lee/Kim/Lin/
AlQuraishi, ICML-W 2026) using our WS5 checkpoint in place of ESMFold: fold each native sequence
single-sequence/no-template, Kabsch-align CA atoms, accept if RMSD < 2 A (their canonical
threshold), report the recall fraction over a sample of ATLAS's 1938 native protein chains.

This only tests the pipeline's baseline sanity ("does our model behave like a reasonable
single-sequence folding oracle at all") -- it does NOT replicate the paper's full ensemble
self-consistency method (needs MD trajectory sampling) or the Rosetta-decoy specificity half
of Phase 3 (Park et al. 2016 decoy set location not yet confirmed -- see PLAN notes; not
implemented here to avoid guessing at an unverified data source).

ATLAS entry list: Zenodo-independent, fetched live from ATLAS's own /api/parsable bulk-download
endpoint (https://www.dsimb.inserm.fr/ATLAS/api/parsable). Native structures: RCSB PDB direct
download (https://files.rcsb.org/download/{pdbid}.pdb) -- ATLAS entries are single small-protein
chains, so plain Bio.PDB auth-chain parsing is fine here (no multi-subunit label/auth divergence
risk).
"""
import csv
import io
import os
import random
import sys
import time
import urllib.request
import zipfile

import numpy as np
from Bio.PDB import PDBParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5, score_sequence, resolve_ws5_ckpt
from openfold.np import residue_constants as rc

CKPT = os.environ.get("CKPT") or None  # resolved lazily below (checkpoint filename rotates while WS5 trains)
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/atlas_native_recall.csv")
DEVICE = os.environ.get("DEVICE", "cuda:0")
N_SAMPLE = int(os.environ.get("N_SAMPLE", "200"))
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))
RMSD_THRESHOLD = 2.0  # Angstrom, matches the paper's canonical-success cutoff

ATLAS_LIST_URL = "https://www.dsimb.inserm.fr/ATLAS/api/parsable"
PDB_LIST_MEMBER = "ATLAS_parsable_latest/2023_03_09_ATLAS_pdb.txt"
LOCAL_LIST_ZIP = os.environ.get("LOCAL_LIST_ZIP", "/home/jupyter-chenxi/prune_work/eval_out/atlas_parsable.zip")

THREE_TO_ONE = rc.restype_3to1


def fetch_pdb_chain_ids():
    if not os.path.exists(LOCAL_LIST_ZIP):
        os.makedirs(os.path.dirname(LOCAL_LIST_ZIP), exist_ok=True)
        urllib.request.urlretrieve(ATLAS_LIST_URL, LOCAL_LIST_ZIP)
    with zipfile.ZipFile(LOCAL_LIST_ZIP) as z:
        with z.open(PDB_LIST_MEMBER) as f:
            return [l.strip() for l in f.read().decode("utf-8").splitlines() if l.strip()]


def fetch_native_chain(pdb_chain, cache_dir):
    pdbid, chain = pdb_chain.split("_")
    path = os.path.join(cache_dir, f"{pdbid}.pdb")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdbid}.pdb", path)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdbid, path)
    model = structure[0]
    if chain not in model:
        return None, None
    seq, ca = [], []
    for res in model[chain]:
        if res.id[0] != " " or "CA" not in res:
            continue
        resname = res.get_resname()
        if resname not in THREE_TO_ONE:
            continue
        seq.append(THREE_TO_ONE[resname])
        ca.append(res["CA"].coord)
    return "".join(seq), np.array(ca, dtype=np.float32)


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    a_aligned = a @ r.T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(-1).mean()))


def main():
    pdb_chains = fetch_pdb_chain_ids()
    print(f"ATLAS has {len(pdb_chains)} entries total", flush=True)
    random.Random(0).shuffle(pdb_chains)

    model = load_ws5(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cache_dir = "/home/jupyter-chenxi/prune_work/eval_out/pdb_cache"
    os.makedirs(cache_dir, exist_ok=True)

    scored = []
    t0 = time.time()
    for pdb_chain in pdb_chains:
        if len(scored) >= N_SAMPLE:
            break
        try:
            seq, native_ca = fetch_native_chain(pdb_chain, cache_dir)
        except Exception as e:
            print(f"{pdb_chain}: fetch failed: {e}", flush=True)
            continue
        if seq is None or len(seq) == 0 or len(seq) > MAX_LEN or len(seq) != len(native_ca):
            continue
        try:
            out = score_sequence(model, seq, device=DEVICE)
        except Exception as e:
            print(f"{pdb_chain}: inference failed: {e}", flush=True)
            continue
        rmsd = kabsch_rmsd(out["ca_coords"], native_ca)
        scored.append({
            "pdb_chain": pdb_chain,
            "length": len(seq),
            "rmsd_to_native": rmsd,
            "canonical_success": rmsd < RMSD_THRESHOLD,
            "our_plddt_mean": out["plddt_mean"],
        })
        if len(scored) % 25 == 0:
            print(f"scored {len(scored)}/{N_SAMPLE} ({time.time() - t0:.0f}s elapsed)", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    print(f"wrote {len(scored)} scored rows -> {OUT_CSV}", flush=True)

    n_success = sum(r["canonical_success"] for r in scored)
    print(f"\ncanonical self-consistency recall: {n_success}/{len(scored)} "
          f"({n_success / len(scored):.3f}) at RMSD < {RMSD_THRESHOLD}A")
    print("comparator: Rethinking-Self-Consistency paper reports ESMFold canonical success "
          "rate 56% (ALL bin, N=1) on the full 1520-protein ATLAS set")


if __name__ == "__main__":
    main()
