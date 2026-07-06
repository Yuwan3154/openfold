"""WS6 FOLLOW-UP #5 control: does the OFFICIAL, FULL (non-pruned, non-distilled) AF2 model also
collapse to near-zero self-consistency recall under single-sequence (no MSA) + no-template
conditions -- or is WS5's near-zero recall (Phase 3: 1.5%; ablation: 2%) specific to WS5's
pruning/distillation/narrow-single-seq training regime? Same Kabsch RMSD<2A recall metric as
Phase 3 (eval_atlas_native_recall.py), on BOTH: (1) the exact same ATLAS-200 sample (reusing
Phase 3's chain-fetch logic verbatim for a clean head-to-head), and (2) WS5's own strict-clean-54
validation chains (natives freshly fetched via local mmcif, matching WS5's own training/eval
distribution -- tests whether even the narrower, "easier" chain set collapses too).
"""
import csv
import os
import random
import sys
import time
import urllib.request
import zipfile

import numpy as np
from Bio.PDB import PDBParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_af2_stock, score_sequence
from openfold.data import mmcif_parsing
from openfold.data.templates import _get_atom_positions
from openfold.np import residue_constants as rc

BASE = "/home/jupyter-chenxi"
DEVICE = os.environ.get("DEVICE", "cuda:0")
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))
RMSD_THRESHOLD = 2.0
CA_IDX = rc.atom_order["CA"]
THREE_TO_ONE = rc.restype_3to1

ATLAS_N_SAMPLE = int(os.environ.get("ATLAS_N_SAMPLE", "200"))
ATLAS_LIST_URL = "https://www.dsimb.inserm.fr/ATLAS/api/parsable"
PDB_LIST_MEMBER = "ATLAS_parsable_latest/2023_03_09_ATLAS_pdb.txt"
LOCAL_LIST_ZIP = f"{BASE}/prune_work/eval_out/atlas_parsable.zip"
ATLAS_PDB_CACHE = f"{BASE}/prune_work/eval_out/pdb_cache"

STRICT54_LIST = f"{BASE}/prune_work/lists_pdb/ws5_val_strict_clean.list"
MMCIF_DIR = f"{BASE}/data/pdb_mmcif/mmcif_files"


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(-1).mean()))


def fetch_atlas_chain_ids():
    if not os.path.exists(LOCAL_LIST_ZIP):
        os.makedirs(os.path.dirname(LOCAL_LIST_ZIP), exist_ok=True)
        urllib.request.urlretrieve(ATLAS_LIST_URL, LOCAL_LIST_ZIP)
    with zipfile.ZipFile(LOCAL_LIST_ZIP) as z:
        with z.open(PDB_LIST_MEMBER) as f:
            return [l.strip() for l in f.read().decode("utf-8").splitlines() if l.strip()]


def fetch_atlas_native(pdb_chain, cache_dir):
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


def run_atlas(model, n_sample):
    pdb_chains = fetch_atlas_chain_ids()
    random.Random(0).shuffle(pdb_chains)  # SAME seed as Phase 3 -- same draw order

    scored = []
    for pdb_chain in pdb_chains:
        if len(scored) >= n_sample:
            break
        try:
            seq, native_ca = fetch_atlas_native(pdb_chain, ATLAS_PDB_CACHE)
        except Exception as e:
            print(f"[atlas] {pdb_chain}: fetch failed: {e}", flush=True)
            continue
        if seq is None or len(seq) == 0 or len(seq) > MAX_LEN or len(seq) != len(native_ca):
            continue
        try:
            out = score_sequence(model, seq, device=DEVICE)
        except Exception as e:
            print(f"[atlas] {pdb_chain}: inference failed: {e}", flush=True)
            continue
        rmsd = kabsch_rmsd(out["ca_coords"], native_ca)
        scored.append({"pdb_chain": pdb_chain, "length": len(seq), "rmsd_to_native": rmsd,
                        "success": rmsd < RMSD_THRESHOLD, "plddt_mean": out["plddt_mean"]})
        if len(scored) % 25 == 0:
            print(f"[atlas] scored {len(scored)}/{n_sample}", flush=True)
    return scored


def run_strict54(model):
    with open(STRICT54_LIST) as f:
        val_chains = [l.strip() for l in f if l.strip()]

    scored = []
    for pdb_chain in val_chains:
        pdbid, chain = pdb_chain.split("_")
        mmcif_path = os.path.join(MMCIF_DIR, f"{pdbid}.cif")
        if not os.path.exists(mmcif_path):
            print(f"[strict54] {pdb_chain}: no local mmcif, skipping", flush=True)
            continue
        try:
            with open(mmcif_path) as f:
                cif_string = f.read()
            parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=cif_string)
            obj = parsed.mmcif_object
            if obj is None or chain not in obj.chain_to_seqres:
                continue
            seq = obj.chain_to_seqres[chain]
            if len(seq) == 0 or len(seq) > MAX_LEN:
                continue
            real_pos, real_mask = _get_atom_positions(
                obj, chain, max_ca_ca_distance=150.0, _zero_center_positions=False)
            real_ca = real_pos[:, CA_IDX, :]
            valid = real_mask[:, CA_IDX].astype(bool)
            if valid.sum() < 5:
                continue
            out = score_sequence(model, seq, device=DEVICE)
        except Exception as e:
            print(f"[strict54] {pdb_chain}: failed: {e}", flush=True)
            continue
        rmsd = kabsch_rmsd(out["ca_coords"][valid], real_ca[valid])
        scored.append({"pdb_chain": pdb_chain, "length": len(seq), "n_resolved": int(valid.sum()),
                        "rmsd_to_native": rmsd, "success": rmsd < RMSD_THRESHOLD,
                        "plddt_mean": out["plddt_mean"]})
        print(f"[strict54] {pdb_chain}: RMSD={rmsd:.2f}A success={rmsd < RMSD_THRESHOLD}", flush=True)
    return scored


def write_and_summarize(scored, out_csv, label):
    if not scored:
        print(f"{label}: no rows scored", flush=True)
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    n_success = sum(r["success"] for r in scored)
    print(f"\n{label}: wrote {len(scored)} rows -> {out_csv}", flush=True)
    print(f"{label}: stock AF2 (single-seq, no-template) recall@2A = "
          f"{n_success}/{len(scored)} ({n_success/len(scored):.3f})", flush=True)


def main():
    t0 = time.time()
    model = load_af2_stock(device=DEVICE)
    print(f"stock AF2 (model_1_ptm, full 48-block, single-seq, no-template) loaded "
          f"({time.time()-t0:.0f}s)", flush=True)

    which = os.environ.get("RUN_SET", "both")  # atlas | strict54 | both
    if which in ("atlas", "both"):
        atlas_scored = run_atlas(model, ATLAS_N_SAMPLE)
        write_and_summarize(
            atlas_scored, f"{BASE}/prune_work/eval_out/stock_af2_atlas.csv", "ATLAS")
    if which in ("strict54", "both"):
        strict54_scored = run_strict54(model)
        write_and_summarize(
            strict54_scored, f"{BASE}/prune_work/eval_out/stock_af2_strict54.csv", "STRICT-CLEAN-54")


if __name__ == "__main__":
    main()
