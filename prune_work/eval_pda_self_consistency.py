"""WS6 FOLLOW-UP #7: the CORRECT replacement for the retracted ATLAS Phase 3 test. The Protein
Design Archive (PDA, Chronowska/Stam/Wood 2024, pragmaticproteindesign.bio.ed.ac.uk/pda) is a
manually-curated database of REAL, experimentally-solved de novo PROTEIN DESIGNS (not natural
proteins, unlike ATLAS) -- this tests genuine self-consistency recall on the right population:
fold each design's sequence single-sequence/no-template (the actual condition a novel design
faces) and check Kabsch RMSD against its real solved structure, for both WS5 and stock AF2.

PDA entries are all PUBLISHED/SOLVED (implicitly successful) designs -- this is a RECALL test
(does the model recover a KNOWN-GOOD design's fold), not an AUROC/discrimination test like
Phase 1a/1b (which have real success/failure labels).
"""
import csv
import json
import os
import random
import ssl
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5, load_af2_stock, score_sequence, resolve_ws5_ckpt
from openfold.data import mmcif_parsing
from openfold.data.templates import _get_atom_positions
from openfold.np import residue_constants as rc

DEVICE = os.environ.get("DEVICE", "cuda:0")
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))
N_SAMPLE = int(os.environ.get("N_SAMPLE", "100"))
RMSD_THRESHOLD = 2.0
WS5_CKPT = os.environ.get("CKPT") or None
CA_IDX = rc.atom_order["CA"]

OUT_DIR = "/home/jupyter-chenxi/prune_work/eval_out"
STUBS_CACHE = f"{OUT_DIR}/pda_all_design_stubs.json"
CIF_CACHE_DIR = f"{OUT_DIR}/pda_mmcif_cache"
OUT_CSV = os.environ.get("OUT_CSV", f"{OUT_DIR}/pda_self_consistency.csv")

STUBS_URL = "https://pragmaticproteindesign.bio.ed.ac.uk/pda-api/all-design-stubs"
DETAIL_URL = "https://pragmaticproteindesign.bio.ed.ac.uk/pda-api/design-details/"

DESIGN_TAGS = {"de novo protein", "de novo design", "de novo", "computational design"}
EXCLUDE_TAGS = {"structural genomics", "psi-biology", "protein structure initiative",
                "northeast structural genomics consortium", "nesg", "unknown function"}

# PDA's server has an incomplete cert chain -- verified manually via curl -k that this is a
# server-side config issue (public, non-sensitive data), not a client problem.
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30, context=SSL_CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def get_clean_stubs():
    if os.path.exists(STUBS_CACHE):
        with open(STUBS_CACHE) as f:
            stubs = json.load(f)
    else:
        stubs = fetch_json(STUBS_URL)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(STUBS_CACHE, "w") as f:
            json.dump(stubs, f)
    clean = []
    for d in stubs:
        tags = set(d.get("tags", []))
        if tags & DESIGN_TAGS and not (tags & EXCLUDE_TAGS):
            clean.append(d)
    return clean


def fetch_mmcif(pdbid):
    os.makedirs(CIF_CACHE_DIR, exist_ok=True)
    path = os.path.join(CIF_CACHE_DIR, f"{pdbid}.cif")
    if not os.path.exists(path):
        with urllib.request.urlopen(
                f"https://files.rcsb.org/download/{pdbid}.cif", timeout=30, context=SSL_CTX) as r:
            content = r.read()
        with open(path, "wb") as f:
            f.write(content)
    return path


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(-1).mean()))


def get_design_chain(pdbid):
    """Returns (chain_id_for_structure_lookup, seq) for the first designed ('D') chain if any,
    else the first available chain -- or None if no usable chain data."""
    detail = fetch_json(f"{DETAIL_URL}{pdbid}")
    chains = detail.get("chains", [])
    if not chains:
        return None
    designed = [c for c in chains if c.get("chain_type") == "D"]
    chain = designed[0] if designed else chains[0]
    seq = chain.get("chain_seq_nat", "")
    chain_id_field = chain.get("chain_id", "")
    chain_id = chain_id_field.split(",")[0].strip() if chain_id_field else None
    if not seq or not chain_id:
        return None
    return chain_id, seq


def main():
    clean = get_clean_stubs()
    print(f"PDA clean de novo subset: {len(clean)} entries", flush=True)
    random.Random(0).shuffle(clean)

    model_ws5 = load_ws5(WS5_CKPT or resolve_ws5_ckpt(), device=DEVICE)
    model_stock = load_af2_stock(device=DEVICE)
    print("both models loaded (WS5 no-template, stock AF2 no-template)", flush=True)

    rows = []
    t0 = time.time()
    for stub in clean:
        if len(rows) >= N_SAMPLE:
            break
        pdbid = stub["pdb"]
        try:
            result = get_design_chain(pdbid)
            if result is None:
                continue
            chain_id, seq = result
            if len(seq) == 0 or len(seq) > MAX_LEN:
                continue

            mmcif_path = fetch_mmcif(pdbid)
            with open(mmcif_path) as f:
                cif_string = f.read()
            parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=cif_string)
            obj = parsed.mmcif_object
            if obj is None or chain_id not in obj.chain_to_seqres:
                print(f"{pdbid}: chain {chain_id} not found in mmcif, skipping", flush=True)
                continue

            real_pos, real_mask = _get_atom_positions(
                obj, chain_id, max_ca_ca_distance=150.0, _zero_center_positions=False)
            real_ca = real_pos[:, CA_IDX, :]
            valid = real_mask[:, CA_IDX].astype(bool)
            if valid.sum() < 5 or abs(len(seq) - len(real_ca)) > 5:
                # length mismatch beyond a small tolerance (e.g. His-tag not in real structure) --
                # skip rather than silently mis-align a garbage comparison.
                continue

            n = min(len(seq), len(real_ca))
            out_ws5 = score_sequence(model_ws5, seq[:n], device=DEVICE)
            out_stock = score_sequence(model_stock, seq[:n], device=DEVICE)
        except Exception as e:
            print(f"{pdbid}: failed: {e}", flush=True)
            continue

        valid_n = valid[:n]
        if valid_n.sum() < 5:
            continue
        rmsd_ws5 = kabsch_rmsd(out_ws5["ca_coords"][valid_n], real_ca[:n][valid_n])
        rmsd_stock = kabsch_rmsd(out_stock["ca_coords"][valid_n], real_ca[:n][valid_n])
        rows.append({
            "pdb": pdbid, "chain_id": chain_id, "length": n,
            "rmsd_ws5": rmsd_ws5, "success_ws5": rmsd_ws5 < RMSD_THRESHOLD,
            "rmsd_stock": rmsd_stock, "success_stock": rmsd_stock < RMSD_THRESHOLD,
        })
        print(f"{pdbid}_{chain_id}: len={n} WS5_RMSD={rmsd_ws5:.2f}A stock_RMSD={rmsd_stock:.2f}A",
              flush=True)
        if len(rows) % 20 == 0:
            print(f"  -- {len(rows)}/{N_SAMPLE} scored ({time.time()-t0:.0f}s elapsed)", flush=True)

    if not rows:
        print("no rows scored -- aborting", flush=True)
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}", flush=True)

    n_ws5 = sum(r["success_ws5"] for r in rows)
    n_stock = sum(r["success_stock"] for r in rows)
    print(f"WS5 (no-template) recall@2A on real de novo designs:   {n_ws5}/{len(rows)} ({n_ws5/len(rows):.3f})")
    print(f"stock AF2 (no-template) recall@2A on real de novo designs: {n_stock}/{len(rows)} ({n_stock/len(rows):.3f})")


if __name__ == "__main__":
    main()
