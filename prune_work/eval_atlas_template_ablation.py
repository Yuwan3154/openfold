"""WS6 follow-up: templates-ON vs templates-OFF ablation on the SAME ATLAS chains used by
Phase 3 (eval_atlas_native_recall.py), to test whether WS5's good (template-assisted)
validation performance is driven by template availability rather than genuine template-free
folding capability -- WS5 was trained with --single_seq_keep_templates (templates enabled for
both train AND val), but Phase 3's self-consistency test (correctly, matching the field-standard
de novo design protocol) ran with templates OFF. This directly measures how much of that gap is
explained by the templates-mismatch alone.

Self-template design: template = the chain's OWN real native structure (whole-chain, AF2Rank-
style self-template), reusing template_feat_builder.build_template_features (built for Phase 5)
with target_offset=0 (template covers the WHOLE query, no partial mapping needed).

Uses the SAME chain list as Phase 3 (identical shuffle seed) but needs mmcif (not just .pdb) for
the OpenFold template pipeline, and folds the mmcif's own full SEQRES sequence (not the
ATOM-derived resolved-only sequence Phase 3 used) so the templated and non-templated runs here
fold the IDENTICAL sequence -- RMSD is computed only over resolved (real atom-mask) positions.
"""
import csv
import os
import random
import sys
import time
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import (
    load_ws5, load_ws5_templated, score_sequence, score_sequence_templated, resolve_ws5_ckpt,
)
from template_feat_builder import build_template_features
from openfold.data import mmcif_parsing
from openfold.data.templates import _get_atom_positions
from openfold.np import residue_constants as rc

CKPT = os.environ.get("CKPT") or None
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/atlas_template_ablation.csv")
N_SAMPLE = int(os.environ.get("N_SAMPLE", "50"))
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))
RMSD_THRESHOLD = 2.0
CIF_CACHE = "/home/jupyter-chenxi/prune_work/eval_out/mmcif_cache_atlas"

ATLAS_LIST_URL = "https://www.dsimb.inserm.fr/ATLAS/api/parsable"
PDB_LIST_MEMBER = "ATLAS_parsable_latest/2023_03_09_ATLAS_pdb.txt"
LOCAL_LIST_ZIP = "/home/jupyter-chenxi/prune_work/eval_out/atlas_parsable.zip"

CA_IDX = rc.atom_order["CA"]


def fetch_pdb_chain_ids():
    if not os.path.exists(LOCAL_LIST_ZIP):
        os.makedirs(os.path.dirname(LOCAL_LIST_ZIP), exist_ok=True)
        urllib.request.urlretrieve(ATLAS_LIST_URL, LOCAL_LIST_ZIP)
    with zipfile.ZipFile(LOCAL_LIST_ZIP) as z:
        with z.open(PDB_LIST_MEMBER) as f:
            return [l.strip() for l in f.read().decode("utf-8").splitlines() if l.strip()]


def fetch_mmcif(pdbid, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdbid}.cif")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdbid}.cif", path)
    return path


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
    random.Random(0).shuffle(pdb_chains)  # same seed as Phase 3 -- same draw order

    model_notempl = load_ws5(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    model_templ = load_ws5_templated(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    print("both WS5 configs loaded (templates off + templates on)", flush=True)

    scored = []
    t0 = time.time()
    for pdb_chain in pdb_chains:
        if len(scored) >= N_SAMPLE:
            break
        pdbid, chain = pdb_chain.split("_")
        try:
            mmcif_path = fetch_mmcif(pdbid, CIF_CACHE)
            with open(mmcif_path) as f:
                cif_string = f.read()
            parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=cif_string)
            if parsed.mmcif_object is None:
                print(f"{pdb_chain}: mmcif parse failed: {parsed.errors}", flush=True)
                continue
            obj = parsed.mmcif_object
            if chain not in obj.chain_to_seqres:
                print(f"{pdb_chain}: chain not in mmcif chain_to_seqres", flush=True)
                continue
            seq = obj.chain_to_seqres[chain]
        except Exception as e:
            print(f"{pdb_chain}: fetch/parse failed: {e}", flush=True)
            continue
        if len(seq) == 0 or len(seq) > MAX_LEN:
            continue

        real_pos, real_mask = _get_atom_positions(
            obj, chain, max_ca_ca_distance=150.0, _zero_center_positions=False)
        real_ca = real_pos[:, CA_IDX, :]
        valid = real_mask[:, CA_IDX].astype(bool)
        if valid.sum() < 10:
            continue

        try:
            template_feats = build_template_features(
                mmcif_path=mmcif_path, pdb_id=pdbid, target_chain_id=chain,
                query_sequence=seq, target_offset=0, device=DEVICE)
            out_templ = score_sequence_templated(model_templ, seq, template_feats, device=DEVICE)
            out_notempl = score_sequence(model_notempl, seq, device=DEVICE)
        except Exception as e:
            print(f"{pdb_chain}: inference failed: {e}", flush=True)
            continue

        rmsd_templ = kabsch_rmsd(out_templ["ca_coords"][valid], real_ca[valid])
        rmsd_notempl = kabsch_rmsd(out_notempl["ca_coords"][valid], real_ca[valid])
        scored.append({
            "pdb_chain": pdb_chain,
            "length": len(seq),
            "n_resolved": int(valid.sum()),
            "rmsd_templated": rmsd_templ,
            "success_templated": rmsd_templ < RMSD_THRESHOLD,
            "plddt_templated": out_templ["plddt_mean"],
            "rmsd_no_template": rmsd_notempl,
            "success_no_template": rmsd_notempl < RMSD_THRESHOLD,
            "plddt_no_template": out_notempl["plddt_mean"],
        })
        if len(scored) % 10 == 0:
            print(f"scored {len(scored)}/{N_SAMPLE} ({time.time() - t0:.0f}s elapsed)", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    print(f"wrote {len(scored)} scored rows -> {OUT_CSV}", flush=True)

    n_templ = sum(r["success_templated"] for r in scored)
    n_notempl = sum(r["success_no_template"] for r in scored)
    print(f"\nRMSD<2A recall, TEMPLATED (self-template):    {n_templ}/{len(scored)} ({n_templ/len(scored):.3f})")
    print(f"RMSD<2A recall, NO TEMPLATE (paired, same seqs): {n_notempl}/{len(scored)} ({n_notempl/len(scored):.3f})")
    print("comparator: Phase 3's original result (different, ATOM-derived seq, N=200): 3/200 (0.015)")


if __name__ == "__main__":
    main()
