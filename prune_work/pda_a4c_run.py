"""A4c batch runner: gap-concat + GT-template re-eval for the hetero-binder / nanocage / homo-oligomer
arms. Uses the SAME WS5 checkpoint as the whole-425 baseline (pda_baseline_full.py), same Kabsch-RMSD
recall metric, so numbers are directly comparable to that baseline's per-entry values -- the only
difference is real multi-chain context (template for arm1, correct stoichiometry for arm2/arm3)
instead of folding the design chain in isolation.
"""
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# MUST point at THIS repo's own worktree, not /home/jupyter-chenxi/openfold/ (a different, unrelated
# worktree). Root-caused 2026-07-15 (full traceback -> single_seq_infer.py, pulled in transitively
# via eval_pda_self_consistency.py, hardcodes sys.path.insert(0, "/home/jupyter-chenxi/openfold/
# openfold/block_replacement_scripts") as an import side effect, poisoning `openfold` resolution for
# everything imported afterward) -- fixed by reimplementing kabsch_rmsd directly in pda_a4c_lib.py
# instead of importing it through that poisoned chain; do NOT import eval_pda_self_consistency or
# single_seq_infer from this script.
sys.path.insert(0, "/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold/block_replacement_scripts")
from pda_a4c_lib import (build_slots_from_components, build_multichain_features, build_cfg,
                          get_native_design_coords, kabsch_rmsd)
from pruned_evoformer import prune_blocks

from openfold.config import model_config
from openfold.data import feature_pipeline
from openfold.model.model import AlphaFold
from openfold.np import residue_constants as rc
from openfold.utils.tensor_utils import tensor_tree_map

ARM = os.environ["ARM"]  # "arm1", "arm2", or "arm3"
DEVICE = os.environ.get("DEVICE", "cuda:0")
CIF_DIR = "/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache"
WS5_CKPT = os.environ.get(
    "WS5_CKPT",
    "/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints/best-063-016336.ckpt")
KAL = "/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign"
MANIFEST = f"/home/jupyter-chenxi/prune_work/eval_out/a4c_{ARM}_{'hetero_binder' if ARM=='arm1' else ('nanocage' if ARM=='arm2' else 'homo')}.json"
OUT_CSV = os.environ.get("OUT_CSV", f"/home/jupyter-chenxi/prune_work/eval_out/a4c_{ARM}_results.csv")
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = no limit, else stop after N entries (for the smoke test)
CA_IDX = rc.atom_order["CA"]


def load_ws5(cfg, ckpt_path):
    m = AlphaFold(cfg)
    prune_blocks(m.evoformer)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = m.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    assert not missing, f"unexpected missing keys: {missing}"
    assert all(k.startswith("template_embedder.") for k in unexpected), \
        f"unexpected non-template keys: {[k for k in unexpected if not k.startswith('template_embedder.')]}"
    return m.to(DEVICE).eval()


def entry_to_components(entry):
    """Attach template info (arm1 only) to the natural-partner component(s)."""
    comps = []
    for c in entry["components"]:
        comp = dict(seq=c["seq"], copies=c["copies"], is_design=c["is_design"])
        if ARM == "arm1" and c.get("is_natural"):
            comp["template"] = dict(
                mmcif_path=f"{CIF_DIR}/{entry['pdb']}.cif",
                pdb_id=entry["pdb"], chain_id=entry["pdb"],  # chain_id resolved below via seq match
            )
        comps.append(comp)
    return comps


@torch.no_grad()
def run_one(model, cfg, entry):
    components = entry_to_components(entry)
    # For template components we need a REAL chain_id present in the cif to align against --
    # find one by scanning the cif's chain_to_seqres for an exact sequence match.
    from openfold.data import mmcif_parsing
    if ARM == "arm1":
        with open(f"{CIF_DIR}/{entry['pdb']}.cif") as f:
            mo = mmcif_parsing.parse(file_id=entry["pdb"], mmcif_string=f.read()).mmcif_object
        for comp in components:
            if comp.get("template") is not None:
                match = next(cid for cid, seq in mo.chain_to_seqres.items() if seq == comp["seq"])
                comp["template"]["chain_id"] = match

    slots = build_slots_from_components(components)
    design_len = len(slots[0]["seq"])
    raw_feats = build_multichain_features(slots, KAL)

    fp = feature_pipeline.FeaturePipeline(cfg.data)
    feats = fp.process_features(raw_feats, mode="eval")
    batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in feats.items()}

    out = model(batch)
    batch_last = tensor_tree_map(lambda t: t[..., -1], batch)

    pred_ca_full = out["final_atom_positions"][0, :, CA_IDX, :].detach().cpu().numpy()
    pred_ca_design = pred_ca_full[:design_len]

    native_ca, native_valid = get_native_design_coords(CIF_DIR, entry["pdb"], entry["chain_id"])
    # native_ca/native_valid are indexed 1:1 with slot 0's own sequence (design_len); mask BOTH
    # arrays by the SAME boolean index so unresolved (non-trailing) native residues can't desync
    # predicted-vs-native correspondence (see get_native_design_coords docstring).
    n = min(design_len, len(native_valid))
    valid = native_valid[:n]
    rmsd = kabsch_rmsd(pred_ca_design[:n][valid], native_ca[:n][valid]) if valid.sum() >= 3 else float("nan")
    return dict(pdb=entry["pdb"], chain_id=entry["chain_id"], total_len=entry["total_len"],
               n_slots=len(slots), design_len=design_len, rmsd=rmsd,
               success_2A=(not np.isnan(rmsd)) and rmsd < 2.0)


def main():
    manifest = json.load(open(MANIFEST))
    if LIMIT:
        manifest = manifest[:LIMIT]
    print(f"ARM={ARM} n_entries={len(manifest)} device={DEVICE}", flush=True)

    cfg = build_cfg(model_config)
    model = load_ws5(cfg, WS5_CKPT)

    rows = []
    for i, entry in enumerate(manifest):
        t0 = time.time()
        try:
            row = run_one(model, cfg, entry)
        except Exception as e:
            print(f"[{ARM}] {entry['pdb']}_{entry['chain_id']}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            rows.append(dict(pdb=entry["pdb"], chain_id=entry["chain_id"], total_len=entry["total_len"],
                             n_slots=None, design_len=None, rmsd=float("nan"), success_2A=False))
            continue
        dt = time.time() - t0
        print(f"[{ARM}] {row['pdb']}_{row['chain_id']}: total_len={row['total_len']} "
              f"n_slots={row['n_slots']} rmsd={row['rmsd']:.2f} success={row['success_2A']} "
              f"({i+1}/{len(manifest)}, {dt:.1f}s)", flush=True)
        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if not np.isnan(r["rmsd"]))
    n_succ = sum(r["success_2A"] for r in rows)
    print(f"\n{ARM} DONE: n={len(rows)} scored={n_ok} recall@2A={n_succ}/{len(rows)} "
          f"({n_succ/len(rows):.3f}) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
