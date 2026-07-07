"""Full PDA baseline: stock AF2 and WS5 (pre-ESMFold2-tricks) on the Foldseek-clustered,
structurally-distinct PDA de novo design set (425 entries). For EACH entry x model, saves:
  - the predicted structure as a PDB (openfold.np.protein.from_prediction + to_pdb -- same
    prediction-saving utility OpenFold's own inference scripts use, not reimplemented)
  - self-consistency Kabsch CA-RMSD against the native structure (same protocol as
    eval_pda_self_consistency.py's kabsch_rmsd, imported directly -- not reimplemented) and
    pass/fail at the standard 2A threshold
  - predicted TM-score (out["ptm_score"], already computed by AuxiliaryHeads.forward() via
    compute_tm whenever config.model.heads.tm.enabled=True, which it is by default for both
    finetuning_ptm and model_1_ptm presets -- just reading an existing output key, not a new
    computation)
  - val/lddt_ca (openfold.utils.loss.lddt_ca, same metric as before)
Saved structures let ANY downstream analysis be done later without rerunning inference.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/jupyter-chenxi/openfold/openfold/block_replacement_scripts")
from pda_dataset import PDASingleSeqDataset
from pruned_evoformer import prune_blocks
from eval_pda_self_consistency import kabsch_rmsd

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.np import residue_constants as rc
from openfold.np.protein import from_prediction, to_pdb
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.loss import lddt_ca
from openfold.utils.tensor_utils import tensor_tree_map

DEVICE = os.environ.get("DEVICE", "cuda:0")
MANIFEST = os.environ["PDA_MANIFEST"]
CIF_CACHE_DIR = os.environ.get(
    "PDA_CIF_CACHE_DIR", "/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache")
CROP_SIZE = int(os.environ.get("CROP_SIZE", "256"))
WS5_CKPT = os.environ.get(
    "WS5_CKPT",
    "/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints/best-063-016336.ckpt")
STOCK_JAX_PARAMS = "/home/jupyter-chenxi/params/params_model_1_ptm.npz"
OUT_DIR = os.environ.get("OUT_DIR", "/home/jupyter-chenxi/prune_work/eval_out/pda_baseline_full")
STRUCT_DIR = os.path.join(OUT_DIR, "structures")
RMSD_THRESHOLD = 2.0
SHARD_IDX = int(os.environ.get("SHARD_IDX", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
CA_IDX = rc.atom_order["CA"]


def build_cfg():
    cfg = model_config("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = 3
    cfg.data.eval.crop_size = CROP_SIZE
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    return cfg


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


def build_cfg_stock():
    cfg = model_config("model_1_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = 3
    cfg.data.common.max_extra_msa = 1
    cfg.data.common.max_msa_clusters = 1
    cfg.data.eval.crop_size = CROP_SIZE
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    return cfg


def load_stock(cfg):
    m = AlphaFold(cfg)
    import_jax_weights_(m, STOCK_JAX_PARAMS, version="model_1_ptm")
    return m.to(DEVICE).eval()


def shard(manifest):
    if NUM_SHARDS <= 1:
        return manifest
    return manifest[SHARD_IDX::NUM_SHARDS]


@torch.no_grad()
def run_eval(model, ds, name):
    rows = []
    struct_subdir = os.path.join(STRUCT_DIR, name)
    os.makedirs(struct_subdir, exist_ok=True)
    for i in range(len(ds)):
        entry = ds.manifest[i]
        pdbid, chain_id = entry["pdb"], entry["chain_id"]
        item = ds[i]
        batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in item.items()}
        out = model(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        lddt = lddt_ca(out["final_atom_positions"], batch["all_atom_positions"],
                        batch["all_atom_mask"], eps=1e-6, per_residue=False)
        ptm = out["ptm_score"]

        # Save the predicted structure -- reuses OpenFold's own prediction-saving utility.
        features_np = {k: v[0].detach().cpu().numpy() for k, v in batch.items()
                       if k in ("aatype", "residue_index")}
        result_np = {
            "final_atom_positions": out["final_atom_positions"][0].detach().cpu().numpy(),
            "final_atom_mask": out["final_atom_mask"][0].detach().cpu().numpy(),
        }
        prot = from_prediction({k: v[None] for k, v in features_np.items()}, result_np)
        pdb_path = os.path.join(struct_subdir, f"{pdbid}_{chain_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(to_pdb(prot))

        # Self-consistency Kabsch CA-RMSD against the native structure (same ground truth
        # already in the batch -- all_atom_positions/all_atom_mask, no separate refetch needed).
        pred_ca = out["final_atom_positions"][0, :, CA_IDX, :].detach().cpu().numpy()
        native_ca = batch["all_atom_positions"][0, :, CA_IDX, :].detach().cpu().numpy()
        valid = batch["all_atom_mask"][0, :, CA_IDX].detach().cpu().numpy().astype(bool)
        rmsd = kabsch_rmsd(pred_ca[valid], native_ca[valid]) if valid.sum() >= 3 else float("nan")

        rows.append({
            "pdb": pdbid, "chain_id": chain_id,
            "lddt_ca": lddt.item(), "ptm": ptm.item(), "rmsd": rmsd,
            "success_2A": (not np.isnan(rmsd)) and rmsd < RMSD_THRESHOLD,
            "n_valid_ca": int(valid.sum()),
        })
        print(f"[{name}] {pdbid}_{chain_id}: lddt_ca={lddt.item():.4f} ptm={ptm.item():.4f} "
              f"rmsd={rmsd:.2f} ({i+1}/{len(ds)})", flush=True)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg_ws5 = build_cfg()
    ds = PDASingleSeqDataset(manifest_path=MANIFEST, cif_cache_dir=CIF_CACHE_DIR,
                              config=cfg_ws5.data, mode="eval")
    ds.manifest = shard(ds.manifest)
    print(f"PDA clustered eval set: {len(ds)} entries (shard {SHARD_IDX}/{NUM_SHARDS})", flush=True)

    ws5 = load_ws5(cfg_ws5, WS5_CKPT)
    rows_ws5 = run_eval(ws5, ds, "WS5")
    del ws5
    torch.cuda.empty_cache()

    cfg_stock = build_cfg_stock()
    ds_stock = PDASingleSeqDataset(manifest_path=MANIFEST, cif_cache_dir=CIF_CACHE_DIR,
                                    config=cfg_stock.data, mode="eval")
    ds_stock.manifest = shard(ds_stock.manifest)
    stock = load_stock(cfg_stock)
    rows_stock = run_eval(stock, ds_stock, "stock_AF2")

    by_key_ws5 = {(r["pdb"], r["chain_id"]): r for r in rows_ws5}
    by_key_stock = {(r["pdb"], r["chain_id"]): r for r in rows_stock}
    combined = []
    for k in by_key_ws5:
        if k in by_key_stock:
            w, s = by_key_ws5[k], by_key_stock[k]
            combined.append({
                "pdb": k[0], "chain_id": k[1],
                "lddt_ca_ws5": w["lddt_ca"], "ptm_ws5": w["ptm"], "rmsd_ws5": w["rmsd"],
                "success_2A_ws5": w["success_2A"],
                "lddt_ca_stock": s["lddt_ca"], "ptm_stock": s["ptm"], "rmsd_stock": s["rmsd"],
                "success_2A_stock": s["success_2A"],
            })

    out_csv = os.path.join(OUT_DIR, "pda_baseline_full.csv") if NUM_SHARDS <= 1 else \
        os.path.join(OUT_DIR, f"pda_baseline_full.shard{SHARD_IDX}.csv")
    fieldnames = ["pdb", "chain_id", "lddt_ca_ws5", "ptm_ws5", "rmsd_ws5", "success_2A_ws5",
                  "lddt_ca_stock", "ptm_stock", "rmsd_stock", "success_2A_stock"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(combined)

    n = len(combined)
    print(f"\nn={n}")
    for model_name in ["ws5", "stock"]:
        mean_lddt = sum(r[f"lddt_ca_{model_name}"] for r in combined) / n
        mean_ptm = sum(r[f"ptm_{model_name}"] for r in combined) / n
        n_success = sum(r[f"success_2A_{model_name}"] for r in combined)
        print(f"{model_name}: mean lddt_ca={mean_lddt:.4f} mean pTM={mean_ptm:.4f} "
              f"recall@2A={n_success}/{n} ({n_success/n:.3f})")
    print(f"wrote per-entry results -> {out_csv}")
    print(f"wrote predicted structures -> {STRUCT_DIR}/{{WS5,stock_AF2}}/")


if __name__ == "__main__":
    main()
