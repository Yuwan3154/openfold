"""Compute val/lddt_ca baselines (stock AF2 and WS5, pre-ESMFold2-tricks) on the Foldseek-clustered,
structurally-distinct PDA de novo design set -- reuses PDASingleSeqDataset (same feature-building
path the training run's own validation will use) so baseline and training-run numbers are directly
comparable, and reuses openfold.utils.loss.lddt_ca (same metric OpenFold's own validation_step logs
as val/lddt_ca) rather than a new metric implementation.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/jupyter-chenxi/openfold/openfold/block_replacement_scripts")
from pda_dataset import PDASingleSeqDataset
from pruned_evoformer import prune_blocks

from openfold.config import model_config
from openfold.model.model import AlphaFold
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
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/pda_baseline_lddt.csv")
SHARD_IDX = int(os.environ.get("SHARD_IDX", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))


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


@torch.no_grad()
def run_eval(model, ds, name):
    rows = []
    for i in range(len(ds)):
        entry = ds.manifest[i]
        item = ds[i]
        batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in item.items()}
        out = model(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        lddt = lddt_ca(out["final_atom_positions"], batch["all_atom_positions"],
                        batch["all_atom_mask"], eps=1e-6, per_residue=False)
        rows.append({"pdb": entry["pdb"], "chain_id": entry["chain_id"], "lddt_ca": lddt.item()})
        print(f"[{name}] {entry['pdb']}_{entry['chain_id']}: lddt_ca={lddt.item():.4f} "
              f"({i+1}/{len(ds)})", flush=True)
    return rows


def shard(manifest):
    """NUM_SHARDS>1: only process every NUM_SHARDS-th entry starting at SHARD_IDX -- lets
    independent parallel processes (one per free GPU) split the eval set, each writing its own
    shard CSV; a separate merge step (pda_baseline_merge.py) combines them afterward."""
    if NUM_SHARDS <= 1:
        return manifest
    return manifest[SHARD_IDX::NUM_SHARDS]


def main():
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

    lddt_ws5 = {(r["pdb"], r["chain_id"]): r["lddt_ca"] for r in rows_ws5}
    lddt_stock = {(r["pdb"], r["chain_id"]): r["lddt_ca"] for r in rows_stock}
    combined = []
    for k in lddt_ws5:
        if k in lddt_stock:
            combined.append({"pdb": k[0], "chain_id": k[1],
                              "lddt_ca_ws5": lddt_ws5[k], "lddt_ca_stock": lddt_stock[k]})

    import csv
    out_csv = OUT_CSV if NUM_SHARDS <= 1 else OUT_CSV.replace(".csv", f".shard{SHARD_IDX}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "chain_id", "lddt_ca_ws5", "lddt_ca_stock"])
        w.writeheader()
        w.writerows(combined)

    mean_ws5 = sum(r["lddt_ca_ws5"] for r in combined) / len(combined)
    mean_stock = sum(r["lddt_ca_stock"] for r in combined) / len(combined)
    print(f"\nn={len(combined)}")
    print(f"WS5 (pre-ESMFold2-tricks) mean val/lddt_ca on this shard:   {mean_ws5:.4f}")
    print(f"stock AF2 mean val/lddt_ca on this shard: {mean_stock:.4f}")
    print(f"wrote per-entry results -> {out_csv}")


if __name__ == "__main__":
    main()
