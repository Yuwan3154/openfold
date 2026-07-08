"""Slide-deck request: memory + runtime comparison of stock (full 48-block) AF2 vs WS5 (pruned
48-block, drop col-attn + triangle-attn), with and without gradient/activation checkpointing
(config.globals.blocks_per_ckpt). One training step (forward + backward) per combination, single-
sequence + templates (WS5's actual deployed regime), at a couple of representative crop lengths.

Random weight init (architecture, not weight VALUES, determines memory/speed) -- no checkpoint
loading needed, keeps this fast and self-contained. Synthetic random template tensors (right
shapes, not real structure -- memory/speed depend only on tensor shape).
"""
import csv
import gc
import os
import sys
import time

import torch
import torch.nn.functional as F

BASE = "/home/jupyter-chenxi"
sys.path.insert(0, f"{BASE}/openfold")
sys.path.insert(0, f"{BASE}/openfold/openfold/block_replacement_scripts")
sys.path.insert(0, f"{BASE}/prune_work")

from openfold.config import model_config
from openfold.model.model import AlphaFold
from hallucination_straight_through import make_feature_batch
from pruned_evoformer import prune_blocks

DEVICE = "cuda:0"
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "128,256").split(",")]
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/bench_memory_runtime.csv")
N_WARMUP = 1
N_TIMED = 3


def build_cfg(pruned, ckpt_on, recycle=3):
    cfg = model_config("model_1_ptm", train=True, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.globals.blocks_per_ckpt = 1 if ckpt_on else None
    cfg.data.common.max_recycling_iters = recycle
    cfg.data.common.max_extra_msa = 1
    cfg.data.common.max_msa_clusters = 1
    cfg.model.template.enabled = True
    cfg.data.common.use_templates = True
    cfg.data.common.use_template_torsion_angles = True
    return cfg


def build_model(pruned, ckpt_on, device):
    cfg = build_cfg(pruned, ckpt_on)
    m = AlphaFold(cfg)
    if pruned:
        prune_blocks(m.evoformer)
    return m.to(device).train()


def synthetic_template_feats(num_res, recycle, device):
    # Shapes match finish_template_features's verified convention (template_feat_builder.py):
    # leading dim is n_templ directly, NOT a separate batch dim -- confirmed via its own
    # asserts (`protein["template_aatype"].shape[0] == 1` where 1 == n_templ, sliced top-1).
    n_templ = 1
    aatype = torch.randint(0, 21, (n_templ, num_res), device=device).long()
    positions = torch.randn(n_templ, num_res, 37, 3, device=device)
    mask = torch.ones(n_templ, num_res, 37, device=device)
    pseudo_beta = torch.randn(n_templ, num_res, 3, device=device)
    pseudo_beta_mask = torch.ones(n_templ, num_res, device=device)
    template_mask = torch.ones(n_templ, device=device)
    torsion_angles_sin_cos = torch.randn(n_templ, num_res, 7, 2, device=device)
    alt_torsion_angles_sin_cos = torch.randn(n_templ, num_res, 7, 2, device=device)
    torsion_angles_mask = torch.ones(n_templ, num_res, 7, device=device)

    def add_cycle(x):
        return x.unsqueeze(-1).expand(*x.shape, recycle + 1)

    feats = {
        "template_aatype": aatype, "template_all_atom_positions": positions,
        "template_all_atom_mask": mask, "template_pseudo_beta": pseudo_beta,
        "template_pseudo_beta_mask": pseudo_beta_mask, "template_mask": template_mask,
        "template_torsion_angles_sin_cos": torsion_angles_sin_cos,
        "template_alt_torsion_angles_sin_cos": alt_torsion_angles_sin_cos,
        "template_torsion_angles_mask": torsion_angles_mask,
    }
    return {k: add_cycle(v) for k, v in feats.items()}


def run_one(pruned, ckpt_on, length, recycle=3):
    torch.cuda.empty_cache()
    gc.collect()
    model = build_model(pruned, ckpt_on, DEVICE)

    aat = torch.randint(0, 20, (length,), device=DEVICE)
    logits = (3.0 * F.one_hot(aat, 20).float())
    ri = torch.arange(length, device=DEVICE)
    batch = make_feature_batch(logits, ri, recycle_dim=recycle + 1)
    batch.update(synthetic_template_feats(length, recycle, DEVICE))

    def step():
        model.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out["final_atom_positions"].float().sum() + out["distogram_logits"].float().sum()
        loss.backward()
        torch.cuda.synchronize()

    for _ in range(N_WARMUP):
        step()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    t0 = time.time()
    for _ in range(N_TIMED):
        step()
    elapsed = (time.time() - t0) / N_TIMED
    peak_mem_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9

    del model, batch
    torch.cuda.empty_cache()
    gc.collect()
    return peak_mem_gb, elapsed


def main():
    rows = []
    for length in LENGTHS:
        for pruned, model_name in [(False, "stock_af2"), (True, "ws5_pruned")]:
            for ckpt_on in [False, True]:
                try:
                    mem, t = run_one(pruned, ckpt_on, length)
                    print(f"L={length} model={model_name} ckpt={ckpt_on}: "
                          f"peak_mem={mem:.2f}GB time/step={t:.3f}s", flush=True)
                    rows.append({"length": length, "model": model_name, "checkpointing": ckpt_on,
                                 "peak_mem_gb": mem, "time_per_step_s": t})
                except torch.cuda.OutOfMemoryError as e:
                    print(f"L={length} model={model_name} ckpt={ckpt_on}: OOM", flush=True)
                    rows.append({"length": length, "model": model_name, "checkpointing": ckpt_on,
                                 "peak_mem_gb": "OOM", "time_per_step_s": "OOM"})
                    torch.cuda.empty_cache()
                    gc.collect()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
