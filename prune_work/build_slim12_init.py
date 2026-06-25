"""Build a 12-block warm-start checkpoint by trimming the converged 24-block slim model (best-037)
down to 12 blocks, with the SAME every-other-plus-keep-last strategy applied to the 24:
  24 slim blocks (re-indexed 0..23) -> keep slim-indices [0,2,4,6,8,10,12,14,16,18,20,23]
  (= every other of the 24, but the LAST block 23 instead of the 2nd-to-last 22).
These map to original-48 indices [0,4,8,12,16,20,24,28,32,36,40,47].

Loads best-037 EMA weights (the val/lddt_ca-selected best) into a 24-block model, slices to 12,
saves a Lightning-style ckpt (state_dict with `model.` prefix) for --resume_model_weights_only.
"""
import argparse

import torch
import torch.nn as nn

from openfold.config import model_config
from openfold.model.model import AlphaFold

KEEP_24 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47]
SLIM12_FROM24 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]  # which of the 24 to keep

ap = argparse.ArgumentParser()
ap.add_argument("--best037", default="/home/jupyter-chenxi/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt")
ap.add_argument("--out", default="/home/jupyter-chenxi/runs/slim12_init.ckpt")
ap.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
args = ap.parse_args()

cfg = model_config("finetuning_ptm")
m = AlphaFold(cfg)
m.evoformer.blocks = nn.ModuleList([m.evoformer.blocks[i] for i in KEEP_24])  # 24-block

ck = torch.load(args.best037, map_location="cpu", weights_only=False)
sd = ck["ema"]["params"] if (args.use_ema and "ema" in ck) else {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
if len(set(sd.keys()) & set(m.state_dict().keys())) == 0:
    raise RuntimeError("0 key overlap loading best-037 into the 24-block model")
miss, unexp = m.load_state_dict(sd, strict=False)
print(f"loaded best-037 (ema={args.use_ema}) into 24-block: missing={len(miss)} unexpected={len(unexp)}")

# trim 24 -> 12
m.evoformer.blocks = nn.ModuleList([m.evoformer.blocks[i] for i in SLIM12_FROM24])
print(f"trimmed to {len(m.evoformer.blocks)} blocks (orig-48 indices {[KEEP_24[i] for i in SLIM12_FROM24]})")

out_sd = {"model." + k: v for k, v in m.state_dict().items()}
torch.save({"state_dict": out_sd}, args.out)
print(f"wrote {args.out} ({len(out_sd)} tensors)")
