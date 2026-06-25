"""Build a fresh-warmup init ckpt for the conf-weight tuning smoke: take the 24-block slim run's LAST
checkpoint and keep only its weights + global_step=0, so the weights-only resume starts a FRESH warmup
(controlled small LR) rather than resuming at the slim run's step ~9500 (full LR).
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--last", default="/home/jupyter-chenxi/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/last.ckpt")
ap.add_argument("--out", default="/home/jupyter-chenxi/runs/conf_smoke_init.ckpt")
args = ap.parse_args()

ck = torch.load(args.last, map_location="cpu", weights_only=False)
torch.save({"state_dict": ck["state_dict"], "global_step": 0, "epoch": 0}, args.out)
print(f"wrote {args.out} (state_dict from {args.last}, global_step=0)")
