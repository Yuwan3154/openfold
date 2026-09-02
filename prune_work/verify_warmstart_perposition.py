"""End-to-end warm-start gate for --per_position_delta, run on CPU.

Replicates train_openfold.py's ACTUAL weights-only warm-start path -- the same flag application,
the same `select_ema_warmstart_weights`, the same `import_openfold_weights_` (which loads twice:
strict=True, then a retry on RuntimeError) -- against a real checkpoint, with the full AlphaFold
model rather than the submodule in isolation.

Gates, all of which must hold before any run is launched:
  1. the EMA weights are what gets loaded, never `state_dict`
  2. EVERY checkpoint tensor lands: unexpected == 0, and missing == exactly the new head/buffer
  3. the effective per-position delta equals the checkpoint's delta
  4. every OTHER contractive parameter is bit-identical to the checkpoint
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# this script lives in prune_work/, so sys.path[0] is prune_work/, not the repo root where
# train_openfold.py is -- and site-packages holds a DIFFERENT, older `openfold` that would win.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfold.config import model_config
from openfold.utils.import_weights import import_openfold_weights_
from train_openfold import OpenFoldWrapper, select_ema_warmstart_weights

PRE = "model.recycling_embedder.contractive_pair_update."

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--delta_floor", type=float, required=True)
ap.add_argument("--config_preset", default="finetuning_ptm")
args = ap.parse_args()

# --- the flags the live launcher sets, plus the two new ones ---------------------------
config = model_config(args.config_preset, train=True, low_prec=True)
config.model.recycling_embedder.use_contractive = True
config.model.recycling_embedder.use_gaussian_pair_init = True
config.model.recycling_embedder.per_position_delta = True
config.model.recycling_embedder.delta_floor = args.delta_floor
config.data.common.max_recycling_iters = 3

model_module = OpenFoldWrapper(config)
n_model = sum(1 for _ in model_module.state_dict())
print(f"model built: {n_model} state_dict entries, "
      f"{sum(p.numel() for p in model_module.parameters())/1e6:.1f}M params")

# --- gate 1: EMA, not state_dict -------------------------------------------------------
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ema_sd = select_ema_warmstart_weights(sd, resume_from_ema=True, ckpt_path=args.ckpt)
assert ema_sd is not None and len(ema_sd) > 1000, len(ema_sd) if ema_sd else None
print(f"\n[1] EMA weights selected: {len(ema_sd)} tensors (state_dict has "
      f"{len(sd['state_dict'])}) -- loading EMA, per reference_offline_eval_needs_ema_weights")

want_delta = F.softplus(ema_sd[PRE + "log_delta"].float().clone())
want_other = {k: v.clone() for k, v in ema_sd.items()
              if k.startswith(PRE) and not k.endswith("log_delta")}

# --- gate 2: the real loader, real strictness (prune_evoformer => strict=False) --------
import_openfold_weights_(model=model_module, state_dict=ema_sd, strict=False)
live = model_module.state_dict()
missing = sorted(set(live) - set(ema_sd))
unexpected = sorted(set(ema_sd) - set(live))
print(f"\n[2] missing={len(missing)} unexpected={len(unexpected)}")
print(f"    missing keys: {missing}")
assert not unexpected, f"checkpoint tensors that found no home: {unexpected[:10]}"
expected_missing = {PRE + "delta_floor", PRE + "delta_head.weight", PRE + "delta_head.bias"}
extra = set(missing) - expected_missing
assert not extra, f"unexpected missing keys beyond the new head: {sorted(extra)}"
assert expected_missing <= set(missing), sorted(expected_missing - set(missing))

# --- gate 3: effective delta preserved -------------------------------------------------
cpu = model_module.model.recycling_embedder.contractive_pair_update
c_z = cpu.c_z
got = cpu.per_position_delta_from_state(torch.zeros(1, 2, 2, c_z))[0, 0, 0]
err = float((got - want_delta).abs().max())
print(f"\n[3] delta: ckpt mean {float(want_delta.mean()):.6f} -> loaded mean "
      f"{float(got.mean()):.6f}   max abs err {err:.3e}")
assert err < 1e-6, "floor migration did not preserve the checkpoint's delta"
assert float(cpu.delta_floor) == args.delta_floor, float(cpu.delta_floor)
print(f"    a_bar mean {float(torch.exp(-got * torch.exp(cpu.log_a.detach())).mean()):.6f}")

# --- gate 4: nothing else moved --------------------------------------------------------
bad = [k for k, v in want_other.items() if not torch.equal(live[k].float(), v.float())]
print(f"\n[4] other contractive tensors bit-identical: {len(want_other) - len(bad)}"
      f"/{len(want_other)}" + (f"  MISMATCHED: {bad}" if bad else ""))
assert not bad, bad
print("\nWARM-START GATE PASSED")
