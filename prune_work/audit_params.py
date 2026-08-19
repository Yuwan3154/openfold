"""Audit the LIVE run's saved parameters: finite, moving where they should, frozen where they should.

Three questions, none of which the training log answers:
  1. Are any saved tensors NaN/Inf? A run can look healthy in the loss curve while a slice of weights
     has already gone non-finite, because the loss is a masked mean.
  2. Are the TRAINABLE groups actually changing? A frozen-by-accident group produces no error and no
     log line -- it just silently never learns. (The contractive params are the live risk: they are
     brand new, randomly initialised, and live OUTSIDE model.evoformer, so an evoformer-only freeze
     would drop them if `freeze_all_except_evoformer` had not been patched to keep them.)
  3. Are the FROZEN groups exactly unchanged? Not "small" -- EXACTLY. Any movement means the freeze
     leaked, and an evoformer-only fine-tune quietly became something else.

⭐ Compares two checkpoints of the SAME run rather than checkpoint-vs-init, because the contractive
parameters have no jax counterpart: their init is random, so a checkpoint-vs-fresh-model diff would
report a meaningless "change" for exactly the group we most need to check.
"""

import argparse
import collections

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-a", required=True)
ap.add_argument("--ckpt-b", default=None, help="a LATER checkpoint of the same run")
a = ap.parse_args()


def load_sd(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    step = ck.get("global_step", "?")
    ema = None
    for k in ("ema", "callbacks"):
        if k in ck and isinstance(ck[k], dict) and "params" in str(ck[k])[:200]:
            ema = ck[k]
    return sd, step, ema


def group_of(name):
    n = name.replace("model.", "", 1)
    if n.startswith("aux_heads."):
        return ".".join(n.split(".")[:2])
    return n.split(".")[0]


sd_a, step_a, _ = load_sd(a.ckpt_a)
print(f"ckpt A: {a.ckpt_a}  global_step={step_a}  tensors={len(sd_a)}")

# ---------------------------------------------------------------- 1. finiteness
print("\n=== 1. FINITENESS of every saved tensor ===")
bad = []
stats = collections.defaultdict(lambda: [0, 0.0, 0.0])
for k, v in sd_a.items():
    if not torch.is_tensor(v) or not v.is_floating_point():
        continue
    vf = v.float()
    if not torch.isfinite(vf).all():
        n_nan = int(torch.isnan(vf).sum())
        n_inf = int(torch.isinf(vf).sum())
        bad.append((k, n_nan, n_inf, tuple(v.shape)))
    g = stats[group_of(k)]
    g[0] += v.numel()
    g[1] = max(g[1], float(vf.abs().max()))
    g[2] += float(vf.pow(2).sum())
if bad:
    print(f"  ⛔ {len(bad)} NON-FINITE tensors:")
    for k, nn, ni, sh in bad[:15]:
        print(f"     {k}  nan={nn} inf={ni} shape={sh}")
else:
    print(f"  ✅ all {sum(1 for v in sd_a.values() if torch.is_tensor(v) and v.is_floating_point())}"
          f" float tensors finite")

print("\n=== 2. PER-GROUP magnitude (max|w| and RMS) -- catches a group that blew up or collapsed ===")
for g in sorted(stats):
    n, mx, sq = stats[g]
    print(f"  {g:34s} n={n:>11,}  max|w|={mx:9.4f}  rms={(sq / max(n,1))**0.5:.6f}")

# ---------------------------------------------------------------- 3. movement
if a.ckpt_b:
    sd_b, step_b, _ = load_sd(a.ckpt_b)
    print(f"\nckpt B: {a.ckpt_b}  global_step={step_b}   (Δ = {int(step_b) - int(step_a)} steps)")
    print("\n=== 3. PARAMETER MOVEMENT between the two checkpoints ===")
    moved = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])   # sumsq_delta, sumsq_w, n, n_changed
    for k, va in sd_a.items():
        if k not in sd_b or not torch.is_tensor(va) or not va.is_floating_point():
            continue
        vb = sd_b[k]
        if vb.shape != va.shape:
            print(f"  ⛔ shape changed: {k} {tuple(va.shape)} -> {tuple(vb.shape)}")
            continue
        d = (vb.float() - va.float())
        m = moved[group_of(k)]
        m[0] += float(d.pow(2).sum())
        m[1] += float(va.float().pow(2).sum())
        m[2] += va.numel()
        m[3] += int((d != 0).sum())
    print(f"  {'group':34s} {'rel ‖Δ‖/‖w‖':>13} {'changed/total':>22}  verdict")
    for g in sorted(moved):
        sq_d, sq_w, n, nc = moved[g]
        rel = (sq_d ** 0.5) / max(sq_w ** 0.5, 1e-12)
        verdict = "MOVING" if nc else "frozen (bit-identical)"
        print(f"  {g:34s} {rel:13.3e} {nc:>10,}/{n:<11,}  {verdict}")
