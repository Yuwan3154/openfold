"""Load the LIVE Run C v2 checkpoint's contractive params into both module modes and verify:

  1. flag OFF reproduces the checkpoint's delta bit-for-bit (no floor migration fires)
  2. flag ON + floor 0.05 reproduces the SAME effective delta (migration fires and is correct)
  3. a verbatim (un-migrated) load would have shifted every channel by exactly the floor
"""
import argparse

import torch
import torch.nn.functional as F

from openfold.model.contractive_recycling import ContractivePairUpdate

FLOOR = 0.05
PRE = "recycling_embedder.contractive_pair_update."

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--c_z", type=int, default=128)
args = ap.parse_args()

ema = torch.load(args.ckpt, map_location="cpu", weights_only=False)["ema"]["params"]
sd = {k[len(PRE):]: v.float() for k, v in ema.items() if k.startswith(PRE)}
print("checkpoint contractive keys:", sorted(sd))

want = F.softplus(sd["log_delta"])
print(f"\ncheckpoint delta: min={float(want.min()):.6f} mean={float(want.mean()):.6f} "
      f"max={float(want.max()):.6f}")

off = ContractivePairUpdate(args.c_z)
print("\n[1] flag OFF ->", off.load_state_dict(dict(sd), strict=True))
got_off = F.softplus(off.log_delta.detach())
print(f"    delta identical to checkpoint: {torch.equal(got_off, want)}")

on = ContractivePairUpdate(args.c_z, per_position_delta=True, delta_floor=FLOOR)
miss, unexp = on.load_state_dict(dict(sd), strict=False)
print(f"\n[2] flag ON  -> missing={miss} unexpected={unexp}")
z = torch.zeros(1, 2, 2, args.c_z)                      # zero-init head => s == 0
got_on = on.per_position_delta_from_state(z)[0, 0, 0]
err = (got_on - want).abs().max()
print(f"    effective delta max abs err vs checkpoint: {float(err):.3e}")
assert float(err) < 1e-6, "floor migration did not preserve delta"
print(f"    a_bar preserved: max abs err "
      f"{float((torch.exp(-got_on * torch.exp(on.log_a.detach())) - torch.exp(-want * torch.exp(off.log_a.detach()))).abs().max()):.3e}")

naive = FLOOR + F.softplus(sd["log_delta"])
print(f"\n[3] un-migrated verbatim load would give delta mean {float(naive.mean()):.6f} "
      f"vs {float(want.mean()):.6f} -- a uniform +{float((naive - want).mean()):.5f} shift "
      f"({100 * float(((naive - want) / want).mean()):.2f}%)")
print("\nALL CHECKS PASSED")
