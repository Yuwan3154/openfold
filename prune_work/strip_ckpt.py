"""Strip a Lightning checkpoint to the weights the SWEEP must use.

⛔⛔ EMA, NOT state_dict. train_openfold.py:487-494 swaps in the EMA params at the start of every
validation epoch and restores the live weights afterwards, so every val/lddt_ca number this project
has ever reported came from the EMA weights. A sweep loading `state_dict` would be measuring a
different model than the one the curves describe.

⭐ The EMA params are keyed on the AlphaFold module directly ("input_embedder...."), whereas
`state_dict` keys carry the LightningModule's "model." prefix -- so these load into AlphaFold as-is.
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
params = ck["ema"]["params"]
n = sum(v.numel() for v in params.values() if torch.is_tensor(v))
print(f"{src}\n  epoch={ck.get('epoch')} global_step={ck.get('global_step')}  "
      f"{len(params)} tensors, {n/1e6:.1f}M params")
print(f"  first keys: {list(params)[:3]}")
print(f"  has contractive: {any('contractive' in k for k in params)}")
torch.save({"ema_params": params, "epoch": ck.get("epoch"),
            "global_step": ck.get("global_step"), "src": src}, dst)
import os
print(f"  wrote {dst}  {os.path.getsize(dst)/1e6:.0f} MB "
      f"(was {os.path.getsize(src)/1e6:.0f} MB)")
