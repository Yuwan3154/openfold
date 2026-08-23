"""Did the EMA warm start actually land? Reports missing/unexpected keys and tensor equality.

⛔ Run C logged `resume_from_ema: loaded the EMA weights (5051 tensors)` -- that is len() of what was
PASSED IN, not what landed. With strict=False (forced by single_seq_mode) a mismatched key is silently
dropped, so that message cannot distinguish a perfect load from a total no-op. This can.
"""
import sys

import torch

sys.path.insert(0, "/home/jupyter-chenxi/openfold-esmfold2-recycling")

from openfold.config import model_config  # noqa: E402
from train_openfold import OpenFoldWrapper, select_ema_warmstart_weights  # noqa: E402

CKPT = ("/home/jupyter-chenxi/runs/runB_full_stack_pda_eval/lightning_logs/"
        "version_1/checkpoints/best-010-008250.ckpt")

# exactly Run C's config path (train_openfold.py:958-1020), precision bf16 -> low_prec True
config = model_config("finetuning_ptm", train=True, low_prec=True)
config.data.common.max_extra_msa = 1
config.data.common.max_msa_clusters = 1
config.data.train.max_extra_msa = 1
config.data.train.max_msa_clusters = 1
# --single_seq_keep_templates: templates stay ENABLED
config.loss.masked_msa.weight = 0.0
config.data.train.crop_size = min(config.data.train.crop_size, 256)
config.model.recycling_embedder.use_contractive = True
config.model.recycling_embedder.use_gaussian_pair_init = True
config.model.recycling_embedder.gaussian_pair_init_scale = 1.0

mm = OpenFoldWrapper(config, replace_block_index=None, replacement_hidden_dim=None,
                     learning_rate=1e-4)
model_keys = set(mm.state_dict().keys())
print(f"model tensors                : {len(model_keys)}")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
ema_sd = select_ema_warmstart_weights(ck, True, CKPT)
live_sd = ck["state_dict"]
print(f"EMA dict passed to the load  : {len(ema_sd)}")
print(f"state_dict in the ckpt       : {len(live_sd)}")

for name, sd in (("EMA", ema_sd), ("state_dict", live_sd)):
    missing = sorted(model_keys - set(sd))
    unexpected = sorted(set(sd) - model_keys)
    print(f"\n=== {name} vs model ===")
    print(f"  missing (in model, not in ckpt)   : {len(missing)}")
    print(f"  unexpected (in ckpt, not in model): {len(unexpected)}")
    for k in missing[:8]:
        print(f"     MISSING    {k}  {tuple(mm.state_dict()[k].shape)}")
    for k in unexpected[:8]:
        print(f"     UNEXPECTED {k}  {tuple(sd[k].shape)}")
    shape_mm = [(k, tuple(sd[k].shape), tuple(mm.state_dict()[k].shape))
                for k in sorted(model_keys & set(sd))
                if hasattr(sd[k], "shape") and sd[k].shape != mm.state_dict()[k].shape]
    print(f"  SHAPE mismatches on shared keys   : {len(shape_mm)}")
    for k, a, b in shape_mm[:8]:
        print(f"     SHAPE      {k}  ckpt{a} vs model{b}")

# the real load, through the real code path
print("\n=== running the ACTUAL load path (import_openfold_weights_) ===")
before = {k: v.detach().clone() for k, v in mm.state_dict().items() if v.is_floating_point()}
from openfold.utils.import_weights import import_openfold_weights_  # noqa: E402
import_openfold_weights_(model=mm, state_dict=ema_sd, strict=False)
after = mm.state_dict()

unchanged, changed, equal_to_ema = 0, 0, 0
never_touched = []
for k, v0 in before.items():
    v1 = after[k]
    if torch.equal(v0, v1):
        unchanged += 1
        never_touched.append(k)
    else:
        changed += 1
    src = ema_sd.get(k)
    if src is not None and src.shape == v1.shape and torch.equal(v1.float(), src.float()):
        equal_to_ema += 1

print(f"  float tensors                     : {len(before)}")
print(f"  CHANGED by the load               : {changed}")
print(f"  unchanged (init == ckpt, or DROPPED): {unchanged}")
print(f"  EXACTLY EQUAL to the EMA value    : {equal_to_ema}  <<< the number that matters")
print(f"\n  first unchanged keys: {never_touched[:10]}")

b = after["model.recycling_embedder.contractive_pair_update.b"]
print(f"\n  contractive b shape after load: {tuple(b.shape)}")
print(f"  b diagonal[:6]  : {torch.diagonal(b)[:6].tolist()}")
print(f"  ema b vector[:6]: {ema_sd['model.recycling_embedder.contractive_pair_update.b'][:6].tolist()}")
off = b - torch.diag(torch.diagonal(b))
print(f"  max |off-diagonal| : {off.abs().max().item():.3e}  (0 => diag(b), lossless migration)")
