"""Gradient-coordination probe for the conf+structure joint training (user-requested diagnostic).

For the slim 24-block model (slim LAST weights, evoformer-only trainable, frozen conf heads), on a few
pre-cutoff training examples, compute from a SINGLE forward (same dropout realization):
  - g_struct = d(structure loss)/d(evoformer params)   [fape, distogram, masked_msa, supervised_chi, violation]
  - g_conf   = d(confidence loss)/d(evoformer params)   [plddt_loss, experimentally_resolved, tm @ AF2 M=1 weights]
and report |g_struct|, |g_conf|, ratio, cosine(g_struct, g_conf), plus a per-conf-component breakdown
(each at weight 1.0) to spot magnitude/direction irregularity or NaN.
fp32 for a clean gradient (the smoke trains bf16; direction/ratio are precision-robust).
"""
import argparse
import copy
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.data.data_modules import OpenFoldDataModule

KEEP = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47]
STRUCT = ("fape", "distogram", "masked_msa", "supervised_chi", "violation")
CONF = ("plddt_loss", "experimentally_resolved", "tm")
CONF_W = {"plddt_loss": 0.01, "experimentally_resolved": 0.01, "tm": 0.1}  # AF2 ratios @ M=1
ALLW = STRUCT + CONF

MM = "/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files"
ALN = "/home/jupyter-chenxi/data/openproteinset_aln"
KAL = "/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign"
OBS = "/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat"
CACHE = "/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json"

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="/home/jupyter-chenxi/runs/conf_smoke_init.ckpt")
ap.add_argument("--list", default="/home/jupyter-chenxi/prune_work/lists_pdb/grad_probe.list")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--crop", type=int, default=128)
args = ap.parse_args()

cfg = model_config("finetuning_ptm")
cfg.data.train.crop_size = args.crop

m = AlphaFold(cfg)
m.evoformer.blocks = nn.ModuleList([m.evoformer.blocks[i] for i in KEEP])
ck = torch.load(args.init, map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
miss, unexp = m.load_state_dict(sd, strict=False)
print(f"loaded init {args.init}: missing={len(miss)} unexpected={len(unexp)}", flush=True)
m = m.cuda().float().eval()  # eval: deterministic (separate-forward-safe), template assert ok; small crop keeps memory low
for p in m.parameters():
    p.requires_grad_(False)
for p in m.evoformer.parameters():
    p.requires_grad_(True)
params = [p for p in m.evoformer.parameters() if p.requires_grad]
print(f"evoformer trainable params: {sum(p.numel() for p in params)/1e6:.1f}M tensors={len(params)}", flush=True)

dm = OpenFoldDataModule(
    config=cfg.data, template_mmcif_dir=MM, max_template_date="2018-04-30",
    train_data_dir=MM, train_alignment_dir=ALN, train_chain_list_path=args.list,
    kalign_binary_path=KAL, obsolete_pdbs_file_path=OBS, template_release_dates_cache_path=CACHE,
    batch_seed=42, train_epoch_len=max(args.n * 2, 8))
dm.prepare_data()
dm.setup()
loader = dm.train_dataloader()


def mk_loss(weights):
    c = copy.deepcopy(cfg.loss)
    for k in ALLW:
        if k in c and "weight" in c[k]:
            c[k].weight = float(weights.get(k, 0.0))
    return AlphaFoldLoss(c)


w_struct = {k: cfg.loss[k].weight for k in STRUCT if k in cfg.loss}  # AF2 structure weights
L_struct = mk_loss(w_struct)
L_conf = mk_loss(CONF_W)
L_each = {k: mk_loss({k: 1.0}) for k in CONF}  # unweighted per-component direction


def to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cuda(v) for v in x)
    return x


def gvec(L, out, b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        val = L(out, b)
    g = torch.autograd.grad(val, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten().float()
                      for gi, p in zip(g, params)])
    return float(val.detach()), flat


print(f"\nstruct weights: { {k: round(v,3) for k,v in w_struct.items()} }  conf weights(M=1): {CONF_W}\n", flush=True)
rows = []
for i, batch in enumerate(loader):
    if i >= args.n:
        break
    batch = to_cuda(batch)
    none_keys = [k for k, v in batch.items() if v is None]
    batch = {k: v for k, v in batch.items() if v is not None}
    if i == 0:
        print(f"batch: {len(batch)} keys; dropped None: {none_keys}", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(batch)
    b = tensor_tree_map(lambda t: t[..., -1], batch)
    _, gs = gvec(L_struct, out, b)
    _, gc = gvec(L_conf, out, b)
    ns, nc = gs.norm().item(), gc.norm().item()
    cos = float(F.cosine_similarity(gs, gc, dim=0))
    length = int(b["seq_length"].float().mean().item())
    print(f"ex{i} len~{length}: |g_struct|={ns:.3e} |g_conf|={nc:.3e} ratio(c/s)={nc/ns:.3e} "
          f"cos(struct,conf)={cos:+.4f}  M_for_parity={ns/nc:.1f}", flush=True)
    for k, lf in L_each.items():
        _, gk = gvec(lf, out, b)
        print(f"     {k:24s} |g|={gk.norm().item():.3e} cos(struct,*)={float(F.cosine_similarity(gs,gk,dim=0)):+.4f} "
              f"nan={bool(torch.isnan(gk).any())}", flush=True)
    rows.append((ns, nc, cos, nc / ns))
    del out, gs, gc
    torch.cuda.empty_cache()

if rows:
    print("\n=== SUMMARY (mean over examples) ===")
    print(f"|g_struct|={statistics.mean(r[0] for r in rows):.3e}  |g_conf|={statistics.mean(r[1] for r in rows):.3e}  "
          f"ratio(c/s)={statistics.mean(r[3] for r in rows):.3e}  cos(struct,conf)={statistics.mean(r[2] for r in rows):+.4f}")
