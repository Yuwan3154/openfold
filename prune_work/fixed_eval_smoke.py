"""Fixed-set before/after structure-stability probe for ONE conf-weight M (user test 2).

The per-step train-loss metric is a data-order artifact (M=0.1 vs M=30 trajectories correlate 0.9998);
the smoke's val uses EMA which barely moves in 100 steps. So here we eval the RAW trained weights on a
FIXED held-out set before and after N steps of conf+structure training at conf-weight M:
  before = eval(init) on K fixed val proteins        [struct = fape+distogram+masked_msa+supervised_chi+violation; conf = plddt+exp_resolved+tm @ AF2 M=1]
  train N steps at conf-weight M (evoformer-only, frozen conf heads, Adam lr, grad-clip 0.1, bf16 autocast)
  after  = eval(raw weights) on the SAME K fixed proteins
struct_delta>0 = structure DEGRADED; conf_delta<0 = confidence loss IMPROVED. Run one M per GPU (4-way).
"""
import argparse
import copy
import os

import torch
import torch.nn as nn

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.data.data_modules import OpenFoldDataModule

KEEP = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47]
STRUCT = ("fape", "distogram", "masked_msa", "supervised_chi", "violation")
CONF = ("plddt_loss", "experimentally_resolved", "tm")
CONF_W = {"plddt_loss": 0.01, "experimentally_resolved": 0.01, "tm": 0.1}  # AF2 ratios @ M=1

MM = "/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files"
ALN = "/home/jupyter-chenxi/data/openproteinset_aln"
KAL = "/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign"
OBS = "/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat"
CACHE = "/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json"

ap = argparse.ArgumentParser()
ap.add_argument("--mult", type=float, required=True)
ap.add_argument("--init", default="/home/jupyter-chenxi/runs/conf_smoke_init.ckpt")
ap.add_argument("--train_list", default="/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_train.list")
ap.add_argument("--eval_list", default="/home/jupyter-chenxi/prune_work/lists_pdb/fixed_eval.list")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--k_eval", type=int, default=16)
args = ap.parse_args()

cfg = model_config("finetuning_ptm")
cfg.data.train.crop_size = args.crop


def build_model():
    m = AlphaFold(cfg)
    m.evoformer.blocks = nn.ModuleList([m.evoformer.blocks[i] for i in KEEP])
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    m.load_state_dict(sd, strict=False)
    return m.cuda().float()


def mk_loss(weights):
    c = copy.deepcopy(cfg.loss)
    for k in STRUCT + CONF:
        if k in c and "weight" in c[k]:
            c[k].weight = float(weights.get(k, 0.0))
    return AlphaFoldLoss(c)


w_struct = {k: cfg.loss[k].weight for k in STRUCT if k in cfg.loss}
L_struct = mk_loss(w_struct)
L_conf = mk_loss(CONF_W)
L_train = mk_loss({**w_struct, **{k: CONF_W[k] * args.mult for k in CONF}})  # struct + conf*M


def to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cuda(v) for v in x)
    return x


def strip(batch):
    batch = to_cuda({k: v for k, v in batch.items() if v is not None})
    return batch


dm = OpenFoldDataModule(
    config=cfg.data, template_mmcif_dir=MM, max_template_date="2018-04-30",
    train_data_dir=MM, train_alignment_dir=ALN, train_chain_list_path=args.train_list,
    val_data_dir=MM, val_alignment_dir=ALN, val_chain_list_path=args.eval_list,
    kalign_binary_path=KAL, obsolete_pdbs_file_path=OBS, template_release_dates_cache_path=CACHE,
    batch_seed=42, train_epoch_len=args.steps + 5)
dm.prepare_data()
dm.setup()


@torch.no_grad()
def evaluate(model):
    model.eval()
    sc, cf, n = 0.0, 0.0, 0
    for j, batch in enumerate(dm.val_dataloader()):
        if j >= args.k_eval:
            break
        batch = strip(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            b = tensor_tree_map(lambda t: t[..., -1], batch)
            sc += float(L_struct(out, b)); cf += float(L_conf(out, b))
        n += 1
        del out
        torch.cuda.empty_cache()
    return sc / n, cf / n


model = build_model()
for p in model.parameters():
    p.requires_grad_(False)
for p in model.evoformer.parameters():
    p.requires_grad_(True)

s_before, c_before = evaluate(model)
print(f"M={args.mult} before: struct={s_before:.4f} conf={c_before:.4f}", flush=True)

model.train()
model.template_embedder.eval()
opt = torch.optim.Adam([p for p in model.evoformer.parameters() if p.requires_grad], lr=args.lr, eps=1e-5)
it = iter(dm.train_dataloader())
for step in range(args.steps):
    batch = strip(next(it))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
        b = tensor_tree_map(lambda t: t[..., -1], batch)
        loss = L_train(out, b)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.evoformer.parameters(), 0.1)
    opt.step()
    if step % 20 == 0:
        print(f"M={args.mult} step {step} train_loss={float(loss):.3f}", flush=True)
    del out, loss

s_after, c_after = evaluate(model)
print(f"M={args.mult} after: struct={s_after:.4f} conf={c_after:.4f}", flush=True)
print(f"RESULT M={args.mult} struct_before={s_before:.4f} struct_after={s_after:.4f} struct_delta={s_after-s_before:+.4f} "
      f"conf_before={c_before:.4f} conf_after={c_after:.4f} conf_delta={c_after-c_before:+.4f}", flush=True)
