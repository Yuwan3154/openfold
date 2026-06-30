"""WS4-BC smoke: full-48 teacher (model_1_ptm), single-seq + NO templates, recycle=1, dropout off.
Build a differentiable soft-seq batch via the WS3 make_feature_batch, run the FULL model (incl IPA),
compute the BindCraft monomer design loss (con+pae+plddt+rg-helix) via the faithful port, and take
the input gradient. Verifies the whole forward+loss+grad path on one chain before caching."""
import os, sys, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.np import residue_constants as rc

sys.path.insert(0, "/home/jupyter-chenxi/openfold/openfold/block_replacement_scripts")
from hallucination_straight_through import make_feature_batch
sys.path.insert(0, "/home/jupyter-chenxi/prune_work")
import ws4_bc_losses as bcl

BASE = "/home/jupyter-chenxi"
JAX = f"{BASE}/params/params_model_1_ptm.npz"
CDC = f"{BASE}/data/pdb_mmcif/chain_data_cache.json"
DEV = "cuda:0"
SCALE = 3.0
RECYCLE = int(os.environ.get("RECYCLE", "1"))

cfg = model_config("finetuning_ptm", train=False, low_prec=False)
cfg.globals.chunk_size = None
cfg.globals.blocks_per_ckpt = None   # disable grad checkpointing (incompatible w/ autograd.grad(inputs=) + double-backprop)
for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
    setattr(cfg.globals, g, False)
cfg.data.common.max_recycling_iters = RECYCLE
cfg.model.template.enabled = False
cfg.data.common.use_templates = False
cfg.data.common.use_template_torsion_angles = False
model = AlphaFold(cfg)
import_jax_weights_(model, JAX, version="model_1_ptm")
model = model.to(DEV).eval()
# disable ALL activation checkpointing (reentrant checkpoint_fn w/o use_reentrant breaks autograd.grad/double-backprop):
# evoformer/template use blocks_per_ckpt; extra-MSA stack + blocks use a per-module `ckpt` flag (evoformer.py:944,1491).
for mod in model.modules():
    if hasattr(mod, "ckpt"):
        mod.ckpt = False
    if hasattr(mod, "blocks_per_ckpt"):
        mod.blocks_per_ckpt = None
for p in model.parameters():
    p.requires_grad_(False)

cdc = json.load(open(CDC))
chain = None
for k, v in cdc.items():
    if 60 <= len(v["seq"]) <= 110:
        chain = k
        break
seq = cdc[chain]["seq"]
Ln = len(seq)
aat = torch.tensor([rc.restype_order.get(a, rc.restype_num) for a in seq])
oh = F.one_hot(aat.clamp(max=19), 20).float()
oh[aat >= 20] = 0.0
seq_logits = (SCALE * oh).to(DEV).requires_grad_(True)        # [L,20] native near-one-hot
ri = torch.arange(Ln, device=DEV)
print(f"chain={chain} L={Ln} RECYCLE={RECYCLE} recycle_dim={RECYCLE+1}", flush=True)

batch = make_feature_batch(seq_logits, ri, recycle_dim=RECYCLE + 1)
print("batch keys:", sorted(batch.keys()), flush=True)
for k in ["target_feat", "msa_feat", "aatype"]:
    if k in batch:
        print(f"  {k}: {tuple(batch[k].shape)}", flush=True)

out = model(batch)
print("out keys (subset):", [k for k in ["distogram_logits", "tm_logits", "lddt_logits", "final_atom_positions"] if k in out], flush=True)
for k in ["distogram_logits", "tm_logits", "lddt_logits", "final_atom_positions"]:
    print(f"  {k}: {tuple(out[k].shape)}", flush=True)

dgl = out["distogram_logits"]; dgl = dgl[0] if dgl.dim() == 4 else dgl       # [L,L,64]
tml = out["tm_logits"]; tml = tml[0] if tml.dim() == 4 else tml             # [L,L,64]
lddt = out["lddt_logits"]; lddt = lddt[0] if lddt.dim() == 3 else lddt       # [L,50]
fap = out["final_atom_positions"]; fap = fap[0] if fap.dim() == 4 else fap   # [L,37,3]
ca = fap[:, rc.atom_order["CA"], :]                                          # [L,3]

losses = bcl.bc_losses({"distogram_logits": dgl, "tm_logits": tml, "lddt_logits": lddt}, ri, ca)
L = bcl.total_loss(losses)
print("losses:", {k: round(float(v), 5) for k, v in losses.items()}, flush=True)
print(f"weighted total L = {float(L):.5f}", flush=True)

g = torch.autograd.grad(L, seq_logits, create_graph=True)[0]                  # [L,20] dL/dseq, graph retained for 2nd order
print(f"grad: shape={tuple(g.shape)} finite={bool(torch.isfinite(g).all())} ||g||={float(g.norm()):.4e} max|g|={float(g.abs().max()):.4e}", flush=True)
g2 = torch.autograd.grad(g.pow(2).sum(), seq_logits)[0]                       # 2nd-order wrt input: confirms double-backprop works (student path)
print(f"2nd-order (double-backprop): finite={bool(torch.isfinite(g2).all())} ||g2||={float(g2.norm()):.4e}", flush=True)
print("DONE")
