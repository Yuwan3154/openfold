"""WS3: monomer hallucination by gradient descent on the input sequence, against a TARGET structure
with a DISTOGRAM loss (IPA-free; only the pair rep z is needed). Frozen model, only seq logits trainable.
Refactor of cache_grads.fwd_S to return distogram logits + an Adam loop with a soft->hard temperature ramp.

Usage: python hallucinate_distogram.py TARGET_PDB [CHAIN=A]
Env: MODEL=slim|full (default slim) ; SLIM_CKPT=<ckpt> ; ITERS=400 ; LR=0.1 ; INIT=randn|zeros|native ;
     SOFT_FRAC=0.8 (fraction of iters at temp=1 before annealing) ; HARD_FRAC_START=0.9 (frac to switch to one-hot)."""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.feats import pseudo_beta_fn
from openfold.np import protein, residue_constants as rc

TARGET_PDB = sys.argv[1]
CHAIN = sys.argv[2] if len(sys.argv) > 2 else "A"
MODEL = os.environ.get("MODEL", "slim")
BASE = "/home/jupyter-chenxi"
JAX = f"{BASE}/params/params_model_1_ptm.npz"
SLIM_CKPT = os.environ.get("SLIM_CKPT",
    f"{BASE}/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt")
KEEP = [int(x) for x in os.environ.get("KEEP",
    "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47").split(",")]
ITERS = int(os.environ.get("ITERS", 400))
LR = float(os.environ.get("LR", 0.1))
INIT = os.environ.get("INIT", "randn")          # randn=from-scratch | zeros | native=redesign
SOFT_FRAC = float(os.environ.get("SOFT_FRAC", 0.8))
HARD_FRAC_START = float(os.environ.get("HARD_FRAC_START", 0.9))
DEV = "cuda:0"
NB = 64  # distogram bins

# ---- frozen model (slim 24-block by default, or full-48 AF2) ----
cfg = model_config("finetuning_ptm", train=True, low_prec=False)
cfg.globals.chunk_size = None
for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
    setattr(cfg.globals, g, False)
cfg.data.common.max_recycling_iters = 0
model = AlphaFold(cfg)
if MODEL == "slim":
    b = model.evoformer.blocks
    model.evoformer.blocks = nn.ModuleList([b[i] for i in KEEP])
    ck = torch.load(SLIM_CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[slim] loaded {SLIM_CKPT} | missing={len(miss)} unexpected={len(unexp)}", flush=True)
else:
    import_jax_weights_(model, JAX, version="model_1_ptm")
    print("[full] loaded jax model_1_ptm", flush=True)
model = model.to(DEV).eval()
model.evoformer.blocks_per_ckpt = 1
for p in model.parameters():
    p.requires_grad_(False)


def fwd_distogram(logits, temp=1.0, hard_frac=0.0):
    """Differentiable soft-seq -> IPA-free forward -> distogram logits [1,L,L,NB]."""
    L = logits.shape[1]
    soft = F.softmax(logits / temp, -1)
    if hard_frac > 0.0:
        hard = F.one_hot(soft.argmax(-1), 20).float()
        hard = (hard - soft).detach() + soft               # straight-through
        seq = (1.0 - hard_frac) * soft + hard_frac * hard
    else:
        seq = soft
    z1 = seq.new_zeros(1, L, 1)
    s21 = torch.cat([seq, z1], -1); s23 = torch.cat([seq, z1, z1, z1], -1)
    tf = torch.cat([seq.new_zeros(1, L, 1), s21], -1)
    zc = seq.new_zeros(1, 1, L, 1)
    msa = torch.cat([s23.unsqueeze(1), zc, zc, s23.unsqueeze(1), zc], -1)
    ri = torch.arange(L, device=DEV)[None]
    sm = seq.new_ones(1, L); mm = seq.new_ones(1, 1, L)
    m, z = model.input_embedder(tf, ri, msa, inplace_safe=False)
    pm = sm[..., None] * sm[..., None, :]
    m, z, s = model.evoformer(m, z, msa_mask=mm, pair_mask=pm, outputs={}, cycle_no=0, chunk_size=None,
                              use_deepspeed_evo_attention=False, use_cuequivariance_attention=False,
                              use_cuequivariance_multiplicative_update=False, use_lma=False, use_flash=False,
                              inplace_safe=False, _mask_trans=True)
    return model.aux_heads.distogram(z)


# ---- target distogram from the PDB (constant) ----
prot = protein.from_pdb_string(open(TARGET_PDB).read(), chain_id=CHAIN)
atom37 = torch.tensor(prot.atom_positions, dtype=torch.float32)[None].to(DEV)
atom37_mask = torch.tensor(prot.atom_mask, dtype=torch.float32)[None].to(DEV)
native_aatype = torch.tensor(prot.aatype, dtype=torch.long)[None].to(DEV)
L = native_aatype.shape[1]
pb, pb_mask = pseudo_beta_fn(native_aatype, atom37, atom37_mask)
boundaries = torch.linspace(2.3125, 21.6875, NB - 1, device=DEV) ** 2
dists = ((pb[..., None, :] - pb[..., None, :, :]) ** 2).sum(-1, keepdim=True)
true_bins = (dists > boundaries).sum(-1)                       # [1,L,L] in {0..NB-1}
true_oh = F.one_hot(true_bins, NB).float().detach()
sq_mask = (pb_mask[..., None] * pb_mask[..., None, :]).detach()
print(f"target {TARGET_PDB}:{CHAIN} L={L} model={MODEL} iters={ITERS} lr={LR} init={INIT}", flush=True)

# ---- trainable sequence logits ----
if INIT == "native":
    seq_logits = nn.Parameter(3.0 * F.one_hot(native_aatype[0].clamp(max=19), 20).float()[None].to(DEV))
elif INIT == "zeros":
    seq_logits = nn.Parameter(torch.zeros(1, L, 20, device=DEV))
else:
    seq_logits = nn.Parameter(0.01 * torch.randn(1, L, 20, device=DEV))
opt = torch.optim.Adam([seq_logits], lr=LR)


def schedule(it):
    frac = it / max(1, ITERS - 1)
    if frac < SOFT_FRAC:
        temp = 1.0
    else:
        temp = max(0.01, 1.0 - (frac - SOFT_FRAC) / max(1e-6, (1.0 - SOFT_FRAC)) * (1.0 - 0.01))
    hard_frac = 1.0 if frac >= HARD_FRAC_START else 0.0
    return temp, hard_frac


for it in range(ITERS):
    temp, hard_frac = schedule(it)
    opt.zero_grad()
    dg = fwd_distogram(seq_logits, temp=temp, hard_frac=hard_frac)
    ce = -(true_oh * F.log_softmax(dg, -1)).sum(-1)           # [1,L,L]
    loss = (ce * sq_mask).sum() / (sq_mask.sum() + 1e-8)
    loss.backward()
    opt.step()
    if it % 25 == 0 or it == ITERS - 1:
        rec = (seq_logits.argmax(-1)[0] == native_aatype[0].clamp(max=19)).float().mean().item()
        print(f"it={it} loss={loss.item():.4f} temp={temp:.3f} hard={hard_frac:.0f} native_recovery={rec:.3f}", flush=True)

final_seq = "".join(rc.restypes[i] for i in seq_logits.argmax(-1)[0].tolist())
print("FINAL_SEQ:", final_seq, flush=True)
