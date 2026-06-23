"""PHASE 1 training: gradient-match the student on the LARGE REAL dataset (grad_cache, native
near-one-hot seqs) with EFFECTIVE BATCH 32 via grad accumulation -> test user hypothesis that
small data+batch caused the heldout~0 failure. Config via env. Metric = held-out input-grad cosine."""
import os, json, glob, math, random
STUDENT = os.environ.get("STUDENT", "converged")   # converged | converged2 | fresh | pruned
BATCH = int(os.environ.get("BATCH", "32"))
LR = float(os.environ.get("LR", "2e-4"))
STEPS = int(os.environ.get("STEPS", "150"))         # optimizer steps (each = BATCH chains)
EVO_ONLY = os.environ.get("EVO_ONLY", "0") == "1"
MAXLEN = int(os.environ.get("MAXLEN", "150"))       # cap train/val chain length (double-backprop mem, no ckpt)
VAL_N = int(os.environ.get("VAL_N", "200"))
TAG = os.environ.get("TAG", "run")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.np import residue_constants as rc

BASE = "/home/jupyter-chenxi"
JAX = f"{BASE}/params/params_model_1_ptm.npz"
CK = f"{BASE}/runs/slim_v1/lightning_logs/version_0/checkpoints/best-017-004500.ckpt"
CK2 = f"{BASE}/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt"  # WS4: converged STRUCTURE slim (val lDDT-Ca 0.889)
CDC = f"{BASE}/data/pdb_mmcif/chain_data_cache.json"
GC = f"{BASE}/data/grad_cache"
KEEP = [int(x) for x in "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47".split(",")]
DEV = "cuda:0"


def build_cfg():
    cfg = model_config("finetuning_ptm", train=True, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = 0
    return cfg


def build_student():
    m = AlphaFold(build_cfg())
    if STUDENT == "pruned":
        import_jax_weights_(m, JAX, version="model_1_ptm")
        from openfold.block_replacement_scripts.pruned_evoformer import prune_blocks
        prune_blocks(m.evoformer)   # within-block: drop col+tri-att, keep all 48 blocks (gradient-structure preserving)
    else:
        b = m.evoformer.blocks
        m.evoformer.blocks = nn.ModuleList([b[i] for i in KEEP])
        if STUDENT in ("converged", "converged2"):
            ck_path = CK2 if STUDENT == "converged2" else CK
            sd = torch.load(ck_path, map_location="cpu", weights_only=False)["state_dict"]
            m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
        else:
            import_jax_weights_(m, JAX, version="model_1_ptm")
    m.evoformer.blocks_per_ckpt = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(m.globals, g, False)
    return m.to(DEV).eval()


def fwd_S(model, logits):
    L = logits.shape[1]
    soft = F.softmax(logits, -1)
    z1 = soft.new_zeros(1, L, 1)
    s21 = torch.cat([soft, z1], -1); s23 = torch.cat([soft, z1, z1, z1], -1)
    tf = torch.cat([soft.new_zeros(1, L, 1), s21], -1)
    zc = soft.new_zeros(1, 1, L, 1)
    msa = torch.cat([s23.unsqueeze(1), zc, zc, s23.unsqueeze(1), zc], -1)
    ri = torch.arange(L, device=DEV)[None]
    sm = soft.new_ones(1, L); mm = soft.new_ones(1, 1, L)
    m, z = model.input_embedder(tf, ri, msa, inplace_safe=False)
    pm = sm[..., None] * sm[..., None, :]
    m, z, s = model.evoformer(m, z, msa_mask=mm, pair_mask=pm, outputs={}, cycle_no=0, chunk_size=None,
                              use_deepspeed_evo_attention=False, use_cuequivariance_attention=False,
                              use_cuequivariance_multiplicative_update=False, use_lma=False, use_flash=False,
                              inplace_safe=False, _mask_trans=True)
    dg = model.aux_heads.distogram(z); tm = model.aux_heads.tm(z)
    nb = dg.shape[-1]
    S_dist = torch.softmax(dg, -1)[..., : nb // 2].sum(-1).mean()
    centers = torch.linspace(0.0, 31.0, tm.shape[-1], device=DEV)
    pae = (torch.softmax(tm, -1) * centers).sum(-1).mean()
    return S_dist - pae


cdc = json.load(open(CDC))


def x_of(chain, scale):
    seq = cdc[chain]["seq"]
    aat = torch.tensor([rc.restype_order.get(a, rc.restype_num) for a in seq])
    oh = F.one_hot(aat.clamp(max=19), 20).float()
    oh[aat >= 20] = 0.0
    return (scale * oh).unsqueeze(0).to(DEV)


# dataset = cached chains within length cap
cached = []
for p in glob.glob(f"{GC}/*.pt"):
    c = os.path.basename(p)[:-3]
    if c in cdc and len(cdc[c]["seq"]) <= MAXLEN:
        cached.append(c)
cached.sort()
random.Random(0).shuffle(cached)
val_ids = cached[:VAL_N]
train_ids = cached[VAL_N:]
print(f"TAG={TAG} STUDENT={STUDENT} BATCH={BATCH} LR={LR} EVO_ONLY={EVO_ONLY} MAXLEN={MAXLEN} | train={len(train_ids)} val={len(val_ids)}", flush=True)


def load_g(chain):
    d = torch.load(f"{GC}/{chain}.pt", map_location=DEV)
    return d["g"].float(), float(d["scale"])


# preload val gradients + scale (val small)
val = [(c, *load_g(c)) for c in val_ids]
tr_sample_ids = train_ids[:60]


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), 0))


@torch.no_grad()
def _noop():
    pass


def igrad(model, x):
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad(fwd_S(model, x), x)[0]


def val_cos(model):
    tot = 0.0
    for c, g, sc in val:
        tot += cos(igrad(model, x_of(c, sc)), g)
    return tot / len(val)


def train_cos(model):
    tot = 0.0
    for c in tr_sample_ids:
        g, sc = load_g(c)
        tot += cos(igrad(model, x_of(c, sc)), g)
    return tot / len(tr_sample_ids)


student = build_student()
if EVO_ONLY:
    for p in student.input_embedder.parameters(): p.requires_grad_(False)
    for p in student.aux_heads.parameters(): p.requires_grad_(False)
    params = [p for p in student.evoformer.parameters() if p.requires_grad]
else:
    params = list(student.parameters())
opt = torch.optim.Adam(params, lr=LR)
print(f"baseline val_cos={val_cos(student):.4f}", flush=True)

WARM = 15
order = train_ids[:]
ptr = 0
best = -2.0
for step in range(1, STEPS + 1):
    lr = LR * (step / WARM if step < WARM else 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM))))
    for pg in opt.param_groups: pg["lr"] = lr
    opt.zero_grad()
    for _ in range(BATCH):
        if ptr >= len(order):
            random.Random(step).shuffle(order); ptr = 0
        c = order[ptr]; ptr += 1
        gt, sc = load_g(c)
        x = x_of(c, sc).detach().requires_grad_(True)
        g = torch.autograd.grad(fwd_S(student, x), x, create_graph=True)[0]
        ((1.0 - F.cosine_similarity(g.flatten(), gt.flatten(), 0)) / BATCH).backward()
    torch.nn.utils.clip_grad_norm_(params, 100.0)
    opt.step()
    if step % 20 == 0:
        vc = val_cos(student)
        if vc > best: best = vc
        print(f"step {step:3d} lr={lr:.1e} train_cos={train_cos(student):.4f} val_cos={vc:.4f} best={best:.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print(f"FINAL best val_cos={best:.4f}", flush=True)
print("DONE")
