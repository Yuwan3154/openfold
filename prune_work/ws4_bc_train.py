"""WS4-BC student trainer: train the slim 24-block student's EVOFORMER to match the full-48 teacher's
BindCraft design-loss input-gradient (cached in grad_cache_bc) at BOTH the native and soft seq points.
Full model (single-seq, no template, recycle=1, IPA on, dropout off), double-backprop. Heads + input
embedder + structure module FROZEN; only evoformer trains. Matching loss = per-residue-normalized
cosine + BETA*log-norm magnitude (optional VAL_W value-anchor for stability).
Config via env: STUDENT STEPS BATCH LR MAXLEN VAL_N POINTS(native|soft|both) BETA VAL_W CLIP WARM EVAL_EVERY."""
import os, sys, json, glob, math, random, hashlib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.np import residue_constants as rc

sys.path.insert(0, "/home/jupyter-chenxi/openfold/openfold/block_replacement_scripts")
from hallucination_straight_through import make_feature_batch
sys.path.insert(0, "/home/jupyter-chenxi/prune_work")
import ws4_bc_losses as bcl

STUDENT = os.environ.get("STUDENT", "converged2")        # converged2 = best-037 slim-24
STEPS = int(os.environ.get("STEPS", "400"))
BATCH = int(os.environ.get("BATCH", "8"))                 # grad-accum chains per optimizer step
LR = float(os.environ.get("LR", "1e-4"))
MAXLEN = int(os.environ.get("MAXLEN", "128"))
VAL_N = int(os.environ.get("VAL_N", "40"))
POINTS = os.environ.get("POINTS", "both")                 # native | soft | both
BETA = float(os.environ.get("BETA", "0.1"))               # log-norm magnitude weight
VAL_W = float(os.environ.get("VAL_W", "0.0"))             # value-anchor (match teacher L); enable if collapse
CLIP = float(os.environ.get("CLIP", "1.0"))
WARM = int(os.environ.get("WARM", "30"))
SCHED = os.environ.get("SCHED", "cosine")
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "20"))
RECYCLE = int(os.environ.get("RECYCLE", "1"))
SCALE = float(os.environ.get("SCALE", "3.0"))
USE_CKPT = os.environ.get("USE_CKPT", "0") == "1"   # enable checkpointed double-backprop (memory for L>128; needs evoformer use_reentrant=False fix)
EPS = 1e-6
BASE = "/home/jupyter-chenxi"
JAX = f"{BASE}/params/params_model_1_ptm.npz"
CK2 = f"{BASE}/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt"
CDC = f"{BASE}/data/pdb_mmcif/chain_data_cache.json"
GC = f"{BASE}/data/grad_cache_bc"
KEEP = [int(x) for x in "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47".split(",")]
DEV = "cuda:0"


def build_cfg():
    cfg = model_config("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = RECYCLE
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    return cfg


def kill_ckpt(m):
    for mod in m.modules():
        if hasattr(mod, "ckpt"):
            mod.ckpt = False
        if hasattr(mod, "blocks_per_ckpt"):
            mod.blocks_per_ckpt = None


def enable_ckpt(m):
    # checkpoint every evoformer + extra-MSA block (max memory savings); needs the use_reentrant=False fix
    if hasattr(m.evoformer, "blocks_per_ckpt"):
        m.evoformer.blocks_per_ckpt = 1
    if hasattr(m, "extra_msa_stack"):
        if hasattr(m.extra_msa_stack, "ckpt"):
            m.extra_msa_stack.ckpt = True
        for b in getattr(m.extra_msa_stack, "blocks", []):
            if hasattr(b, "ckpt"):
                b.ckpt = True


def build_student():
    m = AlphaFold(build_cfg())
    b = m.evoformer.blocks
    m.evoformer.blocks = nn.ModuleList([b[i] for i in KEEP])
    sd = torch.load(CK2, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    m = m.to(DEV).eval()
    (enable_ckpt if USE_CKPT else kill_ckpt)(m)
    return m


cdc = json.load(open(CDC))


def native_logits(seq):
    aat = torch.tensor([rc.restype_order.get(a, rc.restype_num) for a in seq])
    oh = F.one_hot(aat.clamp(max=19), 20).float()
    oh[aat >= 20] = 0.0
    return (SCALE * oh).to(DEV)


def student_grad(model, base_logits, create_graph):
    Ln = base_logits.shape[0]
    sl = base_logits.detach().clone().requires_grad_(True)
    ri = torch.arange(Ln, device=DEV)
    batch = make_feature_batch(sl, ri, recycle_dim=RECYCLE + 1)
    out = model(batch)
    dgl = out["distogram_logits"]; dgl = dgl[0] if dgl.dim() == 4 else dgl
    tml = out["tm_logits"]; tml = tml[0] if tml.dim() == 4 else tml
    lddt = out["lddt_logits"]; lddt = lddt[0] if lddt.dim() == 3 else lddt
    fap = out["final_atom_positions"]; fap = fap[0] if fap.dim() == 4 else fap
    ca = fap[:, rc.atom_order["CA"], :]
    losses = bcl.bc_losses({"distogram_logits": dgl, "tm_logits": tml, "lddt_logits": lddt}, ri, ca)
    L = bcl.total_loss(losses)
    g = torch.autograd.grad(L, sl, create_graph=create_graph)[0]
    return g, L


# build dataset of (chain, point) items from the BindCraft cache
cached = []
for p in sorted(glob.glob(f"{GC}/*.pt")):
    c = os.path.basename(p)[:-3]
    if c in cdc and len(cdc[c]["seq"]) <= MAXLEN:
        cached.append(c)
random.Random(0).shuffle(cached)
val_chains = cached[:VAL_N]
train_chains = cached[VAL_N:]
pts = ["native", "soft"] if POINTS == "both" else [POINTS]
train_items = [(c, pt) for c in train_chains for pt in pts]
val_items = [(c, pt) for c in val_chains for pt in pts]
random.Random(1).shuffle(train_items)
print(f"STUDENT={STUDENT} cached={len(cached)} train_chains={len(train_chains)} val_chains={len(val_chains)} POINTS={POINTS} items(train/val)={len(train_items)}/{len(val_items)} BATCH={BATCH} LR={LR} BETA={BETA} VAL_W={VAL_W}", flush=True)


def teacher_for(c, pt):
    d = torch.load(f"{GC}/{c}.pt", map_location=DEV)
    if pt == "native":
        bl = native_logits(cdc[c]["seq"])
        return bl, d["g_native"].to(DEV), d["L_native"]
    else:
        return d["soft_logits"].to(DEV), d["g_soft"].to(DEV), d["L_soft"]


def perpos_cos(g_s, g_t):
    ns = g_s.norm(dim=-1); nt = g_t.norm(dim=-1); mask = nt > EPS
    c = ((g_s / (ns[:, None] + 1e-8)) * (g_t / (nt[:, None] + 1e-8))).sum(-1)
    return c[mask].mean()


@torch.no_grad()
def _noop():
    pass


def evaluate(model, items):
    accp = accf = 0.0
    for c, pt in items:
        bl, g_t, _ = teacher_for(c, pt)
        g_s, _ = student_grad(model, bl, create_graph=False)
        accp += float(perpos_cos(g_s.detach(), g_t))
        accf += float(F.cosine_similarity(g_s.detach().flatten(), g_t.flatten(), 0))
    return accp / len(items), accf / len(items)


student = build_student()
for p in student.input_embedder.parameters(): p.requires_grad_(False)
for p in student.aux_heads.parameters(): p.requires_grad_(False)
for p in student.structure_module.parameters(): p.requires_grad_(False)
params = [p for p in student.evoformer.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=LR, betas=(0.5, 0.9), weight_decay=1e-4)
b0 = evaluate(student, val_items)
print(f"baseline val: perpos={b0[0]:.4f} flat={b0[1]:.4f}", flush=True)

tr_sample = train_items[:VAL_N]
order = train_items[:]
ptr = 0
best = -2.0
for step in range(1, STEPS + 1):
    if step < WARM:
        lr = LR * step / WARM
    elif SCHED == "const":
        lr = LR
    else:
        lr = LR * 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM)))
    for pg in opt.param_groups: pg["lr"] = lr
    opt.zero_grad()
    for _ in range(BATCH):
        if ptr >= len(order):
            random.Random(step).shuffle(order); ptr = 0
        c, pt = order[ptr]; ptr += 1
        bl, g_t, Lt = teacher_for(c, pt)
        g_s, L_s = student_grad(student, bl, create_graph=True)
        ns = g_s.norm(dim=-1); nt = g_t.norm(dim=-1); mask = nt > EPS
        gs_n = g_s / (ns[:, None] + 1e-8); gt_n = g_t / (nt[:, None] + 1e-8)
        dir_loss = (1.0 - (gs_n * gt_n).sum(-1))[mask].mean()
        mag_loss = ((torch.log(ns + EPS) - torch.log(nt + EPS)) ** 2)[mask].mean()
        loss = dir_loss + BETA * mag_loss
        if VAL_W > 0:
            loss = loss + VAL_W * (L_s - Lt) ** 2
        (loss / BATCH).backward()
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    opt.step()
    if step % EVAL_EVERY == 0:
        tp, tf = evaluate(student, tr_sample)
        vp, vf = evaluate(student, val_items)
        if vp > best: best = vp
        print(f"step {step:3d} lr={lr:.1e} | train perpos={tp:.4f} flat={tf:.4f} | val perpos={vp:.4f} flat={vf:.4f} best={best:.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print(f"FINAL best val perpos={best:.4f} (baseline {b0[0]:.4f})", flush=True)
print("DONE")
