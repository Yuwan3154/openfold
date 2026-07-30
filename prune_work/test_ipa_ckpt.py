"""Correctness + stability test for structure_module.py's new opt-in IPA checkpointing
(self.ckpt / USE_IPA_CKPT), before trusting it for a real L=192 sweep.

Test 1 (CORRECTNESS): on the SAME input, compare student_grad(create_graph=True) with
structure_module.ckpt=False vs =True. Model is in eval() mode (dropout off, matches WS4-BC),
so recomputation during the checkpoint backward should reproduce bit-close (not necessarily
bit-exact, since torch.utils.checkpoint's non-reentrant recompute can hit slightly different
kernel/reduction paths) loss L and gradient g. Large mismatch => bug in the Rigid<->tensor_7
round-trip or the checkpoint wiring, not just "expected float noise".

Test 2 (TRAINING STABILITY): a short (60-step) real training loop with IPA ckpt ON at a length
that's known-good without it, to see the loss/cosine trajectory looks like previously-seen runs
(climbing, not NaN/diverging) -- checkpointing must not silently change optimization dynamics.

Test 3 (MEMORY): PROBE_LONGEST-style longest-chain smoke test at increasing MAXLEN with
evoformer+extra_msa+IPA all checkpointed, to see how far peak memory actually gets pushed.
"""
import os, sys, json, glob, math, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.np import residue_constants as rc

sys.path.insert(0, "/home/jupyter-chenxi/openfold/openfold/block_replacement_scripts")
from hallucination_straight_through import make_feature_batch
sys.path.insert(0, "/home/jupyter-chenxi/prune_work")
import ws4_bc_losses as bcl
# NOTE: ws4_bc_train.py has no `if __name__ == "__main__"` guard -- importing it
# executes its entire module-level training run as a side effect. So its helpers
# are duplicated here (verbatim) instead of imported.

RECYCLE = int(os.environ.get("RECYCLE", "1"))
SCALE = float(os.environ.get("SCALE", "3.0"))
EPS = 1e-6
BASE = "/home/jupyter-chenxi"
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


def enable_ckpt(m, ipa=False):
    if hasattr(m.evoformer, "blocks_per_ckpt"):
        m.evoformer.blocks_per_ckpt = 1
    if hasattr(m, "extra_msa_stack"):
        if hasattr(m.extra_msa_stack, "ckpt"):
            m.extra_msa_stack.ckpt = True
        for b in getattr(m.extra_msa_stack, "blocks", []):
            if hasattr(b, "ckpt"):
                b.ckpt = True
    if ipa and hasattr(m, "structure_module"):
        m.structure_module.ckpt = True


def build_student():
    m = AlphaFold(build_cfg())
    b = m.evoformer.blocks
    m.evoformer.blocks = nn.ModuleList([b[i] for i in KEEP])
    sd = torch.load(CK2, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    return m.to(DEV).eval()


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


cdc = json.load(open(CDC))
cached = sorted(c for p in glob.glob(f"{GC}/*.pt")
                 for c in [os.path.basename(p)[:-3]] if c in cdc and len(cdc[c]["seq"]) <= 128)
random.Random(0).shuffle(cached)
test_chains = cached[:6]

print("=== TEST 1: correctness (ckpt=False vs ckpt=True, same input, create_graph=True) ===", flush=True)
student = build_student()
for c in test_chains:
    d = torch.load(f"{GC}/{c}.pt", map_location=DEV)
    bl = d["soft_logits"].to(DEV)

    kill_ckpt(student)
    g0, L0 = student_grad(student, bl, create_graph=True)
    g0 = g0.detach().clone(); L0 = float(L0.detach())

    enable_ckpt(student, ipa=True)
    g1, L1 = student_grad(student, bl, create_graph=True)
    g1 = g1.detach().clone(); L1 = float(L1.detach())

    kill_ckpt(student)  # restore for next iter

    cos = float(F.cosine_similarity(g0.flatten(), g1.flatten(), 0))
    maxabsdiff = float((g0 - g1).abs().max())
    relnorm = float((g0 - g1).norm() / (g0.norm() + 1e-8))
    print(f"{c} L={cdc[c]['seq'].__len__() if False else len(cdc[c]['seq'])}aa | "
          f"L0={L0:.6f} L1={L1:.6f} dL={abs(L0-L1):.2e} | "
          f"cos(g0,g1)={cos:.6f} max|dg|={maxabsdiff:.2e} rel||dg||={relnorm:.2e}", flush=True)
print("DONE_TEST1", flush=True)

print("=== TEST 2: 60-step training stability with IPA ckpt ON (MAXLEN=128) ===", flush=True)
train_pool = cached[6:206]
val_pool = cached[206:246]
items = lambda pool: [(c, "soft") for c in pool]
train_items = items(train_pool)
val_items = items(val_pool)


def teacher_for(c, pt):
    d = torch.load(f"{GC}/{c}.pt", map_location=DEV)
    return d["soft_logits"].to(DEV), d["g_soft"].to(DEV), d["L_soft"]


def perpos_cos(g_s, g_t):
    ns = g_s.norm(dim=-1); nt = g_t.norm(dim=-1); mask = nt > EPS
    cc = ((g_s / (ns[:, None] + 1e-8)) * (g_t / (nt[:, None] + 1e-8))).sum(-1)
    return cc[mask].mean()


def evaluate(model, its):
    # NOTE: no @torch.no_grad() here -- student_grad needs a live graph (even
    # with create_graph=False) to compute the first-order gradient at all.
    acc = 0.0
    for c, pt in its:
        bl, g_t, _ = teacher_for(c, pt)
        g_s, _ = student_grad(model, bl, create_graph=False)
        acc += float(perpos_cos(g_s.detach(), g_t))
    return acc / len(its)


student2 = build_student()
enable_ckpt(student2, ipa=True)
for p in student2.input_embedder.parameters(): p.requires_grad_(False)
for p in student2.aux_heads.parameters(): p.requires_grad_(False)
for p in student2.structure_module.parameters(): p.requires_grad_(False)
params = [p for p in student2.evoformer.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-4)
b0 = evaluate(student2, val_items)
print(f"baseline val perpos={b0:.4f}", flush=True)
order = train_items[:]
random.Random(1).shuffle(order)
ptr = 0
BATCH = 8
for step in range(1, 61):
    opt.zero_grad()
    for _ in range(BATCH):
        if ptr >= len(order):
            random.Random(step).shuffle(order); ptr = 0
        c, pt = order[ptr]; ptr += 1
        bl, g_t, _ = teacher_for(c, pt)
        g_s, _ = student_grad(student2, bl, create_graph=True)
        ns = g_s.norm(dim=-1); nt = g_t.norm(dim=-1); mask = nt > EPS
        gs_n = g_s / (ns[:, None] + 1e-8); gt_n = g_t / (nt[:, None] + 1e-8)
        dir_loss = (1.0 - (gs_n * gt_n).sum(-1))[mask].mean()
        mag_loss = ((torch.log(ns + EPS) - torch.log(nt + EPS)) ** 2)[mask].mean()
        loss = dir_loss + 0.1 * mag_loss
        (loss / BATCH).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 10 == 0:
        vp = evaluate(student2, val_items)
        print(f"step {step:3d} val perpos={vp:.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print("DONE_TEST2", flush=True)
