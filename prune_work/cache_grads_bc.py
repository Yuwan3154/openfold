"""WS4-BC teacher cache: full-48 model_1_ptm, single-seq + NO templates, recycle=1, dropout off.
For ~2k diverse chains (from slim_struct_train.list) cache the teacher's input-gradient dL/dseq of the
BindCraft monomer DESIGN loss (con+pae+plddt+rg-helix, faithful port) at TWO sequence points:
  - native : scale*onehot(native sequence)            (peaked, ~design-converged regime)
  - soft   : softmax of per-chain gaussian logits     (high-entropy, BindCraft early-design regime)
Saves both gradients + the exact soft logits (so the student matches at the identical point).
Args/env: SHARD NSHARD [MAXLEN=128] [N=2000] [RECYCLE=1] [SCALE=3.0] [SOFT_STD=1.0]. Resumable."""
import os, sys, json, hashlib
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

SHARD = int(sys.argv[1]); NSHARD = int(sys.argv[2])
MAXLEN = int(os.environ.get("MAXLEN", "128"))
N = int(os.environ.get("N", "2000"))
RECYCLE = int(os.environ.get("RECYCLE", "1"))
SCALE = float(os.environ.get("SCALE", "3.0"))
SOFT_STD = float(os.environ.get("SOFT_STD", "1.0"))
USE_CKPT = os.environ.get("USE_CKPT", "0") == "1"   # enable activation checkpointing (fit long L single-backward; needs use_reentrant=False fix)
BASE = "/home/jupyter-chenxi"
JAX = f"{BASE}/params/params_model_1_ptm.npz"
CDC = f"{BASE}/data/pdb_mmcif/chain_data_cache.json"
LIST = f"{BASE}/prune_work/lists_pdb/slim_struct_train.list"
OUT = f"{BASE}/data/grad_cache_bc"
DEV = "cuda:0"
os.makedirs(OUT, exist_ok=True)


def build_teacher():
    cfg = model_config("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = RECYCLE
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    m = AlphaFold(cfg)
    import_jax_weights_(m, JAX, version="model_1_ptm")
    m = m.to(DEV).eval()
    if USE_CKPT:                                    # checkpoint to fit long-L single-backward (use_reentrant=False fix)
        if hasattr(m.evoformer, "blocks_per_ckpt"):
            m.evoformer.blocks_per_ckpt = 1
        if hasattr(m, "extra_msa_stack"):
            if hasattr(m.extra_msa_stack, "ckpt"):
                m.extra_msa_stack.ckpt = True
            for b in getattr(m.extra_msa_stack, "blocks", []):
                if hasattr(b, "ckpt"):
                    b.ckpt = True
    else:
        for mod in m.modules():                     # kill ALL activation checkpointing
            if hasattr(mod, "ckpt"):
                mod.ckpt = False
            if hasattr(mod, "blocks_per_ckpt"):
                mod.blocks_per_ckpt = None
    for p in m.parameters():
        p.requires_grad_(False)
    return m


cdc = json.load(open(CDC))


def seq_of(chain):
    return cdc[chain]["seq"]


def native_logits(seq):
    aat = torch.tensor([rc.restype_order.get(a, rc.restype_num) for a in seq])
    oh = F.one_hot(aat.clamp(max=19), 20).float()
    oh[aat >= 20] = 0.0
    return (SCALE * oh).to(DEV)


def soft_logits(chain, Ln):
    seed = int(hashlib.md5(chain.encode()).hexdigest()[:8], 16)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (SOFT_STD * torch.randn(Ln, 20, generator=gen)).to(DEV)


def grad_at(model, base_logits):
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
    g = torch.autograd.grad(L, sl)[0]
    return g.detach(), float(L), {k: float(v) for k, v in losses.items()}


# chain selection: SOURCE=all (entire chain_data_cache; BindCraft loss is seq-only/unsupervised) or slim list
SOURCE = os.environ.get("SOURCE", "slim")
if SOURCE == "all":
    raw = list(cdc.keys())
else:
    raw = [l.strip().split()[0] for l in open(LIST) if l.strip()]
ids = [c for c in raw if c in cdc and 40 <= len(cdc[c]["seq"]) <= MAXLEN]
seen = set(); ids = [c for c in ids if not (c in seen or seen.add(c))]
import random
random.Random(0).shuffle(ids)
ids = ids[:N]
print(f"[shard{SHARD}/{NSHARD}] candidates={len(ids)} MAXLEN={MAXLEN} RECYCLE={RECYCLE} SCALE={SCALE} SOFT_STD={SOFT_STD}", flush=True)

model = build_teacher()
done = skip = fail = 0
for i, c in enumerate(ids):
    if i % NSHARD != SHARD:
        continue
    outp = f"{OUT}/{c}.pt"
    if os.path.exists(outp):
        skip += 1; continue
    seq = seq_of(c); Ln = len(seq)
    nl = native_logits(seq)
    g_nat, L_nat, ls_nat = grad_at(model, nl)
    sl = soft_logits(c, Ln)
    g_soft, L_soft, ls_soft = grad_at(model, sl)
    if not (torch.isfinite(g_nat).all() and torch.isfinite(g_soft).all()):
        fail += 1; continue
    torch.save({"g_native": g_nat.cpu(), "g_soft": g_soft.cpu(), "soft_logits": sl.cpu(),
                "L": Ln, "scale": SCALE, "soft_std": SOFT_STD,
                "L_native": L_nat, "L_soft": L_soft, "losses_native": ls_nat, "losses_soft": ls_soft}, outp)
    done += 1
    if done % 10 == 0:
        print(f"[shard{SHARD}] done={done} skip={skip} fail={fail} last={c} L={Ln} Lnat={L_nat:.3f} Lsoft={L_soft:.3f} peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
    torch.cuda.empty_cache()
print(f"[shard{SHARD}] FINISHED done={done} skip={skip} fail={fail}", flush=True)
