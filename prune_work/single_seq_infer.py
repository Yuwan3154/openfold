"""Shared single-sequence (no MSA, no template) OpenFold inference for scoring arbitrary sequences.
Reuses the same feature-construction / pruning / loss-extraction helpers as ws4_bc_train.py so the
inference path here is architecturally identical to what WS5 was actually trained under.
"""
import glob
import os
import sys
import torch
import torch.nn.functional as F

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.np import residue_constants as rc

BASE = "/home/jupyter-chenxi"
sys.path.insert(0, f"{BASE}/openfold/openfold/block_replacement_scripts")
from hallucination_straight_through import make_feature_batch
from pruned_evoformer import prune_blocks

sys.path.insert(0, f"{BASE}/prune_work")
import ws4_bc_losses as bcl

SCALE = 3.0


def build_cfg(recycle=3):
    cfg = model_config("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    # NOTE: use_deepspeed_evo_attention JIT-compiles a CUTLASS-dependent CUDA op that fails on
    # this box (missing $CUTLASS_PATH) -- tried it for speed, it broke 614/614 sequences. Reverted.
    # use_flash also doesn't apply (incompatible with AF2's pair-bias attention). Plain attention
    # it is -- slow under WS5 CPU/GPU contention, fine on a free GPU.
    cfg.data.common.max_recycling_iters = recycle
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    return cfg


WS5_CKPT_DIR = f"{BASE}/runs/prune_singleseq_v1/lightning_logs/version_3/checkpoints"


def resolve_ws5_ckpt(ckpt_dir=WS5_CKPT_DIR):
    """WS5 is still training; Lightning's ModelCheckpoint renames/deletes the old 'best-*.ckpt'
    file every time a new best is found, so any hardcoded filename goes stale. Resolve the
    current one by mtime instead of embedding a specific epoch/step in the path."""
    candidates = glob.glob(os.path.join(ckpt_dir, "best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"no best-*.ckpt found in {ckpt_dir}")
    return max(candidates, key=os.path.getmtime)


def load_ws5(ckpt_path, device="cuda:0", recycle=3):
    """WS5 architecture: prune_blocks() slims all 48 blocks in place (drop col-attn + tri-attn,
    keep tri-mul) -- this must match before load_state_dict(strict=True) will succeed."""
    m = AlphaFold(build_cfg(recycle))
    prune_blocks(m.evoformer)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # strict=False: WS5 was trained with templates enabled, so its checkpoint carries
    # template_embedder.* weights that this no-template inference config never instantiates
    # (same convention as ws4_bc_train.py's build_student()).
    missing, unexpected = m.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    assert not missing, f"unexpected missing keys (real mismatch, not template-only): {missing}"
    assert all(k.startswith("template_embedder.") for k in unexpected), \
        f"unexpected non-template keys (real mismatch): {[k for k in unexpected if not k.startswith('template_embedder.')]}"
    return m.to(device).eval()


def seq_to_logits(seq, device, scale=SCALE):
    aat = torch.tensor([rc.restype_order.get(a, rc.restype_num) for a in seq])
    oh = F.one_hot(aat.clamp(max=19), 20).float()
    oh[aat >= 20] = 0.0
    return (scale * oh).to(device)


@torch.no_grad()
def score_sequence(model, seq, device="cuda:0", recycle=3):
    """Single forward pass, no MSA, no template -- matches the field-standard self-consistency
    protocol (Caliby: single-sequence AF2, CA-only Kabsch RMSD downstream)."""
    logits = seq_to_logits(seq, device)
    ri = torch.arange(len(seq), device=device)
    batch = make_feature_batch(logits, ri, recycle_dim=recycle + 1)
    out = model(batch)
    dgl = out["distogram_logits"]; dgl = dgl[0] if dgl.dim() == 4 else dgl
    tml = out["tm_logits"]; tml = tml[0] if tml.dim() == 4 else tml
    lddt = out["lddt_logits"]; lddt = lddt[0] if lddt.dim() == 3 else lddt
    fap = out["final_atom_positions"]; fap = fap[0] if fap.dim() == 4 else fap
    ca = fap[:, rc.atom_order["CA"], :]
    plddt = bcl.get_plddt(lddt)
    pae = bcl.get_pae(tml)
    return {
        "plddt_mean": plddt.mean().item(),
        "plddt_per_res": plddt.detach().cpu().numpy(),
        "pae_mean": pae.mean().item(),
        "ca_coords": ca.detach().cpu().numpy(),
    }
