#!/usr/bin/env python3
"""
Hallucination demo using gradient straight-through replacement blocks.

This script:
- Loads an AlphaFold model and replacement weights.
- Wraps any replacement-enabled Evoformer blocks with a gradient straight-through:
    out = repl(x) + orig(x).detach() - repl(x).detach()
- Runs a sequence-design loop against a ground-truth distogram (default PDB 6mrr),
  optimizing sequence logits for up to 100 steps or until mean distogram loss
  drops below a threshold.
- Saves the final structure (CA-only PDB), loss trajectory plot, and a GIF of
  sequence logits to `SOLab/AFdistill/outputs` by default.

Notes:
- Uses straight-through soft sequence features to keep gradients flowing while
  feeding the standard AlphaFold input interface.
- Templates and extra MSA are disabled for simplicity.
"""

import argparse
import io
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.loss import distogram_loss
from openfold.utils.feats import pseudo_beta_fn
from openfold.data.data_transforms import make_atom14_masks
import openfold.np.residue_constants as rc
from openfold.block_replacement_scripts.custom_evoformer_replacement import SimpleEvoformerReplacement
from openfold.np.protein import from_pdb_string, from_prediction, to_pdb, Protein

class GradientStraightThroughBlock(nn.Module):
    """Wrapper that keeps forward identical to the original block output."""

    def __init__(self, original_block: nn.Module, replacement_block: nn.Module, block_idx: int):
        super().__init__()
        self.original_block = original_block
        self.replacement_block = replacement_block
        self.block_idx = block_idx
        self.use_straight_through = True
        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        if not self.use_straight_through:
            return self.original_block(*args, **kwargs)
        with torch.no_grad():
            orig_m, orig_z = self.original_block(*args, **kwargs)
        rep_m, rep_z = self.replacement_block(*args, **kwargs)
        m_out = rep_m + orig_m.detach() - rep_m.detach()
        z_out = rep_z + orig_z.detach() - rep_z.detach()
        return m_out, z_out


def apply_gradient_straight_through(model: nn.Module) -> int:
    wrapped = 0
    for idx, block in enumerate(model.evoformer.blocks):
        if hasattr(block, "original_block") and hasattr(block, "replacement_block"):
            model.evoformer.blocks[idx] = GradientStraightThroughBlock(
                block.original_block, block.replacement_block, idx
            )
            wrapped += 1
    return wrapped


def fetch_pdb(pdb_id: str) -> str:
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    import urllib.request

    with urllib.request.urlopen(url) as handle:  # noqa: S310
        return handle.read().decode("utf-8")


def load_pdb_string(pdb_path: Path, pdb_id: str) -> str:
    if pdb_path.exists():
        return pdb_path.read_text()
    return fetch_pdb(pdb_id)


def ground_truth_pseudo_beta(prot: Protein, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = prot.aatype != 20  # drop waters/UNK (20)
    aatype_np = prot.aatype[valid_mask]
    all_pos_np = prot.atom_positions[valid_mask]
    all_mask_np = prot.atom_mask[valid_mask]

    aatype = torch.tensor(aatype_np, device=device, dtype=torch.long)
    all_pos = torch.tensor(all_pos_np, device=device, dtype=torch.float32)
    all_mask = torch.tensor(all_mask_np, device=device, dtype=torch.float32)
    pseudo_beta, pseudo_mask = pseudo_beta_fn(aatype, all_pos, all_mask)
    return pseudo_beta, pseudo_mask


def make_feature_batch(
    seq_logits: torch.Tensor,
    residue_index: torch.Tensor,
    recycle_dim: int = 1,
) -> Dict[str, torch.Tensor]:
    """Construct minimal feature dict with differentiable sequence channels."""
    device = seq_logits.device
    L = seq_logits.shape[0]
    seq_probs20 = F.softmax(seq_logits, dim=-1)
    seq_probs21 = torch.cat(
        [seq_probs20, torch.zeros((L, 1), device=device, dtype=seq_probs20.dtype)],
        dim=-1,
    )  # [L,21]

    target_feat = torch.cat(
        [torch.zeros((L, 1), device=device, dtype=seq_probs21.dtype), seq_probs21],
        dim=-1,
    )  # [L,22]

    msa_feat = torch.zeros((1, L, 49), device=device, dtype=seq_probs21.dtype)
    msa_one_hot = torch.cat(
        [seq_probs21, torch.zeros((L, 2), device=device, dtype=seq_probs21.dtype)],
        dim=-1,
    )  # 23 dims
    msa_feat[..., :23] = msa_one_hot.unsqueeze(0)

    seq_mask = torch.ones((L,), device=device, dtype=seq_probs21.dtype)
    msa_mask = torch.ones((1, L), device=device, dtype=seq_probs21.dtype)
    pair_mask = seq_mask[:, None] * seq_mask[None, :]
    extra_msa_mask = torch.zeros((1, L), device=device, dtype=seq_probs21.dtype)
    extra_msa = torch.zeros((1, L), device=device, dtype=torch.long)
    extra_has_deletion = torch.zeros((1, L), device=device, dtype=seq_probs21.dtype)
    extra_deletion_value = torch.zeros((1, L), device=device, dtype=seq_probs21.dtype)
    aatype_int = torch.argmax(seq_probs21, dim=-1).to(torch.long)

    def add_cycle(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1).expand(*x.shape, recycle_dim)

    atom14_masks = make_atom14_masks({"aatype": aatype_int})

    batch = {
        "aatype": add_cycle(aatype_int),
        "target_feat": add_cycle(target_feat),
        "residue_index": add_cycle(residue_index),
        "msa_feat": add_cycle(msa_feat),
        "seq_mask": add_cycle(seq_mask),
        "msa_mask": add_cycle(msa_mask),
        "pair_mask": add_cycle(pair_mask),
        "extra_msa_mask": add_cycle(extra_msa_mask),
        "extra_msa": add_cycle(extra_msa),
        "extra_has_deletion": add_cycle(extra_has_deletion),
        "extra_deletion_value": add_cycle(extra_deletion_value),
        "atom14_atom_exists": add_cycle(atom14_masks["atom14_atom_exists"]),
        "residx_atom14_to_atom37": add_cycle(atom14_masks["residx_atom14_to_atom37"]),
        "residx_atom37_to_atom14": add_cycle(atom14_masks["residx_atom37_to_atom14"]),
        "atom37_atom_exists": add_cycle(atom14_masks["atom37_atom_exists"]),
        "return_representations": False,
    }
    return batch


def ca_from_atom37(atom37: np.ndarray) -> np.ndarray:
    if atom37.ndim == 2:
        atom37 = atom37[None, ...]
    ca_idx = rc.atom_order["CA"]
    return atom37[:, ca_idx, :]


def write_ca_pdb(path: Path, ca_coords: np.ndarray, seq_letters: Optional[np.ndarray] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ca_coords.ndim == 3 and ca_coords.shape[0] == 1:
        ca_coords = ca_coords[0]
    if seq_letters is None:
        seq_letters = np.array(["ALA"] * ca_coords.shape[0])
    assert ca_coords.shape[0] == len(seq_letters), "Sequence length and coordinates length mismatch"
    with open(path, "w", encoding="utf-8") as handle:
        atom_id = 1
        for i, (xyz, resname) in enumerate(zip(ca_coords, seq_letters)):
            x, y, z = xyz
            line = (
                f"ATOM  {atom_id:5d}  CA  {resname:3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            handle.write(line)
            atom_id += 1
        handle.write("END\n")


def save_loss_plot(losses, path: Path, title: str = "Hallucination loss trajectory"):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Mean distogram loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_sequence_gif(seq_history, path: Path, title: str = "Hallucination sequence logits"):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    aa_labels = list(rc.restypes) + ["GAP"]
    x_ticks = np.arange(len(aa_labels))
    for frame in seq_history:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(frame, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Amino acid")
        ax.set_ylabel("Residue")
        ax.set_title("Sequence logits (softmax)")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(aa_labels, rotation=90)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = rgba[..., :3]  # drop alpha
        images.append(image)
        plt.close(fig)
    imageio.mimsave(path, images, fps=4)


def _strip_student_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("student_model.") for k in state_dict):
        return state_dict
    return {k[len("student_model.") :]: v for k, v in state_dict.items() if k.startswith("student_model.")}


def _load_block_submodule(module: nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str):
    sub = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if sub:
        missing, unexpected = module.load_state_dict(sub, strict=False)
        if missing or unexpected:
            print(f"Warning: block load had missing={len(missing)} unexpected={len(unexpected)} for {prefix}")


def _wrap_replacements_from_checkpoint(model: AlphaFold, state_dict: Dict[str, torch.Tensor], c_m: int, c_z: int,linear_type: str = "full") -> int:
    """Replace Evoformer blocks with checkpointed replacements when available."""
    wrapped = 0
    for idx in range(len(model.evoformer.blocks)):
        rep_prefix = f"replacement_blocks.{idx}."
        if any(k.startswith(rep_prefix) for k in state_dict):
            replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type=linear_type)
            _load_block_submodule(replacement, state_dict, rep_prefix)
            replacement = replacement.to(next(model.parameters()).device)
            # Optional: load original block fine-tunes if present
            orig_prefix = f"original_blocks.{idx}."
            _load_block_submodule(model.evoformer.blocks[idx], state_dict, orig_prefix)
            model.evoformer.blocks[idx] = GradientStraightThroughBlock(
                model.evoformer.blocks[idx], replacement, idx
            )
            wrapped += 1
    return wrapped


def _set_straight_through(model: AlphaFold, enabled: bool):
    for blk in model.evoformer.blocks:
        if isinstance(blk, GradientStraightThroughBlock):
            blk.use_straight_through = enabled


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    straight_through: bool = True,
) -> AlphaFold:
    with open(config_path, "r", encoding="utf-8") as handle:
        import yaml

        cfg_yaml = yaml.safe_load(handle)
    preset = cfg_yaml.get("config_preset", "model_1_ptm")
    weights_path = cfg_yaml["weights_path"]
    linear_type = cfg_yaml["linear_type"]

    cfg = model_config(preset)
    cfg.model.template.enabled = False
    cfg.globals.chunk_size = 4
    cfg.model.num_recycle = 1

    model = AlphaFold(cfg).to(device)
    import_jax_weights_(model, os.path.join(Path.home(), weights_path), version=preset)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = _strip_student_prefix(state_dict)

    # Load non-replacement weights first
    base_state = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("replacement_blocks.") and not k.startswith("original_blocks.")
    }
    model.load_state_dict(base_state, strict=False)
    # Handle config schema differences
    evo_cfg = getattr(cfg.model, "evoformer", None) or getattr(cfg.model, "evoformer_stack")
    c_m = evo_cfg.c_m
    c_z = evo_cfg.c_z

    # Inject replacement blocks when present in checkpoint
    wrapped = _wrap_replacements_from_checkpoint(model, state_dict, c_m, c_z, linear_type)
    if device.type == "cuda":
        model = model.bfloat16()

    if straight_through:
        if wrapped == 0:
            wrapped = apply_gradient_straight_through(model)
        if wrapped == 0:
            print("No adaptive/replacement blocks found; running base model.")
        else:
            print(f"Applied gradient straight-through to {wrapped} Evoformer blocks.")
    model.eval()
    return model


def optimize_sequence(
    model: AlphaFold,
    seq_len: int,
    pseudo_beta: torch.Tensor,
    pseudo_mask: torch.Tensor,
    steps: int,
    loss_cutoff: float,
    device: torch.device,
    log_fn=lambda s: print(s, flush=True),
    orig_steps_per_cycle: int = 0,
    repl_steps_per_cycle: int = 1,
    lr: float = 1.0,
) -> Tuple[torch.Tensor, list, list, torch.Tensor]:
    seq_logits = nn.Parameter(torch.randn(seq_len, 20, device=device))
    optimizer = torch.optim.Adam([seq_logits], lr=lr)
    losses = []
    seq_frames = []
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)

    cycle = orig_steps_per_cycle + repl_steps_per_cycle
    t_start = time.perf_counter()
    for step in tqdm(range(steps), desc="Optimizing sequence"):
        optimizer.zero_grad()
        if cycle > 0:
            pos = step % cycle
            use_straight_through = pos >= orig_steps_per_cycle
            _set_straight_through(model, enabled=use_straight_through)
        else:
            _set_straight_through(model, enabled=True)
        batch = make_feature_batch(seq_logits, residue_index)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs = model(batch)
            logits = outputs["distogram_logits"]
            loss = distogram_loss(
                logits=logits,
                pseudo_beta=pseudo_beta,
                pseudo_beta_mask=pseudo_mask,
            )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        seq_frames.append(torch.softmax(seq_logits.detach(), dim=-1).cpu().numpy())
        log_fn(f"Step {step+1}: loss={loss.item():.4f}")
        if loss.item() < loss_cutoff:
            break
    opt_time_s = time.perf_counter() - t_start

    batch_final = make_feature_batch(seq_logits, residue_index)
    _set_straight_through(model, enabled=True)
    outputs = model(batch_final)
    final_seq_probs = torch.softmax(seq_logits.detach(), dim=-1)
    return outputs, losses, seq_frames, final_seq_probs, opt_time_s


def main():
    parser = argparse.ArgumentParser(description="Hallucination with gradient straight-through replacements")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("~/SOLab/AFdistill/configs/repr_distill_per-block.yaml"),
        help="Training config used for replacement blocks",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("~/SOLab/AFdistill/results/repr_distill_per-block/checkpoints/last.ckpt"),
        help="Checkpoint containing replacement weights",
    )
    parser.add_argument("--chain_id", type=str, default="A", help="Chain ID to use")
    parser.add_argument("--pdb_id", type=str, required=True, help="PDB identifier to use")
    parser.add_argument("--pdb_path", type=Path, required=True, help="Optional local PDB path")
    parser.add_argument("--steps", type=int, default=100, help="Maximum optimization steps")
    parser.add_argument("--loss_cutoff", type=float, default=0.05, help="Early-stop distogram loss threshold")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("~/SOLab/AFdistill/outputs"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="Optional log file path (overwrites each run)",
    )
    parser.add_argument(
        "--orig_steps_per_cycle",
        type=int,
        default=0,
        help="Number of original-backprop steps per cycle",
    )
    parser.add_argument(
        "--repl_steps_per_cycle",
        type=int,
        default=1,
        help="Number of replacement-straight-through steps per cycle",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_prefix = f"{args.pdb_id}_orig-{args.orig_steps_per_cycle}_repl-{args.repl_steps_per_cycle}_steps-{args.steps}_lr-{str(args.lr).replace('.', '-')}_loss-cutoff-{str(args.loss_cutoff).replace('.', '-')}"
    out_dir = out_dir / out_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or (out_dir / f"{out_prefix}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle: Optional[object] = open(log_path, "w", encoding="utf-8")

    def tee(msg: str):
        print(msg, flush=True)
        if log_handle is not None:
            print(msg, file=log_handle, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdb_string = load_pdb_string(args.pdb_path, args.pdb_id)
    prot = from_pdb_string(pdb_string, chain_id=args.chain_id)
    pseudo_beta, pseudo_mask = ground_truth_pseudo_beta(prot, device)
    tee(f"Ground truth length: {pseudo_beta.shape[-2]}")

    model = load_model(args.config_path, args.checkpoint_path, device, straight_through=True)
    outputs, losses, seq_frames, final_seq_probs, opt_time_s = optimize_sequence(
        model,
        seq_len=pseudo_beta.shape[-2],
        pseudo_beta=pseudo_beta,
        pseudo_mask=pseudo_mask,
        steps=args.steps,
        loss_cutoff=args.loss_cutoff,
        device=device,
        log_fn=tee,
        orig_steps_per_cycle=args.orig_steps_per_cycle,
        repl_steps_per_cycle=args.repl_steps_per_cycle,
    )

    loss_plot = out_dir / f"{out_prefix}_loss.png"
    gif_path = out_dir / f"{out_prefix}_seq.gif"
    pdb_out = out_dir / f"{out_prefix}_final_ca.pdb"

    save_loss_plot(losses, loss_plot, title=f"{out_prefix} loss trajectory for {args.pdb_id}")
    save_sequence_gif(seq_frames, gif_path, title=f"{out_prefix} sequence for {args.pdb_id}")

    atom37 = outputs["final_atom_positions"].detach().float().cpu().numpy()
    atom37_mask = outputs["final_atom_mask"].detach().bool().cpu().numpy() if "final_atom_mask" in outputs else None
    pred_seq_idx = final_seq_probs.argmax(dim=-1).long().cpu().numpy()
    restype_3 = np.array([rc.restype_1to3[rc.restypes[i]] for i in pred_seq_idx])
    tee(f"Predicted seq length: {len(restype_3)}")
    # Build full-atom PDB via OpenFold utilities
    result = {
        "final_atom_positions": atom37,
        "final_atom_mask": atom37_mask,
    }
    features = {
        "aatype": pred_seq_idx[None, :],
        "residue_index": np.arange(len(pred_seq_idx))[None, :],
    }
    prot = from_prediction(features=features, result=result, remove_leading_feature_dimension=True)
    pdb_str = to_pdb(prot)
    with open(pdb_out, "w", encoding="utf-8") as f:
        f.write(pdb_str)

    tee(f"Saved loss curve to {loss_plot}")
    tee(f"Saved sequence GIF to {gif_path}")
    tee(f"Saved final CA-only PDB to {pdb_out}")
    tee(f"Log written to {log_path}")
    tee(f"Optimization wall time: {opt_time_s:.2f}s")
    if log_handle is not None:
        log_handle.close()


if __name__ == "__main__":
    main()
