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
import importlib.util
import io
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
import csv
import json
import yaml
import re
import subprocess
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.loss import compute_renamed_ground_truth, distogram_loss, fape_loss
from openfold.utils.feats import pseudo_beta_fn
from openfold.data.data_transforms import (
    atom37_to_frames,
    get_backbone_frames,
    make_atom14_masks,
    make_atom14_positions,
)
import openfold.np.residue_constants as rc
from openfold.block_replacement_scripts.custom_evoformer_replacement import SimpleEvoformerReplacement
from openfold.block_replacement_scripts.dilated_conv_evoformer_replacement import (
    DilatedConvEvoformerReplacement,
    TriangleOpConvReplacement,
)
from openfold.np.protein import from_pdb_string, from_prediction, to_pdb, Protein

class GradientStraightThroughBlock(nn.Module):
    """Wrapper that keeps forward identical to the original block output."""

    def __init__(self, original_block: nn.Module, replacement_block: nn.Module, block_idx: int):
        super().__init__()
        self.original_block = original_block
        self.replacement_block = replacement_block
        self.block_idx = block_idx
        self.use_straight_through = True
        self.ckpt_replacement_only = False
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


def build_ground_truth_batch(prot: Protein, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Build the minimal ground-truth dict required for FAPE loss.

    Note: openfold's Protein parser can include water/UNK as aatype==20; we drop those.
    Shapes match model outputs (no explicit batch dim).
    """
    valid_mask = prot.aatype != 20  # drop waters/UNK (20)
    aatype = torch.tensor(prot.aatype[valid_mask], device=device, dtype=torch.long)
    all_atom_positions = torch.tensor(
        prot.atom_positions[valid_mask], device=device, dtype=torch.float32
    )
    all_atom_mask = torch.tensor(
        prot.atom_mask[valid_mask], device=device, dtype=torch.float32
    )
    residue_index = torch.arange(aatype.shape[0], device=device, dtype=torch.long)
    seq_mask = torch.ones((aatype.shape[0],), device=device, dtype=torch.float32)
    seq_length = torch.tensor(aatype.shape[0], device=device, dtype=torch.long)

    batch = {
        "aatype": aatype,
        "all_atom_positions": all_atom_positions,
        "all_atom_mask": all_atom_mask,
        "residue_index": residue_index,
        "seq_mask": seq_mask,
        "seq_length": seq_length,
    }
    batch = make_atom14_masks(batch)
    batch = make_atom14_positions(batch)
    batch = atom37_to_frames(batch)
    batch = get_backbone_frames(batch)
    return batch


def make_feature_batch(
    seq_logits: torch.Tensor,
    residue_index: torch.Tensor,
    recycle_dim: int = 1,
    msa_depth: int = 1,
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

    msa_feat = torch.zeros((msa_depth, L, 49), device=device, dtype=seq_probs21.dtype)
    msa_one_hot = torch.cat(
        [seq_probs21, torch.zeros((L, 2), device=device, dtype=seq_probs21.dtype)],
        dim=-1,
    )  # 23 dims
    msa_feat[..., :23] = msa_one_hot.unsqueeze(0)

    seq_mask = torch.ones((L,), device=device, dtype=seq_probs21.dtype)
    msa_mask = torch.ones((msa_depth, L), device=device, dtype=seq_probs21.dtype)
    pair_mask = seq_mask[:, None] * seq_mask[None, :]
    extra_msa_mask = torch.zeros((msa_depth, L), device=device, dtype=seq_probs21.dtype)
    extra_msa = torch.zeros((msa_depth, L), device=device, dtype=torch.long)
    extra_has_deletion = torch.zeros((msa_depth, L), device=device, dtype=seq_probs21.dtype)
    extra_deletion_value = torch.zeros((msa_depth, L), device=device, dtype=seq_probs21.dtype)
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


def save_loss_plot(
    losses,
    path: Path,
    title: str = "Hallucination loss trajectory",
    y_label: str = "Loss",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(title, wrap=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_sequence_gif(seq_history, path: Path, title: str = "Hallucination sequence logits"):
    if importlib.util.find_spec("imageio") is None:
        print("imageio not available; skipping sequence GIF", flush=True)
        return
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
        ax.set_title(title)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(aa_labels, rotation=90)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = rgba[..., :3]  # drop alpha
        images.append(image)
        plt.close(fig)
    imageio.mimsave(path, images, fps=4)


def _write_gt_pdb_for_usalign(pdb_string: str, chain_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    for line in pdb_string.splitlines():
        if not line:
            continue
        rec = line[0:6]
        if rec.startswith("ATOM  "):
            if len(line) > 21 and line[21] == chain_id:
                resname = line[17:20].strip()
                if resname not in {"HOH", "WAT"}:
                    keep.append(line)
        elif rec.startswith("TER") or rec.startswith("END"):
            keep.append(line)
    if len(keep) == 0:
        keep = ["END"]
    out_path.write_text("\n".join(keep) + "\n", encoding="utf-8")


def _parse_usalign_tm_score(stdout: str) -> Optional[float]:
    # Typical line: "TM-score= 0.1234 (if normalized by length of ...)"
    m = re.search(r"TM-score=\s*([0-9]*\.?[0-9]+)", stdout)
    if m is None:
        return None
    return float(m.group(1))


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


def _resolve_checkpoint_path(path: Path) -> Path:
    """
    Resolve a checkpoint path.
    - If `path` is a directory: prefer `last.ckpt` if present, else pick newest `*.ckpt`.
    - If `path` is a file: return it as-is.
    """
    p = Path(path).expanduser()
    if p.is_dir():
        last = p / "last.ckpt"
        if last.exists():
            return last
        ckpts = [q for q in p.glob("*.ckpt") if q.is_file()]
        if len(ckpts) == 0:
            raise FileNotFoundError(f"No .ckpt files found in checkpoint directory: {p}")
        return max(ckpts, key=lambda q: q.stat().st_mtime)
    return p


def _wrap_replacements_from_checkpoint(
    model: AlphaFold,
    state_dict: Dict[str, torch.Tensor],
    c_m: int,
    c_z: int,
    replacement_type: str = "linear",
    linear_type: str = "full",
    kernel_size: int = 3,
    dilations: Tuple[int, ...] = (1, 2, 4),
    dilation_pattern: Optional[Tuple[int, ...]] = None,
    dilation_repeats: int = 1,
    replacement_mode: str = "per_block",
) -> int:
    """Replace Evoformer blocks with checkpointed replacements when available."""
    wrapped = 0
    for idx in range(len(model.evoformer.blocks)):
        rep_prefix = f"replacement_blocks.{idx}."
        if any(k.startswith(rep_prefix) for k in state_dict):
            if replacement_type == "linear":
                replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type=linear_type)
            elif replacement_type == "conv":
                replacement = DilatedConvEvoformerReplacement(
                    c_m=c_m,
                    c_z=c_z,
                    kernel_size=int(kernel_size),
                    dilations=tuple(int(d) for d in dilations),
                    dilation_pattern=dilation_pattern,
                    dilation_repeats=int(dilation_repeats),
                    mode=str(replacement_mode),
                )
            else:
                raise ValueError(
                    f"Invalid replacement_type: {replacement_type}. Expected 'linear' or 'conv'."
                )
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


def _wrap_remaining_blocks(
    model: AlphaFold,
    c_m: int,
    c_z: int,
    replacement_type: str = "linear",
    linear_type: str = "full",
    kernel_size: int = 3,
    dilations: Tuple[int, ...] = (1, 2, 4),
    dilation_pattern: Optional[Tuple[int, ...]] = None,
    dilation_repeats: int = 1,
    replacement_mode: str = "per_block",
) -> int:
    """Wrap any Evoformer blocks that aren't already wrapped with GradientStraightThroughBlock.
    
    This ensures ALL blocks use replacement-only checkpointing for maximum memory savings.
    Blocks without pretrained weights get randomly initialized replacement blocks.
    """
    wrapped = 0
    for idx, block in enumerate(model.evoformer.blocks):
        if isinstance(block, GradientStraightThroughBlock):
            continue
        if replacement_type == "linear":
            replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type=linear_type)
        elif replacement_type == "conv":
            replacement = DilatedConvEvoformerReplacement(
                c_m=c_m,
                c_z=c_z,
                kernel_size=int(kernel_size),
                dilations=tuple(int(d) for d in dilations),
                dilation_pattern=dilation_pattern,
                dilation_repeats=int(dilation_repeats),
                mode=str(replacement_mode),
            )
        else:
            replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type="full")
        replacement = replacement.to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        wrapped_block = GradientStraightThroughBlock(block, replacement, idx)
        wrapped_block.use_straight_through = True
        wrapped_block.ckpt_replacement_only = True
        model.evoformer.blocks[idx] = wrapped_block
        wrapped += 1
    return wrapped


def _set_straight_through(model: AlphaFold, enabled: bool):
    for blk in model.evoformer.blocks:
        if isinstance(blk, GradientStraightThroughBlock):
            blk.use_straight_through = enabled


def _set_replacement_checkpoint_only(model: AlphaFold, enabled: bool):
    for blk in model.evoformer.blocks:
        if isinstance(blk, GradientStraightThroughBlock):
            blk.ckpt_replacement_only = enabled


class GradientStraightThroughModule(nn.Module):
    """
    Straight-through wrapper for a module operating on pair features.

    Forward: uses original module output (no grad).
    Backward: gradients flow through replacement module.
    """

    def __init__(
        self,
        original: nn.Module,
        replacement: nn.Module,
        ckpt_replacement_only: bool = True,
    ):
        super().__init__()
        self.original = original
        self.replacement = replacement
        self.ckpt_replacement_only = ckpt_replacement_only
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            orig_out = self.original(*args, **kwargs)
        mask = kwargs.get("mask", None)
        if mask is None and len(args) > 1:
            mask = args[1]
        z = args[0] if len(args) > 0 else None
        if self.ckpt_replacement_only:
            repl_out = torch.utils.checkpoint.checkpoint(
                self.replacement,
                z,
                mask,
                use_reentrant=False,
            )
        else:
            repl_out = self.replacement(z, pair_mask=mask)
        return repl_out + (orig_out - repl_out).detach()


def _wrap_triangle_ops(
    model: AlphaFold,
    kernel_size: int,
    dilation_pattern: Tuple[int, ...],
    dilation_repeats: int,
    ckpt_replacement_only: bool = True,
) -> int:
    """
    Set up triangle-only straight-through gradient estimation.

    Instead of wrapping each triangle op individually, stores a single
    replacement layer per block and sets a flag. The actual ST logic
    is handled in checkpoint_blocks via 4-phase execution:
      Phase 1: MSA + OPM (checkpointed)
      Phase 2: all 4 triangle ops (no_grad, not checkpointed)
      Phase 3: single replacement call (checkpointed) + ST combine
      Phase 4: pair_transition (checkpointed)
    """
    wrapped = 0
    c_z = model.evoformer.blocks[0].pair_stack.tri_att_start.c_in
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for block in model.evoformer.blocks:
        replacement = TriangleOpConvReplacement(
            c_z=c_z,
            kernel_size=int(kernel_size),
            dilation_pattern=tuple(int(d) for d in dilation_pattern),
            dilation_repeats=int(dilation_repeats),
        ).to(device=device, dtype=dtype)
        # Store replacement as a submodule so its params are registered
        block._triangle_replacement = replacement
        block._has_triangle_st = True
        # Freeze original triangle op parameters
        for name in ['tri_mul_out', 'tri_mul_in', 'tri_att_start', 'tri_att_end']:
            for param in getattr(block.pair_stack, name).parameters():
                param.requires_grad = False
        wrapped += 4
    return wrapped


def _parse_block_indices(spec: str) -> Tuple[int, ...]:
    """
    Parse a comma-separated list of ints and ranges into block indices.
    Example: "1-4,7,9-10" -> (1,2,3,4,7,9,10)
    """
    s = str(spec).strip()
    if s == "":
        return ()
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid block range: {part}")
            start = int(a)
            end = int(b)
            if end < start:
                raise ValueError(f"Invalid block range (end<start): {part}")
            out.extend(list(range(start, end + 1)))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid block index: {part}")
            out.append(int(part))
    return tuple(sorted(set(out)))


def _set_straight_through_blocks(model: AlphaFold, st_blocks: Tuple[int, ...]):
    st_set = set(int(i) for i in st_blocks)
    for blk in model.evoformer.blocks:
        if isinstance(blk, GradientStraightThroughBlock):
            blk.use_straight_through = blk.block_idx in st_set


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    straight_through: bool = True,
    allow_replacements: bool = True,
    disable_attention_opts: bool = False,
    enable_flash_attention: bool = False,
    enable_deepspeed_attention: bool = False,
    disable_chunking: bool = False,
    wrap_all_blocks: bool = False,
    triangle_st_only: bool = False,
    triangle_kernel_size: int = 3,
    triangle_dilation_pattern: Optional[Tuple[int, ...]] = None,
    triangle_dilation_repeats: int = 1,
) -> AlphaFold:
    with open(config_path, "r", encoding="utf-8") as handle:
        import yaml

        cfg_yaml = yaml.safe_load(handle)
    preset = cfg_yaml.get("config_preset", "model_1_ptm")
    weights_path = cfg_yaml["weights_path"]
    replacement_type = str(cfg_yaml.get("replacement_type", "linear"))
    linear_type = str(cfg_yaml.get("linear_type", "full"))
    kernel_size = int(cfg_yaml.get("kernel_size", 3))
    replacement_mode = str(cfg_yaml.get("replacement_mode", "per_block"))
    dilations_raw = cfg_yaml.get("dilations", "1,2,4")
    if isinstance(dilations_raw, str):
        dilations = tuple(int(d) for d in dilations_raw.split(",") if d.strip() != "")
    else:
        dilations = tuple(int(d) for d in dilations_raw)
    dilation_pattern_raw = cfg_yaml.get("dilation_pattern", None)
    dilation_repeats = int(cfg_yaml.get("dilation_repeats", 1))
    if dilation_pattern_raw is None:
        dilation_pattern = None
    elif isinstance(dilation_pattern_raw, str):
        dilation_pattern = tuple(int(d) for d in dilation_pattern_raw.split(",") if d.strip() != "")
    else:
        dilation_pattern = tuple(int(d) for d in dilation_pattern_raw)

    cfg = model_config(preset)
    cfg.model.template.enabled = False
    cfg.globals.chunk_size = 4
    cfg.model.num_recycle = 1
    # Enable activation checkpointing for hallucination backprop to reduce memory.
    cfg.globals.blocks_per_ckpt = 1
    cfg.model.evoformer_stack.blocks_per_ckpt = 1
    cfg.model.template.template_pair_stack.blocks_per_ckpt = 1
    cfg.model.extra_msa.extra_msa_stack.ckpt = True
    if disable_chunking:
        cfg.globals.chunk_size = None
        cfg.model.evoformer_stack.tune_chunk_size = False
        cfg.model.template.template_pair_stack.tune_chunk_size = False
        cfg.model.extra_msa.extra_msa_stack.tune_chunk_size = False
    if enable_flash_attention and disable_attention_opts:
        raise ValueError("enable_flash_attention and disable_attention_opts are mutually exclusive")
    if enable_deepspeed_attention and disable_attention_opts:
        raise ValueError("enable_deepspeed_attention and disable_attention_opts are mutually exclusive")
    if enable_deepspeed_attention and enable_flash_attention:
        raise ValueError("enable_deepspeed_attention and enable_flash_attention are mutually exclusive")
    if disable_attention_opts:
        cfg.globals.use_flash = False
        cfg.globals.use_lma = False
        cfg.globals.use_deepspeed_evo_attention = False
    if enable_flash_attention:
        cfg.globals.use_flash = True
        cfg.globals.use_lma = False
        cfg.globals.use_deepspeed_evo_attention = False
    if enable_deepspeed_attention:
        cfg.globals.use_flash = False
        cfg.globals.use_lma = False
        cfg.globals.use_deepspeed_evo_attention = True

    model = AlphaFold(cfg).to(device)
    import_jax_weights_(model, os.path.join(Path.home(), weights_path), version=preset)

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
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
    wrapped = 0
    if allow_replacements:
        wrapped = _wrap_replacements_from_checkpoint(
            model,
            state_dict,
            c_m,
            c_z,
            replacement_type=replacement_type,
            linear_type=linear_type,
            kernel_size=kernel_size,
            dilations=dilations,
            dilation_pattern=dilation_pattern,
            dilation_repeats=dilation_repeats,
            replacement_mode=replacement_mode,
        )
    if device.type == "cuda":
        model = model.bfloat16()

    # Wrap remaining blocks for maximum memory savings
    if wrap_all_blocks and allow_replacements:
        remaining = _wrap_remaining_blocks(
            model,
            c_m,
            c_z,
            replacement_type=replacement_type,
            linear_type=linear_type,
            kernel_size=kernel_size,
            dilations=dilations,
            dilation_pattern=dilation_pattern,
            dilation_repeats=dilation_repeats,
            replacement_mode=replacement_mode,
        )
        if remaining > 0:
            print(f"Wrapped {remaining} additional blocks without pretrained weights.")
        wrapped += remaining

    if triangle_st_only:
        if triangle_dilation_pattern is None:
            triangle_dilation_pattern = (1, 2, 4, 8)
        wrapped_ops = _wrap_triangle_ops(
            model,
            kernel_size=triangle_kernel_size,
            dilation_pattern=triangle_dilation_pattern,
            dilation_repeats=triangle_dilation_repeats,
            ckpt_replacement_only=True,
        )
        print(f"Applied triangle-only straight-through to {wrapped_ops} ops.", flush=True)
    if straight_through and allow_replacements and not triangle_st_only:
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
    pseudo_beta: Optional[torch.Tensor],
    pseudo_mask: Optional[torch.Tensor],
    gt_batch: Optional[Dict[str, torch.Tensor]],
    fape_cfg,
    steps: int,
    dist_cutoff: float,
    coor_cutoff: float,
    device: torch.device,
    log_fn=lambda s: print(s, flush=True),
    log_row_fn=None,
    orig_steps_per_cycle: int = 0,
    repl_steps_per_cycle: int = 1,
    lr: float = 1.0,
    dist_scale: float = 1.0,
    coor_scale: float = 0.0,
    disable_repl_coor_loss: bool = False,
    straight_through_selection: str = "cycle",
    straight_through_blocks: Tuple[int, ...] = (),
    init_seq: str = "0",
    softmax_seq: bool = False,
    optimizer: str = "SGD",
    norm_grad: bool = True,
) -> Tuple[torch.Tensor, list, list, torch.Tensor, float, Optional[float], Optional[float]]:
    if init_seq == "0":
        seq_logits = nn.Parameter(torch.zeros(seq_len, 20, device=device))
    elif init_seq == "gaussian":
        seq_logits = nn.Parameter(torch.randn(seq_len, 20, device=device))
    else:
        raise ValueError(f"Invalid initial sequence: {init_seq}")
    if optimizer == "SGD":
        optimizer = torch.optim.SGD([seq_logits], lr=lr)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam([seq_logits], lr=lr)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")
    losses = []
    seq_frames = []
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)

    cycle = orig_steps_per_cycle + repl_steps_per_cycle
    t_start = time.perf_counter()
    final_dist_loss: Optional[float] = None
    final_coor_loss: Optional[float] = None

    if straight_through_selection == "static_blocks":
        _set_straight_through_blocks(model, straight_through_blocks)
        _set_replacement_checkpoint_only(model, enabled=True)
        any_st_enabled = len(straight_through_blocks) > 0
    else:
        _set_replacement_checkpoint_only(model, enabled=False)
        any_st_enabled = True

    for step in tqdm(range(steps), desc="Optimizing sequence"):
        optimizer.zero_grad()
        if straight_through_selection == "static_blocks":
            # Mixed mode (per-block selection), no per-step toggling.
            use_straight_through = any_st_enabled
        else:
            if cycle > 0:
                pos = step % cycle
                use_straight_through = pos >= orig_steps_per_cycle
                _set_straight_through(model, enabled=use_straight_through)
            else:
                use_straight_through = True
                _set_straight_through(model, enabled=True)
        batch = make_feature_batch(seq_logits, residue_index)

        effective_coor_scale = coor_scale
        if disable_repl_coor_loss and use_straight_through:
            effective_coor_scale = 0.0

        # Skip running structure module during optimization when not needed
        if effective_coor_scale == 0.0:
            batch["return_representations"] = True
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs = model(batch)

            dist_loss = None
            coor_loss = None

            if dist_scale != 0.0:
                if "distogram_logits" in outputs:
                    dist_logits = outputs["distogram_logits"]
                else:
                    dist_logits = model.aux_heads.distogram(outputs["pair"])
                dist_loss = distogram_loss(
                    logits=dist_logits,
                    pseudo_beta=pseudo_beta,
                    pseudo_beta_mask=pseudo_mask,
                )

            if effective_coor_scale != 0.0:
                assert gt_batch is not None, "gt_batch is required when coor_scale != 0"
                # Need structure module outputs for FAPE
                out_for_loss = {
                    "sm": {
                        "frames": outputs["sm"]["frames"].float(),
                        "sidechain_frames": outputs["sm"]["sidechain_frames"].float(),
                        "positions": outputs["sm"]["positions"].float(),
                    }
                }
                batch_for_loss = dict(gt_batch)
                batch_for_loss.update(
                    compute_renamed_ground_truth(
                        batch_for_loss,
                        out_for_loss["sm"]["positions"][-1],
                    )
                )
                coor_loss = fape_loss(out_for_loss, batch_for_loss, fape_cfg)

            loss = 0.0
            if dist_loss is not None:
                loss = loss + dist_scale * dist_loss
            if coor_loss is not None:
                loss = loss + effective_coor_scale * coor_loss
        loss.backward()
        if norm_grad:
            seq_logits.grad = seq_logits.grad * seq_logits.shape[0] ** 0.5 / seq_logits.grad.norm()
        optimizer.step()

        losses.append(float(loss.item()))
        if softmax_seq:
            seq_frames.append(torch.softmax(seq_logits.detach(), dim=-1).cpu().numpy())
        else:
            seq_frames.append(seq_logits.detach().softmax(dim=-1).cpu().numpy())
        d_str = "nan" if dist_loss is None else f"{float(dist_loss.item()):.4f}"
        c_str = "nan" if coor_loss is None else f"{float(coor_loss.item()):.4f}"
        step_idx = step + 1
        log_fn(f"Step {step_idx}: total={float(loss.item()):.4f} dist={d_str} coor={c_str}")
        final_dist_loss = None if dist_loss is None else float(dist_loss.item())
        final_coor_loss = None if coor_loss is None else float(coor_loss.item())
        if log_row_fn is not None:
            log_row_fn(
                {
                    "step": step_idx,
                    "total_loss": float(loss.item()),
                    "dist_loss": None if dist_loss is None else float(dist_loss.item()),
                    "coor_loss": None if coor_loss is None else float(coor_loss.item()),
                    "use_straight_through": bool(use_straight_through),
                }
            )

        stop_dist = (dist_loss is not None) and (float(dist_loss.item()) < dist_cutoff)
        stop_coor = (coor_loss is not None) and (float(coor_loss.item()) < coor_cutoff)
        if stop_dist or stop_coor:
            break
    opt_time_s = time.perf_counter() - t_start

    batch_final = make_feature_batch(seq_logits, residue_index)
    _set_straight_through(model, enabled=True)
    outputs = model(batch_final)
    final_seq_probs = torch.softmax(seq_logits.detach(), dim=-1)
    return outputs, losses, seq_frames, final_seq_probs, opt_time_s, final_dist_loss, final_coor_loss


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
    parser.add_argument("--dist_scale", type=float, default=1.0, help="Weight for distogram loss")
    parser.add_argument("--coor_scale", type=float, default=0.0, help="Weight for coordinate (FAPE) loss")
    parser.add_argument("--dist_cutoff", type=float, default=0.05, help="Early-stop threshold for distogram loss")
    parser.add_argument("--coor_cutoff", type=float, default=0.05, help="Early-stop threshold for coordinate (FAPE) loss")
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
    parser.add_argument(
        "--disable-repl-coor-loss",
        action="store_true",
        default=False,
        help="When enabled, disable coordinate (FAPE) loss on straight-through steps",
    )
    parser.add_argument(
        "--straight_through_selection",
        type=str,
        default="cycle",
        choices=["cycle", "static_blocks"],
        help="How to choose straight-through usage: cycle (orig/repl schedule) or static_blocks (fixed per-block selection).",
    )
    parser.add_argument(
        "--straight_through_blocks",
        type=str,
        default="",
        help="Block indices (0-based, Evoformer block idx) to use straight-through when straight_through_selection=static_blocks. "
             "Format: '1-46' or '1,2,5-7'. Others will use original gradients.",
    )
    parser.add_argument(
        "--init_seq",
        type=str,
        default="0",
        help="Initial sequence to use for hallucination. 0 or gaussian",
    )
    parser.add_argument(
        "--softmax_seq",
        action="store_true",
        default=False,
        help="Apply softmax to the initial sequence",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use for hallucination. SGD or Adam",
    )
    parser.add_argument(
        "--norm_grad",
        action="store_true",
        default=False,
        help="Normalize the gradient of the initial sequence. Only works with SGD optimizer",
    )
    args = parser.parse_args()
    if args.dist_scale == 0.0 and args.coor_scale == 0.0:
        raise ValueError("dist_scale and coor_scale cannot both be 0")
    
    st_blocks = _parse_block_indices(args.straight_through_blocks) if args.straight_through_selection == "static_blocks" else ()

    config_path = args.config_path.expanduser()
    output_base_dir = args.output_dir.expanduser()
    config_name = config_path.stem

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg_yaml = yaml.safe_load(handle)
    replacement_type = cfg_yaml["replacement_type"]
    replacement_description = f"_orig-{args.orig_steps_per_cycle}_repl-{args.repl_steps_per_cycle}" if args.straight_through_selection == "cycle" else f"_st-blocks-{args.straight_through_blocks}"

    out_dir = output_base_dir / config_name
    out_prefix = (
        f"{args.pdb_id}_{args.chain_id}"
        f"_repl-type-{replacement_type}"
        f"{replacement_description}"
        f"_steps-{args.steps}_lr-{str(args.lr).replace('.', '-')}"
        f"_init-seq-{args.init_seq}"
        f"_softmax-seq-{args.softmax_seq}"
        f"_optimizer-{args.optimizer}"
        f"_norm-grad-{args.norm_grad}"
    )
    out_dir = out_dir / out_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    losses_csv_path = out_dir / f"{out_prefix}_losses.csv"
    meta_json_path = out_dir / f"{out_prefix}_meta.json"
    losses_csv_handle = open(losses_csv_path, "w", encoding="utf-8", newline="")
    losses_csv_writer = csv.DictWriter(
        losses_csv_handle,
        fieldnames=["step", "total_loss", "dist_loss", "coor_loss", "use_straight_through"],
    )
    losses_csv_writer.writeheader()

    def tee(msg: str):
        print(msg, flush=True)

    def log_row(row: Dict):
        losses_csv_writer.writerow(row)
        losses_csv_handle.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pdb_string = load_pdb_string(args.pdb_path, args.pdb_id)
    prot = from_pdb_string(pdb_string, chain_id=args.chain_id)
    pseudo_beta, pseudo_mask = ground_truth_pseudo_beta(prot, device)
    gt_batch = build_ground_truth_batch(prot, device) if args.coor_scale != 0.0 else None
    gt_len = int(pseudo_beta.shape[-2]) if gt_batch is None else int(gt_batch["aatype"].shape[-1])
    tee(f"Ground truth length: {gt_len}")

    model = load_model(config_path, args.checkpoint_path, device, straight_through=True)
    # Pull the canonical FAPE config from the training preset (same name as used for weights)
    preset = cfg_yaml.get("config_preset", "model_1_ptm")
    fape_cfg = model_config(preset, train=True).loss.fape

    outputs, losses, seq_frames, final_seq_probs, opt_time_s, final_dist_loss, final_coor_loss = optimize_sequence(
        model,
        seq_len=gt_len,
        pseudo_beta=pseudo_beta if args.dist_scale != 0.0 else None,
        pseudo_mask=pseudo_mask if args.dist_scale != 0.0 else None,
        gt_batch=gt_batch,
        fape_cfg=fape_cfg,
        steps=args.steps,
        dist_cutoff=args.dist_cutoff,
        coor_cutoff=args.coor_cutoff,
        device=device,
        log_fn=tee,
        log_row_fn=log_row,
        orig_steps_per_cycle=args.orig_steps_per_cycle,
        repl_steps_per_cycle=args.repl_steps_per_cycle,
        lr=args.lr,
        dist_scale=args.dist_scale,
        coor_scale=args.coor_scale,
        disable_repl_coor_loss=args.disable_repl_coor_loss,
        straight_through_selection=args.straight_through_selection,
        straight_through_blocks=st_blocks,
        init_seq=args.init_seq,
        softmax_seq=args.softmax_seq,
        optimizer=args.optimizer,
        norm_grad=args.norm_grad,
    )

    peak_mem_allocated_bytes = None
    peak_mem_reserved_bytes = None
    if device.type == "cuda":
        peak_mem_allocated_bytes = int(torch.cuda.max_memory_allocated())
        peak_mem_reserved_bytes = int(torch.cuda.max_memory_reserved())

    loss_plot = out_dir / f"{out_prefix}_loss.png"
    gif_path = out_dir / f"{out_prefix}_seq.gif"
    pdb_out = out_dir / f"{out_prefix}_final_ca.pdb"

    y_label = f"Total loss ({args.dist_scale} dist + {args.coor_scale} coor)"
    save_loss_plot(
        losses,
        loss_plot,
        title=f"{out_prefix} loss trajectory for {args.pdb_id}",
        y_label=y_label,
    )
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
    tee(f"Saved final PDB to {pdb_out}")
    tee(f"Optimization wall time: {opt_time_s:.2f}s")
    losses_csv_handle.close()

    gt_pdb_for_usalign = out_dir / f"{out_prefix}_gt_for_usalign.pdb"
    _write_gt_pdb_for_usalign(pdb_string=pdb_string, chain_id=args.chain_id, out_path=gt_pdb_for_usalign)
    usalign = subprocess.run(
        ["USalign", str(gt_pdb_for_usalign), str(pdb_out)],
        capture_output=True,
        text=True,
        check=False,
    )
    usalign_tm_score = _parse_usalign_tm_score(usalign.stdout)

    meta = {
        "pdb_id": args.pdb_id,
        "chain_id": args.chain_id,
        "steps_requested": args.steps,
        "dist_scale": args.dist_scale,
        "coor_scale": args.coor_scale,
        "dist_cutoff": args.dist_cutoff,
        "coor_cutoff": args.coor_cutoff,
        "disable_repl_coor_loss": bool(args.disable_repl_coor_loss),
        "orig_steps_per_cycle": args.orig_steps_per_cycle,
        "repl_steps_per_cycle": args.repl_steps_per_cycle,
        "lr": args.lr,
        "ground_truth_length": gt_len,
        "predicted_length": int(len(pred_seq_idx)),
        "optimization_wall_time_s": float(opt_time_s),
        "final_total_loss": None if len(losses) == 0 else float(losses[-1]),
        "final_dist_loss": final_dist_loss,
        "final_coor_loss": final_coor_loss,
        "peak_gpu_memory_allocated_bytes": peak_mem_allocated_bytes,
        "peak_gpu_memory_reserved_bytes": peak_mem_reserved_bytes,
        "straight_through_selection": args.straight_through_selection,
        "straight_through_blocks": list(st_blocks),
        "usalign": {
            "command": ["USalign", str(gt_pdb_for_usalign), str(pdb_out)],
            "exit_code": int(usalign.returncode),
            "tm_score": usalign_tm_score,
        },
        "losses_csv": str(losses_csv_path),
        "loss_plot": str(loss_plot),
        "sequence_gif": str(gif_path),
        "final_pdb": str(pdb_out),
        "gt_pdb_for_usalign": str(gt_pdb_for_usalign),
    }
    with open(meta_json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)
    tee(f"Wrote losses CSV to {losses_csv_path}")
    tee(f"Wrote metadata JSON to {meta_json_path}")


if __name__ == "__main__":
    main()
