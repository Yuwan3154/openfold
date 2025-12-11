#!/usr/bin/env python3
"""
Hallucination demo using gradient-passthrough replacement blocks.

This script:
- Loads an AlphaFold model and replacement weights.
- Wraps any replacement-enabled Evoformer blocks with a gradient passthrough:
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
from typing import Dict, Tuple

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
import openfold.np.residue_constants as rc


class GradientPassthroughBlock(nn.Module):
    """Wrapper that keeps forward identical to the original block output."""

    def __init__(self, original_block: nn.Module, replacement_block: nn.Module, block_idx: int):
        super().__init__()
        self.original_block = original_block
        self.replacement_block = replacement_block
        self.block_idx = block_idx
        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            orig_m, orig_z = self.original_block(*args, **kwargs)
        rep_m, rep_z = self.replacement_block(*args, **kwargs)
        m_out = rep_m + orig_m.detach() - rep_m.detach()
        z_out = rep_z + orig_z.detach() - rep_z.detach()
        return m_out, z_out


def apply_gradient_passthrough(model: nn.Module) -> int:
    wrapped = 0
    for idx, block in enumerate(model.evoformer.blocks):
        if hasattr(block, "original_block") and hasattr(block, "replacement_block"):
            model.evoformer.blocks[idx] = GradientPassthroughBlock(
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


def ground_truth_pseudo_beta(pdb_string: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    from openfold.np import protein

    prot = protein.from_pdb_string(pdb_string)
    aatype = torch.tensor(prot.aatype, device=device, dtype=torch.long)
    all_pos = torch.tensor(prot.atom_positions, device=device, dtype=torch.float32)
    all_mask = torch.tensor(prot.atom_mask, device=device, dtype=torch.float32)
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
    aatype_int = torch.argmax(seq_probs21, dim=-1).to(torch.long)

    def add_cycle(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1).expand(*x.shape, recycle_dim)

    batch = {
        "aatype": add_cycle(aatype_int),
        "target_feat": add_cycle(target_feat),
        "residue_index": add_cycle(residue_index),
        "msa_feat": add_cycle(msa_feat),
        "seq_mask": add_cycle(seq_mask),
        "msa_mask": add_cycle(msa_mask),
        "pair_mask": add_cycle(pair_mask),
        "extra_msa_mask": add_cycle(extra_msa_mask),
        "return_representations": False,
    }
    return batch


def ca_from_atom37(atom37: np.ndarray) -> np.ndarray:
    ca_idx = rc.atom_order["CA"]
    return atom37[:, ca_idx, :]


def write_ca_pdb(path: Path, ca_coords: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        atom_id = 1
        for i, xyz in enumerate(ca_coords):
            x, y, z = xyz
            line = (
                f"ATOM  {atom_id:5d}  CA  ALA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            handle.write(line)
            atom_id += 1
        handle.write("END\n")


def save_loss_plot(losses, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Mean distogram loss")
    plt.title("Hallucination loss trajectory")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_sequence_gif(seq_history, path: Path):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in seq_history:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(frame, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Amino acid channel")
        ax.set_ylabel("Residue")
        ax.set_title("Sequence logits (softmax)")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    imageio.mimsave(path, images, fps=4)


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    passthrough: bool = True,
) -> AlphaFold:
    with open(config_path, "r", encoding="utf-8") as handle:
        import yaml

        cfg_yaml = yaml.safe_load(handle)
    preset = cfg_yaml.get("config_preset", "model_1_ptm")
    weights_path = cfg_yaml["weights_path"]

    cfg = model_config(preset)
    cfg.model.template.enabled = False
    cfg.model.extra_msa.enabled = False

    model = AlphaFold(cfg).to(device)
    import_jax_weights_(model, str(Path.home() / weights_path), version=preset)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    wrapped = apply_gradient_passthrough(model) if passthrough else 0
    if wrapped == 0:
        print("No adaptive/replacement blocks found; running base model.")
    else:
        print(f"Applied gradient passthrough to {wrapped} Evoformer blocks.")
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
) -> Tuple[torch.Tensor, list, list]:
    seq_logits = nn.Parameter(torch.randn(seq_len, 20, device=device))
    optimizer = torch.optim.Adam([seq_logits], lr=0.5)
    losses = []
    seq_frames = []
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)

    for step in range(steps):
        optimizer.zero_grad()
        batch = make_feature_batch(seq_logits, residue_index)
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
        print(f"Step {step+1}: loss={loss.item():.4f}")
        if loss.item() < loss_cutoff:
            break

    batch_final = make_feature_batch(seq_logits, residue_index)
    outputs = model(batch_final)
    return outputs, losses, seq_frames


def main():
    parser = argparse.ArgumentParser(description="Hallucination with gradient-passthrough replacements")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("/Users/Chenxi/SOLab/AFdistill/configs/repr_distill_per-block.yaml"),
        help="Training config used for replacement blocks",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("/Users/Chenxi/SOLab/AFdistill/results/repr_distill_per-block/checkpoints/last.ckpt"),
        help="Checkpoint containing replacement weights",
    )
    parser.add_argument("--pdb_id", type=str, default="6mrr", help="PDB identifier to use")
    parser.add_argument("--pdb_path", type=Path, default=Path("/Users/Chenxi/6mrr.pdb"), help="Optional local PDB path")
    parser.add_argument("--steps", type=int, default=100, help="Maximum optimization steps")
    parser.add_argument("--loss_cutoff", type=float, default=0.05, help="Early-stop distogram loss threshold")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/Users/Chenxi/SOLab/AFdistill/outputs"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdb_string = load_pdb_string(args.pdb_path, args.pdb_id)
    pseudo_beta, pseudo_mask = ground_truth_pseudo_beta(pdb_string, device)

    model = load_model(args.config_path, args.checkpoint_path, device, passthrough=True)
    outputs, losses, seq_frames = optimize_sequence(
        model,
        seq_len=pseudo_beta.shape[-2],
        pseudo_beta=pseudo_beta,
        pseudo_mask=pseudo_mask,
        steps=args.steps,
        loss_cutoff=args.loss_cutoff,
        device=device,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_plot = out_dir / "hallucination_loss.png"
    gif_path = out_dir / "hallucination_seq.gif"
    pdb_out = out_dir / "hallucination_final_ca.pdb"

    save_loss_plot(losses, loss_plot)
    save_sequence_gif(seq_frames, gif_path)

    atom37 = outputs["final_atom_positions"][0].detach().cpu().numpy()
    ca_coords = ca_from_atom37(atom37)
    write_ca_pdb(pdb_out, ca_coords)

    print(f"Saved loss curve to {loss_plot}")
    print(f"Saved sequence GIF to {gif_path}")
    print(f"Saved final CA-only PDB to {pdb_out}")


if __name__ == "__main__":
    main()
