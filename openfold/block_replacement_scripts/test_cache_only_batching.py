#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
from contextlib import nullcontext

import torch

from openfold.block_replacement_scripts.block_data_io import load_block_sample, sanitize_id
from openfold.block_replacement_scripts.train_per_block_parallel import (
    DilatedConvEvoformerReplacement,
    SimpleEvoformerReplacement,
)


def _find_two_seq_ids_with_cache(
    *,
    dataset_path: Path,
    cache_dir: Path,
    block_idx: int,
    ext: str,
) -> Tuple[Tuple[str, int], Tuple[str, int]]:
    found = []
    with dataset_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            seq_id = parts[0]
            seq = parts[1]
            n = len(seq)
            sid = sanitize_id(seq_id)
            cache_path = cache_dir / f"block_{block_idx:02d}" / f"{sid}.{ext}"
            if not cache_path.exists():
                continue
            found.append((seq_id, n))
            if len(found) >= 2 and found[0][1] != found[1][1]:
                return found[0], found[1]
    raise RuntimeError("Could not find two cached seq_ids with different lengths")


def _masked_mse_m(m_pred: torch.Tensor, m_tgt: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    c_m = int(m_pred.shape[-1])
    return (((m_pred - m_tgt) ** 2) * seq_mask.unsqueeze(-1)).sum() / (seq_mask.sum() * c_m)


def _masked_mse_z(z_pred: torch.Tensor, z_tgt: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    c_z = int(z_pred.shape[-1])
    return (((z_pred - z_tgt) ** 2) * pair_mask.unsqueeze(-1)).sum() / (pair_mask.sum() * c_z)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--block_data_dir", type=str, required=True)
    p.add_argument(
        "--block_data_format",
        type=str,
        default="df11.safetensors",
        choices=["pt", "pt.gz", "safetensors", "safetensors.gz", "safetensors.znn", "df11.safetensors"],
    )
    p.add_argument("--block_idx", type=int, default=1)
    p.add_argument("--replacement_type", type=str, default="conv", choices=["linear", "conv"])
    p.add_argument("--linear_type", type=str, default="full", choices=["full", "diagonal", "affine"])
    p.add_argument("--replacement_mode", type=str, default="shared_proj")
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--dilations", type=str, default="1,2,4,8,16")
    p.add_argument("--trained_models_dir", type=str, required=True)
    args = p.parse_args()

    dataset_path = Path(args.dataset_path)
    cache_dir = Path(args.block_data_dir)
    ext = str(args.block_data_format)
    block_idx = int(args.block_idx)

    if ext == "df11.safetensors":
        if not torch.cuda.is_available():
            raise RuntimeError("df11.safetensors test requires CUDA for GPU decode")
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    (seq_id0, n0), (seq_id1, n1) = _find_two_seq_ids_with_cache(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        block_idx=block_idx,
        ext=ext,
    )
    print(f"picked seqs: {seq_id0} (N={n0}), {seq_id1} (N={n1})", flush=True)

    def load_one(seq_id: str):
        sid = sanitize_id(seq_id)
        cache_path = cache_dir / f"block_{block_idx:02d}" / f"{sid}.{ext}"
        cached = load_block_sample(cache_path, map_location=device if ext == "df11.safetensors" else "cpu")
        m_in = cached["input"]["m"]
        z_in = cached["input"]["z"]
        m_tgt = cached["output"]["m"]
        z_tgt = cached["output"]["z"]
        if m_in.dim() == 3 and m_in.shape[0] == 1:
            m_in = m_in.squeeze(0)
        if z_in.dim() == 4 and z_in.shape[0] == 1:
            z_in = z_in.squeeze(0)
        if m_tgt.dim() == 3 and m_tgt.shape[0] == 1:
            m_tgt = m_tgt.squeeze(0)
        if z_tgt.dim() == 4 and z_tgt.shape[0] == 1:
            z_tgt = z_tgt.squeeze(0)
        return m_in.to(device=device), z_in.to(device=device), m_tgt.to(device=device), z_tgt.to(device=device)

    # Infer channel dims from cache (avoid hardcoding c_m/c_z).
    with torch.no_grad():
        m_tmp, z_tmp, _, _ = load_one(seq_id0)
    c_m = int(m_tmp.shape[-1])
    c_z = int(z_tmp.shape[-1])
    print(f"inferred dims: c_m={c_m} c_z={c_z}", flush=True)

    # Build replacement block with inferred dims
    torch.manual_seed(0)
    if args.replacement_type == "linear":
        replacement_block = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type=args.linear_type)
        checkpoint_subdir = args.linear_type
    else:
        dilations = tuple(int(d) for d in str(args.dilations).split(",") if str(d).strip() != "")
        replacement_block = DilatedConvEvoformerReplacement(
            c_m=c_m,
            c_z=c_z,
            kernel_size=int(args.kernel_size),
            dilations=dilations,
            mode=args.replacement_mode,
        )
        d_str = "-".join(str(d) for d in dilations)
        checkpoint_subdir = f"conv_{args.replacement_mode}_k{int(args.kernel_size)}_d{d_str}"

    block_path = Path(args.trained_models_dir) / f"block_{block_idx:02d}" / checkpoint_subdir / "best_model.ckpt"
    if block_path.exists():
        ckpt = torch.load(block_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        if any(k.startswith("replacement_block.") for k in state_dict.keys()):
            state_dict = {k.replace("replacement_block.", ""): v for k, v in state_dict.items()}
        replacement_block.load_state_dict(state_dict, strict=True)
        print(f"loaded ckpt: {block_path}", flush=True)
    else:
        print(f"ckpt missing; using random init: {block_path}", flush=True)

    replacement_block = replacement_block.to(device=device).eval()

    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
        m0, z0, mt0, zt0 = load_one(seq_id0)
        m1, z1, mt1, zt1 = load_one(seq_id1)

        # Individual runs
        def run_single(m_in: torch.Tensor, z_in: torch.Tensor, m_tgt: torch.Tensor, z_tgt: torch.Tensor):
            n = int(m_in.shape[0])
            seq_mask = torch.ones((1, n), dtype=torch.float32, device=device)
            msa_mask = seq_mask.unsqueeze(1)  # [1, 1, N]
            pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # [1, N, N]
            m_in_msa = m_in.unsqueeze(0).unsqueeze(1)  # [1, 1, N, C]
            z_in_b = z_in.unsqueeze(0)  # [1, N, N, C]
            m_pred_msa, z_pred = replacement_block(
                m_in_msa,
                z_in_b,
                msa_mask,
                pair_mask,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                use_flash=False,
                inplace_safe=False,
                _mask_trans=True,
            )
            m_pred = m_pred_msa[..., 0, :, :]  # [1, N, C]
            z_pred = z_pred  # [1, N, N, C]
            m_loss = _masked_mse_m(m_pred, m_tgt.unsqueeze(0), seq_mask)
            z_loss = _masked_mse_z(z_pred, z_tgt.unsqueeze(0), pair_mask)
            return m_pred.squeeze(0), z_pred.squeeze(0), (m_loss + z_loss).detach()

        with autocast_ctx:
            m0_pred, z0_pred, loss0 = run_single(m0, z0, mt0, zt0)
            m1_pred, z1_pred, loss1 = run_single(m1, z1, mt1, zt1)

        # Padded batch run
        n_max = max(int(m0.shape[0]), int(m1.shape[0]))
        bsz = 2
        c_m = int(m0.shape[-1])
        c_z = int(z0.shape[-1])

        m_in_p = torch.zeros((bsz, n_max, c_m), dtype=m0.dtype, device=device)
        m_tgt_p = torch.zeros((bsz, n_max, c_m), dtype=mt0.dtype, device=device)
        z_in_p = torch.zeros((bsz, n_max, n_max, c_z), dtype=z0.dtype, device=device)
        z_tgt_p = torch.zeros((bsz, n_max, n_max, c_z), dtype=zt0.dtype, device=device)

        m_in_p[0, : m0.shape[0]] = m0
        m_in_p[1, : m1.shape[0]] = m1
        m_tgt_p[0, : mt0.shape[0]] = mt0
        m_tgt_p[1, : mt1.shape[0]] = mt1
        z_in_p[0, : z0.shape[0], : z0.shape[1]] = z0
        z_in_p[1, : z1.shape[0], : z1.shape[1]] = z1
        z_tgt_p[0, : zt0.shape[0], : zt0.shape[1]] = zt0
        z_tgt_p[1, : zt1.shape[0], : zt1.shape[1]] = zt1

        seq_mask = torch.zeros((bsz, n_max), dtype=torch.float32, device=device)
        seq_mask[0, : m0.shape[0]] = 1.0
        seq_mask[1, : m1.shape[0]] = 1.0
        msa_mask = seq_mask.unsqueeze(1)  # [B, 1, N]
        pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # [B, N, N]

        m_in_msa = m_in_p.unsqueeze(1)
        with autocast_ctx:
            m_pred_msa_b, z_pred_b = replacement_block(
                m_in_msa,
                z_in_p,
                msa_mask,
                pair_mask,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                use_flash=False,
                inplace_safe=False,
                _mask_trans=True,
            )
        m_pred_b = m_pred_msa_b[..., 0, :, :]  # [B, N, C]

        # Compare outputs sliced to true lengths
        atol = 2e-2 if m0.dtype in (torch.float16, torch.bfloat16) else 1e-5
        rtol = 2e-2 if m0.dtype in (torch.float16, torch.bfloat16) else 1e-5

        ok_m0 = torch.allclose(m_pred_b[0, : m0.shape[0]], m0_pred, rtol=rtol, atol=atol)
        ok_m1 = torch.allclose(m_pred_b[1, : m1.shape[0]], m1_pred, rtol=rtol, atol=atol)
        ok_z0 = torch.allclose(z_pred_b[0, : z0.shape[0], : z0.shape[1]], z0_pred, rtol=rtol, atol=atol)
        ok_z1 = torch.allclose(z_pred_b[1, : z1.shape[0], : z1.shape[1]], z1_pred, rtol=rtol, atol=atol)

        # Compare per-sample losses computed from batched predictions
        loss0_b = _masked_mse_m(m_pred_b[0:1, : m0.shape[0]], mt0.unsqueeze(0), torch.ones((1, m0.shape[0]), device=device)) + _masked_mse_z(
            z_pred_b[0:1, : z0.shape[0], : z0.shape[1]], zt0.unsqueeze(0), torch.ones((1, z0.shape[0], z0.shape[1]), device=device)
        )
        loss1_b = _masked_mse_m(m_pred_b[1:2, : m1.shape[0]], mt1.unsqueeze(0), torch.ones((1, m1.shape[0]), device=device)) + _masked_mse_z(
            z_pred_b[1:2, : z1.shape[0], : z1.shape[1]], zt1.unsqueeze(0), torch.ones((1, z1.shape[0], z1.shape[1]), device=device)
        )

        print(f"output_match m0={ok_m0} z0={ok_z0} m1={ok_m1} z1={ok_z1}", flush=True)
        if not ok_m1:
            d = (m_pred_b[1, : m1.shape[0]] - m1_pred).abs().max().item()
            print(f"max_abs_diff m1={d}", flush=True)
        if not ok_z1:
            d = (z_pred_b[1, : z1.shape[0], : z1.shape[1]] - z1_pred).abs().max().item()
            print(f"max_abs_diff z1={d}", flush=True)
        print(f"loss_single 0={float(loss0.item()):.6f} 1={float(loss1.item()):.6f}", flush=True)
        print(f"loss_batched 0={float(loss0_b.item()):.6f} 1={float(loss1_b.item()):.6f}", flush=True)

        if not (ok_m0 and ok_z0 and ok_m1 and ok_z1):
            raise RuntimeError("Batched outputs do not match single-sample outputs (sliced to true lengths)")

        if not torch.allclose(loss0, loss0_b, rtol=1e-3, atol=1e-3):
            raise RuntimeError("Sample 0 loss mismatch between single and batched")
        if not torch.allclose(loss1, loss1_b, rtol=1e-3, atol=1e-3):
            raise RuntimeError("Sample 1 loss mismatch between single and batched")

    print("OK: single vs padded-batch equivalence holds", flush=True)


if __name__ == "__main__":
    main()

