#!/usr/bin/env python
"""
FP16 vs FP32 parity test for AF2Rank confidence outputs on 8IM5_A.

Pre-reconstructs CA-only templates via cg2all once, then scores each template
in fp32 and fp16 (autocast), comparing pTM, pLDDT, pAE, and per-residue metrics.
"""
import gc
import os
import sys
import time
import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
REFERENCE_CIF = "/home/gridsan/cou/data/bad_afdb/pdb/IM/8IM5.cif"
INFERENCE_DIR = (
    "/home/gridsan/cou/proteina/inference/"
    "inference_seq_cond_sampling_ca_dssp_beta-2.5-2.0_finetune-all_v1.6_"
    "default-fold_21-seq-S25_128-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise"
)
PROTEIN_ID = "8IM5_A"
TEMPLATE_DIR = os.path.join(
    INFERENCE_DIR, PROTEIN_ID,
    "af2rank_on_proteinebm_top_k", "staged_topk_templates",
)
TEMPLATE_NAMES = ["8IM5_A_0.pdb", "8IM5_A_50.pdb", "8IM5_A_124.pdb",
                  "8IM5_A_167.pdb", "8IM5_A_225.pdb"]


def main():
    # ── Import scorer (needs proteina on path) ───────────────────────────────
    sys.path.insert(0, "/home/gridsan/cou/proteina")
    from proteinfoundation.af2rank_evaluation.af2rank_openfold_scorer import (
        OpenFoldAF2Rank,
        reconstruct_all_atom,
    )
    from openfold.np import residue_constants as rc

    # ── Step 1: Pre-reconstruct CA-only templates ────────────────────────────
    print("=== Step 1: Pre-reconstructing CA-only templates via cg2all ===")
    template_paths = [os.path.join(TEMPLATE_DIR, t) for t in TEMPLATE_NAMES]
    allatom_map = {}  # ca_pdb -> allatom_pdb
    for ca_pdb in template_paths:
        allatom = reconstruct_all_atom(ca_pdb)
        if allatom:
            allatom_map[ca_pdb] = allatom
            print(f"  Reconstructed: {os.path.basename(ca_pdb)}")
        else:
            print(f"  Already all-atom or failed: {os.path.basename(ca_pdb)}")
    print()

    # ── Step 2: Load scorer ──────────────────────────────────────────────────
    print("=== Step 2: Loading AF2Rank scorer (model_1_ptm, fp32) ===")
    scorer = OpenFoldAF2Rank(
        reference_pdb=REFERENCE_CIF,
        chain="A",
        model_name="model_1_ptm",
        recycles=3,
        use_deepspeed_evoformer_attention=False,
        use_cuequivariance_attention=False,
        use_cuequivariance_multiplicative_update=False,
    )
    print(f"  Model dtype: {next(scorer.model.model.parameters()).dtype}")
    print()

    # ── Step 3: Score in fp32 ────────────────────────────────────────────────
    print("=== Step 3: Scoring in FP32 ===")
    fp32_results = []
    fp32_raw = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for ca_pdb in template_paths:
        pdb_to_score = allatom_map.get(ca_pdb, ca_pdb)
        original_pdb = ca_pdb if ca_pdb in allatom_map else None

        # Use _featurize to get batch, then call model directly for raw output
        batch, template_coords = scorer._featurize(
            pdb_to_score, _original_pdb=original_pdb, seed=0,
        )
        with torch.no_grad():
            out = scorer.model.model(batch)

        # Extract raw outputs before averaging
        raw = {
            "ptm": float(out["ptm_score"].item()) if "ptm_score" in out else 0.0,
            "plddt_per_res": out["plddt"].detach().cpu().numpy().copy(),  # [N_res] 0-100
            "plddt_mean": float(out["plddt"].mean().item()) / 100.0,
        }
        if "predicted_aligned_error" in out:
            raw["pae_matrix"] = out["predicted_aligned_error"].detach().cpu().numpy().copy()
            raw["pae_mean"] = float(out["predicted_aligned_error"].mean().item())
        if "final_atom_positions" in out:
            ca_idx = rc.atom_order["CA"]
            raw["ca_coords"] = out["final_atom_positions"][:, ca_idx, :].detach().cpu().numpy().copy()
        raw["composite"] = raw["ptm"] * raw["plddt_mean"]
        fp32_raw.append(raw)

        scores = scorer._extract_scores(out, template_coords)
        fp32_results.append(scores)
        print(f"  {os.path.basename(ca_pdb)}: pTM={raw['ptm']:.6f}  pLDDT={raw['plddt_mean']:.6f}  "
              f"pAE={raw.get('pae_mean', 0):.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    fp32_time = time.perf_counter() - t0
    fp32_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Total: {fp32_time:.1f}s, peak GPU: {fp32_peak:.2f} GB")
    print()

    # ── Step 4: Score in fp16 (model.half() + autocast) ─────────────────────
    # model.half() gives real fp16 weights/activations for memory savings.
    # autocast triggers is_fp16_enabled() → disables attn_core_inplace_cuda
    # (which only supports fp32/bf16) → falls back to naive PyTorch attention.
    # autocast also triggers fp32 upcasting in confidence heads for stability.
    print("=== Step 4: Scoring in FP16 (model.half() + autocast) ===")
    scorer.model.model.half()
    print(f"  Model dtype after .half(): {next(scorer.model.model.parameters()).dtype}")
    fp16_results = []
    fp16_raw = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for ca_pdb in template_paths:
        pdb_to_score = allatom_map.get(ca_pdb, ca_pdb)
        original_pdb = ca_pdb if ca_pdb in allatom_map else None

        batch, template_coords = scorer._featurize(
            pdb_to_score, _original_pdb=original_pdb, seed=0,
        )
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = scorer.model.model(batch)

        raw = {
            "ptm": float(out["ptm_score"].item()) if "ptm_score" in out else 0.0,
            "plddt_per_res": out["plddt"].detach().float().cpu().numpy().copy(),
            "plddt_mean": float(out["plddt"].float().mean().item()) / 100.0,
        }
        if "predicted_aligned_error" in out:
            raw["pae_matrix"] = out["predicted_aligned_error"].detach().float().cpu().numpy().copy()
            raw["pae_mean"] = float(out["predicted_aligned_error"].float().mean().item())
        if "final_atom_positions" in out:
            ca_idx = rc.atom_order["CA"]
            raw["ca_coords"] = out["final_atom_positions"][:, ca_idx, :].detach().float().cpu().numpy().copy()
        raw["composite"] = raw["ptm"] * raw["plddt_mean"]
        fp16_raw.append(raw)

        scores = scorer._extract_scores(out, template_coords)
        fp16_results.append(scores)
        print(f"  {os.path.basename(ca_pdb)}: pTM={raw['ptm']:.6f}  pLDDT={raw['plddt_mean']:.6f}  "
              f"pAE={raw.get('pae_mean', 0):.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    fp16_time = time.perf_counter() - t0
    fp16_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Total: {fp16_time:.1f}s, peak GPU: {fp16_peak:.2f} GB")
    print()

    # ── Step 5: Compare ──────────────────────────────────────────────────────
    print("=" * 70)
    print("=== Comparison: FP32 vs FP16 (model.half + autocast) ===")
    print("=" * 70)

    # Aggregate metrics table
    print(f"\n{'Template':<14}  {'Metric':<12}  {'FP32':>10}  {'FP16':>10}  {'AbsDiff':>10}  {'RelDiff%':>10}")
    print("-" * 72)
    for i, name in enumerate(TEMPLATE_NAMES):
        short = name.replace(".pdb", "")
        for metric in ["ptm", "plddt_mean", "pae_mean", "composite"]:
            v32 = fp32_raw[i].get(metric, 0)
            v16 = fp16_raw[i].get(metric, 0)
            adiff = abs(v32 - v16)
            rdiff = 100 * adiff / max(abs(v32), 1e-10)
            print(f"{short:<14}  {metric:<12}  {v32:>10.6f}  {v16:>10.6f}  {adiff:>10.6f}  {rdiff:>10.4f}")
        print()

    # Per-residue pLDDT comparison
    print("\n--- Per-residue pLDDT comparison ---")
    print(f"{'Template':<14}  {'MaxAbsDiff':>12}  {'MeanAbsDiff':>12}  {'Correlation':>12}")
    print("-" * 56)
    for i, name in enumerate(TEMPLATE_NAMES):
        short = name.replace(".pdb", "")
        p32 = fp32_raw[i]["plddt_per_res"]
        p16 = fp16_raw[i]["plddt_per_res"]
        max_diff = float(np.max(np.abs(p32 - p16)))
        mean_diff = float(np.mean(np.abs(p32 - p16)))
        corr = float(np.corrcoef(p32.flatten(), p16.flatten())[0, 1]) if len(p32) > 1 else 1.0
        print(f"{short:<14}  {max_diff:>12.4f}  {mean_diff:>12.4f}  {corr:>12.8f}")

    # PAE matrix comparison
    if "pae_matrix" in fp32_raw[0]:
        print("\n--- PAE matrix comparison ---")
        print(f"{'Template':<14}  {'MaxAbsDiff':>12}  {'MeanAbsDiff':>12}")
        print("-" * 42)
        for i, name in enumerate(TEMPLATE_NAMES):
            short = name.replace(".pdb", "")
            m32 = fp32_raw[i]["pae_matrix"]
            m16 = fp16_raw[i]["pae_matrix"]
            max_diff = float(np.max(np.abs(m32 - m16)))
            mean_diff = float(np.mean(np.abs(m32 - m16)))
            print(f"{short:<14}  {max_diff:>12.4f}  {mean_diff:>12.4f}")

    # CA coordinate RMSD
    if "ca_coords" in fp32_raw[0]:
        print("\n--- Predicted CA coordinate RMSD (fp32 vs fp16) ---")
        print(f"{'Template':<14}  {'RMSD(A)':>12}  {'MaxDeviation(A)':>16}")
        print("-" * 46)
        for i, name in enumerate(TEMPLATE_NAMES):
            short = name.replace(".pdb", "")
            c32 = fp32_raw[i]["ca_coords"]
            c16 = fp16_raw[i]["ca_coords"]
            rmsd = float(np.sqrt(np.mean(np.sum((c32 - c16) ** 2, axis=-1))))
            max_dev = float(np.max(np.sqrt(np.sum((c32 - c16) ** 2, axis=-1))))
            print(f"{short:<14}  {rmsd:>12.4f}  {max_dev:>16.4f}")

    # Memory summary
    print(f"\n--- Memory & Runtime ---")
    print(f"  FP32: {fp32_peak:.2f} GB peak, {fp32_time:.1f}s")
    print(f"  FP16: {fp16_peak:.2f} GB peak, {fp16_time:.1f}s")
    print(f"  Memory savings: {(1 - fp16_peak / fp32_peak) * 100:.1f}%")
    print(f"  Speedup: {fp32_time / fp16_time:.2f}x")

    # ── Cleanup temp cg2all files ────────────────────────────────────────────
    for tmp in allatom_map.values():
        if os.path.exists(tmp):
            os.unlink(tmp)

    print("\nDone.")


if __name__ == "__main__":
    main()
