"""Phase 5: AF2-initial-guess-style target-conditioned scoring proof-of-concept.

Target = human PD-L1 (chain A of PDB 4ZQK, templated with its real structure).
"Binder" = human PD-1 (chain B of 4ZQK) -- PD-L1's real, natural binding partner. Using a real
known complex (not a synthetic design) means there's real ground truth to check against: does
feeding WS5 [PD1_seq, PDL1_seq] with ONLY PD-L1 templated produce a PD-1 placement that's
anywhere close to where PD-1 REALLY sits relative to PD-L1 in the solved 4ZQK structure?

Method: Kabsch-align the model's predicted PD-L1 (target) CA coords onto PD-L1's REAL CA coords
(this is just checking the model reproduces the thing it was templated on -- a sanity floor),
then apply that SAME rigid transform to the model's predicted PD-1 (binder) CA coords, and
compare against PD-1's REAL CA coords in the same original (uncentered) 4ZQK frame. This is a
feasibility/mechanism test, not a claim about real binder design capability -- WS5 was never
trained in this two-chain configuration (see SLIM_DOWNSTREAM.md WS6 section for the caveat).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5_templated, score_binder_target, resolve_ws5_ckpt
from template_feat_builder import build_template_features
from openfold.data import mmcif_parsing
from openfold.data.templates import _get_atom_positions
from openfold.np import residue_constants as rc

CKPT = os.environ.get("CKPT") or None
DEVICE = os.environ.get("DEVICE", "cuda:0")
MMCIF_PATH = "/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files/4zqk.cif"
TARGET_CHAIN = "A"  # PD-L1
BINDER_CHAIN = "B"  # PD-1 (real natural partner, used here as ground truth, not a design)
CA_IDX = rc.atom_order["CA"]


def kabsch(mobile, ref):
    """Returns (R, t) such that mobile @ R.T + t ~= ref (least squares)."""
    mc, rc_ = mobile.mean(0), ref.mean(0)
    a, b = mobile - mc, ref - rc_
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = rc_ - mc @ r.T
    return r, t


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def main():
    with open(MMCIF_PATH) as f:
        cif_string = f.read()
    parsed = mmcif_parsing.parse(file_id="4zqk", mmcif_string=cif_string)
    obj = parsed.mmcif_object

    target_seq = obj.chain_to_seqres[TARGET_CHAIN]
    binder_seq = obj.chain_to_seqres[BINDER_CHAIN]
    print(f"target (PD-L1, chain {TARGET_CHAIN}): {len(target_seq)} aa", flush=True)
    print(f"binder (PD-1, chain {BINDER_CHAIN}): {len(binder_seq)} aa", flush=True)

    # ground truth, uncentered, both chains in the SAME original frame (for the RMSD check below)
    target_real_pos, target_real_mask = _get_atom_positions(
        obj, TARGET_CHAIN, max_ca_ca_distance=150.0, _zero_center_positions=False)
    binder_real_pos, binder_real_mask = _get_atom_positions(
        obj, BINDER_CHAIN, max_ca_ca_distance=150.0, _zero_center_positions=False)
    target_real_ca = target_real_pos[:, CA_IDX, :]
    binder_real_ca = binder_real_pos[:, CA_IDX, :]
    target_valid = target_real_mask[:, CA_IDX].astype(bool)
    binder_valid = binder_real_mask[:, CA_IDX].astype(bool)
    if not target_valid.all():
        print(f"target: {(~target_valid).sum()} residues have no resolved CA density "
              f"(disordered loop/terminus) -- excluding from RMSD", flush=True)
    if not binder_valid.all():
        print(f"binder: {(~binder_valid).sum()} residues have no resolved CA density "
              f"(disordered loop/terminus) -- excluding from RMSD", flush=True)

    template_feats = build_template_features(
        mmcif_path=MMCIF_PATH, pdb_id="4zqk", target_chain_id=TARGET_CHAIN,
        query_sequence=binder_seq + target_seq, target_offset=len(binder_seq), device=DEVICE)
    print("template features built", flush=True)

    model = load_ws5_templated(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    print("model loaded (templates ON, strict load)", flush=True)

    out = score_binder_target(model, binder_seq, target_seq, template_feats, device=DEVICE)
    print(f"target_plddt_mean={out['target_plddt_mean']:.4f}  "
          f"binder_plddt_mean={out['binder_plddt_mean']:.4f}", flush=True)

    # sanity floor: does the model reproduce the thing it was templated on?
    # AF2's output lives in an arbitrary internal frame, not the PDB deposition's frame -- must
    # Kabsch-align before comparing, or the RMSD is meaningless (raw comparison mistake caught post-hoc).
    r, t = kabsch(out["target_ca"][target_valid], target_real_ca[target_valid])
    target_aligned = out["target_ca"][target_valid] @ r.T + t
    target_self_rmsd = rmsd(target_aligned, target_real_ca[target_valid])
    print(f"predicted-target vs real-PDL1 RMSD, Kabsch-aligned ({target_valid.sum()}/{len(target_valid)} "
          f"resolved residues): {target_self_rmsd:.3f} A", flush=True)

    binder_pred_aligned = out["binder_ca"] @ r.T + t
    binder_rmsd = rmsd(binder_pred_aligned[binder_valid], binder_real_ca[binder_valid])
    print(f"predicted-binder (after aligning target onto real PD-L1) vs real PD-1 RMSD "
          f"({binder_valid.sum()}/{len(binder_valid)} resolved residues): "
          f"{binder_rmsd:.3f} A", flush=True)

    np.savez("/home/jupyter-chenxi/prune_work/eval_out/phase5_pdl1_pd1.npz",
             pred_target_ca=out["target_ca"], pred_binder_ca=out["binder_ca"],
             real_target_ca=target_real_ca, real_binder_ca=binder_real_ca,
             binder_pred_aligned=binder_pred_aligned,
             target_valid=target_valid, binder_valid=binder_valid)
    print("saved raw coords -> eval_out/phase5_pdl1_pd1.npz", flush=True)


if __name__ == "__main__":
    main()
