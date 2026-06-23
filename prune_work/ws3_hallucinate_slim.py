"""WS3: monomer hallucination with the SLIM AF2 model, reusing the validated backprop-hallucination
pipeline (openfold/block_replacement_scripts/hallucination_straight_through.py).

Flow:
  1. Build the slim model (sliced AlphaFold + best-037 EMA), templates off, num_recycle=1.
  2. optimize_sequence (distogram loss only, dist_scale=1, coor_scale=0) against a TARGET natural
     structure's pseudo-beta -> a designed sequence.
  3. Re-fold the HARD (argmax) designed sequence in BOTH the slim model and stock full AF2 (single
     forward each) so the comparison uses the same discrete sequence.
  4. Report USalign TM-scores: TM(slim_pred, target), TM(full_pred, target), and
     TM(full_pred, slim_pred) (cross-model consistency). distogram_loss alone is NOT interpretable.
  5. Write target.pdb / slim_pred.pdb / full_pred.pdb + summary.json + loss.png for rendering.

This does NOT modify hallucination_straight_through.py (other project's file); it imports and reuses it.
We are NOT using the gradient-STE block-replacement path (no blocks are wrapped on the slim model, so
the STE toggles inside optimize_sequence are inert).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.np import residue_constants as rc
from openfold.np.protein import from_pdb_string, from_prediction, to_pdb

# the validated hallucination pipeline lives next to this repo's block_replacement_scripts
_BRS = str(Path(__file__).resolve().parents[1] / "openfold" / "block_replacement_scripts")
sys.path.insert(0, _BRS)
import hallucination_straight_through as H  # noqa: E402


def build_model(kind, slim_ckpt, jax, keep, device):
    cfg = model_config("model_1_ptm")
    cfg.model.template.enabled = False
    cfg.globals.chunk_size = None
    cfg.model.num_recycle = 1
    m = AlphaFold(cfg)
    if kind == "slim":
        m.evoformer.blocks = nn.ModuleList([m.evoformer.blocks[i] for i in keep])
        ck = torch.load(slim_ckpt, map_location="cpu", weights_only=False)
        sd = ck["ema"]["params"]
        if len(set(sd.keys()) & set(m.state_dict().keys())) == 0:
            raise RuntimeError(f"slim ckpt {slim_ckpt}: 0 keys matched the sliced model")
        m.load_state_dict(sd, strict=False)
    else:
        import_jax_weights_(m, jax, version="model_1_ptm")
    m = m.to(device).float().eval()
    m.evoformer.blocks_per_ckpt = 1
    if hasattr(m, "extra_msa_stack"):
        m.extra_msa_stack.ckpt = True
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def fold_sequence(model, seq_idx, device):
    """Single forward folding a fixed (hard) sequence -> model output dict with the predicted structure."""
    L = len(seq_idx)
    seq_logits = 20.0 * F.one_hot(torch.tensor(seq_idx, device=device).clamp(max=19), 20).float()
    residue_index = torch.arange(L, device=device, dtype=torch.long)
    batch = H.make_feature_batch(seq_logits, residue_index)
    with torch.no_grad():
        return model(batch)


def out_to_pdb(out, seq_idx):
    atom37 = out["final_atom_positions"].detach().float().cpu().numpy()
    atom37_mask = out["final_atom_mask"].detach().bool().cpu().numpy() if "final_atom_mask" in out else None
    result = {"final_atom_positions": atom37, "final_atom_mask": atom37_mask}
    features = {"aatype": np.asarray(seq_idx)[None, :], "residue_index": np.arange(len(seq_idx))[None, :]}
    prot = from_prediction(features=features, result=result, remove_leading_feature_dimension=True)
    return to_pdb(prot)


_USALIGN = shutil.which("USalign") or os.path.expanduser("~/.local/bin/USalign")


def usalign_tm(pdb_a, pdb_b):
    if not os.path.exists(_USALIGN) and shutil.which(_USALIGN) is None:
        raise FileNotFoundError(f"USalign not found (tried {_USALIGN}); set PATH or pass full path")
    r = subprocess.run([_USALIGN, str(pdb_a), str(pdb_b)], capture_output=True, text=True, check=False)
    return H._parse_usalign_tm_score(r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_pdb", default="/home/jupyter-chenxi/data/7ad5_example/7ad5_A_cath_3.40.50.720_0_cg2all.pdb")
    ap.add_argument("--chain", default="A")
    ap.add_argument("--slim_ckpt", default="/home/jupyter-chenxi/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt")
    ap.add_argument("--jax", default="/home/jupyter-chenxi/params/params_model_1_ptm.npz")
    ap.add_argument("--keep", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47")
    ap.add_argument("--steps", type=int, default=100)          # main() default
    ap.add_argument("--lr", type=float, default=1.0)           # main() default
    ap.add_argument("--optimizer", default="SGD")              # main() default
    ap.add_argument("--init_seq", default="0")                 # main() default
    ap.add_argument("--norm_grad", action=argparse.BooleanOptionalAction, default=True)  # ColabDesign-standard normalized-grad
    ap.add_argument("--dtype", default="fp32")                 # conservative for a single reported run
    ap.add_argument("--out_dir", default="/home/jupyter-chenxi/runs/ws3_hallucinate")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    keep = [int(x) for x in args.keep.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- target structure ---
    pdb_string = open(args.target_pdb).read()
    prot = from_pdb_string(pdb_string, chain_id=args.chain)
    pseudo_beta, pseudo_mask = H.ground_truth_pseudo_beta(prot, device)
    L = int(pseudo_beta.shape[-2])
    target_pdb = out / "target.pdb"
    H._write_gt_pdb_for_usalign(pdb_string=pdb_string, chain_id=args.chain, out_path=target_pdb)
    print(f"[target] {args.target_pdb} chain={args.chain} L={L}", flush=True)

    # --- design with the slim model (distogram loss only) ---
    slim = build_model("slim", args.slim_ckpt, args.jax, keep, device)
    fape_cfg = model_config("model_1_ptm", train=True).loss.fape  # unused (coor_scale=0) but required arg
    (_outputs, losses, _frames, final_seq_probs,
     opt_time_s, final_dist_loss, _final_coor) = H.optimize_sequence(
        slim, seq_len=L, pseudo_beta=pseudo_beta, pseudo_mask=pseudo_mask,
        gt_batch=None, fape_cfg=fape_cfg, steps=args.steps, dist_cutoff=0.05, coor_cutoff=0.05,
        device=device, lr=args.lr, dist_scale=1.0, coor_scale=0.0,
        init_seq=args.init_seq, optimizer=args.optimizer, norm_grad=args.norm_grad, dtype=args.dtype)
    designed_idx = final_seq_probs.argmax(dim=-1).long().cpu().numpy()
    designed_seq = "".join(rc.restypes[i] for i in designed_idx)

    # --- fold the HARD designed sequence in slim and full (same discrete seq -> fair comparison) ---
    slim_pred_pdb = out / "slim_pred.pdb"
    slim_pred_pdb.write_text(out_to_pdb(fold_sequence(slim, designed_idx, device), designed_idx))
    del slim
    torch.cuda.empty_cache()

    full = build_model("full", args.slim_ckpt, args.jax, keep, device)
    full_pred_pdb = out / "full_pred.pdb"
    full_pred_pdb.write_text(out_to_pdb(fold_sequence(full, designed_idx, device), designed_idx))
    del full
    torch.cuda.empty_cache()

    # --- TM scores (the interpretable metric) ---
    tm_slim_target = usalign_tm(target_pdb, slim_pred_pdb)
    tm_full_target = usalign_tm(target_pdb, full_pred_pdb)
    tm_full_slim = usalign_tm(slim_pred_pdb, full_pred_pdb)

    H.save_loss_plot(losses, out / "loss.png", title="WS3 slim hallucination (distogram loss)",
                     y_label="distogram loss")

    summary = {
        "target_pdb": args.target_pdb,
        "target_name": Path(args.target_pdb).stem,
        "chain": args.chain,
        "L": L,
        "steps": args.steps,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "norm_grad": bool(args.norm_grad),
        "init_seq": args.init_seq,
        "dtype": args.dtype,
        "opt_time_s": float(opt_time_s),
        "final_dist_loss": None if final_dist_loss is None else float(final_dist_loss),
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "designed_seq": designed_seq,
        "tm_slim_target": tm_slim_target,
        "tm_full_target": tm_full_target,
        "tm_full_slim": tm_full_slim,
        "target_pdb_out": str(target_pdb),
        "slim_pred_pdb": str(slim_pred_pdb),
        "full_pred_pdb": str(full_pred_pdb),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=== WS3 RESULTS ===", flush=True)
    print(f"distogram loss: {summary['loss_first']:.4f} -> {summary['loss_last']:.4f} "
          f"(final_dist_loss={summary['final_dist_loss']})", flush=True)
    print(f"designed_seq: {designed_seq}", flush=True)
    print(f"TM(slim_pred, target)           = {tm_slim_target}", flush=True)
    print(f"TM(full_pred, target)           = {tm_full_target}", flush=True)
    print(f"TM(full_pred, slim_pred) [xmodel]= {tm_full_slim}", flush=True)
    print(f"wrote: {out}/summary.json  +  target.pdb/slim_pred.pdb/full_pred.pdb", flush=True)


if __name__ == "__main__":
    main()
