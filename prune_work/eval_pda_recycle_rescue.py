"""Immediate, narrower follow-up to the PDA N=100 result (WS5 41/100, stock AF2 66/100): does
MORE recycling rescue the currently-FAILING cases? This is a simpler diagnostic question than
full compute-matched scaling (see SLIM_DOWNSTREAM.md's "compute-matched recycle scaling" standing
direction) -- just checking whether the model needs more iterations to converge, using the SAME
already-loaded model instance re-scored at different recycle counts (recycle count is purely
input-shape-driven at inference time, confirmed via openfold/model/model.py's
`num_iters = batch["aatype"].shape[-1]` -- no model reload needed between recycle levels).

Reuses eval_pda_self_consistency.py's fetch/scoring helpers directly (no re-derivation).
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pda_self_consistency import get_design_chain, fetch_mmcif, kabsch_rmsd, CA_IDX
from single_seq_infer import load_ws5, load_af2_stock, score_sequence, resolve_ws5_ckpt
from openfold.data import mmcif_parsing
from openfold.data.templates import _get_atom_positions

DEVICE = os.environ.get("DEVICE", "cuda:0")
RMSD_THRESHOLD = 2.0
WS5_CKPT = os.environ.get("CKPT") or None
RESULT_CSV = os.environ.get(
    "RESULT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/pda_self_consistency.csv")
OUT_CSV = os.environ.get(
    "OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/pda_recycle_rescue.csv")
RECYCLE_LEVELS = [int(x) for x in os.environ.get("RECYCLE_LEVELS", "3,6,9,12").split(",")]
N_SAMPLE = int(os.environ.get("N_SAMPLE", "0")) or None  # 0/unset = no cap, test the full failed union


def get_native(pdbid, chain_id):
    mmcif_path = fetch_mmcif(pdbid)
    with open(mmcif_path) as f:
        cif_string = f.read()
    parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=cif_string)
    obj = parsed.mmcif_object
    if obj is None or chain_id not in obj.chain_to_seqres:
        return None, None
    real_pos, real_mask = _get_atom_positions(
        obj, chain_id, max_ca_ca_distance=150.0, _zero_center_positions=False)
    return real_pos[:, CA_IDX, :], real_mask[:, CA_IDX].astype(bool)


def main():
    with open(RESULT_CSV) as f:
        rows = list(csv.DictReader(f))
    failed_ws5 = {(r["pdb"], r["chain_id"]) for r in rows if r["success_ws5"] == "False"}
    failed_stock = {(r["pdb"], r["chain_id"]) for r in rows if r["success_stock"] == "False"}
    failed_union = sorted(failed_ws5 | failed_stock)
    if N_SAMPLE:
        failed_union = failed_union[:N_SAMPLE]
    print(f"WS5 fails: {len(failed_ws5)}  stock fails: {len(failed_stock)}  "
          f"union to re-test: {len(failed_union)}", flush=True)

    model_ws5 = load_ws5(WS5_CKPT or resolve_ws5_ckpt(), device=DEVICE)
    model_stock = load_af2_stock(device=DEVICE)
    print("both models loaded", flush=True)

    out_rows = []
    t0 = time.time()
    for i, (pdbid, chain_id) in enumerate(failed_union):
        try:
            result = get_design_chain(pdbid)
            if result is None:
                continue
            _, seq = result
            real_ca, valid = get_native(pdbid, chain_id)
            if real_ca is None:
                continue
            n = min(len(seq), len(real_ca))
            valid_n = valid[:n]
            if valid_n.sum() < 5:
                continue
            seq = seq[:n]
        except Exception as e:
            print(f"{pdbid}_{chain_id}: setup failed: {e}", flush=True)
            continue

        row = {"pdb": pdbid, "chain_id": chain_id, "length": n,
               "was_ws5_fail": (pdbid, chain_id) in failed_ws5,
               "was_stock_fail": (pdbid, chain_id) in failed_stock}
        for r in RECYCLE_LEVELS:
            try:
                if row["was_ws5_fail"]:
                    rmsd = kabsch_rmsd(
                        score_sequence(model_ws5, seq, device=DEVICE, recycle=r)["ca_coords"][valid_n],
                        real_ca[:n][valid_n])
                    row[f"ws5_rmsd_r{r}"] = rmsd
                    row[f"ws5_rescued_r{r}"] = rmsd < RMSD_THRESHOLD
                if row["was_stock_fail"]:
                    rmsd = kabsch_rmsd(
                        score_sequence(model_stock, seq, device=DEVICE, recycle=r)["ca_coords"][valid_n],
                        real_ca[:n][valid_n])
                    row[f"stock_rmsd_r{r}"] = rmsd
                    row[f"stock_rescued_r{r}"] = rmsd < RMSD_THRESHOLD
            except Exception as e:
                print(f"{pdbid}_{chain_id} @ recycle={r}: failed: {e}", flush=True)
        out_rows.append(row)
        print(f"{pdbid}_{chain_id}: {row}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  -- {i+1}/{len(failed_union)} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    if not out_rows:
        print("no rows scored -- aborting", flush=True)
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_fields = sorted({k for row in out_rows for k in row.keys()})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {OUT_CSV}", flush=True)

    for model_name, failed_set in [("ws5", failed_ws5), ("stock", failed_stock)]:
        applicable = [r for r in out_rows if r[f"was_{model_name}_fail"]]
        print(f"\n{model_name.upper()}: {len(applicable)} originally-failed cases re-tested")
        for r in RECYCLE_LEVELS:
            key = f"{model_name}_rescued_r{r}"
            n_rescued = sum(1 for row in applicable if row.get(key) is True)
            print(f"  recycle={r}: {n_rescued}/{len(applicable)} rescued (RMSD<2A)")


if __name__ == "__main__":
    main()
