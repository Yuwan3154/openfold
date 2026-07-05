"""Phase 1b: score the Overath, Rygaard et al. 2025 binder meta-analysis (bioRxiv 2025.08.14.670059,
Zenodo 10.5281/zenodo.15722219) with our WS5 checkpoint and compare AUROC against their existing
AF2-initial-guess "af2_plddt_binder" column (computed WITH target context, so not a strictly fair
apples-to-apples comparison to our target-free monomer prediction -- reported anyway for context,
see the printed caveat).

Dataset: 3,766 experimentally tested binders across 15 targets; "binder" column is the ground-truth
True/False experimental outcome; "binder_chain" (A/B) + "{chain}_seq" gives the designed sequence.
"""
import csv
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5, score_sequence
from eval_stats import auroc

CKPT = os.environ.get(
    "CKPT",
    "/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_3/checkpoints/best-041-010836.ckpt")
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/binder_designs_scored.csv")
DEVICE = os.environ.get("DEVICE", "cuda:0")
DATASET_URL = "https://zenodo.org/records/15722219/files/final_dataset.csv?download=1"
LOCAL_CSV = os.environ.get("LOCAL_CSV", "/home/jupyter-chenxi/prune_work/eval_out/overath_final_dataset.csv")
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))  # our model's practical single-seq length ceiling


def fetch_dataset():
    if not os.path.exists(LOCAL_CSV):
        os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
        print(f"downloading {DATASET_URL} -> {LOCAL_CSV} (~82MB)", flush=True)
        urllib.request.urlretrieve(DATASET_URL, LOCAL_CSV)
    with open(LOCAL_CSV, newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = fetch_dataset()
    print(f"loaded {len(rows)} rows from Overath et al. final_dataset.csv", flush=True)

    model = load_ws5(CKPT, device=DEVICE)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    scored = []
    skipped_label = skipped_len = skipped_err = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        label_str = row.get("binder", "").strip()
        if label_str not in ("True", "False"):
            skipped_label += 1
            continue
        chain = row.get("binder_chain", "A").strip() or "A"
        seq = row.get(f"{chain}_seq", "").strip().upper()
        if not seq or len(seq) > MAX_LEN:
            skipped_len += 1
            continue
        try:
            out = score_sequence(model, seq, device=DEVICE)
        except Exception as e:
            print(f"[{i}] {row.get('binder_id')} len={len(seq)} FAILED: {e}", flush=True)
            skipped_err += 1
            continue
        scored.append({
            "binder_id": row.get("binder_id"),
            "target_id": row.get("target_id"),
            "source": row.get("source"),
            "length": len(seq),
            "binder": 1 if label_str == "True" else 0,
            "our_plddt_mean": out["plddt_mean"],
            "our_pae_mean": out["pae_mean"],
            "af2_plddt_binder": row.get("af2_plddt_binder"),
            "af2_binder_aligned_rmsd": row.get("af2_binder_aligned_rmsd"),
        })
        if (i + 1) % 200 == 0:
            print(f"processed {i + 1}/{len(rows)}, scored {len(scored)} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    print(f"wrote {len(scored)} scored rows -> {OUT_CSV} "
          f"(skipped: {skipped_label} no-label, {skipped_len} too-long/empty, {skipped_err} inference-error)",
          flush=True)

    labels = [r["binder"] for r in scored]
    our_auc = auroc([r["our_plddt_mean"] for r in scored], labels)
    print(f"\nn={len(scored)} success_rate={sum(labels) / len(labels):.3f}")
    print(f"our WS5 monomer-only pLDDT AUROC: {our_auc:.3f}")
    print("CAVEAT: af2_plddt_binder in this dataset is computed WITH target context (AF2 initial-"
          "guess complex prediction), while our model scores the isolated binder chain alone with "
          "no target -- not a strictly fair comparison, reported for context only:")
    try:
        af2_scores = [float(r["af2_plddt_binder"]) for r in scored if r["af2_plddt_binder"]]
        af2_labels = [r["binder"] for r in scored if r["af2_plddt_binder"]]
        print(f"  af2_plddt_binder (with target context) AUROC: {auroc(af2_scores, af2_labels):.3f}")
    except Exception as e:
        print(f"  (could not compute af2_plddt_binder AUROC: {e})")


if __name__ == "__main__":
    main()
