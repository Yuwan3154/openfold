"""Phase 1a: score the Garcia, Dixit & Rocklin 2025/26 de novo monomer benchmark (Protein Science
33:e70453) with our WS5 checkpoint and compare AUROC against their published ESMFold/AF2/AF3 columns.

Dataset: 614 de novo monomers, 11 studies (2012-2021), binary "Experimental Success" label.
Source: Europe PMC supplementary-files API for PMC12817478 (direct PMC/NCBI downloads are
bot-blocked; Europe PMC mirrors the same CC-BY files and works with plain curl).
"""
import csv
import io
import os
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5, score_sequence, resolve_ws5_ckpt
from eval_stats import auroc

CKPT = os.environ.get("CKPT") or None  # resolved lazily below (checkpoint filename rotates while WS5 trains)
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/monomer_designs_scored.csv")
DEVICE = os.environ.get("DEVICE", "cuda:0")
SUPP_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12817478/supplementaryFiles"
CSV_NAME = "PRO-35-e70453-s002.csv"


def fetch_dataset():
    req = urllib.request.Request(SUPP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(CSV_NAME) as f:
            text = f.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main():
    rows = fetch_dataset()
    print(f"loaded {len(rows)} designs from Garcia/Dixit/Rocklin supplementary CSV", flush=True)

    model = load_ws5(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    scored = []
    t0 = time.time()
    for i, row in enumerate(rows):
        seq = row["sequence"].strip().upper()
        label = 1 if row["Experimental Success"].strip().upper() == "TRUE" else 0
        try:
            out = score_sequence(model, seq, device=DEVICE)
        except Exception as e:
            print(f"[{i}] {row['Name']} len={len(seq)} FAILED: {e}", flush=True)
            continue
        scored.append({
            "name": row["Name"],
            "length": len(seq),
            "experimental_success": label,
            "our_plddt_mean": out["plddt_mean"],
            "our_pae_mean": out["pae_mean"],
            "af2_plddt_3recycle": row["AlphaFold2 pLDDT 3 recycles"],
            "esmfold_plddt": row["ESMFold pLDDT"],
            "af3_plddt": row["AlphaFold3 pLDDT"],
        })
        if (i + 1) % 50 == 0:
            print(f"scored {i + 1}/{len(rows)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    print(f"wrote {len(scored)} scored rows -> {OUT_CSV}", flush=True)

    labels = [r["experimental_success"] for r in scored]
    our_auc = auroc([r["our_plddt_mean"] for r in scored], labels)
    print(f"\nn={len(scored)} success_rate={sum(labels) / len(labels):.3f}")
    print(f"our WS5 pLDDT AUROC:      {our_auc:.3f}")
    print("published baselines (Garcia/Dixit/Rocklin 2025/26, Protein Science 33:e70453):")
    print("  ESMFold pLDDT:  0.72 +/- 0.05")
    print("  AF2 pLDDT:      0.71 +/- 0.17")
    print("  AF3 pLDDT:      0.60 +/- 0.06")
    print("  logistic combo: 0.72 +/- 0.14 (no improvement over best single metric)")


if __name__ == "__main__":
    main()
