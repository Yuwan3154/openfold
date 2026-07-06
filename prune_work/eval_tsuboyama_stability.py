"""Phase 2: correlate our WS5 checkpoint's confidence with real folding-stability ddG from the
Tsuboyama et al. 2023 mega-scale dataset (Nature 620:434-444, Zenodo 10.5281/zenodo.7992926).

Uses Processed_K50_dG_datasets.zip/Single_DMS_list.csv -- the deduplicated per-WT-domain manifest
(983 domains, aa_seq + wt_dg_med), not the raw per-mutant K50_dG_tables (776K rows, DNA sequences,
no clean domain-level summary).

NOTE on natural vs. designed: this manifest has no explicit natural/designed label column. We
classify by name pattern (PDB-ID-format names -> natural; "XX|"/"GG|"/"EA|"-batch-prefixed and
other non-PDB-like names -> designed) -- this is a heuristic, not a ground-truth field, so it's
reported as a secondary breakdown only. The primary correlation is computed on the full set,
which is also how RaSP's r=0.62 zero-shot comparator baseline is reported (on its own subset of
this same lineage of data, not an identical row-for-row match either).
"""
import csv
import io
import os
import re
import sys
import time
import urllib.request
import zipfile

from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5, score_sequence, resolve_ws5_ckpt

CKPT = os.environ.get("CKPT") or None  # resolved lazily below (checkpoint filename rotates while WS5 trains)
OUT_CSV = os.environ.get("OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/tsuboyama_scored.csv")
DEVICE = os.environ.get("DEVICE", "cuda:0")
ZENODO_ZIP_URL = "https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip?download=1"
LOCAL_ZIP = os.environ.get("LOCAL_ZIP", "/home/jupyter-chenxi/prune_work/eval_out/tsuboyama_processed.zip")
MEMBER = "Processed_K50_dG_datasets/Single_DMS_list.csv"

PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}(\.pdb)?$")


def is_natural(name):
    """Heuristic only -- see module docstring."""
    return bool(PDB_ID_RE.match(name.split("|")[0]))


def fetch_dataset():
    if not os.path.exists(LOCAL_ZIP):
        os.makedirs(os.path.dirname(LOCAL_ZIP), exist_ok=True)
        print(f"downloading {ZENODO_ZIP_URL} -> {LOCAL_ZIP} (~1GB)", flush=True)
        urllib.request.urlretrieve(ZENODO_ZIP_URL, LOCAL_ZIP)
    with zipfile.ZipFile(LOCAL_ZIP) as z:
        with z.open(MEMBER) as f:
            text = f.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main():
    rows = fetch_dataset()
    print(f"loaded {len(rows)} WT domains from Single_DMS_list.csv", flush=True)

    model = load_ws5(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    scored = []
    t0 = time.time()
    for i, row in enumerate(rows):
        seq = row["aa_seq"].strip().upper()
        try:
            dg = float(row["wt_dg_med"])
        except (ValueError, KeyError):
            continue
        try:
            out = score_sequence(model, seq, device=DEVICE)
        except Exception as e:
            print(f"[{i}] {row['name']} len={len(seq)} FAILED: {e}", flush=True)
            continue
        scored.append({
            "name": row["name"],
            "length": len(seq),
            "wt_dg_med": dg,
            "our_plddt_mean": out["plddt_mean"],
            "our_pae_mean": out["pae_mean"],
            "natural_heuristic": is_natural(row["name"]),
        })
        if (i + 1) % 100 == 0:
            print(f"scored {i + 1}/{len(rows)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    print(f"wrote {len(scored)} scored rows -> {OUT_CSV}", flush=True)

    def report(subset, label):
        if len(subset) < 3:
            print(f"{label}: n={len(subset)}, too few for correlation")
            return
        plddt = [r["our_plddt_mean"] for r in subset]
        dg = [r["wt_dg_med"] for r in subset]
        pr, _ = pearsonr(plddt, dg)
        sr, _ = spearmanr(plddt, dg)
        print(f"{label}: n={len(subset)}  Pearson r={pr:.3f}  Spearman rho={sr:.3f}")

    print()
    report(scored, "ALL domains")
    report([r for r in scored if r["natural_heuristic"]], "natural (heuristic)")
    report([r for r in scored if not r["natural_heuristic"]], "designed (heuristic)")
    print("\ncomparator: RaSP zero-shot Pearson r=0.62 on this same dataset lineage "
          "(different exact subset, not row-matched -- approximate context only)")


if __name__ == "__main__":
    main()
