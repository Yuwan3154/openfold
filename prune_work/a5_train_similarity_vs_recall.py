import csv
from collections import defaultdict

BASELINE_CSV = "/home/jupyter-chenxi/prune_work/eval_out/pda_baseline_full/pda_baseline_full.csv"
HITS_TSV = "/home/jupyter-chenxi/prune_work/eval_out/pda_vs_train_a5.tsv"

best_identity = defaultdict(float)
with open(HITS_TSV) as f:
    for line in f:
        q, t, pident, alnlen, evalue, bits = line.rstrip("\n").split("\t")
        best_identity[q] = max(best_identity[q], float(pident))

rows = []
with open(BASELINE_CSV) as f:
    for r in csv.DictReader(f):
        qid = f"{r['pdb']}_{r['chain_id']}"
        rows.append({
            "id": qid,
            "success_ws5": r["success_2A_ws5"] == "True",
            "success_stock": r["success_2A_stock"] == "True",
            "identity": best_identity.get(qid, 0.0),
            "has_hit": qid in best_identity,
        })

print(f"total entries: {len(rows)}")

ws5_better = [r for r in rows if r["success_ws5"] and not r["success_stock"]]
stock_better = [r for r in rows if r["success_stock"] and not r["success_ws5"]]
both = [r for r in rows if r["success_ws5"] and r["success_stock"]]
neither = [r for r in rows if not r["success_ws5"] and not r["success_stock"]]

print(f"WS5-better (ws5 succeeds, stock fails): {len(ws5_better)}")
print(f"stock-better (stock succeeds, ws5 fails): {len(stock_better)}")
print(f"both succeed: {len(both)}")
print(f"neither succeeds: {len(neither)}")
print()

def summarize(label, rs):
    if not rs:
        print(f"{label}: n=0")
        return
    ids = [r["identity"] for r in rs]
    n_hit = sum(r["has_hit"] for r in rs)
    print(f"{label}: n={len(rs)}, mean_identity={sum(ids)/len(ids):.1f}%, "
          f"median={sorted(ids)[len(ids)//2]:.1f}%, n_with_any_hit={n_hit}/{len(rs)}")

summarize("WS5-better", ws5_better)
summarize("stock-better", stock_better)
summarize("both succeed", both)
summarize("neither", neither)
summarize("ALL", rows)

print()
print("=== WS5-better entries, full detail ===")
for r in sorted(ws5_better, key=lambda x: -x["identity"]):
    print(f"  {r['id']:12s} identity={r['identity']:5.1f}%  has_hit={r['has_hit']}")
