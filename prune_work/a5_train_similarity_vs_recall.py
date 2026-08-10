import csv
from collections import defaultdict

BASELINE_CSV = "/home/jupyter-chenxi/prune_work/eval_out/pda_baseline_full/pda_baseline_full.csv"
HITS_TSV = "/home/jupyter-chenxi/prune_work/eval_out/pda_vs_train_a5.tsv"
QUERY_FASTA = "/home/jupyter-chenxi/prune_work/eval_out/pda_cluster_rep_seq.fasta"

qlens = {}
qid = None
seq = []
with open(QUERY_FASTA) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if qid:
                qlens[qid] = len("".join(seq))
            qid = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if qid:
        qlens[qid] = len("".join(seq))

# pick the hit maximizing aligned-identical-residue count (alnlen * pident/100), not raw peak
# identity -- a short high-identity fragment (e.g. a common motif) otherwise dominates and
# misrepresents whole-sequence similarity to the training set.
best = {}
with open(HITS_TSV) as f:
    for line in f:
        q, t, pident, alnlen, evalue, bits = line.rstrip("\n").split("\t")
        pident, alnlen = float(pident), int(alnlen)
        score = pident * alnlen / 100.0
        cur = best.get(q)
        if cur is None or score > cur[0]:
            best[q] = (score, pident, alnlen, t)

rows = []
with open(BASELINE_CSV) as f:
    for r in csv.DictReader(f):
        qid = f"{r['pdb']}_{r['chain_id']}"
        b = best.get(qid)
        qlen = qlens.get(qid)
        rows.append({
            "id": qid,
            "success_ws5": r["success_2A_ws5"] == "True",
            "success_stock": r["success_2A_stock"] == "True",
            "identity": b[1] if b else 0.0,
            "alnlen": b[2] if b else 0,
            "coverage": (b[2] / qlen * 100) if (b and qlen) else 0.0,
            "target": b[3] if b else None,
            "has_hit": qid in best,
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
    ids = sorted(r["identity"] for r in rs)
    n_hit = sum(r["has_hit"] for r in rs)
    n_hi = sum(r["identity"] >= 30 for r in rs)
    print(f"{label}: n={len(rs)}, mean_identity={sum(ids)/len(ids):.1f}%, "
          f"median={ids[len(ids)//2]:.1f}%, n_with_any_hit={n_hit}/{len(rs)}, "
          f"n_with_identity>=30%={n_hi}/{len(rs)}")

summarize("WS5-better", ws5_better)
summarize("stock-better", stock_better)
summarize("both succeed", both)
summarize("neither", neither)
summarize("ALL", rows)

print()
print("=== WS5-better entries, full detail (coverage-aware best hit) ===")
for r in sorted(ws5_better, key=lambda x: -x["identity"]):
    print(f"  {r['id']:12s} identity={r['identity']:5.1f}%  alnlen={r['alnlen']:4d}  "
          f"coverage={r['coverage']:5.1f}%  target={r['target']}")
