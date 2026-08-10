import re
from collections import defaultdict

TRAIN_LIST = "/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_train.list"
RESULTS_TSV = "/home/jupyter-chenxi/prune_work/eval_out/a6_search_results.tsv"

# training set as (pdbid, chain) pairs, plus a pdbid -> set(chains) index for the
# bare-pdbid-target case (some foldseek-parsed targets have no chain suffix).
train_pairs = set()
train_chains_by_pdb = defaultdict(set)
with open(TRAIN_LIST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        pdbid, chain = line.split("_", 1)
        train_pairs.add((pdbid, chain))
        train_chains_by_pdb[pdbid].add(chain)

MODEL_RE = re.compile(r"^(?P<pdbid>[0-9a-zA-Z]{4})(_MODEL_\d+)?(_(?P<chain>.+))?$")


def classify_target(target):
    m = MODEL_RE.match(target)
    if not m:
        return None
    pdbid = m.group("pdbid")
    chain = m.group("chain")
    if chain is not None:
        if (pdbid, chain) in train_pairs:
            return (pdbid, chain, "exact")
        return None
    # bare pdbid (no chain suffix) -- confirm the PDB ID itself has >=1 chain in the
    # training list; can't know which specific chain, but the structure is confirmed
    # to be a real training-set entry.
    if pdbid in train_chains_by_pdb:
        chains = sorted(train_chains_by_pdb[pdbid])
        return (pdbid, "|".join(chains), "pdbid-only, chain ambiguous")
    return None


by_query = defaultdict(list)
with open(RESULTS_TSV) as f:
    for line in f:
        query, target, alntm, qtm, ttm, lddt, alnlen, fident, evalue = line.rstrip("\n").split("\t")
        cls = classify_target(target)
        if cls is None:
            continue
        pdbid, chain, note = cls
        by_query[query].append({
            "target_raw": target, "pdbid": pdbid, "chain": chain, "note": note,
            "qtm": float(qtm), "ttm": float(ttm), "alntm": float(alntm),
            "lddt": float(lddt), "alnlen": int(alnlen), "fident": float(fident),
        })

queries = ["6cfa_A", "9g3b_C", "1djf_A", "1pbz_A", "5w9f_A", "6q5q_A", "6os8_A", "7jh6_A"]
print(f"{'query':10s} {'n_train_hits':13s} {'best_target':18s} {'qtm':>6s} {'ttm':>6s} {'fident':>7s} {'alnlen':>7s}")
for q in queries:
    hits = sorted(by_query.get(q, []), key=lambda h: -h["qtm"])
    if not hits:
        print(f"{q:10s} {'0':13s} {'NO TRAINING-SET HIT':18s}")
        continue
    best = hits[0]
    print(f"{q:10s} {len(hits):13d} {best['pdbid']+'_'+best['chain']:18s} "
          f"{best['qtm']:6.3f} {best['ttm']:6.3f} {best['fident']:7.3f} {best['alnlen']:7d}"
          + (f"  [{best['note']}]" if best["note"] != "exact" else ""))

print()
print("=== full detail, top 3 hits per query ===")
for q in queries:
    hits = sorted(by_query.get(q, []), key=lambda h: -h["qtm"])
    print(f"\n--- {q} ({len(hits)} training-set hits total) ---")
    if not hits:
        print("  (no hit against the training set at all, e<10)")
    for h in hits[:3]:
        print(f"  {h['pdbid']}_{h['chain']:8s} qtm={h['qtm']:.3f} ttm={h['ttm']:.3f} "
              f"lddt={h['lddt']:.3f} fident={h['fident']:.3f} alnlen={h['alnlen']} "
              f"raw_target={h['target_raw']}")
