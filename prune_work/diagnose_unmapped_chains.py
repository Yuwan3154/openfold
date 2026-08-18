"""Root-cause the chains that `build_query_index_map.py` could not map.

The builder drops a chain when `residue_index - 1` is not sequence-valid AND the npz sequence is not
a SUBSEQUENCE of the query. That failure is reported as one bucket, which is not enough to fix it.
This classifies each failure into a cause, using only the index, the npz tree and the chain cache.

⛔ Matching in the builder is ASYMMETRIC: `_match(sc, qc)` treats an npz 'X' as a wildcard but NOT a
query 'X'. mmcif seqres renders unknown/modified residues as 'X' while protpardelle's structure parse
resolves many of them to a standard parent (MSE->M etc.), so a query 'X' facing a concrete npz residue
is a predicted failure mode. That hypothesis is TESTED here, not assumed.
"""

import argparse
import collections
import json
import zlib
from pathlib import Path

import numpy as np

import openfold
from openfold.np import residue_constants as rc

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--qmap", required=True)
ap.add_argument("--templates-root", required=True)
ap.add_argument("--chain-cache", required=True)
ap.add_argument("--out-json", required=True)
a = ap.parse_args()

print("openfold from:", openfold.__file__, flush=True)


def embeds(short, long, sym):
    """Leftmost subsequence embedding; `sym` also treats a query 'X' as a wildcard."""
    j = 0
    for ch in long:
        if j < len(short) and (short[j] == ch or short[j] == "X" or (sym and ch == "X")):
            j += 1
    return j == len(short)


def first_divergence(short, long):
    """Greedy walk; report where the embedding gets stuck."""
    j = 0
    for i, ch in enumerate(long):
        if j < len(short) and (short[j] == ch or short[j] == "X"):
            j += 1
    return j  # how many npz residues could be placed before running out of query


zi = np.load(a.index, allow_pickle=False)
zq = np.load(a.qmap, allow_pickle=False)
idx_chains = [str(c) for c in zi["chains"]]
mapped = {str(c) for c in zq["chains"]}
missing = [c for c in idx_chains if c not in mapped]
print(f"index chains {len(idx_chains)}, mapped {len(mapped)}, MISSING {len(missing)}", flush=True)

cache = json.load(open(a.chain_cache))
by_entry = collections.defaultdict(list)
for k in cache:
    by_entry[k.rsplit("_", 1)[0]].append(k)

root = Path(a.templates_root)
buckets = collections.Counter()
records = []

for n, chain in enumerate(missing):
    npz = root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}" / f"{chain}.npz"
    rec = {"chain": chain}
    if not npz.is_file():
        buckets["no_npz"] += 1
        rec["cause"] = "no_npz"
        records.append(rec)
        continue
    entry = chain.rsplit("_", 1)[0]
    cv = cache.get(chain)
    if cv is None or not cv.get("seq"):
        buckets["no_query_in_cache"] += 1
        rec["cause"] = "no_query_in_cache"
        records.append(rec)
        continue
    query = cv["seq"]
    d = np.load(npz, allow_pickle=False)
    aat = d["aatype"].astype(int)
    npz_seq = "".join(rc.restypes[x] if x < len(rc.restypes) else "X" for x in aat)
    rec.update(npz_len=len(npz_seq), query_len=len(query),
               query_X=query.count("X"), npz_X=npz_seq.count("X"),
               placed=first_divergence(npz_seq, query))

    if len(npz_seq) > len(query):
        cause = "npz_longer_than_query"
    elif embeds(npz_seq, query, sym=False):
        cause = "embeds_after_all"          # would mean the builder's own bug, not a data problem
    elif embeds(npz_seq, query, sym=True):
        cause = "query_X_blocks_match"      # the predicted modified-residue case
    else:
        alt = [o for o in by_entry.get(entry, [])
               if o != chain and cache[o].get("seq") and embeds(npz_seq, cache[o]["seq"], sym=True)]
        if alt:
            cause = "matches_other_chain_of_entry"
            rec["alt_chains"] = alt[:4]
        else:
            cause = "real_mismatch"
            rec["npz_head"] = npz_seq[:60]
            rec["query_head"] = query[:60]
    buckets[cause] += 1
    rec["cause"] = cause
    records.append(rec)
    if (n + 1) % 100 == 0:
        print(f"  {n+1}/{len(missing)}", flush=True)

print("\n=== CAUSE BREAKDOWN ===")
for k, v in buckets.most_common():
    print(f"  {k:32s} {v:5d}  ({100*v/len(missing):.1f}%)")

for cause in ["query_X_blocks_match", "matches_other_chain_of_entry", "real_mismatch",
              "npz_longer_than_query", "embeds_after_all"]:
    ex = [r for r in records if r.get("cause") == cause][:4]
    if ex:
        print(f"\n--- examples: {cause}")
        for r in ex:
            print("   ", {k: v for k, v in r.items() if k != "cause"})

json.dump({"missing": len(missing), "buckets": dict(buckets), "records": records},
          open(a.out_json, "w"), indent=1)
print(f"\nwrote {a.out_json}")
