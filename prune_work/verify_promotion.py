"""Audit a live T4 promoted-template pool on disk.

Checks the things that stay SILENT in a running job:
  A. index records vs distinct npz files (the K-samples-overwrite-one-file class of bug)
  B. under --t4_promote_all with K=4, every (chain, epoch, step, rank) should carry 4 samples
  C. promote-all must include gate-FAILING samples (tm_pred < tm_template); if every record passes
     the gate, promote-all is not actually in effect and we are silently back to gated promotion
  D. orphans in both directions: index rows pointing at missing files, files in no index
  E. duplicate (rank, chain, epoch, step, sample) keys -- an override
  F. what retention actually keeps at the run's real max_per_chain
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

POOL = Path(sys.argv[1] if len(sys.argv) > 1
            else "/home/jupyter-chenxi/runs/runC_replica_exchange/t4_pool")
MAX_PER_CHAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 64
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4

if not POOL.exists():
    print(f"pool dir does not exist yet: {POOL}")
    sys.exit(0)

recs = []
for idx in sorted(POOL.glob("rank*/index.jsonl")):
    rank = int(idx.parent.name[4:])
    for line in idx.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["_rank"] = rank
        r["_path"] = idx.parent / r["npz"]
        recs.append(r)

files = sorted(POOL.rglob("*.npz"))
print(f"pool: {POOL}")
print(f"index records : {len(recs)}")
print(f"npz files     : {len(files)}")
if not recs:
    sys.exit(0)

# A ------------------------------------------------------------------ records vs files
paths = [r["_path"] for r in recs]
print(f"\n[A] distinct record paths: {len(set(paths))}  (records {len(recs)}, files {len(files)})")
print(f"    {'OK' if len(set(paths)) == len(recs) == len(files) else '!! MISMATCH'}"
      "  -- records, distinct paths and files must all agree")

# E ------------------------------------------------------------------ duplicate keys
keys = Counter((r["_rank"], r["chain"], r["epoch"], r["step"], r.get("sample", 0)) for r in recs)
dupes = {k: c for k, c in keys.items() if c > 1}
print(f"\n[E] duplicate (rank,chain,epoch,step,sample) keys: {len(dupes)} "
      f"{'OK' if not dupes else '!! OVERRIDE'}")
for k, c in list(dupes.items())[:5]:
    print(f"      {k} x{c}")

# B ------------------------------------------------------------------ samples per step
per_group = defaultdict(set)
for r in recs:
    per_group[(r["_rank"], r["chain"], r["epoch"], r["step"])].add(r.get("sample", 0))
sizes = Counter(len(v) for v in per_group.values())
print(f"\n[B] samples per (rank,chain,epoch,step) group -- expect {K} under promote-all")
for n in sorted(sizes):
    print(f"      {n} sample(s): {sizes[n]:7d} groups")
full = sizes.get(K, 0)
print(f"    groups with all {K}: {full}/{len(per_group)} = {100.0 * full / len(per_group):.1f}%")
print(f"    sample-index histogram: {dict(sorted(Counter(r.get('sample', 0) for r in recs).items()))}")

# C ------------------------------------------------------------------ gate-failers present?
better = sum(1 for r in recs if r["tm_pred"] > r["tm_template"])
worse = len(recs) - better
print(f"\n[C] tm_pred > tm_template : {better} ({100.0 * better / len(recs):.1f}%)")
print(f"    tm_pred <= tm_template: {worse} ({100.0 * worse / len(recs):.1f}%)")
print("    " + ("OK -- gate-failing samples ARE present, so promote-all is really in effect"
                if worse > 0 else
                "!! every record beats its template: promote-all may NOT be in effect"))
tp = sorted(r["tm_pred"] for r in recs)
tt = sorted(r["tm_template"] for r in recs)
med = lambda a: a[len(a) // 2]
print(f"    tm_pred   min/med/max {tp[0]:.3f} / {med(tp):.3f} / {tp[-1]:.3f}")
print(f"    tm_template min/med/max {tt[0]:.3f} / {med(tt):.3f} / {tt[-1]:.3f}")

# D ------------------------------------------------------------------ orphans
missing = [p for p in set(paths) if not p.is_file()]
unindexed = sorted(set(files) - set(paths))
print(f"\n[D] index rows with no file : {len(missing)} {'OK' if not missing else '!!'}")
print(f"    files in no index       : {len(unindexed)} {'OK' if not unindexed else '!!'}")
for p in (missing[:3] + unindexed[:3]):
    print(f"      {p}")

# F ------------------------------------------------------------------ retention
by_chain = defaultdict(list)
for r in recs:
    by_chain[r["chain"]].append(r)
capped = [c for c, v in by_chain.items() if len(v) > MAX_PER_CHAIN]
print(f"\n[F] chains in pool: {len(by_chain)};  epochs present: "
      f"{sorted({r['epoch'] for r in recs})};  steps: "
      f"{min(r['step'] for r in recs)}..{max(r['step'] for r in recs)}")
print(f"    chains over max_per_chain={MAX_PER_CHAIN}: {len(capped)} "
      f"({'no eviction happening yet' if not capped else 'eviction ACTIVE'})")
counts = sorted(len(v) for v in by_chain.values())
print(f"    records/chain min/med/max: {counts[0]} / {counts[len(counts) // 2]} / {counts[-1]}")
if capped:
    c = max(by_chain, key=lambda x: len(by_chain[x]))
    v = sorted(by_chain[c], key=lambda r: (-r["epoch"], -r["step"], -r["_rank"],
                                           r.get("sample", 0), r["npz"]))
    kept = v[:MAX_PER_CHAIN]
    print(f"    e.g. {c}: {len(v)} records -> keeps {len(kept)}; "
          f"kept steps {min(x['step'] for x in kept)}..{max(x['step'] for x in kept)}, "
          f"evicted steps {min(x['step'] for x in v[MAX_PER_CHAIN:])}.."
          f"{max(x['step'] for x in v[MAX_PER_CHAIN:])}")
    assert min(x["step"] for x in kept) >= max(x["step"] for x in v[MAX_PER_CHAIN:]) or \
        len({x["epoch"] for x in v}) > 1, "kept block is not the newest"
    print("    retention keeps the NEWEST block: OK")
