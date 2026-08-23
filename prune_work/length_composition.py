"""If short chains are the damage vector, the FLAT populations must contain almost no short chains,
and the nonneural subset (which fell 3x harder) must be ENRICHED in them."""
import json
import statistics as st
from collections import Counter

E = "/home/jupyter-chenxi/prune_work/eval_out"
V = "/home/jupyter-chenxi/prune_work/val_expanded/v384"

BINS = [(0, 30), (31, 50), (51, 80), (81, 130), (131, 220), (221, 10**9)]


def seqlen(e):
    for k_ in ("seq", "sequence"):
        if k_ in e and e[k_]:
            return len(e[k_])
    return e.get("length") or e.get("seqlen") or 0


def report(label, entries):
    L = [seqlen(e) for e in entries]
    L = [x for x in L if x]
    if not L:
        print(f"{label:26s} (no length field)")
        return
    c = Counter()
    for x in L:
        for lo, hi in BINS:
            if lo <= x <= hi:
                c[(lo, hi)] += 1
                break
    n = len(L)
    cells = "  ".join(f"{100.0 * c[b] / n:5.1f}%" for b in BINS)
    print(f"{label:26s} n={n:4d} med={st.median(L):5.0f}   {cells}")


hdr = "  ".join(f"{(str(lo) + '-' + (str(hi) if hi < 10**9 else '+')):>6}" for lo, hi in BINS)
print(f"{'population':26s} {'':11s}       {hdr}")

mdl = json.load(open(f"{E}/pda_cluster_representatives_modelable.json"))
report("PDA modelable (306)", mdl)
for name in ("val_300_easy.json", "val_300_hard.json"):
    d = json.load(open(f"{V}/{name}"))
    report(f"{name.replace('.json', '')} (natural)", d if isinstance(d, list) else list(d.values()))

nn = json.load(open(f"{E}/pda_nonneural_strict_ids.json"))
nn_set = {f"{e['pdb'].lower()}_{e['chain_id']}" for e in nn}
print(f"\nnonneural id list: {len(nn_set)} ids, sample {sorted(nn_set)[:3]}")


def idk(e):
    return f"{e['pdb'].lower()}_{e['chain_id']}"


inn = [e for e in mdl if idk(e) in nn_set]
oth = [e for e in mdl if idk(e) not in nn_set]
print(f"matched in the modelable 306: {len(inn)}\n")
report("  PDA nonneural", inn)
report("  PDA neural-gated", oth)
