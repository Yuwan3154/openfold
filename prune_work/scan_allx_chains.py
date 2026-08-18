"""Find training/validation chains whose seqres is (almost) entirely "X" -- non-canonical residues.

⛔ WHY THEY MUST GO (user, 2026-08-18): "X" in seqres means the residue is non-standard, so
`mmcif_parsing` maps it to aatype 20 (unknown). A chain that is entirely X gives the model a sequence
it cannot interpret at all -- there is no signal to learn from and the FAPE target is being fitted from
an uninformative input. They were found via the T2 qmap (205 of the 82733 generated chains), but the
training list is larger than the generated set (5422 chains are L>512 and were never generated), so
the lists have to be scanned independently rather than reusing that number.

⭐ THE THRESHOLD IS EXPRESSED ON THE **KNOWN** SIDE (`--min-known-fraction`), not the X side, because
that is the quantity the decision is actually about -- "how much real sequence does the model get" --
and it removes the >=/> ambiguity that an X-fraction threshold has at the boundary. User set it to
0.05 on 2026-08-18 ("cut them too"): a chain with under 5% canonical residues is excluded, which is
the 243 fully-X chains PLUS the 26 that are 90-99.9% X (one is 185 of 189 X). 1.0 would exclude
everything; 0.0 excludes only fully-X.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--chain-cache", required=True, help="chain_data_cache_full.json")
ap.add_argument("--lists", nargs="+", required=True, help="chain list files to scan")
ap.add_argument("--mmcif-dir", default=None, help="fallback for chains absent from the cache")
ap.add_argument("--min-known-fraction", type=float, required=True,
                help="exclude a chain when its fraction of CANONICAL (non-X) residues is BELOW this. "
                     "0.05 = keep only chains with at least 5%% real sequence (user, 2026-08-18). "
                     "Use 0.0 to exclude only fully-X chains.")
ap.add_argument("--out-json", required=True)
a = ap.parse_args()

cache = {k: v.get("seq") for k, v in json.load(open(a.chain_cache)).items()}
print(f"cache: {len(cache)} chains", flush=True)

_parsed = {"id": None, "obj": None}


def seq_from_mmcif(chain):
    """Only used for chains the cache misses -- the cache covered 100% of the generated set, but the
    training list is a superset of it."""
    from openfold.data import mmcif_parsing
    file_id, chain_id = chain.rsplit("_", 1)
    if _parsed["id"] != file_id:
        p = Path(a.mmcif_dir) / f"{file_id}.cif"
        _parsed.update(
            id=file_id,
            obj=(mmcif_parsing.parse(file_id=file_id, mmcif_string=p.read_text()).mmcif_object
                 if p.is_file() else None),
        )
    mo = _parsed["obj"]
    return None if mo is None else mo.chain_to_seqres.get(chain_id)


out = {}
for lst in a.lists:
    chains = [l.strip() for l in open(lst) if l.strip()]
    hist, excl, no_seq, fracs = Counter(), [], [], []
    for c in chains:
        s = cache.get(c)
        if s is None and a.mmcif_dir:
            s = seq_from_mmcif(c)
        if not s:
            no_seq.append(c)
            continue
        f = s.count("X") / len(s)
        known = 1.0 - f
        fracs.append(f)
        # ⭐ bucket on the KNOWN-residue side too, since that is what the model actually has to work
        # with: a 189-residue chain with 4 known residues is not meaningfully different from 0
        key = ("100% X" if f >= 1.0 else ">=99%" if f >= 0.99 else ">=90%" if f >= 0.90
               else ">=50%" if f >= 0.50 else ">=10%" if f >= 0.10 else ">0%" if f > 0 else "0%")
        hist[key] += 1
        # ⛔ strict `<`: a chain sitting exactly AT the threshold is kept, so the flag reads as
        # "at least this much known sequence" with no boundary surprise
        if known < a.min_known_fraction or f >= 1.0:
            excl.append(c)
    name = Path(lst).name
    print(f"\n=== {name}: {len(chains)} chains ===")
    for k in ("0%", ">0%", ">=10%", ">=50%", ">=90%", ">=99%", "100% X"):
        if hist[k]:
            print(f"  X fraction {k:>7}: {hist[k]:6d}  ({100*hist[k]/max(1,len(fracs)):5.2f}%)")
    n_known_lt5 = sum(1 for f in fracs if f > 0.0 and (1 - f) < 0.05)
    print(f"  chains with <5% KNOWN residues (incl. fully-X): {n_known_lt5}")
    print(f"  no sequence available          : {len(no_seq)}"
          + (f"  e.g. {no_seq[:4]}" if no_seq else ""))
    print(f"  EXCLUDED at --min-known-fraction {a.min_known_fraction}: {len(excl)}"
          + (f"  e.g. {excl[:5]}" if excl else ""))
    keep = [c for c in chains if c not in set(excl)]
    out[name] = {"total": len(chains), "excluded": excl, "n_keep": len(keep),
                 "no_seq": no_seq, "hist": dict(hist)}
    if excl:
        kept_path = str(lst) + ".noallx"
        with open(kept_path, "w") as fh:
            fh.write("\n".join(keep) + "\n")
        print(f"  wrote filtered list -> {kept_path}  ({len(keep)} chains)")
        out[name]["filtered_list"] = kept_path

json.dump(out, open(a.out_json, "w"), indent=1)
print(f"\nwrote {a.out_json}")
