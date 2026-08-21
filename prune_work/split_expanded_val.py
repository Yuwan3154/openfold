"""Split the sequence-filtered expanded validation candidates into EASY / HARD by structural
similarity to the training set, then draw a length-stratified sample of each.

Supersedes the inline stage-5 block of build_expanded_val.sh, which keyed the foldseek hits on the
PDB ENTRY (`5bka`) instead of the CHAIN (`5bka_A`) -- so a chain inherited the best TM of any chain
in its entry (5.87 chains/entry here), which systematically over-populates EASY.

Chain-id conventions, all verified to agree on AUTHOR ids (2026-08-21): foldseek's query names equal
the `_atom_site.auth_asym_id` set on 300/300 examined entries where auth != label; openfold's
`chain_to_seqres` is `author_chain_to_sequence` (mmcif_parsing.py:295); OpenProteinSet directory
names match auth.

foldseek names single-chain files `<pdb>` and multi-chain files `<pdb>_<auth>`, so both forms are
resolved here. TM is read from `qtmscore` -- normalised by the QUERY, i.e. the validation chain,
which is the native side of this comparison.
"""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict


def read_lookup(path):
    """foldseek db name -> set of names, and pdb -> set of chain-suffix forms present."""
    names = set()
    by_pdb = defaultdict(set)
    for ln in open(path):
        name = ln.rstrip("\n").split("\t")[1]
        names.add(name)
        by_pdb[name.partition("_")[0]].add(name.partition("_")[2])
    return names, by_pdb


def read_seqs(fasta, keep):
    seqs = {}
    name = None
    for ln in open(fasta):
        if ln.startswith(">"):
            h = ln[1:].strip()
            name = h.split("cand:")[1] if h.startswith("cand:") else None
        elif name is not None:
            if name in keep:
                seqs[name] = ln.strip()
            name = None
    return seqs


def quantile_edges(values, k):
    v = sorted(values)
    return [v[int(round(i * len(v) / k))] for i in range(1, k)]


def stratum_of(length, edges):
    for i, e in enumerate(edges):
        if length < e:
            return i
    return len(edges)


def stratified_sample(pool, edges, n_total, k, rng):
    """pool: {chain: length}. Returns (chosen, deficits) -- deficits records every stratum that
    could not supply its quota, so a shortfall is never silent."""
    by_s = defaultdict(list)
    for c, L in pool.items():
        by_s[stratum_of(L, edges)].append(c)
    for s in by_s:
        by_s[s].sort()
    quota = n_total // k
    chosen, deficits = [], []
    surplus = {}
    for s in range(k):
        avail = by_s.get(s, [])
        take = min(quota, len(avail))
        picked = rng.sample(avail, take)
        chosen += picked
        surplus[s] = [c for c in avail if c not in set(picked)]
        if take < quota:
            deficits.append({"stratum": s, "quota": quota, "available": len(avail), "short": quota - take})
    short = n_total - len(chosen)
    # redistribute a shortfall to the nearest strata that still have members
    for s in sorted(surplus, key=lambda s: -len(surplus[s])):
        if short <= 0:
            break
        take = min(short, len(surplus[s]))
        chosen += rng.sample(surplus[s], take)
        short -= take
    return sorted(set(chosen)), deficits, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/jupyter-chenxi/prune_work/val_expanded")
    ap.add_argument("--tm-split", type=float, required=True,
                    help="EASY if best qtmscore to a training structure > this, else HARD")
    ap.add_argument("--min-len", type=int, required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--n-per-set", type=int, required=True)
    ap.add_argument("--n-strata", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-prefix", default="v2")
    ap.add_argument("--exclude-manifest", action="append", default=[],
                    help="JSON list of {pdb, chain_id} whose chains are removed from the candidate "
                         "pool BEFORE the draw. Needed because the PDA benchmark entries are also "
                         "PDB depositions: 5 of them landed in a 300+300 draw, which both "
                         "double-counts them in a combined validation mean and mislabels a de novo "
                         "design as a natural time-split chain. Excluding up front keeps the draw at "
                         "exactly n-per-set instead of patching afterwards.")
    args = ap.parse_args()

    W = args.work
    O = os.path.join(W, args.out_prefix)
    os.makedirs(O, exist_ok=True)

    chains = [l.strip() for l in open(f"{W}/seqclean.list") if l.strip()]
    excluded_ids, excluded_hits = set(), []

    def _norm(chain_id_str):
        # ⛔ Only the 4-character PDB code is case-insensitive; the chain id is CASE-SENSITIVE
        # (auth ids include both 'A' and 'a' in the same entry -- e.g. 8v2d_y in this very pool).
        # Lowercasing the whole "pdb_chain" string made the first version of this filter match
        # NOTHING while reporting success: "excluded 0" against 5 known duplicates.
        pdb, _, ch = chain_id_str.partition("_")
        return f"{pdb.lower()}_{ch}"

    for mp in args.exclude_manifest:
        for e in json.load(open(mp)):
            excluded_ids.add(_norm(f"{e['pdb']}_{e['chain_id']}"))
    if excluded_ids:
        excluded_hits = [c for c in chains if _norm(c) in excluded_ids]
        chains = [c for c in chains if _norm(c) not in excluded_ids]
        print(f"excluded {len(excluded_hits)} candidate chains present in "
              f"{len(args.exclude_manifest)} exclusion manifest(s): {sorted(excluded_hits)}")
    names, by_pdb = read_lookup(f"{W}/valdb.lookup")
    seqs = read_seqs(f"{W}/combined.fasta", set(chains))
    lens = {c: len(s_) for c, s_ in seqs.items()}

    # chain -> foldseek query name, resolving both naming forms; every unresolved chain is recorded
    key_of, skipped = {}, []
    for c in chains:
        pdb, _, ch = c.partition("_")
        if c in names:
            key_of[c] = c
        elif pdb in names and by_pdb.get(pdb) == {""}:
            key_of[c] = pdb          # foldseek collapses a single-chain file to the bare pdb id
        elif pdb in by_pdb:
            skipped.append({"chain": c, "reason": "chain absent from valdb (entry present)",
                            "valdb_chains": sorted(by_pdb[pdb])[:12]})
        else:
            skipped.append({"chain": c, "reason": "entry absent from valdb (cif unstaged or rejected)"})

    best = defaultdict(float)
    seen_q = set()
    nrows = 0
    for ln in open(f"{W}/hits.tsv"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        nrows += 1
        q = f[0]
        seen_q.add(q)
        tm = float(f[3])
        if tm > best[q]:
            best[q] = tm

    easy, hard, nohit = [], [], []
    for c, k in key_of.items():
        if k not in seen_q:
            nohit.append(c)
        elif best[k] > args.tm_split:
            easy.append(c)
        else:
            hard.append(c)
    hard_all = sorted(hard + nohit)

    for nm, lst in [("easy", sorted(easy)), ("hard", hard_all), ("nohit", sorted(nohit))]:
        with open(f"{O}/{nm}.list", "w") as fh:
            fh.write("\n".join(lst) + "\n")

    elig = {}
    dropped_len = defaultdict(list)
    for nm, lst in [("easy", sorted(easy)), ("hard", hard_all)]:
        e = {}
        for c in lst:
            L = lens.get(c)
            if L is None:
                dropped_len[nm].append((c, None))
            elif args.min_len <= L <= args.max_len:
                e[c] = L
            else:
                dropped_len[nm].append((c, L))
        elig[nm] = e

    edges = quantile_edges(list(elig["easy"].values()) + list(elig["hard"].values()), args.n_strata)
    rng = random.Random(args.seed)
    out = {}
    report = {"tm_split": args.tm_split, "min_len": args.min_len, "max_len": args.max_len,
              "n_per_set": args.n_per_set, "n_strata": args.n_strata, "seed": args.seed,
              "strata_edges": edges, "hits_rows": nrows,
              "counts": {"seqclean": len(chains), "resolved": len(key_of), "skipped": len(skipped),
                         "easy": len(easy), "hard": len(hard), "nohit": len(nohit)},
              "eligible": {k: len(v) for k, v in elig.items()},
              "dropped_by_length": {k: len(v) for k, v in dropped_len.items()},
              "excluded_manifests": args.exclude_manifest,
              "excluded_chains": sorted(excluded_hits)}

    for nm in ["easy", "hard"]:
        picked, deficits, short = stratified_sample(elig[nm], edges, args.n_per_set, args.n_strata, rng)
        out[nm] = picked
        L = sorted(elig[nm][c] for c in picked)
        report[nm] = {"n": len(picked), "deficits": deficits, "unfilled": short,
                      "len_min": L[0], "len_median": statistics.median(L), "len_mean": round(statistics.mean(L), 1),
                      "len_p90": L[int(0.9 * len(L))], "len_max": L[-1]}
        manifest = [{"pdb": c.partition("_")[0], "chain_id": c.partition("_")[2],
                     "seq": seqs[c], "val_source": nm, "length": elig[nm][c],
                     "best_tm_to_train": round(best.get(key_of[c], 0.0), 5)} for c in picked]
        with open(f"{O}/val_{args.n_per_set}_{nm}.json", "w") as fh:
            json.dump(manifest, fh, indent=1)

    with open(f"{O}/skipped.json", "w") as fh:
        json.dump(skipped, fh, indent=1)
    with open(f"{O}/report.json", "w") as fh:
        json.dump(report, fh, indent=1)

    print(json.dumps(report, indent=1))
    print(f"\nwrote {O}/val_{args.n_per_set}_easy.json  {O}/val_{args.n_per_set}_hard.json")
    print(f"skipped chains recorded: {len(skipped)} -> {O}/skipped.json")


if __name__ == "__main__":
    main()
