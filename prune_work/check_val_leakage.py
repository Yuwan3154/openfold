"""Check whether WS5 val-set sequences are near-duplicates of train-set sequences
(k-mer containment as a fast proxy for high sequence identity), since a suspiciously
large val lDDT jump could be explained by train/val leakage rather than genuine
single-seq folding capability."""
import os

ALN = "/home/jupyter-chenxi/data/openproteinset_aln"
TRAIN_LIST = "/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_train.list"
VAL_LIST = "/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_val.list"
K = 8


def query_seq(chain):
    # the FIRST record in any a3m is always the query sequence itself
    for fname in ["uniref90_hits.a3m", "bfd_uniclust_hits.a3m", "mgnify_hits.a3m"]:
        path = f"{ALN}/{chain}/{fname}"
        if os.path.exists(path):
            with open(path) as f:
                header = f.readline()
                if header.startswith(">"):
                    seq = f.readline().strip()
                    if seq:
                        return seq.upper()
    return None


def kmers(seq, k=K):
    return set(seq[i:i + k] for i in range(len(seq) - k + 1))


val_chains = [l.strip() for l in open(VAL_LIST)]
train_chains = [l.strip() for l in open(TRAIN_LIST)]

print(f"loading {len(val_chains)} val + {len(train_chains)} train query sequences...", flush=True)
val_seqs = {c: query_seq(c) for c in val_chains}
val_seqs = {c: s for c, s in val_seqs.items() if s}
print(f"val sequences loaded: {len(val_seqs)}/{len(val_chains)}", flush=True)

train_seqs = {}
missing = 0
for i, c in enumerate(train_chains):
    s = query_seq(c)
    if s:
        train_seqs[c] = s
    else:
        missing += 1
    if (i + 1) % 20000 == 0:
        print(f"  ...{i+1}/{len(train_chains)} train seqs scanned", flush=True)
print(f"train sequences loaded: {len(train_seqs)}/{len(train_chains)} (missing {missing})", flush=True)

# build inverted k-mer index: kmer -> set of train chain ids
print("building k-mer index...", flush=True)
index = {}
train_kmers = {}
for c, s in train_seqs.items():
    km = kmers(s)
    train_kmers[c] = km
    for kk in km:
        index.setdefault(kk, set()).add(c)
print(f"index built: {len(index)} unique {K}-mers", flush=True)

print("=== per-val-chain best train match (k-mer containment) ===", flush=True)
results = []
for c, s in val_seqs.items():
    vk = kmers(s)
    if not vk:
        continue
    candidates = {}
    for kk in vk:
        for tc in index.get(kk, ()):
            candidates[tc] = candidates.get(tc, 0) + 1
    if not candidates:
        results.append((c, len(s), None, 0.0))
        continue
    best_c, best_hits = max(candidates.items(), key=lambda kv: kv[1])
    containment = best_hits / len(vk)
    results.append((c, len(s), best_c, containment))

results.sort(key=lambda r: -r[3])
print("top 20 highest val<->train containment (potential leakage):")
for c, L, best_c, cont in results[:20]:
    print(f"  {c} (L={L}) <-> best train match {best_c}: kmer-containment={cont:.3f}")

n_high = sum(1 for r in results if r[3] > 0.5)
n_med = sum(1 for r in results if 0.2 < r[3] <= 0.5)
print(f"\nSUMMARY: {len(results)} val chains checked; containment>0.5: {n_high}; 0.2-0.5: {n_med}; <=0.2: {len(results)-n_high-n_med}")

OUTDIR = "/home/jupyter-chenxi/prune_work/lists_pdb"
with open(f"{OUTDIR}/leakage_scores.csv", "w") as f:
    f.write("chain,length,best_train_match,containment\n")
    for c, L, best_c, cont in sorted(results, key=lambda r: r[0]):
        f.write(f"{c},{L},{best_c},{cont:.4f}\n")

with open(f"{OUTDIR}/ws5_val_leaked.list", "w") as f:
    f.write("\n".join(c for c, L, bc, cont in results if cont > 0.5) + "\n")
with open(f"{OUTDIR}/ws5_val_clean.list", "w") as f:
    f.write("\n".join(c for c, L, bc, cont in results if cont <= 0.2) + "\n")
with open(f"{OUTDIR}/ws5_val_ambiguous.list", "w") as f:
    f.write("\n".join(c for c, L, bc, cont in results if 0.2 < cont <= 0.5) + "\n")
print(f"wrote leakage_scores.csv + ws5_val_{{leaked,clean,ambiguous}}.list to {OUTDIR}")
