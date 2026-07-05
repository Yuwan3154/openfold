"""Parse the mmseqs2 (--min-seq-id 0.3 -c 0.8) cluster TSV to determine which val chains share a
cluster with >=1 train chain (leaked under the strict definition) vs not (strict-clean)."""
import os
from collections import defaultdict

TSV = "/home/jupyter-chenxi/prune_work/lists_pdb/mmseqs_strict/combined_clu.tsv"
OUTDIR = "/home/jupyter-chenxi/prune_work/lists_pdb"

cluster_members = defaultdict(list)  # rep -> [member, ...]
with open(TSV) as f:
    for line in f:
        rep, member = line.rstrip("\n").split("\t")
        cluster_members[rep].append(member)

val_leaked = []
val_clean = []
for rep, members in cluster_members.items():
    has_train = any(m.startswith("train:") for m in members)
    has_val = any(m.startswith("val:") for m in members)
    if not has_val:
        continue
    val_members = [m[len("val:"):] for m in members if m.startswith("val:")]
    if has_train:
        val_leaked.extend(val_members)
    else:
        val_clean.extend(val_members)

print(f"total val chains: {len(val_leaked) + len(val_clean)}")
print(f"STRICT LEAKED (share a 30%-id/80%-cov cluster with >=1 train chain): {len(val_leaked)}")
print(f"STRICT CLEAN: {len(val_clean)}")

with open(f"{OUTDIR}/ws5_val_strict_leaked.list", "w") as f:
    f.write("\n".join(sorted(val_leaked)) + "\n")
with open(f"{OUTDIR}/ws5_val_strict_clean.list", "w") as f:
    f.write("\n".join(sorted(val_clean)) + "\n")
print("wrote ws5_val_strict_{leaked,clean}.list")

# cross-check against the earlier k-mer-based buckets
kmer_leaked = set(l.strip() for l in open(f"{OUTDIR}/ws5_val_leaked.list"))
kmer_clean = set(l.strip() for l in open(f"{OUTDIR}/ws5_val_clean.list"))
strict_clean_set = set(val_clean)
print(f"\nCross-check vs earlier k-mer buckets:")
print(f"  of the {len(kmer_clean)} k-mer 'clean' chains, {len(kmer_clean & strict_clean_set)} also pass strict clustering")
print(f"  strict clustering caught {len(kmer_clean - strict_clean_set)} chains the k-mer method missed (divergent homologs)")
print(f"  of the {len(kmer_leaked)} k-mer 'leaked' chains, {len(kmer_leaked & strict_clean_set)} are (surprisingly) strict-clean")
