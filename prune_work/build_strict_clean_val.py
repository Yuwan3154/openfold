"""Build a rigorously deduplicated WS5 val set via MMseqs2 clustering (--min-seq-id 0.3 -c 0.8,
the exact recipe already used on this box for pdb30_200513/pdb70), instead of the k-mer containment
proxy used before. Extracts query sequences for train(88155)+val(200), writes one combined FASTA,
clusters it, then reports which val chains share a cluster with >=1 train chain (leaked under the
strict definition) vs not (strict-clean). If too few strict-clean val chains remain, backfills
additional post-cutoff candidates from the openproteinset_aln pool (chains not in train or val),
filtered by mmCIF release date > 2018-04-30, then clusters THOSE against train too before accepting.
"""
import os
import glob

ALN = "/home/jupyter-chenxi/data/openproteinset_aln"
TRAIN_LIST = "/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_train.list"
VAL_LIST = "/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_val.list"
OUT_DIR = "/home/jupyter-chenxi/prune_work/lists_pdb/mmseqs_strict"
os.makedirs(OUT_DIR, exist_ok=True)


def query_seq(chain):
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


val_chains = [l.strip() for l in open(VAL_LIST)]
train_chains = [l.strip() for l in open(TRAIN_LIST)]

print(f"loading {len(val_chains)} val + {len(train_chains)} train query sequences...", flush=True)
val_seqs = {c: query_seq(c) for c in val_chains}
val_seqs = {c: s for c, s in val_seqs.items() if s}
print(f"val sequences loaded: {len(val_seqs)}/{len(val_chains)}", flush=True)

train_seqs = {}
for i, c in enumerate(train_chains):
    s = query_seq(c)
    if s:
        train_seqs[c] = s
    if (i + 1) % 20000 == 0:
        print(f"  ...{i+1}/{len(train_chains)} train seqs scanned", flush=True)
print(f"train sequences loaded: {len(train_seqs)}/{len(train_chains)}", flush=True)

fasta_path = f"{OUT_DIR}/combined.fasta"
with open(fasta_path, "w") as f:
    for c, s in train_seqs.items():
        f.write(f">train:{c}\n{s}\n")
    for c, s in val_seqs.items():
        f.write(f">val:{c}\n{s}\n")
print(f"wrote {fasta_path} ({len(train_seqs) + len(val_seqs)} sequences)", flush=True)
