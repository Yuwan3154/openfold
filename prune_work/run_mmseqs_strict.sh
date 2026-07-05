#!/bin/bash
# Cluster train+val combined sequences at the SAME recipe already used on this box for
# pdb30_200513/pdb70 (--min-seq-id 0.3 -c 0.8 -s 8 --cluster-mode 1), to build a rigorously
# deduplicated WS5 val set (vs the earlier k-mer containment proxy).
# --threads 24 (not 64) to leave headroom for the concurrently-running WS5 training job.
set -e
MMSEQS=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/mmseqs
DIR=/home/jupyter-chenxi/prune_work/lists_pdb/mmseqs_strict
cd "$DIR"
rm -rf combined_db* combined_clu* tmp
$MMSEQS createdb combined.fasta combined_db
$MMSEQS cluster combined_db combined_clu tmp --min-seq-id 0.3 -c 0.8 -s 8 --cluster-mode 1 --threads 24
$MMSEQS createtsv combined_db combined_db combined_clu combined_clu.tsv
echo "DONE"
wc -l combined_clu.tsv
