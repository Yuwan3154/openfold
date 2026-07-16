#!/bin/bash
# ============================================================================================
# DRAFT / CONFIG-ONLY (2026-07-15) -- DO NOT RUN until the flagged choices are confirmed.
# ============================================================================================
# I1: inference study on the PDA de novo set -- random-seed sweep + recycle-count sweep, via
# train_openfold.py --validate_only (the same harness that produced every val/lddt number in this
# project). Priority per user: recycles > random seeds, but TRY RANDOM SEED FIRST -- so this driver
# runs a SEED sweep at a baseline recycle count first, then (set RECYCLES=20) the recycle test.
#
# Mechanisms (both already exist in the code, nothing new needed):
#   - recycle count : env MAX_RECYCLING_ITERS overrides config.data.common.max_recycling_iters
#                     (train_openfold.py:535-537). Eval uses uniform_recycling=False -> runs EXACTLY
#                     that many recycles. finetuning_ptm's own eval default is 3; predict preset = 20.
#   - random seed   : --seed <N> (seed_everything + batch_seed). Varying it changes MSA
#                     sampling / crop position / any stochastic recycling -> a best-of-N ensemble.
#
# ⚠️ OPEN CHOICES flagged for confirmation before running:
#   (a) CKPT + MODEL_FLAGS: which trained model to evaluate, and the architecture flags MUST match
#       that checkpoint. Default below = v4's best (ESMFold2-trick, pruned single-seq + templates),
#       since recycling is the ESMFold2 mechanism under study. Alternatives: stock AF2 (no prune
#       flags) or WS5-continued. If you change CKPT you MUST change MODEL_FLAGS to match, or the
#       weight load / eval will be wrong.
#   (b) SEEDS: default 8-seed sweep {42,1,2,3,4,5,6,7}. Confirm how many.
#   (c) RECYCLES: default 3 (baseline seed sweep). Re-run with RECYCLES=20 for the recycle test;
#       optionally sweep {3,10,20} to see the diminishing-returns curve.
#   (d) Metric of interest: this logs per-seed val/lddt_ca, val/recall_2A, val/alignment_rmsd. For a
#       best-of-N ensemble you likely want per-TARGET best-across-seeds, which --validate_only does
#       NOT compute (it logs set-mean per run). If best-of-N-per-target is the goal, we need a small
#       post-processor over the per-run predictions -- FLAG: confirm mean-per-seed vs best-of-N.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # single GPU; default GPU2 (0,1 busy w/ WS5-continued)
ulimit -n 65536
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
TRAIN=$L/slim_struct_train.list
VAL=$L/ws5_val_strict_clean.list
PDA_MANIFEST=/home/jupyter-chenxi/prune_work/eval_out/pda_cluster_representatives.json
PDA_CIF_DIR=/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache
PDA_TRAIN_OVERLAP=/home/jupyter-chenxi/prune_work/eval_out/pda_train_overlap_ids.json
# --- open choices (env-overridable) ---
CKPT=${CKPT:-$(ls -t /home/jupyter-chenxi/runs/prune_singleseq_esmfold2_v4_pda_eval/lightning_logs/version_7/checkpoints/best-*.ckpt 2>/dev/null | head -1)}
MODEL_FLAGS=${MODEL_FLAGS:-"--prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer --contractive_recycling --gaussian_pair_init"}
SEEDS=${SEEDS:-"42 1 2 3 4 5 6 7"}
RECYCLES=${RECYCLES:-3}
BASE=${OUT_BASE:-/home/jupyter-chenxi/runs/recycle_seed_sweep}
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "ERROR: CKPT not found: $CKPT"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA manifest not found"; exit 1; }
echo "SWEEP ckpt=$CKPT recycles=$RECYCLES seeds=[$SEEDS]"
for s in $SEEDS; do
  OUT="$BASE/rec${RECYCLES}_seed${s}"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "=== seed=$s recycles=$RECYCLES -> $OUT ==="
  MAX_RECYCLING_ITERS=$RECYCLES python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
    --config_preset finetuning_ptm \
    --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
    $MODEL_FLAGS --validate_without_templates \
    --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
    --pda_train_overlap_ids "$PDA_TRAIN_OVERLAP" \
    --resume_from_ckpt "$CKPT" --resume_model_weights_only true --validate_only \
    --train_chain_list_path "$TRAIN" \
    --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
    --precision bf16 --seed "$s" --log_every_n_steps 20 --distributed_backend nccl \
    2>&1 | tee "$OUT/validate.log"
  echo "--- seed=$s result ---"; grep -E "val/lddt_ca|val/recall_2A|val/alignment_rmsd" "$OUT/validate.log" | tail -5
done
echo "SWEEP DONE. Per-seed metrics are the 'Validate metric' tables in each $BASE/rec${RECYCLES}_seed*/validate.log"
