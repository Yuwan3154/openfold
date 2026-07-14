#!/bin/bash
# Control branch, NOT part of the esmfold2-tricks v1-v5 line: continues WS5's OWN training
# (prune_evoformer + enable_single_seq_mode + single_seq_keep_templates + freeze_non_evoformer,
# NO contractive_recycling, NO gaussian_pair_init -- "the earlier simpler recipe") from its TRUE
# last checkpoint (version_4/checkpoints/last.ckpt, PL epoch 64/step 16520 -- one epoch past
# best-063-016336.ckpt, which is what v1-v5 used as their own init source instead), weights-only,
# to see what would have happened if WS5 had simply kept training longer with its original recipe.
# Everything about the recipe itself is unchanged from run_prune_singleseq.sh; the ONLY thing
# updated per user directive (2026-07-12) is the validation set/metrics -- swapped from WS5's
# original natural-protein val list to the same PDA de novo design set + true-single-seq eval
# (--validate_without_templates) + train-overlap split logging that v4/v5 use, so its numbers are
# directly comparable on the same chart. train_epoch_len defaults to 3000 (not WS5's original
# 1000) purely so its epoch axis lines up with v4/v5's for that comparison -- not a recipe change.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold  # MANDATORY, see run_prune_singleseq_esmfold2_v1.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
ulimit -n 65536
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}
VAL=${VAL_LIST:-$L/ws5_val_strict_clean.list}
PDA_MANIFEST=${PDA_VAL_MANIFEST:-/home/jupyter-chenxi/prune_work/eval_out/pda_cluster_representatives.json}
PDA_CIF_DIR=${PDA_CIF_CACHE_DIR:-/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache}
PDA_TRAIN_OVERLAP=${PDA_TRAIN_OVERLAP_IDS:-/home/jupyter-chenxi/prune_work/eval_out/pda_train_overlap_ids.json}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/prune_singleseq_ws5_continued_pda_eval}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-3000}
GRAD_ACCUM=${GRAD_ACCUM:-1}
# WS5's own TRUE LAST checkpoint (not best-063-016336.ckpt, which is what v1-v5 used) -- the
# actual final state of WS5's own training trajectory, PL epoch 64 / step 16520.
WS5_CKPT_DIR=/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints
INIT_CKPT=${INIT_CKPT:-$WS5_CKPT_DIR/last.ckpt}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$INIT_CKPT" ] || { echo "ERROR: no init checkpoint found at $INIT_CKPT"; exit 1; }
# Auto-resume: if THIS run already has its own checkpoint, resume full state from it; else
# weights-only init from WS5's true last checkpoint (first launch).
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_ckpt "$INIT_CKPT" --resume_model_weights_only true)
  echo "INIT (weights-only) from WS5's true last checkpoint: $INIT_CKPT"
fi
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --validate_without_templates \
  --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
  --pda_train_overlap_ids "$PDA_TRAIN_OVERLAP" \
  "${RESUME[@]}" \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --grad_accum_steps "$GRAD_ACCUM" \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
