#!/bin/bash
# ============================================================================================
# DRAFT / CONFIG-ONLY (2026-07-15) -- DO NOT RUN until the flagged design choices are confirmed.
# ============================================================================================
# T1: train STOCK AF2 (full standard protocol) WITH the two ESMFold2 tricks, to test whether the
# tricks improve the stock model (vs the pruned/single-seq WS5 line). "Stock" = full Evoformer, real
# MSA + templates, standard finetuning_ptm config -- i.e. the v4/v5 launcher with the pruning /
# single-seq / freeze modifications REMOVED, and init from stock AF2 jax params (not WS5's ckpt).
#
# vs run_prune_singleseq_esmfold2_v1.sh, this DROPS:
#   --prune_evoformer            (keep the full Evoformer -- column + triangle attention intact)
#   --enable_single_seq_mode     (use real MSA: finetuning_ptm's 512 clusters / 5120 extra, not 1)
#   --single_seq_keep_templates  (templates are on by default in the full model anyway)
#   --freeze_non_evoformer       (stock = ALL parameters trainable)
#   export SINGLE_SEQ_MAX_CROP   (no single-seq crop clamp; use finetuning_ptm's crop_size=384)
# and KEEPS the two tricks:
#   --contractive_recycling --gaussian_pair_init
# Init source is stock AF2 model_1_ptm jax params (the same warm-start WS5's own original
# run_prune_singleseq.sh used), via --resume_from_jax_params.
#
# ⚠️ OPEN DESIGN CHOICES flagged for user confirmation before running (all defaults below are
#    grounded in existing project launchers/config, none invented -- but these three are genuine
#    interpretation calls, not settled):
#   (a) TRAINING SET: defaults to this project's slim_struct_train.list (what every run here uses).
#       "all standard training configurations" MIGHT instead mean a fuller/standard AF2 training
#       set -- that is NOT set up here and would need to be built. Confirm slim vs full.
#   (b) EVAL: defaults to the PDA de novo harness with --validate_without_templates (de novo targets
#       have no templates/MSA anyway; keeps numbers directly comparable to the existing stock-AF2
#       PDA baseline 0.728 lddt / 41.7% recall and to v4/v5/WS5 on the same 425-entry set). The
#       alternative is a standard-split, templates-ON eval. Confirm PDA-single-seq vs standard-split.
#   (c) LR/warmup/crop: lr=1e-4, warmup=3000 (the value every run in this project uses) + crop 384
#       (finetuning_ptm default). AF2's own published finetuning schedule differs; confirm if you
#       want the exact AF2 finetuning schedule instead.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[ "$NGPU" -le 2 ] && export NCCL_P2P_DISABLE=1   # see reference_a6000_2gpu_nccl_p2p_hang
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
JAX=${STOCK_AF2_JAX:-/home/jupyter-chenxi/params/params_model_1_ptm.npz}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/stock_af2_tricks_pda_eval}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-3000}
GRAD_ACCUM=${GRAD_ACCUM:-1}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$JAX" ] || { echo "ERROR: stock AF2 jax params not found: $JAX"; exit 1; }
# Auto-resume this run's own checkpoint (full state); else warm-start from stock AF2 jax params.
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_jax_params "$JAX")
  echo "WARM-START from stock AF2 jax params: $JAX"
fi
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --contractive_recycling --gaussian_pair_init --validate_without_templates \
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
