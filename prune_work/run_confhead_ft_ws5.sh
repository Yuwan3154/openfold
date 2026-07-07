#!/bin/bash
# WS7 Task 1: confidence-head-only fine-tune of WS5's LATEST checkpoint (pruned/single-seq/templated).
# Adapted from run_confhead_ft.sh (WS2, 24-block slim model) -- same freeze_all_except_heads +
# head_only_loss.json mechanism, ported to WS5's actual architecture (--prune_evoformer, not
# --evoformer_keep_block_indices) and actual input regime (--enable_single_seq_mode
# --single_seq_keep_templates, not full MSA). SEPARATE run, clean step count, own lightning_logs
# version -- NOT a continuation of WS5's own structure-training run.
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
ulimit -n 65536
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
# Resolve WS5's current best-*.ckpt by mtime (rotates -- same convention as single_seq_infer.py's
# resolve_ws5_ckpt(), reimplemented here in bash since this is a standalone launcher).
WS5_CKPT_DIR=/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints
WS5_CKPT=${WS5_CKPT:-$(ls -t "$WS5_CKPT_DIR"/best-*.ckpt 2>/dev/null | head -1)}
LOSSJSON=${LOSS_JSON:-/home/jupyter-chenxi/openfold/prune_work/head_only_loss.json}
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}
VAL=${VAL_LIST:-$L/ws5_val_strict_clean.list}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/confhead_ft_ws5_v1}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-1000}
[ -f "$TRAIN" ]     || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]       || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -n "$WS5_CKPT" ] || { echo "ERROR: no WS5 best-*.ckpt found in $WS5_CKPT_DIR"; exit 1; }
[ -f "$LOSSJSON" ]  || { echo "ERROR: loss json not found: $LOSSJSON"; exit 1; }
echo "Fine-tuning confidence heads from WS5 ckpt: $WS5_CKPT"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates \
  --freeze_all_except_heads \
  --experiment_config_json "$LOSSJSON" \
  --resume_from_ckpt "$WS5_CKPT" --resume_model_weights_only true \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate 5e-4 --warmup_no_steps 1000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/plddt_loss --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
