#!/bin/bash
# WS2: confidence-head-only fine-tune of the converged slim model (recalibrate plddt/pTM/pAE heads on the slim reps).
# SAME data regime as slim training (full MSA + templates, finetuning_ptm). Weights-only resume from the slim best ckpt;
# slice to the 24 KEEP blocks first (so the 24-block ckpt loads post-slice), then freeze all but the heads.
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
KEEP="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47"
SLIM_CKPT=${SLIM_CKPT:-/home/jupyter-chenxi/runs/slim_struct_v1/lightning_logs/version_4/checkpoints/best-037-009500.ckpt}
LOSSJSON=${LOSS_JSON:-/home/jupyter-chenxi/prune_work/head_only_loss.json}
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}
VAL=${VAL_LIST:-$L/slim_struct_val.list}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/confhead_ft_v1}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
[ -f "$TRAIN" ]     || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]       || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$SLIM_CKPT" ] || { echo "ERROR: slim ckpt not found: $SLIM_CKPT"; exit 1; }
[ -f "$LOSSJSON" ]  || { echo "ERROR: loss json not found: $LOSSJSON"; exit 1; }
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --evoformer_keep_block_indices "$KEEP" \
  --freeze_all_except_heads \
  --experiment_config_json "$LOSSJSON" \
  --resume_from_ckpt "$SLIM_CKPT" --resume_model_weights_only true \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate 5e-4 --warmup_no_steps 1000 \
  --train_epoch_len 1000 --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/plddt_loss --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
