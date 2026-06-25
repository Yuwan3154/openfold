#!/bin/bash
# Eval ONE checkpoint on a FIXED val set via train_openfold --validate_only (efficient attention, fits
# full MSA). --validate_only refreshes EMA FROM the loaded weights, so it evaluates exactly those weights
# (raw, no EMA lag). Logs val/lddt_ca, val/fape, val/plddt_loss, val/tm, ... to OUT_DIR for the driver.
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ulimit -n 65536
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
KEEP="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47"   # 24-block slim
CKPT=${CKPT}
VAL=${VAL_LIST:-$L/fixed_eval.list}
OUT=${OUT_DIR:-/tmp/fe2_val}
[ -f "$CKPT" ] || { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "VALIDATE ckpt=$CKPT on $(wc -l < $VAL) val chains -> $OUT"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --evoformer_keep_block_indices "$KEEP" --freeze_non_evoformer \
  --resume_from_ckpt "$CKPT" --resume_model_weights_only true --validate_only \
  --train_chain_list_path "$L/slim_struct_train.list" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --seed 42 --log_every_n_steps 1 --distributed_backend nccl
