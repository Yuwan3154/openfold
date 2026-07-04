#!/bin/bash
# Eval ONE WS5 (pruned, single-seq+templates) checkpoint on a given val chain list via
# train_openfold --validate_only. Used to check whether the observed val lDDT surge is
# concentrated in train/val-leaked chains (near-duplicate sequences under different PDB IDs)
# vs genuinely clean, non-redundant validation chains.
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
CKPT=${CKPT}
VAL=${VAL_LIST}
OUT=${OUT_DIR:-/tmp/ws5_validate}
[ -f "$CKPT" ] || { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
[ -f "$VAL" ]  || { echo "ERROR: val list not found: $VAL"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "VALIDATE ckpt=$CKPT on $(wc -l < $VAL) val chains -> $OUT"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --resume_from_ckpt "$CKPT" --resume_model_weights_only true --validate_only \
  --train_chain_list_path "$L/slim_struct_train.list" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --seed 42 --log_every_n_steps 1 --distributed_backend nccl
