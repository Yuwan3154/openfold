#!/bin/bash
# Proper control: evaluate WS5's TRUE LAST checkpoint (version_4/checkpoints/last.ckpt, PL epoch
# 64/step 16520 -- the exact weights-only init source WS5-continued started from) through the
# CURRENT PDA validation harness (train_openfold.py --validate_only, same pda_val_manifest /
# pda_cif_cache_dir / pda_train_overlap_ids / validate_without_templates / prune_evoformer /
# enable_single_seq_mode / single_seq_keep_templates flags as run_prune_singleseq_ws5_continued.sh),
# on the FULL 425-entry PDA set, zero additional training. Purpose: the existing "WS5 baseline"
# chart reference (0.6256 lddt / 29.2% recall / 8.04 RMSD) was computed by a SEPARATE standalone
# script (pda_baseline_full.py) that instantiates PDASingleSeqDataset directly and never goes
# through OpenFoldDataModule/the Trainer -- a different code path from what v4/v5/WS5-continued's
# own validation numbers come from, and it evaluated WS5's epoch-63 "best" checkpoint, not the
# epoch-64 "true last" checkpoint WS5-continued was actually initialized from. This run isolates
# both variables at once by using the SAME harness + SAME checkpoint WS5-continued started from,
# to tell whether WS5-continued's epoch-1 lddt=0.604 (below the 0.626 baseline) reflects a real
# regression from its first epoch of training, or a baseline-methodology/checkpoint mismatch.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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
OUT=/home/jupyter-chenxi/runs/ws5_true_last_validate_only
CKPT=/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints/last.ckpt
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "VALIDATE_ONLY ckpt=$CKPT (WS5 true last, epoch 64) on full PDA set -> $OUT"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --validate_without_templates \
  --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
  --pda_train_overlap_ids "$PDA_TRAIN_OVERLAP" \
  --resume_from_ckpt "$CKPT" --resume_model_weights_only true --validate_only \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --seed 42 --log_every_n_steps 20 --distributed_backend nccl
