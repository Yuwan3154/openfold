#!/bin/bash
# I1 (single-seed, no aggregation needed): v4's best checkpoint (best-023-008484.ckpt, tricks
# enabled -- contractive_recycling + gaussian_pair_init) through validate_only at a configurable
# recycle count (RECYCLES env var -> MAX_RECYCLING_ITERS), on the SAME PDA harness v4's own
# training validation used -- isolates recycle-count as the only variable. Mirrors
# run_ws5_true_last_validate_only.sh's pattern exactly (weights-only load, --validate_only, one
# pass; the harness's own val/* metrics are the whole answer since this is a single seed, not a
# multi-seed sweep needing a mean-vs-best-of-N post-processor).
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?must set explicitly, e.g. 0 or 1 -- GPU2/3 belong to another project, never default to all 4}
export MAX_RECYCLING_ITERS=${RECYCLES:?must set RECYCLES, e.g. 3 or 20}
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
OUT=/home/jupyter-chenxi/runs/v4_recycle${RECYCLES}_validate_only
CKPT=/home/jupyter-chenxi/runs/prune_singleseq_esmfold2_v4_pda_eval/lightning_logs/version_7/checkpoints/best-023-008484.ckpt
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "VALIDATE_ONLY ckpt=$CKPT (v4 best, ep24) recycles=$MAX_RECYCLING_ITERS on full PDA set -> $OUT"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --contractive_recycling --gaussian_pair_init --validate_without_templates \
  --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
  --pda_train_overlap_ids "$PDA_TRAIN_OVERLAP" \
  --resume_from_ckpt "$CKPT" --resume_model_weights_only true --validate_only \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --seed 42 --log_every_n_steps 20 --distributed_backend nccl
