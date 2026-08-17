#!/bin/bash
# ============================================================================================
# Precision-vs-crop decomposition, axis A: same as run_stock_af2_nomsa_t1_validate_only.sh
# (raw jax warm-start, zero training, T1's own uncropped harness) but --precision 32 instead of
# bf16 (PL alias "32" -> "32-true", true fp32, no autocast -- verified against this env's
# lightning_fabric _PRECISION_INPUT_STR_ALIAS_CONVERSION, not guessed). Since args.precision="32"
# is not in train_openfold.py's is_low_precision list, this also gives low_prec=False
# (eps=1e-8, per-module inf) -- matching pda_baseline_full.py's precision config, while crop stays
# uncropped (unlike pda_baseline_full.py's crop=256). Isolates: does precision ALONE (holding crop
# at "uncropped") move the number toward 0.728?
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[ "$NGPU" -le 2 ] && export NCCL_P2P_DISABLE=1
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
JAX=/home/jupyter-chenxi/params/params_model_1_ptm.npz
OUT=/home/jupyter-chenxi/runs/stock_af2_nomsa_t1_validate_only_fp32
[ -f "$JAX" ] || { echo "ERROR: stock AF2 jax params not found: $JAX"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "VALIDATE_ONLY: raw jax warm-start (zero training), fp32 (32-true) + uncropped -> $OUT"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --validate_without_templates \
  --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
  --pda_train_overlap_ids "$PDA_TRAIN_OVERLAP" \
  --resume_from_jax_params "$JAX" --resume_model_weights_only true --validate_only \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision 32 --seed 42 --log_every_n_steps 20 --distributed_backend nccl
