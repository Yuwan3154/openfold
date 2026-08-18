#!/bin/bash
# ⛔⛔ BEHAVIOUR CHANGED 2026-08-18 -- THIS SCRIPT NO LONGER REPRODUCES THE RUN IT IS NAMED FOR.
#   `--enable_single_seq_mode` now also forces a QUERY-ONLY MSA (AF2Rank parity): the a3m files are
#   never opened, so the extra-MSA track attends to fully-masked padding instead of a real homolog, and
#   msa_feat's cluster_profile channels no longer carry homology. The runs that produced the recorded
#   T1/T2 curves did NOT have this. To reproduce those exactly, add --no-force-query-only-msa.
# ============================================================================================
# T1 (2026-08-10 user directive): the SAME no-MSA training recipe as WS5-continued, but applied
# to the FULL STOCK AF2 model instead of the pruned WS5 architecture.
#
# Purpose (user's staging logic): isolate whether the no-MSA training-data recipe ALONE improves
# the full stock model, before any other variable is added. This is stage 1 of a 3-stage ladder:
#   1. T1 (this run)  : stock AF2 + no-MSA recipe, NO tricks, real templates only
#   2. T2             : + partial-diffusion synthetic templates (still no tricks)
#   3. T3             : + both ESMFold2 tricks (the intended maximal config)
# ⛔ Do NOT add the tricks here -- that would collapse stages 1 and 3 and confound the result.
#
# Recipe = run_prune_singleseq_ws5_continued.sh with exactly TWO deliberate differences:
#   (a) NO --prune_evoformer  -> full/unpruned Evoformer. The pruning IS the WS5 architecture, so
#       dropping it is what makes this "the stock model"; everything else is held constant.
#   (b) init from stock AF2 model_1_ptm jax params instead of WS5's checkpoint.
# Everything else (single-seq/no-MSA, keep-templates-in-train, freeze-non-evoformer, PDA
# validate-without-templates, lr 1e-4, warmup 3000, epoch_len 3000, bf16, seed 42) is copied
# verbatim from the WS5-continued launcher so the comparison stays single-variable.
#
# ⚠️ NOT the same as run_stock_af2_tricks.sh (the older DRAFT), which is a DIFFERENT experiment:
# that one has tricks ON and drops --enable_single_seq_mode (i.e. full MSA). Do not conflate.
#
# ⚠️ Memory: the unpruned Evoformer keeps column + triangle ATTENTION, which the pruned WS5 drops.
# At the finetuning_ptm crop of 384 this is substantially heavier than any prior run here. If this
# OOMs, the grounded knobs are GRAD_ACCUM (env) and CUDA_VISIBLE_DEVICES; do NOT silently change
# the crop or lr, since those would break comparability with WS5-continued.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# 2-GPU (0,1) DDP hangs on this box (PHB PCIe P2P deadlock) -> disable direct P2P at <=2 GPUs.
# A 4-GPU ring avoids the broken direct 0<->1 link and is fine with P2P on. Verified 2026-07-24.
# See reference_a6000_2gpu_nccl_p2p_hang.
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[ "$NGPU" -le 2 ] && export NCCL_P2P_DISABLE=1
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
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/stock_af2_nomsa_t1_pda_eval}
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
  echo "INIT (warm-start) from stock AF2 jax params: $JAX"
fi
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
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
