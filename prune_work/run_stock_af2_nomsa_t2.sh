#!/bin/bash
# ⛔⛔ BEHAVIOUR CHANGED 2026-08-18 -- THIS SCRIPT NO LONGER REPRODUCES THE RUN IT IS NAMED FOR.
#   `--enable_single_seq_mode` now also forces a QUERY-ONLY MSA (AF2Rank parity): the a3m files are
#   never opened, so the extra-MSA track attends to fully-masked padding instead of a real homolog, and
#   msa_feat's cluster_profile channels no longer carry homology. The runs that produced the recorded
#   T1/T2 curves did NOT have this. To reproduce those exactly, add --no-force-query-only-msa.
# ============================================================================================
# T2 (2026-08-17 user directive): T1's recipe PLUS Protpardelle-1c partial-diffusion synthetic
# templates mixed into the TRAIN split. Stage 2 of the 3-stage ladder:
#   1. T1 : stock AF2 + no-MSA recipe, NO tricks, real templates only          [DONE, ep30]
#   2. T2 : + partial-diffusion synthetic templates (still no tricks)          [THIS RUN]
#   3. T3 : + both ESMFold2 tricks (the intended maximal config)
# ⛔ Do NOT add the tricks here -- that would collapse stages 2 and 3.
#
# ⭐ INIT FROM THE JAX PARAMS, NOT T1's CHECKPOINT (user, "as a clean comparison"). T2 therefore
# runs single-variable against T1's own curve from step 0, rather than measuring a marginal effect
# on a model T1 had already adapted. A fresh $OUT is what makes this happen: the auto-resume block
# below only finds a last.ckpt inside $OUT, so a new dir falls through to --resume_from_jax_params.
# ⛔ Consequence: do NOT point OUT_DIR at T1's run dir, or this silently becomes a T1 continuation.
#
# Everything else is copied VERBATIM from run_stock_af2_nomsa_t1.sh so the only differences are the
# four --t2_* flags and the output dir.
#
# ⚠️ What "--t2_n_synthetic 4" does (audited in code 2026-08-17, do not re-guess):
#   It adds 4 synthetic hits to the chain's template POOL alongside its ~4 natural ones. It does
#   NOT deliver 4 synthetic templates per step. Per step the model still sees AT MOST 4 templates
#   total, because config.data.train.max_templates = 4. random_crop_to_size(subsample_templates=
#   True) fully permutes the pool and takes a random window, so each delivered slot is a uniform
#   draw from the 8 -> expected mix 50/50. Monte-Carlo over the real torch calls: 2.885 delivered
#   per step, synthetic share 0.4998.
#   ⚠️ Side effect, accepted by the user: a bigger pool also shifts the delivered COUNT, because
#   templates_crop_start ~ Uniform{0..pool} INCLUSIVE. Pool 4 -> mean 2.00, P(0 templates) 20%;
#   pool 8 -> mean 2.89, P(0) 11.1%. So T1-vs-T2 differs in template CONTENT and in template
#   COUNT. To hold the count fixed instead, the pool would have to be capped at 4 by REPLACING
#   natural hits rather than appending -- deliberately not done.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
# ⛔ Single entry, exactly as T1 sets it. The repo root reaches sys.path via the `cd` above +
# `python train_openfold.py` (sys.path[0] = PWD), NOT via this variable. Both are required:
# without the repo root the editable install resolves `openfold` to ~/openfold/openfold, a
# DIFFERENT checkout; without <repo>/openfold the top-level `block_replacement_scripts` import in
# data_modules.py fails, ENHANCED_UTILS_AVAILABLE goes False, and --train_chain_list_path is
# SILENTLY IGNORED (133019 chains instead of 88155). Never drop the `cd`.
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
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}
VAL=${VAL_LIST:-$L/ws5_val_strict_clean.list}
PDA_MANIFEST=${PDA_VAL_MANIFEST:-/home/jupyter-chenxi/prune_work/eval_out/pda_cluster_representatives.json}
PDA_CIF_DIR=${PDA_CIF_CACHE_DIR:-/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache}
PDA_TRAIN_OVERLAP=${PDA_TRAIN_OVERLAP_IDS:-/home/jupyter-chenxi/prune_work/eval_out/pda_train_overlap_ids.json}
JAX=${STOCK_AF2_JAX:-/home/jupyter-chenxi/params/params_model_1_ptm.npz}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/stock_af2_nomsa_t2_pda_eval}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-3000}
GRAD_ACCUM=${GRAD_ACCUM:-1}
# T2-specific: the band-pruned synthetic template tree + its index (transferred + verified
# 2026-08-17; 52 GB, 82730 npz, band-locked to 0.3-0.9).
T2_INDEX=${T2_TEMPLATE_INDEX:-/home/jupyter-chenxi/pp1c_work/index_band.npz}
T2_ROOT=${T2_TEMPLATES_ROOT:-/home/jupyter-chenxi/pp1c_work/templates_band}
T2_MIN_TM=${T2_MIN_TM:-0.3}
T2_MAX_TM=${T2_MAX_TM:-0.9}
T2_N=${T2_N_SYNTHETIC:-4}
# ⛔ REQUIRED. Without it the npz rows are placed by residue_index - 1, which desynchronises at
# the first unresolved residue and is what killed launch #2 (1eis_A, 70/85 positions wrong).
# data_modules.py asserts on its absence rather than silently falling back.
T2_QMAP=${T2_QMAP:-/home/jupyter-chenxi/pp1c_work/qmap_all.npz}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$JAX" ] || { echo "ERROR: stock AF2 jax params not found: $JAX"; exit 1; }
[ -f "$T2_INDEX" ] || { echo "ERROR: T2 template index not found: $T2_INDEX"; exit 1; }
[ -d "$T2_ROOT" ]  || { echo "ERROR: T2 templates root not found: $T2_ROOT"; exit 1; }
[ -f "$T2_QMAP" ] || { echo "ERROR: T2 query-index map not found: $T2_QMAP"; exit 1; }
# Auto-resume this run's own checkpoint (full state); else warm-start from stock AF2 jax params.
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_jax_params "$JAX")
  echo "INIT (warm-start) from stock AF2 jax params: $JAX"
fi
echo "T2 synthetic templates: index=$T2_INDEX root=$T2_ROOT band=$T2_MIN_TM-$T2_MAX_TM n=$T2_N qmap=$T2_QMAP"
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
  --t2_template_index "$T2_INDEX" --t2_templates_root "$T2_ROOT" \
  --t2_qmap "$T2_QMAP" \
  --t2_min_tm "$T2_MIN_TM" --t2_max_tm "$T2_MAX_TM" --t2_n_synthetic "$T2_N" \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --grad_accum_steps "$GRAD_ACCUM" \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
