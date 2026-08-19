#!/bin/bash
# ============================================================================================
# RUN B -- the full stack (user directive 2026-08-18). Warm-started from the STOCK AF2 JAX PARAMS,
# NOT from T1's or T2's checkpoint. Four things on top of the new default recipe:
#
#   1. COUNT-MATCHED synthetic templates   --t2_replace_prob 0.5 --t2_topup_to 20
#   2. ESMFold2 tricks                     --contractive_recycling --gaussian_pair_init
#   3. T4 self-distillation                --t4_self_distill --t4_n_promoted 32 ...
#   4. Explorative modeling (best-of-K)    --explore_k 5 --explore_select plddt
#
# ⭐ 2 and 4 are coupled by design: --gaussian_pair_init draws a fresh z_0 inside iteration() on
#   EVERY forward, which is the only reason the K samples differ. Without it, best-of-K would be K
#   identical forwards and pure wasted compute. Do not enable --explore_k without it.
# ⭐ 3 and 4 also compose: best-of-K hands T4 the best of 5 predictions as promotion candidates
#   rather than a single draw.
#
# ⛔⛔ THINGS THAT WILL BE MISREAD IF NOT WRITTEN DOWN:
#   * `train/loss` is NOT comparable to T1/T2. Best-of-K reports the loss of the SELECTED sample, so
#     it is systematically lower than a K=1 run by construction -- that is the min operator, not
#     progress. Compare val/lddt_ca, never the training curve.
#   * T4 CONTRIBUTES NOTHING BEFORE EPOCH 5. The pool starts empty and --t4_promote_after_epoch 5
#     gates writing; reading only matters once it has content. Epochs 0-5 are a tricks+templates run.
#   * The query-only MSA default means this is NOT comparable to the recorded T1/T2 curves either.
#     A matched baseline (Run A) has to be run before any of this is interpretable.
#   * Four unmeasured changes are stacked at once, and the ONE component with data (T2's synthetic
#     templates) is currently NEGATIVE: T2 trailed T1 at all 9 epochs, mean -0.0048 lDDT. A win here
#     will not say which part won.
#
# ⏱ COST: the K scoring forwards run under no_grad (VRAM = one forward), but time is ~2.3-2.7x per
#   step. At T2's measured 0.06 it/s that is ~0.025 it/s => ~8-9.5 h/epoch vs 3.5 h. So T4 starts
#   promoting ~40-48 h in, and T1's epoch-7 comparison point is ~70 h away. Plan accordingly.
#
# ⛔ Do NOT point OUT_DIR at another run's dir: the auto-resume block finds any last.ckpt inside $OUT
#   and would silently continue that run instead of warm-starting from jax. (T2 launch #3 died of it.)
# ============================================================================================
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
# ⛔ *.noallx: the 243 fully-X training chains are excluded (user, 2026-08-18)
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list.noallx}
VAL=${VAL_LIST:-$L/ws5_val_strict_clean.list.noallx}
PDA_MANIFEST=${PDA_VAL_MANIFEST:-/home/jupyter-chenxi/prune_work/eval_out/pda_cluster_representatives.json}
PDA_CIF_DIR=${PDA_CIF_CACHE_DIR:-/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache}
PDA_TRAIN_OVERLAP=${PDA_TRAIN_OVERLAP_IDS:-/home/jupyter-chenxi/prune_work/eval_out/pda_train_overlap_ids.json}
JAX=${STOCK_AF2_JAX:-/home/jupyter-chenxi/params/params_model_1_ptm.npz}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/runB_full_stack_pda_eval}
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
# ⛔ The two mixing knobs. Both are user-specified values (2026-08-18): p=0.5 and top-up-to-20.
T2_REPLACE_PROB=${T2_REPLACE_PROB:-0.5}
T2_TOPUP_TO=${T2_TOPUP_TO:-20}
# Table of prefiltered natural-hit counts (prune_work/build_prefiltered_counts.py). The top-up rule is
# defined on that number and the featurizer does not report it; the stored release-date cutoff is
# asserted against --max_template_date so a stale table is an error, not a wrong count.
T2_PREF_COUNTS=${T2_PREF_COUNTS:-/home/jupyter-chenxi/pp1c_work/prefiltered_counts.npz}
# T4 promoted-template pool. Under $OUT so it belongs to THIS run and cannot be picked up by
# another; each DDP rank writes only its own rank<N>/ subtree, so no locking is needed.
T4_POOL=${T4_POOL_DIR:-$OUT/t4_pool}
EXPLORE_K=${EXPLORE_K:-5}
EXPLORE_SELECT=${EXPLORE_SELECT:-plddt}
# ⛔ REQUIRED. Without it the npz rows are placed by residue_index - 1, which desynchronises at
# the first unresolved residue and is what killed launch #2 (1eis_A, 70/85 positions wrong).
# data_modules.py asserts on its absence rather than silently falling back.
# ⭐ v2 = the rebuild with the symmetric-X matcher, which recovered the 448 chains the first
# build dropped (all of them via residue_index-1; see prune_work/build_query_index_map.py).
# 82730/82733 usable vs 82282 before. The live T2 run reads the v1 file and is unaffected.
T2_QMAP=${T2_QMAP:-/home/jupyter-chenxi/pp1c_work/qmap_all_v2.npz}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$JAX" ] || { echo "ERROR: stock AF2 jax params not found: $JAX"; exit 1; }
[ -f "$T2_INDEX" ] || { echo "ERROR: T2 template index not found: $T2_INDEX"; exit 1; }
[ -d "$T2_ROOT" ]  || { echo "ERROR: T2 templates root not found: $T2_ROOT"; exit 1; }
[ -f "$T2_QMAP" ] || { echo "ERROR: T2 query-index map not found: $T2_QMAP"; exit 1; }
[ -f "$T2_PREF_COUNTS" ] || { echo "ERROR: prefiltered-count table not found: $T2_PREF_COUNTS"; exit 1; }
# Auto-resume this run's own checkpoint (full state); else warm-start from stock AF2 jax params.
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_jax_params "$JAX")
  echo "INIT (warm-start) from stock AF2 jax params: $JAX"
fi
echo "RUN B (full stack): index=$T2_INDEX root=$T2_ROOT band=$T2_MIN_TM-$T2_MAX_TM qmap=$T2_QMAP"
echo "  tricks: contractive_recycling + gaussian_pair_init | explore: K=$EXPLORE_K select=$EXPLORE_SELECT"
echo "  T4: n_promoted=32 max_per_chain=64 promote_after_epoch=5 pool=$T4_POOL"
echo "  mixing: replace_prob=$T2_REPLACE_PROB  topup_to=$T2_TOPUP_TO  counts=$T2_PREF_COUNTS"
echo "  lists (all-X excluded): train=$TRAIN val=$VAL"
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
  --t2_min_tm "$T2_MIN_TM" --t2_max_tm "$T2_MAX_TM" \
  --t2_replace_prob "$T2_REPLACE_PROB" --t2_topup_to "$T2_TOPUP_TO" \
  --t2_prefiltered_counts "$T2_PREF_COUNTS" \
  --contractive_recycling --gaussian_pair_init \
  --explore_k "$EXPLORE_K" --explore_select "$EXPLORE_SELECT" \
  --t4_self_distill --t4_n_promoted 32 --t4_max_per_chain 64 \
  --t4_promote_after_epoch 5 --t4_pool_dir "$T4_POOL" \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --grad_accum_steps "$GRAD_ACCUM" \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
