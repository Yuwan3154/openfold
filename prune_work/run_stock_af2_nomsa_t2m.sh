#!/bin/bash
# ============================================================================================
# T2-MATCHED (user directives 2026-08-18). T2's recipe with the mixing policy the user specified,
# plus the all-X chain exclusion. Two rules, applied in this order per training example:
#
#   (1) TOP-UP  --t2_topup_to 20
#       A chain with fewer than 20 PREFILTERED natural hits has its pre-shuffle pool topped up to 20
#       with synthetic templates. 20 is config.data.train.shuffle_top_k_prefiltered -- the pool the
#       featurizer actually shuffles before taking max_template_hits=4 -- so this is the template-poor
#       case the synthetic templates exist for. Measured: 11.6% of chains have <20 prefiltered hits,
#       0.5% have none at all.
#       ⚠️ For the 1.3% with <4 hits this RAISES the delivered count above T1's. Intended: the other
#       98.7% stay count-matched, and those are the chains a fair comparison rests on.
#
#   (2) PROBABILISTIC REPLACEMENT  --t2_replace_prob 0.5
#       Each surviving natural template is independently replaced by a synthetic one with p=0.5.
#       ⭐⭐ The delivered-template COUNT distribution is EXACTLY T1's (mean 2.00/step, P(0)=20%) and
#       the delivered synthetic count is Binomial(delivered, 0.5). Implemented per POOL slot, which is
#       distributionally identical to per DELIVERED slot because random_crop_to_size picks its window
#       independently of which slots are synthetic -- verified by Monte Carlo against the real torch
#       calls in tests/test_synthetic_templates.py.
#
# ⭐ SUPERSEDES the earlier --t2_replace_natural design (fixed count of naturals dropped at pool
#   level). That flag is GONE; do not resurrect it.
#
# ⛔⛔ ALL-X CHAINS ARE EXCLUDED (user): 243 training chains and 1 val chain have a seqres that is
#   entirely "X" (non-canonical residues -> aatype 20 everywhere), so the model has no way to know what
#   the sequence means and the example carries no signal. This launcher points at the *.noallx lists
#   (87912 train / 53 val) written by prune_work/scan_allx_chains.py.
#
# ⛔ Do NOT add the tricks here -- that would collapse stages 2 and 3.
# ⛔ Do NOT point OUT_DIR at T1's or T2's run dir: the auto-resume block finds any last.ckpt inside
#    $OUT, and a stale one silently turns this into a continuation instead of a clean jax warm-start.
#    (T2 launch #3 was aborted for exactly this.)
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
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/stock_af2_nomsa_t2m_pda_eval}
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
echo "T2-MATCHED: index=$T2_INDEX root=$T2_ROOT band=$T2_MIN_TM-$T2_MAX_TM qmap=$T2_QMAP"
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
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --grad_accum_steps "$GRAD_ACCUM" \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
