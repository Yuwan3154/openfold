#!/bin/bash
# ============================================================================================
# T2-MATCHED (2026-08-18 user directive: "Get this ready as a possible next run for a comparison").
# T2's recipe with ONE change: the synthetic templates REPLACE natural hits instead of being
# appended, so the template POOL stays exactly the size T1's was.
#
# ⭐⭐ WHY THIS RUN EXISTS. The live T2 differs from T1 in TWO ways at once, not one:
#   content: half the delivered templates are synthetic (TM 0.3-0.9, near-uniform difficulty)
#   COUNT:   `templates_crop_start ~ Uniform{0..pool}` is INCLUSIVE, so appending 4 to a pool of 4
#            moves the delivered count from mean 2.00/step (P(0 templates) 20%) to mean 2.89 (11.1%)
# A T1-vs-T2 gap therefore cannot be attributed to template content. `--t2_replace_natural` clamps
# the request to the number of natural hits available and drops exactly as many naturals as
# synthetic templates actually arrived, so the pool size -- and hence the whole delivered-count
# distribution -- is IDENTICAL to T1's, leaving content as the single variable.
# The surviving natural hits are a UNIFORM random subset, not the top-k by sum_probs, so the natural
# component is also distributed exactly as in T1 (keeping the best-k would hand this run better
# templates than T1's average -- a second difference, biased in the flattering direction).
#
# ⛔⛔ T2_N_SYNTHETIC MEANS SOMETHING DIFFERENT HERE AND HAS NO DEFAULT ON PURPOSE. In append mode
# `--t2_n_synthetic 4` gave a 50/50 content mix (4 synthetic beside 4 natural). In REPLACE mode it
# is the number of natural hits given up, so with the usual 4 natural hits:
#     T2_N_SYNTHETIC=2  -> 2 synthetic + 2 natural = 50/50 content, count-matched  (the direct
#                          content-only analogue of the live T2 run)
#     T2_N_SYNTHETIC=4  -> 4 synthetic + 0 natural = 100% synthetic, count-matched (measures the
#                          synthetic templates alone, with no natural hits to fall back on)
#     T2_N_SYNTHETIC=1  -> 1 synthetic + 3 natural = 25% content
# Which of these to run is a live experimental decision, so this launcher REFUSES to start without
# it rather than picking one. Set it explicitly:  T2_N_SYNTHETIC=2 ./run_stock_af2_nomsa_t2_matched.sh
#
# ⛔ Do NOT add the tricks here -- that would collapse stages 2 and 3.
# ⛔ Do NOT point OUT_DIR at T1's or T2's run dir: the auto-resume block below finds any last.ckpt
#    inside $OUT, and a stale one silently turns this into a continuation of that run instead of a
#    clean warm-start from the jax params. (Launch #3 of T2 was aborted for exactly this.)
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
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}
VAL=${VAL_LIST:-$L/ws5_val_strict_clean.list}
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
# ⛔ no default -- see the header. An unset value is a user decision, not something to guess.
T2_N=${T2_N_SYNTHETIC:?set T2_N_SYNTHETIC explicitly (2 = 50/50 content, 4 = all-synthetic); see this script's header}
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
# Auto-resume this run's own checkpoint (full state); else warm-start from stock AF2 jax params.
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_jax_params "$JAX")
  echo "INIT (warm-start) from stock AF2 jax params: $JAX"
fi
echo "T2-MATCHED (replace mode): index=$T2_INDEX root=$T2_ROOT band=$T2_MIN_TM-$T2_MAX_TM n_replace=$T2_N qmap=$T2_QMAP"
echo "  pool size stays at the natural count -> delivered-count distribution identical to T1"
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
  --t2_replace_natural \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --grad_accum_steps "$GRAD_ACCUM" \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
