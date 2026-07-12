#!/bin/bash
# Single-variable ablation vs run_prune_singleseq_esmfold2_v1.sh (v4, stopped 2026-07-12 to launch
# this run): everything identical (prune_evoformer, freeze_non_evoformer, contractive_recycling,
# gaussian_pair_init, validate_without_templates, PDA eval set) EXCEPT real MSA is left ON
# (finetuning_ptm's own max_msa_clusters=512/max_extra_msa=5120, not clamped to 1) -- tests
# whether OuterProductMean can still extract a genuine covariation/contact signal into the pair
# track without column attention (dropped by --prune_evoformer), approximating classical
# APC-corrected contact-map extraction, as a possible mitigant for the memorization behavior
# found in the PDA train-overlap investigation (see ESMFOLD2_RECYCLE_SCALING.md).
# masked_msa loss is explicitly zeroed (msaon_config_overrides.json) to isolate the OPM/pair-track
# pathway alone, without also reintroducing an MSA-derived (homology-space) training objective.
# crop_size is pinned to 256 (same override file) to match v4 exactly, since finetuning_ptm's own
# default (384) would otherwise be a second, uncontrolled variable.
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold  # MANDATORY, see run_prune_singleseq_esmfold2_v1.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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
MSAON_CONFIG_JSON=${MSAON_CONFIG_JSON:-/home/jupyter-chenxi/openfold-esmfold2-recycling/prune_work/msaon_config_overrides.json}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/prune_evoformer_msaon_v1_pda_eval}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-3000}
GRAD_ACCUM=${GRAD_ACCUM:-4}
# v4's own best checkpoint (by val/lddt_ca) -- the INIT source for this run's weights, per user
# directive (2026-07-12): continue from v4's already-trained state rather than WS5's, so this run
# tests "turn MSA on from here" rather than "retrain from scratch with MSA on".
V4_CKPT_DIR=/home/jupyter-chenxi/runs/prune_singleseq_esmfold2_v4_pda_eval/lightning_logs/version_7/checkpoints
INIT_CKPT=${INIT_CKPT:-$(ls -t "$V4_CKPT_DIR"/best-*.ckpt 2>/dev/null | head -1)}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -f "$MSAON_CONFIG_JSON" ] || { echo "ERROR: MSA-on config overrides json not found: $MSAON_CONFIG_JSON"; exit 1; }
[ -n "$INIT_CKPT" ] || { echo "ERROR: no init checkpoint found in $V4_CKPT_DIR"; exit 1; }
# Auto-resume: if THIS run already has its own checkpoint, resume full state from it; else
# weights-only init from v4's best checkpoint (first launch).
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_ckpt "$INIT_CKPT" --resume_model_weights_only true)
  echo "INIT (weights-only) from v4's best checkpoint: $INIT_CKPT"
fi
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --freeze_non_evoformer \
  --contractive_recycling --gaussian_pair_init --validate_without_templates \
  --experiment_config_json "$MSAON_CONFIG_JSON" \
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
