#!/bin/bash
# WS5 + ESMFold2 tricks: same pruned/single-seq/templated architecture as WS5's own
# run_prune_singleseq.sh, PLUS --contractive_recycling --gaussian_pair_init (Appendix A.2.5,
# arXiv:2604.12946). Weights-only INIT from WS5's latest checkpoint (not stock AF2 jax) on
# first launch; auto-resumes ITS OWN progress (full state) on subsequent launches if a
# checkpoint already exists here. max_len=256 (matches WS5's existing single-seq-mode default,
# made explicit). Recycle count uniformly sampled from {0,1,2,3} at train time -- this is
# OpenFold's OWN existing "uniform_recycling" mechanism (config.py's "train" stage default,
# ALREADY true, no new code needed) combined with max_recycling_iters=3 (finetuning_ptm's
# existing default) -- NOT something this launcher needs to configure further.
# --validate_without_templates: TRAINING keeps templates ON, but VALIDATION (checkpoint
# selection via val/lddt_ca) runs WITHOUT templates -- true single-sequence prediction, not
# template-assisted performance.
# --pda_val_manifest/--pda_cif_cache_dir: validation POPULATION is now real de novo protein
# designs (PDA, Foldseek-clustered to 425 structurally-distinct representatives at TM-score>=0.5),
# not natural-protein chains -- corrects the original directive's intent (de novo protein true
# single-sequence prediction, not natural-protein structure prediction), which the first version
# of this launcher only half-implemented (template-toggle only, wrong population).
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold  # MANDATORY (else block_replacement_scripts.enhanced_data_utils import fails silently -> chain_list_path ignored, eval_dataset = ALL alignment_dir entries instead of the intended val list -- root-caused via direct OpenFoldDataModule construction + diff against WS5's own working launcher convention)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SINGLE_SEQ_MAX_CROP=256
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
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/prune_singleseq_esmfold2_v2_pda_eval}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-1000}
GRAD_ACCUM=${GRAD_ACCUM:-1}
# WS5's own latest checkpoint -- the INIT source for this run's weights (NOT stock full AF2).
WS5_CKPT_DIR=/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints
WS5_INIT_CKPT=${WS5_INIT_CKPT:-$(ls -t "$WS5_CKPT_DIR"/best-*.ckpt 2>/dev/null | head -1)}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$PDA_MANIFEST" ] || { echo "ERROR: PDA val manifest not found: $PDA_MANIFEST"; exit 1; }
[ -d "$PDA_CIF_DIR" ]  || { echo "ERROR: PDA cif cache dir not found: $PDA_CIF_DIR"; exit 1; }
[ -n "$WS5_INIT_CKPT" ] || { echo "ERROR: no WS5 init checkpoint found in $WS5_CKPT_DIR"; exit 1; }
# Auto-resume: if THIS run already has its own checkpoint, resume full state from it; else
# weights-only init from WS5's latest checkpoint (first launch).
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  RESUME=(--resume_from_ckpt "$CK")
  echo "RESUME (full state) from this run's own checkpoint: $CK"
else
  RESUME=(--resume_from_ckpt "$WS5_INIT_CKPT" --resume_model_weights_only true)
  echo "INIT (weights-only) from WS5's latest checkpoint: $WS5_INIT_CKPT"
fi
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --contractive_recycling --gaussian_pair_init --validate_without_templates \
  --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
  "${RESUME[@]}" \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --grad_accum_steps "$GRAD_ACCUM" \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
