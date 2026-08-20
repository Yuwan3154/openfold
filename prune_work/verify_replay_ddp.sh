#!/bin/bash
# ============================================================================================
# DDP REPLAY CORRECTNESS CHECK (user directive 2026-08-20: "unit test the backprop random seed
# issue under DDP ... this is very important correctness check").
#
# Measures, on the REAL path with real DDP across 4 GPUs, whether the grad-carrying forward
# reproduces the no_grad forward that selected it. Two configs:
#   A = Run B's live config (explore_k 5, no ladder)
#   B = Run C's config     (explore_k 4, ladder 0,1,2,4, promote-all from epoch 0)
#
# ⛔ Warm-starts from the JAX params into ITS OWN output dir, so Run B's checkpoints and T4 pool are
#    untouched and Run B can resume cleanly afterwards.
# ⛔ Short epoch (TRAIN_EPOCH_LEN) so it produces the measurement quickly; kill it once enough
#    replay-verify lines have landed. It is a diagnostic, not a training run.
# ============================================================================================
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
TRAIN=$L/slim_struct_train.list.noallx
VAL=$L/ws5_val_strict_clean.list.noallx
PDA_MANIFEST=/home/jupyter-chenxi/prune_work/eval_out/pda_cluster_representatives.json
PDA_CIF_DIR=/home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache
JAX=/home/jupyter-chenxi/params/params_model_1_ptm.npz
T2_INDEX=/home/jupyter-chenxi/pp1c_work/index_band.npz
T2_ROOT=/home/jupyter-chenxi/pp1c_work/templates_band
T2_QMAP=/home/jupyter-chenxi/pp1c_work/qmap_all_v2.npz
T2_PREF=/home/jupyter-chenxi/pp1c_work/prefiltered_counts.npz
CFG=${CFG:?set CFG=A or CFG=B}
OUT=/home/jupyter-chenxi/runs/verify_replay_$CFG
EPL=${TRAIN_EPOCH_LEN:-120}

if [ "$CFG" = "A" ]; then
  EXTRA=(--explore_k 5 --explore_select hybrid --explore_switch_epoch 10
         --t4_self_distill --t4_n_promoted 32 --t4_max_per_chain 64
         --t4_promote_after_epoch 5 --t4_pool_dir "$OUT/t4_pool")
  echo "CONFIG A = Run B's live config (K=5, no ladder) + replay verification"
else
  # ⛔ promote_after_epoch 0 so the promote-all WRITE path actually executes in this short run.
  EXTRA=(--explore_k 4 --explore_noise_ladder 0,1,2,4 --explore_select hybrid
         --explore_switch_epoch 10
         --t4_self_distill --t4_promote_all --t4_n_promoted 64 --t4_max_per_chain 64
         --t4_promote_after_epoch 0 --t4_pool_dir "$OUT/t4_pool")
  echo "CONFIG B = Run C's config (K=4, ladder 0,1,2,4, promote-all) + replay verification"
fi

python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" \
  --template_release_dates_cache_path "$CACHE" \
  --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --validate_without_templates \
  --pda_val_manifest "$PDA_MANIFEST" --pda_cif_cache_dir "$PDA_CIF_DIR" \
  --resume_from_jax_params "$JAX" \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --t2_template_index "$T2_INDEX" --t2_templates_root "$T2_ROOT" --t2_qmap "$T2_QMAP" \
  --t2_min_tm 0.3 --t2_max_tm 0.9 --t2_replace_prob 0.5 --t2_topup_to 20 \
  --t2_prefiltered_counts "$T2_PREF" \
  --contractive_recycling --gaussian_pair_init \
  --explore_verify_replay \
  "${EXTRA[@]}" \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len "$EPL" --max_epochs 1 --num_sanity_val_steps 0 \
  --checkpoint_every_n_steps 100000 --log_lr --log_every_n_steps 5 \
  --seed 42 --distributed_backend nccl
