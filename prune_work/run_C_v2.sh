#!/bin/bash
# RUN C v2 -- identical recipe and identical OBJECTIVE. One change only: fix B is now live.
#
#   --checkpoint_save_top_k 5   was DECLARED BUT NEVER READ (save_top_k hardcoded to 1). Fixed in
#       a3bccfa, so this finally takes effect. With top_k=1 each new best DELETED the previous one,
#       which on 2026-08-23 discarded ep2's weights -- the run's best PDA score -- because the
#       monitored 906-mean fell that epoch. top_k=5 stops the destruction.
#
#   ⛔ --checkpoint_monitor stays val/lddt_ca (the COMBINED 906-entry mean). An earlier proposal to
#      repoint it at val/lddt_ca_src_pda is WITHDRAWN: natural-protein de novo prediction is the
#      project's actual goal and PDA is only the easier instrument for observing change, so the
#      906-mean's 2:1 weighting toward natural chains is CORRECT and deliberate.
#
#   run dir -> runs/runC_v2 with a FRESH t4_pool. The pool is not carried over: runC's ~36k records
#      would hand this run thousands of chains of promoted templates its own training never produced.
#
# Everything else byte-identical, including the EMA warm start from runB best-010 and the scale-leak
# fix (5d25eb1). Training is deterministic under --seed 42, so epochs 0-2 will reproduce exactly.
# Generated from Run B's own argv snapshot (/home/jupyter-chenxi/prune_work/runB_argv.txt) so every hyperparameter not listed
# below is byte-identical to Run B. Changes, each either a user decision or forced by a guard:
#
#   run dir            -> /home/jupyter-chenxi/runs/runC_replica_exchange
#   --resume_from_ckpt -> best-010-008250.ckpt (val/lddt_ca 0.7198 all / 0.7699 modelable)
#   --resume_model_weights_only true   FORCED: cd4afe3 changed the SHAPE of the contractive `b`
#                        (vector -> [c_z,c_z]). Weights migrate losslessly to diag(b) -- verified
#                        BIT-EXACT on this very checkpoint -- but optimizer state cannot follow.
#   --resume_from_ema true             FORCED by correctness, added 2026-08-22: weights-only
#                        resume loads ckpt["state_dict"], the LIVE weights, but validation runs
#                        on the EMA -- so the 0.7198 that SELECTED this checkpoint describes its
#                        EMA weights, and the live tensors at that step were never evaluated.
#                        MEASURED divergence ||live-ema||/||ema|| = 0.00245, all of it in the two
#                        trainable groups (evoformer 0.00252, recycling_embedder 0.00139); every
#                        frozen group is bit-identical. ~1000 steps of drift, i.e. over an epoch.
#                        ⭐ This also BUYS A LAUNCH GATE: epoch-0 val should reproduce ~0.7198,
#                        which end-to-end verifies the warm start AND the b->diag(b) migration.
#   --explore_k 5 -> 4  FORCED: train_openfold asserts len(--explore_noise_ladder) == --explore_k.
#   --explore_noise_ladder 0,8,16,32   the approved ladder. tau=0 is the deterministic rung.
#   --t4_promote_all                   all 4 rungs' samples enter the pool, not just the winner.
#   --t4_n_promoted 32 -> 64           user decision.
#   --t4_pool_dir      -> a FRESH pool. Run B's 8,969 records were produced by a different
#                         combination step (vector b, no ladder); reusing them would seed this run
#                         with templates from a model it no longer is.
#   --pda_val_manifest -> the MODELABLE 306, plus the 300+300 expanded split (906 total).
#   --t4_promote_after_epoch 5 -> 0    user decision: warm-started from a trained checkpoint, so
#                        there is no barely-warmed-up model whose output needs excluding.
#   --explore_switch_epoch 10 -> 0     user decision. The selector is pTM from step 0; the model is
#                        already trained so there is no phase-1 to serve. MEASURED on 5468209: pTM
#                        selection recovers 81.3%% of the oracle gain at K=4 over this ladder.
#   --pda_nonneural_ids  the 51 PDA entries whose paper names NO neural structure predictor, logged
#                        as val/<metric>_nonneural. AF2-circularity-free -- and EXPECTED TO SCORE
#                        LOWER, since pre-DL and rational/manual designs dominate it.
#
# ⛔ UNCHANGED AND WORTH A DELIBERATE LOOK BEFORE LAUNCH:
#
# ⛔ NOT LAUNCHED. Run B is still training -- stop it first, never overlap GPU jobs.
set -e
# Environment mirrored VERBATIM from prune_work/run_B_full_stack.sh:37-47. The generated launcher was
# built from Run B's argv, which does not carry env -- PYTHONPATH in particular is load-bearing.
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold-esmfold2-recycling/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[ "$NGPU" -le 2 ] && export NCCL_P2P_DISABLE=1   # 0<->1 PCIe P2P is broken on this box
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
python \
  train_openfold.py \
  /home/jupyter-chenxi/data/pdb_mmcif/mmcif_files \
  /home/jupyter-chenxi/data/openproteinset_aln \
  /home/jupyter-chenxi/data/pdb_mmcif/mmcif_files \
  /home/jupyter-chenxi/runs/runC_v2 \
  2018-04-30 \
  --config_preset \
  finetuning_ptm \
  --kalign_binary_path \
  /home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign \
  --obsolete_pdbs_file_path \
  /home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat \
  --template_release_dates_cache_path \
  /home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json \
  --enable_single_seq_mode \
  --single_seq_keep_templates \
  --freeze_non_evoformer \
  --validate_without_templates \
  --pda_val_manifest \
  /home/jupyter-chenxi/prune_work/eval_out/pda_cluster_representatives_modelable.json \
  --pda_cif_cache_dir \
  /home/jupyter-chenxi/prune_work/eval_out/pda_mmcif_cache \
  --pda_train_overlap_ids \
  /home/jupyter-chenxi/prune_work/eval_out/pda_train_overlap_ids.json \
  --resume_from_ckpt \
  /home/jupyter-chenxi/runs/runB_full_stack_pda_eval/lightning_logs/version_1/checkpoints/best-010-008250.ckpt \
  --train_chain_list_path \
  /home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_train.list.noallx \
  --val_data_dir \
  /home/jupyter-chenxi/data/pdb_mmcif/mmcif_files \
  --val_alignment_dir \
  /home/jupyter-chenxi/data/openproteinset_aln \
  --val_chain_list_path \
  /home/jupyter-chenxi/prune_work/lists_pdb/ws5_val_strict_clean.list.noallx \
  --t2_template_index \
  /home/jupyter-chenxi/pp1c_work/index_band.npz \
  --t2_templates_root \
  /home/jupyter-chenxi/pp1c_work/templates_band \
  --t2_qmap \
  /home/jupyter-chenxi/pp1c_work/qmap_all_v2.npz \
  --t2_min_tm \
  0.3 \
  --t2_max_tm \
  0.9 \
  --t2_replace_prob \
  0.5 \
  --t2_topup_to \
  20 \
  --t2_prefiltered_counts \
  /home/jupyter-chenxi/pp1c_work/prefiltered_counts.npz \
  --contractive_recycling \
  --gaussian_pair_init \
  --explore_k \
  4 \
  --explore_select \
  hybrid \
  --explore_switch_epoch \
  0 \
  --t4_self_distill \
  --t4_n_promoted \
  64 \
  --t4_max_per_chain \
  64 \
  --t4_promote_after_epoch \
  0 \
  --t4_pool_dir \
  /home/jupyter-chenxi/runs/runC_v2/t4_pool \
  --precision \
  bf16 \
  --learning_rate \
  1e-4 \
  --warmup_no_steps \
  3000 \
  --train_epoch_len \
  3000 \
  --max_epochs \
  100 \
  --num_sanity_val_steps \
  0 \
  --grad_accum_steps \
  1 \
  --checkpoint_every_n_steps \
  20 \
  --checkpoint_monitor \
  val/lddt_ca \
  --checkpoint_save_top_k \
  5 \
  --log_lr \
  --log_every_n_steps \
  20 \
  --seed \
  42 \
  --distributed_backend \
  nccl \
  --resume_model_weights_only \
  true \
  --resume_from_ema \
  true \
  --explore_noise_ladder \
  0,8,16,32 \
  --t4_promote_all \
  --expanded_val_easy \
  /home/jupyter-chenxi/prune_work/val_expanded/v384/val_300_easy.json \
  --expanded_val_hard \
  /home/jupyter-chenxi/prune_work/val_expanded/v384/val_300_hard.json \
  --expanded_val_cif_dir \
  /home/jupyter-chenxi/data/pdb_mmcif/mmcif_files \
  --pda_nonneural_ids \
  /home/jupyter-chenxi/prune_work/eval_out/pda_nonneural_strict_ids.json
