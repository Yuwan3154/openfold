#!/bin/bash
# WS5: single-sequence PRUNED training WITH templates (MSA-free query + template channel; design backbone).
# Prune = drop col-attn + tri-attn (keep tri-mul/row-attn/OPM/transitions), evoformer-only, warm-start AF2 jax.
# Templates KEPT via --single_seq_keep_templates (user requirement); finetuning_ptm preset.
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold     # MANDATORY (else enhanced_data_utils import fails -> 0 chains)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}  # --gpus is a dead arg; control GPUs here
ulimit -n 65536                                              # DataLoader FD fix (pairs with set_sharing_strategy)
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln            # single-seq uses only the QUERY seq; templates from pdb70 hits
JAX=/home/jupyter-chenxi/params/params_model_1_ptm.npz
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}             # ABSOLUTE; temporal split (88155)
VAL=${VAL_LIST:-$L/slim_struct_val.list}                   # ABSOLUTE (200)
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/prune_singleseq_v1}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN (cwd=$(pwd))"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL (cwd=$(pwd))"; exit 1; }
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --prune_evoformer --enable_single_seq_mode --single_seq_keep_templates --freeze_non_evoformer \
  --resume_from_jax_params "$JAX" \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate 1e-4 --warmup_no_steps 3000 \
  --train_epoch_len 1000 --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
