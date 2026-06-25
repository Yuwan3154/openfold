#!/bin/bash
# SLIM-12 structure training: trim the converged 24-block slim model to 12 blocks (orig-48 indices
# 0,4,8,12,16,20,24,28,32,36,40,47 = every 4th + last-instead-of-44), warm-start from best-037 via
# slim12_init.ckpt (build with build_slim12_init.py). Evoformer-only, full MSA + templates, standard
# structure losses. Same protocol as run_slim_struct.sh (lr 5e-4, warmup 1000, monitor val/lddt_ca).
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
ulimit -n 65536
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
KEEP="0,4,8,12,16,20,24,28,32,36,40,47"        # 12-of-48: every 4th + 47 instead of 44
INIT=${INIT_CKPT:-/home/jupyter-chenxi/runs/slim12_init.ckpt}
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}  # SAME data as the 24-block slim (88155 / 200)
VAL=${VAL_LIST:-$L/slim_struct_val.list}
OUT=${OUT_DIR:-/home/jupyter-chenxi/runs/slim12_struct_v1}
MAXEP=${MAX_EPOCHS:-100}
SAVE_TOP_K=${SAVE_TOP_K:-5}
EPL=${TRAIN_EPOCH_LEN:-1000}
[ -f "$TRAIN" ] || { echo "ERROR: train list not found: $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "ERROR: val list not found: $VAL"; exit 1; }
[ -f "$INIT" ]  || { echo "ERROR: init ckpt not found: $INIT (run build_slim12_init.py first)"; exit 1; }
# First launch: warm-start from the 12-block init (weights-only, loaded POST-slice -> keys match, fresh optim/LR).
# Resume after crash: full-state from last.ckpt (continues optim/global_step/LR).
CK=$(ls -t "$OUT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$CK" ]; then WARM=(--resume_from_ckpt "$CK"); echo "RESUME (full-state) from $CK"; else WARM=(--resume_from_ckpt "$INIT" --resume_model_weights_only true); echo "WARM-START (weights-only) from $INIT"; fi
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --evoformer_keep_block_indices "$KEEP" \
  --freeze_non_evoformer \
  "${WARM[@]}" \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate 5e-4 --warmup_no_steps 1000 \
  --train_epoch_len "$EPL" --max_epochs "$MAXEP" --num_sanity_val_steps 0 \
  --checkpoint_every_n_steps 20 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k "$SAVE_TOP_K" \
  --log_lr --log_every_n_steps 20 --seed 42 --distributed_backend nccl
