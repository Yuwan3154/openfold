#!/bin/bash
# Joint conf+structure tuning SMOKE for ONE conf-weight multiplier (env CONF_MULT, default 1).
# EVOFORMER-ONLY (--freeze_non_evoformer): conf heads stay FROZEN, so conf-loss gradients flow only to
# the trunk -> confidence can improve only by the trunk producing reps the fixed/calibrated heads read
# as confident (requires genuinely good structure; no always-unconfident collapse).
# Structure losses at config defaults; conf losses = AF2 finetuning_ptm ratios (plddt 0.01,
# experimentally_resolved 0.01, tm 0.1) x CONF_MULT. Resume weights-only from conf_smoke_init.ckpt
# (slim LAST 24-block weights, fresh warmup). ~100 steps, single GPU, no checkpoint, per-step log.
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # single GPU for the smoke
ulimit -n 65536
M=${CONF_MULT:-1}
STEPS=${STEPS:-100}
LR=${LR:-5e-4}
WARMUP=${WARMUP:-1000}        # slim protocol; first STEPS run on the warmup ramp (controlled small LR)
MM=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files
ALN=/home/jupyter-chenxi/data/openproteinset_aln
KAL=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/kalign
OBS=/home/jupyter-chenxi/data/pdb_mmcif/obsolete.dat
CACHE=/home/jupyter-chenxi/data/pdb_mmcif/mmcif_cache.json
L=/home/jupyter-chenxi/prune_work/lists_pdb
KEEP="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,47"   # 24-block slim
INIT=${INIT_CKPT:-/home/jupyter-chenxi/runs/conf_smoke_init.ckpt}
TRAIN=${TRAIN_LIST:-$L/slim_struct_train.list}
VAL=${VAL_LIST:-$L/slim_struct_val.list}
OUT=${OUT_DIR:-/tmp/conf_smoke_M${M}}
[ -f "$INIT" ] || { echo "ERROR: init ckpt not found: $INIT (run build_conf_smoke_init.py)"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
# conf-loss weights = AF2 ratios x M (structure losses untouched -> keep config defaults)
P=$(python -c "print(0.01*$M)"); E=$(python -c "print(0.01*$M)"); T=$(python -c "print(0.1*$M)")
printf '{"loss.plddt_loss.weight": %s, "loss.experimentally_resolved.weight": %s, "loss.tm.weight": %s}\n' "$P" "$E" "$T" > "$OUT/conf.json"
echo "CONF_MULT=$M  weights: plddt=$P exp_resolved=$E tm=$T  steps=$STEPS lr=$LR"
python train_openfold.py "$MM" "$ALN" "$MM" "$OUT" 2018-04-30 \
  --config_preset finetuning_ptm \
  --kalign_binary_path "$KAL" --obsolete_pdbs_file_path "$OBS" --template_release_dates_cache_path "$CACHE" \
  --evoformer_keep_block_indices "$KEEP" --freeze_non_evoformer \
  --resume_from_ckpt "$INIT" --resume_model_weights_only true \
  --experiment_config_json "$OUT/conf.json" \
  --train_chain_list_path "$TRAIN" \
  --val_data_dir "$MM" --val_alignment_dir "$ALN" --val_chain_list_path "$VAL" \
  --precision bf16 --learning_rate "$LR" --warmup_no_steps "$WARMUP" \
  --train_epoch_len "$STEPS" --max_epochs 1 --num_sanity_val_steps 0 \
  --checkpoint_every_n_steps 999999 --checkpoint_monitor val/lddt_ca --checkpoint_save_top_k 0 \
  --log_lr --log_every_n_steps 1 --seed 42 --distributed_backend nccl
