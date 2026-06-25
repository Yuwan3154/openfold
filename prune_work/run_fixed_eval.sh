#!/bin/bash
# Trivial git-synced launcher for the fixed-eval structure-stability sweep (ngrok-drop safe).
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
ulimit -n 65536
python prune_work/fixed_eval_driver.py \
  --mults ${FE_MULTS:-0.003,0.01,0.03,0.1,0.3,1,3} \
  --steps ${FE_STEPS:-100} --lr ${FE_LR:-5e-4} --gpus 0,1,2,3
