#!/bin/bash
# Minimal launcher for conf_grad_probe.py so the over-the-wire ssh command stays trivial (ngrok-drop safe).
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
ulimit -n 65536
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=${PROBE_GPU:-3} python prune_work/conf_grad_probe.py --n ${PROBE_N:-4} --crop ${PROBE_CROP:-64}
