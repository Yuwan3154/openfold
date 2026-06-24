#!/bin/bash
# WS5 auto-launcher: poll until ALL 4 GPUs are free (comfyui released; <2000 MiB used, for 2 consecutive
# checks to skip a brief gap), then launch the resume-aware WS5 training once and exit.
# Arm detached: setsid nohup bash prune_work/ws5_drive.sh </dev/null >/dev/null 2>&1 &
LOG=/home/jupyter-chenxi/prune_work/ws5_drive.log
TLOG=/home/jupyter-chenxi/prune_work/prune_ss_v1.log
echo "$(date) WS5 auto-launcher armed; polling for free GPUs" >> "$LOG"
streak=0
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
  if [ "$busy" -eq 0 ]; then streak=$((streak+1)); else streak=0; fi
  if [ "$streak" -ge 2 ]; then
    echo "$(date) GPUs free -> launching WS5" >> "$LOG"
    cd /home/jupyter-chenxi/openfold
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash prune_work/run_prune_singleseq.sh > "$TLOG" 2>&1 &
    echo "$(date) launched WS5 pid=$!" >> "$LOG"
    break
  fi
  sleep 60
done
