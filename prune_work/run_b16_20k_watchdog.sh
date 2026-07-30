#!/bin/bash
# WS4-BC push: soft, lr2e-4, batch16, L<=128, STEPS=20000. Now checkpoint/resume-capable
# (ws4_bc_train.py commit c6dee13+), so this watchdog can safely relaunch on any crash --
# it picks up from CKPT_TAG's *_last.pt automatically, no lost progress beyond the last
# EVAL_EVERY-step save.
BASE=/home/jupyter-chenxi
PY=$BASE/miniconda3/envs/cue_openfold_gated/bin/python
L=/tmp/ws4push
LOG=$L/push_soft_b16_20k.log
cd $BASE/prune_work || exit 1
echo "=== watchdog started $(date) ===" >> $L/watchdog_20k.log
while true; do
  if grep -q "^DONE" "$LOG" 2>/dev/null; then
    echo "DONE detected, watchdog exiting $(date)" >> $L/watchdog_20k.log
    break
  fi
  if ! pgrep -f "[w]s4_bc_train.py" > /dev/null; then
    echo "not running, (re)launching $(date)" >> $L/watchdog_20k.log
    CUDA_VISIBLE_DEVICES=3 STUDENT=converged2 POINTS=soft STEPS=20000 BATCH=16 VAL_N=80 MAXLEN=128 LR=2e-4 EVAL_EVERY=50 \
      nohup $PY ws4_bc_train.py >> "$LOG" 2>&1 &
  fi
  sleep 120
done
