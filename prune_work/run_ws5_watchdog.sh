#!/bin/bash
# WS5 (pruned single-seq+templates) continuation watchdog. Validates against the RIGOROUSLY
# deduplicated strict-clean-54 list (ws5_val_strict_clean.list, MMseqs2 --min-seq-id 0.3 -c 0.8
# clustering vs the k-mer-containment proxy's clean-74) per the 2026-07-04 D21 upgrade.
# Uses ALL 4 GPUs per explicit user instruction (WS4-BC push run stays paused).
# run_prune_singleseq.sh is already resume-aware (auto last.ckpt); this just relaunches on crash.
BASE=/home/jupyter-chenxi
L=/tmp/ws4push
cd $BASE/openfold/prune_work || exit 1
echo "=== WS5 watchdog started $(date) ===" >> $L/ws5_watchdog.log
while true; do
  if ! pgrep -f "[t]rain_openfold.py" > /dev/null; then
    echo "not running, (re)launching $(date)" >> $L/ws5_watchdog.log
    CUDA_VISIBLE_DEVICES=0,1,2,3 VAL_LIST=$BASE/prune_work/lists_pdb/ws5_val_strict_clean.list \
      nohup bash run_prune_singleseq.sh >> $L/ws5_resume_strict.log 2>&1 &
  fi
  sleep 120
done
