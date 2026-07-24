#!/bin/bash
# One-shot watcher (2026-07-24 user directive): poll all 4 GPUs every 20 min; once ALL report free
# (mem<300MiB, no compute-apps), confirm free for 5 more minutes (60s sub-polls); if still free,
# launch WS5-continued's own resume-aware launcher (default GPUs 0,1 -- no instruction to claim all
# 4 for this run) and exit. If busy again during the confirmation window, drop back to 20-min polling.
BASE=/home/jupyter-chenxi/openfold-esmfold2-recycling/prune_work
LOG=$BASE/resume_ws5_watcher.log
THRESH_MB=300
POLL_SEC=1200
CONFIRM_SUBPOLL_SEC=60
CONFIRM_TOTAL_SEC=300

all_gpus_free() {
  local apps busy
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [ -n "$apps" ] && return 1
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v t="$THRESH_MB" '$1+0>=t')
  [ -n "$busy" ] && return 1
  return 0
}

echo "$(date): watcher started (PID $$), polling every ${POLL_SEC}s for all-4-GPU-free" >> "$LOG"

while true; do
  if all_gpus_free; then
    echo "$(date): all 4 GPUs report free, entering ${CONFIRM_TOTAL_SEC}s confirmation window" >> "$LOG"
    elapsed=0
    confirmed=1
    while [ "$elapsed" -lt "$CONFIRM_TOTAL_SEC" ]; do
      sleep "$CONFIRM_SUBPOLL_SEC"
      elapsed=$((elapsed + CONFIRM_SUBPOLL_SEC))
      if ! all_gpus_free; then
        echo "$(date): GPU became busy again during confirmation window (at +${elapsed}s) -- aborting, back to ${POLL_SEC}s polling" >> "$LOG"
        confirmed=0
        break
      fi
    done
    if [ "$confirmed" -eq 1 ]; then
      echo "$(date): confirmed free for ${CONFIRM_TOTAL_SEC}s -- launching WS5-continued resume (GPUs 0,1)" >> "$LOG"
      cd /home/jupyter-chenxi/openfold-esmfold2-recycling
      nohup bash prune_work/run_prune_singleseq_ws5_continued.sh >> "$BASE/ws5_continued_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
      echo $! > "$BASE/ws5_continued_resume.pid"
      echo "$(date): launched, PID $(cat "$BASE/ws5_continued_resume.pid")" >> "$LOG"
      exit 0
    fi
  else
    echo "$(date): not all free yet" >> "$LOG"
  fi
  sleep "$POLL_SEC"
done
