#!/bin/bash
# Emits ONLY: validation boundaries, failure signatures, death, hourly heartbeat.
# Errors are read from the log the poller RESOLVED (printed in every event), never a named file.
prev_ev=0; prev_run=""; i=0
while true; do
  i=$((i+1))
  out=$(ssh -n -o ConnectTimeout=30 A6000_offsite 'bash /home/jupyter-chenxi/prune_work/poll_once.sh' 2>/dev/null)
  if [ -z "$out" ]; then
    [ $((i % 12)) -eq 0 ] && echo "HEARTBEAT: ssh unreachable (transient)"
    sleep 300; continue
  fi
  ev=$(echo "$out"  | grep -oE 'VALEV=[0-9]+' | cut -d= -f2)
  al=$(echo "$out"  | grep -oE 'ALIVE=[0-9]+' | cut -d= -f2)
  run=$(echo "$out" | grep -oE 'RUN=[^ ]+'    | cut -d= -f2)
  # a different run dir means a relaunch: reset the validation counter rather than mis-compare
  if [ "$run" != "$prev_run" ]; then prev_ev=0; prev_run="$run"; echo "MONITOR: now tracking RUN=$run"; fi

  err=$(ssh -n -o ConnectTimeout=30 A6000_offsite 'L=$(ls -t /home/jupyter-chenxi/runs/*.log | head -1); tail -c 20000 "$L" | tr "\r" "\n" | grep -aiE "traceback|CUDA error|out of memory|RuntimeError|AssertionError|NCCL.*(timeout|abort)|Killed|Segmentation" | tail -3' 2>/dev/null)
  [ -n "$err" ] && echo "RUN C ERROR SIGNATURE [$run]: $err"

  if [ "${ev:-0}" -gt "$prev_ev" ]; then echo "VALIDATION COMPLETE (event $ev) -- $out"; prev_ev=$ev; fi
  if [ "${al:-0}" -eq 0 ]; then echo "TRAINING IS GONE (ALIVE=0) -- $out"; exit 1; fi
  [ $((i % 12)) -eq 0 ] && echo "HEARTBEAT ok -- $out"
  sleep 300
done
