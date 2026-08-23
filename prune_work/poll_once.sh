#!/bin/bash
. /home/jupyter-chenxi/miniconda3/etc/profile.d/conda.sh
conda activate cue_openfold_gated
# ⛔ Pick the NEWEST launch log, never a hardcoded one: after a relaunch the old log is frozen and
# still holds the previous run's SIGTERM teardown errors, which a grep re-reports as fresh failures.
LOG=$(ls -t /home/jupyter-chenxi/runs/runC_launch*.log 2>/dev/null | head -1)
EV=$(nice -n 19 python /home/jupyter-chenxi/prune_work/dump_epochs.py \
      /home/jupyter-chenxi/runs/runC_replica_exchange 2>/dev/null \
      | grep -oE 'validation\) events: [0-9]+' | grep -oE '[0-9]+$')
BEST=$(ls /home/jupyter-chenxi/runs/runC_replica_exchange/lightning_logs/version_*/checkpoints/best-* 2>/dev/null | wc -l)
AL=$(pgrep -cf "[r]unC_replica_exchange")
PROG=$(tail -c 800 "$LOG" | tr '\r' '\n' \
       | grep -oE 'Epoch [0-9]+: +[0-9]+%[^,]*, +[0-9.]+it/s' | tail -1)
echo "VALEV=${EV:-0} BEST=$BEST ALIVE=$AL LOG=$(basename $LOG) | $PROG"
