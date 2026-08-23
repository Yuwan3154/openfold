#!/bin/bash
# ⛔⛔ THIRD-TIME FIX. Twice before, a poller hardcoded a run artifact and went stale on relaunch:
#   (1) a version_0 glob -> silently 2 days stale with a LIVE pid;
#   (2) runC_launch1.log -> replayed the PREVIOUS run's SIGTERM traceback as a fresh failure;
#   (3) a runC_launch*.log glob + a runC_replica_exchange pgrep pattern -> both missed runC_v2.
# Nothing here is named. Everything is resolved FROM THE RUNNING PROCESS, and what was resolved is
# printed, so a stale read is visible in the event itself.
. /home/jupyter-chenxi/miniconda3/etc/profile.d/conda.sh
conda activate cue_openfold_gated

outdir_of() { tr '\0' '\n' < /proc/$1/cmdline 2>/dev/null | sed -n '6p'; }

ROOT=$(pgrep -f "[t]rain_openfold" | head -1)
if [ -z "$ROOT" ]; then
  echo "ALIVE=0 RUN=? LOG=? | no train_openfold process"
  exit 0
fi
RUN=$(outdir_of "$ROOT")
ALIVE=0
for p in $(pgrep -f "[t]rain_openfold"); do
  [ "$(outdir_of $p)" = "$RUN" ] && ALIVE=$((ALIVE+1))
done
# newest .log by mtime: the live run writes continuously, so it is necessarily the newest
LOG=$(ls -t /home/jupyter-chenxi/runs/*.log 2>/dev/null | head -1)
EV=$(nice -n 19 python /home/jupyter-chenxi/prune_work/dump_epochs.py "$RUN" 2>/dev/null \
      | grep -oE 'validation\) events: [0-9]+' | grep -oE '[0-9]+$')
BEST=$(ls "$RUN"/lightning_logs/version_*/checkpoints/best-* 2>/dev/null | wc -l)
PROG=$(tail -c 800 "$LOG" | tr '\r' '\n' | grep -oE 'Epoch [0-9]+: +[0-9]+%[^,]*, +[0-9.]+it/s' | tail -1)
echo "VALEV=${EV:-0} BEST=$BEST ALIVE=$ALIVE RUN=$(basename $RUN) LOG=$(basename $LOG) | $PROG"
