#!/bin/bash
set -u
R=/home/jupyter-chenxi/runs
echo "=== position at stop ==="
tail -c 400 $R/runC_launch2.log | tr '\r' '\n' | tail -1
# ⛔ Guard on the OUTPUT DIR positional (token 6), never a substring of the whole cmdline: Run C's
# --resume_from_ckpt legitimately names a runB path, which broke two earlier guards.
echo "=== ancestry check ==="
bad=0; n=0
for p in $(pgrep -f "[t]rain_openfold"); do
  out=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | sed -n '6p')
  [ -z "$out" ] && continue
  n=$((n+1))
  case "$out" in
    */runs/runC_replica_exchange) ;;
    *) echo "  !! $p output dir '$out' is NOT runC_replica_exchange -- ABORT"; bad=1;;
  esac
done
[ "$bad" -eq 1 ] && { echo ABORT; exit 1; }
[ "$n" -eq 0 ] && { echo "nothing running"; }
echo "  all $n procs are runC_replica_exchange"
ROOT=$(pgrep -f "[t]rain_openfold" | head -1)
if [ -n "$ROOT" ]; then
  kill -TERM $ROOT $(pgrep -P $ROOT | tr '\n' ' ') 2>&1
  for i in $(seq 1 30); do
    c=$(pgrep -cf "[t]rain_openfold"); [ "${c:-0}" -eq 0 ] && { echo "  stopped after ${i}0s"; break; }
    sleep 10
  done
fi
sleep 40
echo "procs: $(pgrep -cf '[t]rain_openfold')"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "(empty = GPUs released)"
echo "=== code state ==="
cd /home/jupyter-chenxi/openfold-esmfold2-recycling
echo "HEAD=$(git rev-parse --short HEAD)"; grep -c "_top_k" train_openfold.py
echo "=== launch v2 ==="
cd /home/jupyter-chenxi
setsid nohup bash prune_work/run_C_v2.sh > $R/runC_v2_launch1.log 2>&1 < /dev/null &
echo "launched, shell pid $!"
