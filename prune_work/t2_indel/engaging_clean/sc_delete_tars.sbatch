#!/bin/bash
#SBATCH -J eclean_sc_rm
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/cou/logs/eclean_sc_rm-%j.out
# User 10-07: "Yes delete the tars". Literal allowlist: the three staging tars only; the A6000 copy was verified (RAW §37).
X=/home/gridsan/cou/e_clean/xfer
for f in runD_logs.tar runD_t4pool.tar misc.tar; do
  [ -f "$X/$f" ] && [ ! -L "$X/$f" ] && { echo "DELETE $f $(stat -c %s $X/$f)"; rm -f -- "$X/$f"; } || echo "ABSENT $f"
done
ls -la $X
echo rc_final=0
