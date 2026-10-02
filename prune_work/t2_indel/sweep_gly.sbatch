#!/bin/bash
#SBATCH -J t2indel_sweep
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_sweep-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_sweep-%A_%a.err
# Step 3 (user 10-02 "Yes launch"): Gly arm. 4 array tasks x 1 GPU (modest on purpose: the volta cap node=4 is
# SHARED with another agent on this account). Chains are sharded alphabetically; each task runs cc89 then cc91.
# Resumable: a finished npz is skipped, so a re-submit continues. Skips are written to skipped.jsonl.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
for m in cc89 cc91; do
  python run_indel_pd.py --inputs-dir $D/inputs --out-root $D/sweep_gly --model $m --schedule tiered \
      --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
  echo "rc_${m}=$rc"
done
echo "rc_final=$rc"
exit $rc
