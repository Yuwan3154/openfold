#!/bin/bash
#SBATCH -J t2indel_raygun
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_raygun-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_raygun-%A_%a.err
# Raygun arm (option 1, noise 0.5): indel items only (controls are shared with the Gly arm). Same layout as sweep_gly.sbatch.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
for m in cc89 cc91; do
  python run_indel_pd.py --inputs-dir $D/inputs_raygun_n0.5 --out-root $D/sweep_raygun_n0.5 --model $m --kinds indel \
      --schedule tiered --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
  echo "rc_${m}=$rc"
done
echo "rc_final=$rc"
exit $rc
