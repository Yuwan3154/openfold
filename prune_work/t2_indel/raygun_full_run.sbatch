#!/bin/bash
#SBATCH -J t2indel_raygunfull
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_raygunfull-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_raygunfull-%A_%a.err
# Raygun arm v2 (full-atom inputs from cg2all), cc89 only (user 10-05), indel items only.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
python run_indel_pd.py --inputs-dir $D/inputs_raygun_n0.5_full --out-root $D/sweep_raygun_full --model cc89 --kinds indel \
    --schedule tiered --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
echo "rc_final=$rc"
exit $rc
