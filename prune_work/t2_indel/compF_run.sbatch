#!/bin/bash
#SBATCH -J t2indel_compF_run
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_compF_run-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_compF_run-%A_%a.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH=$(ls /home/gridsan/cou/t2_indel_data/inputs | tr "\n" " ")
rc=0
python run_indel_pd.py --inputs-dir $D/inputs_compF --out-root $D/sweep_compF --model cc89 --kinds indel --schedule tiered \
   --rewinds 375 300 250 200 --chains $CH --draws $(seq 0 63) --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
echo "rc_final=$rc"
exit $rc
