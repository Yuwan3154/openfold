#!/bin/bash
#SBATCH -J t2indel_comp_run
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_comp_run-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_comp_run-%A_%a.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="6x61_B 6j22_A 5zo3_A 6qla_A 6ve7_W 7aah_A 6rur_V 6z1p_Bf 6kyf_A 6yw5_QQ 6f43_A 6iu3_A"
rc=0
python run_indel_pd.py --inputs-dir $D/inputs_comp --out-root $D/sweep_comp --model cc89 --kinds indel --schedule tiered \
   --rewinds 300 250 --chains $CH --draws 0 1 2 3 4 5 6 7 --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
echo "rc_final=$rc"
exit $rc
