#!/bin/bash
#SBATCH -J t2indel_step2
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_step2-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_step2-%j.err
# Step 2: ONE chain, ONE draw, cc89, tiered AND grouped, indel + control. Shakedown only.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
IN=/home/gridsan/cou/t2_indel_data/inputs
rc=0
python run_indel_pd.py --inputs-dir $IN --out-root /home/gridsan/cou/t2_indel_data/step2_tiered --model cc89 --chains 7du7_A --draws 0 --schedule tiered || rc=$?
echo "rc_tiered=$rc"
python run_indel_pd.py --inputs-dir $IN --out-root /home/gridsan/cou/t2_indel_data/step2_grouped --model cc89 --chains 7du7_A --draws 0 --schedule grouped || rc=$?
echo "rc_final=$rc"
exit $rc
