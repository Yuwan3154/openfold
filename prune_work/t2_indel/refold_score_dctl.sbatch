#!/bin/bash
#SBATCH -J t2indel_rf_score_dctl
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_score_dctl-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_score_dctl-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_score.py --pool-root $D/pool_t250 --in-dir $D/refold_dctl_esm --method dctl_esm --control --out $D/refold_dctl_scores.csv
echo "rc_final=$?"
