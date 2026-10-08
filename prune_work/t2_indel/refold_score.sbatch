#!/bin/bash
#SBATCH -J t2indel_rf_score
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_score-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_score-%j.err
# usage: METHOD=af2|esm sbatch --export=ALL,METHOD=af2 refold_score.sbatch   (single-token value, safe for --export)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_score.py --pool-root $D/pool_t250 --in-dir $D/refold_$METHOD --method $METHOD --out $D/refold_scores_$METHOD.csv
rc=$?
echo "rc_final=$rc"
exit $rc
