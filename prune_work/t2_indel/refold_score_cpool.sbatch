#!/bin/bash
#SBATCH -J t2_rf_score_cpool
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_rf_score_cpool-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_rf_score_cpool-%j.err
# Score the ESMFold2 refold of the NO-INDEL control pool (job 5851437): paired TM of the prediction to the control template, per design.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_score.py --pool-root $D/pool_ctl --in-dir $D/refold_cpool --method esm --out $D/refold_cpool_scores.csv
rc=$?
echo rc=$rc
exit $rc
