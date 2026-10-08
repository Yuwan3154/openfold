#!/bin/bash
#SBATCH -J esmc_score
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_score-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_score-%j.err
# Score the ESMC pilot refold (paired TM of each prediction to the template).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_score.py --pool-root $D/pool_esmc --in-dir $D/refold_esmc --method esm --out $D/refold_esmc_scores.csv
rc=$?
echo "rc_final=$rc"
exit $rc
