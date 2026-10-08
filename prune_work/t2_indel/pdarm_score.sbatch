#!/bin/bash
#SBATCH -J pdarm_score
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/pdarm_score-%j.out
#SBATCH --error=/home/gridsan/cou/logs/pdarm_score-%j.err
# Score the arm's ESMFold2 refold (paired TM per prediction). usage: sbatch --export=ALL,TAG=cc94gly pdarm_score.sbatch
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_score.py --pool-root $D/pool_$TAG --in-dir $D/refold_$TAG --method esm --out $D/refold_${TAG}_scores.csv
rc=$?
echo "rc_final=$rc"
exit $rc
