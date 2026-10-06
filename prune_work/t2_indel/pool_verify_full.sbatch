#!/bin/bash
#SBATCH -J t2indel_pool_V
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_pool_V-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_pool_V-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
ARMS="--arm comp $D/sweep_compF $D/inputs_compF $D/compF_scores --arm raygun1dir $D/sweep_rg3 $D/inputs_rg3_full $D/scores_rg3"
python verify_pool.py --pool-root $D/pool_t250 --rewind 250 $ARMS --design
rc=$?
echo "rc_final=$rc"
exit $rc
