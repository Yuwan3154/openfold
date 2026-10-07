#!/bin/bash
#SBATCH -J t2indel_rf_local
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_local-%j.out
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_local.py --pool-root $D/pool_t250 --in-dir $D/refold_esm --out $D/refold_local.csv
echo "rc_final=$?"
