#!/bin/bash
#SBATCH -J t2indel_coil
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_coil-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_coil-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protebm
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python coil_diagnosis.py --inputs-dir $D/inputs --out-root $D/sweep_gly --model cc89 --out $D/coil_gly.csv
rc=$?
echo "rc_final=$rc"
exit $rc
