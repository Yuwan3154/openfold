#!/bin/bash
#SBATCH -J t2indel_rgfull_verify
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rgfull_verify-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rgfull_verify-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python verify_raygun_inputs.py --inputs-dir $D/inputs_raygun_n0.5_full --ca-tol 0.5
rc=$?
echo "rc_final=$rc"
exit $rc
