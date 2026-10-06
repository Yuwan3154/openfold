#!/bin/bash
#SBATCH -J t2indel_compF_post
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_compF_post-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_compF_post-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
CH=$(ls /home/gridsan/cou/t2_indel_data/inputs | tr "\n" " ")
rc=0
source activate protpardelle
python score_indel.py --inputs-dir $D/inputs_compF --out-root $D/sweep_compF --score-dir $D/compF_scores --models cc89 --chains $CH --workers 32 || rc=$?
source activate protebm
python diagnose_indel.py --inputs-dir $D/inputs_compF --out-root $D/sweep_compF --models cc89 --kinds indel --diag-dir $D/compF_diag --chains $CH --workers 32 || rc=$?
echo "rc_final=$rc"
exit $rc
