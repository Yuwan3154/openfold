#!/bin/bash
#SBATCH -J t2indel_comp_post
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_comp_post-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_comp_post-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
CH="6x61_B 6j22_A 5zo3_A 6qla_A 6ve7_W 7aah_A 6rur_V 6z1p_Bf 6kyf_A 6yw5_QQ 6f43_A 6iu3_A"
rc=0
source activate protpardelle
python score_indel.py --inputs-dir $D/inputs_comp --out-root $D/sweep_comp --score-dir $D/comp_scores --models cc89 --chains $CH --workers 32 || rc=$?
source activate protebm
python diagnose_indel.py --inputs-dir $D/inputs_comp --out-root $D/sweep_comp --models cc89 --kinds indel --diag-dir $D/comp_diag --chains $CH --workers 32 || rc=$?
echo "rc_final=$rc"
exit $rc
