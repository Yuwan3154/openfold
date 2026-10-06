#!/bin/bash
#SBATCH -J t2indel_rg2_post
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rg2_post-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rg2_post-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
IN=$D/inputs_rg2_full
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
source activate protpardelle
python audit_outputs.py --inputs-dir $IN --out-root $D/sweep_rg2 --models cc89 --kinds indel || rc=$?
python score_indel.py --inputs-dir $IN --out-root $D/sweep_rg2 --score-dir $D/scores_rg2 --models cc89 --workers 32 || rc=$?
source activate protebm
python diagnose_indel.py --inputs-dir $IN --out-root $D/sweep_rg2 --models cc89 --kinds indel --diag-dir $D/diag_rg2 --workers 32 || rc=$?
echo "rc_final=$rc"
exit $rc
