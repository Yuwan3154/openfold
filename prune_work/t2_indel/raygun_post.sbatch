#!/bin/bash
#SBATCH -J t2indel_raygun_post
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_raygun_post-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_raygun_post-%j.err
# CPU: audit + score + diagnose the Raygun arm (indel items only; chain by chain, resumable). Runs after the GPU array.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
IN=$D/inputs_raygun_n0.5
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
source activate protpardelle
python audit_outputs.py --inputs-dir $IN --out-root $D/sweep_raygun_n0.5 --models cc89 cc91 --kinds indel || rc=$?
echo "rc_audit=$rc"
python score_indel.py --inputs-dir $IN --out-root $D/sweep_raygun_n0.5 --score-dir $D/scores_raygun_n0.5 --models cc89 cc91 --workers 32 || rc=$?
echo "rc_score=$rc"
source activate protebm
python diagnose_indel.py --inputs-dir $IN --out-root $D/sweep_raygun_n0.5 --models cc89 cc91 --kinds indel --diag-dir $D/diag_raygun_n0.5 --workers 32 || rc=$?
echo "rc_final=$rc"
exit $rc
