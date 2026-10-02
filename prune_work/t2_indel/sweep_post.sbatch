#!/bin/bash
#SBATCH -J t2indel_post
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_post-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_post-%j.err
# CPU: audit the Gly sweep, verify the Raygun-arm inputs through the real reader, score the sweep (+ the edited inputs)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
python audit_outputs.py --inputs-dir $D/inputs --out-root $D/sweep_gly --models cc89 cc91 || rc=$?
echo "rc_audit=$rc"
python verify_raygun_inputs.py --inputs-dir $D/inputs_raygun_n0.5 || rc=$?
echo "rc_verify_raygun=$rc"
python score_indel.py --inputs-dir $D/inputs --score-dir $D/scores_gly --edited-only --workers 32 || rc=$?
echo "rc_score_edited=$rc"
python score_indel.py --inputs-dir $D/inputs --out-root $D/sweep_gly --score-dir $D/scores_gly --models cc89 cc91 --workers 32 || rc=$?
echo "rc_final=$rc"
exit $rc
