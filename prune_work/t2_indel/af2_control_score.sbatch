#!/bin/bash
#SBATCH -J t2_af2_ctl_sc
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_af2_ctl_sc-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_af2_ctl_sc-%j.err
# Score the AF2 known-good control against the native (RMSD + TM) and the earlier no-template AF2 native refold.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protebm
D=/home/gridsan/cou/t2_indel_data
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
python af2_control_score.py --inputs-dir $D/inputs --ctl-dir $D/af2fix_ctl --notemplate-csv $D/refold_ctl_scores_af2.csv --out $D/af2fix_ctl_scores.csv
rc=$?
echo rc=$rc
exit $rc
