#!/bin/bash
#SBATCH -J t2indel_pilot_post
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_pilot_post-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_pilot_post-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
CH="6x61_B 6j22_A 5zo3_A 6qla_A 6ve7_W 7aah_A 6rur_V 6z1p_Bf 6kyf_A 6yw5_QQ 6f43_A 6iu3_A"
TAGS=$(ls $D/pilot | tr '\n' ' ')
rc=0
source activate protpardelle
python score_indel.py --inputs-dir $D/inputs --out-root $D/pilot --score-dir $D/pilot_scores --models $TAGS --chains $CH --workers 32 || rc=$?
source activate protebm
python diagnose_indel.py --inputs-dir $D/inputs --out-root $D/pilot --models $TAGS --kinds indel --diag-dir $D/pilot_diag --chains $CH --workers 32 || rc=$?
python eval_variants.py --diag-root $D/pilot_diag --score-root $D/pilot_scores --panel ../t2_indel/panel_stats.csv --tags $TAGS \
   --baseline-diag $D/diag_gly_v2 --baseline-score $D/scores_gly --chains $CH > $D/pilot_eval.txt || rc=$?
cat $D/pilot_eval.txt
echo "rc_final=$rc"
exit $rc
