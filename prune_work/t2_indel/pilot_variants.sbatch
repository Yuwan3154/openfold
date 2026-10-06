#!/bin/bash
#SBATCH -J t2indel_pilot
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_pilot-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_pilot-%A_%a.err
# Refinement pilot (autonomous window 10-06): 12 chains x 8 draws x rungs {300,250}; variants are CHOSEN exploratory
# values (CHOICES LEDGER in RAW Phase O): rewinds come from the measured sigma table, 200 = the shipped s_churn.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="6x61_B 6j22_A 5zo3_A 6qla_A 6ve7_W 7aah_A 6rur_V 6z1p_Bf 6kyf_A 6yw5_QQ 6f43_A 6iu3_A"
rc=0
for v in "refine:r=50:n=1" "refine:r=100:n=1" "refine:r=100:n=3" "refine:r=150:n=1" "refine:r=150:n=3" "refine:r=200:n=1" "fresh:churn=200:recur=1" "fresh:churn=0:recur=2"; do
  python run_refine.py --inputs-dir $D/inputs --out-root $D/pilot --variant "$v" --base-root $D/sweep_gly --rungs 300 250 \
      --chains $CH --draws 0 1 2 3 4 5 6 7 --tmp-dir $D/pilot_tmp --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
  echo "rc_${v}=$rc"
done
echo "rc_final=$rc"
exit $rc
