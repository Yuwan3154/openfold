#!/bin/bash
#SBATCH -J t2indel_ref200
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_ref200-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_ref200-%A_%a.err
# User goal "correct bond lengths": the pilot only refined rungs 250/300 where bonds were already native-like.
# Here: the composition arm's t*=200 outputs (C-N median 1.58 A) refined with the same re-noise grid.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="6x61_B 6j22_A 5zo3_A 6qla_A 6ve7_W 7aah_A 6rur_V 6z1p_Bf 6kyf_A 6yw5_QQ 6f43_A 6iu3_A"
rc=0
for v in "refine:r=50:n=1" "refine:r=100:n=1" "refine:r=100:n=3" "refine:r=150:n=1" "refine:r=150:n=3" "refine:r=200:n=1"; do
  python run_refine.py --inputs-dir $D/inputs_compF --out-root $D/pilot200 --variant "$v" --base-root $D/sweep_compF --rungs 200 \
      --chains $CH --draws 0 1 2 3 4 5 6 7 --tmp-dir $D/pilot200_tmp --shard $SLURM_ARRAY_TASK_ID --num-shards 4 || rc=$?
  echo "rc_${v}=$rc"
done
echo "rc_final=$rc"
exit $rc
