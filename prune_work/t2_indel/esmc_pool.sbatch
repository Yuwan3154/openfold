#!/bin/bash
#SBATCH -J esmc_pool
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_pool-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_pool-%j.err
# Score (sequence-independent USalign TM to native), build the pilot pool at rewind 250 and design 32 ProteinMPNN sequences per backbone
# with the pool's settings (mpnn_design_pool.py defaults).
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
rc=0
for ARM in T0.7_p1.0 T0.5_p1.0; do
  python score_indel.py --inputs-dir $D/inputs_esmc_$ARM --out-root $D/sweep_esmc_$ARM --score-dir $D/esmc_${ARM}_scores --models cc89 --chains $CH --workers 8 || rc=$?
done
python build_indel_pool.py --out-root $D/pool_esmc --rewind 250 --chains $CH --draws 0 1 2 3 4 5 6 7 \
  --arm esmc07 $D/sweep_esmc_T0.7_p1.0 $D/inputs_esmc_T0.7_p1.0 $D/esmc_T0.7_p1.0_scores \
  --arm esmc05 $D/sweep_esmc_T0.5_p1.0 $D/inputs_esmc_T0.5_p1.0 $D/esmc_T0.5_p1.0_scores || rc=$?
python mpnn_design_pool.py --pool-root $D/pool_esmc --mpnn-dir /home/gridsan/cou/ProteinMPNN --work-dir $D/pool_esmc_work || rc=$?
echo "rc_final=$rc"
exit $rc
