#!/bin/bash
#SBATCH -J t2indel_rf_af2
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_af2-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_af2-%j.err
# Refold PILOT, AF2: 6 backbones (3 chains x {comp, raygun1dir}, draw 0) x (generation seq + 32 designs). No CUDA module (segfault).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate colabdesign
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_af2.py --pool-root $D/pool_t250 --select 7du7_A:0 7du7_A:64 6kyf_A:0 6kyf_A:64 5zo3_A:0 5zo3_A:64 \
   --out $D/refold_af2 --tmp-dir $D/refold_af2_tmp
rc=$?
echo "rc_final=$rc"
exit $rc
