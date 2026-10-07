#!/bin/bash
#SBATCH -J t2indel_rf_esm
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_esm-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_esm-%j.err
# Refold PILOT, ESMFold2 (esm 3.4.1.post1, fold() defaults, seed 0): the same 6 backbones x (generation seq + 32 designs).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_esmfold2.py --pool-root $D/pool_t250 --select 7du7_A:0 7du7_A:64 6kyf_A:0 6kyf_A:64 5zo3_A:0 5zo3_A:64 --out $D/refold_esm
rc=$?
echo "rc_final=$rc"
exit $rc
