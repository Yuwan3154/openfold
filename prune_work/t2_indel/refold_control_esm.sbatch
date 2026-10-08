#!/bin/bash
#SBATCH -J t2indel_rf_ctl_esm
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_ctl_esm-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_ctl_esm-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python refold_native_control.py --method esm --inputs-dir $D/inputs --chains 7du7_A 6kyf_A 5zo3_A 6ve7_W 6el2_A 6x61_B --out $D/refold_ctl_esm
rc=$?
echo "rc_final=$rc"
exit $rc
