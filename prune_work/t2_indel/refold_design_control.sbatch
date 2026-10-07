#!/bin/bash
#SBATCH -J t2indel_rf_dctl
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_dctl-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_dctl-%j.err
# DESIGN CONTROL: MPNN on NATIVE backbones (same settings as the pool) -> ESMFold2 refold. Chains = the pilot's three + two others.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A 6el2_A 6x61_B"
source activate protpardelle
bash design_control_mpnn.sh $D/inputs $D/dctl_work $CH || { echo rc_final=1; exit 1; }
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python refold_design_control.py --work-dir $D/dctl_work --inputs-dir $D/inputs --chains $CH --out $D/refold_dctl_esm
echo "rc_final=$?"
