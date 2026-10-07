#!/bin/bash
#SBATCH -J t2indel_rf_cpool
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_cpool-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_cpool-%j.err
# NO-INDEL CONTROL POOL refold, ESMFold2 (same settings as refold_pilot_esm): partial diffusion at t* 250 WITHOUT indels,
# then the same MPNN designs + refold. Items 2 and 6 of each chain are the t* 250 rows of pool_ctl/index.csv (draws 0 and 1).
# Separates "partial diffusion at t* 250 hurts designability" from "the indel edits hurt it".
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
SEL=""
for c in 7du7_A 6kyf_A 5zo3_A 6el2_A 6x61_B; do SEL="$SEL $c:2 $c:6"; done
python refold_esmfold2.py --pool-root $D/pool_ctl --select $SEL --out $D/refold_cpool
rc=$?
echo "rc_final=$rc"
exit $rc
