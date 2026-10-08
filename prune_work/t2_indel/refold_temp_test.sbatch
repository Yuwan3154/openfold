#!/bin/bash
#SBATCH -J t2indel_rf_temp
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_temp-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_temp-%j.err
# Lower-temperature test on the SAME 6 template backbones; T grid {0.05, 0.01} is MY choice (ledger). ESMFold2 folds.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
source activate protpardelle
python mpnn_temp_make.py --pool-root $D/pool_t250 --select 7du7_A:0 7du7_A:64 6kyf_A:0 6kyf_A:64 5zo3_A:0 5zo3_A:64 \
   --temps 0.05 0.01 --mpnn-dir /home/gridsan/cou/ProteinMPNN --work-dir $D/temp_work --out $D/temp_seqs.json || { echo rc_final=1; exit 1; }
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python refold_seqs_esm.py --pool-root $D/pool_t250 --seqs-json $D/temp_seqs.json --out $D/refold_temp
rc=$?
echo "rc_final=$rc"
exit $rc
