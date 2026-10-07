#!/bin/bash
#SBATCH -J esmc_pilot
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_pilot-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_pilot-%j.err
# ESMC-300M one-shot fill pilot (user 10-07): T x top-p grid on 3 chains x 8 draws, CPU.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1
D=$HOME/t2_indel_data
python $HOME/of_t2indel/prune_work/t2_indel/esmc_fill_pilot.py --inputs-dir $D/inputs --out-dir $D/esmc_pilot \
  --chains 7du7_A 6kyf_A 5zo3_A --draws 0 1 2 3 4 5 6 7
rc=$?
echo rc=$rc
exit $rc
