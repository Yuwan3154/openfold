#!/bin/bash
#SBATCH -J esmc_run
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_run-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_run-%j.err
# Protpardelle-1c cc89 partial diffusion of the ESMC-fill inputs: same protocol as compF_run.sbatch (tiered, rewinds 375 300 250 200).
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
for ARM in T0.7_p1.0 T0.5_p1.0; do
  python run_indel_pd.py --inputs-dir $D/inputs_esmc_$ARM --out-root $D/sweep_esmc_$ARM --model cc89 --kinds indel --schedule tiered \
    --rewinds 375 300 250 200 --chains 7du7_A 6kyf_A 5zo3_A --draws 0 1 2 3 4 5 6 7 || rc=$?
done
echo "rc_final=$rc"
exit $rc
