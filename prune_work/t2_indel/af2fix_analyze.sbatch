#!/bin/bash
#SBATCH -J t2_af2fix_an
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_af2fix_an-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_af2fix_an-%j.err
# Similarity + validity of the AF2-completed templates vs the partial-diffusion templates (pilot: 3 chains x 8 draws x 3 arms).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protebm
D=/home/gridsan/cou/t2_indel_data
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
python af2fix_analyze.py --inputs-dir $D/inputs --af2-root $D --arms gly comp raygun --pool-root $D/pool_t250 \
  --chains 7du7_A 6kyf_A 5zo3_A --out-prefix $D/af2fix_pilot
rc=$?
echo rc=$rc
exit $rc
