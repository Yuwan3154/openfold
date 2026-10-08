#!/bin/bash
#SBATCH -J t2indel_af3env
#SBATCH -p download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_af3env-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_af3env-%j.err
# ONE download-partition job (the partition allows one submit per user): (1) AF3 env per af3_env_build.sbatch, then
# (2) the ESMFold2 weights pre-fetch (esmfold2_weights.sh).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
rc=0
bash /home/gridsan/cou/of_t2indel/prune_work/t2_indel/af3_env_build.sbatch_body.sh || rc=$?
echo "rc_af3=$rc"
bash /home/gridsan/cou/of_t2indel/prune_work/t2_indel/esmfold2_weights.sh || rc=$?
echo "rc_final=$rc"
exit $rc
