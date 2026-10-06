#!/bin/bash
#SBATCH -J t2indel_pool_smoke
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_pool_smoke-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_pool_smoke-%j.err
# SMOKE (one chain): stage A (assemble) + stage B (ProteinMPNN redesign) before the full build
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
python build_indel_pool.py --out-root $D/pool_smoke --rewind 250 --chains 7du7_A \
  --arm comp $D/sweep_compF $D/inputs_compF $D/compF_scores --arm raygun1dir $D/sweep_rg3 $D/inputs_rg3_full $D/scores_rg3 || rc=$?
python mpnn_design_pool.py --pool-root $D/pool_smoke --mpnn-dir /home/gridsan/cou/ProteinMPNN --work-dir $D/pool_smoke_work || rc=$?
echo "rc_final=$rc"
exit $rc
