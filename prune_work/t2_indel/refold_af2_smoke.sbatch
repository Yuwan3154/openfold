#!/bin/bash
#SBATCH -J t2indel_af2_smoke
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_af2_smoke-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_af2_smoke-%j.err
# SHAKEDOWN of the AF2 refold path on ONE template x 3 sequences; ColabDesign/jax on V100 has NEVER run on SuperCloud before
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate colabdesign
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python -X faulthandler refold_af2.py --pool-root $D/pool_smoke --select 7du7_A:0 --max-seqs 2 --out $D/af2_smoke --tmp-dir $D/af2_smoke_tmp
rc=$?
echo "rc_final=$rc"
exit $rc
