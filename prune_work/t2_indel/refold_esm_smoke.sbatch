#!/bin/bash
#SBATCH -J t2indel_esm_smoke
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_esm_smoke-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_esm_smoke-%j.err
# SHAKEDOWN: ESMFold2 on ONE template x 3 sequences on a V100 (no flash-attn 2 on Volta; bf16 ESMC emulated). HF offline: weights pre-fetched.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python -X faulthandler refold_esmfold2.py --pool-root $D/pool_t250 --select 7du7_A:0 --max-seqs 2 --out $D/esm_smoke
rc=$?
echo "rc_final=$rc"
exit $rc
