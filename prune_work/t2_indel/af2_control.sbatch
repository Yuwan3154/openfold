#!/bin/bash
#SBATCH -J t2_af2_ctl
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_af2_ctl-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_af2_ctl-%j.err
# KNOWN-GOOD control for the AF2 completion arm: native template + native query, nothing removed, then the CA RMSD of the prediction to the
# native (with a native template AF2 should reproduce the structure; if it does not, the template channel is ineffective in this setup).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate colabdesign
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python af2_complete.py --inputs-dir $D/inputs --chains 7du7_A 6kyf_A 5zo3_A 6el2_A 6x61_B --draws 0 --control --out $D/af2fix_ctl
rc=$?
echo rc=$rc
exit $rc
