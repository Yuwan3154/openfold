#!/bin/bash
#SBATCH -J t2_af2fix
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_af2fix-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_af2fix-%j.err
# AF2 completion pilot (user 10-07): survivor-only native template + each arm's post-edit sequence as query. 3 chains x 8 draws x 3 arms.
# usage: sbatch --export=ALL,ARM=gly|comp|raygun af2_complete.sbatch   (single-token values)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate colabdesign
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
case $ARM in gly) IN=inputs;; comp) IN=inputs_compF;; raygun) IN=inputs_rg3_full;; *) echo "bad ARM=$ARM"; exit 2;; esac
python af2_complete.py --inputs-dir $D/$IN --chains 7du7_A 6kyf_A 5zo3_A --draws 0 1 2 3 4 5 6 7 --out $D/af2fix_$ARM
rc=$?
echo rc=$rc
exit $rc
