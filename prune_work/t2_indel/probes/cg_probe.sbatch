#!/bin/bash
#SBATCH -J cg2all_probe
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/gridsan/cou/logs/cg2all_probe-%j.out
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate pottsmpnn
which convert_cg2all
python -c "from cg2all.lib.libconfig import MODEL_HOME; import os; print(MODEL_HOME); print(os.listdir(MODEL_HOME))"
convert_cg2all -h 2>&1 | head -30
D=/home/gridsan/cou/t2_indel_data
mkdir -p $D/cg_test
CUDA_VISIBLE_DEVICES= convert_cg2all -p $D/inputs_raygun_n0.5/7du7_A/d00.pdb -o $D/cg_test/d00_full.pdb --cg BackboneModel 2>&1 | tail -8
ls -la $D/cg_test
grep -c "^ATOM" $D/cg_test/d00_full.pdb $D/inputs_raygun_n0.5/7du7_A/d00.pdb
