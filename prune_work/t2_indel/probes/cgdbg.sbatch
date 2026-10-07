#!/bin/bash
#SBATCH -J cgdbg
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/home/gridsan/cou/logs/cgdbg-%j.out
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate pottsmpnn
D=/home/gridsan/cou/t2_indel_data
mkdir -p $D/cgdbg
CUDA_VISIBLE_DEVICES= convert_cg2all -p $D/inputs_comp_bb/5zo3_A/d03.pdb -o $D/cgdbg/x.pdb --cg BackboneModel --device cpu 2>&1 | tail -12
head -3 $D/inputs_comp_bb/5zo3_A/d03.pdb; head -3 $D/inputs_raygun_n0.5/5zo3_A/d03.pdb
