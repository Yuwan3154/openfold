#!/bin/bash
#SBATCH -J t2indel_cg2all
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_cg2all-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_cg2all-%j.err
# Option (b), user 10-05: build full-atom Raygun-arm inputs with cg2all BackboneModel (backbone + Raygun residue
# types -> all atoms). Weights were cached by a download-partition run (compute nodes have no internet).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate pottsmpnn
export OMP_NUM_THREADS=1
D=/home/gridsan/cou/t2_indel_data
SRC=$D/inputs_raygun_n0.5
DST=$D/inputs_raygun_n0.5_full
mkdir -p $DST
for k in $(ls $SRC); do mkdir -p $DST/$k; cp $SRC/$k/native.pdb $SRC/$k/plans.json $DST/$k/; done
rc=0
ls $SRC/*/d??.pdb | xargs -P 46 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$DST'/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$DST'/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/cg2all_fail.txt || rc=$?
echo "n_in=$(ls $SRC/*/d??.pdb | wc -l) n_out=$(ls $DST/*/d??.pdb | wc -l) n_fail=$(grep -c FAIL $D/cg2all_fail.txt)"
[ "$(ls $SRC/*/d??.pdb | wc -l)" = "$(find $DST -name 'd??.pdb' -size +0 | wc -l)" ] || rc=1
echo "rc_final=$rc"
exit $rc
