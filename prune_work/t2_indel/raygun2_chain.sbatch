#!/bin/bash
#SBATCH -J t2indel_rg2_prep
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rg2_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rg2_prep-%j.err
# Raygun two-step arm: sequences (raygun env) -> alignment-defined backbone edits -> cg2all full atoms (pottsmpnn env)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
export OMP_NUM_THREADS=24
D=/home/gridsan/cou/t2_indel_data
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
source activate raygun
python raygun_generate2.py --inputs-dir $D/inputs --out $D/raygun_seqs2_n0.5.json --noise 0.5 || rc=$?
python make_raygun_inputs.py --inputs-dir $D/inputs --seqs $D/raygun_seqs2_n0.5.json --out-dir $D/inputs_rg2_bb || rc=$?
source activate pottsmpnn
export OMP_NUM_THREADS=1
SRC=$D/inputs_rg2_bb
DST=$D/inputs_rg2_full
mkdir -p $DST
for k in $(ls $SRC); do mkdir -p $DST/$k; cp $SRC/$k/native.pdb $SRC/$k/plans.json $DST/$k/; done
ls $SRC/*/d??.pdb | xargs -P 46 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$DST'/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$DST'/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/rg2_cg_fail.txt || rc=$?
n_in=$(ls $SRC/*/d??.pdb | wc -l); n_out=$(find $DST -name 'd??.pdb' -size +0 | wc -l)
echo "n_in=$n_in n_out=$n_out"
[ "$n_in" = "$n_out" ] || rc=1
echo "rc_final=$rc"
exit $rc
