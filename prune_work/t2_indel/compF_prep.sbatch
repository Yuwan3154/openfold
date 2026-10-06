#!/bin/bash
#SBATCH -J t2indel_compF_prep
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_compF_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_compF_prep-%j.err
# composition-matched real-type arm: backbone-only inputs -> cg2all full atoms -> merge (survivors keep native atoms)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate pottsmpnn
export OMP_NUM_THREADS=1
D=/home/gridsan/cou/t2_indel_data
CH=$(ls /home/gridsan/cou/t2_indel_data/inputs | tr "\n" " ")
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
python make_comp_inputs.py --stage bb --inputs-dir $D/inputs --out-dir $D/inputs_compF_bb --chains $CH --draws $(seq 0 63) || rc=$?
for k in $CH; do mkdir -p $D/inputs_compF_cg/$k; done
ls $D/inputs_compF_bb/*/d??.pdb | xargs -P 46 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_compF_cg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_compF_cg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/compF_cg_fail.txt || rc=$?
n_in=$(ls $D/inputs_compF_bb/*/d??.pdb | wc -l); n_cg=$(find $D/inputs_compF_cg -name 'd??.pdb' -size +0 | wc -l)
echo "n_in=$n_in n_cg=$n_cg"
[ "$n_in" = "$n_cg" ] || rc=1
python make_comp_inputs.py --stage merge --inputs-dir $D/inputs --out-dir $D/inputs_compF --cg-dir $D/inputs_compF_cg --chains $CH --draws $(seq 0 63) || rc=$?
echo "rc_final=$rc"
exit $rc
