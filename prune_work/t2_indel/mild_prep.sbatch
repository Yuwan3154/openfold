#!/bin/bash
#SBATCH -J mild_prep
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/mild_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/mild_prep-%j.err
# MILD-edit test inputs for the 3 pilot chains, 8 draws, insertion/deletion fractions U(0.01, 0.05) of L (my proposal, provisional), all three arms:
#   Gly arm      -> inputs_mild
#   composition  -> inputs_mild_comp  (types drawn from the chain's own composition, cg2all full atoms)
#   Raygun       -> inputs_mild_rg    (raygun_generate3 one-direction sequences, alignment-defined edits, cg2all full atoms)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
source activate pottsmpnn
export OMP_NUM_THREADS=1
python make_mild_inputs.py --inputs-dir $D/inputs --out-dir $D/inputs_mild --chains $CH --n-draws 8 --frac-lo 0.01 --frac-hi 0.05 || rc=$?
python make_comp_inputs.py --stage bb --inputs-dir $D/inputs_mild --out-dir $D/inputs_mild_comp_bb --chains $CH --draws 0 1 2 3 4 5 6 7 || rc=$?
for k in $CH; do mkdir -p $D/inputs_mild_comp_cg/$k; done
ls $D/inputs_mild_comp_bb/*/d??.pdb | xargs -P 24 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_mild_comp_cg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_mild_comp_cg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/mild_comp_cg_fail.txt || rc=$?
n_in=$(ls $D/inputs_mild_comp_bb/*/d??.pdb | wc -l); n_cg=$(find $D/inputs_mild_comp_cg -name 'd??.pdb' -size +0 | wc -l)
echo "comp n_in=$n_in n_cg=$n_cg fails=$(wc -l < $D/mild_comp_cg_fail.txt)"
[ "$n_in" = "$n_cg" ] || rc=1
python make_comp_inputs.py --stage merge --inputs-dir $D/inputs_mild --out-dir $D/inputs_mild_comp --cg-dir $D/inputs_mild_comp_cg --chains $CH --draws 0 1 2 3 4 5 6 7 || rc=$?
export OMP_NUM_THREADS=24
source activate raygun
python raygun_generate3.py --inputs-dir $D/inputs_mild --out $D/raygun_seqs3_mild.json --noise 0.5 --chains $CH || rc=$?
python make_raygun_inputs.py --inputs-dir $D/inputs_mild --seqs $D/raygun_seqs3_mild.json --out-dir $D/inputs_mild_rg_bb || rc=$?
source activate pottsmpnn
export OMP_NUM_THREADS=1
for k in $CH; do mkdir -p $D/inputs_mild_rg/$k; cp $D/inputs_mild_rg_bb/$k/native.pdb $D/inputs_mild_rg_bb/$k/plans.json $D/inputs_mild_rg/$k/; done
ls $D/inputs_mild_rg_bb/*/d??.pdb | xargs -P 24 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_mild_rg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_mild_rg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/mild_rg_cg_fail.txt || rc=$?
n_in=$(ls $D/inputs_mild_rg_bb/*/d??.pdb | wc -l); n_out=$(find $D/inputs_mild_rg -name 'd??.pdb' -size +0 | wc -l)
echo "raygun n_in=$n_in n_out=$n_out fails=$(wc -l < $D/mild_rg_cg_fail.txt)"
[ "$n_in" = "$n_out" ] || rc=1
echo "rc_final=$rc"
exit $rc
