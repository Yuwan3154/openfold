#!/bin/bash
#SBATCH -J mild_prep
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/mild_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/mild_prep-%j.err
# usage: sbatch --export=ALL,PFX=mid,FRAC_LO=0.10,FRAC_HI=0.10 mild_prep.sbatch (no env vars = the mild 1-5 % test)
# MILD-edit test inputs for the 3 pilot chains, 8 draws, insertion/deletion fractions U(0.01, 0.05) of L (my proposal, provisional), all three arms:
#   Gly arm      -> inputs_${P}
#   composition  -> inputs_${P}_comp  (types drawn from the chain's own composition, cg2all full atoms)
#   Raygun       -> inputs_${P}_rg    (raygun_generate3 one-direction sequences, alignment-defined edits, cg2all full atoms)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
D=/home/gridsan/cou/t2_indel_data
P=${PFX:-mild}; FRAC_LO=${FRAC_LO:-0.01}; FRAC_HI=${FRAC_HI:-0.05}   # defaults = the mild test; the mid rung uses PFX=mid FRAC_LO=0.10 FRAC_HI=0.10
CH="7du7_A 6kyf_A 5zo3_A"
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
source activate pottsmpnn
export OMP_NUM_THREADS=1
python make_mild_inputs.py --inputs-dir $D/inputs --out-dir $D/inputs_${P} --chains $CH --n-draws 8 --frac-lo $FRAC_LO --frac-hi $FRAC_HI || rc=$?
python make_comp_inputs.py --stage bb --inputs-dir $D/inputs_${P} --out-dir $D/inputs_${P}_comp_bb --chains $CH --draws 0 1 2 3 4 5 6 7 || rc=$?
for k in $CH; do mkdir -p $D/inputs_${P}_comp_cg/$k; done
ls $D/inputs_${P}_comp_bb/*/d??.pdb | xargs -P 24 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_'${P}'_comp_cg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_'${P}'_comp_cg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/${P}_comp_cg_fail.txt || rc=$?
n_in=$(ls $D/inputs_${P}_comp_bb/*/d??.pdb | wc -l); n_cg=$(find $D/inputs_${P}_comp_cg -name 'd??.pdb' -size +0 | wc -l)
echo "comp n_in=$n_in n_cg=$n_cg fails=$(wc -l < $D/${P}_comp_cg_fail.txt)"
[ "$n_in" = "$n_cg" ] || rc=1
python make_comp_inputs.py --stage merge --inputs-dir $D/inputs_${P} --out-dir $D/inputs_${P}_comp --cg-dir $D/inputs_${P}_comp_cg --chains $CH --draws 0 1 2 3 4 5 6 7 || rc=$?
export OMP_NUM_THREADS=24
source activate raygun
python raygun_generate3.py --inputs-dir $D/inputs_${P} --out $D/raygun_seqs3_${P}.json --noise 0.5 --chains $CH || rc=$?
python make_raygun_inputs.py --inputs-dir $D/inputs_${P} --seqs $D/raygun_seqs3_${P}.json --out-dir $D/inputs_${P}_rg_bb || rc=$?
source activate pottsmpnn
export OMP_NUM_THREADS=1
for k in $CH; do mkdir -p $D/inputs_${P}_rg/$k; cp $D/inputs_${P}_rg_bb/$k/native.pdb $D/inputs_${P}_rg_bb/$k/plans.json $D/inputs_${P}_rg/$k/; done
ls $D/inputs_${P}_rg_bb/*/d??.pdb | xargs -P 24 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_'${P}'_rg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_'${P}'_rg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/${P}_rg_cg_fail.txt || rc=$?
n_in=$(ls $D/inputs_${P}_rg_bb/*/d??.pdb | wc -l); n_out=$(find $D/inputs_${P}_rg -name 'd??.pdb' -size +0 | wc -l)
echo "raygun n_in=$n_in n_out=$n_out fails=$(wc -l < $D/${P}_rg_cg_fail.txt)"
[ "$n_in" = "$n_out" ] || rc=1
echo "rc_final=$rc"
exit $rc
