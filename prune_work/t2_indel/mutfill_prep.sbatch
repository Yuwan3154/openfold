#!/bin/bash
#SBATCH -J mutfill_prep
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/mutfill_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/mutfill_prep-%j.err
# ESMC fill-in + MUTATION arm inputs (user 10-08): 3 pilot chains x 8 draws; insertion, deletion and point-mutation fractions each U(0.05, 0.10) of L
# ('~5-10 %'); inserted + mutated positions filled in ONE ESMC pass (T 0.7, top-p 1.0, native excluded at mutated positions); ESMC 300M vs 600M.
#   inputs_mf (Gly edits + mutation sets) -> mf_seqs_300m.json / mf_seqs_600m.json -> cg2all -> inputs_mf_300m / inputs_mf_600m
source /etc/profile
module load conda/Python-ML-2025b-pytorch
export HF_HUB_OFFLINE=1
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
source activate pottsmpnn
export OMP_NUM_THREADS=1
python make_mild_inputs.py --inputs-dir $D/inputs --out-dir $D/inputs_mf --chains $CH --n-draws 8 --frac-lo 0.05 --frac-hi 0.10 --mut-frac-lo 0.05 --mut-frac-hi 0.10 || rc=$?
source activate esmfold2
export OMP_NUM_THREADS=24
for SZ in 300m 600m; do
  python esmc_fill_mut.py --inputs-dir $D/inputs_mf --out $D/mf_seqs_$SZ.json --esmc esmc_$SZ --chains $CH --draws 0 1 2 3 4 5 6 7 || rc=$?
done
source activate pottsmpnn
export OMP_NUM_THREADS=1
for SZ in 300m 600m; do
  python make_comp_inputs.py --stage bb --inputs-dir $D/inputs_mf --out-dir $D/inputs_mf_${SZ}_bb --chains $CH --draws 0 1 2 3 4 5 6 7 \
    --seqs $D/mf_seqs_$SZ.json --seq-arm esmc_$SZ || rc=$?
  for k in $CH; do mkdir -p $D/inputs_mf_${SZ}_cg/$k; done
  ls $D/inputs_mf_${SZ}_bb/*/d??.pdb | xargs -P 24 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_mf_'${SZ}'_cg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_mf_'${SZ}'_cg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/mf_${SZ}_cg_fail.txt || rc=$?
  n_in=$(ls $D/inputs_mf_${SZ}_bb/*/d??.pdb | wc -l); n_cg=$(find $D/inputs_mf_${SZ}_cg -name 'd??.pdb' -size +0 | wc -l)
  echo "$SZ n_in=$n_in n_cg=$n_cg fails=$(wc -l < $D/mf_${SZ}_cg_fail.txt)"
  [ "$n_in" = "$n_cg" ] || rc=1
  python make_comp_inputs.py --stage merge --inputs-dir $D/inputs_mf --out-dir $D/inputs_mf_$SZ --cg-dir $D/inputs_mf_${SZ}_cg --chains $CH --draws 0 1 2 3 4 5 6 7 \
    --seqs $D/mf_seqs_$SZ.json --seq-arm esmc_$SZ || rc=$?
done
echo "rc_final=$rc"
exit $rc
