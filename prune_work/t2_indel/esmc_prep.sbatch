#!/bin/bash
#SBATCH -J esmc_prep
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_prep-%j.err
# ESMC-fill arm inputs (user 10-08): Gly inputs regenerated under the rigid terminal rule (affected draws only) -> per ESMC setting:
# backbone-only PDBs with the ESMC fill types -> cg2all full atoms -> merge (survivors keep native atoms). Pilot: 3 chains x draws 0-7.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate pottsmpnn
export OMP_NUM_THREADS=1
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
FILLS=/home/gridsan/cou/t2_indel_data/esmc_pilot/fills.json
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
python regen_gly_inputs.py --inputs-dir $D/inputs --out-dir $D/inputs_v2 --chains $CH || rc=$?
for ARM in T0.7_p1.0 T0.5_p1.0; do
  python make_comp_inputs.py --stage bb --inputs-dir $D/inputs_v2 --out-dir $D/inputs_esmc_${ARM}_bb --chains $CH --draws $(seq 0 7) \
    --fills $FILLS --fill-arm esmc_$ARM || rc=$?
  for k in $CH; do mkdir -p $D/inputs_esmc_${ARM}_cg/$k; done
  ls $D/inputs_esmc_${ARM}_bb/*/d??.pdb | xargs -P 15 -I{} sh -c 'k=$(basename $(dirname {})); f=$(basename {}); [ -s '$D'/inputs_esmc_'${ARM}'_cg/$k/$f ] || CUDA_VISIBLE_DEVICES= convert_cg2all -p {} -o '$D'/inputs_esmc_'${ARM}'_cg/$k/$f --cg BackboneModel --device cpu > /dev/null 2>&1 || echo FAIL {}' > $D/esmc_${ARM}_cg_fail.txt || rc=$?
  n_in=$(ls $D/inputs_esmc_${ARM}_bb/*/d??.pdb | wc -l); n_cg=$(find $D/inputs_esmc_${ARM}_cg -name 'd??.pdb' -size +0 | wc -l)
  echo "$ARM n_in=$n_in n_cg=$n_cg"
  [ "$n_in" = "$n_cg" ] || rc=1
  python make_comp_inputs.py --stage merge --inputs-dir $D/inputs_v2 --out-dir $D/inputs_esmc_$ARM --cg-dir $D/inputs_esmc_${ARM}_cg --chains $CH --draws $(seq 0 7) \
    --fills $FILLS --fill-arm esmc_$ARM || rc=$?
done
echo "rc_final=$rc"
exit $rc
