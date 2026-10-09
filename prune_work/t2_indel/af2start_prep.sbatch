#!/bin/bash
#SBATCH -J af2start_prep
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/af2start_prep-%j.out
#SBATCH --error=/home/gridsan/cou/logs/af2start_prep-%j.err
# AF2-completed structures (af2fix_gly/comp/raygun, 3 pilot chains x draws 0-7, 10-30 % edits) -> partial-diffusion input dirs
# inputs_af2_gly / inputs_af2_comp / inputs_af2_rg (all atoms of the query residue types), source plans from inputs / inputs_compF / inputs_rg3_full.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
rc=0
for spec in "gly inputs" "comp inputs_compF" "rg inputs_rg3_full"; do
  set -- $spec
  arm=$1; src=$2; [ $arm = rg ] && ad=af2fix_raygun || ad=af2fix_$arm
  python af2_to_inputs.py --af2-dir $D/$ad --src-inputs $D/$src --out-dir $D/inputs_af2_$arm --chains $CH --draws 0 1 2 3 4 5 6 7 || rc=$?
done
n=$(find $D/inputs_af2_gly $D/inputs_af2_comp $D/inputs_af2_rg -name "d??.pdb" | wc -l); echo "pdbs=$n (expected 72)"; [ "$n" = "72" ] || rc=1
echo "rc_final=$rc"
exit $rc
