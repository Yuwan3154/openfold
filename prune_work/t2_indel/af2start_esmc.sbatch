#!/bin/bash
#SBATCH -J af2start_esmc
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/af2start_esmc-%j.out
#SBATCH --error=/home/gridsan/cou/logs/af2start_esmc-%j.err
# localfold completions of the ESMC T0.7 arm (af2fix_esmc07_lf2, RAW 116) -> PD input dir inputs_af2_esmc07lf; then: bash pdarm_submit.sh esmclf cc89 inputs_af2_esmc07lf
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
python af2_to_inputs.py --af2-dir $D/af2fix_esmc07_lf2 --src-inputs $D/inputs_esmc_T0.7_p1.0 --out-dir $D/inputs_af2_esmc07lf --chains 7du7_A 6kyf_A 5zo3_A --draws 0 1 2 3 4 5 6 7
rc=$?
n=$(find $D/inputs_af2_esmc07lf -name "d??.pdb" | wc -l); echo "pdbs=$n (expected 24)"; [ "$n" = "24" ] || rc=1
echo "rc_final=$rc"
exit $rc
