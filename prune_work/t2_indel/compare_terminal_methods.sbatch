#!/bin/bash
#SBATCH -J t2_term_cmp
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_term_cmp-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_term_cmp-%j.err
# Per-atom rescaling (current) vs rigid translation for terminal insertions, on every chain/draw of the Gly-arm inputs.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
D=/home/gridsan/cou/t2_indel_data
CH=$(for f in $D/inputs/*/plans.json; do basename $(dirname $f); done)
python $HOME/of_t2indel/prune_work/t2_indel/compare_terminal_methods.py --inputs-dir $D/inputs --chains $CH --out-csv $D/terminal_methods.csv
rc=$?
echo rc=$rc
exit $rc
