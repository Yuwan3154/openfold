#!/bin/bash
#SBATCH -J t2_regen_scope
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2_regen_scope-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2_regen_scope-%j.err
# usage: sbatch --export=ALL,INPUTS=inputs[,CA_ONLY=1] verify_regen_scope.sbatch (INPUTS = the arm's input dir under t2_indel_data; single token).
# Which Gly-arm draws change under the rigid terminal rule (verify before regenerating only those).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
D=/home/gridsan/cou/t2_indel_data
CH=$(for f in $D/$INPUTS/*/plans.json; do basename $(dirname $f); done)
git -C $HOME/of_t2indel rev-parse HEAD
python $HOME/of_t2indel/prune_work/t2_indel/verify_regen_scope.py --inputs-dir $D/$INPUTS --chains $CH --out-csv $D/regen_scope_$INPUTS.csv ${CA_ONLY:+--ca-only}
rc=$?
echo rc=$rc
exit $rc
