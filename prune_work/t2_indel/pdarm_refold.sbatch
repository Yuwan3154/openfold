#!/bin/bash
#SBATCH -J pdarm_refold
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/pdarm_refold-%j.out
#SBATCH --error=/home/gridsan/cou/logs/pdarm_refold-%j.err
# ESMFold2 refold (refold_pilot_esm settings) of draws 0-3 of each chain of the arm's pilot pool (3 chains x 4 = 12 templates x 33 folds).
# index.csv columns: chain,file,i,arm,draw,rewind,L,tm_native. usage: sbatch --export=ALL,TAG=cc94gly[,DMAX=1] pdarm_refold.sbatch
# RW (optional) = keep only rows with this rewind (control pools hold 4 rungs per draw; the arms refold rewind 250).
# DMAX = highest draw refolded (default 3 -> 12 templates; a cost limit, not a selection on results; the mild-edit arms use DMAX=1 -> 6 templates).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
DMAX=${DMAX:-3}
RW=${RW:-0}
SEL=$(awk -F, -v dmax=$DMAX -v rw=$RW 'NR>1 && $5<=dmax && (rw==0 || $6==rw) {printf "%s:%s ", $1, $3}' $D/pool_$TAG/index.csv)
echo "selected: $SEL"
[ "$(echo $SEL | wc -w)" = "$((3 * (DMAX + 1)))" ] || { echo "expected $((3 * (DMAX + 1))) templates"; exit 1; }
python refold_esmfold2.py --pool-root $D/pool_$TAG --select $SEL --out $D/refold_$TAG
rc=$?
echo "rc_final=$rc"
exit $rc
