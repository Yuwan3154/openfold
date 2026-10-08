#!/bin/bash
# Chain the four generic partial-diffusion-arm stages with afterok. usage: bash pdarm_submit.sh TAG MODEL INPUTS
set -e
[ $# -eq 3 ] || { echo "usage: bash pdarm_submit.sh TAG MODEL INPUTS"; exit 1; }
TAG=$1; MODEL=$2; INPUTS=$3
S=$(dirname "$0")
a=$(sbatch --parsable --export=ALL,TAG=$TAG,MODEL=$MODEL,INPUTS=$INPUTS $S/pdarm_run.sbatch)
b=$(sbatch --parsable --dependency=afterok:$a --export=ALL,TAG=$TAG,MODEL=$MODEL,INPUTS=$INPUTS $S/pdarm_pool.sbatch)
c=$(sbatch --parsable --dependency=afterok:$b --export=ALL,TAG=$TAG $S/pdarm_refold.sbatch)
d=$(sbatch --parsable --dependency=afterok:$c --export=ALL,TAG=$TAG $S/pdarm_score.sbatch)
echo "$TAG: run=$a pool=$b refold=$c score=$d"
