#!/bin/bash
#SBATCH -J pdarm_run
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/pdarm_run-%j.out
#SBATCH --error=/home/gridsan/cou/logs/pdarm_run-%j.err
# Generic partial-diffusion arm on the 3 pilot chains x draws 0-7 (same protocol as compF_run.sbatch: tiered, rewinds 375 300 250 200).
# usage: sbatch --export=ALL,TAG=cc94gly,MODEL=cc94,INPUTS=inputs_v2[,KINDS=control][,SEEDSC=1] pdarm_run.sbatch   (TAG = arm key; KINDS default indel; no commas inside values)
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
EXTRA=""
if [ -n "$SEEDSC" ]; then export PYTHONPATH=$HOME/pp1c_seedsc/src:$PYTHONPATH; EXTRA="--seed-self-cond"; fi   # SEEDSC=1: patched worktree + first-step self-cond seed
python run_indel_pd.py $EXTRA --inputs-dir $D/$INPUTS --out-root $D/sweep_$TAG --model $MODEL --kinds ${KINDS:-indel} --schedule tiered \
  --rewinds 375 300 250 200 --chains 7du7_A 6kyf_A 5zo3_A --draws 0 1 2 3 4 5 6 7
rc=$?
echo "rc_final=$rc"
exit $rc
