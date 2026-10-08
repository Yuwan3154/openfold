#!/bin/bash
#SBATCH -J ctlarm_pool
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/ctlarm_pool-%j.out
#SBATCH --error=/home/gridsan/cou/logs/ctlarm_pool-%j.err
# NO-INDEL CONTROL pool of a partial-diffusion model (user rule: a gate must pass a known-good control before a failing arm is evidence):
# score (USalign TM) the control sweep, build the control pool (draws 0-1 x 4 rungs per chain), design 32 ProteinMPNN sequences per backbone.
# usage: sbatch --export=ALL,TAG=cc94ctl,MODEL=cc94,INPUTS=inputs_v2 ctlarm_pool.sbatch   (needs sweep_$TAG from pdarm_run with KINDS=control)
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
rc=0
python score_indel.py --inputs-dir $D/$INPUTS --out-root $D/sweep_$TAG --score-dir $D/${TAG}_scores --models $MODEL --chains $CH --workers 8 || rc=$?
python build_ctl_pool.py --sweep $D/sweep_$TAG --inputs-dir $D/$INPUTS --scores $D/${TAG}_scores --out-root $D/pool_$TAG --draws 2 --model $MODEL --chains $CH || rc=$?
python mpnn_design_pool.py --pool-root $D/pool_$TAG --mpnn-dir /home/gridsan/cou/ProteinMPNN --work-dir $D/pool_${TAG}_work || rc=$?
echo "rc_final=$rc"
exit $rc
