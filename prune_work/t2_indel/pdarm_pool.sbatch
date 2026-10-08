#!/bin/bash
#SBATCH -J pdarm_pool
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/pdarm_pool-%j.out
#SBATCH --error=/home/gridsan/cou/logs/pdarm_pool-%j.err
# Score (USalign TM to native), build the pilot pool at rewind 250 and design 32 ProteinMPNN sequences per backbone (mpnn_design_pool.py defaults).
# usage: sbatch --export=ALL,TAG=cc94gly,MODEL=cc94,INPUTS=inputs_v2 pdarm_pool.sbatch
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
CH="7du7_A 6kyf_A 5zo3_A"
rc=0
python score_indel.py --inputs-dir $D/$INPUTS --out-root $D/sweep_$TAG --score-dir $D/${TAG}_scores --models $MODEL --chains $CH --workers 8 || rc=$?
python build_indel_pool.py --out-root $D/pool_$TAG --rewind 250 --model $MODEL --chains $CH --draws 0 1 2 3 4 5 6 7 \
  --arm $TAG $D/sweep_$TAG $D/$INPUTS $D/${TAG}_scores || rc=$?
python mpnn_design_pool.py --pool-root $D/pool_$TAG --mpnn-dir /home/gridsan/cou/ProteinMPNN --work-dir $D/pool_${TAG}_work || rc=$?
echo "rc_final=$rc"
exit $rc
