#!/bin/bash
#SBATCH -J esmc_refold
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_refold-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_refold-%j.err
# ESMFold2 refold (refold_pilot_esm settings) of the ESMC pilot pool: draws 0-3 of each chain and arm (24 templates x 33 folds; the cost
# limit, not a selection on results). index.csv columns: chain,file,i,arm,draw,rewind,L,tm_native.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
SEL=$(awk -F, 'NR>1 && $5<=3 {printf "%s:%s ", $1, $3}' $D/pool_esmc/index.csv)
echo "selected: $SEL"
[ "$(echo $SEL | wc -w)" = "24" ] || { echo "expected 24 templates"; exit 1; }
python refold_esmfold2.py --pool-root $D/pool_esmc --select $SEL --out $D/refold_esmc
rc=$?
echo "rc_final=$rc"
exit $rc
