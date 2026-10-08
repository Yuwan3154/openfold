#!/bin/bash
#SBATCH -J af2_compat_sc
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/af2_compat_sc-%j.out
#SBATCH --error=/home/gridsan/cou/logs/af2_compat_sc-%j.err
# Score the AF2 compatibility outputs of every set (pTM, pLDDT, CA RMSD and TM to the template; input sequence vs designs).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protebm
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
for spec in control:pool_ctl esmc07:pool_esmc milg:pool_milg milc:pool_milc milr:pool_milr mdg:pool_mdg mdc:pool_mdc mdr:pool_mdr; do
  L=${spec%%:*}; P=${spec##*:}
  [ -d $D/af2compat_$L ] || { echo "SKIP $L"; continue; }
  python af2_compat_score.py --pool-root $D/$P --in-dir $D/af2compat_$L --label $L --out $D/af2compat_${L}_scores.csv || rc=$?
done
echo "rc_final=$rc"
exit $rc
