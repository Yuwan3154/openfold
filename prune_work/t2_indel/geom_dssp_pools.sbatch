#!/bin/bash
#SBATCH -J geom_dssp_pools
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/geom_dssp_pools-%j.out
#SBATCH --error=/home/gridsan/cou/logs/geom_dssp_pools-%j.err
# Geometry panel (bond lengths/angles, omega, Ramachandran, clashes; geom_metrics.py) and DSSP distribution (pool_dssp.py) of every pilot pool built so far
# (the geometry gate candidates for using the synthetic templates as-is). Pools that do not exist are skipped. usage: sbatch [--export=ALL,POOLS="a b"] geom_dssp_pools.sbatch
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protebm
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
LIST=${POOLS:-"pool_t250 pool_ctl pool_esmc pool_milg pool_milc pool_milr pool_mdg pool_mdc pool_mdr pool_cc94gly pool_cc94ctl pool_esmc07s pool_mf300 pool_mf600"}
R=""
for p in $LIST; do [ -f $D/$p/index.csv ] && R="$R $D/$p" || echo "SKIP $p"; done
rc=0
python geom_metrics.py --inputs-dir $D/inputs --sweep $D/sweep_gly --pool-roots $R --out $D/geom_pools.csv || rc=$?
python pool_dssp.py --inputs-dir $D/inputs --pool-roots $R --out $D/dssp_pools.csv || rc=$?
echo "rc_final=$rc"
exit $rc
