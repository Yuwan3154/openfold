#!/bin/bash
#SBATCH -J t2indel_invest
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_invest-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_invest-%j.err
# Designability investigation, CPU part: (1) no-indel control pool, (2) geometry metrics for native / control / indel backbones,
# (3) ProteinMPNN (same settings as the pool) on the control backbones.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
export PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
python build_ctl_pool.py --sweep $D/sweep_gly --inputs-dir $D/inputs --scores $D/scores_gly --out-root $D/pool_ctl --draws 2 || rc=$?
# geom_metrics imports diagnose_indel -> pool_sse -> mdtraj, which only the protebm env has (the first run died here)
source activate protebm
python geom_metrics.py --inputs-dir $D/inputs --sweep $D/sweep_gly --pool-roots $D/pool_t250 $D/pool_ctl --out $D/geom.csv || rc=$?
echo "rc_geom_pool=$rc"
source activate protpardelle
python mpnn_design_pool.py --pool-root $D/pool_ctl --mpnn-dir /home/gridsan/cou/ProteinMPNN --work-dir $D/pool_ctl_work || rc=$?
echo "rc_final=$rc"
exit $rc
