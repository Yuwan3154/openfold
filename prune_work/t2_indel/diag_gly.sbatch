#!/bin/bash
#SBATCH -J t2indel_diag
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_diag-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_diag-%j.err
# CPU: seam-geometry + SSE diagnostics over the Gly sweep (env protebm has mdtraj 1.11.0)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protebm
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
python diagnose_indel.py --inputs-dir $D/inputs --out-root $D/sweep_gly --models cc89 cc91 --kinds indel control --diag-dir $D/diag_gly --workers 32 || rc=$?
echo "rc_final=$rc"
exit $rc
