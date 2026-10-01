#!/bin/bash
#SBATCH -J t2indel_scoretest
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_scoretest-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_scoretest-%j.err
# CPU only: scores the EDITED inputs (t*->0 limit) of one chain; native-vs-native must give TM 1.0
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
python score_indel.py --inputs-dir /home/gridsan/cou/t2_indel_data/inputs --score-dir /home/gridsan/cou/t2_indel_data/scores_selftest --edited-only --chains 7du7_A --workers 8 || rc=$?
python - <<'PY' || rc=$?
import numpy as np
from score_indel import ca_from_pdb, score_one
nat = "/home/gridsan/cou/t2_indel_data/inputs/7du7_A/native.pdb"
r = score_one(("x", "7du7_A", "selfcheck", 0, 0, ca_from_pdb(nat), nat, len(ca_from_pdb(nat))))
print("native-vs-native:", r)
assert abs(r["tm_native"] - 1.0) < 1e-4 and abs(r["tm_template"] - 1.0) < 1e-4 and r["n_aligned"] == r["L_native"]
PY
echo "rc=$rc"
exit $rc
