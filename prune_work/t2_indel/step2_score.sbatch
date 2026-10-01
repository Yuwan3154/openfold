#!/bin/bash
#SBATCH -J t2indel_step2score
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_step2score-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_step2score-%j.err
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
for s in tiered grouped; do
  python score_indel.py --inputs-dir $D/inputs --out-root $D/step2_$s --score-dir $D/scores_step2_$s --models cc89 --chains 7du7_A --workers 8 || rc=$?
done
python - <<'PY' || rc=$?
import pandas as pd
D = "/home/gridsan/cou/t2_indel_data"
for s in ("tiered", "grouped"):
    d = pd.read_csv(f"{D}/scores_step2_{s}/cc89_7du7_A.csv")
    print(s); print(d[["kind", "draw", "rewind", "L_template", "tm_native", "tm_template", "rmsd", "n_aligned"]].round(3).to_string(index=False))
PY
echo "rc=$rc"
exit $rc
