#!/bin/bash
#SBATCH -J hdrscan
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/home/gridsan/cou/logs/hdrscan-%j.out
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate protpardelle
python - <<'PY'
import glob, re
bad = 0
for f in sorted(glob.glob("/home/gridsan/cou/t2_indel_data/pool_work/*/seqs/*.fa")):
    for ln in open(f):
        if ln.startswith(">") and not (re.search(r"score=([-0-9.eE]+)", ln) and re.search(r"global_score=([-0-9.eE]+)", ln)):
            bad += 1
            if bad <= 3: print(f, ln.strip()[:200])
print("bad headers:", bad)
PY
