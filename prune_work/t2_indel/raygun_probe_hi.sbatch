#!/bin/bash
#SBATCH -J raygun_probe_hi
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/raygun_probe_hi-%j.out
#SBATCH --error=/home/gridsan/cou/logs/raygun_probe_hi-%j.err
# user 10-02: Raygun noise 0.3 0.4 0.5 (the 0-0.2 grid is raygun_probe.sbatch)
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate raygun
export OMP_NUM_THREADS=24
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
rc=0
python raygun_probe.py --inputs-dir /home/gridsan/cou/t2_indel_data/inputs --out /home/gridsan/cou/t2_indel_data/raygun_probe_hi.json --noise 0.3 0.4 0.5 || rc=$?
echo "rc=$rc"
exit $rc
