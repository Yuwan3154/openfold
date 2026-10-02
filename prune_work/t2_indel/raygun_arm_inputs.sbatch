#!/bin/bash
#SBATCH -J raygun_arm_inputs
#SBATCH -p xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/raygun_arm_inputs-%j.out
#SBATCH --error=/home/gridsan/cou/logs/raygun_arm_inputs-%j.err
# Raygun arm (option 1, noise 0.5): sequences for all 24 x 64 draws, then the alignment-defined edited inputs. CPU only.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate raygun
export OMP_NUM_THREADS=24
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
python raygun_generate.py --inputs-dir $D/inputs --out $D/raygun_seqs_n0.5.json --noise 0.5 || rc=$?
python make_raygun_inputs.py --inputs-dir $D/inputs --seqs $D/raygun_seqs_n0.5.json --out-dir $D/inputs_raygun_n0.5 || rc=$?
echo "rc=$rc"
exit $rc
