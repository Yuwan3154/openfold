#!/bin/bash
#SBATCH -J seedsc_test
#SBATCH -p debug-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/cou/logs/seedsc_test-%j.out
#SBATCH --error=/home/gridsan/cou/logs/seedsc_test-%j.err
# Regression + effect test of the seed_self_cond patch on one real item (7du7_A d00 of the ESMC T0.7 arm, cc89, tiered 375/300/250/200):
# flag OFF must reproduce the stock-checkout sweep bit-for-bit, flag ON must change the result.
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate protpardelle
export PYTHONUNBUFFERED=1
export PYTHONPATH=$HOME/pp1c_seedsc/src:$PYTHONPATH
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
IN=$D/inputs_esmc_T0.7_p1.0
rm -rf $D/seedsc_test_off $D/seedsc_test_on
rc=0
python -c "import inspect, protpardelle.core.models as m; print('protpardelle from', m.__file__); assert 'seed_self_cond' in inspect.getsource(m.Protpardelle.sample), 'patched worktree NOT loaded'" || exit 1
python run_indel_pd.py --inputs-dir $IN --out-root $D/seedsc_test_off --model cc89 --kinds indel --chains 7du7_A --draws 0 || rc=$?
python run_indel_pd.py --inputs-dir $IN --out-root $D/seedsc_test_on --model cc89 --kinds indel --chains 7du7_A --draws 0 --seed-self-cond || rc=$?
python seedsc_compare.py --ref $D/sweep_esmc_T0.7_p1.0/cc89/7du7_A/d00.npz --off $D/seedsc_test_off/cc89/7du7_A/d00.npz --on $D/seedsc_test_on/cc89/7du7_A/d00.npz || rc=$?
echo "rc_final=$rc"
exit $rc
