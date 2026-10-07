#!/bin/bash
#SBATCH -J t2indel_af3_smoke
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_af3_smoke-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_af3_smoke-%j.err
# SHAKEDOWN: AF3 on ONE template x 3 sequences (generation + 2 designs), V100, xla attention. NOT the CUDA module (jax segfault lesson).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"   # AF3's own requirement for compute capability 7.x (run_alphafold.py:975)
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
source activate protpardelle
python refold_af3.py make --pool-root $D/pool_t250 --select 7du7_A:0 --max-seqs 2 --json-dir $D/af3_smoke_json || exit 1
conda deactivate
cd /home/gridsan/cou/alphafold3_sc
.venv/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
.venv/bin/python run_alphafold.py --input_dir=$D/af3_smoke_json --output_dir=$D/af3_smoke_out --model_dir=/home/gridsan/cou/params \
  --norun_data_pipeline --flash_attention_implementation=xla --num_diffusion_samples=1 --jax_compilation_cache_dir=$D/af3_jaxcache
rc=$?
echo "rc_run=$rc"
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
source activate protpardelle
python refold_af3.py collect --pool-root $D/pool_t250 --select 7du7_A:0 --max-seqs 2 --json-dir $D/af3_smoke_json --out $D/af3_smoke_out --out2 $D/refold_af3_smoke || rc=$?
echo "rc_final=$rc"
exit $rc
