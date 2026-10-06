#!/bin/bash
#SBATCH -J t2indel_af2_segv
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_af2_segv-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_af2_segv-%j.err
# Which environment variant lets jax 0.5.3 compile the AF2 graph on a V100 without a segfault?
source /etc/profile
source activate colabdesign 2>/dev/null || { module load conda/Python-ML-2025b-pytorch; source activate colabdesign; }
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
run() {
  name=$1; shift
  rm -rf $D/af2_probe_$name $D/af2_probe_tmp_$name
  ( env "$@" timeout 900 python -X faulthandler refold_af2.py --pool-root $D/pool_smoke --select 7du7_A:0 --max-seqs 1 \
      --out $D/af2_probe_$name --tmp-dir $D/af2_probe_tmp_$name 2>&1 | tail -3 | cut -c1-160 )
  echo "VARIANT $name rc=${PIPESTATUS[0]} npz=$(ls $D/af2_probe_$name 2>/dev/null | wc -l)"
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "A: module-free (no system CUDA module), default flags"
run nomodule XLA_PYTHON_CLIENT_PREALLOCATE=false
module load cuda/12.6
echo "B: cuda/12.6 module + triton gemm off"
run tritonoff XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "C: unlimited stack + autotune off"
ulimit -s unlimited
run stack XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false" XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "rc_final=0"
