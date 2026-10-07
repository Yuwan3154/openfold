#!/bin/bash
#SBATCH -J t2indel_rf_af3
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_rf_af3-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_rf_af3-%j.err
# AF3 refold PILOT (same 6 backbones x 33 sequences) + the native-sequence CONTROL (6 chains). No MSA, no templates.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
D=/home/gridsan/cou/t2_indel_data
SEL="7du7_A:0 7du7_A:64 6kyf_A:0 6kyf_A:64 5zo3_A:0 5zo3_A:64"
CH="7du7_A 6kyf_A 5zo3_A 6ve7_W 6el2_A 6x61_B"
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
source activate protpardelle
python refold_af3.py make --pool-root $D/pool_t250 --select $SEL --json-dir $D/af3_pilot_json || exit 1
python refold_af3.py make_ctl --inputs-dir $D/inputs --chains $CH --json-dir $D/af3_ctl_json || exit 1
conda deactivate
cd /home/gridsan/cou/alphafold3_sc
rc=0
for kind in ctl pilot; do
  .venv/bin/python run_alphafold.py --input_dir=$D/af3_${kind}_json --output_dir=$D/af3_${kind}_out --model_dir=/home/gridsan/cou/params \
    --norun_data_pipeline --flash_attention_implementation=xla --num_diffusion_samples=1 --jax_compilation_cache_dir=$D/af3_jaxcache || rc=$?
  echo "rc_run_$kind=$rc"
done
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
source activate protpardelle
python refold_af3.py collect_ctl --inputs-dir $D/inputs --chains $CH --json-dir $D/af3_ctl_json --out $D/af3_ctl_out --out2 $D/refold_ctl_af3 || rc=$?
python refold_af3.py collect --pool-root $D/pool_t250 --select $SEL --json-dir $D/af3_pilot_json --out $D/af3_pilot_out --out2 $D/refold_af3 || rc=$?
echo "rc_final=$rc"
exit $rc
