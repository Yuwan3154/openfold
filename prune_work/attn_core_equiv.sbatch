#!/bin/bash
#SBATCH -J attn_equiv
#SBATCH -p pi_so3
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -o /orcd/pool/006/chenxiou/logs/attn_equiv-%j.out
#SBATCH -e /orcd/pool/006/chenxiou/logs/attn_equiv-%j.out
#
# Dump (and optionally compare) the attn_core extension on whatever GPU this lands on.
# ⛔ --time=1:00:00, NOT the pi_so3 max: a 2-day request cannot fit before the 09-15 maintenance
#    window and SLURM would defer this ~2-minute job to 09-16, which defeats its purpose. Same
#    value and same reasoning as the sm120 control jobs (22755853) already recorded in the plan.
#
#   sbatch --gres=gpu:h200:1        --export=ALL,OF3_ENV=$S/envs/of3_env,EQ_OUT=<trusted>.npz
#   sbatch --gres=gpu:h200:1        --export=ALL,OF3_ENV=$S/envs/of3_env_sm120,EQ_OUT=…,EQ_TRUSTED=…
#   sbatch --gres=gpu:rtx_pro_6000:1 --export=ALL,OF3_ENV=$S/envs/of3_env_sm120,EQ_OUT=…,EQ_TRUSTED=…
set -eo pipefail

S=/orcd/scratch/orcd/011/chenxiou
ENV=${OF3_ENV:-$S/envs/of3_env_sm120}
REPO=$S/openfold-esmfold2-recycling
OUT=${EQ_OUT:?EQ_OUT must name the npz this job writes}
TRUSTED=${EQ_TRUSTED:-}

echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "ENV=$ENV"
echo "OUT=$OUT"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
module load cuda/12.9.1
mkdir -p "$(dirname "$OUT")"
cd "$REPO"
# ⛔ $REPO must be on PYTHONPATH: this is run as a FILE, so sys.path[0] is
#    $REPO/prune_work, not $REPO. The old heredoc-fed gate got `openfold` from the cwd.

if [ -n "$TRUSTED" ]; then
    echo "mode: --check against $TRUSTED"
    PYTHONPATH="$REPO:$REPO/openfold" "$ENV/bin/python" prune_work/attn_core_equiv.py \
        --check "$TRUSTED" --scratch "$OUT"
else
    echo "mode: --dump (this run becomes the trusted reference)"
    PYTHONPATH="$REPO:$REPO/openfold" "$ENV/bin/python" prune_work/attn_core_equiv.py --dump "$OUT"
fi

echo "DONE $(date -Is)"
