#!/bin/bash
#SBATCH -J t8b_cpu_step
#SBATCH -p mit_normal,mit_preemptable
#SBATCH --exclude=node1800,node1801,node1305
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
# One CPU step of the T8b template regeneration (pack build / append, top-up planning, diversity selection, verify) in the
# t2 pipeline's pp1c env (numpy, pandas; USalign at ~/.local/bin). node1800/1801/1305 lack lmod/conda (10-23).
#   env CMD="python t2_pack_build.py ..." REPO_DIR=<t8b-regen checkout> sbatch --export=ALL -o <log> t8b_cpu_step.sbatch
set -uo pipefail
REPO="${REPO_DIR:?REPO_DIR = the t8b-regen checkout}"
CODE=$REPO/prune_work/t2_indel
export PATH=/orcd/scratch/orcd/011/chenxiou/pp1c/env/bin:$PATH PYTHONPATH=$CODE PYTHONUNBUFFERED=1
cd "$CODE" || exit 3
git -C "$REPO" log -1 --format="[repo] %h %s"
echo "[python] $(command -v python)"
echo "CMD: ${CMD:?set CMD}"
eval "$CMD"
rc=$?
echo "T8B_STEP_DONE rc=$rc"
exit $rc
