#!/bin/bash
#SBATCH -J af2_compat
#SBATCH -p xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/af2_compat-%j.out
#SBATCH --error=/home/gridsan/cou/logs/af2_compat-%j.err
# AF2 compatibility test (user 10-08): partially diffused pool template as AF2 template + its INPUT sequence (and 4 MPNN designs) as query.
# Sets = the templates already refolded with ESMFold2 (same draws, so per-template results pair up): no-indel control (3 pilot chains, rewind 250,
# draws 0-1), ESMC T0.7 fills (arm esmc07, draws 0-3), mild 1-5 % edits (milg/milc/milr, draws 0-1), mid 10 % edits (mdg/mdc/mdr, draws 0-1).
# Resumable per template. usage: sbatch [-p debug-gpu --time=02:00:00] af2_compat.sbatch
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate colabdesign
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/gridsan/cou/of_t2indel/prune_work/t2_indel
D=/home/gridsan/cou/t2_indel_data
rc=0
# label pool arm(or -) dmax rewind(or 0)
while read -r LABEL POOL ARM DMAX RW; do
  [ -d $D/$POOL ] || { echo "SKIP $LABEL: no pool $POOL"; continue; }
  SEL=$(awk -F, -v dmax=$DMAX -v rw=$RW -v arm=$ARM 'NR>1 && ($1=="7du7_A"||$1=="6kyf_A"||$1=="5zo3_A") && $5<=dmax && (rw==0||$6==rw) && (arm=="-"||$4==arm) {printf "%s:%s ", $1, $3}' $D/$POOL/index.csv)
  echo "$LABEL: $(echo $SEL | wc -w) templates"
  [ -n "$SEL" ] || { echo "no templates for $LABEL"; rc=1; continue; }
  python af2_compat.py --pool-root $D/$POOL --select $SEL --out $D/af2compat_$LABEL < /dev/null || rc=$?
done <<LIST
control pool_ctl - 1 250
esmc07 pool_esmc esmc07 3 0
milg pool_milg - 1 0
milc pool_milc - 1 0
milr pool_milr - 1 0
mdg pool_mdg - 1 0
mdc pool_mdc - 1 0
mdr pool_mdr - 1 0
LIST
echo "rc_final=$rc"
exit $rc
