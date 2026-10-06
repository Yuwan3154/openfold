#!/bin/bash
#SBATCH -J eclean_sc_manifest
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/eclean_sc_manifest-%j.out
# SuperCloud-side manifest of the template tree that Engaging holds a pulled copy of
mkdir -p /home/gridsan/cou/e_clean
bash /home/gridsan/cou/of_t2indel/prune_work/t2_indel/engaging_clean/manifest_local.sh /home/gridsan/cou/pp1c_work/templates_merged /home/gridsan/cou/e_clean/sc_templates_merged.tsv.gz
echo "rc_final=$?"
