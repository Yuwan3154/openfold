#!/bin/bash
#SBATCH -J eclean_sc_hash
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/cou/logs/eclean_sc_hash-%j.out
bash /home/gridsan/cou/of_t2indel/prune_work/t2_indel/engaging_clean/hash_sample.sh /home/gridsan/cou/pp1c_work/templates_merged /home/gridsan/cou/e_clean/sample_templates_merged.txt /home/gridsan/cou/e_clean/hash_sc_templates_merged.txt
