#!/bin/bash
#SBATCH -J t2indel_af3env
#SBATCH -p download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_af3env-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_af3env-%j.err
bash /home/gridsan/cou/of_t2indel/prune_work/t2_indel/af3_env_build.sbatch_body.sh
echo "rc_final=$?"
