#!/bin/bash
#SBATCH -J eclean_new
#SBATCH -p mit_data_transfer
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o /home/chenxiou/e_clean/new-%j.log
R=/orcd/compute/so3/002/chenxi/of_run
find $R -type f -newermt "2026-10-06 18:00" -printf '%TY-%Tm-%Td %TH:%TM  %s  %P\n' | sort | head -80
echo "count: $(find $R -type f -newermt '2026-10-06 18:00' | wc -l)"
