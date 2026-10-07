#!/bin/bash
#SBATCH -J eclean_net
#SBATCH -p mit_data_transfer
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:20:00
#SBATCH -o /home/chenxiou/e_clean/net-%j.log
hostname
ls ~/.ssh | head
grep -A6 "Host SuperCloud" ~/.ssh/config
ssh -o BatchMode=yes -o ConnectTimeout=20 SuperCloud 'hostname; df -h ~ | tail -1; mkdir -p ~/e_clean/xfer && echo xfer-dir-ok'
echo "rc=$?"
