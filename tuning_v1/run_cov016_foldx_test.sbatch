#!/bin/bash
#SBATCH -p test
#SBATCH -t 0-00:40
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -o slurm_logs/cov016foldxtest_%j.out
#SBATCH -e slurm_logs/cov016foldxtest_%j.err
#SBATCH -J cov016fxtest
export COV016_TEST=3
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
source /n/sw/Miniforge3-24.11.3-0-fasrc02/etc/profile.d/conda.sh 2>/dev/null
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/fold_x_for_cov016.py
echo "=== outputs ==="
ls -la data/dms_cov016/cov016_cache/wildtype/ data/dms_cov016/cov016_cache/optimized/ 2>/dev/null
