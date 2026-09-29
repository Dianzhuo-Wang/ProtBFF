#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -t 0-01:30
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -o /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/slurm_logs/wpool_build_%j.out
#SBATCH -e /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/slurm_logs/wpool_build_%j.out
#SBATCH -J wpool_build
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
echo "=== ProSST weighted-mean ==="
$PY data_pipeline/build_wpool_cache.py \
  --merged_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/merged_output \
  --pool mean --out model_benchmarking/score_caches/skempi_prosst_wmean_cache.npz
echo "=== ESM-C weighted-mean ==="
$PY data_pipeline/build_wpool_cache.py \
  --merged_dir data/merged_output_esmc \
  --pool mean --out model_benchmarking/score_caches/skempi_esmc_wmean_cache.npz
echo "=== DONE ==="
