#!/bin/bash
#SBATCH --job-name=phono3py-gpu-test
#SBATCH --account=wlu-liushi
#SBATCH --partition=16V100,4V100,8V100V0
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/12.4.1
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
export CUPY_CACHE_DIR="$PWD/.cupy_cache"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python -m pytest "$@" -v
