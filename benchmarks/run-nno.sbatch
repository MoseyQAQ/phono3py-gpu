#!/bin/bash
#SBATCH --job-name=phono3py-gpu-nno
#SBATCH --account=wlu-liushi
#SBATCH --partition=16V100
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
if [ "$1" = "scale" ]; then
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method rta --mesh 9 --devices 0
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method rta --mesh 9 --devices 0 1
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method rta --mesh 9 --devices 0 1 2 3
elif [ "$1" = "profile" ]; then
    .venv/bin/python -u benchmarks/profile_nno.py
elif [ "$1" = "solve" ]; then
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method lbte --mesh 6 --devices 0
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method lbte --mesh 6 --devices 0 1 2 3 --lbte-solver mg
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method wte --mesh 6 --devices 0 1 2 3
    .venv/bin/python -u benchmarks/nno.py --backend cpu --method wte --mesh 6 --read-gamma-from benchmarks/nno/wte-gpu-m6-q1-t64-g4-tiled16.npz
elif [ "$1" = "lbte" ]; then
    .venv/bin/python -u benchmarks/nno.py --backend cpu --method lbte --mesh 6
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method lbte --mesh 6
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method lbte --mesh 6 --q-batch 8 --triplet-batch 256
elif [ "$1" = "wte-reference" ]; then
    .venv/bin/python -u benchmarks/nno.py --backend cpu --method wte --read-gamma-from benchmarks/nno/rta-gpu-m12-q1-t64.npz
else
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method rta
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method rta --q-batch 8 --triplet-batch 256
    .venv/bin/python -u benchmarks/nno.py --backend gpu --method wte --q-batch 8 --triplet-batch 256
fi
