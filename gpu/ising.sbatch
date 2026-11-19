#!/usr/bin/env bash
#SBATCH --job-name=ising-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out
#
# Submit from the repo root:   sbatch gpu/ising.sbatch
# Pass a different script/args: sbatch gpu/ising.sbatch python gpu/run_thermo.py --L 64 128 256 512
# Default (no args) runs the full pipeline (gpu/run_all.sh).
set -euo pipefail

VENV="${VENV:-$HOME/ising-venv}"
source "$VENV/bin/activate"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  python: $(which python)"

if [ $# -gt 0 ]; then
    "$@"
else
    bash gpu/run_all.sh
fi
