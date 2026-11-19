#!/usr/bin/env bash
# Full GPU pipeline: correctness checks, benchmark, then sections 4 and 5.
# Usage: bash gpu/run_all.sh            (logs to gpu/out/logs)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/logs
nvidia-smi --query-gpu=name,memory.total --format=csv | tee out/logs/gpus.txt

python selftest.py        2>&1 | tee out/logs/selftest.log
python benchmark.py       2>&1 | tee out/logs/benchmark.log
python run_snapshots.py   2>&1 | tee out/logs/snapshots.log
python run_correlation.py 2>&1 | tee out/logs/correlation.log
python run_thermo.py      2>&1 | tee out/logs/thermo.log
echo "done: plots in gpu/out/plots"
