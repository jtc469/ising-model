# GPU 2D Ising (PyTorch, multi-GPU)

GPU port of sections 4 and 5 of `main.ipynb` (2D model, $J = k_B = 1$), written for 2x H100.

## What changed vs the notebook

| Notebook | GPU version |
|---|---|
| Random-site Metropolis, one spin at a time | **Checkerboard Metropolis**: all black sites, then all white sites, in parallel. Same Boltzmann distribution, slightly different dynamics. |
| One lattice per temperature, run in a Python loop | Every temperature x replica is one batch, so each half-sweep is one fused kernel (`torch.compile`). |
| `int` spins, full lattice | `int8` spins packed per colour (`(B, L, L/2)`), so every random number does useful work. The acceptance probabilities $e^{-4/T}, e^{-8/T}$ are looked up, not computed with `exp` per site. |
| L = 20 / 40, no error bars | L = 16 to 256 (configurable), 16 to 32 independent replicas per T, so you get error bars from the spread across replicas. |
| CPU | Batches split into chunks (`--max-sites`) and spread across all GPUs, one process per GPU. |

Additions: the exact Onsager $e(T)$ and $C(T)$ curves, the Binder cumulant crossing, finite-size scaling of the $C$ peak ($C_{max}$ vs $\ln L$, $T_{peak}(L) \to T_c$), and error bars on $\xi$ and $\nu$.

## Files

- `ising_gpu.py`: the engine (layout, kernels, multi-GPU `simulate`, replica statistics)
- `onsager.py`: exact infinite-lattice results
- `run_thermo.py`: section 4a. $e$, $C$, $\langle|m|\rangle$ and $U_4$ vs T for several L, plus finite-size scaling
- `run_snapshots.py`: section 4 snapshots (default 1024 x 1024)
- `run_correlation.py`: section 5. $G(r)$, $\xi(T)$, $\nu$
- `selftest.py`: checks the layout against the notebook's numpy code, checks the Onsager formulas, and compares a 4x4 run with exact enumeration
- `benchmark.py`: Gflips/s per GPU
- `run_all.sh`: runs everything in order and logs to `out/logs`

## Running on the Launchpad instance

```bash
git clone <repo> && cd ising-model
pip install -r gpu/requirements.txt      # skip torch if you're in an NGC PyTorch container
python gpu/selftest.py                   # must say ALL PASSED
python gpu/benchmark.py                  # throughput per GPU
bash gpu/run_all.sh                      # everything; or run the scripts one at a time
```

Run inside `tmux` so a dropped SSH connection doesn't kill the job. Outputs go to `gpu/out/plots` (same filenames as the notebook, plus `ising_2d_energy.png`, `ising_2d_binder.png` and `ising_2d_finite_size_scaling.png`) and `gpu/out/data` (`.npz` raw per-replica results and `*_summary.json`).

Useful flags (all scripts support `--help`):

- `--L 32 64 128 256 512`: lattice sizes for `run_thermo.py`
- `--equil`, `--measure`: sweep counts. Near $T_c$ Metropolis slows down roughly as $L^{2.17}$, so scale these up if you add L = 512 or larger.
- `--replicas`: independent lattices per T; error bars shrink as $1/\sqrt{R}$
- `--devices 0` (one GPU), `--devices 0,1`, `--devices cpu`
- `--max-sites`: spins per chunk per GPU (about 16 bytes each). The default $2^{27}$ uses about 2 GB; raise it if the benchmark shows small L under-using the GPU.
- `--no-compile`: fall back to eager PyTorch if `torch.compile` causes trouble
- `--quick`: tiny smoke test (`run_thermo.py`, `run_correlation.py`)

## Cost

The total work is (sum of $L^2$) x lattices x sweeps. The default `run_thermo.py` is about $2 \times 10^{13}$ flip attempts: 59 temperatures x 16 replicas x 87k sites (L = 16 to 256 combined) x 250k sweeps. Divide by twice the benchmark's Gflips/s to get the wall time on 2 GPUs. `torch.compile` adds about a minute of compilation for each new lattice shape.

## Verified locally (CPU only)

`selftest.py` passes, and every script runs end to end with `--quick` on CPU, as does the 2-process multi-device path (it gives identical results to 1 process). Not yet tested on CUDA: `torch.compile` on GPU and real H100 throughput.
