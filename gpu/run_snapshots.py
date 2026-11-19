"""
Section 4 snapshots on GPU: an L x L lattice below, at and above Tc.

    python gpu/run_snapshots.py --L 1024 --sweeps 5000
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ising_gpu import RunConfig, add_common_args, resolve_devices, simulate

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=1024)
    p.add_argument("--T", type=float, nargs="+", default=[1.5, 2.27, 3.5])
    p.add_argument("--sweeps", type=int, default=5000)
    p.add_argument("--start", choices=["cold", "hot"], default="cold",
                   help="cold = all up (notebook); hot = random, shows domain coarsening below Tc")
    add_common_args(p)
    args = p.parse_args()

    out = args.out or os.path.join(HERE, "out")
    os.makedirs(f"{out}/plots", exist_ok=True)
    cfg = RunConfig(L=args.L, equil_sweeps=args.sweeps, measure_sweeps=0, measure="none",
                    start=args.start, compile=not args.no_compile, seed=args.seed, return_spins=True)
    # One lattice per temperature on a single device: they are tiny for an H100.
    spins = simulate(cfg, args.T, 1, resolve_devices(args.devices)[:1], args.max_sites)["spins"][:, 0]

    fig, axes = plt.subplots(1, len(args.T), figsize=(4 * len(args.T), 4.2))
    for ax, T, s in zip(np.atleast_1d(axes), args.T, spins):
        ax.imshow(s, cmap="plasma", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"T = {T}")
        ax.axis("off")
    fig.suptitle(f"{args.L} x {args.L}, {args.sweeps} sweeps, {args.start} start")
    fig.tight_layout()
    fig.savefig(f"{out}/plots/ising_2d_snapshots.png", dpi=200)
    print(f"saved {out}/plots/ising_2d_snapshots.png")


if __name__ == "__main__":
    main()
