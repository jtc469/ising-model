"""
Section 5 on GPU: row/column-average correlation G(r) above Tc, fit G ~ exp(-r/xi),
then xi ~ (T - Tc)^-nu.

    python gpu/run_correlation.py            # L = 256, 32 replicas per T
    python gpu/run_correlation.py --quick
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import onsager
from ising_gpu import RunConfig, add_common_args, replica_mean, resolve_devices, simulate

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_xi(r, G, err, r_min=1, n_sigma=3.0):
    """Weighted fit of log G = a - r / xi over r_min <= r up to the first point where
    G is no longer n_sigma above its error bar."""
    ok = (G > n_sigma * err) & (G > 0)
    end = np.argmin(ok[r_min:]) + r_min if not ok[r_min:].all() else len(r)
    sel = slice(r_min, end)
    if end - r_min < 2:
        return np.nan, np.nan, sel
    w = G[sel] / err[sel]  # 1 / sigma(log G)
    (slope, _), cov = np.polyfit(r[sel], np.log(G[sel]), 1, w=w, cov="unscaled")
    xi = -1 / slope
    return xi, np.sqrt(cov[0, 0]) * xi**2, sel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=256)
    p.add_argument("--T", type=float, nargs="+",
                   default=[2.35, 2.4, 2.45, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.4])
    p.add_argument("--replicas", type=int, default=32)
    p.add_argument("--equil", type=int, default=20_000)
    p.add_argument("--measure", type=int, default=100_000)
    p.add_argument("--measure-every", type=int, default=1)
    p.add_argument("--r-max", type=int, default=None, help="default L // 4")
    p.add_argument("--quick", action="store_true")
    add_common_args(p)
    args = p.parse_args()
    if args.quick:
        args.L, args.replicas, args.equil, args.measure = 32, 4, 500, 2000
        args.T = [2.6, 2.8, 3.0, 3.4]
    r_max = args.r_max or args.L // 4

    out = args.out or os.path.join(HERE, "out")
    os.makedirs(f"{out}/plots", exist_ok=True)
    os.makedirs(f"{out}/data", exist_ok=True)
    T = np.array(sorted(args.T))

    cfg = RunConfig(L=args.L, equil_sweeps=args.equil, measure_sweeps=args.measure,
                    measure_every=args.measure_every, measure="corr", r_max=r_max,
                    compile=not args.no_compile, seed=args.seed)
    res = simulate(cfg, T, args.replicas, resolve_devices(args.devices), args.max_sites)
    np.savez_compressed(f"{out}/data/correlation_L{args.L}.npz", T=T, **res)

    r = np.arange(r_max + 1)
    G, G_err = replica_mean(res["G"])
    xis, xi_errs = [], []
    plt.figure(figsize=(7, 5))
    for k, t in enumerate(T):
        xi, xi_err, sel = fit_xi(r, G[k], G_err[k])
        xis.append(xi)
        xi_errs.append(xi_err)
        line = plt.errorbar(r, G[k], yerr=G_err[k], fmt="o", ms=3, label=f"T = {t}")[0]
        if np.isfinite(xi):
            rs = r[sel]
            amp = np.exp(np.mean(np.log(G[k][sel]) + rs / xi))
            plt.plot(rs, amp * np.exp(-rs / xi), color=line.get_color(), lw=1)
    plt.yscale("log")
    plt.xlabel("$r$")
    plt.ylabel("$G(r)$")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/ising_2d_correlation.png", dpi=150)
    plt.close()

    xis, xi_errs = np.array(xis), np.array(xi_errs)
    xi_exact = onsager.xi_row(T)
    good = np.isfinite(xis)
    dT = T - onsager.TC
    slope, icpt = np.polyfit(np.log(dT[good]), np.log(xis[good]), 1, w=(xis / xi_errs)[good])
    nu = -slope
    # Effective exponent of the *exact* xi over the same window (corrections to scaling).
    nu_exact_window = -np.polyfit(np.log(dT), np.log(xi_exact), 1)[0]

    print("T          :", T)
    print("Measured xi:", np.round(xis, 3))
    print("+-         :", np.round(xi_errs, 3))
    print("Exact xi   :", np.round(xi_exact, 3))
    print(f"nu = {nu:.3f}  (exact xi over the same T window gives {nu_exact_window:.3f}; asymptotic nu = 1)")

    plt.figure(figsize=(7, 5))
    plt.errorbar(dT, xis, yerr=xi_errs, fmt="o", label="Measured")
    plt.loglog(dT, xi_exact, "k--", lw=1, label="Exact")
    xs = np.geomspace(dT.min(), dT.max(), 100)
    plt.loglog(xs, np.exp(icpt) * xs**slope, label=rf"Fit, $\nu$ = {nu:.3f}")
    plt.xlabel("$T - T_c$")
    plt.ylabel(r"$\xi$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/plots/ising_2d_xi_power_law.png", dpi=150)
    plt.close()

    with open(f"{out}/data/correlation_summary.json", "w") as f:
        json.dump({"L": args.L, "T": T.tolist(), "xi": xis.tolist(), "xi_err": xi_errs.tolist(),
                   "xi_exact": xi_exact.tolist(), "nu": nu, "nu_exact_window": nu_exact_window},
                  f, indent=2)
    print(f"plots in {out}/plots, data in {out}/data")


if __name__ == "__main__":
    main()
