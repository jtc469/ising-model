"""
Section 4a on GPU: energy, specific heat, |m| and Binder cumulant vs T for several
lattice sizes, compared with Onsager, plus finite-size scaling of the C peak.

    python gpu/run_thermo.py                 # full run (see README for timings)
    python gpu/run_thermo.py --quick         # small smoke test
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import onsager
from ising_gpu import RunConfig, add_common_args, binder, replica_mean, resolve_devices, simulate

HERE = os.path.dirname(os.path.abspath(__file__))


def temperature_grid(args):
    coarse = np.linspace(args.T_min, args.T_max, args.n_coarse)
    fine = np.linspace(args.fine_min, args.fine_max, args.n_fine)
    return np.unique(np.round(np.concatenate([coarse, fine]), 6))


def peak(T, C_reps, n_boot=500, half_width=3, rng=None):
    """Location and height of the C peak: parabola through the points around the
    maximum of the replica-averaged curve, with bootstrap errors over replicas."""
    rng = rng or np.random.default_rng(0)

    def vertex(C):
        i = int(np.argmax(C))
        lo, hi = max(0, i - half_width), min(len(T), i + half_width + 1)
        if hi - lo < 3:
            return T[i], C[i]
        a, b, c = np.polyfit(T[lo:hi], C[lo:hi], 2)
        if a >= 0:
            return T[i], C[i]
        t = np.clip(-b / (2 * a), T[lo], T[hi - 1])
        return t, np.polyval([a, b, c], t)

    t0, c0 = vertex(C_reps.mean(1))
    R = C_reps.shape[1]
    if R < 2:
        return t0, np.nan, c0, np.nan
    boots = np.array([vertex(C_reps[:, rng.integers(R, size=R)].mean(1)) for _ in range(n_boot)])
    return t0, boots[:, 0].std(), c0, boots[:, 1].std()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--T-min", type=float, default=1.0)
    p.add_argument("--T-max", type=float, default=4.0)
    p.add_argument("--n-coarse", type=int, default=31)
    p.add_argument("--fine-min", type=float, default=2.15)
    p.add_argument("--fine-max", type=float, default=2.45)
    p.add_argument("--n-fine", type=int, default=31)
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--equil", type=int, default=50_000)
    p.add_argument("--measure", type=int, default=200_000)
    p.add_argument("--measure-every", type=int, default=1)
    p.add_argument("--start", choices=["cold", "hot"], default="cold")
    p.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    add_common_args(p)
    args = p.parse_args()
    if args.quick:
        args.L, args.replicas, args.equil, args.measure = [8, 16], 4, 500, 2000
        args.n_coarse, args.n_fine = 13, 7

    out = args.out or os.path.join(HERE, "out")
    os.makedirs(f"{out}/plots", exist_ok=True)
    os.makedirs(f"{out}/data", exist_ok=True)
    devices = resolve_devices(args.devices)
    T = temperature_grid(args)
    print(f"devices: {devices}\n{len(T)} temperatures in [{T[0]}, {T[-1]}], L = {args.L}")

    data = {}
    t_start = time.perf_counter()
    for L in args.L:
        cfg = RunConfig(L=L, equil_sweeps=args.equil, measure_sweeps=args.measure,
                        measure_every=args.measure_every, start=args.start,
                        compile=not args.no_compile, seed=args.seed + L)
        r = simulate(cfg, T, args.replicas, devices, args.max_sites)
        data[L] = r
        np.savez_compressed(f"{out}/data/thermo_L{L}.npz", T=T, **r)
    print(f"simulation done in {time.perf_counter() - t_start:.1f}s")

    # ---- summary ---------------------------------------------------------
    summary = {"Tc_exact": onsager.TC, "L": {}}
    peaks = []
    for L, r in data.items():
        tp, tp_err, cp, cp_err = peak(T, r["C"])
        peaks.append((L, tp, tp_err, cp, cp_err))
        summary["L"][L] = {"T_peak": tp, "T_peak_err": tp_err, "C_max": cp, "C_max_err": cp_err}
        print(f"L={L:5d}: C peaks at T = {tp:.4f} +- {tp_err:.4f}, C_max = {cp:.4f} +- {cp_err:.4f}")
    peaks = np.array(peaks)

    T_fine = np.linspace(T[0], T[-1], 2000)
    T_fine = T_fine[np.abs(T_fine - onsager.TC) > 1e-4]
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(data)))

    def per_L_plot(key, ylabel, fname, exact=None, exact_label="Onsager", T_exact=T_fine):
        plt.figure(figsize=(7, 5))
        for (L, r), col in zip(data.items(), colours):
            y, err = replica_mean(r[key])
            plt.errorbar(T, y, yerr=err, fmt="o-", ms=3, lw=1, color=col, label=f"L = {L}")
        if exact is not None:
            plt.plot(T_exact, exact, "k", lw=1.5, label=exact_label)
        plt.axvline(onsager.TC, color="black", ls="--", lw=1, label="Exact $T_c$")
        plt.xlabel("$T/J$")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out}/plots/{fname}", dpi=150)
        plt.close()

    per_L_plot("C", "$C(T)/k_B$", "ising_2d_specific_heat.png",
               onsager.specific_heat(T_fine), "Onsager ($L=\\infty$)")
    per_L_plot("abs_m", r"$\langle |m| \rangle$", "ising_2d_magnetisation.png",
               onsager.magnetisation(T_fine))
    per_L_plot("e", "$e(T)/J$", "ising_2d_energy.png", onsager.energy(T_fine))

    # Binder cumulant: curves for different L cross at Tc.
    plt.figure(figsize=(7, 5))
    win = (T >= args.fine_min) & (T <= args.fine_max)
    for (L, r), col in zip(data.items(), colours):
        U, err = binder(r["m2"], r["m4"])
        plt.errorbar(T[win], U[win], yerr=err[win], fmt="o-", ms=3, lw=1, color=col, label=f"L = {L}")
    plt.axvline(onsager.TC, color="black", ls="--", lw=1, label="Exact $T_c$")
    plt.xlabel("$T/J$")
    plt.ylabel("$U_4 = 1 - \\langle m^4\\rangle / 3\\langle m^2\\rangle^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/plots/ising_2d_binder.png", dpi=150)
    plt.close()

    # Finite-size scaling of the peak (needs >= 3 sizes to say anything).
    if len(peaks) >= 3:
        Ls, tp, tp_err, cp, cp_err = peaks.T
        slope, icpt = np.polyfit(np.log(Ls), cp, 1)
        a, Tc_fit = np.polyfit(1 / Ls, tp, 1)
        summary.update({"C_max_log_slope": slope, "C_max_log_slope_exact": onsager.CMAX_LOG_SLOPE,
                        "Tc_extrapolated": Tc_fit})
        print(f"C_max vs ln L slope = {slope:.4f} (exact {onsager.CMAX_LOG_SLOPE:.4f})")
        print(f"T_peak(L) -> Tc extrapolation: {Tc_fit:.4f} (exact {onsager.TC:.4f})")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.errorbar(Ls, cp, yerr=cp_err, fmt="o", label="Metropolis")
        xs = np.geomspace(Ls.min(), Ls.max(), 100)
        ax1.plot(xs, slope * np.log(xs) + icpt, label=f"fit, slope {slope:.3f}")
        ax1.plot(xs, onsager.CMAX_LOG_SLOPE * (np.log(xs) - np.log(Ls).mean()) + cp.mean(), "--",
                 label=f"exact slope {onsager.CMAX_LOG_SLOPE:.3f}")
        ax1.set_xscale("log")
        ax1.set_xlabel("$L$")
        ax1.set_ylabel("$C_{max}$")
        ax1.legend()
        ax2.errorbar(1 / Ls, tp, yerr=tp_err, fmt="o", label="$T_{peak}(L)$")
        xs = np.linspace(0, 1 / Ls.min(), 100)
        ax2.plot(xs, a * xs + Tc_fit, label=f"linear fit, $T_c$ = {Tc_fit:.4f}")
        ax2.axhline(onsager.TC, color="black", ls="--", lw=1, label=f"exact $T_c$ = {onsager.TC:.4f}")
        ax2.set_xlabel("$1/L$")
        ax2.set_ylabel("$T$")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(f"{out}/plots/ising_2d_finite_size_scaling.png", dpi=150)
        plt.close(fig)

    with open(f"{out}/data/thermo_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"plots in {out}/plots, data in {out}/data")


if __name__ == "__main__":
    main()
