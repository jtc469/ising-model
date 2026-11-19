"""
Correctness checks. Run before the big jobs (takes ~1 min on CPU, less on GPU):

    python gpu/selftest.py

1. Packed checkerboard layout round-trips and gives the same neighbour sums, energy and
   row/column correlations as the notebook's full-lattice numpy code.
2. The Onsager formulas are self-consistent (c = du/dT).
3. Monte Carlo on a 4x4 lattice matches exact enumeration of all 2^16 states.
"""
import argparse
import sys

import numpy as np
import torch

import onsager
from ising_gpu import (RunConfig, circular_autocorr, from_full, neighbours_black, neighbours_white,
                       odd_rows, replica_mean, resolve_devices, row_col_means, simulate, to_full,
                       total_energy)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def energy_2d(spins):  # from main.ipynb
    return -np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)))


def test_layout(device):
    rng = np.random.default_rng(0)
    for L in [4, 6, 8, 32]:
        full = rng.choice(np.array([-1, 1], dtype=np.int8), size=(3, L, L))
        ft = torch.as_tensor(full, device=device)
        b, w = from_full(ft)
        check(f"roundtrip L={L}", np.array_equal(to_full(b, w).cpu().numpy(), full))

        nb_full = (np.roll(full, 1, 1) + np.roll(full, -1, 1) + np.roll(full, 1, 2) + np.roll(full, -1, 2))
        odd = odd_rows(L, device)
        nb_b, nb_w = from_full(torch.as_tensor(nb_full, device=device))
        check(f"neighbours L={L}",
              torch.equal(neighbours_black(w, odd), nb_b) and torch.equal(neighbours_white(b, odd), nb_w))

        E = total_energy(b, w, odd).cpu().numpy()
        check(f"energy L={L}", np.array_equal(E, [energy_2d(s) for s in full]))

        rows, cols = row_col_means(b, w)
        rm, cm = full.mean(2), full.mean(1)
        G_ref = np.array([[np.mean(x * np.roll(x, r)) for r in range(L // 2)] for x in rm])
        G_ref += np.array([[np.mean(x * np.roll(x, r)) for r in range(L // 2)] for x in cm])
        G = (circular_autocorr(rows, L // 2 - 1) + circular_autocorr(cols, L // 2 - 1)).cpu().numpy()
        check(f"row/col correlation L={L}", np.allclose(G, G_ref))


def test_onsager():
    T = np.array([1.2, 1.8, 2.1, 2.5, 3.0, 4.0])
    h = 1e-5
    c_num = (onsager.energy(T + h) - onsager.energy(T - h)) / (2 * h)
    check("Onsager c = du/dT", np.allclose(c_num, onsager.specific_heat(T), rtol=1e-5),
          f"max rel err {np.max(np.abs(c_num / onsager.specific_heat(T) - 1)):.1e}")
    check("Onsager u(T->inf) ~ -2 tanh(1/T)", abs(onsager.energy(50.0) + 2 * np.tanh(1 / 50)) < 1e-4)


def exact_4x4(T):
    L = 4
    states = np.arange(2 ** (L * L))
    spins = (((states[:, None] >> np.arange(L * L)) & 1) * 2 - 1).reshape(-1, L, L)
    E = -np.sum(spins * (np.roll(spins, 1, 1) + np.roll(spins, 1, 2)), axis=(1, 2)).astype(float)
    absM = np.abs(spins.sum((1, 2))).astype(float)
    out = []
    for t in T:
        w = np.exp(-(E - E.min()) / t)
        w /= w.sum()
        mE, mE2 = w @ E, w @ E**2
        out.append(((mE / 16), (mE2 - mE**2) / (16 * t**2), w @ absM / 16))
    return np.array(out)


def test_exact(devices, use_compile):
    T = np.array([1.5, 2.27, 3.0])
    cfg = RunConfig(L=4, equil_sweeps=1000, measure_sweeps=20000, compile=use_compile, verbose=False)
    res = simulate(cfg, T, replicas=64, devices=devices[:1])
    exact = exact_4x4(T)
    for k, name in enumerate(["e", "C", "abs_m"]):
        mean, err = replica_mean(res[name])
        z = np.abs(mean - exact[:, k]) / err
        check(f"4x4 exact {name}", np.all(z < 5),
              f"MC {np.round(mean, 4)} exact {np.round(exact[:, k], 4)} |z| {np.round(z, 1)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="auto")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()
    devices = resolve_devices(args.devices)
    print(f"devices: {devices}")
    test_layout(devices[0])
    test_onsager()
    test_exact(devices, not args.no_compile)
    print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
