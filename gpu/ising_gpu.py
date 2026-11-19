"""
Batched checkerboard Metropolis for the 2D Ising model (J = k_B = 1) in PyTorch.

Why checkerboard?
    The notebook's sweep picks L*L random sites one after another, which is inherently
    serial. Colour the lattice like a chessboard: every neighbour of a black site is
    white, so all black sites can be updated at the same time without affecting each
    other, then all white sites. Each half-update satisfies detailed balance, so the
    chain samples the same Boltzmann distribution as the notebook (only the dynamics,
    i.e. the autocorrelation times, differ slightly).

Layout
    A batch of B independent L x L periodic lattices (each with its own temperature) is
    stored as two int8 tensors of shape (B, L, L // 2):
        black[b, i, j] = spin at (i, 2j + (i & 1))
        white[b, i, j] = spin at (i, 2j + 1 - (i & 1))
    Storing only one colour per tensor means every random number and every flip does
    useful work (no masked-out half of the lattice). With this packing the four
    neighbours of black[i, j] are white[i, j], white[i +- 1, j] and white[i, j -+ 1]
    (the sign depending on whether row i is even or odd), and symmetrically for white.

Parallelism
    * All temperatures x replicas are one batch -> one kernel per half-sweep.
    * Large batches are split into chunks (bounded by --max-sites) and chunks are spread
      round-robin over all visible GPUs, one process per GPU.
    * On CUDA the sweep and the measurement are fused with torch.compile.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

TC = 2 / math.log(1 + math.sqrt(2))  # Onsager critical temperature, ~2.269


# ---------------------------------------------------------------------------
# Lattice layout helpers
# ---------------------------------------------------------------------------
def odd_rows(L: int, device) -> torch.Tensor:
    """Bool tensor of shape (1, L, 1): True on odd rows."""
    return (torch.arange(L, device=device) % 2 == 1).view(1, L, 1)


def neighbours_black(white: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """Sum of the 4 (white) neighbours of every black site."""
    side = torch.where(odd, white.roll(-1, -1), white.roll(1, -1))
    return white + white.roll(1, -2) + white.roll(-1, -2) + side


def neighbours_white(black: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """Sum of the 4 (black) neighbours of every white site."""
    side = torch.where(odd, black.roll(1, -1), black.roll(-1, -1))
    return black + black.roll(1, -2) + black.roll(-1, -2) + side


def to_full(black: torch.Tensor, white: torch.Tensor) -> torch.Tensor:
    """(B, L, L//2) x 2 -> (B, L, L) full lattice."""
    B, L, H = black.shape
    full = torch.empty((B, L, 2 * H), dtype=black.dtype, device=black.device)
    full[:, 0::2, 0::2] = black[:, 0::2]
    full[:, 1::2, 1::2] = black[:, 1::2]
    full[:, 0::2, 1::2] = white[:, 0::2]
    full[:, 1::2, 0::2] = white[:, 1::2]
    return full


def from_full(full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L, L) full lattice -> (black, white) packed sublattices."""
    B, L, _ = full.shape
    black = torch.empty((B, L, L // 2), dtype=full.dtype, device=full.device)
    white = torch.empty_like(black)
    black[:, 0::2] = full[:, 0::2, 0::2]
    black[:, 1::2] = full[:, 1::2, 1::2]
    white[:, 0::2] = full[:, 0::2, 1::2]
    white[:, 1::2] = full[:, 1::2, 0::2]
    return black, white


def total_energy(black: torch.Tensor, white: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """E = -sum_<ij> s_i s_j. Every bond joins a black and a white site, so summing
    black * (its neighbours) counts each bond exactly once."""
    return -(black * neighbours_black(white, odd)).sum((-2, -1), dtype=torch.int32)


# ---------------------------------------------------------------------------
# Kernels (fused by torch.compile on CUDA)
# ---------------------------------------------------------------------------
def _metropolis_half(S, nb, p4, p8):
    # dE = 2 * s * (sum of neighbours) = 2x with x in {-4, -2, 0, 2, 4}.
    # Accept if dE <= 0, else with probability exp(-dE / T) (looked up, no exp per site).
    x = S * nb
    u = torch.rand(S.shape, device=S.device)
    accept = (x <= 0) | (u < torch.where(x == 2, p4, p8))
    return torch.where(accept, -S, S)


def _sweep(black, white, odd, p4, p8):
    black = _metropolis_half(black, neighbours_black(white, odd), p4, p8)
    white = _metropolis_half(white, neighbours_white(black, odd), p4, p8)
    return black, white


def _measure(black, white, odd, acc, E_ref, inv_N):
    # Running sums of (E - E_ref), (E - E_ref)^2, |m|, m^2, m^4 in float64.
    # Shifting by E_ref (energy at the start of measurement) avoids cancellation in var(E).
    E = total_energy(black, white, odd)
    M = black.sum((-2, -1), dtype=torch.int32) + white.sum((-2, -1), dtype=torch.int32)
    de = (E - E_ref).double()
    m = M.double() * inv_N
    m2 = m * m
    return acc + torch.stack([de, de * de, m.abs(), m2, m2 * m2])


_KERNELS: dict[bool, tuple] = {}


def _kernels(use_compile: bool):
    if use_compile not in _KERNELS:
        if use_compile:
            _KERNELS[use_compile] = (torch.compile(_sweep), torch.compile(_measure))
        else:
            _KERNELS[use_compile] = (_sweep, _measure)
    return _KERNELS[use_compile]


def row_col_means(black: torch.Tensor, white: torch.Tensor):
    """Row and column averages of every lattice, each (B, L) float64."""
    B, L, H = black.shape
    i32 = torch.int32
    rows = black.sum(-1, dtype=i32) + white.sum(-1, dtype=i32)
    even_cols = black[:, 0::2].sum(-2, dtype=i32) + white[:, 1::2].sum(-2, dtype=i32)  # k = 2j
    odd_cols = black[:, 1::2].sum(-2, dtype=i32) + white[:, 0::2].sum(-2, dtype=i32)   # k = 2j+1
    cols = torch.stack([even_cols, odd_cols], -1).reshape(B, L)
    return rows.double() / L, cols.double() / L


def circular_autocorr(x: torch.Tensor, r_max: int) -> torch.Tensor:
    """g[r] = mean_i x[i] x[i + r] (periodic), via FFT. Same as the notebook's
    np.mean(x * np.roll(x, r)) for every r at once."""
    L = x.shape[-1]
    F = torch.fft.rfft(x, dim=-1)
    g = torch.fft.irfft(F.real**2 + F.imag**2, n=L, dim=-1) / L
    return g[..., : r_max + 1]


# ---------------------------------------------------------------------------
# Running one chunk of lattices on one device
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    L: int
    equil_sweeps: int = 1000
    measure_sweeps: int = 3000
    measure_every: int = 1
    start: str = "cold"          # "cold" (all up, as in the notebook) or "hot" (random)
    measure: str = "thermo"      # "thermo", "corr" (thermo + G(r)) or "none"
    r_max: int = 10
    compile: bool = True
    seed: int = 1234
    return_spins: bool = False
    verbose: bool = True


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def run_chunk(cfg: RunConfig, temps: np.ndarray, device: str, chunk_id: int = 0) -> dict:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.manual_seed(cfg.seed + 7919 * chunk_id)

    L, B = cfg.L, len(temps)
    if L % 2 or L < 4:
        raise ValueError("L must be even and >= 4")
    N = L * L
    T = torch.as_tensor(temps, dtype=torch.float64, device=dev)
    p4 = torch.exp(-4 / T).float().view(B, 1, 1)
    p8 = torch.exp(-8 / T).float().view(B, 1, 1)
    odd = odd_rows(L, dev)

    shape = (B, L, L // 2)
    if cfg.start == "cold":
        black = torch.ones(shape, dtype=torch.int8, device=dev)
        white = torch.ones(shape, dtype=torch.int8, device=dev)
    else:
        black = (torch.randint(0, 2, shape, device=dev) * 2 - 1).to(torch.int8)
        white = (torch.randint(0, 2, shape, device=dev) * 2 - 1).to(torch.int8)

    sweep, measure = _kernels(cfg.compile and dev.type == "cuda")

    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(cfg.equil_sweeps):
        black, white = sweep(black, white, odd, p4, p8)

    out: dict = {}
    n_meas = 0
    if cfg.measure != "none" and cfg.measure_sweeps > 0:
        acc = torch.zeros((5, B), dtype=torch.float64, device=dev)
        E_ref = total_energy(black, white, odd)
        G = torch.zeros((B, cfg.r_max + 1), dtype=torch.float64, device=dev)
        for k in range(cfg.measure_sweeps):
            black, white = sweep(black, white, odd, p4, p8)
            if (k + 1) % cfg.measure_every:
                continue
            acc = measure(black, white, odd, acc, E_ref, 1.0 / N)
            if cfg.measure == "corr":
                rows, cols = row_col_means(black, white)
                G += circular_autocorr(rows, cfg.r_max) + circular_autocorr(cols, cfg.r_max)
            n_meas += 1

        acc = acc / n_meas
        mean_de, mean_de2 = acc[0], acc[1]
        var_E = (mean_de2 - mean_de**2).clamp_min(0)
        out["e"] = ((E_ref.double() + mean_de) / N).cpu().numpy()
        out["C"] = (var_E / (N * T**2)).cpu().numpy()
        out["abs_m"] = acc[2].cpu().numpy()
        out["m2"] = acc[3].cpu().numpy()
        out["m4"] = acc[4].cpu().numpy()
        if cfg.measure == "corr":
            out["G"] = (G / (2 * n_meas)).cpu().numpy()

    if cfg.return_spins:
        out["spins"] = to_full(black, white).cpu().numpy()

    _sync(dev)
    dt = time.perf_counter() - t0
    sweeps = cfg.equil_sweeps + cfg.measure_sweeps
    if cfg.verbose:
        rate = B * N * sweeps / dt / 1e9
        print(f"  [{device}] chunk {chunk_id}: L={L} B={B} sweeps={sweeps} "
              f"in {dt:.1f}s ({rate:.2f} Gflips/s incl. measurement + compile)", flush=True)
    return out


def _run_device(cfg: RunConfig, jobs: list[tuple[int, np.ndarray]], device: str):
    return [(cid, run_chunk(cfg, temps, device, cid)) for cid, temps in jobs]


# ---------------------------------------------------------------------------
# Public entry point: many temperatures x replicas over all GPUs
# ---------------------------------------------------------------------------
def resolve_devices(spec: str = "auto") -> list[str]:
    if spec == "cpu":
        return ["cpu"]
    if spec == "auto":
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n else ["cpu"]
    return [f"cuda:{int(i)}" for i in spec.split(",")]


def simulate(cfg: RunConfig, temperatures, replicas: int = 1,
             devices: list[str] | None = None, max_sites: int = 2**27) -> dict:
    """Run `replicas` independent lattices at each temperature.

    Returns a dict of numpy arrays with leading shape (n_T, replicas):
    e, C, abs_m, m2, m4 (and G (n_T, R, r_max+1) for measure="corr",
    spins (n_T, R, L, L) if return_spins).
    """
    devices = devices or resolve_devices()
    temperatures = np.asarray(temperatures, dtype=np.float64)
    nT = len(temperatures)
    temps = np.repeat(temperatures, replicas)
    B = len(temps)

    per_chunk = max(1, max_sites // (cfg.L * cfg.L))
    n_chunks = max(math.ceil(B / per_chunk), min(len(devices), B))
    chunks = np.array_split(np.arange(B), n_chunks)

    jobs: list[tuple[str, list]] = [(d, []) for d in devices]
    for cid, idx in enumerate(chunks):
        jobs[cid % len(devices)][1].append((cid, temps[idx]))
    jobs = [(d, j) for d, j in jobs if j]

    if cfg.verbose:
        print(f"L={cfg.L}: {nT} temperatures x {replicas} replicas = {B} lattices, "
              f"{n_chunks} chunk(s) on {[d for d, _ in jobs]}", flush=True)

    results: dict[int, dict] = {}
    if len(jobs) == 1:
        d, j = jobs[0]
        results.update(_run_device(cfg, j, d))
    else:
        # One process per GPU so each has its own CUDA context and Python thread.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
            futures = [pool.submit(_run_device, cfg, j, d) for d, j in jobs]
            for f in futures:
                results.update(f.result())

    ordered = [results[c] for c in range(n_chunks)]
    merged = {}
    for key in ordered[0]:
        arr = np.concatenate([r[key] for r in ordered], axis=0)
        merged[key] = arr.reshape(nT, replicas, *arr.shape[1:])
    return merged


# ---------------------------------------------------------------------------
# Statistics over replicas
# ---------------------------------------------------------------------------
def replica_mean(x: np.ndarray):
    """Mean and standard error over the replica axis (axis 1)."""
    R = x.shape[1]
    err = x.std(axis=1, ddof=1) / np.sqrt(R) if R > 1 else np.full(x.mean(1).shape, np.nan)
    return x.mean(axis=1), err


def binder(m2: np.ndarray, m4: np.ndarray):
    """Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2) from pooled replicas, jackknife error."""
    R = m2.shape[1]
    U = 1 - m4.mean(1) / (3 * m2.mean(1) ** 2)
    if R < 2:
        return U, np.full(U.shape, np.nan)
    s2, s4 = m2.sum(1, keepdims=True), m4.sum(1, keepdims=True)
    jk = 1 - ((s4 - m4) / (R - 1)) / (3 * ((s2 - m2) / (R - 1)) ** 2)
    err = np.sqrt((R - 1) / R * ((jk - jk.mean(1, keepdims=True)) ** 2).sum(1))
    return U, err


def add_common_args(parser):
    parser.add_argument("--devices", default="auto",
                        help="'auto' (all GPUs, else CPU), 'cpu', or e.g. '0,1'")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    parser.add_argument("--max-sites", type=int, default=2**27,
                        help="max spins per chunk on one GPU (memory ~ 16 bytes/spin)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=None, help="output directory (default gpu/out)")
    return parser
