"""
Throughput benchmark: spin-flip attempts per second for each GPU.

    python gpu/benchmark.py
"""
import argparse
import time

import torch

from ising_gpu import _kernels, odd_rows, resolve_devices


@torch.no_grad()
def bench(L, B, device, use_compile, sweeps, warmup=20):
    dev = torch.device(device)
    shape = (B, L, L // 2)
    black = torch.ones(shape, dtype=torch.int8, device=dev)
    white = torch.ones(shape, dtype=torch.int8, device=dev)
    T = torch.linspace(1.5, 3.5, B, device=dev, dtype=torch.float64)
    p4 = torch.exp(-4 / T).float().view(B, 1, 1)
    p8 = torch.exp(-8 / T).float().view(B, 1, 1)
    odd = odd_rows(L, dev)
    sweep, _ = _kernels(use_compile and dev.type == "cuda")
    for _ in range(warmup):
        black, white = sweep(black, white, odd, p4, p8)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(sweeps):
        black, white = sweep(black, white, odd, p4, p8)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return B * L * L * sweeps / dt / 1e9, dt / sweeps * 1e3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", default="auto")
    p.add_argument("--sites", type=int, default=2**27, help="spins per batch")
    p.add_argument("--L", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--sweeps", type=int, default=200)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    for d in resolve_devices(args.devices):
        name = torch.cuda.get_device_name(d) if d.startswith("cuda") else "cpu"
        print(f"== {d} ({name}), torch {torch.__version__}")
        for L in args.L:
            B = max(1, args.sites // (L * L))
            rate, ms = bench(L, B, d, not args.no_compile, args.sweeps)
            print(f"  L={L:5d} x {B:6d} lattices: {rate:7.2f} Gflips/s  ({ms:.3f} ms/sweep)")


if __name__ == "__main__":
    main()
