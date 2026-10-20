"""性能基准：报告 MLUPS = Nx*Ny*steps / 1e6 / 秒。

不要拿不同网格、是否编译混在一起比。报告时把参数一起打出来。

计时是分块累加的：每块只计 ``solver.step()`` 加同步的时间，进度条的
刷新落在计时窗口之外，所以带不带 tqdm 得到的 MLUPS 是一样的。
"""

from __future__ import annotations

import argparse
import time

import torch

from lbm import create_solver
from lbm.progress import progress


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench(
    shape, steps: int, device: str, compile: bool, solid=None, chunk: int = 50
) -> tuple[float, int]:
    s = create_solver("D2Q9", shape, solid, nu=0.1, device=device, compile=compile)
    s.initialize()

    # 预热：编译、分配、首次同步都不计入
    s.step(min(50, steps))
    _sync(device)

    chunk = max(1, min(chunk, steps))
    n_chunks = max(1, steps // chunk)
    done = 0
    dt = 0.0

    with progress(n_chunks * chunk, desc=f"bench[{device}]") as bar:
        for _ in range(n_chunks):
            t0 = time.perf_counter()
            s.step(chunk)
            _sync(device)
            dt += time.perf_counter() - t0
            done += chunk
            bar.update(chunk)

    nodes = 1
    for n in shape:
        nodes *= n
    return nodes * done / 1e6 / dt, done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs="+", default=[1024, 1024])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--chunk", type=int, default=50, help="每块步数，只影响计时粒度")
    args = p.parse_args()

    mlups, done = bench(
        tuple(args.shape), args.steps, args.device, args.compile, chunk=args.chunk
    )
    print(
        f"shape={tuple(args.shape)} steps={done} "
        f"device={args.device} compile={args.compile} -> {mlups:.1f} MLUPS"
    )


if __name__ == "__main__":
    main()
