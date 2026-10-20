"""Poiseuille 算例：周期通道 + 均匀体积力，验证抛物线速度剖面。

按 solver.step(interval) → save → diagnose 的循环写，和论文里
需要"每 N 步存一次原始 PDF"的用法一致。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from lbm import create_solver
from lbm.progress import progress

NX, NY = 4, 64
NU = 0.1
F = 2.5e-6  # u_max ≈ 2.4e-2，Ma ≈ 0.042
TOTAL_STEPS = 200000
SAVE_INTERVAL = 20000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--out", default="out/poiseuille")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    h = NY - 2
    u_max = F * h * h / (8 * NU)
    ma = u_max * math.sqrt(3.0)
    print(f"u_max = {u_max:.4e}, Ma = {ma:.4e}")
    if ma >= 0.1:
        raise SystemExit("马赫数超出低马赫区间，请下调 F")

    solid = torch.zeros((NX, NY), dtype=torch.bool)
    solid[:, 0] = True
    solid[:, -1] = True

    solver = create_solver(
        "D2Q9",
        (NX, NY),
        solid,
        nu=NU,
        force=(F, 0.0),
        device=args.device,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,  # 要做定量对比，别用 float16
    )
    solver.initialize(rho=1.0, u=None)

    out = Path(args.out)
    if not args.no_save:
        solver.dump_setup(
            out,
            total_steps=args.steps,
            save_interval=SAVE_INTERVAL,
            extra={"case": "poiseuille"},
        )

    # 保存间隔不能超过总步数，否则块数取整成 0，一步都不跑
    interval = min(SAVE_INTERVAL, args.steps)
    done = 0
    with progress(args.steps, desc=f"poiseuille[{args.device}]") as bar:
        while done < args.steps:
            n = min(interval, args.steps - done)
            solver.step(n)
            done += n
            bar.update(n)

            d = solver.diagnose()
            bar.set_postfix_str(
                f"|u|max={d.max_velocity:.3e} rho=[{d.rho_min:.4f},{d.rho_max:.4f}]"
            )
            if not args.no_save:
                path = solver.save_npz(out)
                bar.write(f"saved {path.name}")

    _report(solver)


def _report(solver) -> None:
    rho, ux, uy = solver.macro()
    ux = ux.cpu().to(torch.float64)
    uy = uy.cpu().to(torch.float64)

    y = torch.arange(NY, dtype=torch.float64)
    ana = F / (2 * NU) * (y - 0.5) * (NY - 1.5 - y)

    print("\n  y      ux(num)      ux(analytic)   rel.err")
    for j in range(1, NY - 1):
        a = ana[j].item()
        n = ux[1, j].item()
        print(f"{j:3d}  {n: .6e}  {a: .6e}  {(n - a) / a: .3%}")

    lo, hi = 2, NY - 1
    rel = ((ux[1, lo:hi] - ana[lo:hi]).norm() / ana[lo:hi].norm()).item()
    print(
        f"\ninterior rel L2 = {rel:.3%}, max|uy| = {uy[1, 1:-1].abs().max().item():.2e}"
    )
    print(solver.diagnose())


if __name__ == "__main__":
    main()
