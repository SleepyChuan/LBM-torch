"""Karman 涡街：圆柱绕流，左 Zou/He 速度入口 + 右 Zou/He 密度出口。

用法::

    python examples/karman.py --device mps
    python examples/karman.py --device mps --nx 360 --ny 240 --d 10 --quick

产出 ``out/karman.gif``（涡量场随时间的动画）。

几个约定：

* y 方向仍是周期边界（域高 24D，圆柱在中间，上下各留 12D，周期像的影响很小）。
* 网格 ``(Nx, Ny)``，axis 0 是 x。``solid`` 为 ``(Nx, Ny)`` 的 bool。
* 涡量用二阶中心差分，固体节点置 NaN，不参与配色归一化。
* 配色用一个**全局**的 ``TwoSlopeNorm``，不是逐帧归一化——逐帧会让颜色
  随涡量极值抖动，看不出涡的生成与脱落。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lbm import create_solver
from lbm.progress import progress, run_steps


def build_cylinder_mask(
    nx: int, ny: int, cx: float, cy: float, r: float
) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float32).view(-1, 1)
    y = torch.arange(ny, dtype=torch.float32).view(1, -1)
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r


def vorticity(ux: np.ndarray, uy: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """omega = du_y/dx - du_x/dy，数组是 (Nx, Ny)。"""
    duy_dx = np.gradient(uy, axis=0)
    dux_dy = np.gradient(ux, axis=1)
    w = duy_dx - dux_dy
    w[solid] = np.nan
    return w


def run(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    nu = args.u_inf * args.d / args.re
    tau = 3.0 * nu + 0.5
    r = args.d / 2.0
    cx, cy = args.cx * args.d, args.ny // 2

    print(f"grid=({args.nx},{args.ny}) D={args.d} Re={args.re}")
    print(
        f"u_inf={args.u_inf}  nu={nu:.5f}  tau={tau:.4f}  "
        f"Ma={args.u_inf * np.sqrt(3):.4f}"
    )
    if tau <= 0.51:
        print(
            "注意：tau 接近 0.5。BGK 在这个区间会发散；MRT 用 magic parameter "
            "把 ghost（能量通量）模式压住（s_q << 1），能量模式保持 1.9。"
        )

    solid = build_cylinder_mask(args.nx, args.ny, cx, cy, r)
    solver = create_solver(
        "D2Q9",
        (args.nx, args.ny),
        solid,
        nu=nu,
        collision="mrt",
        Lambda=args.Lambda,
        boundary="inlet_outlet",
        u_inf=args.u_inf,
        rho_inf=args.rho_inf,
        device=args.device,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
        compile=args.compile,
    )
    print(
        f"collision=mrt  Lambda={args.Lambda:.4f}  s_nu={solver.collision.s_nu:.4f}  "
        f"s_e={solver.collision.s_e:.4f}  s_q={solver.collision.s_q:.4f}"
    )
    solver.initialize(rho=args.rho_inf, u=(args.u_inf, 0.0))

    solid_np = solid.numpy()
    probe = (int(cx + 3 * args.d), int(cy))
    probe_hist: list[float] = []
    frames: list[np.ndarray] = []

    run_steps(solver, args.spinup, desc=f"spin-up[{args.device}]", chunk=500)

    # 探针点单独取：只算一个节点的 u_y，比整场 macro() 便宜得多，
    # 所以可以加密采样（FFT 的频带宽度由总窗口决定，采样太稀会看不出 St）。
    total = args.frames * args.interval
    every = args.probe_every
    n_chunk = total // every
    frame_steps = set(range(args.interval, total + 1, args.interval))

    with progress(total, desc=f"capture[{args.device}]") as bar:
        for k in range(1, n_chunk + 1):
            solver.step(every)
            probe_hist.append(_probe_uy(solver, *probe))

            if k * every in frame_steps:
                _, ux, uy = solver.macro()
                frames.append(
                    vorticity(ux.cpu().numpy(), uy.cpu().numpy(), solid_np).astype(
                        np.float16
                    )
                )
                d = solver.diagnose()
                bar.set_postfix_str(
                    f"frame {len(frames)}/{args.frames} "
                    f"|u|max={d.max_velocity:.3f} rho_max={d.rho_max:.4f}"
                )

            bar.update(every)

    return np.stack(frames), solid_np, np.array(probe_hist), float(nu), float(cx)


def _probe_uy(solver, px: int, py: int) -> float:
    """单个节点的 u_y。只做 (Q,) 的运算 + 一次同步。"""
    f = solver.pdf[solver.cur_idx][:, px, py]
    return float((f * solver._st.c[1].flatten()).sum() / f.sum())


def estimate_strouhal(
    probe: np.ndarray, interval: int, d: float, u_inf: float
) -> float:
    sig = probe - probe.mean()
    if sig.size < 8 or np.allclose(sig, 0):
        return float("nan")
    spec = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, d=interval)  # 单位：1/步
    spec[0] = 0.0
    f = freqs[int(np.argmax(spec))]
    return float(f * d / u_inf)


def animate(
    frames: np.ndarray,
    solid: np.ndarray,
    out: Path,
    fps: int,
    stride: int,
    vmax: float | None = None,
    vmax_percentile: float = 99.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list(
        "vorticity_cmap", ["blue", "white", "red"], N=256
    )

    # 全局归一化，中心锁在 0，所有帧共用同一个 norm（逐帧归一化会让颜色抖动）。
    # 上下限不用绝对极值：圆柱壁面的涡量比尾迹大一个量级，用极值会把
    # 涡街压成一片灰白。默认取 |ω| 的 99% 分位数。
    if vmax is None:
        flat = frames[np.isfinite(frames)]
        step = max(1, flat.size // 1_000_000)
        vmax = float(np.percentile(np.abs(flat[::step]), vmax_percentile))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    nx, ny = frames.shape[1], frames.shape[2]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    image = ax.imshow(
        frames[0].T.astype(np.float32),
        origin="lower",
        extent=(0, nx, 0, ny),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")
    fig.colorbar(image, ax=ax, label=r"$\omega$", fraction=0.035, pad=0.02)
    fig.tight_layout()

    def update(i: int):
        f = frames[i].T.astype(np.float32)
        image.set_data(f)
        title.set_text(f"Karman vortex street   frame {i + 1}/{len(frames)}")
        return image, title

    anim = FuncAnimation(
        fig,
        update,
        frames=range(0, len(frames), stride),
        interval=1000 / fps,
        blit=False,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=720)
    p.add_argument("--ny", type=int, default=480)
    p.add_argument("--d", type=int, default=20, help="圆柱直径（格子单位）")
    p.add_argument("--re", type=float, default=100.0)
    p.add_argument("--u-inf", type=float, default=0.05)
    p.add_argument("--rho-inf", type=float, default=1.0)
    p.add_argument(
        "--Lambda",
        type=float,
        default=3.0 / 16.0,
        help="TRT magic parameter，(tau+-0.5)(tau--0.5)，默认 3/16",
    )
    p.add_argument("--cx", type=float, default=5.0, help="圆心 x，以 D 为单位")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--spinup",
        type=int,
        default=20000,
        help="抛掉的发展期步数；Re=100 时脱落要 t*=U t/D≈50 之后才明显",
    )
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--interval", type=int, default=120, help="帧间隔步数")
    p.add_argument(
        "--probe-every", type=int, default=20, help="探针采样步数；越小 FFT 频带越窄"
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=1, help="抽帧，>1 可减小文件")
    p.add_argument(
        "--vmax", type=float, default=None, help="色标上下限（对称）；不给就用分位数"
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="色标上下限取 |omega| 的该分位数，默认 99",
    )
    p.add_argument(
        "--save-frames", default=None, help="把涡量帧存成 npz，之后可只重渲染不重算"
    )
    p.add_argument(
        "--load-frames", default=None, help="直接从 npz 载入涡量帧，跳过仿真"
    )
    p.add_argument("--out", default="out/karman.gif")
    p.add_argument("--quick", action="store_true", help="小网格快跑，仅用于自检")
    args = p.parse_args()

    if args.quick:
        args.nx, args.ny, args.d = 360, 240, 10
        args.spinup, args.frames, args.interval = 3000, 60, 120

    if args.load_frames:
        data = np.load(args.load_frames)
        frames = data["frames"]
        solid = data["solid"]
        print(f"loaded {args.load_frames}: {frames.shape}")
    else:
        frames, solid, probe, nu, cx = run(args)
        st = estimate_strouhal(probe, args.probe_every, args.d, args.u_inf)
        print(f"probe u_y 峰峰值 {np.ptp(probe):.3e}")
        print(f"Strouhal ≈ {st:.4f}   (Re={args.re} 的文献值约 0.165)")
        if args.save_frames:
            np.savez_compressed(args.save_frames, frames=frames, solid=solid)
            print(f"saved frames -> {args.save_frames}")

    animate(
        frames,
        solid,
        Path(args.out),
        args.fps,
        args.stride,
        vmax=args.vmax,
        vmax_percentile=args.vmax_percentile,
    )


if __name__ == "__main__":
    main()
