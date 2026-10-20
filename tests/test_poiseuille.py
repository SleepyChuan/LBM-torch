from __future__ import annotations

import math

import pytest
import torch

from lbm import create_solver
from lbm.progress import run_steps

# 算例参数。u_max 由 F、nu、H 决定，必须落在低马赫数区间。
NX, NY = 4, 16
NU = 0.1
F = 1.0e-4
STEPS = 20000


def _analytic(y: torch.Tensor) -> torch.Tensor:
    """半程反弹的有效壁面在 y=0.5 与 y=Ny-1.5，不是节点索引。"""
    return F / (2 * NU) * (y - 0.5) * (NY - 1.5 - y)


@pytest.mark.slow
def test_poiseuille_profile(device, progress_out):
    solid = torch.zeros((NX, NY), dtype=torch.bool)
    solid[:, 0] = True
    solid[:, -1] = True

    s = create_solver("D2Q9", (NX, NY), solid, nu=NU, force=(F, 0.0), device=device)
    s.initialize(rho=1.0, u=None)
    mass0 = s.diagnose().total_mass

    run_steps(s, STEPS, desc=f"poiseuille[{device}]", chunk=2000, file=progress_out)

    rho, ux, uy = s.macro()
    ux = ux.cpu().to(torch.float64)
    uy = uy.cpu().to(torch.float64)

    y = torch.arange(NY, dtype=torch.float64)
    ana = _analytic(y)

    # 马赫数必须落在低马赫数区间
    assert ana.max().item() * math.sqrt(3.0) < 0.1

    # 入口效应：x 方向周期且充分发展，不同 x 截面必须一致
    assert (ux[1] - ux[NX - 2]).abs().max().item() < 1e-6

    # uy 应为 0
    assert uy[:, 1:-1].abs().max().item() < 1e-5

    # 内部剖面（去掉最靠墙的 2 行）。
    # TRT + Λ=3/16 的壁面位置是精确的，实测相对 L2 约 1e-5 量级；
    # BGK 在这个分辨率下是 0.35%。阈值定 1%，留三个数量级余量，
    # 同时能抓住任何把 TRT 的 magic parameter 改坏的改动。
    lo, hi = 2, NY - 1
    num = ux[1, lo:hi]
    ref = ana[lo:hi]
    rel = (num - ref).norm() / ref.norm()
    assert rel.item() < 0.01, f"相对 L2 误差 {rel.item():.3%} 超过 1%"

    # 峰值也要对上
    assert abs(num.max().item() - ref.max().item()) / ref.max().item() < 0.01

    # 质量守恒。容差按 float32 的舍入平台给：TRT 的碰撞比 BGK 多几步运算，
    # 平台实测为 2.3e-5 相对（BGK 是 3.5e-6），且 5k 步之后不再增长；
    # float64 下两者都在 1e-11 量级。真正的质量泄漏会随步数线性增长，
    # 1e-4 这个量级足够区分。
    assert abs(s.diagnose().total_mass - mass0) < 1e-4 * mass0


@pytest.mark.slow
@pytest.mark.parametrize("ny,steps", [(16, 20000), (32, 80000)])
def test_poiseuille_grid_convergence(device, progress_out, ny, steps):
    """固定 u_max（不是固定 F），加密网格后误差应下降。

    固定 F 的话 u_max ∝ H^2，Ny=64 时 Ma 已到 0.68，测的是压缩性误差。
    """
    u_target = 0.02
    h = ny - 2
    f = 8 * NU * u_target / h**2

    solid = torch.zeros((NX, ny), dtype=torch.bool)
    solid[:, 0] = True
    solid[:, -1] = True

    s = create_solver("D2Q9", (NX, ny), solid, nu=NU, force=(f, 0.0), device=device)
    s.initialize()
    run_steps(
        s,
        steps,
        desc=f"convergence Ny={ny}[{device}]",
        chunk=4000,
        file=progress_out,
    )

    _, ux, _ = s.macro()
    ux = ux.cpu().to(torch.float64)
    y = torch.arange(ny, dtype=torch.float64)
    ana = f / (2 * NU) * (y - 0.5) * (ny - 1.5 - y)

    lo, hi = 2, ny - 1
    rel = (ux[1, lo:hi] - ana[lo:hi]).norm() / ana[lo:hi].norm()
    # 两档都应该在 3% 以内；收敛阶在 validation.md 里记录，不写成硬断言
    assert rel.item() < 0.03, f"Ny={ny} 相对误差 {rel.item():.3%}"
