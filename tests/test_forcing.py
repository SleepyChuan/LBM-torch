from __future__ import annotations

import pytest
import torch

from lbm import create_solver


def test_uniform_force_grows_momentum_linearly(device):
    """全域周期 + 均匀体积力：u_x = (n + 1/2) * F。

    那个 1/2 来自宏观速度定义 u = (M + F/2)/rho，M 每步增加 F。
    写成 n*F 会有一个 1/(2n) 的相对偏差。

    步数不取太大：float32 累积误差随 n 增长（n=500 时约 2e-4），
    而 F/2 的信号是 1/(2n)，n 太大会把两者混在一起。
    n=100 时信号 5e-3，噪声 ~4e-5，差两个数量级。
    """
    F = 1e-5
    n = 100
    s = create_solver("D2Q9", (8, 8), nu=0.1, force=(F, 0.0), device=device)
    s.initialize(rho=1.0, u=None)
    s.step(n)

    _, ux, uy = s.macro()
    expected = (n + 0.5) * F
    # 用绝对容差：动量是 sum(f*c) 的抵消结果，float32 下绝对误差固定在
    # 1e-7 量级，与步数无关。相对容差在小 u 上会失真。
    assert abs(ux.mean().item() - expected) < 1e-6
    # 横向分量只剩 float32 在 sum 里的抵消误差，量级 1e-7
    assert uy.abs().max().item() < 1e-6


def test_first_step_velocity_equals_one_and_a_half_f(device):
    """单步之后 u = (F + F/2)/rho = 1.5F。这里噪声极小，可以卡得很紧，
    专门盯住 F/2 与 (1 - 1/(2 tau)) 这两个因子。"""
    F = 1e-4
    s = create_solver("D2Q9", (8, 8), nu=0.1, force=(F, 0.0), device=device)
    s.initialize(rho=1.0, u=None)
    s.step(1)
    _, ux, _ = s.macro()
    assert abs(ux.mean().item() - 1.5 * F) < 1e-7


def test_zero_force_keeps_fluid_at_rest(device):
    s = create_solver("D2Q9", (8, 8), nu=0.1, device=device)
    s.initialize(rho=1.0, u=None)
    s.step(200)
    _, ux, uy = s.macro()
    # float32 的 sum(f*c) 是抵消求和，绝对误差 ~1e-9（CUDA 实测 1.9e-9），
    # 阈值不能低于 fp32 的分辨率。
    assert ux.abs().max().item() < 1e-8
    assert uy.abs().max().item() < 1e-8


def test_force_is_applied_only_along_given_axis(device):
    s = create_solver("D2Q9", (8, 8), nu=0.1, force=(0.0, 2e-5), device=device)
    s.initialize()
    s.step(100)
    _, ux, uy = s.macro()
    assert abs(uy.mean().item() - 100.5 * 2e-5) < 1e-6
    assert ux.abs().max().item() < 1e-6


def test_force_on_solid_node_does_not_explode(device):
    """回归测试：F/2 曾经被无条件加到固体节点上，rho=0 时 u 爆到 1e25。"""
    solid = torch.zeros((12, 12), dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    s = create_solver("D2Q9", (12, 12), solid, nu=0.1, force=(1e-3, 0.0), device=device)
    s.initialize()
    s.step(500)

    assert torch.isfinite(s.pdf).all()
    rho, ux, _ = s.macro()
    assert torch.isfinite(rho).all() and torch.isfinite(ux).all()
    # 固体节点必须仍然是 0，不能是 1e25 之类的数
    assert ux[solid].abs().max().item() == 0.0


def test_negative_nu_rejected(device):
    with pytest.raises(ValueError):
        create_solver("D2Q9", (8, 8), nu=-0.1, device=device)


def test_invalid_output_dtype_on_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS 不可用")
    with pytest.raises(ValueError):
        create_solver("D2Q9", (8, 8), nu=0.1, device="mps", compute_dtype=torch.float64)
