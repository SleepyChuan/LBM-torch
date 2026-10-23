from __future__ import annotations

import pytest
import torch

from lbm import create_solver
from lbm.collide import BGK, MRT
from lbm.collide.mrt import LL_M, ll_matrix
from lbm.equilibrium import equilibrium
from lbm.macros import macroscopic
from lbm.stencils import D2Q9
from lbm.stencils.base import make_tensors

# LL 原矩阵的列 -> 本项目方向表的列
_PERM = (0, 1, 4, 3, 2, 5, 8, 7, 6)


def _tensors(device: str):
    return make_tensors(D2Q9, torch.device(device), torch.float32)


def _random_f(device: str, shape=(6, 5), seed=2):
    g = torch.Generator().manual_seed(seed)
    return (0.1 + 0.05 * torch.rand((D2Q9.q, *shape), generator=g)).to(device)


def test_matrix_is_ll_with_permuted_columns():
    expected = tuple(tuple(row[j] for j in _PERM) for row in LL_M)
    assert ll_matrix(D2Q9.c) == expected
    # 恒等置换是 LL 的原始列序，必须和本项目不同，防止置换方向写反
    assert ll_matrix(D2Q9.c) != LL_M


def test_matrix_rows_are_the_documented_moments(device):
    st = _tensors(device)
    op = MRT(0.1, 3.0 / 16.0, st.w, st.c, D2Q9.c)
    f = _random_f(device)
    m = torch.tensordot(op.M, f, dims=1)

    c0, c1 = st.c
    assert torch.allclose(m[0], f.sum(0), atol=1e-6)
    assert torch.allclose(m[3], (f * c0).sum(0), atol=1e-6)
    assert torch.allclose(m[5], (f * c1).sum(0), atol=1e-6)
    assert torch.allclose(m[7], (f * (c0 * c0 - c1 * c1)).sum(0), atol=1e-6)
    assert torch.allclose(m[8], (f * c0 * c1).sum(0), atol=1e-6)

    # 逆矩阵在构造时算好，必须能精确还原 f
    assert torch.allclose(torch.tensordot(op.M_inv, m, dims=1), f, atol=1e-6)


def test_relaxation_vector_matches_spec(device):
    st = _tensors(device)
    nu, lam = 0.1, 3.0 / 16.0
    op = MRT(nu, lam, st.w, st.c, D2Q9.c)

    tau = 3.0 * nu + 0.5
    assert op.s_nu == pytest.approx(1.0 / tau)
    # 魔法参数只约束 ghost 模式
    assert op.s_q == pytest.approx(1.0 / (0.5 + lam / (tau - 0.5)))
    # 能量模式必须快速松弛
    assert op.s_e == pytest.approx(1.9)

    s = [float(v) for v in op.s.flatten()]
    assert s[0] == 0.0 and s[3] == 0.0 and s[5] == 0.0
    for i in (1, 2):
        assert s[i] == pytest.approx(op.s_e)
    for i in (4, 6):
        assert s[i] == pytest.approx(op.s_q)
    for i in (7, 8):
        assert s[i] == pytest.approx(op.s_nu)


def test_conserves_mass_and_momentum(device):
    st = _tensors(device)
    op = MRT(0.1, 3.0 / 16.0, st.w, st.c, D2Q9.c)

    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    f_post = op(f, rho, u, None)

    assert torch.allclose(f_post.sum(0), f.sum(0), atol=1e-6)
    for d in range(2):
        assert torch.allclose(
            (f_post * st.c[d]).sum(0), (f * st.c[d]).sum(0), atol=1e-6
        )


def test_equilibrium_is_fixed_point(device):
    st = _tensors(device)
    op = MRT(0.1, 3.0 / 16.0, st.w, st.c, D2Q9.c)

    shape = (6, 5)
    rho = torch.full(shape, 1.0, device=device)
    ux = torch.full(shape, 0.03, device=device)
    uy = torch.full(shape, -0.02, device=device)
    feq = equilibrium(st.w, st.c, rho, (ux, uy))

    assert torch.allclose(op(feq, rho, (ux, uy), None), feq, atol=1e-7)


def test_reduces_to_bgk_when_all_rates_equal(device):
    """把非守恒松弛率全设成 s_nu 时，MRT 与 BGK 只差守恒行的松弛率。

    平衡态附近动量矩的差值 ρu - Σf·c 只有舍入误差，所以两者数值一致。
    """
    st = _tensors(device)
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    mrt = MRT(nu, 3.0 / 16.0, st.w, st.c, D2Q9.c)
    for i in (1, 2, 4, 6):
        mrt.s[i] = mrt.s_nu
    bgk = BGK(tau, st.w, st.c)

    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    assert torch.allclose(mrt(f, rho, u, None), bgk(f, rho, u, None), atol=1e-6)


def test_force_growth_is_half_step_corrected(device):
    F, n = 1e-5, 100
    s = create_solver(
        "D2Q9", (8, 8), nu=0.1, force=(F, 0.0), device=device, collision="mrt"
    )
    s.initialize()
    s.step(n)

    _, ux, uy = s.macro()
    assert abs(ux.mean().item() - (n + 0.5) * F) < 1e-6
    assert uy.abs().max().item() < 1e-6


def test_single_step_velocity(device):
    F = 1e-4
    s = create_solver(
        "D2Q9", (8, 8), nu=0.1, force=(F, 0.0), device=device, collision="mrt"
    )
    s.initialize()
    s.step(1)
    assert abs(s.macro()[1].mean().item() - 1.5 * F) < 1e-7


def test_lambda_defaults_to_magic_parameter(device):
    s = create_solver("D2Q9", (8, 8), nu=0.1, device=device, collision="mrt")
    assert s.Lambda == pytest.approx(3.0 / 16.0)
    assert s.collision_info["collision"] == "mrt"
    assert s.collision_info["Lambda"] == pytest.approx(3.0 / 16.0)


def test_invalid_lambda_rejected(device):
    with pytest.raises(ValueError):
        create_solver(
            "D2Q9", (8, 8), nu=0.1, device=device, collision="mrt", Lambda=0.0
        )


def test_non_d2q9_velocity_set_rejected():
    with pytest.raises(ValueError):
        ll_matrix(((0, 0), (1, 0)))
    with pytest.raises(ValueError):
        ll_matrix(((0, 0),) * 9)


@pytest.mark.slow
def test_open_boundary_low_viscosity_stays_finite(device):
    """回归：karman 型绕流（Zou/He 入口出口，tau≈0.515）。

    能量模式的松弛率跟着魔法参数走时，出口会在 1000 步内把 rho 推到 inf；
    s_e=1.9 / s_q≈0.077 则能稳定跑下去。
    """
    nx, ny, d = 180, 120, 10
    u_inf, re = 0.05, 100.0
    nu = u_inf * d / re
    cx, cy, r = 2.5 * d, ny // 2, d / 2.0

    x = torch.arange(nx, dtype=torch.float32).view(-1, 1)
    y = torch.arange(ny, dtype=torch.float32).view(1, -1)
    solid = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r

    s = create_solver(
        "D2Q9",
        (nx, ny),
        solid,
        nu=nu,
        collision="mrt",
        boundary="inlet_outlet",
        u_inf=u_inf,
        rho_inf=1.0,
        device=device,
    )
    s.initialize(rho=1.0, u=(u_inf, 0.0))
    s.step(8000)

    rho, ux, uy = s.macro()
    fluid = ~solid.to(rho.device)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(ux).all() and torch.isfinite(uy).all()
    assert rho[fluid].max().item() < 1.5
