from __future__ import annotations

import pytest
import torch

from lbm import create_solver
from lbm.collide import BGK, TRT
from lbm.equilibrium import equilibrium
from lbm.macros import macroscopic
from lbm.stencils import D2Q9
from lbm.stencils.base import make_tensors


def _tensors(device: str):
    return make_tensors(D2Q9, torch.device(device), torch.float32)


def _random_f(device: str, shape=(6, 5), seed=3):
    g = torch.Generator().manual_seed(seed)
    return (0.1 + 0.05 * torch.rand((D2Q9.q, *shape), generator=g)).to(device)


def test_trt_reduces_to_bgk_when_taus_equal(device):
    """Λ = (tau+ - 0.5)^2 时 tau- = tau+，TRT 必须和 BGK 完全一致。"""
    st = _tensors(device)
    tau = 0.8
    lam = (tau - 0.5) ** 2

    trt = TRT.from_lambda(tau, lam, st.w, st.c, st.opp)
    bgk = BGK(tau, st.w, st.c)
    assert trt.tau_minus == pytest.approx(tau, rel=1e-12)

    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    assert torch.allclose(trt(f, rho, u), bgk(f, rho, u), atol=1e-7)


def test_trt_conserves_mass_and_momentum(device):
    st = _tensors(device)
    trt = TRT.from_lambda(0.53, 3.0 / 16.0, st.w, st.c, st.opp)

    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    f_post = trt(f, rho, u, None)

    assert torch.allclose(f_post.sum(0), f.sum(0), atol=1e-6)
    for d in range(2):
        assert torch.allclose(
            (f_post * st.c[d]).sum(0), (f * st.c[d]).sum(0), atol=1e-6
        )


def test_trt_equilibrium_is_fixed_point(device):
    st = _tensors(device)
    trt = TRT.from_lambda(0.53, 3.0 / 16.0, st.w, st.c, st.opp)

    shape = (6, 5)
    rho = torch.full(shape, 1.0, device=device)
    ux = torch.full(shape, 0.03, device=device)
    uy = torch.full(shape, -0.02, device=device)
    feq = equilibrium(st.w, st.c, rho, (ux, uy))

    assert torch.allclose(trt(feq, rho, (ux, uy), None), feq, atol=1e-7)


def test_trt_force_keeps_momentum_growth(device):
    """力项的半格点修正必须配 tau-，否则 u 会偏离 (n+1/2)F。"""
    F, n = 1e-5, 100
    s = create_solver(
        "D2Q9", (8, 8), nu=0.1, force=(F, 0.0), device=device, Lambda=3.0 / 16.0
    )
    s.initialize()
    s.step(n)
    _, ux, _ = s.macro()
    assert abs(ux.mean().item() - (n + 0.5) * F) < 1e-6


def test_trt_single_step_velocity(device):
    F = 1e-4
    s = create_solver("D2Q9", (8, 8), nu=0.1, force=(F, 0.0), device=device)
    s.initialize()
    s.step(1)
    assert abs(s.macro()[1].mean().item() - 1.5 * F) < 1e-7


def test_low_viscosity_trt_is_stable_where_bgk_is_not(device):
    """tau+=0.515 这种 BGK 会炸的区间，TRT 必须撑住。"""
    nu = 0.005
    solid = torch.zeros((80, 40), dtype=torch.bool)
    solid[30:34, 18:22] = True

    s = create_solver(
        "D2Q9",
        (80, 40),
        solid,
        nu=nu,
        u_inf=0.05,
        rho_inf=1.0,
        boundary="inlet_outlet",
        device=device,
        Lambda=3.0 / 16.0,
    )
    assert s.collision.tau_minus > 5.0
    s.initialize(rho=1.0, u=(0.05, 0.0))
    s.step(2000)
    rho, ux, _ = s.macro()
    assert torch.isfinite(rho).all()
    assert rho[~solid].min().item() > 0.5


def test_invalid_lambda_rejected(device):
    with pytest.raises(ValueError):
        create_solver("D2Q9", (8, 8), nu=0.1, device=device, Lambda=0.0)


def test_bgk_with_lambda_warns(device):
    with pytest.warns(RuntimeWarning):
        s = create_solver(
            "D2Q9", (8, 8), nu=0.1, device=device, collision="bgk", Lambda=3.0 / 16.0
        )
    assert s.Lambda is None


def test_unknown_collision_rejected(device):
    with pytest.raises(ValueError):
        create_solver("D2Q9", (8, 8), nu=0.1, device=device, collision="lattice")
