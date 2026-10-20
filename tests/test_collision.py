from __future__ import annotations

import pytest
import torch

from lbm.collide import BGK
from lbm.equilibrium import equilibrium
from lbm.macros import macroscopic
from lbm.stencils import D2Q9
from lbm.stencils.base import make_tensors


def _random_f(device: str, shape=(6, 5), seed=1):
    g = torch.Generator().manual_seed(seed)
    return (0.1 + 0.05 * torch.rand((D2Q9.q, *shape), generator=g)).to(device)


def test_collision_conserves_mass_and_momentum(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    bgk = BGK(0.8, st.w, st.c)

    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    f_post = bgk(f, rho, u, None)

    assert torch.allclose(f_post.sum(0), f.sum(0), atol=1e-6)
    for d in range(2):
        assert torch.allclose(
            (f_post * st.c[d]).sum(0), (f * st.c[d]).sum(0), atol=1e-6
        )


def test_equilibrium_is_fixed_point(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    bgk = BGK(0.7, st.w, st.c)

    shape = (6, 5)
    rho = torch.full(shape, 1.0, device=device)
    ux = torch.full(shape, 0.03, device=device)
    uy = torch.full(shape, -0.02, device=device)
    feq = equilibrium(st.w, st.c, rho, (ux, uy))

    f_post = bgk(feq, rho, (ux, uy), None)
    assert torch.allclose(f_post, feq, atol=1e-7)


@pytest.mark.parametrize("tau", [0.55, 0.8, 1.5])
def test_collision_obeys_bgk_relation(device, tau):
    """f_post - feq 必须等于 (1 - 1/tau) * (f - feq)。

    注意 (1 - 1/tau) 在 tau=1 处过零，所以「tau 越小松弛越快」只在
    指数衰减的意义上成立，不能拿 |f_post - feq| 直接比大小。
    """
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    bgk = BGK(tau, st.w, st.c)

    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    feq = equilibrium(st.w, st.c, rho, u)

    f_post = bgk(f, rho, u, None)
    expected = f - (1.0 / tau) * (f - feq)
    assert torch.allclose(f_post, expected, atol=1e-6)


def test_zero_force_matches_no_force(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    f = _random_f(device)
    rho, u = macroscopic(f, st.c)
    bgk = BGK(0.8, st.w, st.c)
    assert torch.allclose(bgk(f, rho, u, None), bgk(f, rho, u, (0.0, 0.0)), atol=1e-7)


def test_invalid_tau_rejected():
    with pytest.raises(ValueError):
        BGK(0.5, torch.ones(9, 1, 1), (torch.ones(9, 1, 1), torch.ones(9, 1, 1)))
