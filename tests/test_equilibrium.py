from __future__ import annotations

import pytest
import torch

from lbm.equilibrium import equilibrium
from lbm.stencils import D2Q9
from lbm.stencils.base import CS2, make_tensors


def _fields(device: str, dtype=torch.float32, shape=(7, 5)):
    g = torch.Generator().manual_seed(0)
    rho = 1.0 + 0.1 * torch.rand(shape, generator=g)
    ux = 0.2 * (torch.rand(shape, generator=g) - 0.5)
    uy = 0.2 * (torch.rand(shape, generator=g) - 0.5)
    return (
        rho.to(device=device, dtype=dtype),
        ux.to(device=device, dtype=dtype),
        uy.to(device=device, dtype=dtype),
    )


def test_zeroth_moment(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    rho, ux, uy = _fields(device)
    feq = equilibrium(st.w, st.c, rho, (ux, uy))
    assert torch.allclose(feq.sum(dim=0), rho, atol=1e-5)


def test_first_moment(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    rho, ux, uy = _fields(device)
    feq = equilibrium(st.w, st.c, rho, (ux, uy))
    assert torch.allclose((feq * st.c[0]).sum(dim=0), rho * ux, atol=1e-6)
    assert torch.allclose((feq * st.c[1]).sum(dim=0), rho * uy, atol=1e-6)


def test_second_moment(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    rho, ux, uy = _fields(device)
    feq = equilibrium(st.w, st.c, rho, (ux, uy))

    m11 = (feq * st.c[0] * st.c[0]).sum(dim=0)
    m22 = (feq * st.c[1] * st.c[1]).sum(dim=0)
    m12 = (feq * st.c[0] * st.c[1]).sum(dim=0)

    assert torch.allclose(m11, rho * (CS2 + ux * ux), atol=1e-6)
    assert torch.allclose(m22, rho * (CS2 + uy * uy), atol=1e-6)
    assert torch.allclose(m12, rho * ux * uy, atol=1e-6)


def test_symmetry(device):
    """feq_i(u) == feq_opp_i(-u)。"""
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    rho, ux, uy = _fields(device)
    f_pos = equilibrium(st.w, st.c, rho, (ux, uy))
    f_neg = equilibrium(st.w, st.c, rho, (-ux, -uy))
    assert torch.allclose(f_pos, f_neg[st.opp], atol=1e-7)


def test_uniform_rest_equal_to_weights(device):
    st = make_tensors(D2Q9, torch.device(device), torch.float32)
    rho = torch.ones((3, 4), device=device)
    zero = torch.zeros((3, 4), device=device)
    feq = equilibrium(st.w, st.c, rho, (zero, zero))
    expected = st.w.expand_as(feq)
    assert torch.allclose(feq, expected, atol=1e-7)
