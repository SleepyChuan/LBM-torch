from __future__ import annotations

import pytest
import torch

from lbm import create_solver
from lbm.solvers.d2q9 import write_shifted


@pytest.mark.parametrize("sx", [-1, 0, 1])
@pytest.mark.parametrize("sy", [-1, 0, 1])
def test_write_shifted_matches_roll(device, sx, sy):
    """逐方向切片版必须和 torch.roll 的语义完全一致（周期回绕）。"""
    g = torch.Generator().manual_seed(0)
    src = torch.rand((5, 6), generator=g).to(device)
    dst = torch.zeros_like(src)
    write_shifted(dst, src, sx, sy)
    expected = torch.roll(src, shifts=(sx, sy), dims=(0, 1))
    assert torch.equal(dst, expected)


def test_uniform_equilibrium_is_fixed_point(device):
    """均匀平衡态在无固体周期域里是精确解。这一条能同时抓到
    opp 表、方向表、切片方向、双缓冲下标、feq 公式的错误。"""
    s = create_solver("D2Q9", (16, 16), nu=0.1, device=device)
    s.initialize(rho=1.0, u=(0.02, 0.0))
    ref = s.pdf[0].clone()
    s.step(1000)
    drift = (s.pdf[s.cur_idx] - ref).abs().max().item()
    assert drift < 1e-6


def test_static_fluid_with_solid_stays_static(device):
    """u=0 时 feq_i == feq_opp_i，反射分支与正常分支给出同一个值。"""
    solid = torch.zeros((16, 16), dtype=torch.bool)
    solid[4:8, 4:8] = True
    solid[10:12, 2:6] = True

    s = create_solver("D2Q9", (16, 16), solid, nu=0.1, device=device)
    s.initialize(rho=1.0, u=None)
    ref = s.pdf[0].clone()
    mass0 = s.diagnose().total_mass

    s.step(1000)

    assert (s.pdf[s.cur_idx] - ref).abs().max().item() < 1e-6
    assert abs(s.diagnose().total_mass - mass0) < 1e-7 * mass0


def test_all_solid_does_not_crash(device):
    s = create_solver(
        "D2Q9", (6, 6), torch.ones((6, 6), dtype=torch.bool), nu=0.1, device=device
    )
    s.initialize(rho=1.0, u=None)
    s.step(10)
    assert torch.isfinite(s.pdf).all()
    assert s.diagnose().total_mass == 0.0


def test_double_buffer_is_not_aliased(device):
    s = create_solver("D2Q9", (8, 8), nu=0.1, device=device)
    assert s.pdf[0].data_ptr() != s.pdf[1].data_ptr()


def test_cur_idx_flips_each_step(device):
    s = create_solver("D2Q9", (8, 8), nu=0.1, device=device)
    s.initialize()
    assert s.cur_idx == 0
    s.step(1)
    assert s.cur_idx == 1 and s.step_count == 1
    s.step(3)
    assert s.step_count == 4


def test_solid_nodes_hold_zero(device):
    solid = torch.zeros((10, 10), dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    s = create_solver("D2Q9", (10, 10), solid, nu=0.1, force=(1e-4, 0.0), device=device)
    s.initialize()
    s.step(200)
    assert s.pdf[s.cur_idx][:, solid].abs().max().item() == 0.0
