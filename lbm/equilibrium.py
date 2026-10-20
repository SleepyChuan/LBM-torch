"""平衡态分布函数。

对 2D 与 3D 都成立：``c`` 是每个空间方向一个 ``(Q,1,1)`` 张量，
``u`` 是同长度的 ``(Nx,Ny[,Nz])`` 张量元组。维数在打包时定死，
热循环里没有额外分支。
"""

from __future__ import annotations

from typing import Sequence

import torch


def equilibrium_with_cu(
    w: torch.Tensor,
    c: Sequence[torch.Tensor],
    rho: torch.Tensor,
    u: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (feq, c_i·u)。

    ``cu`` 一起返回是因为 Guo 源项还要用它，避免在碰撞里重算一遍。
    """
    cu = c[0] * u[0]
    for ci, ui in zip(c[1:], u[1:]):
        cu = cu + ci * ui

    usq = u[0] * u[0]
    for ui in u[1:]:
        usq = usq + ui * ui

    feq = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq, cu


def equilibrium(
    w: torch.Tensor,
    c: Sequence[torch.Tensor],
    rho: torch.Tensor,
    u: Sequence[torch.Tensor],
) -> torch.Tensor:
    feq, _ = equilibrium_with_cu(w, c, rho, u)
    return feq
