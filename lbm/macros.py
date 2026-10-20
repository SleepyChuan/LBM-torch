"""从分布函数求宏观量。

速度定义只有一种，全局唯一：

    u = ( sum_i f_i * c_i + F/2 ) / rho

feq、Guo 源项、诊断都必须用这一个版本。

固体节点上的 pdf 恒为 0，所以 rho = 0。除法前把 rho 夹到一个极小正数，
避免 0/0 产生 NaN 之后污染整场。夹完之后分子也是 0，结果是 0，不影响物理。
"""

from __future__ import annotations

from typing import Sequence

import torch

_TINY = 1e-30


def macroscopic(
    f: torch.Tensor,
    c: Sequence[torch.Tensor],
    force: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """返回 (rho, (u0, u1[, u2]))。"""
    rho = f.sum(dim=0)
    inv_rho = 1.0 / rho.clamp_min(_TINY)

    # 固体节点上 rho 恒为 0，那里没有动量，F/2 也必须为 0。
    # 少了这个 scale，u = (0 + F/2) * 1e30 会爆到 1e25 量级，
    # 再进 feq 的 cu^2 项直接把 float32 撑成 inf。
    scale = (rho > 0).to(rho.dtype) if force is not None else None

    mom = []
    for d, cd in enumerate(c):
        m = (f * cd).sum(dim=0)
        if force is not None:
            m = m + 0.5 * force[d] * scale
        mom.append(m * inv_rho)

    return rho, tuple(mom)
