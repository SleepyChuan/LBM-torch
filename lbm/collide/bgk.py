"""BGK 碰撞 + Guo forcing。

    f_post_i = f_i - (1/tau) (f_i - feq_i) + S_i

    S_i = w_i (1 - 1/(2 tau)) [ (c_i - u)/cs^2 + (c_i·u)/cs^4 * c_i ] · F

cs^2 = 1/3，所以 1/cs^2 = 3、1/cs^4 = 9。

两个容易漏的地方：
  * S_i 前面的 (1 - 1/(2 tau))，漏掉后稳态速度系统性偏低且随 tau 变化；
  * 这里用的 u 必须是 macros 里带 F/2 修正的那个 u。
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..equilibrium import equilibrium_with_cu
from .base import CollisionOperator


class BGK(CollisionOperator):
    def __init__(
        self,
        tau: float,
        w: torch.Tensor,
        c: Sequence[torch.Tensor],
    ) -> None:
        if tau <= 0.5:
            raise ValueError(f"tau 必须 > 0.5，当前 {tau}")
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.force_factor = 1.0 - 0.5 / self.tau
        self.w = w
        self.c = tuple(c)

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: Sequence[torch.Tensor],
        force: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        feq, cu = equilibrium_with_cu(self.w, self.c, rho, u)
        f_post = f - self.omega * (f - feq)

        if force is not None:
            s = 0.0
            for d, cd in enumerate(self.c):
                # (c_d - u_d)/cs^2 + (c·u)/cs^4 * c_d
                s = s + (3.0 * (cd - u[d]) + 9.0 * cu * cd) * force[d]
            f_post = f_post + self.w * self.force_factor * s

        return f_post
