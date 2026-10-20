"""TRT（Two-Relaxation-Time）碰撞 + Guo forcing。

把分布函数按 stencil 的反射对称性拆成对称/反对称两部分：

    f_i^± = (f_i ± f_ī)/2
    feq_i^± = (feq_i ± feq_ī)/2

分别用两个松弛率碰撞：

    f_i^post = f_i - ω+ (f_i^+ - feq_i^+) - ω- (f_i^- - feq_i^-)

运动粘度只由 τ+ 决定，τ- 是自由参数，用 magic parameter 约束：

    ν  = cs² (τ+ - 0.5) = (τ+ - 0.5)/3
    Λ  = (τ+ - 0.5)(τ- - 0.5)
    τ- = 0.5 + Λ / (τ+ - 0.5)

τ+ → 0.5（低粘度）时 τ- 反而变大，把"幽灵模态"压住——这正是 BGK
在 τ≈0.52 附近发散、而 TRT 能撑住的原因。

**力项的半格点修正因子用 τ- 而不是 τ+。**
动量由反对称部分携带，碰撞带来的动量变化是 ω- F/2，Guo 源项必须给出
(1 - ω-/2) F 才能让总和回到 F。用 τ+ 会留下 (1/τ- - 1/τ+)F/2 的系统偏差，
表现为"力驱动算例的速度对不上 (n + 1/2)F"。τ- = τ+ 时退化为 BGK。
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..equilibrium import equilibrium_with_cu
from .base import CollisionOperator


class TRT(CollisionOperator):
    def __init__(
        self,
        tau_plus: float,
        tau_minus: float,
        w: torch.Tensor,
        c: Sequence[torch.Tensor],
        opp: torch.Tensor,
    ) -> None:
        if tau_plus <= 0.5:
            raise ValueError(f"tau_plus 必须 > 0.5，当前 {tau_plus}")
        if tau_minus <= 0.5:
            raise ValueError(f"tau_minus 必须 > 0.5，当前 {tau_minus}")

        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.omega_plus = 1.0 / self.tau_plus
        self.omega_minus = 1.0 / self.tau_minus
        self.force_factor = 1.0 - 0.5 / self.tau_minus
        self.w = w
        self.c = tuple(c)
        self.opp = opp

    @classmethod
    def from_lambda(
        cls,
        tau_plus: float,
        Lambda: float,
        w: torch.Tensor,
        c: Sequence[torch.Tensor],
        opp: torch.Tensor,
    ) -> "TRT":
        if Lambda <= 0.0:
            raise ValueError(f"Lambda 必须为正，当前 {Lambda}")
        if tau_plus <= 0.5:
            raise ValueError(f"tau_plus 必须 > 0.5，当前 {tau_plus}")
        tau_minus = 0.5 + Lambda / (tau_plus - 0.5)
        return cls(tau_plus, tau_minus, w, c, opp)

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: Sequence[torch.Tensor],
        force: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        feq, cu = equilibrium_with_cu(self.w, self.c, rho, u)

        f_opp = f[self.opp]
        feq_opp = feq[self.opp]

        f_plus = 0.5 * (f + f_opp)
        f_minus = 0.5 * (f - f_opp)
        feq_plus = 0.5 * (feq + feq_opp)
        feq_minus = 0.5 * (feq - feq_opp)

        f_post = (
            f
            - self.omega_plus * (f_plus - feq_plus)
            - self.omega_minus * (f_minus - feq_minus)
        )

        if force is not None:
            s = 0.0
            for d, cd in enumerate(self.c):
                # (c_d - u_d)/cs^2 + (c·u)/cs^4 * c_d
                s = s + (3.0 * (cd - u[d]) + 9.0 * cu * cd) * force[d]
            f_post = f_post + self.w * self.force_factor * s

        return f_post
