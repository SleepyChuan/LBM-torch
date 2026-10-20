"""碰撞算子接口。

只要求"给我 f、rho、u、force，还我碰撞后的 f"。
TRT / MRT 将来实现在同一接口下，solver 不用改。
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch


class CollisionOperator(Protocol):
    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: Sequence[torch.Tensor],
        force: tuple[float, ...] | None = None,
    ) -> torch.Tensor: ...


__all__ = ["CollisionOperator"]
