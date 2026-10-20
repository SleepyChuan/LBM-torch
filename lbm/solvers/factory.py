"""求解器工厂。"""

from __future__ import annotations

from typing import Sequence

import torch

from .base import BaseSolver
from .d2q9 import D2Q9Solver
from .d3q19 import D3Q19Solver
from .d3q27 import D3Q27Solver

_REGISTRY: dict[str, type[BaseSolver]] = {
    "D2Q9": D2Q9Solver,
    "D3Q19": D3Q19Solver,
    "D3Q27": D3Q27Solver,
}


def create_solver(
    model: str,
    shape: Sequence[int],
    solid: torch.Tensor | None = None,
    **kwargs,
) -> BaseSolver:
    """按模型名创建 solver。

    未知模型名抛 ``ValueError``；已知但未实现的模型会正常返回实例，
    直到调用 ``step`` / ``initialize`` 时才抛 ``NotImplementedError``。
    """
    key = str(model).upper().replace(" ", "").replace("_", "")
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"未知模型 {model!r}，可选：{', '.join(sorted(_REGISTRY))}")
    return cls(shape=shape, solid=solid, **kwargs)
