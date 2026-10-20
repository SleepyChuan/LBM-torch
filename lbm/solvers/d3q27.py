"""D3Q27 求解器 —— v1 未实现。"""

from __future__ import annotations

from ..stencils.d3q27 import D3Q27
from .base import BaseSolver


class D3Q27Solver(BaseSolver):
    spec = D3Q27
    model = "D3Q27"
    implemented = False
