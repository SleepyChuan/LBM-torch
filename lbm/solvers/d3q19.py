"""D3Q19 求解器 —— v1 未实现。

工厂能识别这个名字并正常构造实例，但任何实际计算都会抛
``NotImplementedError``。元数据（速度集）已经就位，接实现时
只需要补 ``_setup`` / ``_stream``。
"""

from __future__ import annotations

from ..stencils.d3q19 import D3Q19
from .base import BaseSolver


class D3Q19Solver(BaseSolver):
    spec = D3Q19
    model = "D3Q19"
    implemented = False
