"""D3Q27 速度集。v1 只提供元数据，求解器未实现。

用张量积方式生成：每个轴取 (0, ±1)，权重按轴乘积
（0 -> 2/3，±1 -> 1/6），这样子集 D3Q19 / D3Q15 天然一致。
"""

from __future__ import annotations

import itertools

from .base import LatticeSpec

_ORDER = (0, 1, -1)  # 静止在中心，先正后负，便于人眼检查

_C = tuple(itertools.product(_ORDER, repeat=3))
_W_AXIS = {0: 2.0 / 3.0, 1: 1.0 / 6.0, -1: 1.0 / 6.0}

_W = tuple(_W_AXIS[cx] * _W_AXIS[cy] * _W_AXIS[cz] for cx, cy, cz in _C)

_INDEX = {v: i for i, v in enumerate(_C)}
_OPP = tuple(_INDEX[(-cx, -cy, -cz)] for cx, cy, cz in _C)

D3Q27 = LatticeSpec(name="D3Q27", c=_C, w=_W, opp=_OPP)
