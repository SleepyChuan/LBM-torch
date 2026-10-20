"""VTK 输出 —— v1 未实现。

将来用来把宏观量（速度、涡量）丢进 ParaView。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_vtk(path: str | Path, rho: np.ndarray, u: np.ndarray, **kwargs) -> Path:
    raise NotImplementedError("VTK 输出尚未实现，v1 请使用 save_npz")
