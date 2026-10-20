"""LBM on PyTorch —— 格子单位 D2Q9。

公开入口只有 ``create_solver`` 和 ``Diagnostics``。
"""

from .diagnostics import Diagnostics
from .progress import progress, run_steps
from .solvers.factory import create_solver

__version__ = "0.1.0"

__all__ = ["create_solver", "Diagnostics", "progress", "run_steps", "__version__"]
