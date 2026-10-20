"""HDF5 输出 —— v1 未实现。

签名和 npz 版保持一致，将来替换时调用方不用改。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_pdf_h5(path: str | Path, pdf: np.ndarray, **kwargs) -> Path:
    raise NotImplementedError("h5 输出尚未实现，v1 请使用 save_npz")
