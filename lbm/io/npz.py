"""npz 输出。v1 唯一真正实现的输出格式。

约定：

    metadata.json        模拟开始前写一次
    solid.npz            键 "solid"，bool，(Nx, Ny)
    pdf_<step:08d>.npz   键 "pdf"，(Q, Nx, Ny)，只存当前 buffer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_pdf(path: str | Path, pdf: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, pdf=pdf)
    return p


def read_pdf(path: str | Path) -> np.ndarray:
    with np.load(Path(path)) as data:
        return data["pdf"]


def write_solid(path: str | Path, solid: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, solid=solid)
    return p


def read_solid(path: str | Path) -> np.ndarray:
    with np.load(Path(path)) as data:
        return data["solid"]


def write_metadata(path: str | Path, meta: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
