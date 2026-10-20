"""进度条。tqdm 是可选依赖，没装就退化成一个空壳，不影响任何功能。

测试里用的时候注意 pytest 默认会捕获 stderr，进度条看不见。
要把 ``file=sys.__stderr__`` 传进来绕过捕获。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:  # pragma: no cover - 取决于环境
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class _NullBar:
    """tqdm 缺失时的替身，只实现本项目用到的那几个方法。"""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, s: str = "") -> None:
        pass

    def write(self, s: str) -> None:
        print(s)

    def close(self) -> None:
        pass


@contextmanager
def progress(total: int, desc: str | None = None, **kwargs: Any) -> Iterator[Any]:
    if _tqdm is None or total <= 0:
        yield _NullBar()
        return

    kwargs.setdefault("unit", "step")
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.2)
    kwargs.setdefault("leave", True)

    bar = _tqdm(total=total, desc=desc, **kwargs)
    try:
        yield bar
    finally:
        bar.close()


def run_steps(
    solver,
    total_steps: int,
    *,
    desc: str | None = None,
    chunk: int = 500,
    **kwargs: Any,
) -> None:
    """把 ``total_steps`` 拆成若干块推进，顺带刷新进度条。

    分块只是为了让进度条有机会刷新，结果和一次性 ``solver.step(n)`` 完全一样。
    """
    if total_steps <= 0:
        return

    chunk = max(1, min(int(chunk), int(total_steps)))
    with progress(total_steps, desc=desc, **kwargs) as bar:
        done = 0
        while done < total_steps:
            n = min(chunk, total_steps - done)
            solver.step(n)
            done += n
            bar.update(n)
