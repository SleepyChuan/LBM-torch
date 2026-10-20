from __future__ import annotations

import pytest
import torch


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.fixture(params=_available_devices())
def device(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def progress_out():
    """tqdm 往哪儿写。

    pytest 默认在 fd 层捕获 stderr，连 ``sys.__stderr__`` 都会被吞掉。
    直接开 /dev/tty 才能真正看到进度条；拿不到（CI、无 tty）就返回 None，
    进度条静默，不影响测试结果。
    """
    try:
        f = open("/dev/tty", "w", encoding="utf-8")
    except OSError:
        yield None
        return
    try:
        yield f
    finally:
        f.close()
