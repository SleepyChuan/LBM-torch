"""格子速度集（stencil）的元数据与 device 张量。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class LatticeSpec:
    """纯 python 的速度集描述，不绑定 device。"""

    name: str
    c: tuple[tuple[int, ...], ...]
    w: tuple[float, ...]
    opp: tuple[int, ...]

    @property
    def q(self) -> int:
        return len(self.c)

    @property
    def dim(self) -> int:
        return len(self.c[0])

    def check(self) -> None:
        """元数据自洽性。构造 solver 时调用一次。"""
        q, dim = self.q, self.dim
        if len(self.w) != q or len(self.opp) != q:
            raise ValueError(f"{self.name}: c/w/opp 长度不一致")
        if any(len(ci) != dim for ci in self.c):
            raise ValueError(f"{self.name}: 速度分量维数不一致")

        if abs(sum(self.w) - 1.0) > 1e-12:
            raise ValueError(f"{self.name}: sum(w) != 1")

        for i in range(q):
            j = self.opp[i]
            if self.opp[j] != i:
                raise ValueError(f"{self.name}: opp 不是对合，i={i}")
            if tuple(-v for v in self.c[i]) != self.c[j]:
                raise ValueError(f"{self.name}: c[{i}] != -c[{j}]")

        for d in range(dim):
            m1 = sum(wi * ci[d] for wi, ci in zip(self.w, self.c))
            if abs(m1) > 1e-12:
                raise ValueError(f"{self.name}: 一阶矩不为零，d={d}")

        for a in range(dim):
            for b in range(dim):
                m2 = sum(wi * ci[a] * ci[b] for wi, ci in zip(self.w, self.c))
                target = CS2 if a == b else 0.0
                if abs(m2 - target) > 1e-12:
                    raise ValueError(f"{self.name}: 二阶矩不等于 cs^2*I，({a},{b})")


@dataclass(frozen=True)
class StencilTensors:
    """放到 device 上的常量。形状按 (Q, 1, 1) 预展开，方便和 (Nx, Ny) 广播。"""

    w: torch.Tensor
    c: tuple[torch.Tensor, ...]
    opp: torch.Tensor
    q: int
    dim: int


def make_tensors(
    spec: LatticeSpec, device: torch.device, dtype: torch.dtype
) -> StencilTensors:
    c_cols = []
    for d in range(spec.dim):
        col = torch.tensor([ci[d] for ci in spec.c], dtype=dtype, device=device)
        c_cols.append(col.view(-1, 1, 1))
    w = torch.tensor(spec.w, dtype=dtype, device=device).view(-1, 1, 1)
    opp = torch.tensor(spec.opp, dtype=torch.long, device=device)
    return StencilTensors(w=w, c=tuple(c_cols), opp=opp, q=spec.q, dim=spec.dim)
