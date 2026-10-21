"""Solver 基类：与维数无关的公共部分。

这里放的是构造校验、双缓冲、时间推进循环、诊断、IO。
**不**放 streaming —— 那是每个模型各自实现的部分。

双缓冲是 ``pdf: (2, Q, *shape)``，``cur_idx`` 指示当前时刻的分布函数。
不做 ``pdf[0] = pdf[1]``，也切换引用；只翻转一个 int。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from ..diagnostics import Diagnostics
from ..equilibrium import equilibrium
from ..io.npz import write_metadata, write_pdf, write_solid
from ..macros import macroscopic
from ..stencils.base import LatticeSpec, StencilTensors, make_tensors

_SUPPORTED_DTYPES = (torch.float32, torch.float64)


class BaseSolver:
    spec: LatticeSpec
    model: str = "base"
    implemented: bool = True

    def __init__(
        self,
        shape: Sequence[int],
        solid: torch.Tensor | None = None,
        *,
        nu: float,
        Lambda: float | None = None,
        collision: str = "trt",
        force: Sequence[float] | torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        compute_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float16,
        compile: bool = False,
    ) -> None:
        # ---- 参数校验 -------------------------------------------------
        if nu <= 0.0:
            raise ValueError(f"nu 必须为正，当前 {nu}")
        self.nu = float(nu)
        self.tau = 3.0 * self.nu + 0.5
        if self.tau <= 0.5:
            raise ValueError(f"tau 必须 > 0.5，当前 {self.tau}")
        self.Lambda = None if Lambda is None else float(Lambda)

        kind = str(collision).lower()
        if kind not in ("bgk", "trt"):
            raise ValueError(f"collision 只支持 'bgk' / 'trt'，收到 {collision!r}")
        self.collision_kind = kind

        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != self.spec.dim:
            raise ValueError(
                f"{self.model} 需要 {self.spec.dim} 维 shape，收到 {self.shape}"
            )
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape 必须为正，收到 {self.shape}")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda 但当前环境不可用")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("指定了 mps 但当前环境不可用")

        if compute_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"compute_dtype 只支持 float32/float64，收到 {compute_dtype}"
            )
        if self.device.type == "mps" and compute_dtype == torch.float64:
            raise ValueError("MPS 不支持 float64，请改用 float32 或把设备换成 cpu")
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype

        self.spec.check()

        # ---- 常量与缓冲 -----------------------------------------------
        self._st: StencilTensors = make_tensors(
            self.spec, self.device, self.compute_dtype
        )
        self.pdf = torch.zeros(
            (2, self.spec.q, *self.shape),
            dtype=self.compute_dtype,
            device=self.device,
        )
        self.cur_idx = 0
        self._step_count = 0

        self.solid = self._prepare_solid(solid)
        self._has_solid = bool(self.solid.any().item())

        self.collision = None
        self._force: tuple[float, ...] | None = None
        self.set_force(force)

        self._init_desc: dict[str, Any] = {}

        self._setup()

        if compile:
            # 只编译单步。step() 的多步循环留在 python 层，
            # 这样 delta_step 每次不同也不会引起重新编译。
            self._advance = torch.compile(self._advance)

    # ------------------------------------------------------------------
    # 子类钩子
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """子类在这里预计算 mask、构造碰撞算子等。"""

    def _stream(self, f_post: torch.Tensor, dst: torch.Tensor) -> None:
        raise NotImplementedError

    def _apply_open_boundary(self, dst: torch.Tensor) -> None:
        """Zou/He 入口出口的挂载点。periodic 模式下为空实现。"""

    def _extra_meta(self) -> dict[str, Any]:
        """子类往 metadata.json 里加的额外字段。"""
        return {}

    # ------------------------------------------------------------------
    # 构造辅助
    # ------------------------------------------------------------------
    def _prepare_solid(self, solid: torch.Tensor | None) -> torch.Tensor:
        if solid is None:
            return torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        t = torch.as_tensor(solid, dtype=torch.bool, device=self.device)
        if tuple(t.shape) != self.shape:
            raise ValueError(f"solid 形状应为 {self.shape}，收到 {tuple(t.shape)}")
        return t.contiguous()

    def _check_implemented(self) -> None:
        if not self.implemented:
            raise NotImplementedError(
                f"{self.model} 求解器尚未实现，v1 只提供 D2Q9 的可用实现"
            )

    def _as_field(self, value: Any, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.to(device=self.device, dtype=self.compute_dtype)
            if tuple(t.shape) != self.shape:
                raise ValueError(f"{name} 形状应为 {self.shape}，收到 {tuple(t.shape)}")
            return t
        if isinstance(value, (int, float)):
            return torch.full(
                self.shape, float(value), dtype=self.compute_dtype, device=self.device
            )
        raise TypeError(f"{name} 只支持标量或 tensor，收到 {type(value)}")

    def _as_velocity(self, u: Any) -> list[torch.Tensor]:
        dim = self.spec.dim
        if u is None:
            return [
                torch.zeros(self.shape, dtype=self.compute_dtype, device=self.device)
                for _ in range(dim)
            ]
        if isinstance(u, torch.Tensor):
            if u.shape[0] != dim or tuple(u.shape[1:]) != self.shape:
                raise ValueError(
                    f"速度场形状应为 ({dim}, *{self.shape})，收到 {tuple(u.shape)}"
                )
            return [
                u[d].to(device=self.device, dtype=self.compute_dtype)
                for d in range(dim)
            ]
        if isinstance(u, (tuple, list)):
            if len(u) != dim:
                raise ValueError(f"速度分量个数应为 {dim}，收到 {len(u)}")
            return [self._as_field(c, f"u[{d}]") for d, c in enumerate(u)]
        raise TypeError(f"u 不支持的类型 {type(u)}")

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initialize(self, rho: Any = 1.0, u: Any = None) -> None:
        """用平衡态初始化 ``pdf[0]``，并把 ``pdf[1]`` 清零；固体节点置 0。

        会把 ``step_count`` 归零、``cur_idx`` 归零。
        """
        self._check_implemented()
        rho_f = self._as_field(rho, "rho")
        u_comps = self._as_velocity(u)

        feq = equilibrium(self._st.w, self._st.c, rho_f, u_comps)
        if self._has_solid:
            feq = torch.where(self.solid, torch.zeros_like(feq), feq)

        self.pdf[0].copy_(feq)
        self.pdf[1].zero_()
        self.cur_idx = 0
        self._step_count = 0

        self._init_desc = {"rho": _describe(rho), "u": _describe(u)}

    def set_force(self, force: Sequence[float] | torch.Tensor | None) -> None:
        if force is None:
            self._force = None
            return
        if isinstance(force, torch.Tensor):
            vals = tuple(float(x) for x in force.detach().to("cpu").tolist())
        else:
            vals = tuple(float(x) for x in force)
        if len(vals) != self.spec.dim:
            raise ValueError(f"force 分量个数应为 {self.spec.dim}，收到 {len(vals)}")
        self._force = vals

    @property
    def force(self) -> tuple[float, ...] | None:
        return self._force

    # ------------------------------------------------------------------
    # 时间推进
    # ------------------------------------------------------------------
    def _advance(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """单步核心。compile=True 时被编译的就是这个函数。

        参数是"当前"和"下一个"两块 buffer 的视图。两块视图的
        shape/dtype/stride 完全一样，所以翻转 cur_idx 不会触发重新编译，
        也不需要把 cur_idx 引进图里。
        """
        rho, u = macroscopic(src, self._st.c, self._force)
        f_post = self.collision(src, rho, u, self._force)
        self._stream(f_post, dst)
        self._apply_open_boundary(dst)

    def step(self, delta_step: int = 1) -> None:
        if delta_step < 0:
            raise ValueError("delta_step 不能为负")
        self._check_implemented()
        with torch.inference_mode():
            for _ in range(delta_step):
                src = self.pdf[self.cur_idx]
                dst = self.pdf[1 - self.cur_idx]
                self._advance(src, dst)
                self.cur_idx = 1 - self.cur_idx
                self._step_count += 1

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def step_count(self) -> int:
        return self._step_count

    def macro(self) -> tuple[torch.Tensor, ...]:
        """返回 (rho, u0, u1[, u2])，完整场，仍在 device 上。

        固体节点上的值没有物理意义（rho 为 0），需要时自己 mask 掉。
        """
        self._check_implemented()
        rho, u = macroscopic(self.pdf[self.cur_idx], self._st.c, self._force)
        return (rho, *u)

    def diagnose(self) -> Diagnostics:
        rho, u = macroscopic(self.pdf[self.cur_idx], self._st.c, self._force)
        fluid = ~self.solid
        rho_f = rho[fluid]
        if rho_f.numel() == 0:
            return Diagnostics(self._step_count, 0.0, 0.0, math.nan, math.nan)

        speed2 = u[0] * u[0]
        for ui in u[1:]:
            speed2 = speed2 + ui * ui
        speed_f = torch.sqrt(speed2)[fluid]

        ok = bool(torch.isfinite(rho_f).all().item()) and bool(
            torch.isfinite(speed_f).all().item()
        )
        if not ok:
            return Diagnostics(self._step_count, math.nan, math.nan, math.nan, math.nan)

        return Diagnostics(
            step=self._step_count,
            total_mass=float(rho_f.sum().item()),
            max_velocity=float(speed_f.max().item()),
            rho_min=float(rho_f.min().item()),
            rho_max=float(rho_f.max().item()),
        )

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def save_npz(self, directory: str | Path, prefix: str = "pdf") -> Path:
        """保存当前 buffer 的 PDF，文件名带时间步。"""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arr = self.pdf[self.cur_idx].detach().cpu().to(self.output_dtype).numpy()
        return write_pdf(d / f"{prefix}_{self._step_count:08d}.npz", arr)

    def dump_setup(
        self,
        directory: str | Path,
        *,
        total_steps: int,
        save_interval: int,
        extra: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """模拟开始前调用一次，写 metadata.json 与 solid.npz。"""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "model": self.model,
            "shape": list(self.shape),
            "nu": self.nu,
            "tau": self.tau,
            "Lambda": self.Lambda,
            "force": list(self._force) if self._force is not None else None,
            "device": str(self.device),
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "output_dtype": str(self.output_dtype).replace("torch.", ""),
            "total_steps": int(total_steps),
            "save_interval": int(save_interval),
            "initial_rho": self._init_desc.get("rho"),
            "initial_u": self._init_desc.get("u"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "lbm_version": _version(),
        }
        meta.update(self._extra_meta())
        if extra:
            meta.update(extra)

        meta_path = write_metadata(d / "metadata.json", meta)
        solid_path = write_solid(d / "solid.npz", self.solid.detach().cpu().numpy())
        return meta_path, solid_path


def _describe(value: Any) -> str:
    if value is None:
        return "default"
    if isinstance(value, torch.Tensor):
        return "field"
    if isinstance(value, (tuple, list)):
        return "field" if any(isinstance(v, torch.Tensor) for v in value) else "scalar"
    if isinstance(value, (int, float)):
        return f"scalar({float(value):g})"
    return "field"


def _version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"
