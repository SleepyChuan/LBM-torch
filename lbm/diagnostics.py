"""诊断量。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnostics:
    step: int
    total_mass: float
    max_velocity: float
    rho_min: float
    rho_max: float

    def __str__(self) -> str:  # 方便直接 print
        return (
            f"step={self.step} mass={self.total_mass:.8e} "
            f"|u|max={self.max_velocity:.6e} "
            f"rho=[{self.rho_min:.6e}, {self.rho_max:.6e}]"
        )
