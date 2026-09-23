"""Airy 均衡与地壳厚度演化（《第一层完善》§3–§4）。

- :func:`airy_elevation`：由地壳厚度到地表高程的 Airy-Heiskanen 均衡公式，
  是构造层高程分配的核心（§3.2）。
- :func:`integrate_crust_thickness`：地壳增厚/减薄 + 重力松弛的显式时间积分
  （§4.4），用 Numba ``@njit(parallel=True, fastmath=True)`` 加速（§6.3）。
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ===== Airy 均衡 =====


def airy_elevation(
    crust_thickness: float | np.ndarray,
    c_ref: float = 30.0,
    rho_crust: float = 2800.0,
    rho_mantle: float = 3300.0,
) -> float | np.ndarray:
    """地壳厚度 → 地表高程 (m)（§3.2）。

    ``H = (rho_m - rho_c) / rho_m * (C - T) * 1000``。
    ``c_ref`` 为对应于海平面的参考地壳厚度 (km)，厚度与参考值的单位均为 km。
    """
    if np.any(np.asarray(crust_thickness) < 0.0):
        raise ValueError("地壳厚度不能为负")
    if rho_mantle <= rho_crust:
        raise ValueError("地幔密度必须大于地壳密度")
    return (rho_mantle - rho_crust) / rho_mantle * (crust_thickness - c_ref) * 1000.0


# ===== 地壳厚度时间积分（Numba 加速）=====


@njit(parallel=True, fastmath=True, cache=True)
def _integrate_kernel(
    crust: np.ndarray,
    thickening: np.ndarray,
    thinning: np.ndarray,
    c_eq: np.ndarray,
    kappa: float,
    dt: float,
    n_steps: int,
) -> None:
    """显式时间积分核心（原地更新，§6.3）。"""
    for _ in range(n_steps):
        for i in range(crust.shape[0]):
            for j in range(crust.shape[1]):
                crust[i, j] += (thickening[i, j] - thinning[i, j]) * dt
                crust[i, j] -= kappa * (crust[i, j] - c_eq[i, j]) * dt


def integrate_crust_thickness(
    crust: np.ndarray,
    thickening: np.ndarray | float,
    thinning: np.ndarray | float,
    c_eq: np.ndarray,
    kappa: float = 0.0,
    dt: float = 1.0,
    n_steps: int = 1,
) -> np.ndarray:
    """地壳厚度显式时间积分（§4.4），返回更新后的副本。

    每步更新：``C += (thickening - thinning)*dt - kappa*(C - C_eq)*dt``。
    ``thickening`` / ``thinning`` 为已含效率系数的体速率（单位与 ``crust`` 一致），
    ``dt`` 为时间步长，``n_steps`` 为步数（Numba 内核整体 JIT 编译）。
    """
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")
    if n_steps <= 0:
        raise ValueError(f"n_steps 必须为正，实际为 {n_steps}")
    crust = np.asarray(crust, dtype=np.float64).copy()
    thickening = np.broadcast_to(np.asarray(thickening, dtype=np.float64), crust.shape).copy()
    thinning = np.broadcast_to(np.asarray(thinning, dtype=np.float64), crust.shape).copy()
    c_eq = np.broadcast_to(np.asarray(c_eq, dtype=np.float64), crust.shape).copy()
    _integrate_kernel(crust, thickening, thinning, c_eq, float(kappa), float(dt), int(n_steps))
    return crust
