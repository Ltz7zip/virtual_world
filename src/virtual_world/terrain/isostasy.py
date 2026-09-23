"""Airy 均衡与地壳厚度演化（《第一层完善》§3–§4）。

- :func:`airy_elevation`：由地壳厚度到地表高程的 Airy-Heiskanen 均衡公式，
  是构造层高程分配的核心（§3.2）。
- :func:`integrate_crust_thickness`：地壳增厚/减薄 + 重力松弛的显式时间积分
  （§4.4），用 Numba ``@njit(parallel=True, fastmath=True)`` 加速（§6.3）。
- §4.2 质量守恒：地壳缩短量必须等于增厚量（``L_0*C_0 = L*C``），缩短因子
  ``beta = L_0/L = C/C_0``——见 :func:`shortening_factor` / :func:`mass_conserving_thickness`。
- §4.3 造山带高宽比：``H/W ∝ (d_rho*g*C_0)/(eta_eff*|v_rel|)``——见
  :func:`orogen_aspect_ratio` / :func:`ridge_half_width_km`。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..core.constants import DENSITY_CRUST, DENSITY_MANTLE, EARTH_GRAVITY

# ===== 物理常数与标定（§3.1、§4.1、§4.3）=====

#: 地壳密度 (kg/m^3)（§3.1）
RHO_CRUST = DENSITY_CRUST
#: 地幔密度 (kg/m^3)（§3.1）
RHO_MANTLE = DENSITY_MANTLE
#: 增厚效率系数 eta_eff（§4.1 的 0.1–0.3 区间中值）
DEFAULT_THICKENING_EFFICIENCY = 0.2

# ===== §4.3 造山带高宽比的参考标定 =====
#: 参考造山带的高宽比：取喜马拉雅量级（H ≈ 5 km、半宽 W ≈ 100 km ⇒ H/W ≈ 0.05）。
REFERENCE_ASPECT_RATIO = 0.05
#: 参考条件（方案 §3.1/§4.1 的典型值），用于把 §4.3 的比例式无量纲化
REFERENCE_DELTA_RHO = RHO_MANTLE - RHO_CRUST
REFERENCE_C0_KM = 36.0
REFERENCE_V_REL = 1.0

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
    """显式时间积分核心（原地更新，§6.3）。

    展平后按 ``prange`` 并行：``parallel=True`` 只有配合 ``prange`` 才真正启用
    多线程（§6.3 要求）。各网格单元互不依赖，因此并行不改变数值结果。
    用 ``size``/``reshape(-1)`` 而非二维下标，因此对立方球 ``(6, n, n)`` 等任意
    形状同样适用（方案 §1.5 的地形网格互转）。
    """
    n_cells = crust.size
    flat_crust = crust.reshape(n_cells)
    flat_thickening = thickening.reshape(n_cells)
    flat_thinning = thinning.reshape(n_cells)
    flat_eq = c_eq.reshape(n_cells)
    for _ in range(n_steps):
        # numba 的 prange 无类型存根，mypy 无法解析
        for k in prange(n_cells):  # type: ignore[no-untyped-call, attr-defined]
            flat_crust[k] += (flat_thickening[k] - flat_thinning[k]) * dt
            flat_crust[k] -= kappa * (flat_crust[k] - flat_eq[k]) * dt


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


# ===== §4.2 质量守恒与缩短因子 =====


def shortening_factor(crust_thickness_km: float, c0_km: float) -> float:
    """缩短因子 ``beta = C/C_0``（§4.2）。

    质量守恒 ``L_0*C_0 = L*C`` 给出 ``beta = L_0/L = C/C_0``，即**增厚多少倍
    就必须缩短多少倍**。方案给出的安第斯中部基准值为 ``beta_0 = 2.34 +- 0.13``。
    """
    if c0_km <= 0.0:
        raise ValueError(f"初始地壳厚度必须为正，实际为 {c0_km}")
    if crust_thickness_km < c0_km:
        raise ValueError(
            f"增厚后的地壳厚度 {crust_thickness_km} 不得小于初始厚度 {c0_km}（beta >= 1）"
        )
    return float(crust_thickness_km / c0_km)


def crust_from_shortening(c0_km: float, beta: float) -> float:
    """由缩短因子反推地壳厚度 ``C = C_0*beta``（§4.2）。"""
    if c0_km <= 0.0:
        raise ValueError(f"初始地壳厚度必须为正，实际为 {c0_km}")
    if beta < 1.0:
        raise ValueError(f"缩短因子 beta 必须 >= 1（缩短而非拉张），实际为 {beta}")
    return float(c0_km * beta)


def shortening_distance_km(beta: float, plate_length_km: float) -> float:
    """缩短量 ``d_L = L_0*(1 - 1/beta)`` (km)（§4.2）。"""
    if beta < 1.0:
        raise ValueError(f"缩短因子 beta 必须 >= 1，实际为 {beta}")
    if plate_length_km <= 0.0:
        raise ValueError(f"板块长度必须为正，实际为 {plate_length_km}")
    return float(plate_length_km * (1.0 - 1.0 / beta))


def max_shortening_factor(plate_length_km: float, convergence_km: float) -> float:
    """给定汇聚距离，质量守恒允许的最大缩短因子 ``beta = 1/(1 - d_L/L_0)``（§4.2）。"""
    if plate_length_km <= 0.0:
        raise ValueError(f"板块长度必须为正，实际为 {plate_length_km}")
    if convergence_km < 0.0:
        raise ValueError(f"汇聚距离不能为负，实际为 {convergence_km}")
    if convergence_km >= plate_length_km:
        raise ValueError(
            f"汇聚距离 {convergence_km} km 不得达到或超过板块长度 {plate_length_km} km（板块已被吃光）"
        )
    return float(1.0 / (1.0 - convergence_km / plate_length_km))


def mass_conserving_thickness(
    c0_km: float,
    plate_length_km: float,
    convergence_km: float,
) -> tuple[float, float]:
    """质量守恒约束下可达到的 (地壳厚度 km, 缩短因子 beta)（§4.2）。

    返回 ``(C, beta)``，其中 ``C = C_0*beta``、``beta = 1/(1 - d_L/L_0)``。
    这是碰撞带增厚的**上限**：超过该厚度的地壳无法由既有缩短量支撑，
    建模时应以 :func:`max_shortening_factor` 先行截断。
    """
    beta = max_shortening_factor(plate_length_km, convergence_km)
    return crust_from_shortening(c0_km, beta), beta


# ===== §4.3 造山带高宽比 =====


def orogen_aspect_ratio(
    delta_rho: float,
    c0_km: float,
    v_rel: float,
    *,
    g: float = EARTH_GRAVITY,
    eta_eff: float = DEFAULT_THICKENING_EFFICIENCY,
) -> float:
    """造山带高宽比 ``H/W``（§4.3，方案比例式的无量纲标定形式）。

    方案给的是比例式 ``H/W ∝ (d_rho*g*C_0)/(eta_eff*|v_rel|)``，没有常数。这里用
    参考条件（:data:`REFERENCE_ASPECT_RATIO` 等）把同一组物理量相除，得到无量纲形式：

        H/W = 0.05 * [(d_rho*g*C_0)/(eta_eff*|v_rel|)] / [(d_rho_ref*g*C_0_ref)/(eta_ref*v_ref)]

    参考点取喜马拉雅量级（H≈5 km、半宽 W≈100 km）。这样既保留了方案对
    ``d_rho / C_0 / eta_eff / v_rel`` 的全部依赖，又能直接由高度反推造山带宽度。

    注意：方案 §4.3 的**公式**给出 ``H/W ∝ 1/|v_rel|``（汇聚越快，造山带越宽越矮），
    但同段**文字结论**写的是"汇聚速率越快，造山带越窄越高"，二者方向相反。
    本实现以显式公式为准；若需按文字结论，调用方对 ``v_rel`` 取倒数即可。
    """
    if c0_km <= 0.0:
        raise ValueError(f"初始地壳厚度必须为正，实际为 {c0_km}")
    if eta_eff <= 0.0:
        raise ValueError(f"增厚效率必须为正，实际为 {eta_eff}")
    if v_rel <= 0.0:
        raise ValueError(f"汇聚速率必须为正，实际为 {v_rel}")
    reference = (
        REFERENCE_DELTA_RHO * g * REFERENCE_C0_KM * 1000.0
        / (DEFAULT_THICKENING_EFFICIENCY * REFERENCE_V_REL)
    )
    current = delta_rho * g * c0_km * 1000.0 / (eta_eff * v_rel)
    return float(REFERENCE_ASPECT_RATIO * current / reference)


def ridge_half_width_km(height_m: float, aspect_ratio: float) -> float:
    """由高宽比反推造山带半宽 ``W = H/(H/W)`` (km)（§4.3）。"""
    if aspect_ratio <= 0.0:
        raise ValueError(f"高宽比必须为正，实际为 {aspect_ratio}")
    return float(abs(height_m) / aspect_ratio / 1000.0)
