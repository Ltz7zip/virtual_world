"""第三层·热力侵蚀：休止角约束与坡面扩散（《第三层完善：侵蚀模拟》§三）。

两个模型同属"重力驱动的坡面物质迁移"，本模块都实现：

- :func:`thermal_erode` —— **休止角管道模型**（§3.4），第三层管线的第 4 步。
  相邻单元高差超过 ``ΔH_max = tanβ·Δx`` 时产生滑移通量
  ``F = K(ΔH − ΔH_max)``，高处物质向低处迁移，直到所有坡度低于休止角。
  物理上休止角即摩擦角（§3.5），典型值沙砾 33°、湿土 15–25°。
- :func:`hillslope_diffusion` —— **坡面扩散**（Culling 1960，§3.2 线性 / §3.3 非线性）
  ``∂H/∂t = κ∇²H``，通量与坡面梯度成正比；``nonlinear=True`` 时改用
  ``q = -κ∇H/(1 - (|∇H|/tanβ)²)``——坡度趋近休止角时分母趋于零、通量趋于无穷，
  从而**逼近**休止角而不越界（硬约束仍由 §3.4 的管道模型保证）。它是景观演化方程中
  不可省的坡面项（§2.2：缺了它 ``m/n = 0.5`` 的河流功率律会给出尺度不变的假景观）。

两者都是**纯物质再分配**：面通量在两侧共享，因此面积加权体积严格守恒。

分辨率与量级提醒：休止角是"坡度"判据，在粗网格上几乎不可能被触发——
1° 网格（111 km）要达到 33° 需要 72 km 高差。这是物理正确的：休止角/坡面
扩散是 100 m–1 km 尺度过程，只有细网格或陡崖（断层、海沟壁）才会激活。
因此地球尺度粗网格上 :func:`thermal_erode` 近似恒等变换，属于预期行为。
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit, prange

from ..core import spherical
from ..core.constants import EARTH_RADIUS

#: 默认休止角 (deg)：§3.1 沙砾典型值
DEFAULT_TALUS_ANGLE_DEG = 33.0
#: 滑移松弛系数（每迭代步），<= 0.5 才能保证不越过休止角
K_THERMAL = 0.5
#: 默认迭代次数
DEFAULT_ITERATIONS = 5
#: 非线性扩散分母的放大上限（§3.3 的 ``1/(1-(|∇H|/tanβ)²)`` 在坡度趋近休止角时发散，
#: 该上限把分母正则化到 ``>= 1/max``，避免显式积分炸掉）
MAX_DIFFUSION_AMPLIFICATION = 20.0


# ===== 休止角管道模型（§3.4）=====


@njit(parallel=True, fastmath=True, cache=True)
def _thermal_loop(
    height: np.ndarray,
    land: np.ndarray,
    area: np.ndarray,
    dlat_m: float,
    dlon_m: np.ndarray,
    talus_tan: float,
    rate: float,
    iterations: int,
) -> None:
    """休止角滑移迭代松弛（面通量 + 双缓冲，避免并行读写竞态）。"""
    nlat, nlon = height.shape
    f_east = np.zeros((nlat, nlon), dtype=np.float64)
    f_north = np.zeros((nlat, nlon), dtype=np.float64)
    scale = np.ones((nlat, nlon), dtype=np.float64)

    for _ in range(iterations):
        # 1 面通量（每个面只计算一次；符号 = 由 i 侧流向 i+1 侧为正）
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                f_east[i, j] = 0.0
                f_north[i, j] = 0.0
                if not land[i, j]:
                    continue
                jn = j + 1
                if jn >= nlon:
                    jn -= nlon
                face_e = dlat_m * dlon_m[i]
                dh = height[i, j] - height[i, jn]
                limit_e = talus_tan * dlon_m[i]
                if dh > limit_e:
                    f_east[i, j] = rate * (dh - limit_e) * face_e
                elif dh < -limit_e:
                    f_east[i, j] = -rate * (-dh - limit_e) * face_e
                if i + 1 < nlat:
                    face_n = 0.5 * dlat_m * (dlon_m[i] + dlon_m[i + 1])
                    dh_n = height[i, j] - height[i + 1, j]
                    limit_n = talus_tan * dlat_m
                    if dh_n > limit_n:
                        f_north[i, j] = rate * (dh_n - limit_n) * face_n
                    elif dh_n < -limit_n:
                        f_north[i, j] = -rate * (-dh_n - limit_n) * face_n
        # 2 出流钳制：一次迭代转移量不得超过"到最低邻居的余量"
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                scale[i, j] = 1.0
                if not land[i, j]:
                    continue
                jw = j - 1
                if jw < 0:
                    jw += nlon
                outflow = 0.0
                if f_east[i, j] > 0.0:
                    outflow += f_east[i, j]
                if f_east[i, jw] < 0.0:
                    outflow -= f_east[i, jw]
                if f_north[i, j] > 0.0:
                    outflow += f_north[i, j]
                if i - 1 >= 0 and f_north[i - 1, j] < 0.0:
                    outflow -= f_north[i - 1, j]
                if outflow > 0.0:
                    jp = j + 1
                    if jp >= nlon:
                        jp -= nlon
                    low = height[i, jp]
                    if height[i, jw] < low:
                        low = height[i, jw]
                    if i + 1 < nlat and height[i + 1, j] < low:
                        low = height[i + 1, j]
                    if i - 1 >= 0 and height[i - 1, j] < low:
                        low = height[i - 1, j]
                    room = height[i, j] - low
                    if room < 0.0:
                        room = 0.0
                    limit = room * area[i, j] / outflow
                    if limit < 1.0:
                        scale[i, j] = limit
        # 3 按失水方缩放面通量
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                jn = j + 1
                if jn >= nlon:
                    jn -= nlon
                fe = f_east[i, j]
                if fe > 0.0:
                    f_east[i, j] = fe * scale[i, j]
                elif fe < 0.0:
                    f_east[i, j] = fe * scale[i, jn]
                fn = f_north[i, j]
                if fn > 0.0:
                    f_north[i, j] = fn * scale[i, j]
                elif fn < 0.0 and i + 1 < nlat:
                    f_north[i, j] = fn * scale[i + 1, j]
        # 4 地形更新（面通量两侧共享 → 体积守恒）
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                if not land[i, j]:
                    continue
                jw = j - 1
                if jw < 0:
                    jw += nlon
                net = f_east[i, jw] - f_east[i, j] - f_north[i, j]
                if i - 1 >= 0:
                    net += f_north[i - 1, j]
                height[i, j] += net / area[i, j]


@dataclasses.dataclass(frozen=True)
class ThermalErosionResult:
    """热力侵蚀输出（§七 第 4 步）。"""

    elevation: np.ndarray  # 侵蚀后地形 H (m)
    drop: np.ndarray  # ΔH_thermal = H_in − H_out (m)
    report: dict[str, float | int | str]


def thermal_erode(
    elevation: np.ndarray,
    is_ocean: np.ndarray,
    *,
    talus_angle_deg: float = DEFAULT_TALUS_ANGLE_DEG,
    rate: float = K_THERMAL,
    iterations: int = DEFAULT_ITERATIONS,
    radius: float = EARTH_RADIUS,
) -> ThermalErosionResult:
    """休止角热力侵蚀（§3.4）。

    参数：
        elevation: 输入高程场 H (m)
        is_ocean: 海洋掩码（海洋不参与山体滑移，保持海底形态）
        talus_angle_deg: 休止角 β (deg)
        rate: 滑移松弛系数，必须 <= 0.5（否则单步可越过休止角产生震荡）
        iterations: 松弛迭代次数
        radius: 行星半径 (m)，决定网格度规间距

    返回 :class:`ThermalErosionResult`；坡度已低于休止角的区域逐点不变。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    if elev.shape != ocean.shape:
        raise ValueError(f"elevation 与 is_ocean 形状不一致: {elev.shape} vs {ocean.shape}")
    if not 0.0 <= talus_angle_deg < 90.0:
        raise ValueError(f"talus_angle_deg 必须在 [0, 90) 内，实际为 {talus_angle_deg}")
    if not 0.0 <= rate <= 0.5:
        raise ValueError(f"rate 必须在 [0, 0.5] 内（稳定性要求），实际为 {rate}")
    if iterations < 0:
        raise ValueError(f"iterations 不能为负，实际为 {iterations}")
    if radius <= 0.0:
        raise ValueError(f"radius 必须为正，实际为 {radius}")

    nlat, nlon = elev.shape
    dlat_m, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    area = spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)
    height = np.array(elev, dtype=np.float64, copy=True)
    if iterations > 0:
        _thermal_loop(
            height,
            np.ascontiguousarray(~ocean),
            np.ascontiguousarray(area),
            float(dlat_m),
            np.ascontiguousarray(dlon_m),
            float(np.tan(np.deg2rad(talus_angle_deg))),
            float(rate),
            int(iterations),
        )
    drop = elev - height
    report: dict[str, float | int | str] = {
        "talus_angle_deg": float(talus_angle_deg),
        "iterations": int(iterations),
        "moved_volume_m3": float(np.sum(np.maximum(drop, 0.0) * area)),
        "max_abs_drop_m": float(np.abs(drop).max()) if drop.size else 0.0,
    }
    return ThermalErosionResult(elevation=height, drop=drop, report=report)


# ===== 坡面扩散（§3.2）=====


@njit(parallel=True, fastmath=True, cache=True)
def _diffusion_loop(
    height: np.ndarray,
    land: np.ndarray,
    area: np.ndarray,
    dlat_m: float,
    dlon_m: np.ndarray,
    kappa: float,
    dt: float,
    n_steps: int,
) -> None:
    """线性坡面扩散显式积分（面通量形式，体积守恒）。"""
    nlat, nlon = height.shape
    f_east = np.zeros((nlat, nlon), dtype=np.float64)
    f_north = np.zeros((nlat, nlon), dtype=np.float64)
    for _ in range(n_steps):
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                f_east[i, j] = 0.0
                f_north[i, j] = 0.0
                if not land[i, j]:
                    continue
                jn = j + 1
                if jn >= nlon:
                    jn -= nlon
                if land[i, jn]:
                    f_east[i, j] = kappa * dt * (height[i, j] - height[i, jn]) / dlon_m[i] * dlat_m
                if i + 1 < nlat and land[i + 1, j]:
                    face_n = 0.5 * dlat_m * (dlon_m[i] + dlon_m[i + 1])
                    f_north[i, j] = kappa * dt * (height[i, j] - height[i + 1, j]) / dlat_m * face_n
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                if not land[i, j]:
                    continue
                jw = j - 1
                if jw < 0:
                    jw += nlon
                net = f_east[i, jw] - f_east[i, j] - f_north[i, j]
                if i - 1 >= 0:
                    net += f_north[i - 1, j]
                height[i, j] += net / area[i, j]


@njit(parallel=True, fastmath=True, cache=True)
def _nonlinear_diffusion_loop(
    height: np.ndarray,
    land: np.ndarray,
    area: np.ndarray,
    dlat_m: float,
    dlon_m: np.ndarray,
    kappa: float,
    dt: float,
    n_steps: int,
    talus_tan: float,
    f_cap: float,
) -> None:
    """非线性坡面扩散（§3.3）显式积分，面通量形式、体积守恒。

    扩散系数随面坡度放大：``k_eff = kappa/(1 - (|grad H|/tan beta)^2)``。
    ``|grad H|/tan beta`` 被截断到 ``f_cap < 1``（由 ``max_amplification`` 决定），
    从而正则化休止角处的奇异性。

    **斜率限制器**：每个面的搬运量不超过"把该面坡度降到休止角所需量的一半"，
    因此单步之内坡度不会被搬过头。这既让格式**无条件稳定**（不受放大系数与
    极区小格距的 CFL 限制），也正是 §3.3 想要的物理行为——坡度趋近休止角时
    通量趋于无穷，即"超出休止角的部分被立刻搬走"。
    """
    nlat, nlon = height.shape
    f_east = np.zeros((nlat, nlon), dtype=np.float64)
    f_north = np.zeros((nlat, nlon), dtype=np.float64)
    for _ in range(n_steps):
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                f_east[i, j] = 0.0
                f_north[i, j] = 0.0
                if not land[i, j]:
                    continue
                jn = j + 1
                if jn >= nlon:
                    jn -= nlon
                if land[i, jn]:
                    dist = dlon_m[i]
                    drop = height[i, j] - height[i, jn]
                    ratio = abs(drop) / dist / talus_tan
                    if ratio > f_cap:
                        ratio = f_cap
                    flux = (kappa / (1.0 - ratio * ratio)) * dt * drop / dist * dlat_m
                    room = abs(drop) - talus_tan * dist
                    if room > 0.0:
                        cap = 0.5 * room * (area[i, j] if area[i, j] < area[i, jn] else area[i, jn])
                        if abs(flux) > cap:
                            flux = cap if flux > 0.0 else -cap
                    f_east[i, j] = flux
                if i + 1 < nlat and land[i + 1, j]:
                    face_n = 0.5 * dlat_m * (dlon_m[i] + dlon_m[i + 1])
                    drop_n = height[i, j] - height[i + 1, j]
                    ratio_n = abs(drop_n) / dlat_m / talus_tan
                    if ratio_n > f_cap:
                        ratio_n = f_cap
                    flux_n = (kappa / (1.0 - ratio_n * ratio_n)) * dt * drop_n / dlat_m * face_n
                    room_n = abs(drop_n) - talus_tan * dlat_m
                    if room_n > 0.0:
                        a0 = area[i, j]
                        a1 = area[i + 1, j]
                        cap_n = 0.5 * room_n * (a0 if a0 < a1 else a1)
                        if abs(flux_n) > cap_n:
                            flux_n = cap_n if flux_n > 0.0 else -cap_n
                    f_north[i, j] = flux_n
        for i in prange(nlat):  # type: ignore[no-untyped-call, attr-defined]
            for j in range(nlon):
                if not land[i, j]:
                    continue
                jw = j - 1
                if jw < 0:
                    jw += nlon
                net = f_east[i, jw] - f_east[i, j] - f_north[i, j]
                if i - 1 >= 0:
                    net += f_north[i - 1, j]
                height[i, j] += net / area[i, j]


def hillslope_diffusion(
    elevation: np.ndarray,
    kappa: float,
    dt: float,
    *,
    n_steps: int = 1,
    is_ocean: np.ndarray | None = None,
    radius: float = EARTH_RADIUS,
    nonlinear: bool = False,
    talus_angle_deg: float = DEFAULT_TALUS_ANGLE_DEG,
    max_amplification: float = MAX_DIFFUSION_AMPLIFICATION,
) -> np.ndarray:
    """坡面扩散 ``∂H/∂t = κ∇²H``（§3.2）或其非线性形式（§3.3），返回新高程场。

    参数：
        elevation: 输入高程场 (m)
        kappa: 扩散系数 κ，量纲随 ``dt`` 一致（内核只用 ``kappa*dt`` 的乘积，
            故 (m²/s, s) 与 (m²/Ma, Ma) 等价）。方案 §3.2 的 4e-4–40e-4 m²/yr
            换算后约为 1.3e-11–1.3e-10 m²/s
        dt: 时间步，单位与 ``kappa`` 一致
        n_steps: 步数
        is_ocean: 海洋掩码；海洋为**零通量边界**（不参与坡面扩散）
        radius: 行星半径 (m)
        nonlinear: ``True`` 时用 §3.3 的非线性扩散
            ``q = -κ∇H/(1-(|∇H|/tanβ)²)``，坡度趋近休止角时通量急剧放大
        talus_angle_deg: 非线性形式的休止角 β（§3.1 沙砾 33°）
        max_amplification: 非线性分母的放大上限 ``1/(1-f_cap²)``，用于正则化
            ``|∇H| → tanβ`` 的奇异性（硬约束仍由 :func:`thermal_erode` 保证）

    显式格式的稳定性要求 ``κ·dt·4/Δx² <= 0.5``（一维 CFL；非线性形式按放大上限
    同样估算），**Δx 取最小格距**——经纬网格高纬度的经向格距随 ``cos(lat)`` 收缩，
    极区才是限制步长的地方。超出时调用方会看到振荡——请减小 ``dt``。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    if kappa < 0.0:
        raise ValueError(f"kappa 不能为负，实际为 {kappa}")
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps 必须 >= 1，实际为 {n_steps}")
    if not 0.0 <= talus_angle_deg < 90.0:
        raise ValueError(f"talus_angle_deg 必须在 [0, 90) 内，实际为 {talus_angle_deg}")
    if nonlinear and talus_angle_deg <= 0.0:
        raise ValueError(f"非线性扩散要求休止角为正，实际为 {talus_angle_deg}")
    if max_amplification <= 1.0:
        raise ValueError(f"max_amplification 必须 > 1，实际为 {max_amplification}")
    nlat, nlon = elev.shape
    land = (
        np.ones((nlat, nlon), dtype=bool) if is_ocean is None else ~np.asarray(is_ocean, dtype=bool)
    )
    if land.shape != elev.shape:
        raise ValueError(f"is_ocean 形状 {land.shape} 与 elevation {elev.shape} 不一致")
    dlat_m, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    area = spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)
    height = np.array(elev, dtype=np.float64, copy=True)
    if nonlinear:
        _nonlinear_diffusion_loop(
            height,
            np.ascontiguousarray(land),
            np.ascontiguousarray(area),
            float(dlat_m),
            np.ascontiguousarray(dlon_m),
            float(kappa),
            float(dt),
            int(n_steps),
            float(np.tan(np.deg2rad(talus_angle_deg))),
            float(1.0 - 1.0 / max_amplification),
        )
    else:
        _diffusion_loop(
            height,
            np.ascontiguousarray(land),
            np.ascontiguousarray(area),
            float(dlat_m),
            np.ascontiguousarray(dlon_m),
            float(kappa),
            float(dt),
            int(n_steps),
        )
    return np.asarray(height, dtype=np.float64)


# ===== 坡度诊断 =====


@njit(cache=True)
def _slope_kernel(height: np.ndarray, dlat_m: float, dlon_m: np.ndarray, out: np.ndarray) -> None:
    """|∇H|（= 坡度角的正切），中心差分，纬度边界单侧差分。"""
    nlat, nlon = height.shape
    for i in range(nlat):
        for j in range(nlon):
            jm = j - 1
            if jm < 0:
                jm += nlon
            jp = j + 1
            if jp >= nlon:
                jp -= nlon
            hx = (height[i, jp] - height[i, jm]) / (2.0 * dlon_m[i])
            if i == 0:
                hy = (height[1, j] - height[0, j]) / dlat_m
            elif i == nlat - 1:
                hy = (height[i, j] - height[i - 1, j]) / dlat_m
            else:
                hy = (height[i + 1, j] - height[i - 1, j]) / (2.0 * dlat_m)
            out[i, j] = np.sqrt(hx * hx + hy * hy)


def slope_field(elevation: np.ndarray, *, radius: float = EARTH_RADIUS) -> np.ndarray:
    """坡度场 ``|∇H|``（无量纲，等于坡度角的正切）。

    按度规距离计算梯度，因此可用于休止角约束校验与坡度统计（§1.6-4）。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    nlat, nlon = elev.shape
    if nlat < 2:
        raise ValueError("纬度方向至少需要 2 个点才能计算坡度")
    dlat_m, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    out = np.zeros((nlat, nlon), dtype=np.float64)
    _slope_kernel(np.ascontiguousarray(elev), float(dlat_m), np.ascontiguousarray(dlon_m), out)
    return out


__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_TALUS_ANGLE_DEG",
    "K_THERMAL",
    "MAX_DIFFUSION_AMPLIFICATION",
    "ThermalErosionResult",
    "hillslope_diffusion",
    "slope_field",
    "thermal_erode",
]
