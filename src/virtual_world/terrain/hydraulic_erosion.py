"""第三层·水力侵蚀：虚拟管道模型（《第三层完善：侵蚀模拟》§一、§5.3）。

物理：把每个格点视作一个水柱，相邻水柱由"虚拟管道"连接。管道流量由水头差
驱动，取小孔出流（orifice）形式——这正是方案 §1.2 的 ``Q = A·sign(Δh)·√|Δh|``
补上重力标定后的完整量纲形式：

    Q_ij = C_pipe · A_face · sign(Δh_ij) · sqrt(2g|Δh_ij|)      [m^3/s]

其中 ``A_face`` 是共享面的面积、``Δh`` 是水头差（``H + d`` 之差）。水量更新
``d_i += Δt·(ΣQ_in − ΣQ_out)/A_cell``，面通量由两侧共享，因此**水量体积严格守恒**。

稳定性（§5.2）：
- **出流钳制**：每步先算出流总量，若 ``Δt·ΣQ_out > d_i·A_cell`` 则按"失水方"
  统一缩放该面通量。既保证 ``d >= 0``，又不破坏守恒（缩放后的通量仍是两侧共享）。
- **自适应时间步**：``Δt = CFL·Δx_min/|v|_max``（用上一步的流速场估计），
  并按 ``[dt_min, dt_max]`` 截断。
- **单步地形改动上限**：侵蚀不得把单元挖到低于其最低邻居，沉积不得抬高到
  高于其最高邻居——既防震荡，也保证河道单调下切。

侵蚀/沉积（§1.3）：``C_cap = K_c·slope·|v|·d``；``s > C_cap`` 沉积
``D = K_d(s−C_cap)``，``s < C_cap`` 侵蚀 ``E = K_s(C_cap−s)``，``H ← H − E + D``。
Shields 阈值（§1.4）以临界流速 ``u_c`` 实现：``|v| <= u_c`` 时容量为 0，不侵蚀。
``u_c`` 由 Shields 准则按河床粒径反解（:func:`critical_velocity_from_shields`，
默认 1 mm 中砂床 ⇒ ≈0.38 m/s），见 :data:`CRITICAL_VELOCITY`。

沉积物输运（§1.3、§5.3）：半拉格朗日回溯 + 双线性插值，再用 **MacCormack**
反扩散校正 ``s^{n+1} = s' + (s^n − s'^n)/2``，并以回溯采样四角的局部极值作
通量限制器抑制过冲。

单位与标定：高程 m、水量 m、流量 m^3/s、时间 s。``C_pipe`` 默认 1e-6 是**标定量**
——它决定"降水累积速率 / 排水速率"的平衡水深（量级 1 mm），进而决定坡度-流量
乘积的量级。侵蚀总量随 ``n_steps`` 线性增长：本层定位是 100 m–1 km 尺度的
微观刻蚀（§1.1），不负责造山。
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit, prange

from ..core import spherical
from ..core.constants import DENSITY_WATER, EARTH_GRAVITY, EARTH_RADIUS

# ===== §1.4 Shields 准则 =====
#: Shields 参数 theta（§1.4：0.03–0.06）
SHIELDS_PARAMETER = 0.045
#: 河床沉积物粒径 d_s (m)：默认 1 mm 中砂
GRAIN_SIZE_M = 1.0e-3
#: 沉积物密度 (kg/m^3)（石英砂）
DENSITY_SEDIMENT = 2650.0
#: 河床阻力系数 C_f：剪切应力与流速的闭合 ``tau = rho*C_f*u^2``
#: （典型光滑床面 0.002–0.005，取中值）
BED_FRICTION_COEFFICIENT = 0.005


def shields_critical_shear(
    grain_size_m: float = GRAIN_SIZE_M,
    *,
    shields_parameter: float = SHIELDS_PARAMETER,
    density_sediment: float = DENSITY_SEDIMENT,
    density_water: float = DENSITY_WATER,
    g: float = EARTH_GRAVITY,
) -> float:
    """Shields 临界剪切应力 ``tau_ct = theta * d_s * (gamma_s - gamma_w)`` (Pa)（§1.4）。

    ``gamma = rho*g`` 为容重 (N/m^3)。方案取值 ``theta = 0.03–0.06``；粒径越大
    临界应力越高（砂 < 砾石 < 漂石）。
    """
    if grain_size_m <= 0.0:
        raise ValueError(f"粒径必须为正，实际为 {grain_size_m}")
    if not 0.0 < shields_parameter < 1.0:
        raise ValueError(f"Shields 参数应在 (0, 1) 内，实际为 {shields_parameter}")
    if density_sediment <= density_water:
        raise ValueError("沉积物密度必须大于水密度")
    return float(shields_parameter * grain_size_m * (density_sediment - density_water) * g)


def critical_velocity_from_shields(
    grain_size_m: float = GRAIN_SIZE_M,
    *,
    shields_parameter: float = SHIELDS_PARAMETER,
    density_sediment: float = DENSITY_SEDIMENT,
    density_water: float = DENSITY_WATER,
    friction_coefficient: float = BED_FRICTION_COEFFICIENT,
    g: float = EARTH_GRAVITY,
) -> float:
    """由 Shields 临界剪切应力反解临界流速 ``u_c = sqrt(tau_ct/(rho_w*C_f))`` (m/s)。

    剪切应力与流速的闭合取 ``tau = rho_w*C_f*u^2``（C_f 为河床阻力系数）。方案
    §1.4 的 ``tau_ct = theta*d_s*(gamma_s-gamma_w)``、``E_s = K_3*(u-u_c)`` 中
    ``u_c`` 就是本函数的返回值——管道模型用 ``|v| <= u_c`` 时容量归零来实现。
    """
    if friction_coefficient <= 0.0:
        raise ValueError(f"阻力系数必须为正，实际为 {friction_coefficient}")
    tau_ct = shields_critical_shear(
        grain_size_m,
        shields_parameter=shields_parameter,
        density_sediment=density_sediment,
        density_water=density_water,
        g=g,
    )
    return float(np.sqrt(tau_ct / (density_water * friction_coefficient)))


#: 管道排水系数（标定量，量级决定平衡水深，见模块 docstring）
PIPE_COEFFICIENT = 1.0e-6
#: 沉积物容量常数 K_c（§1.3）。方案未给具体值（K 族跨越 4–5 个数量级），
#: 此处标定为：1° 网格 + 1 m/yr 降水 + 数百步 ⇒ 分米–米级的局部下切
CAPACITY_CONSTANT = 1000.0
#: 侵蚀速率 K_s（每步松弛系数，需 <= 1 才稳定）
EROSION_RATE = 0.5
#: 沉积速率 K_d（每步松弛系数，需 <= 1 才稳定）
DEPOSITION_RATE = 0.5
#: 临界流速 u_c (m/s)：Shields 准则的等效实现（§1.4），默认 1 mm 中砂床
CRITICAL_VELOCITY = critical_velocity_from_shields()
#: 流速上限 (m/s)：管道闭合在陡坡/薄水层下会给出非物理大流速，故设上限
MAX_VELOCITY = 10.0
#: CFL 安全系数（§5.2）
CFL_SAFETY = 0.5
#: 自适应时间步的上下限 (s)
DT_MIN_S = 1.0e-1
DT_MAX_S = 1.0e7
#: 计算流速时的最小水深 (m)，避免除零
MIN_WATER_DEPTH = 1.0e-12


# ===== 双线性采样（经度循环、纬度外推）=====


@njit(cache=True, inline="always")
def _bilinear(
    field: np.ndarray, fi: float, fj: float, nlat: int, nlon: int
) -> tuple[float, float, float]:
    """在 ``(fi, fj)`` 处双线性采样，返回 ``(值, 四角最小值, 四角最大值)``。

    后两者供 MacCormack 通量限制器使用（§5.3）。
    """
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    wi = fi - i0
    wj = fj - j0
    i0c = i0
    if i0c < 0:
        i0c = 0
    elif i0c > nlat - 1:
        i0c = nlat - 1
    i1c = i0 + 1
    if i1c < 0:
        i1c = 0
    elif i1c > nlat - 1:
        i1c = nlat - 1
    j0c = j0 % nlon
    j1c = (j0 + 1) % nlon
    f00 = field[i0c, j0c]
    f01 = field[i0c, j1c]
    f10 = field[i1c, j0c]
    f11 = field[i1c, j1c]
    top = f00 + (f01 - f00) * wj
    bottom = f10 + (f11 - f10) * wj
    value = top + (bottom - top) * wi
    lo = min(min(f00, f01), min(f10, f11))
    hi = max(max(f00, f01), max(f10, f11))
    return value, lo, hi


@njit(parallel=True, cache=True)
def _advect_pass(
    field: np.ndarray,
    out: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    sign: float,
    dt: float,
    dlat_m: float,
    dlon_m: np.ndarray,
    land: np.ndarray,
) -> None:
    """半拉格朗日单次对流：``sign=-1`` 反向回溯（求 ``s'``），``sign=+1`` 正向。"""
    nlat, nlon = field.shape
    for i in prange(nlat):
        for j in range(nlon):
            if not land[i, j]:
                out[i, j] = 0.0
                lo[i, j] = 0.0
                hi[i, j] = 0.0
                continue
            fi = i + sign * vy[i, j] * dt / dlat_m
            fj = j + sign * vx[i, j] * dt / dlon_m[i]
            value, vlo, vhi = _bilinear(field, fi, fj, nlat, nlon)
            out[i, j] = value
            lo[i, j] = vlo
            hi[i, j] = vhi


@njit(parallel=True, cache=True)
def _apply_macormack(
    field: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    out: np.ndarray,
    land: np.ndarray,
) -> None:
    """MacCormack 校正 + 局部极值限制（§5.3）。"""
    nlat, nlon = field.shape
    for i in prange(nlat):
        for j in range(nlon):
            if not land[i, j]:
                out[i, j] = 0.0
                continue
            value = s1[i, j] + 0.5 * (field[i, j] - s2[i, j])
            if value < lo[i, j]:
                value = lo[i, j]
            elif value > hi[i, j]:
                value = hi[i, j]
            if value < 0.0:
                value = 0.0
            out[i, j] = value


def advect_sediment(
    field: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    dt: float,
    dlat_m: float,
    dlon_m: np.ndarray,
    macormack: bool = True,
    land: np.ndarray | None = None,
) -> np.ndarray:
    """沉积物半拉格朗日输运（§1.3、§5.3），返回输运后的新场。

    ``macormack=True`` 时做反向 + 正向两次对流并按 ``s' + (s − s'')/2`` 反扩散，
    显著减轻半拉格朗日的数值耗散；``land`` 给定时非陆地单元输出 0。
    """
    f = np.asarray(field, dtype=np.float64)
    ux = np.asarray(vx, dtype=np.float64)
    uy = np.asarray(vy, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"field 必须为 2D，实际 {f.shape}")
    if ux.shape != f.shape or uy.shape != f.shape:
        raise ValueError("速度场形状必须与 field 一致")
    nlat, nlon = f.shape
    dlon = np.asarray(dlon_m, dtype=np.float64)
    if dlon.shape != (nlat,):
        raise ValueError(f"dlon_m 长度必须为 nlat={nlat}，实际 {dlon.shape}")
    if dt < 0.0:
        raise ValueError(f"dt 不能为负，实际为 {dt}")
    mask = np.ones((nlat, nlon), dtype=bool) if land is None else np.asarray(land, dtype=bool)
    if mask.shape != f.shape:
        raise ValueError(f"land 形状 {mask.shape} 与 field {f.shape} 不一致")

    f = np.ascontiguousarray(f)
    ux = np.ascontiguousarray(ux)
    uy = np.ascontiguousarray(uy)
    mask = np.ascontiguousarray(mask)
    dlon = np.ascontiguousarray(dlon)
    s1 = np.zeros_like(f)
    lo = np.zeros_like(f)
    hi = np.zeros_like(f)
    s2 = np.zeros_like(f)
    tmp_a = np.zeros_like(f)
    tmp_b = np.zeros_like(f)
    _advect_pass(f, s1, lo, hi, ux, uy, -1.0, float(dt), float(dlat_m), dlon, mask)
    if not macormack:
        return np.asarray(np.where(mask, s1, 0.0), dtype=np.float64)
    _advect_pass(s1, s2, tmp_a, tmp_b, ux, uy, 1.0, float(dt), float(dlat_m), dlon, mask)
    out = np.zeros_like(f)
    _apply_macormack(f, s1, s2, lo, hi, out, mask)
    return np.asarray(out, dtype=np.float64)


# ===== 时间循环（Numba）=====


@njit(parallel=True, fastmath=True, cache=True)
def _hydraulic_loop(
    height: np.ndarray,
    depth: np.ndarray,
    sediment: np.ndarray,
    land: np.ndarray,
    area: np.ndarray,
    dlat_m: float,
    dlon_m: np.ndarray,
    precip: np.ndarray,
    pipe_coeff: float,
    kc: float,
    ks: float,
    kd: float,
    u_c: float,
    v_cap: float,
    dt0: float,
    dt_min: float,
    dt_max: float,
    cfl: float,
    adaptive: bool,
    n_steps: int,
    macormack: bool,
) -> float:
    """水力侵蚀显式时间积分，返回最后一个时间步长。"""
    nlat, nlon = height.shape
    f_east = np.zeros((nlat, nlon), dtype=np.float64)
    f_north = np.zeros((nlat, nlon), dtype=np.float64)
    scale = np.ones((nlat, nlon), dtype=np.float64)
    delta_h = np.zeros((nlat, nlon), dtype=np.float64)
    delta_s = np.zeros((nlat, nlon), dtype=np.float64)
    vx = np.zeros((nlat, nlon), dtype=np.float64)
    vy = np.zeros((nlat, nlon), dtype=np.float64)
    speed = np.zeros((nlat, nlon), dtype=np.float64)
    s1 = np.zeros((nlat, nlon), dtype=np.float64)
    s2 = np.zeros((nlat, nlon), dtype=np.float64)
    adv_lo = np.zeros((nlat, nlon), dtype=np.float64)
    adv_hi = np.zeros((nlat, nlon), dtype=np.float64)
    tmp_a = np.zeros((nlat, nlon), dtype=np.float64)
    tmp_b = np.zeros((nlat, nlon), dtype=np.float64)

    dx_min = dlat_m
    for i in range(nlat):
        if dlon_m[i] < dx_min:
            dx_min = dlon_m[i]

    dt = dt0
    for _ in range(n_steps):
        # 1 降水补给
        for i in prange(nlat):
            for j in range(nlon):
                if land[i, j]:
                    depth[i, j] += precip[i, j] * dt
        # 2 管道流量（每个面只由西/南侧单元计算一次，符号表示净输运方向）
        for i in prange(nlat):
            for j in range(nlon):
                f_east[i, j] = 0.0
                f_north[i, j] = 0.0
                if not land[i, j]:
                    depth[i, j] = 0.0
                    continue
                head = height[i, j] + depth[i, j]
                jn = j + 1
                if jn >= nlon:
                    jn -= nlon
                dh = head - (height[i, jn] + depth[i, jn])
                f_east[i, j] = pipe_coeff * (dlat_m * dlon_m[i]) * np.sign(dh) * np.sqrt(abs(dh))
                if i + 1 < nlat:
                    dh_n = head - (height[i + 1, j] + depth[i + 1, j])
                    face = 0.5 * dlat_m * (dlon_m[i] + dlon_m[i + 1])
                    f_north[i, j] = pipe_coeff * face * np.sign(dh_n) * np.sqrt(abs(dh_n))
        # 3 出流钳制因子：保证本步排水量不超过现有水量
        for i in prange(nlat):
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
                    limit = depth[i, j] * area[i, j] / (dt * outflow)
                    if limit < 1.0:
                        scale[i, j] = limit
        # 4 面通量按失水方的因子缩放（仍是两侧共享，守恒不破坏）
        for i in prange(nlat):
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
        # 5 水量更新
        for i in prange(nlat):
            for j in range(nlon):
                if not land[i, j]:
                    depth[i, j] = 0.0
                    continue
                jw = j - 1
                if jw < 0:
                    jw += nlon
                net = f_east[i, jw] - f_east[i, j] - f_north[i, j]
                if i - 1 >= 0:
                    net += f_north[i - 1, j]
                depth[i, j] += dt * net / area[i, j]
                if depth[i, j] < 0.0:
                    depth[i, j] = 0.0
        # 6 流速场（净流量 / (水深 × 面宽)，并施加流速上限）
        for i in prange(nlat):
            for j in range(nlon):
                vx[i, j] = 0.0
                vy[i, j] = 0.0
                speed[i, j] = 0.0
                if not land[i, j]:
                    continue
                dep = depth[i, j]
                if dep <= MIN_WATER_DEPTH:
                    continue
                jw = j - 1
                if jw < 0:
                    jw += nlon
                qx = f_east[i, j] - f_east[i, jw]
                qy = f_north[i, j]
                if i - 1 >= 0:
                    qy -= f_north[i - 1, j]
                ux = qx / (dep * dlat_m)
                uy = qy / (dep * dlon_m[i])
                sp = np.sqrt(ux * ux + uy * uy)
                if sp > v_cap:
                    shrink = v_cap / sp
                    ux *= shrink
                    uy *= shrink
                    sp = v_cap
                vx[i, j] = ux
                vy[i, j] = uy
                speed[i, j] = sp
        # 7 侵蚀/沉积（算增量再统一应用，避免并行读写竞态）
        for i in prange(nlat):
            for j in range(nlon):
                delta_h[i, j] = 0.0
                delta_s[i, j] = 0.0
                if not land[i, j]:
                    sediment[i, j] = 0.0
                    continue
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
                slope = np.sqrt(hx * hx + hy * hy)
                active = speed[i, j] - u_c
                if active < 0.0:
                    active = 0.0
                capacity = kc * slope * active * depth[i, j]
                h_min = height[i, jp]
                if height[i, jm] < h_min:
                    h_min = height[i, jm]
                if i + 1 < nlat and height[i + 1, j] < h_min:
                    h_min = height[i + 1, j]
                if i - 1 >= 0 and height[i - 1, j] < h_min:
                    h_min = height[i - 1, j]
                if sediment[i, j] > capacity:
                    transfer = kd * (sediment[i, j] - capacity)
                    delta_h[i, j] = transfer
                    delta_s[i, j] = -transfer
                else:
                    transfer = ks * (capacity - sediment[i, j])
                    room = height[i, j] - h_min
                    if transfer > room:
                        transfer = room
                    if transfer < 0.0:
                        transfer = 0.0
                    delta_h[i, j] = -transfer
                    delta_s[i, j] = transfer
        # 8 应用地形与悬移质增量
        for i in prange(nlat):
            for j in range(nlon):
                if not land[i, j]:
                    continue
                height[i, j] += delta_h[i, j]
                sediment[i, j] += delta_s[i, j]
                if sediment[i, j] < 0.0:
                    sediment[i, j] = 0.0
        # 9 沉积物输运（半拉格朗日 + MacCormack）
        _advect_pass(sediment, s1, adv_lo, adv_hi, vx, vy, -1.0, dt, dlat_m, dlon_m, land)
        if macormack:
            _advect_pass(s1, s2, tmp_a, tmp_b, vx, vy, 1.0, dt, dlat_m, dlon_m, land)
            for i in prange(nlat):
                for j in range(nlon):
                    if not land[i, j]:
                        sediment[i, j] = 0.0
                        continue
                    value = s1[i, j] + 0.5 * (sediment[i, j] - s2[i, j])
                    if value < adv_lo[i, j]:
                        value = adv_lo[i, j]
                    elif value > adv_hi[i, j]:
                        value = adv_hi[i, j]
                    if value < 0.0:
                        value = 0.0
                    sediment[i, j] = value
        else:
            for i in prange(nlat):
                for j in range(nlon):
                    sediment[i, j] = s1[i, j] if land[i, j] else 0.0
        # 10 自适应时间步（用上一步流速场估计）
        if adaptive:
            vmax = 0.0
            for i in range(nlat):
                for j in range(nlon):
                    if speed[i, j] > vmax:
                        vmax = speed[i, j]
            if vmax < 1.0e-9:
                dt = dt_max
            else:
                dt = cfl * dx_min / vmax
            if dt < dt_min:
                dt = dt_min
            elif dt > dt_max:
                dt = dt_max
    return dt


# ===== 顶层接口 =====


@dataclasses.dataclass(frozen=True)
class HydraulicErosionResult:
    """水力侵蚀输出（§七 第 3 步）。"""

    elevation: np.ndarray  # 侵蚀后地形 H (m)
    drop: np.ndarray  # ΔH_hydraulic = H_in − H_out (m)，正值表示净下切
    water_depth: np.ndarray  # 末态水深 d (m)
    sediment: np.ndarray  # 末态悬移质含量 s (m)
    report: dict[str, float | int | str]


def hydraulic_erode(
    elevation: np.ndarray,
    is_ocean: np.ndarray,
    *,
    precipitation: float | np.ndarray,
    n_steps: int = 200,
    dt: float | None = None,
    pipe_coefficient: float = PIPE_COEFFICIENT,
    capacity_constant: float = CAPACITY_CONSTANT,
    erosion_rate: float = EROSION_RATE,
    deposition_rate: float = DEPOSITION_RATE,
    critical_velocity: float = CRITICAL_VELOCITY,
    max_velocity: float = MAX_VELOCITY,
    macormack: bool = True,
    cfl: float = CFL_SAFETY,
    radius: float = EARTH_RADIUS,
    gravity: float = EARTH_GRAVITY,
) -> HydraulicErosionResult:
    """虚拟管道水力侵蚀（§一）。

    参数：
        elevation: 输入高程场 H (m)
        is_ocean: 海洋掩码（海洋即水位基准面，不受侵蚀也不改变高程）
        precipitation: 降水速率 (m/s)，标量或逐格场
        n_steps: 时间步数（侵蚀总量近似随其线性增长）
        dt: 固定时间步 (s)；None 时用 CFL 自适应步长
        pipe_coefficient / capacity_constant / erosion_rate / deposition_rate:
            见模块常量说明
        critical_velocity: 侵蚀临界流速 u_c (m/s)（§1.4）。缺省由 Shields 准则按
            1 mm 中砂床反解（:func:`critical_velocity_from_shields`），也可传 0 关闭阈值
        max_velocity: 流速上限 (m/s)
        macormack: 是否启用 MacCormack 反扩散输运（§5.3）
        cfl / radius / gravity: 稳定性与几何参数

    海洋单元的高程与悬移质严格不变（海岸是物质汇，不是沉积区——否则长时间
    积分会把海盆填平）。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    if elev.shape != ocean.shape:
        raise ValueError(f"elevation 与 is_ocean 形状不一致: {elev.shape} vs {ocean.shape}")
    nlat, nlon = elev.shape
    if nlat < 2 or nlon < 2:
        raise ValueError(f"网格至少 2x2，实际 {elev.shape}")
    if n_steps < 1:
        raise ValueError(f"n_steps 必须 >= 1，实际为 {n_steps}")
    if not 0.0 <= erosion_rate <= 1.0 or not 0.0 <= deposition_rate <= 1.0:
        raise ValueError(
            f"侵蚀/沉积速率必须在 [0, 1] 内（每步松弛系数），实际 {erosion_rate}, {deposition_rate}"
        )
    if capacity_constant < 0.0:
        raise ValueError(f"capacity_constant 不能为负，实际为 {capacity_constant}")
    if pipe_coefficient <= 0.0:
        raise ValueError(f"pipe_coefficient 必须为正，实际为 {pipe_coefficient}")
    if max_velocity <= 0.0 or critical_velocity < 0.0:
        raise ValueError("max_velocity 必须为正、critical_velocity 不能为负")
    if not 0.0 < cfl <= 1.0:
        raise ValueError(f"cfl 必须在 (0, 1] 内，实际为 {cfl}")
    if dt is not None and dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")

    precip = np.asarray(precipitation, dtype=np.float64)
    if precip.ndim == 0:
        if float(precip) < 0.0:
            raise ValueError(f"precipitation 不能为负，实际为 {float(precip)}")
        precip = np.full((nlat, nlon), float(precip), dtype=np.float64)
    elif precip.shape != (nlat, nlon):
        raise ValueError(f"precipitation 形状 {precip.shape} 与网格 {elev.shape} 不一致")
    elif float(precip.min()) < 0.0:
        raise ValueError("precipitation 不能为负")

    dlat_m, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    area = spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)
    # 管流的口径标定：Q = C_pipe·A_face·sign(Δh)·√(2g|Δh|)
    pipe = pipe_coefficient * np.sqrt(2.0 * gravity)

    height = np.array(elev, dtype=np.float64, copy=True)
    depth = np.zeros((nlat, nlon), dtype=np.float64)
    sediment = np.zeros((nlat, nlon), dtype=np.float64)
    land = np.ascontiguousarray(~ocean)
    if dt is None:
        dt0 = float(
            np.clip(cfl * min(dlat_m, float(dlon_m.min())) / max_velocity, DT_MIN_S, DT_MAX_S)
        )
        adaptive = True
    else:
        dt0 = float(dt)
        adaptive = False
    dt_final = float(
        _hydraulic_loop(
            height,
            depth,
            sediment,
            land,
            np.ascontiguousarray(area),
            float(dlat_m),
            np.ascontiguousarray(dlon_m),
            np.ascontiguousarray(precip),
            float(pipe),
            float(capacity_constant),
            float(erosion_rate),
            float(deposition_rate),
            float(critical_velocity),
            float(max_velocity),
            dt0,
            DT_MIN_S,
            DT_MAX_S,
            float(cfl),
            adaptive,
            int(n_steps),
            bool(macormack),
        )
    )

    drop = elev - height
    drop_land = np.where(land, drop, 0.0)
    report: dict[str, float | int | str] = {
        "n_steps": int(n_steps),
        "dt_s": dt_final,
        "adaptive_dt": adaptive,
        "macormack": bool(macormack),
        "total_incision_m3": float(np.sum(np.maximum(drop_land, 0.0) * area)),
        "total_deposition_m3": float(np.sum(np.maximum(-drop_land, 0.0) * area)),
        "mean_incision_m": float(np.sum(drop_land[land] * area[land]) / max(area[land].sum(), 1.0))
        if land.any()
        else 0.0,
        "max_abs_elevation_change_m": float(np.abs(drop_land).max()) if land.any() else 0.0,
        "max_water_depth_m": float(depth.max()),
        "max_sediment_m": float(sediment.max()),
    }
    return HydraulicErosionResult(
        elevation=height,
        drop=drop,
        water_depth=depth,
        sediment=sediment,
        report=report,
    )


__all__ = [
    "BED_FRICTION_COEFFICIENT",
    "CAPACITY_CONSTANT",
    "CFL_SAFETY",
    "CRITICAL_VELOCITY",
    "DENSITY_SEDIMENT",
    "DEPOSITION_RATE",
    "DT_MAX_S",
    "DT_MIN_S",
    "EROSION_RATE",
    "GRAIN_SIZE_M",
    "MAX_VELOCITY",
    "PIPE_COEFFICIENT",
    "SHIELDS_PARAMETER",
    "HydraulicErosionResult",
    "advect_sediment",
    "critical_velocity_from_shields",
    "hydraulic_erode",
    "shields_critical_shear",
]
