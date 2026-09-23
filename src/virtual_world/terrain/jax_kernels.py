"""§6.3/§6.4 JAX 内核（可选依赖）。

方案 §6.4：当模拟时间尺度达到数百万年、需要数千个时间步时，JAX 的 ``jit`` +
``lax.scan`` 会把整个时间循环编译成 XLA 计算图，消除 Python 循环开销，并在 GPU
上获得极高吞吐（对比见《第一层完善》§6.5 加速分工表）。

本模块提供与 Numba 内核 :func:`virtual_world.terrain.isostasy.integrate_crust_thickness`
**数值一致**的实现，采用同一分裂格式（先增厚/减薄、再用更新后的厚度做重力松弛）。

方案 §6.3 进一步要求把 **水力侵蚀的管道模型**（第三层）也做成 JAX 内核：邻居访问用
:func:`jax.numpy.roll`（经度环绕、纬度不环绕），时间积分用 :func:`jax.lax.scan`，从而
在高分辨率网格上把数百步的显式循环整体编译为 XLA 计算图（自适应时间步作为 ``scan``
的 carry 之一随步更新）。:func:`hydraulic_erode_jax` 与 Numba 实现
:func:`virtual_world.terrain.hydraulic_erosion.hydraulic_erode` 逐格数值等价。

JAX 是可选依赖：本模块顶层不导入 jax，缺失时 :func:`integrate_crust_thickness_jax` /
:func:`hydraulic_erode_jax` 抛出含安装指引的 :class:`RuntimeError`，而不是静默退化为
Numba——静默降级会让调用方以为拿到了 GPU 加速的结果。

精度：JAX 默认关闭 64 位浮点（float32），与本项目地形层的 float64 约定不符，因此两个
内核都会先调用 :func:`enable_x64` 打开 x64（该开关幂等，是 JAX 的既定用法）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import spherical
from ..core.constants import EARTH_GRAVITY, EARTH_RADIUS
from . import hydraulic_erosion as he
from .hydraulic_erosion import HydraulicErosionResult


def jax_available() -> bool:
    """JAX 是否可用（供调用方选择内核，避免依赖异常做流程控制）。"""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


def _require_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - 取决于可选依赖
        raise RuntimeError(
            "JAX 内核需要安装 jax（pip install jax）；"
            "也可改用 Numba 内核 isostasy.integrate_crust_thickness"
        ) from exc
    return jax, jnp


def enable_x64() -> bool:
    """打开 JAX 的 64 位浮点（幂等）；JAX 不可用时返回 ``False``。"""
    if not jax_available():
        return False
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    return True


def integrate_crust_thickness_jax(
    crust: np.ndarray,
    thickening: np.ndarray | float,
    thinning: np.ndarray | float,
    c_eq: np.ndarray,
    *,
    kappa: float = 0.0,
    dt: float = 1.0,
    n_steps: int = 1,
) -> np.ndarray:
    """地壳厚度显式时间积分的 JAX 实现（方案 §6.4），返回 NumPy 数组。

    参数与 :func:`virtual_world.terrain.isostasy.integrate_crust_thickness` 一致，
    数值结果相同；差别只在于时间循环由 ``jax.lax.scan`` 编译为 XLA 计算图
    （可放到 GPU 上跑数千步）。

    分裂格式（与 Numba 内核一致）：``C += (T - S)*dt`` 之后再
    ``C -= kappa*(C - C_eq)*dt``，因此该格式的定点为 ``C_eq + T*(1-kappa*dt)/kappa``
    （松弛作用在增厚之后的厚度上），而非 ``C_eq + T/kappa``。
    """
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")
    if n_steps <= 0:
        raise ValueError(f"n_steps 必须为正，实际为 {n_steps}")

    jax, jnp = _require_jax()
    enable_x64()
    base = np.asarray(crust, dtype=np.float64)
    if base.ndim < 1:
        raise ValueError(f"crust 至少需要 1 维，实际 {base.shape}")
    thick = jnp.asarray(np.broadcast_to(np.asarray(thickening, dtype=np.float64), base.shape))
    thin = jnp.asarray(np.broadcast_to(np.asarray(thinning, dtype=np.float64), base.shape))
    equilibrium = jnp.asarray(np.broadcast_to(np.asarray(c_eq, dtype=np.float64), base.shape))
    state = jnp.asarray(base)
    step_dt = float(dt)
    relax = float(kappa)

    def step(current: Any, _: Any) -> tuple[Any, None]:
        current = current + (thick - thin) * step_dt
        current = current - relax * (current - equilibrium) * step_dt
        return current, None

    result, _ = jax.lax.scan(step, state, None, length=int(n_steps))
    return np.asarray(jax.block_until_ready(result), dtype=np.float64)


# ===== §6.3 水力侵蚀管道模型（JAX）=====


def _build_hydraulic_scan(jax: Any, jnp: Any) -> Any:
    """构建水力侵蚀时间循环并 ``jit``，返回可调用内核。

    方案 §6.3 的硬性要求：邻居访问一律用 :func:`jax.numpy.roll`（经度环绕、纬度靠
    掩码截断），时间积分用 :func:`jax.lax.scan`（自适应时间步作为 carry 之一）。
    计算图只依赖网格**形状**，物理常数与场都以参数传入，因此同一形状下只编译一次。
    运算顺序刻意与 Numba 内核逐条对应（含 ``jnp.where`` 代替 ``if``），以保证数值等价。
    """

    def bilinear(field: Any, fi: Any, fj: Any) -> tuple[Any, Any, Any]:
        """双线性采样，返回 ``(值, 四角最小值, 四角最大值)``（语义同 Numba 的 ``_bilinear``）。

        纬度索引越界**钳制**（不环绕，极点处退化为边界外推），经度索引**取模环绕**；
        四角极值供 MacCormack 通量限制器使用。运算顺序与 Numba 实现逐位一致。
        """
        nlat, nlon = field.shape
        i0 = jnp.floor(fi)
        j0 = jnp.floor(fj)
        wi = fi - i0
        wj = fj - j0
        i0i = i0.astype(jnp.int32)
        j0i = j0.astype(jnp.int32)
        i0c = jnp.clip(i0i, 0, nlat - 1)
        i1c = jnp.clip(i0i + 1, 0, nlat - 1)
        j0c = jnp.mod(j0i, nlon)
        j1c = jnp.mod(j0i + 1, nlon)
        f00 = field[i0c, j0c]
        f01 = field[i0c, j1c]
        f10 = field[i1c, j0c]
        f11 = field[i1c, j1c]
        top = f00 + (f01 - f00) * wj
        bottom = f10 + (f11 - f10) * wj
        value = top + (bottom - top) * wi
        lo = jnp.minimum(jnp.minimum(f00, f01), jnp.minimum(f10, f11))
        hi = jnp.maximum(jnp.maximum(f00, f01), jnp.maximum(f10, f11))
        return value, lo, hi

    def advect(
        field: Any,
        vx: Any,
        vy: Any,
        veer: float,
        dt: Any,
        dlat_m: float,
        dlon: Any,
        land: Any,
        fi_row: Any,
        fj_col: Any,
    ) -> tuple[Any, Any, Any]:
        """半拉格朗日单次对流：``veer=-1`` 反向回溯（求 ``s'``）、``+1`` 正向。"""
        fi = fi_row + veer * vy * dt / dlat_m
        fj = fj_col + veer * vx * dt / dlon
        value, lo, hi = bilinear(field, fi, fj)
        return (
            jnp.where(land, value, 0.0),
            jnp.where(land, lo, 0.0),
            jnp.where(land, hi, 0.0),
        )

    def run(
        height: Any,
        depth: Any,
        sediment: Any,
        dt: Any,
        land: Any,
        area: Any,
        dlon: Any,
        precip: Any,
        area_east: Any,
        face_north: Any,
        *,
        dlat_m: float,
        pipe: float,
        kc: float,
        ks: float,
        kd: float,
        u_c: float,
        v_cap: float,
        cfl: float,
        dt_min: float,
        dt_max: float,
        dx_min: float,
        n_steps: int,
        macormack: bool,
        adaptive: bool,
    ) -> tuple[Any, Any, Any, Any]:
        """编译后的显式时间循环，返回末态 ``(height, depth, sediment, dt)``。"""
        nlat, nlon = height.shape
        fi_row = jnp.arange(nlat, dtype=jnp.float64)[:, None]
        fj_col = jnp.arange(nlon, dtype=jnp.float64)[None, :]
        rows = jnp.arange(nlat, dtype=jnp.int32)[:, None]
        north_ok = rows + 1 < nlat  # 纬度方向不环绕：末行没有北向面
        south_ok = rows - 1 >= 0  # 首行没有南向邻居（用于最低邻居掩码）
        row_first = rows == 0
        row_last = rows == nlat - 1
        two_dlon = 2.0 * dlon
        min_wdepth = he.MIN_WATER_DEPTH

        def step(carry: Any, _: Any) -> tuple[Any, Any]:
            height, depth, sediment, dt = carry
            # 1 降水补给（只有陆地接受降水）
            depth = jnp.where(land, depth + precip * dt, depth)
            # 2 管道流量：每个面只由西/南侧单元计算一次，符号表示净输运方向
            head = height + depth
            dh_east = head - jnp.roll(head, -1, axis=1)
            f_east = jnp.where(
                land, pipe * area_east * jnp.sign(dh_east) * jnp.sqrt(jnp.abs(dh_east)), 0.0
            )
            dh_north = head - jnp.roll(head, -1, axis=0)
            f_north = jnp.where(
                land & north_ok,
                pipe * face_north * jnp.sign(dh_north) * jnp.sqrt(jnp.abs(dh_north)),
                0.0,
            )
            # 3 出流钳制因子：保证本步排水量不超过现有水量
            e_west = jnp.roll(f_east, 1, axis=1)
            n_south = jnp.roll(f_north, 1, axis=0)  # 首行环绕值恒为 0（末行北向通量为 0）
            outflow = jnp.where(f_east > 0.0, f_east, 0.0)
            outflow = outflow - jnp.where(e_west < 0.0, e_west, 0.0)
            outflow = outflow + jnp.where(f_north > 0.0, f_north, 0.0)
            outflow = outflow - jnp.where(n_south < 0.0, n_south, 0.0)
            limit = depth * area / jnp.where(outflow > 0.0, dt * outflow, 1.0)
            scale = jnp.where(land & (outflow > 0.0), jnp.minimum(1.0, limit), 1.0)
            # 4 面通量按失水方的因子缩放（仍是两侧共享，体积守恒不破坏）
            f_east = jnp.where(
                f_east > 0.0,
                f_east * scale,
                jnp.where(f_east < 0.0, f_east * jnp.roll(scale, -1, axis=1), f_east),
            )
            f_north = jnp.where(
                f_north > 0.0,
                f_north * scale,
                jnp.where(f_north < 0.0, f_north * jnp.roll(scale, -1, axis=0), f_north),
            )
            # 5 水量更新
            e_west = jnp.roll(f_east, 1, axis=1)
            n_south = jnp.roll(f_north, 1, axis=0)
            net = e_west - f_east - f_north + n_south
            depth = jnp.where(land, jnp.maximum(depth + dt * net / area, 0.0), 0.0)
            # 6 流速场（净流量 /(水深 × 面宽)），并施加速度上限
            qx = f_east - e_west
            qy = f_north - n_south
            wet = land & (depth > min_wdepth)
            dep = jnp.where(wet, depth, 1.0)
            ux = qx / (dep * dlat_m)
            uy = qy / (dep * dlon)
            sp = jnp.sqrt(ux * ux + uy * uy)
            shrink = jnp.where(sp > v_cap, v_cap / jnp.where(sp > 0.0, sp, 1.0), 1.0)
            vx = jnp.where(wet, ux * shrink, 0.0)
            vy = jnp.where(wet, uy * shrink, 0.0)
            speed = jnp.where(wet, jnp.where(sp > v_cap, v_cap, sp), 0.0)
            # 7 侵蚀/沉积（先算增量再统一应用，避免并行读写竞态）
            h_east = jnp.roll(height, -1, axis=1)
            h_west = jnp.roll(height, 1, axis=1)
            h_north = jnp.roll(height, -1, axis=0)
            h_south = jnp.roll(height, 1, axis=0)
            h_min = jnp.minimum(
                jnp.minimum(h_east, h_west),
                jnp.minimum(
                    jnp.where(north_ok, h_north, jnp.inf),
                    jnp.where(south_ok, h_south, jnp.inf),
                ),
            )
            hx = (h_east - h_west) / two_dlon
            hy = jnp.where(
                row_first,
                (h_north - height) / dlat_m,
                jnp.where(
                    row_last,
                    (height - h_south) / dlat_m,
                    (h_north - h_south) / (2.0 * dlat_m),
                ),
            )
            slope = jnp.sqrt(hx * hx + hy * hy)
            active = jnp.maximum(speed - u_c, 0.0)
            capacity = kc * slope * active * depth
            room = height - h_min
            deposit = sediment > capacity
            transfer_d = kd * (sediment - capacity)
            transfer_e = jnp.maximum(jnp.minimum(ks * (capacity - sediment), room), 0.0)
            delta_h = jnp.where(deposit, transfer_d, -transfer_e)
            delta_s = jnp.where(deposit, -transfer_d, transfer_e)
            # 8 应用地形与悬移质增量
            height = jnp.where(land, height + delta_h, height)
            sediment = jnp.where(land, jnp.maximum(sediment + delta_s, 0.0), 0.0)
            # 9 沉积物输运（半拉格朗日回溯 + MacCormack 反扩散与通量限制）
            s1, adv_lo, adv_hi = advect(
                sediment, vx, vy, -1.0, dt, dlat_m, dlon, land, fi_row, fj_col
            )
            if macormack:
                s2, _, _ = advect(s1, vx, vy, 1.0, dt, dlat_m, dlon, land, fi_row, fj_col)
                value = s1 + 0.5 * (sediment - s2)
                value = jnp.where(value < adv_lo, adv_lo, jnp.where(value > adv_hi, adv_hi, value))
                value = jnp.maximum(value, 0.0)
                sediment = jnp.where(land, value, 0.0)
            else:
                sediment = jnp.where(land, s1, 0.0)
            # 10 自适应时间步（用本步流速场上限估计，随 carry 传给下一步）
            if adaptive:
                vmax = jnp.max(speed)
                dt = jnp.where(
                    vmax < 1.0e-9, dt_max, cfl * dx_min / jnp.where(vmax > 0.0, vmax, 1.0)
                )
                dt = jnp.where(dt < dt_min, dt_min, jnp.where(dt > dt_max, dt_max, dt))
            return (height, depth, sediment, dt), None

        (height, depth, sediment, dt), _ = jax.lax.scan(
            step, (height, depth, sediment, dt), None, length=int(n_steps)
        )
        return height, depth, sediment, dt

    return jax.jit(run, static_argnames=("n_steps", "macormack", "adaptive"))


#: 编译后的扫描内核缓存（按进程复用，避免每次调用重新 trace/编译）
_SCAN_CACHE: dict[str, Any] = {}


def _hydraulic_scan(jax: Any, jnp: Any) -> Any:
    """取（必要时构建）编译后的水力侵蚀时间循环。"""
    kernel = _SCAN_CACHE.get("hydraulic")
    if kernel is None:
        kernel = _build_hydraulic_scan(jax, jnp)
        _SCAN_CACHE["hydraulic"] = kernel
    return kernel


def hydraulic_erode_jax(
    elevation: np.ndarray,
    is_ocean: np.ndarray,
    *,
    precipitation: float | np.ndarray,
    n_steps: int = 200,
    dt: float | None = None,
    pipe_coefficient: float = he.PIPE_COEFFICIENT,
    capacity_constant: float = he.CAPACITY_CONSTANT,
    erosion_rate: float = he.EROSION_RATE,
    deposition_rate: float = he.DEPOSITION_RATE,
    critical_velocity: float = he.CRITICAL_VELOCITY,
    max_velocity: float = he.MAX_VELOCITY,
    macormack: bool = True,
    cfl: float = he.CFL_SAFETY,
    radius: float = EARTH_RADIUS,
    gravity: float = EARTH_GRAVITY,
) -> HydraulicErosionResult:
    """虚拟管道水力侵蚀的 JAX 实现（方案 §6.3），返回 :class:`HydraulicErosionResult`。

    参数、默认值、校验与返回结构均与
    :func:`virtual_world.terrain.hydraulic_erosion.hydraulic_erode` 一致，数值结果
    逐格相同；差别只在于时间循环由 ``jax.lax.scan`` 编译为 XLA 计算图（可在 GPU 上
    跑数百/数千步），邻居访问用 ``jnp.roll``。步骤语义见
    :func:`virtual_world.terrain.hydraulic_erosion._hydraulic_loop` 的 1–10 步注释。

    数值等价性：固定 dt 时高程场与 Numba 内核**逐位相同**（实测 200 步内偏差 0）；
    自适应 dt 时同一地形、60 步内高程最大偏差 ~3e-14 m（相对 3000 m 地形约 1e-17）。
    残差来自 Numba 内核的 ``fastmath=True``（允许 LLVM 重排/近似、偏离严格 IEEE），
    本实现则与严格 IEEE 的 NumPy 逐步重放逐位一致。
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
    if dt is None:
        dt0 = float(
            np.clip(cfl * min(dlat_m, float(dlon_m.min())) / max_velocity, he.DT_MIN_S, he.DT_MAX_S)
        )
        adaptive = True
    else:
        dt0 = float(dt)
        adaptive = False
    # 面几何与网格最小间距（表达式与 Numba 内核内的逐格写法逐位一致）
    dx_min = min(float(dlat_m), float(dlon_m.min()))
    area_east = (float(dlat_m) * dlon_m).reshape(nlat, 1)
    face_north = (0.5 * float(dlat_m) * (dlon_m + np.roll(dlon_m, -1))).reshape(nlat, 1)
    dlon = dlon_m.reshape(nlat, 1)

    jax, jnp = _require_jax()
    enable_x64()
    kernel = _hydraulic_scan(jax, jnp)
    height, depth, sediment, dt_final = kernel(
        jnp.asarray(np.array(elev, dtype=np.float64, copy=True)),
        jnp.zeros((nlat, nlon), dtype=jnp.float64),
        jnp.zeros((nlat, nlon), dtype=jnp.float64),
        jnp.asarray(dt0, dtype=jnp.float64),
        jnp.asarray(np.ascontiguousarray(~ocean)),
        jnp.asarray(np.ascontiguousarray(area)),
        jnp.asarray(np.ascontiguousarray(dlon)),
        jnp.asarray(np.ascontiguousarray(precip)),
        jnp.asarray(np.ascontiguousarray(area_east)),
        jnp.asarray(np.ascontiguousarray(face_north)),
        dlat_m=float(dlat_m),
        pipe=float(pipe),
        kc=float(capacity_constant),
        ks=float(erosion_rate),
        kd=float(deposition_rate),
        u_c=float(critical_velocity),
        v_cap=float(max_velocity),
        cfl=float(cfl),
        dt_min=he.DT_MIN_S,
        dt_max=he.DT_MAX_S,
        dx_min=float(dx_min),
        n_steps=int(n_steps),
        macormack=bool(macormack),
        adaptive=adaptive,
    )
    height = np.asarray(jax.block_until_ready(height), dtype=np.float64)
    depth = np.asarray(jax.block_until_ready(depth), dtype=np.float64)
    sediment = np.asarray(jax.block_until_ready(sediment), dtype=np.float64)
    dt_final = float(np.asarray(jax.block_until_ready(dt_final), dtype=np.float64))

    land = np.ascontiguousarray(~ocean)
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
    "enable_x64",
    "hydraulic_erode_jax",
    "integrate_crust_thickness_jax",
    "jax_available",
]
