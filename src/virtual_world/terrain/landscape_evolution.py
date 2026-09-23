"""§2.1 河流功率定律与 §2.2 景观演化方程（《第三层完善：侵蚀模拟》§二）。

本模块实现方案 §2.2 的完整景观演化方程：

    dH/dt = U - K*A^m*S^n + kappa*nabla^2 H

其中

- **U**：构造抬升速率 (m/Ma)，来自第一层板块构造的 ``uplift_rate_m_per_ma`` 逐格场
  ——这是方案要求的跨层因果链（抬升与下切在同一时间导数里竞争）；
- **K*A^m*S^n**：河流功率定律（§2.1）的基岩河道下切，``A`` 为汇水面积 (m²)、
  ``S`` 为河道坡度、``m/n = 0.5``、``n`` 取 1 或 2；
- **kappa*nabla^2 H**：坡面扩散（§3.2 线性 / §3.3 非线性）。方案 §2.2 明确指出该项
  **不可省略**：忽略它时 ``m/n = 0.5`` 会让稳态景观对水平拉伸不变（10 km² 的景观
  可以拉伸成 1000 km² 的"同一景观"），坡面项打破了这种尺度不变性。

与 :func:`virtual_world.terrain.erosion.simulate_erosion` 的分工：后者是方案 §七 的
"微观刻蚀"管线（100 m–1 km 尺度、秒级时间步、管道模型+MacCormack），本模块是
**Ma 尺度的长期景观演化**（显式欧拉 + D8 汇流，见 §5.1）。两者可串联使用：先用本模块
在构造基底上积分出长期河网与坡面，再用微观管线做局部精修。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core import spherical
from ..core.constants import EARTH_RADIUS
from . import hydrology as hd
from . import thermal_erosion as te

#: 河流功率定律指数（§2.1：m/n = 0.5，n 取 1）
DEFAULT_M = 0.5
DEFAULT_N = 1.0
#: 侵蚀系数 K 的标定值（量纲 1/Ma，当 m=0.5、n=1、A 用 m²）。
#: 标定依据：参考河道 A=10^10 m²（10^4 km² 量级流域）、S=10^-2 ⇒ 下切约 50 m/Ma
#: （活跃造山带的量级）。方案 §2.1 指出 K 可跨 4–5 个数量级（取决于岩性与气候），
#: 因此这里只给出一个可解释的默认值，实际使用应按岩性调整。
DEFAULT_K = 0.05
#: 坡面扩散系数 kappa (m²/Ma)：方案 §3.2 的 4e-4–40e-4 m²/yr 换算为 400–40000 m²/Ma
DEFAULT_KAPPA_M2_PER_MA = 4.0e3
#: 单步下切不得超过到下游单元高差的比例（显式格式的稳定性保护，防止河道反转）
DEFAULT_INCISION_LIMIT = 0.5


@dataclasses.dataclass(frozen=True)
class LandscapeEvolutionResult:
    """景观演化输出（§2.2）。"""

    elevation: np.ndarray  # 末态高程 H (m)
    uplift_total: np.ndarray  # 累计构造抬升 (m)，海洋为 0
    incision_total: np.ndarray  # 累计河流下切 (m，正值表示降低)
    diffusion_total: np.ndarray  # 累计坡面扩散净变化 (m，可正可负)
    flow_accumulation: np.ndarray  # 末态汇流累积（单元数，海洋为 0）
    report: dict[str, float | int | str]
    seed: int


def stream_power_incision(
    elevation: np.ndarray,
    accumulation: np.ndarray,
    direction: np.ndarray,
    *,
    coefficient: float = DEFAULT_K,
    m: float = DEFAULT_M,
    n: float = DEFAULT_N,
    dt: float = 1.0,
    cell_area: np.ndarray | float,
    is_ocean: np.ndarray | None = None,
    incision_limit: float = DEFAULT_INCISION_LIMIT,
    radius: float = EARTH_RADIUS,
) -> np.ndarray:
    """河流功率定律下切量 ``E*dt = K*A^m*S^n*dt`` (m)（§2.1）。

    ``A`` 由汇流累积换算为汇水面积 (m²)，``S`` 取该单元沿 D8 最陡下降方向的坡度
    (m/m)。海洋与无出流单元不下切；单步下切被限制在到下游高差的 ``incision_limit``
    倍以内（显式积分的稳定性保护，避免河道被挖到低于下游）。
    """
    height = np.asarray(elevation, dtype=np.float64)
    acc = np.asarray(accumulation, dtype=np.float64)
    dirs = np.asarray(direction, dtype=np.int16)
    if height.shape != acc.shape or height.shape != dirs.shape:
        raise ValueError("elevation / accumulation / direction 形状必须一致")
    if coefficient < 0.0:
        raise ValueError(f"coefficient 不能为负，实际为 {coefficient}")
    if m <= 0.0 or n <= 0.0:
        raise ValueError(f"m 与 n 必须为正，实际为 {m}, {n}")
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")
    if not 0.0 <= incision_limit <= 1.0:
        raise ValueError(f"incision_limit 必须在 [0, 1] 内，实际为 {incision_limit}")

    nlat, nlon = height.shape
    dlat_m, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    area = np.broadcast_to(np.asarray(cell_area, dtype=np.float64), height.shape)

    drop = np.zeros_like(height)
    slope = np.zeros_like(height)
    for code, (di, dj) in hd.D8_OFFSETS.items():
        mask = dirs == code
        if not mask.any():
            continue
        rows, cols = np.nonzero(mask)
        nrows = rows + di
        keep = (nrows >= 0) & (nrows < nlat)
        if not keep.any():
            continue
        rows, cols, nrows = rows[keep], cols[keep], nrows[keep]
        ncols = (cols + dj) % nlon
        step = (
            abs(dlat_m) if dj == 0 else (dlon_m[rows] if di == 0 else np.hypot(dlat_m, dlon_m[rows]))
        )
        local_drop = height[rows, cols] - height[nrows, ncols]
        drop[rows, cols] = np.maximum(local_drop, 0.0)
        slope[rows, cols] = np.maximum(local_drop, 0.0) / step

    drainage = acc * area
    erosion = coefficient * np.power(np.maximum(drainage, 0.0), m) * np.power(slope, n) * dt
    erosion = np.minimum(erosion, incision_limit * drop)
    ocean = np.zeros_like(height, dtype=bool) if is_ocean is None else np.asarray(is_ocean, dtype=bool)
    return np.asarray(np.where(ocean, 0.0, np.maximum(erosion, 0.0)), dtype=np.float64)


def landscape_evolve(
    elevation: np.ndarray,
    is_ocean: np.ndarray,
    *,
    uplift: float | np.ndarray = 0.0,
    coefficient: float = DEFAULT_K,
    m: float = DEFAULT_M,
    n: float = DEFAULT_N,
    kappa: float = DEFAULT_KAPPA_M2_PER_MA,
    dt_ma: float = 0.1,
    n_steps: int = 10,
    update_flow_every: int = 1,
    nonlinear_diffusion: bool = False,
    talus_angle_deg: float = te.DEFAULT_TALUS_ANGLE_DEG,
    min_elevation: float = 0.0,
    radius: float = EARTH_RADIUS,
    seed: int = 0,
) -> LandscapeEvolutionResult:
    """显式积分景观演化方程（§2.2），返回末态高程与各项累计贡献。

    参数：
        elevation: 初始高程 H (m)
        is_ocean: 海洋掩码（海洋不抬升、不下切，固定海陆边界）
        uplift: 构造抬升速率 U (m/Ma)，标量或逐格场；第一层构造层给出
            :attr:`~virtual_world.terrain.plate_tectonics.TectonicFieldResult.uplift_rate_m_per_ma`
        coefficient / m / n: 河流功率定律参数（§2.1）
        kappa: 坡面扩散系数 (m²/Ma)（§3.2 线性 / §3.3 非线性）
        dt_ma / n_steps: 时间步与步数（显式欧拉，§5.1）
        update_flow_every: 每多少步重算一次 D8 流向与汇流累积（1 = 每步都算；
            调大可显著加速长期积分）
        nonlinear_diffusion / talus_angle_deg: 传给 :func:`thermal_erosion.hillslope_diffusion`
        min_elevation: 判定海洋的高程阈值
        radius / seed: 行星半径与记录用种子（本函数确定性，无随机成分）

    返回 :class:`LandscapeEvolutionResult`；同一输入 + 参数 ⇒ 逐位相同输出。
    """
    height = np.asarray(elevation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    if height.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {height.shape}")
    if height.shape != ocean.shape:
        raise ValueError(f"elevation 与 is_ocean 形状不一致: {height.shape} vs {ocean.shape}")
    if dt_ma <= 0.0:
        raise ValueError(f"dt_ma 必须为正，实际为 {dt_ma}")
    if n_steps < 1:
        raise ValueError(f"n_steps 必须 >= 1，实际为 {n_steps}")
    if update_flow_every < 1:
        raise ValueError(f"update_flow_every 必须 >= 1，实际为 {update_flow_every}")
    if kappa < 0.0:
        raise ValueError(f"kappa 不能为负，实际为 {kappa}")
    uplift_field = np.broadcast_to(np.asarray(uplift, dtype=np.float64), height.shape)
    if np.any(uplift_field < 0.0):
        raise ValueError("uplift 不能为负（本模型不模拟沉降）")

    height = np.array(height, dtype=np.float64, copy=True)
    nlat, nlon = height.shape
    cell_area = spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)
    uplift_total = np.zeros_like(height)
    incision_total = np.zeros_like(height)
    diffusion_total = np.zeros_like(height)

    direction = np.zeros((nlat, nlon), dtype=np.int16)
    accumulation = np.zeros((nlat, nlon), dtype=np.float64)
    for step in range(int(n_steps)):
        if step % int(update_flow_every) == 0:
            filled = hd.fill_pits(height, ocean)
            direction = hd.flow_directions(filled, ocean, radius=radius)
            accumulation = hd.flow_accumulation(direction, ocean)

        # U：构造抬升（仅陆地；洋壳由第一层的洋脊热沉降决定，不参与抬升）
        gain = np.where(ocean, 0.0, uplift_field * dt_ma)
        height += gain
        uplift_total += gain

        # -K*A^m*S^n：河流下切
        incision = stream_power_incision(
            height,
            accumulation,
            direction,
            coefficient=coefficient,
            m=m,
            n=n,
            dt=dt_ma,
            cell_area=cell_area,
            is_ocean=ocean,
            radius=radius,
        )
        height -= incision
        incision_total += incision

        # +kappa*nabla^2 H：坡面扩散（§3.2 线性 / §3.3 非线性）
        before = height
        height = te.hillslope_diffusion(
            height,
            kappa,
            dt_ma,
            is_ocean=ocean,
            radius=radius,
            nonlinear=nonlinear_diffusion,
            talus_angle_deg=talus_angle_deg,
        )
        diffusion_total += height - before

    land = ~ocean
    report: dict[str, float | int | str] = {
        "n_steps": int(n_steps),
        "dt_ma": float(dt_ma),
        "coefficient": float(coefficient),
        "m": float(m),
        "n": float(n),
        "kappa_m2_per_ma": float(kappa),
        "nonlinear_diffusion": bool(nonlinear_diffusion),
        "update_flow_every": int(update_flow_every),
        "mean_uplift_m": float(np.sum(uplift_total[land] * cell_area[land]) / max(cell_area[land].sum(), 1.0))
        if land.any()
        else 0.0,
        "total_uplift_m3": float(np.sum(uplift_total * cell_area)),
        "total_incision_m3": float(np.sum(incision_total * cell_area)),
        "total_diffusion_m3": float(np.sum(diffusion_total * cell_area)),
        "max_incision_m": float(incision_total.max()) if incision_total.size else 0.0,
    }
    return LandscapeEvolutionResult(
        elevation=height,
        uplift_total=uplift_total,
        incision_total=incision_total,
        diffusion_total=diffusion_total,
        flow_accumulation=accumulation,
        report=report,
        seed=int(seed),
    )


__all__ = [
    "DEFAULT_INCISION_LIMIT",
    "DEFAULT_K",
    "DEFAULT_KAPPA_M2_PER_MA",
    "DEFAULT_M",
    "DEFAULT_N",
    "LandscapeEvolutionResult",
    "landscape_evolve",
    "stream_power_incision",
]
