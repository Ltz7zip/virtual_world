"""第三层·侵蚀模拟管线编排（《第三层完善：侵蚀模拟》§七）。

    H_second + 海陆掩码 + 降水 P
      → ① 洼地填充（Priority-Flood，§4.2）
      → ② 构造抬升叠加（§2.2 的 U 项：按本次演化累计抬升量加到侵蚀基底上）
      → ③ 水力侵蚀时间循环（虚拟管道 + Shields 阈值 + MacCormack 输运，§一）
      → ④ 热力侵蚀时间循环（休止角滑移，§三）
      → ⑤ 坡面扩散（§3.2 线性 / §3.3 非线性；方案 §2.2 视为不可省的坡面项）
      → ⑥ 最终路由基底再填充 → D8 流向 → 汇流累积 → 河网 → 河宽（§四）
      → H_final + river_network + flow_accumulation

**相位恒等式**（本模块的核心不变量）：

    H_final = erosion_bed + uplift − ΔH_hydraulic − ΔH_thermal

其中 ``erosion_bed = fill_pits(H_second)``、``uplift`` 为本次演化叠加的构造抬升
（缺省 0）。方案 §七 把洼地填充排在侵蚀**之前**（第 2 步），因此侵蚀是在无洼地基底上
进行的——否则闭合盆地会截留水流、河网断裂。填充量（``erosion_bed − H_second``）在
:attr:`ErosionResult.report` 中有统计，通常只在少数内陆盆地上非零。

第 ⑥ 步用**再填充**后的 :attr:`ErosionResult.filled_elevation` 做流向计算：
沉积可以在侵蚀后的地形上重新造出闭合洼地，而 D8 要求无洼地。

与 :mod:`virtual_world.terrain.landscape_evolution` 的分工见该模块文档：本模块是
100 m–1 km 尺度的微观刻蚀（秒级时间步、管道模型），Ma 尺度的长期景观演化（含构造
抬升 U 与河流功率定律）走 :func:`landscape_evolution.landscape_evolve`。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core import spherical
from ..core.constants import DAYS_PER_YEAR, EARTH_RADIUS, SECONDS_PER_DAY
from ..core.grid import FIELDS
from . import hydraulic_erosion as he
from . import hydrology as hd
from . import thermal_erosion as te

#: 默认降水速率 (m/s)：地球年降水 1 m（后续由水循环模块的降水场替代）
DEFAULT_PRECIPITATION_M_PER_S = 1.0 / (DAYS_PER_YEAR * SECONDS_PER_DAY)
#: 默认水力侵蚀步数（侵蚀总量近似随其线性增长）
DEFAULT_HYDRAULIC_STEPS = 200
#: 默认热力侵蚀迭代次数
DEFAULT_THERMAL_ITERATIONS = te.DEFAULT_ITERATIONS
#: 默认坡面扩散时间步 (s)：约 116 天
DEFAULT_HILLSLOPE_DT_S = 1.0e7
#: 默认坡面扩散系数 (m²/s)：方案 §3.2 的 40e-4 m²/yr（区间上端）。
#: 方案 §2.2 明确指出坡面扩散项不可省略，因此**默认启用**；量级很小，
#: 粗网格（1°）上 κ·dt/Δx² ≈ 1e-13 几乎无影响，细网格或长时积分才逐步显现。
DEFAULT_HILLSLOPE_KAPPA = 1.27e-10


@dataclasses.dataclass(frozen=True)
class ErosionResult:
    """第三层输出（§七 第 9 步）。"""

    elevation: np.ndarray  # H_final (m)
    erosion_bed: np.ndarray  # 侵蚀基底 = fill_pits(H_second) (m)
    uplift: np.ndarray  # §2.2 的构造抬升 U 叠加量 (m)，缺省全 0
    filled_elevation: np.ndarray  # 最终路由基底 = fill_pits(H_final) (m)
    hydraulic_drop: np.ndarray  # ΔH_hydraulic = H_bed+uplift − H_after_hydraulic (m)
    thermal_drop: np.ndarray  # ΔH_thermal = H_after_hydraulic − H_final (m)
    flow_direction: np.ndarray  # ESRI D8 方向码 int16
    flow_accumulation: np.ndarray  # 上游单元数（含自身），海洋为 0
    river_network: np.ndarray  # 河流掩码 bool
    river_width: np.ndarray  # 河宽 m
    report: dict[str, float | int | str]
    seed: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def total_drop(self) -> np.ndarray:
        """总降低量 ``ΔH_hydraulic + ΔH_thermal`` (m)。"""
        return np.asarray(self.hydraulic_drop + self.thermal_drop, dtype=np.float64)


def simulate_erosion(
    elevation: np.ndarray,
    is_ocean: np.ndarray,
    *,
    precipitation: float | np.ndarray | None = None,
    uplift: float | np.ndarray = 0.0,
    seed: int = 0,
    hydraulic_steps: int = DEFAULT_HYDRAULIC_STEPS,
    hydraulic_dt: float | None = None,
    hydraulic_macormack: bool = True,
    critical_velocity: float = he.CRITICAL_VELOCITY,
    talus_angle_deg: float = te.DEFAULT_TALUS_ANGLE_DEG,
    thermal_iterations: int = DEFAULT_THERMAL_ITERATIONS,
    hillslope_kappa: float = DEFAULT_HILLSLOPE_KAPPA,
    hillslope_dt: float = DEFAULT_HILLSLOPE_DT_S,
    hillslope_steps: int = 1,
    hillslope_nonlinear: bool = False,
    min_river_cells: int = hd.DEFAULT_RIVER_CELLS,
    radius: float = EARTH_RADIUS,
) -> ErosionResult:
    """运行完整第三层侵蚀管线（§七 伪代码 1–9）。

    参数：
        elevation: 第二层输出高程 H_second (m)
        is_ocean: 海洋掩码
        precipitation: 降水速率 (m/s)，标量或逐格场；None 用
            :data:`DEFAULT_PRECIPITATION_M_PER_S`
        uplift: §2.2 的构造抬升 U：本次侵蚀演化期间累计叠加的抬升量 (m)，标量或
            逐格场，缺省 0。第一层给出的速率场（m/Ma）需先按演化时间换算：
            ``uplift = uplift_rate_m_per_ma * 演化时间(Ma)``；Ma 尺度的长期积分
            请直接用 :func:`landscape_evolution.landscape_evolve`
        seed: 记录用种子（本层确定性，不含随机成分）
        hydraulic_steps / hydraulic_dt: 水力侵蚀步数与步长（None = CFL 自适应）
        hydraulic_macormack: 沉积物输运用 MacCormack 反扩散
        critical_velocity: Shields 临界流速 u_c (m/s)（§1.4），缺省由 Shields 准则
            按 1 mm 中砂床反解；传 0 可关闭阈值
        talus_angle_deg / thermal_iterations: 热力侵蚀休止角与迭代次数
        hillslope_kappa: 坡面扩散系数 κ (m²/s)（§3.2），缺省启用（方案 §2.2 要求）；
            传 0 关闭。粗网格上量级极小，细网格或长时积分才显著
        hillslope_dt / hillslope_steps: 坡面扩散的时间步与步数
        hillslope_nonlinear: 用 §3.3 的非线性扩散（休止角约束下通量急剧放大）
        river_min_cells: 河道形成阈值（上游汇流单元数，§4.4）
        radius: 行星半径 (m)

    返回 :class:`ErosionResult`；同一输入 + 参数 ⇒ 逐位相同输出（确定性）。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    if elev.shape != ocean.shape:
        raise ValueError(f"elevation 与 is_ocean 形状不一致: {elev.shape} vs {ocean.shape}")
    if seed < 0:
        raise ValueError(f"seed 不能为负，实际为 {seed}")
    if hydraulic_steps < 1:
        raise ValueError(f"hydraulic_steps 必须 >= 1，实际为 {hydraulic_steps}")
    if thermal_iterations < 0:
        raise ValueError(f"thermal_iterations 不能为负，实际为 {thermal_iterations}")
    if hillslope_kappa < 0.0:
        raise ValueError(f"hillslope_kappa 不能为负，实际为 {hillslope_kappa}")
    if hillslope_kappa > 0.0 and hillslope_dt <= 0.0:
        raise ValueError(f"启用坡面扩散时 hillslope_dt 必须为正，实际为 {hillslope_dt}")
    if radius <= 0.0:
        raise ValueError(f"radius 必须为正，实际为 {radius}")
    rain = DEFAULT_PRECIPITATION_M_PER_S if precipitation is None else precipitation
    uplift_field = np.broadcast_to(np.asarray(uplift, dtype=np.float64), elev.shape)
    if np.any(uplift_field < 0.0):
        raise ValueError("uplift 不能为负（本层不含沉降）")

    nlat, nlon = elev.shape
    area = np.asarray(
        spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius), dtype=np.float64
    )

    # ① 洼地填充（§七 第 2 步）：无洼地基底是水力循环与 D8 的前提
    erosion_bed = hd.fill_pits(elev, ocean)
    fill_gain = erosion_bed - elev

    # ② 构造抬升（§2.2 的 U 项）：抬升后的面才是水力侵蚀的基底
    uplift_applied = np.where(ocean, 0.0, uplift_field)
    bed = erosion_bed + uplift_applied

    # ③ 水力侵蚀（§七 第 3 步）
    hydraulic = he.hydraulic_erode(
        bed,
        ocean,
        precipitation=rain,
        n_steps=int(hydraulic_steps),
        dt=hydraulic_dt,
        macormack=bool(hydraulic_macormack),
        critical_velocity=float(critical_velocity),
        radius=radius,
    )

    # ④ 热力侵蚀（§七 第 4 步）：休止角滑移
    thermal = te.thermal_erode(
        hydraulic.elevation,
        ocean,
        talus_angle_deg=float(talus_angle_deg),
        iterations=int(thermal_iterations),
        radius=radius,
    )
    final = thermal.elevation
    # ⑤ 坡面扩散（§3.2 线性 / §3.3 非线性）
    if hillslope_kappa > 0.0:
        final = te.hillslope_diffusion(
            final,
            float(hillslope_kappa),
            float(hillslope_dt),
            n_steps=int(hillslope_steps),
            is_ocean=ocean,
            radius=radius,
            nonlinear=bool(hillslope_nonlinear),
            talus_angle_deg=float(talus_angle_deg),
        )

    # ⑥ 河流网络（§七 第 5–8 步）：在**最终**地形上重新填充后做 D8
    hydro = hd.analyze_hydrology(final, ocean, river_min_cells=int(min_river_cells), radius=radius)

    hydraulic_drop = bed - hydraulic.elevation
    thermal_drop = hydraulic.elevation - final
    land = ~ocean
    report: dict[str, float | int | str] = {
        "hydraulic_steps": int(hydraulic_steps),
        "hydraulic_dt_s": float(hydraulic.report["dt_s"]),
        "critical_velocity_m_per_s": float(critical_velocity),
        "thermal_iterations": int(thermal_iterations),
        "talus_angle_deg": float(talus_angle_deg),
        "hillslope_kappa": float(hillslope_kappa),
        "hillslope_nonlinear": bool(hillslope_nonlinear),
        "pit_fill_volume_m3": float(np.sum(np.maximum(fill_gain, 0.0) * area)),
        "pit_fill_max_m": float(np.max(fill_gain)) if fill_gain.size else 0.0,
        "uplift_total_m3": float(np.sum(uplift_applied * area)),
        "hydraulic_incision_m3": float(np.sum(np.maximum(hydraulic_drop, 0.0) * area)),
        "thermal_move_m3": float(np.sum(np.abs(thermal_drop) * area)),
        "mean_land_drop_m": float(
            np.sum((hydraulic_drop + thermal_drop)[land] * area[land]) / max(area[land].sum(), 1.0)
        )
        if land.any()
        else 0.0,
        "max_total_drop_m": float(np.max(np.abs(hydraulic_drop + thermal_drop)))
        if fill_gain.size
        else 0.0,
        **hydro.report,
    }
    return ErosionResult(
        elevation=final,
        erosion_bed=erosion_bed,
        uplift=uplift_applied,
        filled_elevation=hydro.filled_elevation,
        hydraulic_drop=hydraulic_drop,
        thermal_drop=thermal_drop,
        flow_direction=hydro.flow_direction,
        flow_accumulation=hydro.flow_accumulation,
        river_network=hydro.river_network,
        river_width=hydro.river_width,
        report=report,
        seed=int(seed),
    )


def validate_erosion(
    result: ErosionResult,
    *,
    radius: float = EARTH_RADIUS,
    talus_angle_deg: float = te.DEFAULT_TALUS_ANGLE_DEG,
    slope_tol: float = 0.05,
    check_identity: bool = True,
) -> list[str]:
    """复核第三层输出是否满足方案约束，返回问题描述列表（空列表即通过）。

    检查项：
      - 各场有限性
      - ``H_final`` 落在 :data:`core.grid.FIELDS` 的 elevation 有效范围内
      - 相位恒等式 ``H_final = H_bed + uplift − ΔH_hyd − ΔH_thermal``
      - 路由基底不低于最终高程（无洼地保证）
      - 陆地单元均有出流（``direction != 0``）或位于纬度边界出流口
      - 汇流累积：陆地 >= 1、海洋 = 0
      - 河流只出现在陆地
      - 末态坡度不超过休止角（容差 ``slope_tol``）

    "海洋"由 ``flow_accumulation == 0`` 判定（管线保证陆地累积 >= 1、海洋为 0）。
    """
    problems: list[str] = []
    elevation = np.asarray(result.elevation, dtype=np.float64)
    for name, field in (
        ("elevation", elevation),
        ("uplift", np.asarray(result.uplift, dtype=np.float64)),
        ("hydraulic_drop", np.asarray(result.hydraulic_drop, dtype=np.float64)),
        ("thermal_drop", np.asarray(result.thermal_drop, dtype=np.float64)),
        ("filled_elevation", np.asarray(result.filled_elevation, dtype=np.float64)),
        ("river_width", np.asarray(result.river_width, dtype=np.float64)),
    ):
        if not np.all(np.isfinite(field)):
            problems.append(f"{name} 含非有限值 (NaN/Inf)")
    if problems:
        return problems

    accumulation = np.asarray(result.flow_accumulation, dtype=np.float64)
    ocean = accumulation <= 0.0
    land = ~ocean
    nlat = elevation.shape[0]

    lo, hi = FIELDS["elevation"].valid_range
    if float(elevation.min()) < lo or float(elevation.max()) > hi:
        problems.append(
            f"H_final 超出 elevation 有效范围 [{lo}, {hi}]: "
            f"[{float(elevation.min()):.1f}, {float(elevation.max()):.1f}]"
        )

    if check_identity:
        bed = np.asarray(result.erosion_bed, dtype=np.float64)
        uplift = np.asarray(result.uplift, dtype=np.float64)
        residual = (
            bed
            + uplift
            - np.asarray(result.hydraulic_drop)
            - np.asarray(result.thermal_drop)
        )
        scale = max(float(np.abs(bed).max()), 1.0)
        if float(np.abs(residual - elevation).max()) > 1e-6 * scale:
            problems.append("相位恒等式违反：H_final != H_bed + uplift − ΔH_hydraulic − ΔH_thermal")

    filled = np.asarray(result.filled_elevation, dtype=np.float64)
    if float((elevation - filled).max()) > 1e-6 * max(float(np.abs(elevation).max()), 1.0):
        problems.append("路由基底低于最终高程：洼地未填充，D8 会出现死点")

    direction = np.asarray(result.flow_direction)
    if land.any():
        rows = np.arange(nlat)[:, None]
        interior = (rows > 0) & (rows < nlat - 1)
        dead = land & interior & (direction == 0)
        if bool(dead.any()):
            problems.append(f"存在无出流的陆地单元 {int(dead.sum())} 个（洼地未填充）")
        if bool((accumulation[land] < 1.0).any()):
            problems.append("陆地单元汇流累积 < 1")
    if bool((accumulation[ocean] != 0.0).any()):
        problems.append("海洋单元汇流累积非零")

    river = np.asarray(result.river_network, dtype=bool)
    if bool((river & ocean).any()):
        problems.append("河流出现在海洋单元")

    if land.any():
        slope = te.slope_field(elevation, radius=radius)
        talus = float(np.tan(np.deg2rad(talus_angle_deg)))
        worst = float(slope[land].max())
        if worst > talus * (1.0 + slope_tol):
            problems.append(
                f"坡度超过休止角 {talus_angle_deg}°: |∇H|max={worst:.4f} > {talus * (1 + slope_tol):.4f}"
            )
    return problems


__all__ = [
    "DEFAULT_HILLSLOPE_DT_S",
    "DEFAULT_HYDRAULIC_STEPS",
    "DEFAULT_PRECIPITATION_M_PER_S",
    "DEFAULT_THERMAL_ITERATIONS",
    "ErosionResult",
    "simulate_erosion",
    "validate_erosion",
]
