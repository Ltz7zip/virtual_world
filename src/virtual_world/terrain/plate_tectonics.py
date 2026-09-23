"""板块构造模拟管线（《第一层完善》§1–§7 伪代码）。

流程：Fibonacci 种子点 → 域扭曲归属微板块 → 归并为大板块（含连通性修复）→
分配欧拉极点 → 板块边界配对与法向 → 相对速度与边界类型判定 → 地壳属性
（大陆/洋壳，由 ``ocean_fraction`` 控制）→ **§4.2 质量守恒给出增厚上限** →
地壳厚度时间积分（Numba 加速，见 :mod:`virtual_world.terrain.isostasy`）→
大陆用 Airy 均衡、洋壳用洋脊年龄热沉降 → **§2.5 陆-陆碰撞带角动量守恒（伪代码
10d）** → 俯冲带海沟/火山弧叠加 → 派生 ``is_ocean`` / ``ocean_depth`` /
``max_shortening_factor`` / ``collision_mask`` 等。

单位约定：地壳厚度 km，高程 m，时间 Ma。欧拉角速度在单位球上取无量纲量级
（与 §6.2 的向量化示例一致），``dt_ma`` 为时间步长；转物理角速度/汇聚距离的
标定见 :data:`MODEL_RATE_TO_RAD_PER_S` 与 :data:`CONVERGENCE_KM_PER_MA_PER_RATE`。

输出 :class:`TectonicFieldResult`，供后续噪声精修与侵蚀模拟消费。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..core.constants import (
    DAYS_PER_YEAR,
    EARTH_GRAVITY,
    EARTH_RADIUS,
    SECONDS_PER_DAY,
)
from ..core.cubed_sphere import CubedSphere
from . import isostasy, voronoi
from .euler_poles import (
    BoundaryType,
    cap_moment_of_inertia,
    collision_kinetic_budget,
    merge_angular_momentum,
    pick_euler_poles,
)
from .isostasy import integrate_crust_thickness
from .voronoi import assign_plates, fibonacci_sphere, merge_micro_plates, plate_angular_radius

# ===== 地壳增厚/减薄与松弛（§4.1、§4.4、§7）=====
#: 汇聚边界增厚效率 η（§4.1，0.1–0.3）
THICKENING_EFFICIENCY = 0.2
#: 离散边界减薄效率 δ（§7 伪代码 10b）
THINNING_EFFICIENCY = 0.15
#: 重力松弛速率 κ（§4.4，时间尺度约 25 Ma）
RELAXATION_RATE = 0.04

# ===== 地壳厚度参考值（§1.2.3、§3.2、§3.3）=====
#: 大陆参考地壳厚度（对应海平面高程），km
REF_CRUST_KM = 30.0
#: 大陆板块均衡地壳厚度（§1.2.3 的 30–70 km 区间内）
#: 比海平面参考厚度厚约 6 km，对应约 +900 m 的陆壳自由板——否则大陆内部平原会
#: 恰好落在 0 m，与下游 ``land_mask = elevation > 0`` 的判据冲突
CONTINENTAL_CRUST_KM = 36.0
#: 洋壳均衡厚度（§1.2.3：5–10 km）
OCEANIC_CRUST_KM = 8.0
#: 地壳厚度下限（km），防止长时间离散减薄把洋壳减成负值
MIN_CRUST_KM = 5.0

# ===== 海沟深度（§5.1）=====
#: 海沟深度范围（m）：年轻板片浅、老板片深
TRENCH_DEPTH_MIN = -11000.0
TRENCH_DEPTH_MAX = -4000.0
#: 经验关系 D = -8000 - 2000·log10(age/10)
TRENCH_DEPTH_BASE = -8000.0
TRENCH_DEPTH_SLOPE = -2000.0
TRENCH_REF_AGE_MA = 10.0

# ===== 洋壳热沉降与洋脊年龄（§1.2.3）=====
#: 洋中脊处海底深度 (m)
RIDGE_DEPTH_M = -2500.0
#: 热沉降平方根项系数 (m/Ma^0.5)，对应 depth ∝ √age
SUBSIDENCE_COEFF = 350.0
#: 平方根律适用的最大年龄 (Ma)，之后转为板模型（Parsons–Sclater）
PLATE_MODEL_START_MA = 70.0
#: 板模型渐近深度与特征时间
PLATE_MODEL_DEPTH_M = -6400.0
PLATE_MODEL_TAU_MA = 62.8
#: 洋壳年龄上限 (Ma)
OCEAN_AGE_CAP_MA = 200.0
#: 海底扩张速率（度/Ma）：约 3 cm/a ≈ 30 km/Ma ÷ 111 km/度
SPREADING_RATE_DEG_PER_MA = 0.27

#: 默认海洋面积占比（地球约 0.71，方案 §1.6 要求可配置）
DEFAULT_OCEAN_FRACTION = 0.7

# ===== §4.2 质量守恒标定 =====
#: 模型角速度单位 → 物理汇聚速率的标定 (km/Ma per 速率单位)。方案 §6.2 的角速度在
#: 单位球上取无量纲量级，需要显式标定才能换算汇聚距离；取 1 单位 ≡ 5 km/Ma
#: （≈5 cm/yr，真实板块汇聚速率的典型量级）。
CONVERGENCE_KM_PER_MA_PER_RATE = 5.0
#: 质量守恒允许的缩短因子上限，避免 ``beta = 1/(1 - d_L/L_0)`` 在板块被吃光时奇异。
#: 取值高于方案 §4.2 给出的安第斯基准 ``beta_0 = 2.34``。
MAX_SHORTENING_FACTOR_CAP = 4.0

# ===== §2.5 碰撞带角动量守恒 =====
#: 板块转动惯量用的地壳厚度 (m)——球冠薄壳近似取大陆地壳均衡厚度
INERTIA_THICKNESS_M = CONTINENTAL_CRUST_KM * 1000.0
#: 模型角速度单位 → 物理角速度 (rad/s)。由 :data:`CONVERGENCE_KM_PER_MA_PER_RATE` 推出的
#: 地表速度除以行星半径得到（``v = omega x R``），用于把碰撞动能预算折算成 SI 能量。
MODEL_RATE_TO_RAD_PER_S = (
    CONVERGENCE_KM_PER_MA_PER_RATE * 1000.0
    / (1.0e6 * DAYS_PER_YEAR * SECONDS_PER_DAY)
    / EARTH_RADIUS
)


# ===== 地壳厚度时间积分的后端选择（§6.4）=====


def _resolve_crust_integrator(use_jax: bool) -> Any:
    """选地壳厚度时间积分的实现：``use_jax`` 时返回 JAX 版，否则返回 Numba 版。

    两者签名与数值结果一致（同一分裂格式，见
    :func:`virtual_world.terrain.jax_kernels.integrate_crust_thickness_jax`）。JAX 是
    可选依赖，因此这里**惰性导入**：未安装 jax 时不在导入期报错，而是在调用
    JAX 版时抛出含安装指引的 :class:`RuntimeError`（不静默退回 Numba，否则调用方
    会以为拿到了 GPU 加速的结果）。
    """
    if not use_jax:
        return integrate_crust_thickness
    from .jax_kernels import integrate_crust_thickness_jax

    return integrate_crust_thickness_jax


@dataclasses.dataclass(frozen=True)
class TectonicFieldResult:
    """板块构造层输出场。"""

    elevation: np.ndarray  # (nlat, nlon) 构造高程 m
    plate_map: np.ndarray  # (nlat, nlon) int32 大板块 id ∈ [0, n_major)
    boundary_type: np.ndarray  # (nlat, nlon) int32 BoundaryType 像素值
    crust_thickness: np.ndarray  # (nlat, nlon) 最终地壳厚度 km
    is_ocean: np.ndarray  # (nlat, nlon) bool 海洋掩码（elevation < 0）
    ocean_depth: np.ndarray  # (nlat, nlon) m 海洋深度，陆地为 0
    ridge_mask: np.ndarray  # (nlat, nlon) bool 洋中脊（洋壳离散边界）
    plate_is_oceanic: np.ndarray  # (n_major,) bool 各板块的洋壳属性
    seed: int
    n_major: int
    # ===== §4.2 质量守恒 =====
    max_shortening_factor: np.ndarray  # (nlat, nlon) 质量守恒允许的缩短因子上限 beta_max
    plate_shortening_factor: np.ndarray  # (n_major,) 各板块达到的 beta = C/C0（洋壳为 1）
    plate_shortening_km: np.ndarray  # (n_major,) 由 beta 反推的地壳缩短量 d_L (km)
    # ===== §4.3 造山带高宽比 =====
    orogen_aspect_ratio: float  # 最强汇聚边界的 H/W（方案 §4.3 比例关系）
    orogen_half_width_km: float  # 由该高宽比与其造山带高度反推的半宽 W (km)
    # ===== §2.5 碰撞带角动量守恒 =====
    collision_mask: np.ndarray  # (nlat, nlon) bool 陆-陆碰撞带单元
    plate_omega_merged: np.ndarray  # (n_major, 3) 角动量守恒合并后的角速度（未参与碰撞者不变）
    collision_energy_j: float  # 碰撞耗散动能（势能预算）总和 (J)
    collision_uplift: np.ndarray  # (nlat, nlon) 由动能预算转成的抬升 (m)
    # ===== §2.2（第三层）构造抬升速率 U =====
    uplift_rate_m_per_ma: np.ndarray  # (nlat, nlon) 长期构造抬升速率 (m/Ma)，海洋为 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape


# ===== 经验关系（§1.2.3、§4.1、§5）=====


def trench_depth(age_ma: float | np.ndarray) -> float | np.ndarray:
    """海沟深度 (m)（§5.1）：``D = -8000 - 2000*log10(age/10)``，限定在 §5.1 范围内。"""
    age = np.maximum(np.asarray(age_ma, dtype=np.float64), 1e-6)
    depth = TRENCH_DEPTH_BASE + TRENCH_DEPTH_SLOPE * np.log10(age / TRENCH_REF_AGE_MA)
    result = np.clip(depth, TRENCH_DEPTH_MIN, TRENCH_DEPTH_MAX)
    return float(result) if np.asarray(age_ma).ndim == 0 else result


def ocean_floor_depth(age_ma: float | np.ndarray) -> float | np.ndarray:
    """洋壳海底深度 (m)（§1.2.3）：洋中脊约 -2500 m，随年龄按 √age 冷却下沉。

    年龄小于 :data:`PLATE_MODEL_START_MA` 时用平方根律；更老时用 Parsons–Sclater
    板模型 ``d = 6400 - 3200·exp(-t/62.8)``，使深海平原稳定在 -6400 m 量级。
    """
    age = np.maximum(np.asarray(age_ma, dtype=np.float64), 0.0)
    sqrt_law = RIDGE_DEPTH_M - SUBSIDENCE_COEFF * np.sqrt(age)
    plate_model = -(abs(PLATE_MODEL_DEPTH_M) - 3200.0 * np.exp(-age / PLATE_MODEL_TAU_MA))
    result = np.where(age <= PLATE_MODEL_START_MA, sqrt_law, plate_model)
    return float(result) if np.asarray(age_ma).ndim == 0 else result


def volcanic_arc_uplift(distance_km: float) -> float:
    """火山弧抬升 (m)（§5.2）：距海沟 100–200 km 的弧带抬升 +1000 ~ +3000 m。"""
    d = min(max(distance_km, 100.0), 200.0)
    frac = (d - 100.0) / 100.0
    return 2500.0 - 700.0 * abs(frac - 0.5) * 2.0


def convergence_factor(v_rel: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """碰撞角修正因子 ``f(θ) = |v_rel·n̂| / |v_rel|``（§4.1）。

    正面碰撞取 1（增厚效率最高），斜碰撞按 ``cos θ`` 衰减，纯走滑为 0
    （能量转为走滑运动而非地壳增厚）。
    """
    v = np.asarray(v_rel, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"v_rel 的最后一维必须为 3，实际 {v.shape}")
    speeds = np.linalg.norm(v, axis=-1)
    dots = np.einsum("...i,...i->...", v, n)
    safe = np.where(speeds > 0.0, speeds, 1.0)
    return np.where(speeds > 0.0, np.abs(dots) / safe, 0.0)


# ===== 边界配对与几何 =====


def boundary_index_pairs(
    plate_map: np.ndarray, neighbours: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """相邻但属不同板块的**扁平索引**配对 ``(index_a, index_b)``，每对仅出现一次。

    网格无关版：邻接由 ``neighbours`` 给出（缺省为经纬网格，见
    :func:`voronoi.latlon_neighbours`），因此立方球网格可直接传入其邻接表。
    去重规则为"扁平索引较小者为起点"，对无向网格边恰好保留一次。
    """
    plate = np.asarray(plate_map)
    nbr = voronoi.grid_neighbours(plate.shape, neighbours)
    flat = plate.ravel()
    source = np.arange(flat.size, dtype=np.int64)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for d in range(4):
        nb = nbr[d]
        valid = nb >= 0
        src = source[valid]
        dst = nb[valid]
        keep = (flat[src] != flat[dst]) & (src < dst)
        first.append(src[keep])
        second.append(dst[keep])
    return np.concatenate(first), np.concatenate(second)


def boundary_pairs(plate_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """经纬网格上的相邻异板块配对 ``(i, j, ni, nj)``，每对仅出现一次。

    经度方向循环、**纬度方向非循环**（极点约束，见 :mod:`virtual_world.core.spherical`）：
    南北极行之间不产生伪配对。网格无关版本见 :func:`boundary_index_pairs`。
    """
    plate = np.asarray(plate_map)
    if plate.ndim != 2:
        raise ValueError(f"boundary_pairs 仅支持 2D 经纬网格，实际 {plate.shape}")
    nlon = plate.shape[1]
    index_a, index_b = boundary_index_pairs(plate)
    return index_a // nlon, index_a % nlon, index_b // nlon, index_b % nlon


def _surface_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """网格中心的单位球面向量，形状 ``(nlat, nlon, 3)``。"""
    lat_2d, lon_2d = np.broadcast_arrays(
        np.asarray(lat_deg, dtype=np.float64), np.asarray(lon_deg, dtype=np.float64)
    )
    phi = np.deg2rad(lat_2d)
    lam = np.deg2rad(lon_2d)
    cos_phi = np.cos(phi)
    return np.stack([cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi)], axis=-1)


def _boundary_tangent(
    points: np.ndarray,
    index_a: np.ndarray,
    index_b: np.ndarray,
) -> np.ndarray:
    """边界法向：``P_cell - P_nbr`` 的球面切向分量并归一（指向本单元内部）。

    ``points`` 为扁平化的单元球面坐标 ``(N, 3)``，索引为扁平索引。
    """
    p_cell = points[index_a]
    d = p_cell - points[index_b]
    tangent = d - np.einsum("ij,ij->i", d, p_cell)[:, None] * p_cell
    length = np.linalg.norm(tangent, axis=-1)
    out = np.zeros_like(tangent)
    ok = length > 1e-12
    out[ok] = tangent[ok] / length[ok, None]
    return np.asarray(out)


# ===== 板块地壳属性与洋脊年龄 =====


def plate_crust_types(plate_map: np.ndarray, n_major: int, ocean_fraction: float) -> np.ndarray:
    """把大板块标为洋壳，使洋壳面积占比逼近 ``ocean_fraction``。

    大陆/洋壳是**板块本身**的属性（§1.2.3），因此海陆比可通过该参数配置
    （§1.6-1）。用 best-fit 贪心挑选板块（优先取不超过剩余需求的较大板块，
    否则取最小板块收尾），把误差压到"最小板块面积"量级。
    """
    if not 0.0 <= ocean_fraction <= 1.0:
        raise ValueError(f"ocean_fraction 必须在 [0, 1] 内，实际为 {ocean_fraction}")
    counts = np.bincount(np.asarray(plate_map).ravel(), minlength=n_major).astype(np.float64)
    total = float(counts.sum())
    flags = np.zeros(n_major, dtype=bool)
    if total <= 0.0:
        return flags

    need = ocean_fraction * total
    available = [int(p) for p in np.argsort(-counts, kind="stable")]
    accumulated = 0.0
    while available and accumulated < need:
        remaining = need - accumulated
        fits = [p for p in available if counts[p] <= remaining]
        chosen = (
            max(fits, key=lambda p: counts[p]) if fits else min(available, key=lambda p: counts[p])
        )
        flags[chosen] = True
        accumulated += counts[chosen]
        available.remove(chosen)
    return flags


def _ridge_age_field(
    oceanic_cell: np.ndarray,
    ridge_mask: np.ndarray,
    neighbours: np.ndarray,
    cell_angle_deg: float,
) -> np.ndarray:
    """洋壳年龄场 (Ma)：由到最近洋中脊的网格跳数与扩张速率换算（§1.2.3）。

    多源 BFS（邻接由 ``neighbours`` 给出，网格无关），只在洋壳内部传播；超出
    :data:`OCEAN_AGE_CAP_MA` 的单元取年龄上限（深海平原）。``cell_angle_deg``
    为该网格的平均单元角距（经纬网格取纬向格距，立方球取面均角距）。
    """
    nbr = np.asarray(neighbours)
    age_per_cell = cell_angle_deg / SPREADING_RATE_DEG_PER_MA
    max_layers = int(np.ceil(OCEAN_AGE_CAP_MA / age_per_cell)) + 1

    oceanic_flat = np.asarray(oceanic_cell, dtype=bool).ravel()
    distance = np.full(oceanic_flat.size, np.inf, dtype=np.float64)
    frontier = np.asarray(ridge_mask, dtype=bool).ravel() & oceanic_flat
    distance[frontier] = 0.0
    layer = 0
    while frontier.any() and layer < max_layers:
        layer += 1
        reached = np.zeros(oceanic_flat.size, dtype=bool)
        for d in range(4):
            nb = nbr[d]
            valid = nb >= 0
            sel = frontier & valid
            reached[nb[sel]] = True
        new = reached & oceanic_flat & np.isinf(distance)
        if not new.any():
            break
        distance[new] = layer
        frontier = new

    age = distance * age_per_cell
    age = np.where(np.isfinite(age), age, OCEAN_AGE_CAP_MA)
    return age.reshape(np.asarray(oceanic_cell).shape)


# ===== §4.2 质量守恒约束 =====


def plate_length_scale_km(
    plate_map: np.ndarray, n_major: int, radius: float = EARTH_RADIUS
) -> np.ndarray:
    """各板块的等效长度尺度 ``L0 = sqrt(板块面积)`` (km)，形状 ``(n_major,)``。

    这是 §4.2 质量守恒里的缩短基准长度 ``L_0``：把板块面积折算成线尺度。
    """
    counts = np.bincount(np.asarray(plate_map).ravel(), minlength=n_major).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros(n_major, dtype=np.float64)
    area = counts / total * 4.0 * np.pi * radius**2
    return np.asarray(np.sqrt(area) / 1000.0, dtype=np.float64)


def convergence_distance_km(
    convergence_rate: np.ndarray | float,
    time_ma: float,
    *,
    calibration: float = CONVERGENCE_KM_PER_MA_PER_RATE,
) -> np.ndarray:
    """模型汇聚速率 → 物理汇聚距离 ``d_L = rate * time * calibration`` (km)（§4.2）。

    方案 §6.2 的角速度在单位球上取无量纲量级，必须经 :data:`CONVERGENCE_KM_PER_MA_PER_RATE`
    标定才能与板块长度比较。
    """
    if time_ma <= 0.0:
        raise ValueError(f"time_ma 必须为正，实际为 {time_ma}")
    if calibration <= 0.0:
        raise ValueError(f"calibration 必须为正，实际为 {calibration}")
    rate = np.asarray(convergence_rate, dtype=np.float64)
    return np.asarray(rate * float(time_ma) * float(calibration), dtype=np.float64)


def mass_conservation_beta(
    convergence_rate: np.ndarray,
    plate_length_km: np.ndarray,
    time_ma: float,
    *,
    calibration: float = CONVERGENCE_KM_PER_MA_PER_RATE,
    beta_cap: float = MAX_SHORTENING_FACTOR_CAP,
) -> np.ndarray:
    """逐格质量守恒允许的最大缩短因子 ``beta_max``（§4.2）。

    ``beta_max = 1/(1 - d_L/L_0)``，其中 ``d_L`` 由该格的汇聚速率与演化时间反解
    （:func:`convergence_distance_km`），``L_0`` 为该板块的等效长度。``d_L/L_0``
    被截断到 ``1 - 1/beta_cap``，避免板块被"吃光"时的奇异性。
    """
    if beta_cap <= 1.0:
        raise ValueError(f"beta_cap 必须 > 1，实际为 {beta_cap}")
    rate = np.asarray(convergence_rate, dtype=np.float64)
    length = np.asarray(plate_length_km, dtype=np.float64)
    distance = convergence_distance_km(rate, time_ma, calibration=calibration)
    ratio = np.divide(distance, length, out=np.zeros_like(distance), where=length > 0.0)
    ratio = np.clip(ratio, 0.0, 1.0 - 1.0 / beta_cap)
    return np.asarray(1.0 / (1.0 - ratio), dtype=np.float64)


# ===== §2.5 碰撞带角动量守恒 =====


@dataclasses.dataclass(frozen=True)
class CollisionResult:
    """陆-陆碰撞带的角动量守恒结果（§2.5、§7 伪代码 10d）。"""

    mask: np.ndarray  # (nlat, nlon) bool 碰撞带单元
    omega_merged: np.ndarray  # (n_major, 3) 合并后角速度（未参与碰撞者保持原值）
    energy_j: float  # 碰撞耗散动能总和 (J)，即转化为抬升的"势能预算"
    uplift: np.ndarray  # (nlat, nlon) m，碰撞带抬升
    components: list[list[int]]  # 参与碰撞的板块连通分量


def _plate_components(first: np.ndarray, second: np.ndarray, n_plates: int) -> list[list[int]]:
    """把碰撞板块对并查集成连通分量（每个分量是一组互相碰撞的板块）。"""
    parent = list(range(n_plates))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for p in range(n_plates):
        groups.setdefault(find(p), []).append(p)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def collide_plates(
    plate_map: np.ndarray,
    plate_is_oceanic: np.ndarray,
    index_a: np.ndarray,
    index_b: np.ndarray,
    is_convergent: np.ndarray,
    omegas: np.ndarray,
    *,
    radius: float = EARTH_RADIUS,
    density: float = isostasy.RHO_CRUST,
    thickness: float = INERTIA_THICKNESS_M,
    g: float = EARTH_GRAVITY,
    rate_to_rad_per_s: float = MODEL_RATE_TO_RAD_PER_S,
) -> CollisionResult:
    """陆-陆碰撞带的角动量守恒处理（§2.5、§7 伪代码 10d）。

    方案要求碰撞**不能简单地让板块停止运动**，而应守恒角动量：互相碰撞的板块
    按转动惯量加权合并角速度（:func:`euler_poles.merge_angular_momentum`），
    损耗的动能 ``dE`` 转为碰撞带地壳增厚的"势能预算"。

    ``index_a`` / ``index_b`` 为边界配对的**扁平索引**（见 :func:`boundary_index_pairs`），
    ``is_convergent`` 为逐配对的布尔掩码。只有**两侧都是大陆地壳**的配对属于碰撞
    （洋-陆为俯冲，由 §5 的海沟/火山弧处理）。

    量纲：模型角速度是无量纲量级，动能预算必须用物理角速度
    ``omega_phys = omega_model * rate_to_rad_per_s``（见 :data:`MODEL_RATE_TO_RAD_PER_S`）
    才能得到 SI 能量；角动量合并只用比值，因此 ``omega_merged`` 仍以模型单位返回。

    抬升换算：``dH = dE/(rho*g*A)``（把能量均匀铺在碰撞带上）。
    注意：真实板块角速度（~1e-14 rad/s）下该项为**亚毫米量级**——板块运动的动能
    远小于地壳势能。本函数把它如实算出并保留在输出中，而不是人为放大。
    """
    plate = np.asarray(plate_map)
    oceanic = np.asarray(plate_is_oceanic, dtype=bool)
    conv = np.asarray(is_convergent, dtype=bool)
    flat_plate = plate.ravel()
    n_major = int(oceanic.size)

    sel = np.nonzero(conv)[0]
    if sel.size > 0:
        both_continental = ~oceanic[flat_plate[index_a[sel]]] & ~oceanic[flat_plate[index_b[sel]]]
        sel = sel[both_continental]

    mask_flat = np.zeros(flat_plate.size, dtype=bool)
    uplift_flat = np.zeros(flat_plate.size, dtype=np.float64)
    omega_merged = np.array(omegas, dtype=np.float64, copy=True)
    if sel.size == 0:
        return CollisionResult(
            mask_flat.reshape(plate.shape), omega_merged, 0.0, uplift_flat.reshape(plate.shape), []
        )

    components = _plate_components(
        flat_plate[index_a[sel]], flat_plate[index_b[sel]], n_major
    )

    counts = np.bincount(flat_plate, minlength=n_major).astype(np.float64)
    total_cells = float(counts.sum())
    cell_area = 4.0 * np.pi * radius**2 / max(total_cells, 1.0)
    energy_total = 0.0
    for members in components:
        idx = np.asarray(members, dtype=np.int64)
        # 板块面积占比 → 球冠角半径 → 薄壳转动惯量（仅比值参与合并，与 R 无关）
        fractions = counts[idx] / max(total_cells, 1.0)
        alphas = np.arccos(np.clip(1.0 - 2.0 * fractions, -1.0, 1.0))
        inertias = np.array(
            [cap_moment_of_inertia(float(a), radius, density, thickness) for a in alphas]
        )
        omega_merged[idx] = merge_angular_momentum(omegas[idx], inertias)
        energy = collision_kinetic_budget(omegas[idx] * float(rate_to_rad_per_s), inertias)
        energy_total += energy

        member_a = np.isin(flat_plate[index_a[sel]], idx)
        mask_flat[index_a[sel][member_a]] = True
        mask_flat[index_b[sel][member_a]] = True
        n_cells = int(mask_flat.sum())
        if n_cells > 0 and energy > 0.0:
            uplift_flat += np.where(mask_flat, energy / (density * g * n_cells * cell_area), 0.0)
    return CollisionResult(
        mask_flat.reshape(plate.shape),
        omega_merged,
        energy_total,
        uplift_flat.reshape(plate.shape),
        components,
    )


# ===== 主管线 =====


def generate_tectonic_field(
    nlat: int | None = None,
    nlon: int | None = None,
    n_seeds: int = 300,
    n_major: int = 7,
    seed: int = 0,
    warp_fraction: float = 0.15,
    time_ma: float = 200.0,
    dt_ma: float = 1.0,
    ocean_fraction: float = DEFAULT_OCEAN_FRACTION,
    subduction: bool = True,
    threshold: float = 0.2,
    *,
    grid: str = "latlon",
    n_side: int | None = None,
    radius: float = EARTH_RADIUS,
    use_jax: bool = False,
) -> TectonicFieldResult:
    """运行完整板块构造管线（§7 伪代码 1–12）。

    ``grid="latlon"``（默认）用 ``nlat``/``nlon`` 建经纬网格，输出 ``(nlat, nlon)``；
    ``grid="cubed_sphere"`` 用 ``n_side`` 建立方球网格（§1.5），输出 ``(6, n_side, n_side)``
    ——立方球**无极点奇点、单元面积接近均匀**，是方案推荐的地形生成网格；需要经纬布局时
    用 :func:`regrid_tectonic_result` 插值转换。

    ``warp_fraction`` 为域扭曲幅度相对板块平均角半径的比例（§1.2.1 步骤三：
    板块平均边长的 10%–20%）。``ocean_fraction`` 控制海洋面积占比（§1.6-1）。

    ``use_jax=True`` 时把第 9–10 步的地壳厚度时间积分交给
    :func:`jax_kernels.integrate_crust_thickness_jax`（方案 §6.4：``lax.scan`` 把
    数千步的时间循环编译成 XLA 计算图，长时积分在 GPU 上收益显著）。两者数值一致
    （同一分裂格式），缺 JAX 时会抛出含安装指引的错误，而不是静默退回 Numba。
    """
    if grid not in {"latlon", "cubed_sphere"}:
        raise ValueError(f"grid 必须为 'latlon' 或 'cubed_sphere'，实际为 {grid!r}")
    if n_major < 2:
        raise ValueError(f"大板块数量必须 >= 2，实际为 {n_major}")
    if time_ma <= 0.0 or dt_ma <= 0.0:
        raise ValueError("time_ma 与 dt_ma 必须为正")
    if not 0.0 <= ocean_fraction <= 1.0:
        raise ValueError(f"ocean_fraction 必须在 [0, 1] 内，实际为 {ocean_fraction}")
    if warp_fraction < 0.0:
        raise ValueError(f"warp_fraction 不能为负，实际为 {warp_fraction}")
    if radius <= 0.0:
        raise ValueError(f"radius 必须为正，实际为 {radius}")

    # 网格准备：坐标场、单元球面坐标、邻接表、单元平均角距（洋脊年龄换算用）
    if grid == "latlon":
        if nlat is None or nlon is None:
            raise ValueError("grid='latlon' 时必须给出 nlat 与 nlon")
        if nlat <= 0 or nlon <= 0:
            raise ValueError("nlat/nlon 必须为正")
        step_lat = 180.0 / nlat
        step_lon = 360.0 / nlon
        lat_1d = -90.0 + step_lat / 2 + step_lat * np.arange(nlat)
        lon_1d = -180.0 + step_lon / 2 + step_lon * np.arange(nlon)
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")
        shape: tuple[int, ...] = (nlat, nlon)
        xyz = _surface_xyz(lat_2d, lon_2d)
        neighbours = voronoi.latlon_neighbours(nlat, nlon)
        cell_angle_deg = step_lat
    else:
        if n_side is None:
            raise ValueError("grid='cubed_sphere' 时必须给出 n_side")
        sphere = CubedSphere(n_side, radius=radius)
        shape = sphere.shape
        lat_2d, lon_2d = sphere.centers_latlon()
        xyz = sphere.centers_xyz()
        neighbours = sphere.neighbors()
        # 一面覆盖 90 度、边长 n_side 格 ⇒ 单元平均角距约 90/n_side（面内近似）
        cell_angle_deg = 90.0 / n_side


    # 1–4：微板块 → 大板块（域扭曲幅度由板块角半径标定，§1.2.1/§1.3）
    seeds = fibonacci_sphere(n_seeds)
    warp_amp = warp_fraction * plate_angular_radius(n_seeds)
    micro = assign_plates(lat_2d, lon_2d, seeds, seed=seed, warp_amp=warp_amp)
    plate_map = merge_micro_plates(micro, n_major=n_major, seed=seed + 12345, neighbours=neighbours)

    # 5：欧拉极点
    omegas = pick_euler_poles(n_major, seed=seed + 777, rate_min=0.3, rate_max=1.5)

    # 6–8：边界配对 → 法向 → 相对速度 → 边界类型（含碰撞角因子 f(θ)，§4.1）
    # 为兼容立方球网格，内部统一用**扁平索引**运算，最后再 reshape 回网格形状。
    points = np.ascontiguousarray(xyz.reshape(-1, 3))
    flat_plate = plate_map.ravel()
    index_a, index_b = boundary_index_pairs(plate_map, neighbours)
    conforming = index_a.size > 0
    if conforming:
        tangent = _boundary_tangent(points, index_a, index_b)
        omega_rel = omegas[flat_plate[index_a]] - omegas[flat_plate[index_b]]
        v_rel = np.cross(omega_rel, points[index_a])
        speeds = np.linalg.norm(v_rel, axis=-1)
        factor = convergence_factor(v_rel, tangent)
        signed = np.einsum("ij,ij->i", v_rel, tangent) / np.where(speeds > 0.0, speeds, 1.0)
        is_conv = signed < -threshold
        is_div = signed > threshold
    else:
        speeds = np.zeros(0)
        factor = np.zeros(0)
        is_conv = np.zeros(0, dtype=bool)
        is_div = np.zeros(0, dtype=bool)

    size = int(np.prod(shape))
    btype_flat = np.full(size, BoundaryType.INTERIOR, dtype=np.int32)
    conv_rate_flat = np.zeros(size, dtype=np.float64)
    div_rate_flat = np.zeros(size, dtype=np.float64)

    def _pair_cells(mask: np.ndarray) -> np.ndarray:
        """该配对掩码涉及的全部单元（扁平索引，两侧各算一次）。"""
        return np.concatenate([index_a[mask], index_b[mask]])

    if conforming:
        # 增厚/减薄速率按 f(θ) 加权（斜碰撞只有法向分量参与增厚）
        weighted = speeds * factor
        for mask, rate in ((is_conv, conv_rate_flat), (is_div, div_rate_flat)):
            np.add.at(rate, _pair_cells(mask), np.concatenate([weighted[mask], weighted[mask]]))
        # 边界类型：汇聚优先，其次离散，最后转换；其余保持 INTERIOR
        for mask, code in ((is_conv, BoundaryType.CONVERGENT), (is_div, BoundaryType.DIVERGENT)):
            btype_flat[_pair_cells(mask)] = code
        rest = ~is_conv & ~is_div
        if rest.any():
            cells = _pair_cells(rest)
            unset = btype_flat[cells] == BoundaryType.INTERIOR
            btype_flat[cells[unset]] = BoundaryType.TRANSFORM

    btype = btype_flat.reshape(shape)
    conv_rate = conv_rate_flat.reshape(shape)
    div_rate = div_rate_flat.reshape(shape)

    # 板块地壳属性（§1.2.3）→ 均衡厚度目标
    plate_is_oceanic = plate_crust_types(plate_map, n_major, ocean_fraction)
    oceanic_cell = plate_is_oceanic[plate_map]
    c_eq = np.where(oceanic_cell, OCEANIC_CRUST_KM, CONTINENTAL_CRUST_KM).astype(np.float64)

    # §4.2：质量守恒给出逐格增厚上限——地壳增厚多少就必须缩短多少（L_0*C_0 = L*C）
    plate_length = plate_length_scale_km(plate_map, n_major)
    beta_max = mass_conservation_beta(conv_rate, plate_length[plate_map], time_ma)
    continental_cap = CONTINENTAL_CRUST_KM * beta_max

    # 9–10：地壳厚度时间积分（§6.4：缺省 Numba 内核；use_jax 时走 JAX lax.scan）
    integrator = _resolve_crust_integrator(use_jax)
    crust = integrator(
        c_eq,
        thickening=THICKENING_EFFICIENCY * conv_rate * dt_ma,
        thinning=THINNING_EFFICIENCY * div_rate * dt_ma,
        c_eq=c_eq,
        kappa=RELAXATION_RATE,
        dt=1.0,
        n_steps=int(round(time_ma / dt_ma)),
    )
    crust = np.maximum(crust, MIN_CRUST_KM)
    # §4.2 约束：大陆增厚受质量守恒上限截断（洋壳由洋脊热沉降决定，不受此约束）
    crust = np.where(oceanic_cell, crust, np.minimum(crust, continental_cap))

    # 11：高程——大陆用 Airy 均衡；洋壳用洋脊年龄热沉降（§1.2.3、§3.2）
    oceanic_flat = oceanic_cell.ravel()
    ridge_mask = (btype == BoundaryType.DIVERGENT) & oceanic_cell
    age = _ridge_age_field(oceanic_cell, ridge_mask, neighbours, cell_angle_deg)
    age_flat = age.ravel()
    elevation = np.asarray(
        isostasy.airy_elevation(crust, c_ref=REF_CRUST_KM), dtype=np.float64
    ).ravel()
    elevation[oceanic_flat] = np.asarray(ocean_floor_depth(age)).ravel()[oceanic_flat]

    # 10d：陆-陆碰撞带用角动量守恒合并角速度（而不是"让板块停止"），碰撞耗散动能
    # 转为抬升预算叠加到碰撞带（§2.5、§7 伪代码 10d）
    collision = collide_plates(plate_map, plate_is_oceanic, index_a, index_b, is_conv, omegas)
    collision_mask_flat = np.asarray(collision.mask, dtype=bool).ravel()
    if collision_mask_flat.any():
        elevation[collision_mask_flat] += np.asarray(collision.uplift).ravel()[collision_mask_flat]

    # 12：俯冲带——海沟位于俯冲（洋壳）侧，火山弧抬升位于上盘（陆壳）侧（§5）
    if subduction and conforming and is_conv.any():
        # 海沟：洋壳侧取海沟深度；两侧皆洋壳（洋内俯冲）时两侧都成沟
        for side in (index_a, index_b):
            mask = is_conv & oceanic_flat[side]
            if mask.any():
                cells = side[mask]
                np.minimum.at(elevation, cells, np.asarray(trench_depth(age_flat[cells])))

        # 火山弧：上盘为陆壳、对侧为俯冲洋壳时，在上盘侧抬升
        for up, down in ((index_a, index_b), (index_b, index_a)):
            mask = is_conv & ~oceanic_flat[up] & oceanic_flat[down]
            if mask.any():
                elevation[up[mask]] += volcanic_arc_uplift(150.0)

    # §4.3：造山带高宽比 H/W ∝ (d_rho*g*C_0)/(eta_eff*|v_rel|)，取最强汇聚边界的速率
    convergent_continental = ((btype == BoundaryType.CONVERGENT) & ~oceanic_cell).ravel()
    v_strongest = float(speeds[is_conv].max()) if (conforming and is_conv.any()) else 0.0
    if v_strongest > 0.0:
        aspect = isostasy.orogen_aspect_ratio(
            isostasy.RHO_MANTLE - isostasy.RHO_CRUST, CONTINENTAL_CRUST_KM, v_strongest
        )
        peak = (
            max(float(elevation[convergent_continental].max()), 0.0)
            if convergent_continental.any()
            else 0.0
        )
        half_width = isostasy.ridge_half_width_km(peak, aspect)
    else:
        aspect = 0.0
        half_width = 0.0

    # §4.2：各板块达到的缩短因子 beta = C/C_0 与对应缩短量（洋壳不参与碰撞增厚，恒为 1）
    max_crust_per_plate = np.full(n_major, -np.inf, dtype=np.float64)
    np.maximum.at(max_crust_per_plate, plate_map.ravel(), crust.ravel())
    beta_plate = np.where(
        ~plate_is_oceanic & (max_crust_per_plate > CONTINENTAL_CRUST_KM),
        max_crust_per_plate / CONTINENTAL_CRUST_KM,
        1.0,
    )
    beta_plate = np.maximum(beta_plate, 1.0)
    shortening_km = np.array(
        [
            isostasy.shortening_distance_km(float(b), float(max(length, 1.0)))
            for b, length in zip(beta_plate, plate_length, strict=True)
        ]
    )

    # §2.2（第三层）的构造抬升速率 U：地壳增厚相对初始均衡厚度的抬升除以演化时间。
    # 这是第一层交给第三层景观演化方程的 U（第三层方案 §2.2 的跨层因果链）。
    reference_elevation = float(isostasy.airy_elevation(CONTINENTAL_CRUST_KM, c_ref=REF_CRUST_KM))
    airy_crust = np.asarray(isostasy.airy_elevation(crust, c_ref=REF_CRUST_KM), dtype=np.float64)
    uplift_rate = np.where(
        oceanic_cell,
        0.0,
        np.maximum(airy_crust - reference_elevation, 0.0) / time_ma,
    )

    # 派生字段（§3.1 GridState 的 terrain 字段）
    elevation = elevation.reshape(shape)
    is_ocean = elevation < 0.0
    ocean_depth = np.where(is_ocean, -elevation, 0.0)

    return TectonicFieldResult(
        elevation=elevation,
        plate_map=plate_map,
        boundary_type=btype,
        crust_thickness=crust,
        is_ocean=is_ocean,
        ocean_depth=ocean_depth,
        ridge_mask=ridge_mask,
        plate_is_oceanic=plate_is_oceanic,
        seed=seed,
        n_major=n_major,
        max_shortening_factor=np.asarray(beta_max, dtype=np.float64),
        plate_shortening_factor=np.asarray(beta_plate, dtype=np.float64),
        plate_shortening_km=np.asarray(shortening_km, dtype=np.float64),
        orogen_aspect_ratio=float(aspect),
        orogen_half_width_km=float(half_width),
        collision_mask=np.asarray(collision.mask, dtype=bool),
        plate_omega_merged=np.asarray(collision.omega_merged, dtype=np.float64),
        collision_energy_j=float(collision.energy_j),
        collision_uplift=np.asarray(collision.uplift, dtype=np.float64),
        uplift_rate_m_per_ma=np.asarray(uplift_rate, dtype=np.float64),
    )


#: 规则网格上的连续场（重网格用双线性插值）
CONTINUOUS_TECTONIC_FIELDS: tuple[str, ...] = (
    "elevation",
    "crust_thickness",
    "ocean_depth",
    "max_shortening_factor",
    "collision_uplift",
    "uplift_rate_m_per_ma",
)
#: 编码/掩码场（重网格用最近邻，避免产生新类别）
DISCRETE_TECTONIC_FIELDS: tuple[str, ...] = (
    "plate_map",
    "boundary_type",
    "is_ocean",
    "ridge_mask",
    "collision_mask",
)


def regrid_tectonic_result(
    result: TectonicFieldResult,
    sphere: CubedSphere,
    nlat: int,
    nlon: int,
) -> TectonicFieldResult:
    """把**立方球网格**上的构造场插值到经纬网格（方案 §1.5 的互转桥接）。

    连续场（高程、地壳厚度等）用面内双线性；编码/掩码场（板块图、边界类型、海陆、
    洋中脊、碰撞带）用最近邻，避免插值造出不存在的类别。逐板块向量（``(n_major,)``）
    与标量（造山带高宽比等）与网格无关，原样保留。
    """
    if result.elevation.shape != sphere.shape:
        raise ValueError(
            f"结果网格 {result.elevation.shape} 与立方球 {sphere.shape} 不一致，无需重网格"
        )
    values: dict[str, Any] = {}
    cells = None
    for name in CONTINUOUS_TECTONIC_FIELDS:
        values[name] = sphere.to_latlon(np.asarray(getattr(result, name)), nlat, nlon)
    for name in DISCRETE_TECTONIC_FIELDS:
        field = np.asarray(getattr(result, name))
        if cells is None:
            cells = sphere.latlon_to_cell(nlat, nlon)
        if field.dtype == bool:
            values[name] = field.ravel()[cells]
        else:
            values[name] = field.ravel()[cells].astype(field.dtype)

    return dataclasses.replace(result, **values)

