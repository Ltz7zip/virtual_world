"""板块构造模拟管线（《第一层完善》§7 伪代码）。

流程：Fibonacci 种子点 → 域扭曲归属微板块 → 归并为大板块 → 分配欧拉极点 →
计算板块边界与法向 → 逐点相对速度 → 边界类型判定 → 地壳厚度时间积分
（Numba 加速，见 :mod:`virtual_world.terrain.isostasy`）→ Airy 均衡高程 →
俯冲带海沟/火山弧叠加。

输出 :class:`TectonicFieldResult`，供后续噪声精修与侵蚀模拟消费。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import isostasy
from .euler_poles import BoundaryType, pick_euler_poles
from .isostasy import integrate_crust_thickness
from .voronoi import assign_plates, fibonacci_sphere, merge_micro_plates

#: 汇聚边界增厚效率 η（§4.1，0.1–0.3）
THICKENING_EFFICIENCY = 0.2
#: 离散边界减薄效率 δ（§7 伪代码 10b）
THINNING_EFFICIENCY = 0.15
#: 重力松弛速率 κ（时间尺度约 25 Ma，§4.4）
RELAXATION_RATE = 0.04
#: 参考地壳厚度（对应海平面高程），km（§3.2）
REF_CRUST_KM = 30.0
#: 洋壳均衡厚度（§1.2.3：5–10 km），离散边界松弛目标
OCEANIC_CRUST_KM = 8.0
#: 地壳厚度下限（km），防止长时间离散减薄把洋壳减成负值
MIN_CRUST_KM = 5.0


@dataclasses.dataclass(frozen=True)
class TectonicFieldResult:
    """板块构造层输出场。"""

    elevation: np.ndarray  # (nlat, nlon) 构造高程 m
    plate_map: np.ndarray  # (nlat, nlon) int32 大板块 id ∈ [0, n_major)
    boundary_type: np.ndarray  # (nlat, nlon) int32 BoundaryType 像素值
    crust_thickness: np.ndarray  # (nlat, nlon) 最终地壳厚度 km
    seed: int
    n_major: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape


def trench_depth(age_ma: float, ref_age_ma: float = 20.0) -> float:
    """海沟深度 (m)（§5.1）：``D = -8000 - 2000*log10(age/10)``。"""
    return float(-8000.0 - 2000.0 * np.log10(age_ma / 10.0))


def volcanic_arc_uplift(distance_km: float) -> float:
    """火山弧抬升 (m)（§5.2）：弧带 +1000 ~ +3000 m 的简化钟形形态。"""
    d = min(max(distance_km, 100.0), 200.0)
    frac = (d - 100.0) / 100.0
    return 2500.0 - 700.0 * abs(frac - 0.5) * 2.0


def _surface_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """网格中心的单位球面向量，形状 ``(nlat, nlon, 3)``。

    ``lat_deg`` / ``lon_deg`` 为网格化后的 2D 坐标（广播到同形状）。
    """
    lat_2d, lon_2d = np.broadcast_arrays(
        np.asarray(lat_deg, dtype=np.float64), np.asarray(lon_deg, dtype=np.float64)
    )
    phi = np.deg2rad(lat_2d)
    lam = np.deg2rad(lon_2d)
    cos_phi = np.cos(phi)
    return np.stack([cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi)], axis=-1)


def _boundary_fields(
    plate: np.ndarray, vel: np.ndarray, xyz: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算边界诊断场：``(boundary_type, conv_rate, div_rate, arc_normal)``。

    对每个单元检查东/南两邻（经度循环）：板块不同即为边界。边界法向取
    ``P_cell - P_nbr`` 的球面切向分量并归一（指向本单元内部）；相对速度
    ``v_rel = (omega_a - omega_b) x P``。收敛/离散速率叠加到两侧单元。
    """
    nlat, nlon = plate.shape
    btype = np.full((nlat, nlon), BoundaryType.TRANSFORM, dtype=np.int32)
    conv_rate = np.zeros((nlat, nlon), dtype=np.float64)
    div_rate = np.zeros((nlat, nlon), dtype=np.float64)
    normals = np.zeros((nlat, nlon, 3), dtype=np.float64)

    # (di, dj) for south / east neighbor; di=1 → 行方向（纬度减一格即南邻？）；
    # 约定：邻居 = roll(plate, -1, axis)（东邻或南邻）
    for di, dj in ((1, 0), (0, 1)):
        nbr = np.roll(np.roll(plate, -di, axis=0), -dj, axis=1)
        bnd_with = plate != nbr
        idx_i, idx_j = np.nonzero(bnd_with)
        if idx_i.size == 0:
            continue
        nbr_i, nbr_j = (idx_i + di) % nlat, (idx_j + dj) % nlon
        pa, pb = plate[idx_i, idx_j], plate[nbr_i, nbr_j]

        p_cell = xyz[idx_i, idx_j]
        p_nbr = xyz[nbr_i, nbr_j]
        d = p_cell - p_nbr
        # 切向法向（在球面切平面上）
        tangent = d - np.einsum("ij,ij->i", d, p_cell)[:, None] * p_cell
        tlen = np.linalg.norm(tangent, axis=-1)
        ok = tlen > 1e-12
        t_n = np.zeros_like(tangent)
        t_n[ok] = tangent[ok] / tlen[ok, None]

        v = np.cross(vel[pa] - vel[pb], p_cell)
        dots = np.einsum("ij,ij->i", v, t_n)
        speeds = np.linalg.norm(v, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dots / np.where(speeds > 0.0, speeds, 1.0)
        is_conv = ratio < -threshold
        is_div = ratio > threshold

        np.add.at(normals, (idx_i, idx_j), t_n)
        np.add.at(conv_rate, (idx_i, idx_j), np.where(is_conv, speeds, 0.0))
        np.add.at(div_rate, (idx_i, idx_j), np.where(is_div, speeds, 0.0))
        # 邻居侧共享同一边界类型与速率
        np.add.at(normals, (nbr_i, nbr_j), -t_n)
        np.add.at(conv_rate, (nbr_i, nbr_j), np.where(is_conv, speeds, 0.0))
        np.add.at(div_rate, (nbr_i, nbr_j), np.where(is_div, speeds, 0.0))
        # 边界类型（内部与邻居侧均标记）
        btype[idx_i, idx_j] = np.where(
            is_conv,
            BoundaryType.CONVERGENT,
            np.where(is_div, BoundaryType.DIVERGENT, BoundaryType.TRANSFORM),
        )
        btype[nbr_i, nbr_j] = btype[idx_i, idx_j]

    normals = normals / np.clip(np.linalg.norm(normals, axis=-1)[..., None], 1e-12, None)
    return btype, conv_rate, div_rate, normals


def generate_tectonic_field(
    nlat: int,
    nlon: int,
    n_seeds: int = 300,
    n_major: int = 7,
    seed: int = 0,
    warp_amp: float = 0.15,
    time_ma: float = 200.0,
    dt_ma: float = 1.0,
    subduction: bool = True,
    threshold: float = 0.2,
) -> TectonicFieldResult:
    """运行完整板块构造管线（§7 伪代码 1–12）。

    地壳厚度单位 km，高程单位 m；时间步长默认 1 Ma。
    """
    if nlat <= 0 or nlon <= 0:
        raise ValueError("nlat/nlon 必须为正")
    if n_major < 2:
        raise ValueError(f"大板块数量必须 >= 2，实际为 {n_major}")
    if time_ma <= 0.0 or dt_ma <= 0.0:
        raise ValueError("time_ma 与 dt_ma 必须为正")

    step_lat = 180.0 / nlat
    step_lon = 360.0 / nlon
    lat = -90.0 + step_lat / 2 + step_lat * np.arange(nlat)
    lon = -180.0 + step_lon / 2 + step_lon * np.arange(nlon)
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    xyz = _surface_xyz(lat_2d, lon_2d)

    # 1–4：微板块 → 大板块（§7 伪代码 1–4）
    seeds = fibonacci_sphere(n_seeds)
    micro = assign_plates(lat_2d, lon_2d, seeds, seed=seed, warp_amp=warp_amp)
    plate_map = merge_micro_plates(micro, n_major=n_major, seed=seed + 12345)

    # 5：欧拉极点（§7 伪代码 5）
    omegas = pick_euler_poles(n_major, seed=seed + 777, rate_min=0.3, rate_max=1.5)

    # 6–8：边界类型与收敛/离散速率场（§7 伪代码 6–8）
    btype, conv_rate, div_rate, normals = _boundary_fields(plate_map, omegas, xyz, threshold)

    # 9–10：地壳厚度时间积分（§7 伪代码 9–10a/b/c）
    c_ref = REF_CRUST_KM
    # 均衡厚度按板块属性区分：离散边界（洋中脊）松弛到洋壳厚度，其余为大陆厚度
    c_eq = np.full((nlat, nlon), c_ref, dtype=np.float64)
    c_eq[div_rate > 0.0] = OCEANIC_CRUST_KM
    crust0 = np.full((nlat, nlon), c_ref, dtype=np.float64)
    crust = integrate_crust_thickness(
        crust0,
        thickening=THICKENING_EFFICIENCY * conv_rate * dt_ma,
        thinning=THINNING_EFFICIENCY * div_rate * dt_ma,
        c_eq=c_eq,
        kappa=RELAXATION_RATE,
        dt=1.0,
        n_steps=int(round(time_ma / dt_ma)),
    )

    # 11：Airy 均衡高程（§7 伪代码 11）
    crust = np.maximum(crust, MIN_CRUST_KM)
    elevation = np.asarray(isostasy.airy_elevation(crust, c_ref=c_ref), dtype=np.float64)

    # 12：俯冲带（海沟 + 火山弧，§7 伪代码 12）
    if subduction:
        conv_i, conv_j = np.nonzero(btype == BoundaryType.CONVERGENT)
        if conv_i.size:
            # 海沟只作用于洋壳俯冲侧（当前为海洋/低高程单元）；大陆碰撞侧
            # 保留地壳增厚形成的造山带
            trench_mask = elevation[conv_i, conv_j] < 0.0
            ages = np.full(conv_i.size, 20.0, dtype=np.float64)
            trenches = np.array([trench_depth(a) for a in ages])
            elevation[conv_i[trench_mask], conv_j[trench_mask]] += trenches[trench_mask]
            # 火山弧：沿边界法向方向外推一格，同板块单元抬升
            for ii, jj, nn in zip(conv_i, conv_j, normals[conv_i, conv_j], strict=True):
                di = int(np.rint(nn[0] * 2.0))
                dj = int(np.rint(nn[1] * 2.0))
                if di == 0 and dj == 0:
                    di, dj = (0, 1)
                ai, aj = (ii + di) % nlat, (jj + dj) % nlon
                if plate_map[ai, aj] == plate_map[ii, jj]:
                    elevation[ai, aj] += volcanic_arc_uplift(150.0)

    return TectonicFieldResult(
        elevation=elevation,
        plate_map=plate_map,
        boundary_type=btype,
        crust_thickness=crust,
        seed=seed,
        n_major=n_major,
    )
