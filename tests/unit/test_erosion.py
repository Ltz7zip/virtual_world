"""第三层 侵蚀模拟测试（《第三层完善：侵蚀模拟》§一–§七）。

覆盖四个模块：

- :mod:`virtual_world.terrain.hydrology` —— Priority-Flood 洼地填充、D8 流向
  （ESRI 编码）、拓扑汇流累积、汇水面积、河流网络与水力几何宽度（§四）
- :mod:`virtual_world.terrain.hydraulic_erosion` —— 虚拟管道模型、水量守恒、
  侵蚀/沉积、Shields 阈值、MacCormack 沉积物输运（§一、§5.3）
- :mod:`virtual_world.terrain.thermal_erosion` —— 休止角约束、物质守恒、
  线性坡面扩散（§三）
- :mod:`virtual_world.terrain.erosion` —— 完整第三层管线与输出一致性验证（§七）

测试取向：断言**物理不变量**（体积守恒、非负、无环、下坡单调）与**定性行为**，
不锁定绝对侵蚀量——侵蚀幅度随步数缩放，属于模型标定参数。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from virtual_world.core import constants as const
from virtual_world.core import spherical
from virtual_world.core.grid import FIELDS
from virtual_world.terrain import erosion as ero
from virtual_world.terrain import hydraulic_erosion as he
from virtual_world.terrain import hydrology as hd
from virtual_world.terrain import jax_kernels
from virtual_world.terrain import noise_refine as nr
from virtual_world.terrain import thermal_erosion as te

#: 地球年降水 1 m 折算的降水速率 (m/s)
P_EARTH = 1.0 / (const.DAYS_PER_YEAR * const.SECONDS_PER_DAY)
#: 细网格测试用的虚拟行星半径 (m)：使格距缩到百米量级，休止角才能被触发
SMALL_RADIUS = 1000.0


# ===== 测试夹具 =====


def _cell_area(nlat: int, nlon: int, radius: float = const.EARTH_RADIUS) -> np.ndarray:
    return spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)


def _cone_island(
    nlat: int = 24,
    nlon: int = 48,
    *,
    peak: float = 2000.0,
    half_width_rad: float = 0.6,
    sea_floor: float = -3000.0,
    roughness: float = 120.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """正圆锥岛 + 分形起伏：赤道、0 经度为中心，锥顶 2000 m，锥外深海。

    圆锥给出明确的"高处→海岸"排水方向，分形起伏制造汇流不均——真实地形
    的河网正是靠这种不均形成的（纯圆锥的径向流几乎不汇聚，刻不出河谷）。
    """
    lat = spherical.lat_centers(nlat)
    lon = spherical.lon_centers(nlon)
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    cp = np.cos(phi)
    xyz = np.stack(
        [cp * np.cos(lam), cp * np.sin(lam), np.sin(phi) * np.ones((nlat, nlon))], axis=-1
    )
    cos_d = np.cos(phi) * np.cos(lam)
    dist = np.arccos(np.clip(cos_d, -1.0, 1.0))
    land = dist < half_width_rad
    elevation = np.where(land, peak * (1.0 - dist / half_width_rad), sea_floor)
    if roughness > 0.0:
        rough = roughness * (
            nr.simplex_noise(xyz[..., 0] * 6.0, xyz[..., 1] * 6.0, xyz[..., 2] * 6.0, seed=seed)
            + 0.5
            * nr.simplex_noise(
                xyz[..., 0] * 15.0, xyz[..., 1] * 15.0, xyz[..., 2] * 15.0, seed=seed + 1
            )
        )
        elevation = elevation + np.where(land, rough, 0.0)
    is_ocean = elevation < 0.0
    return np.asarray(elevation, dtype=np.float64), is_ocean


def _valley_plane(nlat: int = 8, nlon: int = 12) -> np.ndarray:
    """两侧向中间排水的"V 形谷"：中央两列为海洋，是唯一出流口。

    经度是循环边界，把海洋放在**网格中部**（而非端列）才能排除"跨接缝伪下坡"：
    接缝两侧（j=0 与 j=nlon-1）等高，两端不会互相引流，于是汇流累积有精确期望。
    """
    j = np.arange(nlon)
    mid = nlon // 2
    elev = np.abs(j - (mid - 0.5)) * 100.0
    elev = np.tile(elev[None, :], (nlat, 1))
    elev[:, mid - 1 : mid + 1] = -1000.0
    return elev


def _continent(
    nlat: int = 32,
    nlon: int = 64,
    *,
    grade: float = 3.0e-4,
    offset: float = 500.0,
    roughness: float = 220.0,
    seed: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """南向倾斜的大陆：北高南低，低纬为海。

    排水路径横跨十几个纬行，配合分形起伏形成**长流程汇聚**的河网——这是
    检验汇流累积、河道提取与"汇流量越大下切越强"的标准地形。纯圆锥的
    径向流会迅速发散，几乎不汇聚，不适合做河网测试。
    """
    lat = spherical.lat_centers(nlat)
    lon = spherical.lon_centers(nlon)
    y = const.EARTH_RADIUS * np.deg2rad(lat)
    base = offset + grade * y
    elevation = np.tile(base[:, None], (1, nlon))
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    cp = np.cos(phi)
    xyz = np.stack(
        [cp * np.cos(lam), cp * np.sin(lam), np.sin(phi) * np.ones((nlat, nlon))], axis=-1
    )
    rough = roughness * (
        nr.fbm_noise(xyz[..., 0] * 5.0, xyz[..., 1] * 5.0, xyz[..., 2] * 5.0, seed=seed, octaves=3)
    )
    elevation = elevation + rough
    is_ocean = elevation < 0.0
    return np.asarray(elevation, dtype=np.float64), is_ocean


def _is_outlet(i: int, nlat: int) -> bool:
    """纬度边界行在本项目中是开口（纬向非循环），可视作出流口。"""
    return i == 0 or i == nlat - 1


def _downstream_ok(direction: np.ndarray, is_ocean: np.ndarray) -> tuple[int, int]:
    """沿 D8 流向追踪：返回 (未到出流口的陆地单元数, 路径长度超限单元数)。"""
    nlat, nlon = direction.shape
    limit = nlat * nlon + 4
    dead = 0
    looping = 0
    for i in range(nlat):
        for j in range(nlon):
            if is_ocean[i, j]:
                continue
            ci, cj = i, j
            for _ in range(limit):
                if is_ocean[ci, cj] or _is_outlet(ci, nlat):
                    break
                code = int(direction[ci, cj])
                if code == 0:
                    dead += 1
                    break
                di, dj = hd.D8_OFFSETS[code]
                ci, cj = ci + di, (cj + dj) % nlon
            else:
                looping += 1
    return dead, looping


# ===== 行为 1：Priority-Flood 洼地填充（§4.2）=====


def _basin_field() -> tuple[np.ndarray, np.ndarray]:
    """南侧为海的台地，台地中央有一处高于海平面但低于四周的封闭洼地。"""
    nlat, nlon = 24, 48
    elev = np.full((nlat, nlon), 500.0)
    elev[:2, :] = -800.0  # 南侧为海
    elev[nlat // 2, nlon // 2] = 100.0  # 内陆封闭洼地（仍为陆地）
    return elev, elev < 0.0


def test_fill_pits_raises_only_basins() -> None:
    elev, ocean = _basin_field()
    filled = hd.fill_pits(elev, ocean)
    assert filled.shape == elev.shape
    assert np.all(filled >= elev - 1e-12)
    # 洼地内部被显著抬升；未受影响的单元只被 ε 量级抬升（严格单调的最小代价）
    assert filled[12, 24] > elev[12, 24] + 100.0
    assert filled[20, 40] == pytest.approx(elev[20, 40], abs=1e-2)


def test_flow_directions_acyclic_on_quasi_flat_terrain() -> None:
    """回归：准平地（毫厘级高差）不得产生流环。

    早期实现用"容差内等高"做平地导流，会把单元指向**略高**的邻居而闭环；
    ε 抬升保证填充面严格单调，从结构上排除该问题。
    """
    nlat, nlon = 8, 16
    elev = np.full((nlat, nlon), 823.664674)
    elev[3:5, 6:10] = 823.664673  # 1e-6 级差，且与邻近单元几乎等高
    ocean = np.zeros_like(elev, dtype=bool)
    elev[:, -1] = -500.0  # 东侧为海，提供出流口
    ocean[:, -1] = True
    filled = hd.fill_pits(elev, ocean)
    direction = hd.flow_directions(filled, ocean)
    hd.flow_accumulation(direction, ocean)  # 有环会抛 ValueError
    dead, looping = _downstream_ok(direction, ocean)
    assert dead == 0 and looping == 0


def test_fill_pits_keeps_ocean_unchanged() -> None:
    elev, ocean = _basin_field()
    filled = hd.fill_pits(elev, ocean)
    assert np.array_equal(filled[ocean], elev[ocean])


def test_fill_pits_removes_all_pits() -> None:
    """填充后每个陆地单元都有严格下坡路径通向海洋或纬度边界。"""
    elev, ocean = _basin_field()
    filled = hd.fill_pits(elev, ocean)
    direction = hd.flow_directions(filled, ocean)
    dead, looping = _downstream_ok(direction, ocean)
    assert dead == 0
    assert looping == 0


def test_fill_pits_is_deterministic() -> None:
    elev, ocean = _basin_field()
    assert np.array_equal(hd.fill_pits(elev, ocean), hd.fill_pits(elev, ocean))


def test_fill_pits_validation() -> None:
    with pytest.raises(ValueError):
        hd.fill_pits(np.zeros((4, 8)), np.zeros((4, 7), dtype=bool))


# ===== 行为 3：D8 流向（§4.1）=====


def test_flow_direction_steepest_descent_codes() -> None:
    """V 形谷：西侧陆地一律向东（码 1），东侧一律向西（码 16），海洋为 0。"""
    nlat, nlon = 8, 12
    elev = _valley_plane(nlat, nlon)
    ocean = elev < 0.0
    direction = hd.flow_directions(hd.fill_pits(elev, ocean), ocean)
    mid = nlon // 2
    land = ~ocean
    cols = np.arange(nlon)[None, :] * np.ones((nlat, 1), dtype=int)
    assert np.all(direction[land & (cols < mid - 1)] == 1)
    assert np.all(direction[land & (cols > mid)] == 16)
    assert np.all(direction[ocean] == 0)


def test_flow_direction_diagonal_and_cyclic() -> None:
    """对角最陡下降取对角码；经度方向循环（j=0 的最陡邻居是 j=nlon-1）。"""
    nlat, nlon = 6, 8
    ocean = np.zeros((nlat, nlon), dtype=bool)
    # 沿 NE 方向单调下降 → (3,3) 指向 NE(128)
    ii, jj = np.indices((nlat, nlon))
    ne_field = 10000.0 - (ii + jj) * 100.0
    d_ne = hd.flow_directions(hd.fill_pits(ne_field, ocean), ocean)
    assert d_ne[3, 3] == 128  # NE
    # 仅西侧循环邻居是极低海面 → (3,0) 指向 W(16)
    west_field = np.full((nlat, nlon), 300.0)
    west_field[3, 0] = 400.0
    ocean2 = np.zeros((nlat, nlon), dtype=bool)
    ocean2[3, nlon - 1] = True
    west_field[3, nlon - 1] = -5000.0
    d_w = hd.flow_directions(hd.fill_pits(west_field, ocean2), ocean2)
    assert d_w[3, 0] == 16  # W


def test_flow_direction_codes_are_valid() -> None:
    elev, ocean = _cone_island()
    direction = hd.flow_directions(hd.fill_pits(elev, ocean), ocean)
    assert direction.dtype.kind == "i"
    assert set(np.unique(direction)) <= {0, *hd.D8_CODES}


def test_flow_direction_leaves_pits_as_sinks() -> None:
    """未填充的洼地单元无下坡邻居 → 出流码 0（说明必须先填充）。"""
    elev, ocean = _basin_field()
    direction = hd.flow_directions(elev, ocean)
    assert direction[12, 24] == 0


def test_flow_direction_ondemand_runoff_to_ocean() -> None:
    """圆锥岛：填充后所有陆地单元都能入海，无死点无环。"""
    elev, ocean = _cone_island()
    direction = hd.flow_directions(hd.fill_pits(elev, ocean), ocean)
    dead, looping = _downstream_ok(direction, ocean)
    assert dead == 0
    assert looping == 0


# ===== 行为 4：汇流累积（§4.3）=====


def test_flow_accumulation_valley_plane() -> None:
    """V 形谷：每列汇流累积精确等于"本列及上游列"的陆地格数。"""
    nlat, nlon = 8, 12
    elev = _valley_plane(nlat, nlon)
    ocean = elev < 0.0
    direction = hd.flow_directions(hd.fill_pits(elev, ocean), ocean)
    acc = hd.flow_accumulation(direction, ocean)
    assert np.allclose(acc[:, :5], np.arange(1, 6)[None, :])
    assert np.allclose(acc[:, 7:], np.arange(5, 0, -1)[None, :])
    assert np.all(acc[:, 5:7] == 0.0)


def test_flow_accumulation_monotone_downstream() -> None:
    elev, ocean = _cone_island()
    direction = hd.flow_directions(hd.fill_pits(elev, ocean), ocean)
    acc = hd.flow_accumulation(direction, ocean)
    nlat, nlon = direction.shape
    checked = 0
    for i in range(nlat):
        for j in range(nlon):
            code = int(direction[i, j])
            if ocean[i, j] or code == 0:
                continue
            di, dj = hd.D8_OFFSETS[code]
            ni, nj = i + di, (j + dj) % nlon
            if ocean[ni, nj]:
                continue
            assert acc[ni, nj] >= acc[i, j]
            checked += 1
    assert checked > 0


def test_flow_accumulation_matches_recursion() -> None:
    """逐格核对定义式 ``A[i] = 1 + Σ_{上游} A[u]``（§4.3），包含入海终止。"""
    elev, ocean = _continent()
    direction = hd.flow_directions(hd.fill_pits(elev, ocean), ocean)
    acc = hd.flow_accumulation(direction, ocean)
    nlat, nlon = direction.shape
    offsets = list(hd.D8_OFFSETS.values())
    for i in range(nlat):
        for j in range(nlon):
            if ocean[i, j]:
                assert acc[i, j] == 0.0
                continue
            expected = 1.0
            for di, dj in offsets:
                ni, nj = i - di, (j - dj) % nlon
                if ni < 0 or ni >= nlat or ocean[ni, nj]:
                    continue
                code = int(direction[ni, nj])
                if code == 0:
                    continue
                rdi, rdj = hd.D8_OFFSETS[code]
                if (ni + rdi, (nj + rdj) % nlon) == (i, j):
                    expected += acc[ni, nj]
            assert acc[i, j] == pytest.approx(expected), f"({i},{j}) 的汇流累积不符定义式"


def test_flow_accumulation_detects_cycle() -> None:
    """人为构造两格互指 → 环绕无出流，必须报错而不是静默给出错值。"""
    ocean = np.zeros((3, 4), dtype=bool)
    direction = np.zeros((3, 4), dtype=np.int16)
    direction[1, 1] = int(hd.D8_CODES[0])  # E
    direction[1, 2] = int(hd.D8_CODES[4])  # W
    with pytest.raises(ValueError, match="环"):
        hd.flow_accumulation(direction, ocean)


# ===== 行为 5：河流网络与水力几何（§4.4）=====


def test_drainage_area_scales_with_cell_area() -> None:
    acc = np.full((4, 8), 10.0)
    area = _cell_area(4, 8)
    assert np.allclose(hd.drainage_area(acc, area), 10.0 * area)


def test_extract_river_network_threshold_and_land_only() -> None:
    elev, ocean = _continent()
    filled = hd.fill_pits(elev, ocean)
    direction = hd.flow_directions(filled, ocean)
    acc = hd.flow_accumulation(direction, ocean)
    river = hd.extract_river_network(acc, ocean, min_cells=20)
    assert river.dtype == bool
    assert river.any()
    assert not np.any(river & ocean)
    # 阈值提高后河网收缩（严格子集）
    sparse = hd.extract_river_network(acc, ocean, min_cells=60)
    assert sparse.any()
    assert sparse.sum() < river.sum()
    assert not np.any(sparse & ~river)


def test_river_width_hydraulic_geometry() -> None:
    """w ∝ √A：宽度随汇水面积单调增加，非河道为 0，且有上下限。"""
    area = np.array([[0.0, 4.0e6, 4.0e8]])
    river = np.zeros_like(area, dtype=bool)
    river[0, 1:] = True
    width = hd.river_width(area, river)
    assert width[0, 0] == 0.0
    assert width[0, 1] < width[0, 2]
    assert width[0, 1] >= hd.RIVER_WIDTH_MIN_M
    assert width[0, 2] <= hd.RIVER_WIDTH_MAX_M


def test_analyze_hydrology_consistency() -> None:
    elev, ocean = _continent()
    result = hd.analyze_hydrology(elev, ocean, river_min_cells=20)
    assert result.filled_elevation.shape == elev.shape
    assert np.all(result.filled_elevation >= elev - 1e-12)
    assert np.all(result.flow_accumulation[ocean] == 0.0)
    assert np.all(result.flow_accumulation[~ocean] >= 1.0)
    assert not np.any(result.river_network & ocean)
    assert result.river_network.any()
    assert np.all(result.river_width[~result.river_network] == 0.0)
    assert result.report["river_cells"] == int(result.river_network.sum())


# ===== 行为 6：水力侵蚀（§一）=====


def test_hydraulic_zero_precipitation_no_change() -> None:
    """无降水 → 无水 → 不侵蚀，高程逐点不变。"""
    elev, ocean = _cone_island()
    result = he.hydraulic_erode(elev, ocean, precipitation=0.0, n_steps=20)
    assert np.array_equal(result.elevation, elev)
    assert np.allclose(result.drop, 0.0)


def test_hydraulic_flat_terrain_no_erosion() -> None:
    """平地无坡度 → 容量为 0 → 不侵蚀（只积水）。"""
    elev = np.full((12, 24), 500.0)
    ocean = np.zeros_like(elev, dtype=bool)
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=30)
    assert np.allclose(result.drop, 0.0)
    assert result.water_depth.max() > 0.0


def test_hydraulic_carves_localized_incision() -> None:
    """下切高度局部化（河谷），而非均匀削平：极值远大于标准差。"""
    elev, ocean = _continent()
    land = ~ocean
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=120)
    drop = result.drop[land]
    assert drop.max() > 0.0
    assert drop.max() > 4.0 * drop.std()
    assert drop.std() > 0.0
    assert np.all(np.isfinite(result.elevation))


def test_hydraulic_erosion_increases_downstream() -> None:
    """南向大陆：汇流沿流向累积 → 下游（南侧）平均下切大于上游（北侧）。"""
    elev, ocean = _continent()
    nlat = elev.shape[0]
    land = ~ocean
    rows = np.arange(nlat)[:, None]
    south = land & (rows < nlat * 0.55)
    north = land & (rows >= nlat * 0.55)
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=120)
    assert result.drop[south].mean() > result.drop[north].mean()


def test_hydraulic_ocean_untouched() -> None:
    elev, ocean = _cone_island()
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=60)
    assert np.array_equal(result.elevation[ocean], elev[ocean])
    assert np.all(result.sediment[ocean] == 0.0)


def test_hydraulic_water_is_bounded_and_nonnegative() -> None:
    """强降水下水量必须非负且不失控（出流通量钳制的效果）。"""
    elev, ocean = _cone_island()
    result = he.hydraulic_erode(elev, ocean, precipitation=100.0 * P_EARTH, n_steps=150)
    assert np.all(result.water_depth >= 0.0)
    assert np.all(np.isfinite(result.water_depth))
    assert result.water_depth[~ocean].max() < 1.0e4


def test_hydraulic_sediment_nonnegative_and_finite() -> None:
    elev, ocean = _cone_island()
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=80)
    assert np.all(result.sediment >= 0.0)
    assert np.all(np.isfinite(result.sediment))


def test_hydraulic_bounded_step_change() -> None:
    """地形改动受"不得低于最低邻居"约束 → 总降低量不超过局部起伏量级。"""
    elev, ocean = _cone_island()
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=150)
    assert float(result.report["max_abs_elevation_change_m"]) <= 1.0e4


def test_hydraulic_deterministic() -> None:
    elev, ocean = _cone_island()
    a = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=40)
    b = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=40)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.sediment, b.sediment)


def test_hydraulic_report_fields() -> None:
    elev, ocean = _cone_island()
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=20)
    assert result.report["n_steps"] == 20
    assert result.report["dt_s"] > 0.0
    assert np.isfinite(result.report["dt_s"])
    assert result.report["total_incision_m3"] >= 0.0


def test_hydraulic_precipitation_validation() -> None:
    elev, ocean = _cone_island(8, 16)
    with pytest.raises(ValueError):
        he.hydraulic_erode(elev, ocean, precipitation=-1.0, n_steps=5)
    with pytest.raises(ValueError):
        he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=0)
    with pytest.raises(ValueError):
        he.hydraulic_erode(elev, ocean, precipitation=np.zeros((4, 8)), n_steps=5)


def test_hydraulic_accepts_precipitation_field() -> None:
    """可传入逐格降水场（后续气候模块的接入点）。"""
    elev, ocean = _cone_island()
    precip = np.where(ocean, 0.0, P_EARTH)
    result = he.hydraulic_erode(elev, ocean, precipitation=precip, n_steps=20)
    assert result.elevation.shape == elev.shape


# ===== 行为 7：沉积物输运（§5.3 MacCormack）=====


def test_semilagrangian_preserves_constant_field() -> None:
    """零流速 → 半拉格朗日回溯原地采样，沉积物场严格不变。"""
    field = np.linspace(0.0, 1.0, 12)[:, None] * np.ones((1, 24))
    vx = np.zeros_like(field)
    vy = np.zeros_like(field)
    dlon_m = np.full(12, 1.0e5)
    out = he.advect_sediment(field, vx, vy, dt=10.0, dlat_m=1.0e5, dlon_m=dlon_m)
    assert np.allclose(out, field)


def test_semilagrangian_uniform_flow_matches_analytic_shift() -> None:
    """匀速东向流：均匀场平移后仍为同一常数（输运不产生源汇）。"""
    field = np.full((6, 12), 3.0)
    vx = np.full_like(field, 1.0)  # 向东 1 m/s
    vy = np.zeros_like(field)
    out = he.advect_sediment(field, vx, vy, dt=0.5, dlat_m=1.0e5, dlon_m=np.full(6, 1.0e5))
    assert np.allclose(out, 3.0)


def test_macormack_keeps_front_steeper_than_semilagrangian() -> None:
    """MacCormack 反扩散应比纯半拉格朗日给出更陡的沉积物锋面（§5.3）。"""
    field = np.zeros((8, 40))
    field[:, 12:20] = 1.0
    vx = np.full_like(field, 4000.0)
    vy = np.zeros_like(field)
    dlon_m = np.full(8, 1.0e5)
    plain = he.advect_sediment(field, vx, vy, dt=5.0, dlat_m=1.0e5, dlon_m=dlon_m, macormack=False)
    sharp = he.advect_sediment(field, vx, vy, dt=5.0, dlat_m=1.0e5, dlon_m=dlon_m, macormack=True)
    plain_grad = float(np.abs(np.diff(plain, axis=1)).max())
    sharp_grad = float(np.abs(np.diff(sharp, axis=1)).max())
    assert sharp_grad > plain_grad
    assert float(sharp.max()) == pytest.approx(1.0, abs=1e-9)  # 峰值不被削平


# ===== 行为 8：热力侵蚀（§三）=====


def test_slope_field_matches_analytic_gradient() -> None:
    """线性斜坡 H = a·i → |∇H| = a / dlat_m。"""
    radius = const.EARTH_RADIUS
    nlat, nlon = 12, 24
    a = 500.0
    elev = np.repeat((a * np.arange(nlat))[:, None], nlon, axis=1).astype(np.float64)
    dlat_m, _ = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    slope = te.slope_field(elev, radius=radius)
    assert np.allclose(slope, a / dlat_m, rtol=1e-9)


def test_thermal_erosion_enforces_talus_angle() -> None:
    """超过休止角的陡坎必须被削平到休止角以内（细网格 + 小半径）。"""
    nlat, nlon = 16, 32
    elev = np.zeros((nlat, nlon))
    elev[nlat // 2 :, :] = 1000.0
    ocean = np.zeros_like(elev, dtype=bool)
    result = te.thermal_erode(
        elev, ocean, talus_angle_deg=33.0, iterations=400, radius=SMALL_RADIUS
    )
    talus = np.tan(np.deg2rad(33.0))
    slope = te.slope_field(result.elevation, radius=SMALL_RADIUS)
    assert float(slope[~ocean].max()) <= talus * 1.05
    assert float(slope[~ocean].max()) < 1.0e6


def test_thermal_erosion_conserves_mass() -> None:
    """休止角滑移是纯物质再分配 → 面积加权体积严格守恒。"""
    nlat, nlon = 16, 32
    elev = np.zeros((nlat, nlon))
    elev[nlat // 2 :, :] = 1000.0
    ocean = np.zeros_like(elev, dtype=bool)
    area = _cell_area(nlat, nlon, SMALL_RADIUS)
    before = float((elev * area).sum())
    after = float(
        (te.thermal_erode(elev, ocean, iterations=50, radius=SMALL_RADIUS).elevation * area).sum()
    )
    assert after == pytest.approx(before, rel=1e-9)


def test_thermal_erosion_flat_terrain_unchanged() -> None:
    elev = np.full((12, 24), 300.0)
    ocean = np.zeros_like(elev, dtype=bool)
    result = te.thermal_erode(elev, ocean, iterations=10, radius=SMALL_RADIUS)
    assert np.array_equal(result.elevation, elev)


def test_thermal_erosion_below_talus_untouched() -> None:
    """地球尺度粗网格：坡度远低于休止角 → 热力侵蚀为恒等变换（§3.1 物理正确）。"""
    elev, ocean = _cone_island()
    result = te.thermal_erode(elev, ocean, iterations=20, radius=const.EARTH_RADIUS)
    assert np.allclose(result.elevation, elev)
    assert np.allclose(result.drop, 0.0)


def test_thermal_erosion_ocean_untouched() -> None:
    nlat, nlon = 16, 32
    elev = np.zeros((nlat, nlon))
    elev[nlat // 2 :, :] = 1000.0
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    elev[:2, :] = -500.0
    result = te.thermal_erode(elev, ocean, iterations=50, radius=SMALL_RADIUS)
    assert np.array_equal(result.elevation[ocean], elev[ocean])


def test_thermal_erosion_deterministic() -> None:
    nlat, nlon = 16, 32
    elev = np.zeros((nlat, nlon))
    elev[nlat // 2 :, :] = 1000.0
    ocean = np.zeros_like(elev, dtype=bool)
    a = te.thermal_erode(elev, ocean, iterations=30, radius=SMALL_RADIUS)
    b = te.thermal_erode(elev, ocean, iterations=30, radius=SMALL_RADIUS)
    assert np.array_equal(a.elevation, b.elevation)


def test_thermal_erosion_zero_talus_moves_everything() -> None:
    """休止角为 0 → 任何正坡度都搬运物质。"""
    elev = np.zeros((8, 16))
    elev[4:, :] = 100.0
    ocean = np.zeros_like(elev, dtype=bool)
    result = te.thermal_erode(elev, ocean, talus_angle_deg=0.0, iterations=20, radius=SMALL_RADIUS)
    assert not np.allclose(result.elevation, elev)


def test_hillslope_diffusion_analytic_decay() -> None:
    """线性扩散对正弦场的单步衰减与解析解一致（§3.2）。"""
    radius = const.EARTH_RADIUS
    nlat, nlon = 8, 32
    amp = 200.0
    kappa, dt = 1.0e-4, 1.0e6
    j = np.arange(nlon)
    elev = np.repeat((amp * np.sin(2.0 * np.pi * j / nlon))[None, :], nlat, axis=0)
    out = te.hillslope_diffusion(elev, kappa, dt, radius=radius)
    _, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    decay = 4.0 * np.sin(np.pi / nlon) ** 2 / dlon_m**2
    expected = elev * (1.0 - kappa * dt * decay)[:, None]
    assert np.allclose(out, expected, rtol=1e-9)


def test_hillslope_diffusion_preserves_area_weighted_mean() -> None:
    nlat, nlon, radius = 12, 24, const.EARTH_RADIUS
    rng = np.random.default_rng(7)
    elev = rng.normal(size=(nlat, nlon)) * 500.0
    area = _cell_area(nlat, nlon, radius)
    out = te.hillslope_diffusion(elev, kappa=1.0e-4, dt=1.0e7, n_steps=5, radius=radius)
    before = float((elev * area).sum())
    after = float((out * area).sum())
    assert after == pytest.approx(before, rel=1e-9)
    assert float(out.std()) < float(elev.std())


def test_hillslope_diffusion_constant_field_unchanged() -> None:
    elev = np.full((8, 16), 42.0)
    out = te.hillslope_diffusion(elev, kappa=1.0e-3, dt=1.0e9, radius=const.EARTH_RADIUS)
    assert np.allclose(out, 42.0)


def test_hillslope_diffusion_validation() -> None:
    elev = np.zeros((4, 8))
    with pytest.raises(ValueError):
        te.hillslope_diffusion(elev, kappa=-1.0, dt=1.0)
    with pytest.raises(ValueError):
        te.hillslope_diffusion(elev, kappa=1.0, dt=0.0)
    with pytest.raises(ValueError):
        te.hillslope_diffusion(elev, kappa=1.0, dt=1.0, n_steps=0)


# ===== 行为 9：第三层完整管线（§七）=====


def test_simulate_erosion_shapes_and_finiteness() -> None:
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=60)
    assert result.elevation.shape == elev.shape
    assert result.hydraulic_drop.shape == elev.shape
    assert result.thermal_drop.shape == elev.shape
    assert result.river_network.dtype == bool
    assert np.all(np.isfinite(result.elevation))
    assert np.all(np.isfinite(result.hydraulic_drop))


def test_simulate_erosion_phase_identity() -> None:
    """H_final = H_bed − ΔH_hydraulic − ΔH_thermal，且 H_bed 是输入洼地填充结果（§七 1–4）。"""
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=40)
    assert np.allclose(result.erosion_bed, hd.fill_pits(elev, ocean))
    residual = result.erosion_bed - result.hydraulic_drop - result.thermal_drop - result.elevation
    assert np.allclose(residual, 0.0, atol=1e-6)


def test_simulate_erosion_reduces_land_elevation() -> None:
    """管线在陆地上产生真实的局部下切（极值远大于标准差），并改变地形。"""
    elev, ocean = _continent()
    land = ~ocean
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=120)
    drop = result.total_drop[land]
    assert drop.max() > 0.0
    assert drop.max() > 4.0 * drop.std()
    assert not np.allclose(result.elevation, elev)


def test_simulate_erosion_river_network_reaches_ocean() -> None:
    elev, ocean = _continent()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=60)
    assert result.river_network.any()
    assert not np.any(result.river_network & ocean)
    dead, looping = _downstream_ok(result.flow_direction, ocean)
    assert dead == 0
    assert looping == 0


def test_simulate_erosion_routing_bed_has_no_pits() -> None:
    elev, ocean = _continent()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=40)
    assert np.all(result.filled_elevation >= result.elevation - 1e-12)


def test_simulate_erosion_ocean_floor_untouched() -> None:
    elev, ocean = _continent()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=40)
    assert np.array_equal(result.elevation[ocean], elev[ocean])


def test_simulate_erosion_elevation_within_gridstate_range() -> None:
    elev, ocean = _continent()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=80)
    lo, hi = FIELDS["elevation"].valid_range
    assert float(result.elevation.min()) >= lo
    assert float(result.elevation.max()) <= hi


def test_simulate_erosion_deterministic() -> None:
    elev, ocean = _continent()
    a = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, seed=3, hydraulic_steps=30)
    b = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, seed=3, hydraulic_steps=30)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.river_network, b.river_network)


def test_simulate_erosion_hillslope_diffusion_branch() -> None:
    """开启坡面扩散（κ>0）时热力项不再为零，且仍守恒面积加权均值。"""
    nlat, nlon = 16, 32
    rng = np.random.default_rng(11)
    elev = 500.0 + rng.normal(size=(nlat, nlon)) * 200.0
    ocean = np.zeros_like(elev, dtype=bool)
    result = ero.simulate_erosion(
        elev,
        ocean,
        precipitation=0.0,
        hydraulic_steps=1,
        hillslope_kappa=1.0e-4,
        hillslope_dt=1.0e7,
        hillslope_steps=3,
    )
    assert float(np.abs(result.thermal_drop).max()) > 0.0
    assert float(result.elevation.std()) < float(elev.std())


def test_simulate_erosion_no_precipitation_no_hydraulic_drop() -> None:
    """无降水 ⇒ 无水力下切；坡面扩散关闭时整条管线为恒等变换。"""
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(
        elev, ocean, precipitation=0.0, hydraulic_steps=20, hillslope_kappa=0.0
    )
    assert np.allclose(result.hydraulic_drop, 0.0)
    assert np.allclose(result.elevation, result.erosion_bed)


# ===== §1.4 Shields 准则（侵蚀阈值） =====


def test_shields_critical_shear_matches_formula() -> None:
    """tau_ct = theta*d_s*(gamma_s - gamma_w)（§1.4）。"""
    grain = 2.0e-3
    tau = he.shields_critical_shear(grain)
    expected = he.SHIELDS_PARAMETER * grain * (he.DENSITY_SEDIMENT - const.DENSITY_WATER) * const.EARTH_GRAVITY
    assert tau == pytest.approx(expected, rel=1e-12)


def test_critical_velocity_scales_with_grain_size() -> None:
    """u_c = sqrt(tau_ct/(rho_w*C_f)) ∝ sqrt(d_s)：粒径 4 倍 ⇒ 临界流速 2 倍。"""
    u_small = he.critical_velocity_from_shields(1.0e-3)
    u_large = he.critical_velocity_from_shields(4.0e-3)
    assert u_large == pytest.approx(2.0 * u_small, rel=1e-12)
    # 沙(1mm) / 砾石(40mm) 的临界流速落在水力学合理区间
    assert 0.1 < u_small < 1.0
    assert 0.5 < he.critical_velocity_from_shields(4.0e-2) < 5.0


def test_shields_validation() -> None:
    with pytest.raises(ValueError):
        he.shields_critical_shear(0.0)
    with pytest.raises(ValueError):
        he.shields_critical_shear(1e-3, shields_parameter=0.0)
    with pytest.raises(ValueError):
        he.shields_critical_shear(1e-3, density_sediment=900.0)
    with pytest.raises(ValueError):
        he.critical_velocity_from_shields(friction_coefficient=0.0)


def test_pipeline_shields_threshold_reduces_incision() -> None:
    """默认启用 Shields 阈值：相对无阈值(u_c=0)应有更少的侵蚀（§1.4）。"""
    elev, ocean = _continent()
    kwargs = dict(precipitation=P_EARTH, hydraulic_steps=60)
    off = ero.simulate_erosion(elev, ocean, critical_velocity=0.0, **kwargs)
    on = ero.simulate_erosion(elev, ocean, **kwargs)  # 缺省 = Shields(1 mm 中砂)
    assert on.report["critical_velocity_m_per_s"] > 0.0
    assert off.report["critical_velocity_m_per_s"] == 0.0
    assert float(on.report["hydraulic_incision_m3"]) < float(off.report["hydraulic_incision_m3"])


# ===== §3.3 非线性坡面扩散 =====


def _stepped_field(nlat: int = 16, nlon: int = 32, *, step: float = 100.0) -> np.ndarray:
    """沿纬度方向的阶梯地形：细网格下坡面接近休止角，触发非线性放大。"""
    rows = np.arange(nlat)[:, None]
    return np.asarray(rows * step, dtype=np.float64) * np.ones((1, nlon))


def test_nonlinear_diffusion_flat_terrain_unchanged() -> None:
    elev = np.full((12, 24), 250.0)
    out = te.hillslope_diffusion(
        elev, 1.0e-3, 1.0e7, n_steps=3, radius=SMALL_RADIUS, nonlinear=True
    )
    assert np.allclose(out, elev)


def test_nonlinear_diffusion_conserves_volume() -> None:
    """面通量两侧共享 ⇒ 面积加权体积严格守恒（线性/非线性都成立）。"""
    nlat, nlon = 16, 32
    rng = np.random.default_rng(3)
    elev = 500.0 + rng.normal(size=(nlat, nlon)) * 150.0
    area = _cell_area(nlat, nlon, SMALL_RADIUS)
    before = float((elev * area).sum())
    out = te.hillslope_diffusion(
        elev, 1.0e-5, 1.0e7, n_steps=2, radius=SMALL_RADIUS, nonlinear=True
    )
    assert float((out * area).sum()) == pytest.approx(before, rel=1e-9)


def test_nonlinear_diffusion_moves_more_than_linear_on_steep_terrain() -> None:
    """坡度接近休止角时，非线性扩散的平滑作用强于线性扩散（§3.3）。"""
    elev = _stepped_field()
    kappa, dt = 1.0e-6, 1.0e7
    linear = te.hillslope_diffusion(elev, kappa, dt, n_steps=2, radius=SMALL_RADIUS)
    nonlinear = te.hillslope_diffusion(
        elev, kappa, dt, n_steps=2, radius=SMALL_RADIUS, nonlinear=True
    )
    assert not np.allclose(linear, nonlinear)
    assert float(nonlinear.std()) < float(linear.std()) < float(elev.std())


def test_nonlinear_diffusion_validation() -> None:
    elev = np.zeros((8, 16))
    with pytest.raises(ValueError):
        te.hillslope_diffusion(elev, 1e-6, 1e7, radius=SMALL_RADIUS, nonlinear=True, talus_angle_deg=0.0)
    with pytest.raises(ValueError):
        te.hillslope_diffusion(elev, 1e-6, 1e7, radius=SMALL_RADIUS, max_amplification=1.0)
    with pytest.raises(ValueError):
        te.hillslope_diffusion(elev, 1e-6, 1e7, radius=SMALL_RADIUS, talus_angle_deg=95.0)


def test_nonlinear_diffusion_never_crosses_talus_angle() -> None:
    """斜率限制器：单步搬运不会把面坡度搬到休止角以下（§3.3 的"强制不越界"），
    即便 κ·dt 远超线性 CFL 也保持有界。"""
    nlat, nlon = 16, 32
    elev = _stepped_field(nlat, nlon, step=200.0)
    tan_beta = np.tan(np.deg2rad(te.DEFAULT_TALUS_ANGLE_DEG))
    dlat_m, _ = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, SMALL_RADIUS)
    limit = tan_beta * dlat_m
    assert 200.0 > limit  # 前提：初始面坡度超过休止角

    out = te.hillslope_diffusion(
        elev, 1.0e-2, 1.0e7, n_steps=1, radius=SMALL_RADIUS, nonlinear=True
    )
    assert np.all(np.isfinite(out))
    before = np.abs(elev[:-1, :] - elev[1:, :])
    after = np.abs(out[:-1, :] - out[1:, :])
    above = before > limit
    assert above.all()
    # 向休止角靠拢但绝不越过
    assert np.all(after[above] >= limit - 1e-9)
    assert np.all(after[above] <= before[above])


def test_simulate_erosion_hillslope_nonlinear_branch() -> None:
    """管线暴露非线性坡面扩散开关（§3.3）。"""
    elev, ocean = _cone_island()
    linear = ero.simulate_erosion(
        elev, ocean, precipitation=P_EARTH, hydraulic_steps=20, hillslope_kappa=1.0e-4,
        hillslope_dt=1.0e7, hillslope_steps=2, radius=SMALL_RADIUS,
    )
    nonlinear = ero.simulate_erosion(
        elev, ocean, precipitation=P_EARTH, hydraulic_steps=20, hillslope_kappa=1.0e-4,
        hillslope_dt=1.0e7, hillslope_steps=2, hillslope_nonlinear=True, radius=SMALL_RADIUS,
    )
    assert nonlinear.report["hillslope_nonlinear"] is True
    assert linear.report["hillslope_nonlinear"] is False
    assert not np.allclose(linear.thermal_drop, nonlinear.thermal_drop)


def test_simulate_erosion_hillslope_diffusion_enabled_by_default() -> None:
    """方案 §2.2 认为坡面扩散不可省，因此管线默认启用（可传 0 关闭）。"""
    elev, ocean = _cone_island()
    default = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=10)
    off = ero.simulate_erosion(
        elev, ocean, precipitation=P_EARTH, hydraulic_steps=10, hillslope_kappa=0.0
    )
    assert float(default.report["hillslope_kappa"]) == pytest.approx(ero.DEFAULT_HILLSLOPE_KAPPA)
    assert float(default.report["hillslope_kappa"]) > 0.0
    assert float(off.report["hillslope_kappa"]) == 0.0


# ===== §2.2 构造抬升 U =====


def test_simulate_erosion_uplift_raises_land_and_keeps_identity() -> None:
    """U 只作用于陆地；相位恒等式 H_final = H_bed + uplift − ΔH 仍成立（§2.2）。"""
    elev, ocean = _continent()
    kwargs = dict(precipitation=0.0, hydraulic_steps=20, hillslope_kappa=0.0)
    base = ero.simulate_erosion(elev, ocean, **kwargs)
    lifted = ero.simulate_erosion(elev, ocean, uplift=6.0, **kwargs)
    land = ~ocean
    assert np.allclose((lifted.elevation - base.elevation)[land], 6.0, atol=1e-9)
    assert np.allclose((lifted.elevation - base.elevation)[ocean], 0.0, atol=1e-12)
    assert np.allclose(lifted.uplift[land], 6.0)
    assert np.allclose(lifted.uplift[ocean], 0.0)
    assert lifted.report["uplift_total_m3"] > 0.0
    assert ero.validate_erosion(lifted) == []


def test_simulate_erosion_uplift_validation() -> None:
    elev, ocean = _continent()
    with pytest.raises(ValueError):
        ero.simulate_erosion(elev, ocean, uplift=-1.0)


def test_simulate_erosion_uplift_offsets_incision() -> None:
    """抬升与下切在同一演化里竞争：净降低量被抬升抵消一部分（§2.2）。"""
    elev, ocean = _continent()
    kwargs = dict(precipitation=P_EARTH, hydraulic_steps=60)
    base = ero.simulate_erosion(elev, ocean, **kwargs)
    lifted = ero.simulate_erosion(elev, ocean, uplift=5.0, **kwargs)
    land = ~ocean
    assert float(lifted.elevation[land].mean()) > float(base.elevation[land].mean())


# ===== 行为 10：输出一致性验证（§七 输出与校验）=====


def test_validate_erosion_clean_on_pipeline_output() -> None:
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=50)
    assert ero.validate_erosion(result) == []


def test_validate_erosion_detects_non_finite() -> None:
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=20)
    bad = dataclasses.replace(result, elevation=result.elevation.copy())
    bad.elevation[3, 3] = np.nan
    assert any("非有限" in p for p in ero.validate_erosion(bad))


def test_validate_erosion_detects_river_on_ocean() -> None:
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=20)
    river = result.river_network.copy()
    river[ocean] = True
    bad = dataclasses.replace(result, river_network=river)
    assert any("河流" in p for p in ero.validate_erosion(bad))


def test_validate_erosion_detects_dead_end_land_cell() -> None:
    """陆地单元无出流（非极点行）→ 违反无洼地约束。"""
    elev, ocean = _continent()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=20)
    i, j = elev.shape[0] - 3, 5
    assert not ocean[i, j]  # 确认测的是内陆陆地单元
    direction = result.flow_direction.copy()
    direction[i, j] = 0
    bad = dataclasses.replace(result, flow_direction=direction)
    assert any("出流" in p for p in ero.validate_erosion(bad))


def test_validate_erosion_detects_phase_identity_violation() -> None:
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=20)
    bad = dataclasses.replace(result, hydraulic_drop=result.hydraulic_drop + 100.0)
    assert any("恒等" in p for p in ero.validate_erosion(bad))


def test_validate_erosion_detects_talus_violation() -> None:
    """细网格下人为造出超休止角陡壁 → 校验应报出。"""
    nlat, nlon = 16, 32
    elev = np.zeros((nlat, nlon))
    elev[nlat // 2 :, :] = 1000.0
    ocean = np.zeros_like(elev, dtype=bool)
    filled = hd.fill_pits(elev, ocean)
    direction = hd.flow_directions(filled, ocean, radius=SMALL_RADIUS)
    acc = hd.flow_accumulation(direction, ocean)
    bad = ero.ErosionResult(
        elevation=elev,
        erosion_bed=filled,
        uplift=np.zeros_like(elev),
        filled_elevation=filled,
        hydraulic_drop=np.zeros_like(elev),
        thermal_drop=np.zeros_like(elev),
        flow_direction=direction,
        flow_accumulation=acc,
        river_network=np.zeros_like(elev, dtype=bool),
        river_width=np.zeros_like(elev),
        report={},
        seed=0,
    )
    problems = ero.validate_erosion(
        bad, radius=SMALL_RADIUS, talus_angle_deg=33.0, check_identity=False
    )
    assert any("休止角" in p for p in problems)


# ===== §6.3 水力侵蚀的 JAX 内核后端（可选依赖） =====


def test_simulate_erosion_rejects_unknown_hydraulic_backend() -> None:
    """后端必须是已注册的实现之一，拼错时立即报错而不是静默换内核。"""
    elev, ocean = _cone_island()
    with pytest.raises(ValueError, match="hydraulic_backend"):
        ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_backend="cuda")


def test_simulate_erosion_records_hydraulic_backend() -> None:
    """report 必须记录实际使用的内核，保证"跑了哪个后端"可观测。"""
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(elev, ocean, precipitation=P_EARTH, hydraulic_steps=20)
    assert result.report["hydraulic_backend"] == "numba"
    assert set(ero.HYDRAULIC_BACKENDS) == {"numba", "jax"}


@pytest.mark.skipif(not jax_kernels.jax_available(), reason="未安装 jax（pip install jax）")
def test_simulate_erosion_jax_backend_matches_numba() -> None:
    """``hydraulic_backend="jax"`` 与缺省 Numba 路径在第三层管线级等价。

    热力侵蚀与河网步骤是共用的，因此这里验证的是"换内核后整条第三层管线结果不变"。
    """
    elev, ocean = _cone_island()
    kwargs = dict(precipitation=P_EARTH, hydraulic_steps=40, seed=3)
    numba = ero.simulate_erosion(elev, ocean, **kwargs)
    jax_result = ero.simulate_erosion(elev, ocean, hydraulic_backend="jax", **kwargs)

    assert jax_result.report["hydraulic_backend"] == "jax"
    assert numba.report["hydraulic_backend"] == "numba"
    for name in ("elevation", "hydraulic_drop", "river_width"):
        deviation = float(
            np.abs(
                np.asarray(getattr(numba, name), dtype=np.float64)
                - np.asarray(getattr(jax_result, name), dtype=np.float64)
            ).max()
        )
        assert deviation <= 1e-9, f"{name} 偏差 {deviation:.3e} 超限"
    # 河网是分类结果，必须逐格一致
    np.testing.assert_array_equal(numba.river_network, jax_result.river_network)


@pytest.mark.skipif(not jax_kernels.jax_available(), reason="未安装 jax（pip install jax）")
def test_simulate_erosion_jax_backend_passes_validation() -> None:
    """JAX 后端产出的结果同样满足第三层的不变量复核。"""
    elev, ocean = _cone_island()
    result = ero.simulate_erosion(
        elev, ocean, precipitation=P_EARTH, hydraulic_steps=40, hydraulic_backend="jax"
    )
    assert ero.validate_erosion(result) == []
