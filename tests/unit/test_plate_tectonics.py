"""完整板块构造管线测试（《第一层完善》§1–§7）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import spherical
from virtual_world.core.constants import EARTH_RADIUS
from virtual_world.core.cubed_sphere import CubedSphere
from virtual_world.core.grid import FIELDS
from virtual_world.terrain import isostasy, voronoi
from virtual_world.terrain import plate_tectonics as pt
from virtual_world.terrain.euler_poles import BoundaryType

# ===== §5 俯冲带经验关系 =====


def test_trench_depth_young_and_old_plates() -> None:
    """§5.1：D = -8000 - 2000*log10(age/10)，年龄越大海沟越深。"""
    assert pt.trench_depth(age_ma=10.0) == pytest.approx(-8000.0)
    assert pt.trench_depth(age_ma=100.0) == pytest.approx(-10000.0)
    assert pt.trench_depth(age_ma=1.0) == pytest.approx(-6000.0)
    ages = np.array([5.0, 20.0, 80.0, 150.0])
    depths = np.array([pt.trench_depth(a) for a in ages])
    assert np.all(np.diff(depths) < 0)


def test_trench_depth_clamped_to_spec_range() -> None:
    """§5.1：海沟深度限定在 -4000 ~ -11000 m（年轻板片浅、老板片深）。"""
    assert pt.trench_depth(0.01) >= -11000.0
    assert pt.trench_depth(5000.0) == pytest.approx(-11000.0)
    assert pt.trench_depth(1e-6) == pytest.approx(-4000.0)


def test_volcanic_arc_uplift_positive() -> None:
    """§5.2：火山弧为抬升带，幅度在 [1000, 3000] m 量级。"""
    for distance in (100.0, 150.0, 200.0):
        assert 1000.0 <= pt.volcanic_arc_uplift(distance) <= 3000.0


def test_oceanic_floor_age_subsidence() -> None:
    """§1.2.3：洋中脊最浅（≈-2500 m），随年龄冷却下沉（depth ∝ √age）。"""
    ridge = pt.ocean_floor_depth(0.0)
    young = pt.ocean_floor_depth(20.0)
    old = pt.ocean_floor_depth(150.0)
    assert ridge == pytest.approx(-2500.0, abs=1.0)
    assert young < ridge and old < young
    # 老洋壳稳定在深海平原量级
    assert -7000.0 < old < -5000.0


# ===== §4.1 碰撞角修正因子 =====


def test_convergence_factor_head_on_oblique_parallel() -> None:
    """f(θ)=|v_rel·n̂|/|v_rel|：正面碰撞=1，45°≈0.707，纯走滑=0。"""
    v = np.array([0.0, 1.0, 0.0])
    head_on = pt.convergence_factor(v, np.array([0.0, 1.0, 0.0]))
    oblique = pt.convergence_factor(v, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
    parallel = pt.convergence_factor(v, np.array([1.0, 0.0, 0.0]))
    assert head_on == pytest.approx(1.0)
    assert oblique == pytest.approx(np.sqrt(0.5))
    assert parallel == pytest.approx(0.0)
    assert pt.convergence_factor(np.zeros(3), np.array([0.0, 1.0, 0.0])) == 0.0


# ===== §1.4 边界配对与连通性 =====


def test_boundary_pairs_latitude_not_cyclic() -> None:
    """边界配对纬度非循环（极点约束），经度循环。"""
    plate = np.zeros((6, 5), dtype=np.int32)
    plate[3:, :] = 1  # 南北分界；若纬度循环会产生 0 行与 5 行的伪配对
    i, j, ni, nj = pt.boundary_pairs(plate)
    nlat = plate.shape[0]
    wrapped = ((i == 0) & (ni == nlat - 1)) | ((i == nlat - 1) & (ni == 0))
    assert not wrapped.any()
    assert i.size == 5  # 仅行 2/行 3 的 5 个界面


def test_boundary_pairs_longitude_is_cyclic() -> None:
    plate = np.zeros((3, 6), dtype=np.int32)
    plate[:, 3:] = 1
    i, j, ni, nj = pt.boundary_pairs(plate)
    # 列 2/3 与列 5/0 两处界面 × 3 行
    assert i.size == 6
    assert np.all(ni == i)


# ===== 管线整体行为 =====


def test_generate_tectonic_field_shape_and_qc() -> None:
    result = pt.generate_tectonic_field(
        nlat=48, nlon=96, n_seeds=150, n_major=7, seed=42, time_ma=200.0
    )
    assert result.elevation.shape == (48, 96)
    assert result.plate_map.shape == (48, 96)
    assert result.plate_map.dtype.kind == "i"
    elev = np.asarray(result.elevation)
    assert np.all(np.isfinite(elev))
    assert len(np.unique(result.plate_map)) == 7


def test_generate_tectonic_field_elevation_within_gridstate_range() -> None:
    """产出必须落在 GridState 的 elevation 有效范围内（集成就绪）。"""
    result = pt.generate_tectonic_field(nlat=48, nlon=96, seed=42, time_ma=250.0)
    lo, hi = FIELDS["elevation"].valid_range
    elev = np.asarray(result.elevation)
    assert elev.min() >= lo
    assert elev.max() <= hi


def test_generate_tectonic_field_derived_ocean_fields() -> None:
    """§3.1：派生 is_ocean / ocean_depth，且深度落在字段有效范围内。"""
    result = pt.generate_tectonic_field(nlat=48, nlon=96, seed=9, time_ma=200.0)
    elev = np.asarray(result.elevation)
    is_ocean = np.asarray(result.is_ocean)
    depth = np.asarray(result.ocean_depth)
    assert is_ocean.dtype == bool
    assert np.array_equal(is_ocean, elev < 0.0)
    assert np.allclose(depth[is_ocean], -elev[is_ocean])
    assert np.allclose(depth[~is_ocean], 0.0)
    lo, hi = FIELDS["ocean_depth"].valid_range
    assert depth.min() >= lo and depth.max() <= hi


def test_continental_interior_is_above_sea_level() -> None:
    """大陆板块内部必须高于海平面，否则下游 ``land_mask = elevation > 0`` 会判错。"""
    result = pt.generate_tectonic_field(nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0)
    elev = np.asarray(result.elevation)
    continental = ~result.plate_is_oceanic[result.plate_map]
    interior = np.asarray(result.boundary_type) == BoundaryType.INTERIOR
    cells = continental & interior
    assert cells.any()
    assert np.all(elev[cells] > 0.0)
    assert np.all(np.asarray(result.is_ocean)[cells] == False)  # noqa: E712 - 掩码语义检查


def test_generate_tectonic_field_plates_connected() -> None:
    """§1.4 关键约束：每个大板块在球面上连通。"""
    result = pt.generate_tectonic_field(nlat=48, nlon=96, seed=42, time_ma=200.0)
    for lab in np.unique(result.plate_map):
        _, n = voronoi.connected_components(result.plate_map == lab)
        assert n == 1, f"板块 {lab} 有 {n} 个连通分量"


def test_generate_tectonic_field_interior_marked_interior() -> None:
    """内部单元标为 INTERIOR，与真实转换边界可区分。"""
    result = pt.generate_tectonic_field(nlat=48, nlon=96, seed=3, time_ma=150.0)
    bt = np.asarray(result.boundary_type)
    for bt_val in (
        BoundaryType.CONVERGENT,
        BoundaryType.DIVERGENT,
        BoundaryType.TRANSFORM,
        BoundaryType.INTERIOR,
    ):
        assert int((bt == bt_val).sum()) > 0, f"缺少边界类型 {bt_val}"


def test_generate_tectonic_field_deterministic() -> None:
    a = pt.generate_tectonic_field(nlat=32, nlon=64, seed=7, time_ma=100.0)
    b = pt.generate_tectonic_field(nlat=32, nlon=64, seed=7, time_ma=100.0)
    assert np.array_equal(a.plate_map, b.plate_map)
    assert np.allclose(a.elevation, b.elevation)


def test_generate_tectonic_field_time_scales_elevation() -> None:
    """更长演化时间 → 汇聚边界抬升更强（高程差更大）。"""
    short = pt.generate_tectonic_field(nlat=48, nlon=96, seed=11, time_ma=50.0)
    long = pt.generate_tectonic_field(nlat=48, nlon=96, seed=11, time_ma=400.0)
    assert np.ptp(np.asarray(long.elevation)) > np.ptp(np.asarray(short.elevation))


# ===== 海陆比可配置（§1.6-1）与洋壳属性（§1.2.3）=====


def test_ocean_fraction_controls_ocean_area() -> None:
    """陆地面积占比可配置；误差受板块面积粒度与陆内裂谷下沉限制。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.6
    )
    achieved = float(np.asarray(result.is_ocean).mean())
    assert abs(achieved - 0.6) <= 0.12


def test_merged_plates_are_reasonably_balanced() -> None:
    """大板块面积不得由单一板块主导（否则海陆比不可控）。"""
    result = pt.generate_tectonic_field(nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0)
    areas = np.bincount(np.asarray(result.plate_map).ravel(), minlength=8) / result.plate_map.size
    assert areas.max() < 0.35
    assert areas.min() > 0.01


def test_ocean_fraction_monotonic() -> None:
    """更大的 ocean_fraction 必须给出更大的海洋面积。"""
    low = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.3
    )
    high = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.8
    )
    assert float(np.asarray(high.is_ocean).mean()) > float(np.asarray(low.is_ocean).mean())


def test_ocean_fraction_validation() -> None:
    with pytest.raises(ValueError):
        pt.generate_tectonic_field(nlat=32, nlon=64, ocean_fraction=1.5)
    with pytest.raises(ValueError):
        pt.generate_tectonic_field(nlat=32, nlon=64, ocean_fraction=-0.1)


def test_mid_ocean_ridge_shallower_than_abyssal_plain() -> None:
    """§1.2.3：洋中脊（离散边界附近）应比远离洋脊的深海平原浅。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=5, time_ma=200.0, ocean_fraction=0.7
    )
    elev = np.asarray(result.elevation)
    is_ocean = np.asarray(result.is_ocean)
    ridge = np.asarray(result.ridge_mask)
    assert ridge.any() and (is_ocean & ~ridge).any()
    assert elev[ridge].mean() > elev[is_ocean & ~ridge].mean()


# ===== 俯冲带：海沟与火山弧分居边界两侧 =====


def test_subduction_trench_and_arc_on_opposite_sides() -> None:
    """海沟位于俯冲洋壳侧，火山弧抬升位于上盘（陆壳）侧。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=5, time_ma=300.0, ocean_fraction=0.7
    )
    elev = np.asarray(result.elevation)
    is_ocean = np.asarray(result.is_ocean)
    bt = np.asarray(result.boundary_type)
    conv = bt == BoundaryType.CONVERGENT

    trench = conv & is_ocean & (elev < -4000.0)
    assert trench.any(), "未生成海沟"
    # 海沟深度不得超出方案范围
    assert elev[trench].min() >= pt.TRENCH_DEPTH_MIN - 1e-6
    # 存在陆壳侧汇聚单元被抬升到造山带高度
    assert (conv & ~is_ocean & (elev > 2000.0)).any(), "未生成造山带"


def test_subduction_disabled_leaves_no_trench() -> None:
    """关闭俯冲后不应出现海沟；开启后最深高程显著更深。"""
    off = pt.generate_tectonic_field(nlat=48, nlon=96, seed=5, time_ma=300.0, subduction=False)
    on = pt.generate_tectonic_field(nlat=48, nlon=96, seed=5, time_ma=300.0, subduction=True)
    off_min = float(np.asarray(off.elevation).min())
    on_min = float(np.asarray(on.elevation).min())
    assert off_min > on_min
    # 无俯冲时最深仅到深海平原量级
    assert off_min >= -7000.0


# ===== §4.2 质量守恒 =====


def test_plate_length_scale_km_matches_plate_area() -> None:
    """L0 = sqrt(板块面积)，与板块面积占比一致（§4.2）。"""
    plate_map = np.zeros((4, 8), dtype=np.int32)
    plate_map[:, 4:] = 1
    lengths = pt.plate_length_scale_km(plate_map, 2)
    assert lengths.shape == (2,)
    assert lengths[0] == pytest.approx(lengths[1], rel=1e-12)
    # 面积占比 1/2 ⇒ 面积 = 2*pi*R^2 ⇒ L0 = sqrt(2*pi)*R
    expected_km = np.sqrt(2.0 * np.pi) * pt.EARTH_RADIUS / 1000.0
    assert lengths[0] == pytest.approx(expected_km, rel=1e-6)


def test_convergence_distance_and_beta_limits() -> None:
    """d_L = rate*time*calibration；beta_max = 1/(1 - d_L/L0) 且被 beta_cap 截断（§4.2）。"""
    distance = pt.convergence_distance_km(np.array([1.0, 2.0]), 100.0)
    assert np.allclose(distance, [500.0, 1000.0])
    beta = pt.mass_conservation_beta(np.array([0.0, 1.0]), np.array([1000.0, 1000.0]), 100.0)
    assert beta[0] == pytest.approx(1.0)
    assert beta[1] == pytest.approx(2.0)
    # 汇聚距离超过板块长度时被 beta_cap 截断而非发散
    extreme = pt.mass_conservation_beta(np.array([1.0e3]), np.array([100.0]), 100.0)
    assert extreme[0] == pytest.approx(pt.MAX_SHORTENING_FACTOR_CAP)


def test_mass_conservation_is_never_violated() -> None:
    """地壳厚度绝不超过质量守恒上限（§4.2 硬约束）。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.7
    )
    crust = np.asarray(result.crust_thickness)
    cap = pt.CONTINENTAL_CRUST_KM * np.asarray(result.max_shortening_factor)
    continental = ~result.plate_is_oceanic[result.plate_map]
    assert np.all(crust[continental] <= cap[continental] + 1e-9)
    assert np.all(np.asarray(result.max_shortening_factor) >= 1.0)


def test_mass_conservation_limits_short_timescale_orogeny() -> None:
    """缩短量不足时增厚被截断：短时间无法造出与长时间同等的高山（§4.2）。"""
    short = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=20.0, ocean_fraction=0.7
    )
    long = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.7
    )
    assert np.asarray(short.crust_thickness).max() < np.asarray(long.crust_thickness).max()
    assert np.asarray(short.elevation).max() < np.asarray(long.elevation).max()


def test_mass_conservation_cap_loose_at_long_timescale() -> None:
    """长时间尺度下汇聚距离充裕，质量守恒上限不再收紧（不改变既有标定）。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.7
    )
    crust = np.asarray(result.crust_thickness)
    cap = pt.CONTINENTAL_CRUST_KM * np.asarray(result.max_shortening_factor)
    peak = int(np.argmax(crust))
    assert crust.ravel()[peak] < cap.ravel()[peak] - 1e-6


def test_plate_shortening_factor_consistency() -> None:
    """beta = C/C_0（仅大陆），缩短量 = L0*(1 - 1/beta)（§4.2）。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.7
    )
    beta = np.asarray(result.plate_shortening_factor)
    shortening = np.asarray(result.plate_shortening_km)
    crust = np.asarray(result.crust_thickness)
    plate_map = np.asarray(result.plate_map)
    lengths = pt.plate_length_scale_km(plate_map, result.n_major)

    assert np.all(beta[result.plate_is_oceanic] == 1.0)
    assert np.all(shortening[result.plate_is_oceanic] == 0.0)
    for p in range(result.n_major):
        if result.plate_is_oceanic[p]:
            continue
        peak = float(crust[plate_map == p].max())
        assert beta[p] == pytest.approx(max(peak / pt.CONTINENTAL_CRUST_KM, 1.0), rel=1e-9)
        assert shortening[p] == pytest.approx(lengths[p] * (1.0 - 1.0 / beta[p]), rel=1e-9)


# ===== §4.3 造山带高宽比 =====


def test_orogen_aspect_ratio_from_pipeline() -> None:
    """高宽比与由其反推的造山带半宽落在合理量级（§4.3）。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=5, time_ma=300.0, ocean_fraction=0.7
    )
    assert 0.0 < result.orogen_aspect_ratio < 1.0
    assert 10.0 < result.orogen_half_width_km < 1000.0


# ===== §2.5 碰撞带角动量守恒 =====


def test_plate_components_groups_colliding_plates() -> None:
    """互相碰撞的板块被并查集成同一分量。"""
    comps = pt._plate_components(
        np.array([0, 1, 3]), np.array([1, 2, 4]), n_plates=5
    )
    assert [0, 1, 2] in comps
    assert [3, 4] in comps


def test_collide_plates_conserves_angular_momentum() -> None:
    """合并角速度满足角动量守恒 L = I_A*w_A + I_B*w_B（§2.5）。"""
    plate_map = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    index_a = np.array([1, 5])  # (0,1),(1,1)
    index_b = np.array([2, 6])  # (0,2),(1,2)
    omegas = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    result = pt.collide_plates(
        plate_map,
        np.array([False, False]),
        index_a,
        index_b,
        np.array([True, True]),
        omegas,
    )
    assert result.components == [[0, 1]]
    assert result.mask.sum() == 4  # 两个边界面的 4 个单元
    # 两板块面积相同 ⇒ 转动惯量相同 ⇒ 合并角速度为算平均
    assert np.allclose(result.omega_merged[0], [0.5, 0.0, 0.0])
    assert np.allclose(result.omega_merged[1], [0.5, 0.0, 0.0])
    assert result.energy_j > 0.0
    assert np.all(result.uplift >= 0.0)


def test_collide_plates_skips_oceanic_pairs() -> None:
    """洋-洋/洋-陆汇聚是俯冲而非碰撞，不参与角动量合并（§2.5）。"""
    plate_map = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    result = pt.collide_plates(
        plate_map,
        np.array([True, True]),
        np.array([1]),
        np.array([2]),
        np.array([True]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    )
    assert result.components == []
    assert not result.mask.any()
    assert result.energy_j == 0.0


def test_collision_energy_scales_with_angular_velocity_squared() -> None:
    """动能预算 ∝ |omega|^2（§2.5），且真实板块速率下量级极小。"""
    plate_map = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    index_a = np.array([1, 5])
    index_b = np.array([2, 6])
    omegas = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    kwargs = dict(index_a=index_a, index_b=index_b, is_convergent=np.array([True, True]), omegas=omegas)
    base = pt.collide_plates(plate_map, np.array([False, False]), **kwargs)
    scaled = pt.collide_plates(
        plate_map,
        np.array([False, False]),
        rate_to_rad_per_s=10.0 * pt.MODEL_RATE_TO_RAD_PER_S,
        **kwargs,
    )
    assert scaled.energy_j == pytest.approx(100.0 * base.energy_j, rel=1e-9)
    # 真实速率（~2.5e-17 rad/s）下抬升不到 1 mm——如实反映而非人为放大
    assert float(np.asarray(base.uplift).max()) < 1.0e-3


def test_collision_mask_only_on_continental_collision() -> None:
    """全洋壳世界无陆-陆碰撞带（§2.5）。"""
    result = pt.generate_tectonic_field(
        nlat=48, nlon=96, n_major=8, seed=5, time_ma=200.0, ocean_fraction=1.0
    )
    assert not np.asarray(result.collision_mask).any()
    assert result.collision_energy_j == 0.0


def test_pipeline_marks_continental_collision_zone() -> None:
    """大陆板块相撞时给出碰撞带与守恒的合并角速度（§2.5）。"""
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=200.0, ocean_fraction=0.7
    )
    mask = np.asarray(result.collision_mask)
    assert mask.any()
    assert np.asarray(result.plate_omega_merged).shape == (result.n_major, 3)
    assert result.collision_energy_j > 0.0
    assert np.all(np.isfinite(np.asarray(result.collision_uplift)))


# ===== §2.2（第三层）构造抬升速率 U =====


def test_uplift_rate_matches_crust_thickening() -> None:
    """U = max(airy(C) − airy(C_eq), 0)/time_ma，海洋为 0（§2.2 的跨层因果链）。"""
    time_ma = 200.0
    result = pt.generate_tectonic_field(
        nlat=60, nlon=120, n_major=8, seed=21, time_ma=time_ma, ocean_fraction=0.7
    )
    rate = np.asarray(result.uplift_rate_m_per_ma)
    crust = np.asarray(result.crust_thickness)
    oceanic = result.plate_is_oceanic[result.plate_map]

    assert rate.shape == result.shape
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0.0)
    assert np.all(rate[oceanic] == 0.0)
    assert float(rate[~oceanic].max()) > 0.0

    reference = float(isostasy.airy_elevation(pt.CONTINENTAL_CRUST_KM, c_ref=pt.REF_CRUST_KM))
    airy = np.asarray(isostasy.airy_elevation(crust, c_ref=pt.REF_CRUST_KM))
    expected = np.where(oceanic, 0.0, np.maximum(airy - reference, 0.0) / time_ma)
    assert np.allclose(rate, expected, rtol=1e-9, atol=1e-12)


# ===== §1.5 立方球网格 =====


def test_cubed_sphere_grid_validation() -> None:
    with pytest.raises(ValueError):
        pt.generate_tectonic_field(grid="sphere", nlat=8, nlon=16)
    with pytest.raises(ValueError):
        pt.generate_tectonic_field(grid="cubed_sphere")  # 缺 n_side
    with pytest.raises(ValueError):
        pt.generate_tectonic_field(grid="latlon")  # 缺 nlat/nlon


def test_cubed_sphere_pipeline_shapes_and_quality() -> None:
    """立方球上跑完整板块构造管线（§1.5）：形状、板块数、边界类型、字段范围。"""
    sphere = CubedSphere(14)
    result = pt.generate_tectonic_field(
        grid="cubed_sphere", n_side=14, n_major=7, seed=42, time_ma=200.0, ocean_fraction=0.7
    )
    assert result.elevation.shape == (6, 14, 14)
    assert result.shape == (6, 14, 14)
    assert len(np.unique(result.plate_map)) == 7
    for code in (
        BoundaryType.CONVERGENT,
        BoundaryType.DIVERGENT,
        BoundaryType.TRANSFORM,
        BoundaryType.INTERIOR,
    ):
        assert int((np.asarray(result.boundary_type) == code).sum()) > 0
    assert np.asarray(result.is_ocean).dtype == bool
    assert np.array_equal(np.asarray(result.is_ocean), np.asarray(result.elevation) < 0.0)
    # 落到 GridState 的 elevation 有效范围内（与经纬网格同一约束）
    lo, hi = FIELDS["elevation"].valid_range
    assert float(np.asarray(result.elevation).min()) >= lo
    assert float(np.asarray(result.elevation).max()) <= hi
    # 海陆比可配置（§1.6-1）
    assert abs(float(np.asarray(result.is_ocean).mean()) - 0.7) < 0.15
    # 碰撞带与造山带高宽比同样产出
    assert np.asarray(result.collision_mask).any()
    assert 0.0 < result.orogen_aspect_ratio < 1.0
    # 每格都有 4 个合法邻居（立方球无域外）
    assert sphere.neighbors().min() >= 0


def test_cubed_sphere_plates_connected_and_deterministic() -> None:
    """大板块在立方球上连通（§1.4），且结果确定。"""
    sphere = CubedSphere(12)
    kwargs = dict(grid="cubed_sphere", n_side=12, n_major=6, seed=7, time_ma=150.0)
    a = pt.generate_tectonic_field(**kwargs)
    b = pt.generate_tectonic_field(**kwargs)
    assert np.array_equal(a.plate_map, b.plate_map)
    assert np.allclose(a.elevation, b.elevation)
    for lab in np.unique(a.plate_map):
        _, count = voronoi.connected_components(np.asarray(a.plate_map) == lab, sphere.neighbors())
        assert count == 1, f"板块 {lab} 不连通"


def test_cubed_sphere_cell_uniformity_makes_ocean_fraction_area_accurate() -> None:
    """§1.5 的直接收益：单元近等面积 ⇒ 按单元数控制的海陆比同时是面积精确的。

    经纬网格在高纬度的单元面积极小，按单元数控制的海洋占比与其真实面积占比
    相差可达 5% 以上；立方球把这一差距压到 1% 以内。
    """
    sphere = CubedSphere(14)
    cs = pt.generate_tectonic_field(
        grid="cubed_sphere", n_side=14, n_major=7, seed=42, time_ma=200.0, ocean_fraction=0.7
    )
    areas = sphere.areas()
    cs_count = float(np.asarray(cs.is_ocean).mean())
    cs_area = float((areas * np.asarray(cs.is_ocean)).sum() / areas.sum())
    assert abs(cs_count - cs_area) < 0.02

    latlon = pt.generate_tectonic_field(
        nlat=42, nlon=84, n_major=7, seed=42, time_ma=200.0, ocean_fraction=0.7
    )
    latlon_areas = spherical.cell_areas(spherical.lat_edges(42), 84, EARTH_RADIUS)
    ll_count = float(np.asarray(latlon.is_ocean).mean())
    ll_area = float((latlon_areas * np.asarray(latlon.is_ocean)).sum() / latlon_areas.sum())
    assert abs(ll_count - ll_area) > 0.05
    assert abs(cs_count - cs_area) < abs(ll_count - ll_area)


def test_regrid_tectonic_result_to_latlon() -> None:
    """立方球 → 经纬的互转桥接（§1.5）：连续场双线性、编码场最近邻。"""
    sphere = CubedSphere(14)
    cs = pt.generate_tectonic_field(
        grid="cubed_sphere", n_side=14, n_major=7, seed=5, time_ma=200.0, ocean_fraction=0.7
    )
    latlon = pt.regrid_tectonic_result(cs, sphere, 36, 72)
    assert latlon.elevation.shape == (36, 72)
    assert latlon.plate_map.shape == (36, 72)
    assert latlon.crust_thickness.shape == (36, 72)

    # 连续场：与直接双线性一致
    direct = sphere.to_latlon(np.asarray(cs.elevation), 36, 72)
    assert np.allclose(latlon.elevation, direct, atol=1e-9)
    # 编码场：值域不扩张（最近邻不会造出新类别）
    assert set(np.unique(latlon.plate_map)) <= set(np.unique(cs.plate_map))
    assert set(np.unique(latlon.boundary_type)) <= set(np.unique(cs.boundary_type))
    assert latlon.boundary_type.dtype == np.asarray(cs.boundary_type).dtype
    assert latlon.is_ocean.dtype == bool
    # 逐板块向量与标量与网格无关，原样保留
    assert np.array_equal(latlon.plate_is_oceanic, cs.plate_is_oceanic)
    assert np.array_equal(latlon.plate_shortening_factor, cs.plate_shortening_factor)
    assert latlon.orogen_aspect_ratio == cs.orogen_aspect_ratio
    assert latlon.n_major == cs.n_major
    # 连续场（含第三层的抬升速率 U）随重网格一起转换
    assert latlon.uplift_rate_m_per_ma.shape == (36, 72)
    assert np.allclose(
        latlon.uplift_rate_m_per_ma,
        sphere.to_latlon(np.asarray(cs.uplift_rate_m_per_ma), 36, 72),
        atol=1e-9,
    )
    # 掩码场与逐格连续场均保持形状与有限性
    assert np.asarray(latlon.collision_mask).shape == (36, 72)
    assert np.all(np.isfinite(np.asarray(latlon.collision_uplift)))
    assert np.all(np.isfinite(np.asarray(latlon.max_shortening_factor)))


def test_regrid_rejects_latlon_result() -> None:
    """经纬结果无需重网格，调用即报错（避免误用）。"""
    sphere = CubedSphere(6)
    latlon = pt.generate_tectonic_field(nlat=16, nlon=32, n_major=6, seed=1, time_ma=100.0)
    with pytest.raises(ValueError):
        pt.regrid_tectonic_result(latlon, sphere, 16, 32)



# ===== §6.4 JAX 长时间积分后端（可选依赖） =====


def test_crust_integration_jax_matches_numba() -> None:
    """``use_jax=True`` 走 lax.scan 内核，结果须与 Numba 内核逐位一致。

    两个内核共用同一分裂格式（先增厚/减薄、再用更新后的厚度做重力松弛），
    因此这里要求**逐位相同**而不只是接近。
    """
    pytest.importorskip("jax")
    kwargs = {"nlat": 24, "nlon": 48, "n_major": 6, "seed": 3, "time_ma": 100.0}
    numba = pt.generate_tectonic_field(**kwargs)
    jax_result = pt.generate_tectonic_field(**kwargs, use_jax=True)

    np.testing.assert_array_equal(
        np.asarray(numba.crust_thickness), np.asarray(jax_result.crust_thickness)
    )
    np.testing.assert_array_equal(np.asarray(numba.elevation), np.asarray(jax_result.elevation))


def test_crust_integrator_resolver_defaults_to_numba() -> None:
    """缺省（use_jax=False）必须用 Numba 内核，避免可选依赖变成硬依赖。"""
    from virtual_world.terrain.isostasy import integrate_crust_thickness

    assert pt._resolve_crust_integrator(False) is integrate_crust_thickness


def test_crust_integrator_resolver_selects_jax() -> None:
    """use_jax=True 时必须返回 JAX 内核（而非静默退回 Numba）。"""
    pytest.importorskip("jax")
    from virtual_world.terrain.jax_kernels import integrate_crust_thickness_jax

    assert pt._resolve_crust_integrator(True) is integrate_crust_thickness_jax
    assert pt._resolve_crust_integrator(True) is not pt._resolve_crust_integrator(False)
