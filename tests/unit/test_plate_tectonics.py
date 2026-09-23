"""完整板块构造管线测试（《第一层完善》§7 伪代码管线）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import plate_tectonics as pt


def test_trench_depth_young_and_old_plates() -> None:
    """§5.1：D = -8000 - 2000*log10(age/10)，年龄越大海沟越深。"""
    assert pt.trench_depth(age_ma=10.0) == pytest.approx(-8000.0)
    assert pt.trench_depth(age_ma=100.0) == pytest.approx(-10000.0)
    assert pt.trench_depth(age_ma=1.0) == pytest.approx(-6000.0)
    # 单调性
    ages = np.array([5.0, 20.0, 80.0, 150.0])
    depths = np.array([pt.trench_depth(a) for a in ages])
    assert np.all(np.diff(depths) < 0)


def test_volcanic_arc_uplift_positive() -> None:
    """§5.2：火山弧为抬升带，幅度在 [1000, 3000] m 量级。"""
    uplift = pt.volcanic_arc_uplift(distance_km=150.0)
    assert 1000.0 <= uplift <= 3000.0


def test_generate_tectonic_field_shape_and_qc() -> None:
    result = pt.generate_tectonic_field(
        nlat=48, nlon=96, n_seeds=150, n_major=7, seed=42, time_ma=200.0
    )
    assert result.elevation.shape == (48, 96)
    assert result.plate_map.shape == (48, 96)
    assert result.plate_map.dtype.kind == "i"
    elev = np.asarray(result.elevation)
    assert np.all(np.isfinite(elev))
    # 高程量级合理（构造层：-8000 ~ 8000 m）
    assert np.nanmin(elev) > -12000.0
    assert np.nanmax(elev) < 9000.0
    # 至少存在 6 个板块
    assert len(np.unique(result.plate_map)) >= 6


def test_generate_tectonic_field_deterministic() -> None:
    a = pt.generate_tectonic_field(nlat=32, nlon=64, seed=7, time_ma=100.0)
    b = pt.generate_tectonic_field(nlat=32, nlon=64, seed=7, time_ma=100.0)
    assert np.array_equal(a.plate_map, b.plate_map)
    assert np.allclose(a.elevation, b.elevation)


def test_generate_tectonic_field_time_scales_elevation() -> None:
    """更长演化时间 → 汇聚边界抬升更强（高程差更大）。"""
    short = pt.generate_tectonic_field(nlat=48, nlon=96, seed=11, time_ma=50.0)
    long = pt.generate_tectonic_field(nlat=48, nlon=96, seed=11, time_ma=400.0)
    spread_s = np.ptp(np.asarray(short.elevation))
    spread_l = np.ptp(np.asarray(long.elevation))
    assert spread_l > spread_s


def test_generate_tectonic_field_boundary_types_present() -> None:
    result = pt.generate_tectonic_field(nlat=48, nlon=96, seed=3, time_ma=150.0)
    bt = np.asarray(result.boundary_type)
    from virtual_world.terrain.euler_poles import BoundaryType

    for bt_val in (BoundaryType.CONVERGENT, BoundaryType.DIVERGENT, BoundaryType.TRANSFORM):
        assert int((bt == bt_val).sum()) > 0, f"缺少边界类型 {bt_val}"


def test_generate_tectonic_field_subduction_uplift() -> None:
    """俯冲带：汇聚边界两侧应有海沟（洋壳俯冲侧）与造山高山（陆壳碰撞侧）。"""
    result = pt.generate_tectonic_field(
        nlat=48, nlon=96, seed=5, time_ma=300.0, subduction=True
    )
    elev = np.asarray(result.elevation)
    bt = np.asarray(result.boundary_type)
    from virtual_world.terrain.euler_poles import BoundaryType

    # 全局深沟单元占比受限（海沟只沿边界出现，不会压平全图）
    trench_count = int((elev < -6000.0).sum())
    assert trench_count > 0
    assert trench_count / elev.size < 0.05
    # 汇聚边界上同时存在海沟侧（海洋俯冲）与造山侧（> 2000 m）
    conv = bt == BoundaryType.CONVERGENT
    assert int((elev[conv] > 2000.0).sum()) > 0
    assert int((elev[conv] < -6000.0).sum()) > 0
