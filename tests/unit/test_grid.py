"""网格数据模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import backend
from virtual_world.core.grid import (
    FIELDS,
    GridState,
    get_resolution_level,
    group_fields,
    load_resolution_levels,
)
from virtual_world.core.planet import PlanetParams

EARTH = PlanetParams.from_preset("earth")


@pytest.fixture
def grid(backend_name: str) -> GridState:
    """2 度全球网格（90 x 180），面积计算使用地球半径。"""
    return GridState.from_resolution(2.0, planet=EARTH, seed=7, backend_name=backend_name)


# ===== 网格元数据 =====


def test_geometry_metadata(grid: GridState, backend_name: str) -> None:
    assert grid.shape == (90, 180)
    assert grid.size == 16200
    assert grid.resolution == 2.0
    assert grid.backend_name == backend_name
    assert grid.is_global
    assert grid.lat[0] == pytest.approx(-89.0)
    assert grid.lat[-1] == pytest.approx(89.0)
    assert grid.lon[0] == pytest.approx(-179.0)
    assert backend.to_numpy(grid.cell_area).shape == (90, 180)
    assert grid.total_area == pytest.approx(4 * np.pi * EARTH.radius**2, rel=1e-9)
    assert grid.planet.name == "earth"
    assert grid.seed == 7


def test_resolution_must_match_shape() -> None:
    with pytest.raises(ValueError, match="不一致"):
        GridState(nlat=90, nlon=180, resolution=1.0)
    with pytest.raises(ValueError, match="不一致"):
        GridState(nlat=90, nlon=360, resolution=2.0)


def test_unsupported_grid_type() -> None:
    with pytest.raises(NotImplementedError):
        GridState(nlat=90, nlon=180, resolution=2.0, grid_type="cubed_sphere")


def test_constructors() -> None:
    assert GridState.from_resolution(1.0).shape == (180, 360)
    assert GridState.from_level("fine").shape == (180, 360)
    assert GridState.from_planet(EARTH, 5.0).shape == (36, 72)
    with pytest.raises(ValueError, match="整除"):
        GridState.from_resolution(0.7)
    with pytest.raises(KeyError):
        GridState.from_level("no_such_level")


def test_resolution_levels_config() -> None:
    levels = load_resolution_levels()
    assert {"preview", "coarse", "medium", "fine", "high", "ultra"} <= set(levels)
    assert get_resolution_level("medium").cells == 16200
    assert levels["preview"].resolution == 10.0


def test_field_specs() -> None:
    assert FIELDS["elevation"].unit == "m"
    assert FIELDS["is_ocean"].dtype == "bool"
    assert FIELDS["T_soil"].dims == ("depth", "lat", "lon")
    assert FIELDS["elevation"].expected_shape(90, 180) == (90, 180)
    assert "elevation" in group_fields("terrain")
    assert "T_jan" in group_fields("temperature")
    assert GridState.field_names() == list(FIELDS)


# ===== 字段读写与校验 =====


def test_set_and_get_field(grid: GridState, rng: np.random.Generator) -> None:
    data = rng.uniform(-500, 3000, size=grid.shape)
    grid.set("elevation", data)
    assert grid.has("elevation")
    stored = grid.as_numpy("elevation")
    assert stored.dtype == np.float32
    assert np.allclose(stored, data, atol=1e-3)
    assert backend.backend_of(grid.get("elevation")) is grid.xp


def test_set_wrong_shape(grid: GridState) -> None:
    with pytest.raises(ValueError, match="形状"):
        grid.set("elevation", np.zeros((10, 10)))


def test_set_out_of_range(grid: GridState) -> None:
    with pytest.raises(ValueError, match="值 >"):
        grid.set("elevation", np.full(grid.shape, 1.0e5))
    with pytest.raises(ValueError, match="值 <"):
        grid.set("T_annual", np.full(grid.shape, -500.0))
    with pytest.raises(ValueError, match="非有限值"):
        grid.set("T_annual", np.full(grid.shape, np.nan))


def test_set_rejects_bad_input(grid: GridState) -> None:
    with pytest.raises(TypeError):
        grid.set("elevation", [[0.0]])
    with pytest.raises(KeyError):
        grid.set("no_such_field", np.zeros(grid.shape))
    with pytest.raises(KeyError):
        grid.get("T_annual")


def test_category_and_mask_dtypes(grid: GridState, rng: np.random.Generator) -> None:
    grid.set("is_ocean", rng.random(grid.shape) < 0.5)
    assert grid.as_numpy("is_ocean").dtype == np.bool_
    grid.set("climate_code", np.full(grid.shape, 12))
    assert grid.as_numpy("climate_code").dtype == np.int8
    # int8 越界在转换前就被拦截，不会被静默截断
    with pytest.raises(ValueError, match="值 >"):
        grid.set("climate_code", np.full(grid.shape, 200))


def test_three_dimensional_field(backend_name: str) -> None:
    state = GridState.from_resolution(10.0, planet=EARTH, backend_name=backend_name, nz=4)
    state.set("T_soil", np.full((4, 18, 36), 5.0))
    assert state.has("T_soil")
    assert state.as_numpy("T_soil").shape == (4, 18, 36)
    with pytest.raises(ValueError, match="形状"):
        state.set("T_soil", np.full((3, 18, 36), 5.0))


def test_constructor_prefilled_field(backend_name: str) -> None:
    state = GridState(
        nlat=18, nlon=36, resolution=10.0, backend=backend_name, elevation=np.zeros((18, 36))
    )
    assert state.has("elevation")
    with pytest.raises(ValueError):
        GridState(
            nlat=18,
            nlon=36,
            resolution=10.0,
            backend=backend_name,
            elevation=np.full((18, 36), 1.0e6),
        )


# ===== 诊断与统计 =====


def test_area_weighted_diagnostics(grid: GridState) -> None:
    grid.set("T_annual", np.full(grid.shape, 12.0))
    assert grid.global_mean("T_annual") == pytest.approx(12.0, rel=1e-6)
    assert grid.global_integral("T_annual") == pytest.approx(12.0 * grid.total_area, rel=1e-6)

    # 面积加权平均 cos(phi) 应为 pi/4，比按纬度均匀采样的算术平均更接近赤道
    cos_phi = np.cos(np.deg2rad(grid.lat))
    field = np.repeat(cos_phi[:, None], grid.nlon, axis=1)
    grid.set("T_annual", field)
    assert grid.global_mean("T_annual") == pytest.approx(np.pi / 4, rel=1e-3)
    assert grid.global_mean("T_annual") > float(field.mean())
    assert grid.zonal_mean("T_annual") == pytest.approx(cos_phi, rel=1e-6)


def test_nearest_index_and_value_at(grid: GridState) -> None:
    grid.set("T_annual", np.arange(grid.size, dtype=np.float64).reshape(grid.shape) / 1000.0)
    i, j = grid.nearest_index(0.0, 0.0)
    assert abs(grid.lat[i]) <= 1.0
    assert abs(grid.lon[j]) <= 1.0
    assert grid.value_at("T_annual", grid.lat[i], grid.lon[j]) == pytest.approx(
        grid.as_numpy("T_annual")[i, j]
    )


# ===== 网格变换 =====


def test_slice_region(grid: GridState) -> None:
    grid.set("T_annual", np.full(grid.shape, 3.0))
    sub = grid.slice_region(-10, 10, -30, 30)
    assert sub.shape == (10, 30)
    assert sub.resolution == 2.0
    assert not sub.is_global
    assert sub.lat.min() == pytest.approx(-9.0)
    assert sub.lat.max() == pytest.approx(9.0)
    assert sub.lon.min() == pytest.approx(-29.0)
    assert np.allclose(sub.as_numpy("T_annual"), 3.0)
    # 面积仍按相同半径计算
    assert sub.cell_area is not None


def test_slice_region_wraps_longitude(grid: GridState) -> None:
    sub = grid.slice_region(-2, 2, 170, -170)
    assert sub.nlon == 10
    assert sub.lon[0] == pytest.approx(171.0)
    assert sub.lon[-1] == pytest.approx(-171.0)


def test_slice_region_empty_range(grid: GridState) -> None:
    with pytest.raises(ValueError, match="未覆盖"):
        grid.slice_region(95, 96, -10, 10)


def test_coarsen_conserves_area_weighted_mean(grid: GridState, rng: np.random.Generator) -> None:
    field = np.cumsum(rng.normal(size=grid.shape), axis=1) * 10.0
    grid.set("elevation", field)
    before = grid.global_mean("elevation")
    coarse = grid.coarsen(3)
    assert coarse.shape == (30, 60)
    assert coarse.resolution == 6.0
    assert coarse.is_global
    assert coarse.planet.name == "earth"
    assert coarse.global_mean("elevation") == pytest.approx(before, rel=1e-6)


def test_coarsen_category_uses_nearest(grid: GridState) -> None:
    codes = np.arange(grid.size, dtype=np.int64).reshape(grid.shape) % 100
    grid.set("climate_code", codes)
    coarse = grid.coarsen(3)
    assert coarse.as_numpy("climate_code").dtype == np.int8
    assert np.array_equal(coarse.as_numpy("climate_code")[0, 0], codes[0, 0])


def test_coarsen_invalid_factor(grid: GridState) -> None:
    with pytest.raises(ValueError, match="整除"):
        grid.coarsen(7)
    with pytest.raises(ValueError, match="factor"):
        grid.coarsen(0)


def test_resample_preserves_constant_field(grid: GridState) -> None:
    grid.set("T_annual", np.full(grid.shape, 7.5))
    fine = grid.resample(180, 360)
    assert fine.shape == (180, 360)
    assert fine.resolution == 1.0
    assert np.allclose(fine.as_numpy("T_annual"), 7.5, atol=1e-5)
    dense = grid.resample(45, 90)
    assert dense.shape == (45, 90)
    assert np.allclose(dense.as_numpy("T_annual"), 7.5, atol=1e-5)


def test_resample_conserves_area_weighted_mean(grid: GridState) -> None:
    field = np.repeat(np.cos(np.deg2rad(grid.lat))[:, None] * 100.0, grid.nlon, axis=1)
    grid.set("T_annual", field)
    before = grid.global_mean("T_annual")
    assert grid.resample(45, 90).global_mean("T_annual") == pytest.approx(before, rel=1e-3)


def test_copy_is_independent(grid: GridState) -> None:
    grid.set("elevation", np.ones(grid.shape))
    clone = grid.copy()
    grid.set("elevation", np.ones(grid.shape) * 2.0)
    assert clone.as_numpy("elevation").max() == pytest.approx(1.0)
    assert grid.as_numpy("elevation").max() == pytest.approx(2.0)


# ===== 校验与序列化 =====


def test_validate_detects_bad_values(grid: GridState) -> None:
    grid.set("elevation", np.zeros(grid.shape))
    assert grid.validate() == []
    # 绕过 set 直接赋值，validate 应能发现问题
    grid.elevation = backend.asarray(np.full(grid.shape, 1.0e6), backend=grid.backend_name)
    problems = grid.validate()
    assert problems and "elevation" in problems[0]


def test_dict_roundtrip(grid: GridState, rng: np.random.Generator) -> None:
    grid.set("elevation", rng.uniform(-500, 3000, size=grid.shape))
    grid.set("is_ocean", rng.random(grid.shape) < 0.5)
    grid.set("climate_code", rng.integers(0, 30, size=grid.shape))
    restored = GridState.from_dict(grid.to_dict())
    assert restored.shape == grid.shape
    assert restored.seed == grid.seed
    assert restored.planet_params == grid.planet_params
    assert restored.backend == grid.backend
    for name in ("elevation", "is_ocean", "climate_code"):
        np.testing.assert_allclose(restored.as_numpy(name), grid.as_numpy(name))


def test_summary_and_nbytes(grid: GridState) -> None:
    grid.set("elevation", np.zeros(grid.shape, dtype=np.float64))
    summary = grid.summary()
    assert summary["shape"] == (90, 180)
    assert summary["cells"] == 16200
    assert summary["filled_fields"] == 1
    assert summary["planet"] == "earth"
    assert summary["nbytes"] == 90 * 180 * 4  # 统一为 float32
    assert grid.nbytes == 90 * 180 * 4
