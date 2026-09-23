"""网格数据模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import backend
from virtual_world.core.grid import (
    FIELDS,
    FieldSpec,
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


def test_unknown_grid_type_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="grid_type"):
        GridState(nlat=90, nlon=180, resolution=2.0, grid_type="healpix2")
    with pytest.raises(ValueError, match="n_side"):
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
    # 水平维按网格自身的形状展开：经纬 (90, 180)、立方球 (6, 16, 16)、HEALPix (npix,)
    assert FIELDS["elevation"].expected_shape((90, 180)) == (90, 180)
    assert FIELDS["elevation"].expected_shape((6, 16, 16)) == (6, 16, 16)
    assert FIELDS["elevation"].expected_shape((12288,)) == (12288,)
    assert FIELDS["T_soil"].expected_shape((90, 180), nz=6) == (6, 90, 180)
    assert "elevation" in group_fields("terrain")
    assert "T_jan" in group_fields("temperature")
    assert GridState.field_names() == list(FIELDS)


def test_field_spec_rejects_broken_dims() -> None:
    broken = FieldSpec(name="x", unit="-", group="test", dims=("lat",))
    with pytest.raises(ValueError, match="水平维必须成对出现"):
        broken.expected_shape((10, 10))


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


# ===== 三种网格对等（方案《地形生成混合方案》§1.5） =====


@pytest.fixture
def cubed(backend_name: str) -> GridState:
    """立方球网格状态（12 度的等效分辨率，1536 单元）。"""
    return GridState.from_cubed_sphere(16, planet=EARTH, seed=11, backend_name=backend_name)


@pytest.fixture
def healpix(backend_name: str) -> GridState:
    """HEALPix 网格状态（nside=16，3072 单元）。"""
    return GridState.from_healpix(16, planet=EARTH, seed=11, backend_name=backend_name)


def _fill(state: GridState) -> GridState:
    """用解析的 ``sin(lat)`` 场填充状态（三种网格通用）。"""
    lat, _ = state.cell_latlon
    state.set("T_annual", (np.sin(np.deg2rad(lat)) * 30.0).reshape(state.shape))
    state.set("climate_code", np.abs(np.rint(np.sin(np.deg2rad(lat)) * 10)).astype(np.int64).reshape(state.shape))
    return state


@pytest.mark.parametrize(
    "grid_type,shape,size",
    [
        ("cubed_sphere", (6, 16, 16), 1536),
        ("healpix", (3072,), 3072),
    ],
)
def test_non_latlon_geometry(grid_type: str, shape: tuple[int, ...], size: int) -> None:
    state = (
        GridState.from_cubed_sphere(16, planet=EARTH)
        if grid_type == "cubed_sphere"
        else GridState.from_healpix(16, planet=EARTH)
    )
    assert state.grid_type == grid_type
    assert state.shape == shape
    assert state.size == size
    assert state.is_global
    assert state.total_area == pytest.approx(4 * np.pi * EARTH.radius**2, rel=1e-12)
    assert backend.to_numpy(state.cell_area).shape == shape
    lat, lon = state.cell_latlon
    assert lat.shape == (size,) and lon.shape == (size,)
    assert state.validate() == []
    # 经纬网格专属属性对这些网格不再有意义
    assert state.lat is None and state.lat_edges is None and state.cos_lat is None
    with pytest.raises(NotImplementedError, match="切片"):
        state.slice_region(-10, 10, -30, 30)
    with pytest.raises(NotImplementedError, match="resample"):
        state.resample(45, 90)
    with pytest.raises(NotImplementedError, match="cell_index"):
        state.nearest_index(0.0, 0.0)


def test_non_latlon_field_validation(cubed: GridState) -> None:
    cubed.set("T_annual", np.zeros(cubed.shape))
    with pytest.raises(ValueError, match="形状"):
        cubed.set("T_annual", np.zeros((90, 180)))
    with pytest.raises(ValueError, match="值 >"):
        cubed.set("T_annual", np.full(cubed.shape, 1.0e5))
    assert cubed.as_numpy("T_annual").shape == cubed.shape


@pytest.mark.parametrize("grid_type", ["cubed_sphere", "healpix"])
def test_non_latlon_statistics_and_lookup(grid_type: str, backend_name: str) -> None:
    """面积加权统计与最近单元查找（三种网格通用语义）。"""
    state = GridState.from_resolution(5.0, planet=EARTH, grid_type=grid_type, backend_name=backend_name)
    _fill(state)
    assert state.global_mean("T_annual") == pytest.approx(0.0, abs=0.2)  # sin(lat) 的全球面积加权均值 ~ 0
    assert state.global_integral("T_annual") == pytest.approx(
        state.global_mean("T_annual") * state.total_area, rel=1e-9
    )
    profile = state.zonal_mean("T_annual", nbands=12)
    assert profile.shape == (12,)
    assert np.all(np.diff(profile) > 0.0)  # 随纬度单调递增
    # 最近单元查找：格心处的查询应回到自身
    lat, lon = state.cell_latlon
    lat7, lon7 = float(np.asarray(lat)[7]), float(np.asarray(lon)[7])
    index = state.cell_index(lat7, lon7)
    assert 0 <= index < state.size
    assert state.value_at("T_annual", lat7, lon7) == pytest.approx(
        float(np.asarray(state.as_numpy("T_annual")).reshape(-1)[index]), abs=1e-6
    )


@pytest.mark.parametrize("grid_type", ["cubed_sphere", "healpix"])
def test_non_latlon_coarsen_conserves_mean(grid_type: str, backend_name: str) -> None:
    """限制算子：三种网格的粗化都严格守恒全球面积加权平均。"""
    state = _fill(GridState.from_resolution(5.0, planet=EARTH, grid_type=grid_type, backend_name=backend_name))
    coarse = state.coarsen(2)
    assert coarse.grid_type == grid_type
    assert coarse.size == state.size // 4
    assert coarse.global_mean("T_annual") == pytest.approx(state.global_mean("T_annual"), abs=1e-9)
    # 分类字段不产生新类别
    codes_before = set(np.unique(state.as_numpy("climate_code")).tolist())
    codes_after = set(np.unique(coarse.as_numpy("climate_code")).tolist())
    assert codes_after <= codes_before
    if grid_type == "healpix":
        with pytest.raises(ValueError, match="2 的幂"):
            state.coarsen(3)  # HEALPix 只在四叉树层次上对齐
    else:
        # 立方球的块在面内对齐，任意整除因子都合法
        assert state.coarsen(3).size * 9 == state.size


def test_healpix_coarsen_matches_hierarchy(healpix: GridState) -> None:
    """HEALPix 粗化等于四叉树层次聚合：4p..4p+3 的均值。"""
    values = np.arange(healpix.size, dtype=np.float64)
    from virtual_world.core.healpix import HealpixGrid

    grid = HealpixGrid(16)
    manual = values.reshape(-1, 4).mean(axis=1)
    np.testing.assert_allclose(grid.coarsen(values, 2), manual)


@pytest.mark.parametrize(
    "target,kwargs,shape",
    [
        ("latlon", {"nlat": 45, "nlon": 90}, (45, 90)),
        ("cubed_sphere", {"n_side": 16}, (6, 16, 16)),
        ("healpix", {"nside": 32}, (12288,)),
    ],
)
def test_regrid_from_cubed_sphere(
    cubed: GridState, target: str, kwargs: dict, shape: tuple[int, ...]
) -> None:
    """立方球 -> 三种目标网格：形状正确、全球均值守恒（误差在插值阶量级）。"""
    _fill(cubed)
    before = cubed.global_mean("T_annual")
    converted = cubed.regrid(target, **kwargs)
    assert converted.grid_type == target
    assert converted.shape == shape
    assert converted.global_mean("T_annual") == pytest.approx(before, abs=1.0)
    assert set(np.unique(converted.as_numpy("climate_code")).tolist()) <= set(
        np.unique(cubed.as_numpy("climate_code")).tolist()
    )
    assert converted.validate() == []


def test_regrid_roundtrip_cubed_and_healpix(cubed: GridState, healpix: GridState) -> None:
    """立方球 <-> HEALPix 往返（经经纬网格桥接），光滑场信息基本保真。"""
    _fill(cubed)
    healed = cubed.regrid("healpix", nside=64)
    back = healed.regrid("cubed_sphere", n_side=16)
    assert back.shape == cubed.shape
    error = np.abs(back.as_numpy("T_annual") - cubed.as_numpy("T_annual"))
    assert error.max() < 2.0  # 场幅值 30，往返误差应在插值阶量级
    assert back.global_mean("T_annual") == pytest.approx(cubed.global_mean("T_annual"), abs=0.5)
    # 反方向：HEALPix -> 立方球 -> HEALPix
    _fill(healpix)
    roundtrip = healpix.regrid("cubed_sphere", n_side=32).regrid("healpix", nside=16)
    assert roundtrip.shape == healpix.shape
    assert np.abs(roundtrip.as_numpy("T_annual") - healpix.as_numpy("T_annual")).max() < 2.0


def test_regrid_latlon_to_non_latlon(grid: GridState) -> None:
    """经纬 -> 立方球 / HEALPix（含分类字段的最近邻语义）。"""
    _fill(grid)
    for grid_type, kwargs, shape in (
        ("cubed_sphere", {"n_side": 24}, (6, 24, 24)),
        ("healpix", {"nside": 32}, (12288,)),
    ):
        converted = grid.regrid(grid_type, **kwargs)
        assert converted.shape == shape
        assert converted.global_mean("T_annual") == pytest.approx(
            grid.global_mean("T_annual"), abs=1.0
        )
        assert set(np.unique(converted.as_numpy("climate_code")).tolist()) <= set(
            np.unique(grid.as_numpy("climate_code")).tolist()
        )


@pytest.mark.parametrize("grid_type", ["cubed_sphere", "healpix"])
def test_non_latlon_serialization_roundtrip(grid_type: str, backend_name: str) -> None:
    state = _fill(
        GridState.from_resolution(10.0, planet=EARTH, grid_type=grid_type, backend_name=backend_name)
    )
    payload = state.to_dict()
    assert payload["grid"]["grid_type"] == grid_type
    restored = GridState.from_dict(payload)
    assert restored.grid_type == grid_type
    assert restored.shape == state.shape
    assert restored.resolution == pytest.approx(state.resolution)
    np.testing.assert_allclose(
        restored.as_numpy("T_annual"), state.as_numpy("T_annual"), atol=1e-3
    )


@pytest.mark.parametrize("grid_type", ["cubed_sphere", "healpix"])
def test_non_latlon_operators(grid_type: str) -> None:
    """微分算子在非结构网格上可用：z 场梯度、旋转场涡度、laplacian 全球积分 ~ 0。"""
    radius = EARTH.radius
    state = GridState.from_resolution(5.0, planet=EARTH, grid_type=grid_type)
    lat_flat, _ = state.cell_latlon
    z = np.sin(np.deg2rad(lat_flat))
    cos_lat = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    state.set("T_annual", (z * 30.0).reshape(state.shape))
    state.set("u_jan", (200.0 * cos_lat).reshape(state.shape))
    state.set("v_jan", np.zeros(state.shape))

    # 精度细节由 test_operators.py 负责；这里只验证 GridState 的三种网格调度可用
    gx, gy = state.gradient("T_annual")
    gy_flat = np.asarray(gy).reshape(-1)
    assert np.abs(gy_flat / 30.0 * radius - cos_lat).max() < 1e-2
    # 用 u = U cos(lat)（等价于以 U/R 为角速度的刚体旋转）检验涡度与散度
    spin = 200.0 / radius
    vorticity = np.asarray(state.vorticity("u_jan", "v_jan")).reshape(-1)
    assert np.abs(vorticity - 2.0 * spin * z).max() / (2.0 * spin) < 5e-2
    divergence = np.asarray(state.divergence("u_jan", "v_jan")).reshape(-1)
    # 散度靠差分相消得到，相对误差比涡度大一档（收敛性见 test_operators.py）
    assert np.abs(divergence).max() / spin < 0.1
    # 拉普拉斯是二阶量：float32 存储下噪声被 1/h^2 放大，这里只验证量级与全球积分为 0
    laplacian = np.asarray(state.laplacian("T_annual")).reshape(-1)
    expected = -2.0 * z * 30.0 / radius**2
    assert np.abs(laplacian - expected).max() < 0.35 * np.abs(expected).max()
    integral = abs(float(np.sum(laplacian * state._area_flat()))) / state.total_area
    assert integral < 0.05 * np.abs(expected).max()


def test_from_resolution_derives_non_latlon_sizes() -> None:
    """按分辨率构造非经纬网格：立方球取 90/res，HEALPix 取最近的 2 的幂。"""
    cubed = GridState.from_resolution(5.0, grid_type="cubed_sphere")
    assert cubed.n_side == 18 and cubed.shape == (6, 18, 18)
    healpix = GridState.from_resolution(5.0, grid_type="healpix")
    assert healpix.nside == 16 and healpix.shape == (3072,)
    from virtual_world.core.healpix import HealpixGrid

    assert abs(healpix.resolution - HealpixGrid(16).resolution) < 1e-9
    # 分级分辨率按"最接近的 2 的幂"换算（5° 对应 nside=11.7 → 16）
    level = GridState.from_level("coarse", grid_type="healpix")
    assert level.nside == 16
    with pytest.raises(ValueError, match="网格类型"):
        GridState.from_resolution(5.0, grid_type="nope")
