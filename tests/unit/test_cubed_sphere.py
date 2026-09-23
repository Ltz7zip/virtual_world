"""立方球网格测试（方案《地形生成混合方案》§1.5）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import spherical
from virtual_world.core.constants import EARTH_RADIUS
from virtual_world.core.cubed_sphere import DIRECTIONS, CubedSphere


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        CubedSphere(0)
    with pytest.raises(ValueError):
        CubedSphere(8, radius=0.0)


def test_shape_and_size() -> None:
    cs = CubedSphere(12)
    assert cs.shape == (6, 12, 12)
    assert cs.size == 6 * 12 * 12
    assert cs.total_area == pytest.approx(4.0 * np.pi * EARTH_RADIUS**2)


def test_areas_tile_the_sphere_exactly() -> None:
    """6 面单元面积之和严格等于球面积（球面三角形超量精确）。"""
    for n_side in (4, 9, 16):
        cs = CubedSphere(n_side)
        total = float(cs.areas().sum())
        assert total == pytest.approx(cs.total_area, rel=1e-12)


def test_area_uniformity_is_far_better_than_latlon() -> None:
    """方案 §1.5：立方球单元面积接近均匀，远优于等经纬网格。"""
    cs = CubedSphere(45)  # 6*45^2 = 12150 单元，与 90x180 的 16200 同量级
    latlon_areas = spherical.cell_areas(spherical.lat_edges(90), 180, EARTH_RADIUS)
    latlon_ratio = float(latlon_areas.max() / latlon_areas.min())
    assert cs.area_ratio() < 1.6
    assert latlon_ratio > 20.0
    assert cs.area_ratio() < latlon_ratio


def test_equiangular_projection_is_more_uniform_than_gnomonic() -> None:
    """等角投影（方案 §1.5 的推荐）面积比显著优于线性 gnomonic。"""
    equi = CubedSphere(16, equiangular=True).area_ratio()
    linear = CubedSphere(16, equiangular=False).area_ratio()
    assert equi < linear


def test_centers_are_unit_vectors_and_pole_free() -> None:
    """单元中心为单位向量；立方球无极点单元（极点落在立方体角点上）。"""
    cs = CubedSphere(10)
    xyz = cs.centers_xyz()
    assert xyz.shape == (6, 10, 10, 3)
    assert np.allclose(np.linalg.norm(xyz, axis=-1), 1.0, atol=1e-12)
    lat, lon = cs.centers_latlon()
    assert lat.shape == (6, 10, 10)
    assert np.abs(lat).max() < 90.0
    assert np.abs(lon).max() <= 180.0


def test_no_degenerate_cells_near_poles() -> None:
    """极区单元面积不塌缩——这正是等经纬网格的硬伤（方案 §1.5）。"""
    cs = CubedSphere(32)
    areas = cs.areas()
    lat, _ = cs.centers_latlon()
    polar = np.abs(lat) > 75.0
    assert polar.any()
    assert areas[polar].min() > 0.5 * areas.mean()
    latlon_areas = spherical.cell_areas(spherical.lat_edges(90), 180, EARTH_RADIUS)
    latlon_polar = np.abs(spherical.lat_centers(90)) > 75.0
    assert latlon_areas[latlon_polar].min() < 0.2 * latlon_areas.mean()


def test_spacing_is_sqrt_area() -> None:
    cs = CubedSphere(8)
    assert np.allclose(cs.spacing(), np.sqrt(cs.areas()))


def test_neighbours_are_distinct_and_symmetric() -> None:
    """每个单元 4 个邻居互异，且邻接关系对称（跨面接缝自洽）。"""
    cs = CubedSphere(12)
    nbr = cs.neighbors()
    assert nbr.shape == (4, cs.size)
    assert len(DIRECTIONS) == 4
    identity = np.arange(cs.size)
    for cell in (0, 5, cs.size // 2, cs.size - 1):
        assert len(set(nbr[d][cell] for d in range(4))) == 4
    for d in range(4):
        target = nbr[d]
        back = np.stack([nbr[e][target] for e in range(4)], axis=1)
        assert (back == identity[:, None]).any(axis=1).all()


def test_shift_matches_neighbour_gather() -> None:
    cs = CubedSphere(6)
    field = np.arange(cs.size, dtype=np.float64).reshape(cs.shape)
    for d, (di, dj) in enumerate(DIRECTIONS):
        shifted = cs.shift(field, di, dj)
        assert np.array_equal(shifted.ravel(), field.ravel()[cs.neighbors()[d]])


def test_shift_works_with_leading_dimensions() -> None:
    cs = CubedSphere(5)
    field = np.stack([np.arange(cs.size).reshape(cs.shape), np.ones(cs.shape) * 7.0], axis=0)
    shifted = cs.shift(field, 0, 1)
    assert shifted.shape == (2, *cs.shape)
    assert np.array_equal(shifted[1], np.full(cs.shape, 7.0))


def test_shift_validates_field_shape() -> None:
    cs = CubedSphere(4)
    with pytest.raises(ValueError):
        cs.shift(np.zeros((6, 5, 5)), 0, 1)


def test_to_latlon_accuracy_on_smooth_field() -> None:
    """光滑场（z 分量 = sin(lat)）的重网格误差应在插值阶量级。"""
    cs = CubedSphere(16)
    z = cs.centers_xyz()[..., 2]
    nlat, nlon = 40, 80
    out = cs.to_latlon(z, nlat, nlon)
    lat = -90.0 + (180.0 / nlat) * (np.arange(nlat) + 0.5)
    expected = np.sin(np.deg2rad(lat))[:, None] * np.ones((1, nlon))
    assert out.shape == (nlat, nlon)
    assert np.abs(out - expected).max() < 1.0e-2


def test_regrid_round_trip_and_leading_dimensions() -> None:
    cs = CubedSphere(16)
    z = cs.centers_xyz()[..., 2]
    batch = np.stack([z, np.full(cs.shape, 3.0)], axis=0)
    latlon = cs.to_latlon(batch, 40, 80)
    assert latlon.shape == (2, 40, 80)
    back = cs.from_latlon(latlon)
    assert back.shape == (2, *cs.shape)
    assert np.abs(back[1] - 3.0).max() < 1.0e-12
    assert np.abs(back[0] - z).max() < 5.0e-3


def test_from_latlon_longitude_is_periodic() -> None:
    """0/360 度接缝处不得出现不连续：常数经度场重网格后仍为常数。"""
    cs = CubedSphere(8)
    constant = np.full((18, 36), 2.5)
    assert np.allclose(cs.from_latlon(constant), 2.5)
    latlon = cs.to_latlon(np.full(cs.shape, 2.5), 18, 36)
    assert np.allclose(latlon, 2.5)


def test_from_latlon_validates_dimensions() -> None:
    cs = CubedSphere(4)
    with pytest.raises(ValueError):
        cs.from_latlon(np.zeros(5))


def test_geometry_is_deterministic() -> None:
    a = CubedSphere(7)
    b = CubedSphere(7)
    assert np.array_equal(a.centers_xyz(), b.centers_xyz())
    assert np.array_equal(a.areas(), b.areas())
    assert np.array_equal(a.neighbors(), b.neighbors())


def test_neighbor_index_rejects_multi_cell_offsets() -> None:
    cs = CubedSphere(4)
    with pytest.raises(ValueError):
        cs.neighbor_index(2, 0)
