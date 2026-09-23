"""球面几何工具测试。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import backend, spherical
from virtual_world.core import constants as const


def test_lat_lon_centers_and_edges() -> None:
    lat = spherical.lat_centers(180)
    lon = spherical.lon_centers(360)
    assert lat[0] == pytest.approx(-89.5)
    assert lat[-1] == pytest.approx(89.5)
    assert lon[0] == pytest.approx(-179.5)
    assert lon[-1] == pytest.approx(179.5)
    assert spherical.lat_edges(180)[0] == pytest.approx(-90.0)
    assert spherical.lat_edges(180)[-1] == pytest.approx(90.0)
    assert np.allclose(spherical.cos_lat_weights(np.array([0.0, 60.0])), [1.0, 0.5])


def test_cell_areas_sum_to_sphere() -> None:
    radius = const.EARTH_RADIUS
    edges = spherical.lat_edges(90)
    areas = spherical.cell_areas(edges, 180, radius)
    assert spherical.total_area(edges, radius) == pytest.approx(4 * np.pi * radius**2, rel=1e-12)
    # 每个环带内的单元面积相等，且赤道单元最大
    assert np.allclose(areas[44], areas[45])
    assert areas[45, 0] > areas[0, 0]


def test_global_mean_and_integral(backend_name: str) -> None:
    radius = const.EARTH_RADIUS
    nlat, nlon = 18, 36
    edges = spherical.lat_edges(nlat)
    areas = spherical.cell_areas(edges, nlon, radius)
    const_field = backend.asarray(np.full((nlat, nlon), 7.0), backend=backend_name)
    assert spherical.global_mean(const_field, areas) == pytest.approx(7.0, rel=1e-6)
    assert spherical.global_integral(const_field, areas) == pytest.approx(
        7.0 * 4 * np.pi * radius**2, rel=1e-6
    )


def test_coriolis_and_beta() -> None:
    omega = const.EARTH_ROTATION_RATE
    assert spherical.coriolis_parameter(omega, 90.0) == pytest.approx(2 * omega, rel=1e-9)
    assert spherical.coriolis_parameter(omega, -30.0) == pytest.approx(-omega, rel=1e-9)
    assert spherical.beta_parameter(omega, 0.0, const.EARTH_RADIUS) == pytest.approx(
        2 * omega / const.EARTH_RADIUS, rel=1e-12
    )


def test_meridional_gradient_of_linear_field(backend_name: str) -> None:
    radius = const.EARTH_RADIUS
    nlat, nlon = 90, 180
    lat = spherical.lat_centers(nlat)
    # 取 z = R*phi（单位 m），则 dz/dy = 1
    field = backend.asarray(
        np.repeat((radius * np.deg2rad(lat))[:, None], nlon, axis=1), backend=backend_name
    )
    grad = backend.to_numpy(spherical.meridional_gradient(field, radius))
    assert np.allclose(grad[1:-1, :], 1.0, rtol=1e-5)


def test_vorticity_of_solid_body_rotation(backend_name: str) -> None:
    omega = const.EARTH_ROTATION_RATE
    radius = const.EARTH_RADIUS
    nlat, nlon = 90, 180
    lat = spherical.lat_centers(nlat)
    cos_phi = spherical.cos_lat_weights(lat)
    u = np.repeat((omega * radius * cos_phi)[:, None], nlon, axis=1)
    v = np.zeros_like(u)
    u_b = backend.asarray(u, backend=backend_name)
    v_b = backend.asarray(v, backend=backend_name)

    zeta = backend.to_numpy(spherical.vorticity(u_b, v_b, lat, radius))
    div = backend.to_numpy(spherical.divergence(u_b, v_b, lat, radius))
    expected = 2 * omega * np.sin(np.deg2rad(lat))

    mid = slice(2, -2)
    assert np.allclose(zeta[mid, :], expected[mid, None], rtol=1e-3)
    assert np.allclose(div, 0.0, atol=1e-15)


def test_zonal_gradient(backend_name: str) -> None:
    radius = const.EARTH_RADIUS
    nlat, nlon = 90, 180
    lat = spherical.lat_centers(nlat)
    lon = spherical.lon_centers(nlon)
    lam = np.deg2rad(lon)
    field = np.cos(lam)[None, :] + 0.0 * lat[:, None]
    grad = backend.to_numpy(
        spherical.zonal_gradient(backend.asarray(field, backend=backend_name), lat, radius)
    )
    # d(cos lam)/dx = -sin(lam) / (R cos phi)
    expected = -np.sin(lam)[None, :] / (radius * spherical.cos_lat_weights(lat)[:, None])
    assert np.allclose(grad[40:50], expected[40:50], rtol=1e-3)


def test_distance_and_coordinate_transform() -> None:
    radius = const.EARTH_RADIUS
    assert spherical.haversine_distance(0.0, 0.0, 0.0, 90.0, radius) == pytest.approx(
        np.pi / 2 * radius
    )
    assert spherical.haversine_distance(0.0, 0.0, 90.0, 123.0, radius) == pytest.approx(
        np.pi / 2 * radius
    )

    xyz = spherical.latlon_to_xyz(np.array([0.0, 45.0]), np.array([0.0, 90.0]), radius)
    assert xyz[0] == pytest.approx([radius, 0.0, 0.0])
    lat, lon = spherical.xyz_to_latlon(xyz, radius)
    assert np.allclose(lat, [0.0, 45.0])
    assert np.allclose(lon, [0.0, 90.0])


def test_latitude_of_max() -> None:
    lat = spherical.lat_centers(90)
    profile = np.exp(-((lat / 20.0) ** 2))
    assert spherical.latitude_of_max(profile, lat) == pytest.approx(-1.0, abs=1.0)


def test_cell_spacing_metric() -> None:
    """纬向格距恒定，经向格距随 cos(lat) 向极点收缩。"""
    radius = const.EARTH_RADIUS
    nlat, nlon = 18, 36
    lat = spherical.lat_centers(nlat)
    dlat_m, dlon_m = spherical.cell_spacing(lat, nlon, radius)
    assert dlat_m == pytest.approx(np.deg2rad(180.0 / nlat) * radius)
    assert dlon_m.shape == (nlat,)
    assert np.allclose(dlon_m, dlat_m * np.cos(np.deg2rad(lat)))
    # 经向格距在赤道最大、向两极单调收缩
    assert np.all(dlon_m <= dlat_m + 1e-9)
    assert int(np.argmax(dlon_m)) in (nlat // 2 - 1, nlat // 2)


def test_cell_spacing_matches_cell_area() -> None:
    """纬向格距 × 经向格距 ≈ 单元面积（小角度近似，相对误差 O(dlat²)）。"""
    radius = const.EARTH_RADIUS
    nlat, nlon = 180, 360
    lat = spherical.lat_centers(nlat)
    dlat_m, dlon_m = spherical.cell_spacing(lat, nlon, radius)
    areas = spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)[:, 0]
    assert np.allclose(dlat_m * dlon_m, areas, rtol=1e-4)
