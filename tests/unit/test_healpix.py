"""HEALPix 网格测试（:mod:`virtual_world.core.healpix`）。

几何/拓扑与参考实现 ``astropy_healpix`` 逐像素交叉验证；多分辨率算子、重网格
与切平面算子用解析场（``z = sin(lat)``、刚体旋转）检验。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from virtual_world.core import spherical
from virtual_world.core.constants import EARTH_RADIUS
from virtual_world.core.healpix import HealpixGrid

_MISSING = -1  # 参考实现中的"无邻居"哨兵
_R = EARTH_RADIUS
_OMEGA = 7.292e-5  # 刚体旋转角速度 (1/s)
_CORNER_PIXELS = 24  # 基础像素角点像素数：这些像素只有 7 个邻居


@lru_cache(maxsize=8)
def _grid(nside: int) -> HealpixGrid:
    """按 ``nside`` 缓存的网格（拓扑表构造较贵，测试内复用）。"""
    return HealpixGrid(nside)


@lru_cache(maxsize=8)
def _lat_deg(nside: int) -> np.ndarray:
    """单元中心纬度 (deg)，形状 ``(npix,)``。"""
    return _grid(nside).centers_latlon()[0]


@lru_cache(maxsize=8)
def _corner_mask(nside: int) -> np.ndarray:
    """角点像素掩码：邻居数不足 8 的像素。"""
    return ~_grid(nside).neighbor_mask().all(axis=0)


def _reference(nside: int):
    """参考实现 ``astropy_healpix.HEALPix``（未安装时跳过相关测试）。"""
    healpix = pytest.importorskip("astropy_healpix")
    return healpix.HEALPix(nside=nside, order="nested")


def _rotation(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """刚体旋转的东/北分量：``u = Omega*R*cos(lat)``、``v = 0``。"""
    u = _OMEGA * _R * np.cos(np.deg2rad(_lat_deg(nside)))
    return u, np.zeros_like(u)


def test_construction_and_nside_one_limits() -> None:
    """``nside`` 必须是 2 的幂、``radius`` 为正；``nside=1`` 无邻居表与算子。"""
    for bad in (0, 3, 6):
        with pytest.raises(ValueError):
            HealpixGrid(bad)
    with pytest.raises(ValueError):
        HealpixGrid(4, radius=0.0)
    grid = _grid(1)
    assert grid.npix == 12
    assert np.array_equal(
        grid.interpolator().nearest_index(grid.centers_xyz()), np.arange(grid.npix)
    )
    for method in (grid.edge_neighbors, grid.all_neighbors, grid.stencil):
        with pytest.raises(ValueError):
            method()


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16])
def test_geometry_matches_reference_implementation(nside: int) -> None:
    """等面积性精确（面积比恒为 1）；单元中心与 ``astropy_healpix`` 一致到 1e-10 度。"""
    grid = _grid(nside)
    areas = grid.areas()
    assert grid.npix == 12 * nside**2
    assert grid.size == grid.npix and grid.shape == (grid.npix,)
    assert grid.n_level == int(np.log2(nside))
    assert np.all(areas == grid.total_area / grid.npix)
    assert areas.sum() == pytest.approx(grid.total_area, rel=1e-15)
    assert grid.area_ratio() == 1.0
    assert np.array_equal(grid.spacing(), np.sqrt(areas))
    expected = np.degrees(np.sqrt(grid.cell_area()) / _R)
    assert grid.resolution == pytest.approx(expected, rel=1e-15) and grid.resolution > 0.0
    lon, lat = _reference(nside).healpix_to_lonlat(np.arange(grid.npix))
    lat_mine, lon_mine = grid.centers_latlon()
    assert np.abs(lat_mine - np.degrees(np.asarray(lat))).max() < 1.0e-10
    wrapped = (lon_mine - np.degrees(np.asarray(lon)) + 180.0) % 360.0 - 180.0
    assert np.abs(wrapped).max() < 1.0e-10
    xyz = grid.centers_xyz()
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-15)
    assert np.allclose(xyz[:, 2], np.sin(np.deg2rad(lat_mine)), atol=1e-15)


@pytest.mark.parametrize("nside", [2, 8, 32])
def test_neighbors_match_reference_and_edge_properties(nside: int) -> None:
    """8 邻居**集合**与参考实现一致（差异数写进断言消息）；共边邻居规则对称。"""
    grid = _grid(nside)
    edge, table, mask = grid.edge_neighbors(), grid.all_neighbors(), grid.neighbor_mask()
    pixel = np.arange(grid.npix)
    ref_table = np.asarray(_reference(nside).neighbours(pixel))
    if ref_table.shape != (grid.npix, 8):
        ref_table = ref_table.T
    mine = [set(int(v) for v in table[:, p][mask[:, p]]) for p in range(grid.npix)]
    theirs = [set(int(v) for v in ref_table[p] if int(v) != _MISSING) for p in range(grid.npix)]
    mismatched = [p for p in range(grid.npix) if mine[p] != theirs[p]]
    assert not mismatched, f"nside={nside} 有 {len(mismatched)} 个像素邻居集合不一致"
    assert int((~mask).sum()) == _CORNER_PIXELS
    assert {len(item) for item in mine} == {7, 8}
    assert edge.shape == (4, grid.npix) and np.array_equal(grid.neighbors(), edge)
    assert not (edge == pixel).any()
    for k in range(4):
        assert np.isin(pixel, edge[:, edge[k]]).all()  # 共边关系对称
    for p in range(grid.npix):
        assert set(int(v) for v in edge[:, p]) <= set(int(v) for v in table[:, p][mask[:, p]])
    for k in range(8):
        source = np.where(mask[k])[0]
        assert np.isin(pixel[source], table[:, table[k][source]][mask[:, table[k][source]]]).all()


def test_shift_gathers_neighbor_direction() -> None:
    """``shift`` 按方向表取邻居；方向越界、字段长度不符、缺邻居方向都报错。"""
    grid = _grid(8)
    field = np.arange(grid.npix, dtype=np.float64)
    for direction in range(4):
        assert np.array_equal(grid.shift(field, direction), field[grid.edge_neighbors()[direction]])
    batch = np.stack([field, np.full(grid.npix, 2.5)], axis=0)
    assert np.array_equal(grid.shift(batch, 1)[1], np.full(grid.npix, 2.5))
    with pytest.raises(ValueError):
        grid.shift(field, 4)
    with pytest.raises(ValueError):
        grid.shift(field[:-1], 0)
    # 角点像素缺一个共角邻居，该方向无法整体平移（缺失槽位排在最后）
    with pytest.raises(ValueError):
        grid.shift(field, 7, diagonal=True)


def test_hierarchy_coarsening_and_upsampling() -> None:
    """子像素为 ``4p..4p+3``；粗化严格守恒平均且可复合；延长算子保守/保常数。"""
    rng = np.random.default_rng(20240923)
    grid, coarse_grid = _grid(8), _grid(4)
    pixel = np.arange(grid.npix)
    children = HealpixGrid.children(pixel)
    assert children.shape == (grid.npix, 4)
    assert np.array_equal(children, np.stack([4 * pixel + k for k in range(4)], axis=-1))
    assert (HealpixGrid.parent(children) == pixel[:, None]).all()
    assert np.array_equal(
        HealpixGrid.children(np.array([3, 7])), np.array([[12, 13, 14, 15], [28, 29, 30, 31]])
    )
    field = rng.normal(size=grid.npix)
    coarse = grid.coarsen(field, 2)
    assert np.array_equal(coarse, field.reshape(-1, 4).mean(axis=1))
    assert float(coarse.mean()) == pytest.approx(float(field.mean()), abs=1e-12)
    assert np.allclose(grid.coarsen(field, 4), coarse_grid.coarsen(coarse, 2), atol=1e-14)
    for bad in (1, 3):
        with pytest.raises(ValueError):
            grid.coarsen(field, bad)
    with pytest.raises(ValueError):
        grid.coarsen(field[:-1], 2)
    block = np.repeat(rng.normal(size=grid.npix // 4), 4)
    assert np.array_equal(grid.upsample(grid.coarsen(block, 2), 2, smooth=False), block)
    smooth = grid.upsample(np.full(coarse_grid.npix, 3.5), 2, smooth=True)
    assert np.abs(smooth - 3.5).max() < 1.0e-10
    assert abs(float(smooth.mean()) - 3.5) < 1.0e-10
    with pytest.raises(ValueError):
        grid.upsample(np.zeros(grid.npix), 2)
    with pytest.raises(ValueError):
        grid.upsample(np.full(coarse_grid.npix, 1.0), 3)


def test_regridding_round_trip_and_nearest_semantics() -> None:
    """常数场往返仍是常数；光滑场往返误差 < 0.05；最近邻不产生新类别值。"""
    grid = _grid(16)
    nlat, nlon = 18, 36
    assert np.abs(grid.from_latlon(np.full((nlat, nlon), 2.5)) - 2.5).max() < 1.0e-12
    assert np.abs(grid.to_latlon(np.full(grid.npix, 2.5), nlat, nlon) - 2.5).max() < 1.0e-9
    lat_2d, lon_2d = np.meshgrid(
        spherical.lat_centers(nlat), spherical.lon_centers(nlon), indexing="ij"
    )
    field = np.cos(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(lon_2d))
    back = grid.to_latlon(grid.from_latlon(field), nlat, nlon)
    assert back.shape == (nlat, nlon)
    assert np.abs(back - field).max() < 0.05
    categories = np.arange(grid.npix) % 5
    coarsened = grid.to_latlon(categories, nlat, nlon, order="nearest")
    assert set(np.unique(coarsened).tolist()) <= set(np.unique(categories).tolist())
    cells = grid.latlon_to_cell(nlat, nlon)
    assert cells.shape == (nlat, nlon) and cells.min() >= 0 and cells.max() < grid.npix
    with pytest.raises(ValueError):
        grid.to_latlon(categories, nlat, nlon, order="cubic")
    lat, lon = grid.centers_latlon()
    index = np.arange(0, grid.npix, 37)
    assert np.array_equal(grid.pixel_of_latlon(lat[index], lon[index]), index)


def test_global_and_zonal_statistics() -> None:
    """全球平均/积分自洽；``zonal_mean`` 与直接条带平均一致且随纬度单调增。"""
    grid = _grid(32)
    z = grid.centers_xyz()[:, 2]
    lat = _lat_deg(32)
    assert grid.global_mean(np.full(grid.npix, 2.0)) == pytest.approx(2.0, abs=1e-15)
    assert grid.global_integral(z) == pytest.approx(grid.global_mean(z) * grid.total_area)
    profile = grid.zonal_mean(z, nbands=18)
    assert profile.shape == (18,) and np.all(np.diff(profile) > 0.0)
    edges = np.linspace(-90.0, 90.0, 19)
    for band in (0, 9, 17):
        band_sel = (lat >= edges[band]) & (lat < edges[band + 1])
        assert profile[band] == pytest.approx(float(z[band_sel].mean()), abs=1e-9)
    assert np.abs(grid.zonal_mean(np.full(grid.npix, 4.25), nbands=18) - 4.25).max() < 1e-12
    assert grid.zonal_mean(z).shape == (int(round(180.0 / grid.resolution)),)


def test_gradient_of_z_matches_analytic_value() -> None:
    """nside=32：``z`` 的北向梯度为 ``cos(lat)/R``，合成 RMS 误差 < 2e-3。"""
    grid = _grid(32)
    gx, gy = grid.gradient(grid.centers_xyz()[:, 2])
    cos_lat = np.cos(np.deg2rad(_lat_deg(32)))
    assert np.abs(gy * _R - cos_lat).max() < 6.0e-3
    assert np.sqrt(np.mean((gy * _R - cos_lat) ** 2 + (gx * _R) ** 2)) < 2.0e-3


def test_vorticity_divergence_and_finiteness() -> None:
    """刚体旋转涡度误差 < 2%；常数场近似无辐散；全场算子输出有限。

    常数矢量场的散度误差集中在极点相邻像素且不随加密下降（nside=64 时约 1.7e-6），
    故全场阈值按解析量级 ``1/R`` 放宽，并额外约束极区外的严格界。
    """
    grid = _grid(32)
    u, v = _rotation(32)
    z = grid.centers_xyz()[:, 2]
    assert np.abs(grid.vorticity(u, v) - 2.0 * _OMEGA * z).max() / (2.0 * _OMEGA) < 2.0e-2
    divergence = grid.divergence(np.full(grid.npix, 1.0), np.zeros(grid.npix))
    assert np.abs(divergence).max() < 1.0e-6
    assert np.abs(divergence[np.abs(_lat_deg(32)) < 80.0]).max() < 1.0e-7
    results = [*grid.gradient(z), grid.vorticity(u, v), divergence, grid.laplacian(z)]
    for result in results:
        assert np.isfinite(result).all() and result.shape == (grid.npix,)
    assert int(_corner_mask(32).sum()) == _CORNER_PIXELS


def test_laplacian_of_z_and_weighted_mean() -> None:
    """``laplacian(z) = -2z/R^2``；角点像素误差不收敛，故全场界略宽。

    nside=16/32/64 的 24 个角点像素误差稳定在 0.35~0.38（不随加密减小），其余像素
    收敛（nside=32 时最大 0.14），因此同时给出全场界（0.4）与角点外的强约束（0.15）。
    """
    grid = _grid(32)
    z = grid.centers_xyz()[:, 2]
    error = np.abs(grid.laplacian(z) + 2.0 * z / _R**2) * _R**2
    assert error.max() < 0.4
    assert error[~_corner_mask(32)].max() < 0.15
    weights = grid.areas()
    assert abs(float(np.sum(grid.laplacian(z) * weights) / weights.sum())) * _R**2 < 1.0e-3


def test_interpolator_identity_constant_and_determinism() -> None:
    """最近邻索引恒等、常数场插值回常数；重复调用结果逐位相同。"""
    grid = _grid(16)
    interpolator = grid.interpolator()
    assert np.array_equal(interpolator.nearest_index(grid.centers_xyz()), np.arange(grid.npix))
    points = np.random.default_rng(20240923).normal(size=(64, 3))
    query = points / np.linalg.norm(points, axis=1, keepdims=True)
    assert np.abs(interpolator(np.full(grid.npix, 1.75), query) - 1.75).max() < 1.0e-10
    z = grid.centers_xyz()[:, 2]
    assert np.array_equal(grid.laplacian(z), grid.laplacian(z))
    assert np.array_equal(grid.all_neighbors(), _grid(16).all_neighbors())


def test_neighbor_caches_are_read_only() -> None:
    """内部邻居缓存数组不可就地改写（防止调用方污染缓存）。"""
    grid = _grid(8)
    for cached in (*grid.edge_neighbors(), *grid.all_neighbors(), *grid.neighbor_mask()):
        assert cached.flags.writeable is False
    assert grid.face_xy()[0].flags.writeable is False
    with pytest.raises(ValueError):
        grid.edge_neighbors()[0, 0] = 1