"""球面域扭曲 Voronoi 板块划分测试（《第一层完善》§1）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import voronoi


def test_fibonacci_sphere_radius_and_count() -> None:
    seeds = voronoi.fibonacci_sphere(300)
    assert seeds.shape == (300, 3)
    norms = np.linalg.norm(seeds, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_fibonacci_sphere_covers_both_hemispheres() -> None:
    seeds = voronoi.fibonacci_sphere(200)
    z = seeds[:, 2]
    # 南/北极都有种子点，且没有全部挤在赤道
    assert z.min() < -0.99 and z.max() > 0.99
    assert np.mean(z) == pytest.approx(0.0, abs=0.1)


def test_fibonacci_sphere_no_clustered_seeds() -> None:
    """任意两个种子点的角距离应大于阈值（N=200 时约 9 度，取 5 度判据）。"""
    seeds = voronoi.fibonacci_sphere(200)
    dist = np.arccos(np.clip(seeds @ seeds.T, -1.0, 1.0))
    np.fill_diagonal(dist, np.inf)
    assert dist.min() > np.deg2rad(5.0)


def test_fibonacci_sphere_seed_count_validation() -> None:
    with pytest.raises(ValueError):
        voronoi.fibonacci_sphere(0)
    with pytest.raises(ValueError):
        voronoi.fibonacci_sphere(-3)


def test_assign_plates_no_warp_is_nearest_seed() -> None:
    """warp_amp=0 时，隶属应等于笛卡尔坐标最近邻（经纬循环隐含正确）。"""
    seeds = voronoi.fibonacci_sphere(60)
    lat = np.array([-89.5, 0.0, 45.0, 89.5])
    lon = np.array([-179.5, 0.0, 90.0, 179.5])
    plate = voronoi.assign_plates(lat, lon, seeds, seed=1)
    assert plate.shape == (4, 4)
    assert plate.dtype.kind == "i"
    # 最近邻（点积最大即为角距离最近）
    latg, long = np.meshgrid(lat, lon, indexing="ij")
    pts = np.stack(
        [
            np.cos(np.deg2rad(latg)) * np.cos(np.deg2rad(long)),
            np.cos(np.deg2rad(latg)) * np.sin(np.deg2rad(long)),
            np.sin(np.deg2rad(latg)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    expected = seeds @ pts.T
    assert np.all(plate.ravel() == expected.argmax(axis=0))


def test_assign_plates_deterministic() -> None:
    seeds = voronoi.fibonacci_sphere(120)
    lat = np.linspace(-80, 80, 30)
    lon = np.linspace(-175, 175, 60)
    a = voronoi.assign_plates(lat, lon, seeds, seed=7, warp_amp=0.15)
    b = voronoi.assign_plates(lat, lon, seeds, seed=7, warp_amp=0.15)
    assert np.array_equal(a, b)


def test_assign_plates_warp_changes_boundaries() -> None:
    """域扭曲应使部分单元归属改变，且仍只产生 0..n-1 的微板块。"""
    seeds = voronoi.fibonacci_sphere(120)
    lat = np.linspace(-85, 85, 36)
    lon = np.linspace(-175, 175, 72)
    plain = voronoi.assign_plates(lat, lon, seeds, seed=3, warp_amp=0.0)
    warped = voronoi.assign_plates(lat, lon, seeds, seed=3, warp_amp=0.2)
    assert not np.array_equal(plain, warped)
    assert warped.min() >= 0 and warped.max() <= 119
    seen = set(np.unique(warped))
    assert len(seen) > 90  # 大参数下绝大多数种子点仍覆盖到单元


def test_assign_plates_point_cloud_growth_multiplier() -> None:
    """小种子集 + 域扭曲：所有网格单元都必须归属到某个种子。"""
    seeds = voronoi.fibonacci_sphere(30)
    lat = np.linspace(-85, 85, 18)
    lon = np.linspace(-175, 175, 36)
    plate = voronoi.assign_plates(lat, lon, seeds, seed=42, warp_amp=0.3)
    assert plate.shape == (18, 36)
    assert np.all(plate >= 0)


# ===== 微板块归并 =====


def _micro_plate_field(n_seeds: int = 200, nlat: int = 90, nlon: int = 180) -> np.ndarray:
    """构造微板块归属场的确定性辅助函数。"""
    seeds = voronoi.fibonacci_sphere(n_seeds)
    lat = np.linspace(-89, 89, nlat)
    lon = np.linspace(-179.5, 179.5, nlon)
    return voronoi.assign_plates(lat, lon, seeds, seed=11, warp_amp=0.12)


def _count_connected_components(labels: np.ndarray, ncomp: int) -> list[int]:
    """4-邻（经度循环）连通分量计数，返回每个标签的分量数。"""
    nlat, nlon = labels.shape
    seen = np.zeros_like(labels, dtype=bool)
    counts = [0] * ncomp
    from collections import deque

    for i0 in range(nlat):
        for j0 in range(nlon):
            lab = labels[i0, j0]
            if seen[i0, j0] or lab < 0:
                continue
            counts[lab] += 1
            seen[i0, j0] = True
            queue = deque([(i0, j0)])
            while queue:
                i, j = queue.popleft()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, (j + dj) % nlon
                    if 0 <= ni < nlat and not seen[ni, nj] and labels[ni, nj] == lab:
                        seen[ni, nj] = True
                        queue.append((ni, nj))
    return counts


def test_merge_micro_plates_reduces_to_n_major() -> None:
    micro = _micro_plate_field()
    merged = voronoi.merge_micro_plates(micro, n_major=7, seed=5)
    assert merged.shape == micro.shape
    unique = np.unique(merged)
    assert unique.min() == 0
    assert len(unique) == 7
    assert unique.max() == 6


def test_merge_micro_plates_all_plates_connected() -> None:
    micro = _micro_plate_field()
    merged = voronoi.merge_micro_plates(micro, n_major=7, seed=5)
    counts = _count_connected_components(merged, 7)
    assert all(c == 1 for c in counts), f"存在不连通板块: {counts}"


def test_merge_micro_plates_deterministic() -> None:
    micro = _micro_plate_field()
    a = voronoi.merge_micro_plates(micro, n_major=8, seed=3)
    b = voronoi.merge_micro_plates(micro, n_major=8, seed=3)
    assert np.array_equal(a, b)


def test_merge_micro_plates_keeps_distinct_selections() -> None:
    """不同 seed 应产生不同的归并结果（核心选点随机化有效）。"""
    micro = _micro_plate_field()
    a = voronoi.merge_micro_plates(micro, n_major=7, seed=5)
    b = voronoi.merge_micro_plates(micro, n_major=7, seed=99)
    assert not np.array_equal(a, b)


def test_merge_micro_plates_n_major_validation() -> None:
    micro = _micro_plate_field()
    with pytest.raises(ValueError):
        voronoi.merge_micro_plates(micro, n_major=0)
    with pytest.raises(ValueError):
        voronoi.merge_micro_plates(micro, n_major=500)


def test_merge_micro_plates_at_least_one_large_plate() -> None:
    """归并后最大板块应占据可观比例（少数大板块主导，符合尺度分布）。"""
    micro = _micro_plate_field()
    merged = voronoi.merge_micro_plates(micro, n_major=7, seed=5)
    counts = np.bincount(merged.ravel())
    largest_frac = counts.max() / counts.sum()
    assert largest_frac > 0.15
