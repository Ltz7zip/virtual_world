"""第二层噪声与扩散精修测试（《第二层完善：噪声与扩散精修》）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import noise_refine as nr


def _xyz_grid(nlat: int, nlon: int, scale: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单位球面上的 3D 噪声输入坐标（《第二层完善》§4.1）。"""
    lat = -90.0 + (180.0 / nlat) * (np.arange(nlat) + 0.5)
    lon = -180.0 + (360.0 / nlon) * (np.arange(nlon) + 0.5)
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    cp = np.cos(phi)
    x = cp * np.cos(lam) * scale
    y = cp * np.sin(lam) * scale
    z = np.sin(phi) * np.ones((nlat, nlon)) * scale
    return x, y, z


# ===== 行为 1：噪声核确定性 =====


def test_simplex_deterministic_same_seed() -> None:
    x, y, z = _xyz_grid(24, 48)
    a = nr.simplex_noise(x, y, z, seed=7)
    b = nr.simplex_noise(x, y, z, seed=7)
    assert np.array_equal(a, b)


def test_simplex_different_seeds_differ() -> None:
    x, y, z = _xyz_grid(24, 48)
    a = nr.simplex_noise(x, y, z, seed=1)
    b = nr.simplex_noise(x, y, z, seed=999)
    assert not np.allclose(a, b)


def test_simplex_range() -> None:
    """Simplex 输出落在 [-1, 1]。"""
    x, y, z = _xyz_grid(32, 64)
    out = nr.simplex_noise(x, y, z, seed=3)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_simplex_mean_about_zero() -> None:
    """光滑噪声的大样本均值应接近零。"""
    x, y, z = _xyz_grid(48, 96)
    out = nr.simplex_noise(x, y, z, seed=5)
    assert abs(float(np.mean(out))) < 0.1


# ===== 行为 2：噪声核时空分布 =====


def test_worley_nonnegative() -> None:
    """Worley F1 距离场必然非负。"""
    x, y, z = _xyz_grid(32, 64)
    out = nr.worley_noise(x, y, z, seed=2)
    assert np.all(out >= 0.0)


def test_ridged_nonnegative() -> None:
    """Ridged fBm（Σ|noise|）必然非负。"""
    x, y, z = _xyz_grid(32, 64)
    out = nr.ridged_noise(x, y, z, seed=4, octaves=4)
    assert np.all(out >= 0.0)


def test_turbulence_range() -> None:
    """Turbulence（1-|n|）落在 [0, 1]。"""
    x, y, z = _xyz_grid(32, 64)
    out = nr.turbulence_noise(x, y, z, seed=4, octaves=4)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_fbm_adds_octaves() -> None:
    """fBm 多 octave ≠ 单 octave，且保持确定性。"""
    x, y, z = _xyz_grid(24, 48)
    single = nr.fbm_noise(x, y, z, seed=9, octaves=1)
    multi = nr.fbm_noise(x, y, z, seed=9, octaves=5)
    assert not np.allclose(single, multi)
    again = nr.fbm_noise(x, y, z, seed=9, octaves=5)
    assert np.array_equal(multi, again)


# ===== 行为 3：域扭曲 =====


def test_domain_warp_changes_field() -> None:
    x, y, z = _xyz_grid(32, 64)
    plain = nr.fbm_noise(x, y, z, seed=11, octaves=4)
    warped = nr.warped_noise(x, y, z, seed=11, octaves=4, warp_amp=0.15)
    assert not np.allclose(plain, warped)


def test_domain_warp_deterministic() -> None:
    x, y, z = _xyz_grid(32, 64)
    a = nr.warped_noise(x, y, z, seed=17, octaves=4, warp_amp=0.15)
    b = nr.warped_noise(x, y, z, seed=17, octaves=4, warp_amp=0.15)
    assert np.array_equal(a, b)


def test_domain_warp_amplitude_scales_difference() -> None:
    x, y, z = _xyz_grid(32, 64)
    base = nr.fbm_noise(x, y, z, seed=23, octaves=4)
    small = nr.warped_noise(x, y, z, seed=23, octaves=4, warp_amp=0.05)
    large = nr.warped_noise(x, y, z, seed=23, octaves=4, warp_amp=0.30)
    d_small = float(np.mean(np.abs(small - base)))
    d_large = float(np.mean(np.abs(large - base)))
    assert d_large > d_small


def test_domain_warp_zero_amp_identity() -> None:
    x, y, z = _xyz_grid(24, 48)
    plain = nr.fbm_noise(x, y, z, seed=7, octaves=4)
    warped = nr.warped_noise(x, y, z, seed=7, octaves=4, warp_amp=0.0)
    assert np.allclose(plain, warped)


# ===== 行为 4：按构造环境选择噪声核 =====


def test_select_noise_kernel_matches_environment_table() -> None:
    """§1.3 分配策略表：
    造山带（汇聚）→ RIDGED；洋中脊（离散）→ WORLEY；海沟（汇聚+深海洋壳）→ TURBULENCE；
    大陆内部（稳定地盾，非边界）→ SIMPLEX。
    """
    from virtual_world.terrain.euler_poles import BoundaryType

    nlat, nlon = 24, 48
    btype = np.full((nlat, nlon), BoundaryType.TRANSFORM, dtype=np.int32)
    elev = np.full((nlat, nlon), 300.0)  # 大陆内部低地
    btype[10:14, :] = BoundaryType.CONVERGENT  # 造山带
    btype[4:8, :] = BoundaryType.DIVERGENT  # 洋中脊
    btype[17:20, :] = BoundaryType.CONVERGENT  # 俯冲带海沟
    elev[10:14, :] = 1500.0  # 造山带高海拔
    elev[4:8, :] = -2500.0  # 洋中脊海底
    elev[17:20, :] = -7000.0  # 海沟（深海洋壳）

    kernel = nr.select_noise_kernel(btype, elev)
    assert kernel.shape == (nlat, nlon)
    # 造山带 → RIDGED
    assert np.all(kernel[10:14, :] == nr.Kernel.RIDGED)
    # 洋中脊 → WORLEY
    assert np.all(kernel[4:8, :] == nr.Kernel.WORLEY)
    # 海沟（汇聚 + 深海洋壳）→ TURBULENCE
    assert np.all(kernel[17:20, :] == nr.Kernel.TURBULENCE)
    # 大陆内部（非边界）→ SIMPLEX
    assert np.all(kernel[0:4, :] == nr.Kernel.SIMPLEX)


# ===== 行为 5：残差与融合 =====


def test_refine_noise_residual_zero_mean() -> None:
    """H_residual 均值为零（§核心原则）。"""
    nlat, nlon = 48, 96
    x, y, z = _xyz_grid(nlat, nlon)
    tectonic = 2000.0 * nr.simplex_noise(x, y, z, seed=31) * 4.0  # 伪构造场
    btype = np.full((nlat, nlon), 2, dtype=np.int32)  # TRANSFORM
    out = nr.refine_noise(tectonic, btype, seed=13, freq_scale=4.0)
    assert out.elevation.shape == tectonic.shape
    assert np.allclose(out.tectonic, tectonic)
    assert abs(float(np.mean(out.residual))) < 1e-8


def test_refine_noise_final_is_tectonic_plus_residual() -> None:
    nlat, nlon = 24, 48
    x, y, z = _xyz_grid(nlat, nlon)
    tectonic = 3000.0 * nr.simplex_noise(x, y, z, seed=41) * 3.0
    btype = np.full((nlat, nlon), 2, dtype=np.int32)
    out = nr.refine_noise(tectonic, btype, seed=19, freq_scale=4.0)
    assert np.allclose(out.elevation, tectonic + out.residual)


def test_refine_noise_low_frequency_preserved() -> None:
    """低频分量受构造层约束：残差的粗网格平均应远小于构造层（§核心原则）。"""
    nlat, nlon = 64, 128
    x, y, z = _xyz_grid(nlat, nlon)

    def lowpass(field: np.ndarray) -> np.ndarray:
        coarse = field.reshape(nlat // 4, 4, nlon // 4, 4).mean(axis=(1, 3))
        return coarse

    tectonic = 2500.0 * nr.simplex_noise(x, y, z, seed=47) * 3.0
    btype = np.full((nlat, nlon), 2, dtype=np.int32)
    out = nr.refine_noise(tectonic, btype, seed=29, freq_scale=4.0)
    # 残差在 16×16 块上的粗网格均值 std 应远小于构造层同尺度 std
    res_coarse = lowpass(out.residual)
    tec_coarse = lowpass(out.tectonic)
    assert float(res_coarse.std()) < 0.25 * (float(tec_coarse.std()) + 1e-9)


def test_refine_noise_deterministic() -> None:
    nlat, nlon = 24, 48
    x, y, z = _xyz_grid(nlat, nlon)
    tectonic = nr.simplex_noise(x, y, z, seed=55)
    btype = np.full((nlat, nlon), 2, dtype=np.int32)
    a = nr.refine_noise(tectonic, btype, seed=33, freq_scale=4.0)
    b = nr.refine_noise(tectonic, btype, seed=33, freq_scale=4.0)
    assert np.allclose(a.elevation, b.elevation)
    assert np.allclose(a.residual, b.residual)


def test_refine_noise_non_divisible_block_shape() -> None:
    """低分辨率网格（如 preview 18×36）下 block 不整除时形状保持（§7 管线）。"""
    nlat, nlon = 18, 36
    x, y, z = _xyz_grid(nlat, nlon)
    tectonic = 1000.0 * nr.simplex_noise(x, y, z, seed=89) * 2.0
    btype = np.full((nlat, nlon), 2, dtype=np.int32)
    out = nr.refine_noise(tectonic, btype, seed=43, freq_scale=4.0, block=4)
    assert out.elevation.shape == (nlat, nlon)
    assert out.residual.shape == (nlat, nlon)
    assert np.all(np.isfinite(out.elevation))


# ===== 行为 6：Numba 加速（正确性一致） =====


def test_numba_kernel_matches_python_reference() -> None:
    """Numba 加速的 Simplex 内核与纯 Python 参考实现一致（§6.1）。"""
    x, y, z = _xyz_grid(8, 16)
    pg = nr.python_simplex_reference(x, y, z, seed=61)
    nj = nr.simplex_noise(x, y, z, seed=61)
    assert np.allclose(nj, pg, atol=1e-7)


def test_noise_inputs_validation() -> None:
    x, y, z = _xyz_grid(24, 48)
    with pytest.raises(ValueError):
        nr.simplex_noise(x, y, z, seed=-1)
    with pytest.raises(ValueError):
        nr.fbm_noise(x, y, z, seed=0, octaves=0)
    with pytest.raises(ValueError):
        nr.warped_noise(x, y, z, seed=0, warp_amp=-0.1)


def test_boundary_distance_field_zero_on_boundary() -> None:
    """边界距离场：非转换边界处距离为 0，远处为正（§3.2 条件通道 3）。"""
    from virtual_world.terrain.euler_poles import BoundaryType

    nlat, nlon = 16, 32
    btype = np.full((nlat, nlon), BoundaryType.TRANSFORM, dtype=np.int32)
    btype[8, :] = BoundaryType.CONVERGENT
    dist = nr.boundary_distance_field(btype)
    assert np.all(dist[8, :] == 0.0)
    assert np.all(dist[0, :] > 0.0) and np.all(dist[15, :] > 0.0)
    # 距离随接近边界递减（行 0 → 8 单调下降）
    assert np.all(np.diff(dist[0:4, 0]) < 0)


def test_boundary_distance_field_all_transform_ok() -> None:
    """全转换边界（无真实边界）不应产生 NaN。"""
    from virtual_world.terrain.euler_poles import BoundaryType

    btype = np.full((8, 16), BoundaryType.TRANSFORM, dtype=np.int32)
    dist = nr.boundary_distance_field(btype)
    assert np.all(np.isfinite(dist))