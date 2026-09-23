"""扩散精修测试（《扩散精修方案》§三–§六）。

覆盖：条件通道（4 通道）、D8 汇流累积、tile 划分与拼接、约束强制
（零均值 / 低频截断 / 海岸线）、内置结构化精修器、模型适配器、
三频带融合、顶层管线与约束验证、条件通道缓存。
"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import diffusion as df
from virtual_world.terrain import noise_refine as nr
from virtual_world.terrain.euler_poles import BoundaryType


def _sphere_coords(nlat: int, nlon: int, scale: float = 4.0) -> tuple[np.ndarray, ...]:
    lat = -90.0 + (180.0 / nlat) * (np.arange(nlat) + 0.5)
    lon = -180.0 + (360.0 / nlon) * (np.arange(nlon) + 0.5)
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    cp = np.cos(phi)
    return (
        cp * np.cos(lam) * scale,
        cp * np.sin(lam) * scale,
        np.sin(phi) * np.ones((nlat, nlon)) * scale,
    )


def _tectonic(nlat: int, nlon: int, seed: int = 7, amp: float = 2500.0) -> np.ndarray:
    x, y, z = _sphere_coords(nlat, nlon)
    return amp * nr.simplex_noise(x, y, z, seed=seed) * 2.0


def _boundary_types(nlat: int, nlon: int) -> np.ndarray:
    bt = np.full((nlat, nlon), BoundaryType.TRANSFORM, dtype=np.int32)
    bt[nlat // 3, :] = BoundaryType.CONVERGENT
    bt[2 * nlat // 3, :] = BoundaryType.DIVERGENT
    return bt


# ===== 行为 1：D8 汇流累积（§3.2 条件通道 4） =====


def _east_draining_plane(nlat: int, nlon: int) -> np.ndarray:
    """西高东低的倾斜面：最东列为海洋，陆地单元一律向东汇流。

    东侧设海洋是为了排除经度环绕造成的伪陡坡（最西格与最东格相邻）。
    """
    elev = np.tile((nlon - 1 - np.arange(nlon))[None, :] * 10.0, (nlat, 1))
    elev[:, -1] = -1000.0  # 最东列为海
    return elev


def test_d8_accumulation_eastward_plane() -> None:
    """向东倾斜的平面：每格汇流 = 本列及以西的陆地格数（水流直行向东）。"""
    nlat, nlon = 8, 12
    elev = _east_draining_plane(nlat, nlon)
    acc = df.d8_flow_accumulation(elev, min_elevation=0.0)
    assert acc.shape == (nlat, nlon)
    # 陆地列 j=0..nlon-2 汇流依次为 1..nlon-1
    expected = np.arange(1, nlon).astype(np.float64)
    assert np.allclose(acc[:, :-1], expected[None, :])
    assert np.allclose(acc[:, -1], 0.0)  # 海洋列不计汇流


def test_d8_accumulation_land_only() -> None:
    """海洋单元（elev <= min_elevation）不产生汇流累积。"""
    nlat, nlon = 8, 12
    elev = _east_draining_plane(nlat, nlon)
    elev[:, :3] = -500.0  # 西侧也为海
    acc = df.d8_flow_accumulation(elev, min_elevation=0.0)
    assert np.all(acc[:, :3] == 0.0)
    assert np.all(acc[:, -1] == 0.0)


def test_d8_accumulation_monotone_along_flow() -> None:
    """下游单元的汇流累积严格大于上游单元。"""
    nlat, nlon = 8, 12
    elev = _east_draining_plane(nlat, nlon)
    acc = df.d8_flow_accumulation(elev, min_elevation=0.0)
    assert np.all(np.diff(acc[0, :-1]) > 0)


# ===== 行为 2：条件通道（§3.2 四通道） =====


def test_build_condition_channels_shapes_and_masks() -> None:
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    bt = _boundary_types(nlat, nlon)
    cond = df.build_condition_channels(tec, bt, block=4)
    assert cond.lowpass.shape == (nlat, nlon)
    assert cond.land_mask.shape == (nlat, nlon)
    assert cond.land_mask.dtype == np.bool_
    assert cond.boundary_distance.shape == (nlat, nlon)
    assert cond.river_network.shape == (nlat, nlon)
    # 海陆掩码与构造高程符号一致
    assert np.array_equal(cond.land_mask, tec > 0.0)
    # 河流只出现在陆地上
    assert not np.any(cond.river_network & ~cond.land_mask)


def test_condition_channels_stack_four_channels() -> None:
    nlat, nlon = 16, 32
    cond = df.build_condition_channels(_tectonic(nlat, nlon), _boundary_types(nlat, nlon), block=4)
    stack = cond.stack()
    assert stack.shape == (4, nlat, nlon)


def test_condition_lowpass_of_constant_is_constant() -> None:
    tec = np.full((16, 32), 700.0)
    bt = _boundary_types(16, 32)
    cond = df.build_condition_channels(tec, bt, block=4)
    assert np.allclose(cond.lowpass, 700.0)


def test_condition_lowpass_removes_high_frequency() -> None:
    """低频通道应抑制逐格交替的高频分量。"""
    nlat, nlon = 16, 32
    checker = np.indices((nlat, nlon)).sum(axis=0) % 2 * 1000.0
    bt = _boundary_types(nlat, nlon)
    cond = df.build_condition_channels(checker, bt, block=4)
    assert float(cond.lowpass.std()) < float(checker.std()) * 0.2


def test_condition_boundary_distance_zero_on_boundary() -> None:
    nlat, nlon = 16, 32
    bt = _boundary_types(nlat, nlon)
    cond = df.build_condition_channels(_tectonic(nlat, nlon), bt, block=4)
    assert np.all(cond.boundary_distance[nlat // 3, :] == 0.0)
    assert np.all(cond.boundary_distance[2 * nlat // 3, :] == 0.0)


def test_condition_river_network_detects_channels() -> None:
    """默认阈值下应真的检出河道（不只是空集），且位于下游汇聚处。"""
    nlat, nlon = 12, 24
    elev = _east_draining_plane(nlat, nlon)
    cond = df.build_condition_channels(
        elev, np.full((nlat, nlon), BoundaryType.TRANSFORM, dtype=np.int32), block=4
    )
    assert int(cond.river_network.sum()) > 0
    # 汇流量沿流向累积，河道只应出现在下游（东）侧
    cols = np.nonzero(cond.river_network.any(axis=0))[0]
    assert cols.min() > 0


def test_condition_river_network_accepts_precomputed() -> None:
    """可传入预算河流网络，跳过 D8 计算（§3.2 通道 4）。"""
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    pre = np.zeros((nlat, nlon), dtype=bool)
    pre[8, 10:20] = True
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4, river_network=pre)
    assert np.array_equal(cond.river_network, pre)


# ===== 行为 3：tile 划分与拼接（§4.2 批量 tile 推理） =====


def test_iter_tiles_full_coverage() -> None:
    """tile 必须无遗漏地覆盖整个网格。"""
    nlat, nlon = 20, 36
    tiles = df.iter_tiles(nlat, nlon, tile_size=8, overlap=2)
    covered = np.zeros((nlat, nlon), dtype=bool)
    for t in tiles:
        covered[t.i0 : t.i1, t.j0 : t.j1] = True
    assert covered.all()


def test_iter_tiles_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        df.iter_tiles(16, 32, tile_size=0)
    with pytest.raises(ValueError):
        df.iter_tiles(16, 32, tile_size=4, overlap=4)


def test_stitch_identity_for_consistent_tiles() -> None:
    """拼接权重归一：所有 tile 给出同一常量场时，拼接结果等于该常量。"""
    nlat, nlon = 20, 36
    tiles = df.iter_tiles(nlat, nlon, tile_size=8, overlap=2)
    field = np.full((nlat, nlon), 3.5)
    pieces = [field[t.i0 : t.i1, t.j0 : t.j1] for t in tiles]
    stitched = df.stitch_tiles(tiles, pieces, (nlat, nlon))
    assert np.allclose(stitched, 3.5)


def test_tile_weight_map_covers_all_cells() -> None:
    nlat, nlon = 20, 36
    tiles = df.iter_tiles(nlat, nlon, tile_size=8, overlap=2)
    weights = df.tile_weight_map(tiles, (nlat, nlon))
    assert weights.shape == (nlat, nlon)
    assert np.all(weights > 0.0)


# ===== 行为 4：约束强制（§三 数学约束） =====


def test_constraints_residual_zero_mean() -> None:
    """FF1：残差均值约束 E[H_residual] = 0。"""
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    residual = np.full((nlat, nlon), 120.0)
    out = df.enforce_residual_constraints(residual, tec, block=4)
    assert abs(float(out.mean())) < 1e-9


def test_constraints_remove_low_frequency() -> None:
    """FF2：低频截断——粗块均值应远小于原始低频分量。"""
    nlat, nlon = 32, 64
    tec = _tectonic(nlat, nlon)
    # 纯低频信号（线性斜坡）应被显著抑制
    ramp = np.add.outer(np.linspace(-500.0, 500.0, nlat), np.linspace(-300.0, 300.0, nlon))
    out = df.enforce_residual_constraints(ramp, tec, block=8)
    block = out.reshape(nlat // 8, 8, nlon // 8, 8).mean(axis=(1, 3))
    assert float(np.abs(block).max()) < 1.0
    assert float(out.std()) < float(ramp.std()) * 0.5


def test_constraints_coastline_forced_to_zero() -> None:
    """FF3：海岸线处 |H_tectonic| < eps 时残差被清零。"""
    nlat, nlon = 16, 32
    tec = np.full((nlat, nlon), 500.0)
    tec[8, :] = 0.0  # 一条海岸线
    residual = np.full((nlat, nlon), 50.0)
    out = df.enforce_residual_constraints(residual, tec, block=4, coastline_eps=1.0)
    assert np.allclose(out[8, :], 0.0)


def test_constraints_preserve_high_frequency_detail() -> None:
    """约束不应抹平真正的高频细节（周期 4 格，块尺度以下）。"""
    nlat, nlon = 32, 64
    tec = _tectonic(nlat, nlon)
    ii, jj = np.indices((nlat, nlon))
    high = (((ii // 2 + jj // 2) % 2) * 2.0 - 1.0) * 120.0  # 周期 4 格
    out = df.enforce_residual_constraints(high, tec, block=8)
    assert float(out.std()) > float(high.std()) * 0.5


# ===== 行为 5：内置结构化精修器（§5.2 可独立使用的精修层） =====


def test_structured_refiner_deterministic_and_shaped() -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    refiner = df.StructuredDiffusionRefiner()
    tile = df.Tile(0, nlat, 0, nlon)
    tcond = df.TileConditions.from_global(tile, tec, cond, _sphere_coords(nlat, nlon, scale=6.0))
    a = refiner.refine_tile(tcond, seed=5)
    b = refiner.refine_tile(tcond, seed=5)
    assert a.shape == (nlat, nlon)
    assert np.array_equal(a, b)


def test_structured_refiner_different_seed_differs() -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    refiner = df.StructuredDiffusionRefiner()
    tcond = df.TileConditions.from_global(
        df.Tile(0, nlat, 0, nlon), tec, cond, _sphere_coords(nlat, nlon, scale=6.0)
    )
    assert not np.allclose(refiner.refine_tile(tcond, seed=5), refiner.refine_tile(tcond, seed=6))


def test_structured_refiner_seamless_across_overlapping_tiles() -> None:
    """重叠区一致：噪声由全局坐标决定，重叠部分逐点相同（无接缝）。"""
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    refiner = df.StructuredDiffusionRefiner()
    xyz = _sphere_coords(nlat, nlon, scale=6.0)
    tiles = df.iter_tiles(nlat, nlon, tile_size=16, overlap=8)
    pieces = [
        refiner.refine_tile(df.TileConditions.from_global(t, tec, cond, xyz), seed=9) for t in tiles
    ]
    left, right = pieces[0], pieces[1]
    left_cols = tiles[0].j1 - tiles[1].j0
    assert left_cols > 0
    assert np.allclose(left[:, -left_cols:], right[:, :left_cols])


# ===== 行为 6：模型适配器（可选重依赖） =====


def test_model_refiner_reports_missing_dependency() -> None:
    """无可用的扩散模型时给出可操作的错误，而非静默降级。"""
    refiner = df.TerrainDiffusionRefiner(
        model_id="xandergos/terrain-diffusion-90m", allow_download=False
    )
    nlat, nlon = 8, 16
    tec = _tectonic(nlat, nlon, seed=1)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    tcond = df.TileConditions.from_global(
        df.Tile(0, nlat, 0, nlon), tec, cond, _sphere_coords(nlat, nlon, scale=4.0)
    )
    with pytest.raises(RuntimeError, match="terrain-diffusion"):
        refiner.refine_tile(tcond, seed=0)


# ===== 行为 7：三频带融合（§5.1） =====


def test_frequency_merge_partition_of_unity() -> None:
    """FF6：窗函数构成单位分解——三个输入相同时输出等于该输入。"""
    nlat, nlon = 16, 32
    x, y, z = _sphere_coords(nlat, nlon)
    field = nr.simplex_noise(x, y, z, seed=11) * 500.0
    merged = df.frequency_merge(field, field, field, k_low=0.1, k_mid=0.35, k_high=0.6)
    assert np.allclose(merged, field, atol=1e-6)


def test_frequency_windows_partition_of_unity() -> None:
    """FF6：三频带窗严格构成单位分解且非负。"""
    w_low, w_mid, w_high = df.frequency_windows((32, 64), k_low=0.1, k_mid=0.35, k_high=0.6)
    assert np.allclose(w_low + w_mid + w_high, 1.0)
    assert np.all(w_low >= 0.0) and np.all(w_mid >= 0.0) and np.all(w_high >= 0.0)


def test_frequency_windows_passbands() -> None:
    """通带性质：``k <= k_low`` 全走构造层，``k >= k_high`` 全走扩散层。"""
    nlat, nlon = 32, 64
    w_low, w_mid, w_high = df.frequency_windows((nlat, nlon), k_low=0.1, k_mid=0.35, k_high=0.6)
    k = np.sqrt(np.fft.fftfreq(nlat)[:, None] ** 2 + np.fft.fftfreq(nlon)[None, :] ** 2)
    low = k <= 0.1
    high = k >= 0.6
    assert np.allclose(w_low[low], 1.0)
    assert np.allclose(w_mid[low], 0.0)
    assert np.allclose(w_high[low], 0.0)
    assert np.allclose(w_high[high], 1.0)
    assert np.allclose(w_low[high], 0.0)


def test_frequency_merge_preserves_low_band() -> None:
    """低频段（>100 km 波长）完全由构造层决定，与噪声/扩散层无关（§5.1）。

    用严格谱投影（截止取 ``k_low/2``，稳在低通带内）验证：投影后的输出与
    投影后的构造层逐点一致；同时验证高频段确实被细节层改变。
    """
    nlat, nlon = 32, 64
    k_low, k_mid, k_high = 0.1, 0.35, 0.6
    tec = _tectonic(nlat, nlon)
    x, y, z = _sphere_coords(nlat, nlon, scale=14.0)
    noise = nr.simplex_noise(x, y, z, seed=21) * 800.0
    detail = nr.simplex_noise(x, y, z, seed=31) * 800.0

    k = np.sqrt(np.fft.fftfreq(nlat)[:, None] ** 2 + np.fft.fftfreq(nlon)[None, :] ** 2)

    def project(field: np.ndarray, keep: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(field) * keep).real

    merged = df.frequency_merge(tec, noise, detail, k_low=k_low, k_mid=k_mid, k_high=k_high)
    low_keep = k <= k_low / 2.0
    assert np.allclose(project(merged, low_keep), project(tec, low_keep), atol=1e-6)

    high_keep = k >= k_high
    assert not np.allclose(project(merged, high_keep), project(tec, high_keep))
    assert float(np.std(project(merged, high_keep))) > 0.0


def test_frequency_merge_windows_validated() -> None:
    nlat, nlon = 8, 16
    f = np.zeros((nlat, nlon))
    with pytest.raises(ValueError):
        df.frequency_merge(f, f, f, k_low=0.5, k_mid=0.2, k_high=0.6)


# ===== 行为 8：顶层管线（§6） =====


def test_refine_diffusion_result_shape_and_identity() -> None:
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    res = df.refine_diffusion(
        tec, _boundary_types(nlat, nlon), seed=3, tile_size=16, overlap=4, block=4
    )
    assert res.elevation.shape == (nlat, nlon)
    assert res.residual.shape == (nlat, nlon)
    assert np.allclose(res.tectonic, tec)
    assert np.all(np.isfinite(res.elevation))


def test_refine_diffusion_deterministic() -> None:
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    bt = _boundary_types(nlat, nlon)
    a = df.refine_diffusion(tec, bt, seed=17, tile_size=16, overlap=4, block=4)
    b = df.refine_diffusion(tec, bt, seed=17, tile_size=16, overlap=4, block=4)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.residual, b.residual)


def test_refine_diffusion_accepts_external_refiner() -> None:
    """可注入自定义精修器（模型适配器即走此路径）。"""
    nlat, nlon = 16, 32

    class HalfRefiner:
        def refine_tile(self, conditions: df.TileConditions, seed: int) -> np.ndarray:
            return np.full(conditions.tectonic.shape, 0.5)

    tec = _tectonic(nlat, nlon)
    res = df.refine_diffusion(
        tec,
        _boundary_types(nlat, nlon),
        seed=1,
        tile_size=16,
        overlap=0,
        block=4,
        refiner=HalfRefiner(),
    )
    assert res.elevation.shape == (nlat, nlon)
    # 常数 0.5 残差经约束后被清零（零均值 + 低频截断）
    assert np.allclose(res.residual, 0.0, atol=1e-6)


def test_refine_diffusion_constraint_report_clean() -> None:
    """顶层管线自身输出应通过全部约束验证（§6 第 7 步）。"""
    nlat, nlon = 32, 64
    tec = _tectonic(nlat, nlon)
    res = df.refine_diffusion(
        tec, _boundary_types(nlat, nlon), seed=23, tile_size=16, overlap=4, block=8
    )
    problems = df.validate_constraints(res, block=8, coastline_eps=1.0)
    assert problems == []


def test_refine_diffusion_coastline_pinned_to_sea_level() -> None:
    """FF3：海岸线处 H_final = 0，细节层不得推移海陆边界（§3.3）。"""
    nlat, nlon = 32, 64
    tec = _tectonic(nlat, nlon)
    tec[nlat // 2, :] = 0.0  # 显式构造一条岸线（构造场粗网格不一定落在 ±1 m 内）
    x, y, z = _sphere_coords(nlat, nlon, scale=14.0)
    noise = nr.simplex_noise(x, y, z, seed=51) * 900.0
    res = df.refine_diffusion(
        tec, _boundary_types(nlat, nlon), seed=9, noise=noise, tile_size=16, overlap=4, block=8
    )
    coast = np.abs(res.tectonic) < df.COASTLINE_EPS
    assert coast.any()
    assert np.all(res.elevation[coast] == 0.0)


def test_validate_constraints_detects_violations() -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    bad = df.DiffusionRefineResult(
        elevation=tec + 100.0,
        residual=np.full((nlat, nlon), 100.0),
        tectonic=tec,
        conditions=cond,
        noise=np.zeros((nlat, nlon)),
        report={},
        seed=0,
    )
    problems = df.validate_constraints(bad, block=4, coastline_eps=1.0)
    assert any("均值" in p for p in problems)


# ===== 行为 9：条件通道缓存（§4.3） =====


def test_condition_cache_hits_and_roundtrip(tmp_path) -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    bt = _boundary_types(nlat, nlon)
    calls = {"n": 0}

    def builder() -> df.ConditionChannels:
        calls["n"] += 1
        return df.build_condition_channels(tec, bt, block=4)

    cache = df.ConditionsCache(tmp_path)
    first = cache.get_or_build(df.condition_cache_key(tec, bt, block=4), builder)
    second = cache.get_or_build(df.condition_cache_key(tec, bt, block=4), builder)
    assert calls["n"] == 1
    assert np.allclose(first.lowpass, second.lowpass)
    assert np.array_equal(first.land_mask, second.land_mask)

    # 落盘后再从磁盘读回
    reloaded = df.ConditionsCache(tmp_path).get_or_build(
        df.condition_cache_key(tec, bt, block=4), builder
    )
    assert calls["n"] == 1
    assert np.allclose(reloaded.lowpass, first.lowpass)
    assert np.array_equal(reloaded.river_network, first.river_network)


def test_condition_cache_key_reacts_to_input() -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    bt = _boundary_types(nlat, nlon)
    k1 = df.condition_cache_key(tec, bt, block=4)
    k2 = df.condition_cache_key(tec * 1.001, bt, block=4)
    k3 = df.condition_cache_key(tec, bt, block=8)
    assert k1 != k2
    assert k1 != k3
    assert k1 == df.condition_cache_key(tec, bt, block=4)


# ===== 行为 10：§1.3 选核图（两条路径纹理一致） =====


def test_condition_channels_include_kernel_map() -> None:
    """条件通道携带 §1.3 选核图；模型输入仍是 §3.2 的四通道。"""
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    bt = _boundary_types(nlat, nlon)
    cond = df.build_condition_channels(tec, bt, block=4)
    assert np.array_equal(cond.kernel_map, nr.select_noise_kernel(bt, tec))
    assert cond.kernel_map.shape == (nlat, nlon)
    assert cond.stack().shape == (4, nlat, nlon)


def test_tile_conditions_carry_kernel_map_and_init_noise() -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    coords = _sphere_coords(nlat, nlon, scale=6.0)
    noise = np.full((nlat, nlon), 123.0)
    tile = df.Tile(4, 12, 8, 24)

    plain = df.TileConditions.from_global(tile, tec, cond, coords)
    assert np.all(plain.init_noise == 0.0)
    assert np.array_equal(plain.kernel_map, cond.kernel_map[4:12, 8:24])
    assert np.array_equal(plain.kernel_map, nr.select_noise_kernel(
        _boundary_types(nlat, nlon), tec
    )[4:12, 8:24])

    with_noise = df.TileConditions.from_global(tile, tec, cond, coords, noise)
    assert np.all(with_noise.init_noise == 123.0)
    assert with_noise.tectonic.shape == (8, 16)


def test_structured_refiner_uses_kernel_map() -> None:
    """精修器按 §1.3 选核图逐单元选核——换核图即换纹理。"""
    import dataclasses

    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    xyz = _sphere_coords(nlat, nlon, scale=6.0)
    tcond = df.TileConditions.from_global(df.Tile(0, nlat, 0, nlon), tec, cond, xyz)
    refiner = df.StructuredDiffusionRefiner()

    simplex_only = dataclasses.replace(
        tcond, kernel_map=np.full((nlat, nlon), int(nr.Kernel.SIMPLEX), dtype=np.int32)
    )
    worley_only = dataclasses.replace(
        tcond, kernel_map=np.full((nlat, nlon), int(nr.Kernel.WORLEY), dtype=np.int32)
    )
    a = refiner.refine_tile(simplex_only, seed=5)
    b = refiner.refine_tile(worley_only, seed=5)
    assert not np.allclose(a, b)
    # 与默认选核图（含造山带 Ridged / 洋中脊 Worley）也不同
    assert not np.allclose(a, refiner.refine_tile(tcond, seed=5))


def test_structured_refiner_uses_initial_noise() -> None:
    """§5.2 第 2 步：有初始噪声时以其为细节基底，且重叠 tile 仍逐点一致。"""
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    cond = df.build_condition_channels(tec, _boundary_types(nlat, nlon), block=4)
    xyz = _sphere_coords(nlat, nlon, scale=6.0)
    noise = nr.simplex_noise(*_sphere_coords(nlat, nlon, scale=20.0), seed=77) * 400.0

    tile = df.Tile(0, nlat, 0, nlon)
    base = df.TileConditions.from_global(tile, tec, cond, xyz)
    seeded = df.TileConditions.from_global(tile, tec, cond, xyz, noise)
    refiner = df.StructuredDiffusionRefiner()
    out = refiner.refine_tile(seeded, seed=5)
    corr = float(np.corrcoef(out.ravel(), noise.ravel())[0, 1])
    assert corr > 0.9
    assert not np.allclose(out, refiner.refine_tile(base, seed=5))

    # 重叠区一致性：初始噪声逐点切片 + 逐点增益，不应产生接缝
    tiles = df.iter_tiles(nlat, nlon, tile_size=16, overlap=8)
    pieces = [
        refiner.refine_tile(df.TileConditions.from_global(t, tec, cond, xyz, noise), seed=9)
        for t in tiles
    ]
    overlap_cols = tiles[0].j1 - tiles[1].j0
    assert overlap_cols > 0
    assert np.allclose(pieces[0][:, -overlap_cols:], pieces[1][:, :overlap_cols])


def test_refine_diffusion_uses_noise_as_initial_noise() -> None:
    """顶层管线把 noise 同时作为中频带与扩散初始噪声（§5.1、§5.2）。"""
    nlat, nlon = 24, 48
    tec = _tectonic(nlat, nlon)
    noise = nr.simplex_noise(*_sphere_coords(nlat, nlon, scale=20.0), seed=13) * 500.0
    res = df.refine_diffusion(
        tec, _boundary_types(nlat, nlon), seed=5, noise=noise, tile_size=16, overlap=4, block=4
    )
    assert float(np.corrcoef(res.residual.ravel(), noise.ravel())[0, 1]) > 0.5


def test_refine_diffusion_rejects_non_2d_grid() -> None:
    """tile + FFT 融合建立在矩形周期网格上，立方球需走噪声路径或先重网格。"""
    tec = np.zeros((6, 8, 8))
    bt = np.zeros((6, 8, 8), dtype=np.int32)
    with pytest.raises(ValueError, match="经纬网格"):
        df.refine_diffusion(tec, bt, seed=0)


def test_condition_cache_preserves_kernel_map(tmp_path) -> None:
    nlat, nlon = 16, 32
    tec = _tectonic(nlat, nlon)
    bt = _boundary_types(nlat, nlon)
    cache = df.ConditionsCache(tmp_path)
    key = df.condition_cache_key(tec, bt, block=4)
    first = cache.get_or_build(key, lambda: df.build_condition_channels(tec, bt, block=4))
    second = df.ConditionsCache(tmp_path).get_or_build(key, lambda: df.build_condition_channels(tec, bt, block=4))
    assert np.array_equal(first.kernel_map, second.kernel_map)
    assert np.array_equal(second.kernel_map, nr.select_noise_kernel(bt, tec))
