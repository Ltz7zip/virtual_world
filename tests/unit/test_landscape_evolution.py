"""§2.1 河流功率定律与 §2.2 景观演化方程测试（《第三层完善：侵蚀模拟》§二）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import spherical
from virtual_world.core.constants import EARTH_RADIUS
from virtual_world.terrain import hydrology as hd
from virtual_world.terrain import landscape_evolution as le


def _north_ramp(nlat: int = 16, nlon: int = 32, *, drop: float = 10.0) -> np.ndarray:
    """北低南高的直坡：行索引越大越高，故所有单元的 D8 流向为北(64)。"""
    rows = np.arange(nlat)[:, None]
    return np.asarray(1000.0 - drop * rows, dtype=np.float64) * np.ones((1, nlon))


def _north_ramp_setup(nlat: int = 16, nlon: int = 32, *, drop: float = 10.0):
    elev = _north_ramp(nlat, nlon, drop=drop)
    ocean = np.zeros_like(elev, dtype=bool)
    filled = hd.fill_pits(elev, ocean)
    direction = hd.flow_directions(filled, ocean, radius=EARTH_RADIUS)
    return elev, ocean, direction


def _cell_area(nlat: int, nlon: int) -> np.ndarray:
    return spherical.cell_areas(spherical.lat_edges(nlat), nlon, EARTH_RADIUS)


# ===== §2.1 河流功率定律 =====


def test_stream_power_scales_with_drainage_area() -> None:
    """E ∝ A^m：汇水面积扩大 4 倍，下切扩大 4^0.5 = 2 倍（m = 0.5）。"""
    nlat, nlon = 16, 32
    elev, ocean, direction = _north_ramp_setup(nlat, nlon)
    area = _cell_area(nlat, nlon)
    base = le.stream_power_incision(
        elev, np.full_like(elev, 10.0), direction, coefficient=0.05, m=0.5, n=1.0,
        dt=0.1, cell_area=area, is_ocean=ocean,
    )
    larger = le.stream_power_incision(
        elev, np.full_like(elev, 40.0), direction, coefficient=0.05, m=0.5, n=1.0,
        dt=0.1, cell_area=area, is_ocean=ocean,
    )
    inner = slice(0, nlat - 1)  # 末行无下游，不下切
    assert np.all(base[inner, :] > 0.0)
    assert np.allclose(larger[inner, :] / base[inner, :], 2.0, rtol=1e-9)
    assert np.all(larger[nlat - 1, :] == 0.0)


def test_stream_power_scales_with_slope_exponent() -> None:
    """E ∝ S^n：坡度翻倍时 n=1 下切翻倍、n=2 下切 4 倍。"""
    nlat, nlon = 16, 32
    area = _cell_area(nlat, nlon)
    results: dict[tuple[float, float], np.ndarray] = {}
    for drop in (10.0, 20.0):
        elev, ocean, direction = _north_ramp_setup(nlat, nlon, drop=drop)
        for n in (1.0, 2.0):
            results[(drop, n)] = le.stream_power_incision(
                elev, np.full_like(elev, 5.0), direction, coefficient=0.05, m=0.5, n=n,
                dt=0.1, cell_area=area, is_ocean=ocean,
            )
    row = slice(1, nlat - 2)  # 避开末行（无下游）与首行（填洼影响）
    assert np.allclose(
        results[(20.0, 1.0)][row, :] / results[(10.0, 1.0)][row, :], 2.0, rtol=1e-9
    )
    assert np.allclose(
        results[(20.0, 2.0)][row, :] / results[(10.0, 2.0)][row, :], 4.0, rtol=1e-9
    )


def test_stream_power_zero_on_ocean() -> None:
    elev, ocean, direction = _north_ramp_setup()
    ocean[8:, :] = True
    incision = le.stream_power_incision(
        elev, np.full_like(elev, 10.0), direction, dt=0.1, cell_area=_cell_area(*elev.shape),
        is_ocean=ocean,
    )
    assert np.all(incision[ocean] == 0.0)


def test_stream_power_respects_incision_limit() -> None:
    """单步下切不超过到下游高差的 incision_limit 倍（显式积分稳定性保护）。"""
    nlat, nlon = 12, 24
    drop = 10.0
    elev, ocean, direction = _north_ramp_setup(nlat, nlon, drop=drop)
    limit = 0.25
    incision = le.stream_power_incision(
        elev, np.full_like(elev, 1e6), direction, coefficient=1e6, m=0.5, n=1.0,
        dt=1.0, cell_area=_cell_area(nlat, nlon), is_ocean=ocean, incision_limit=limit,
    )
    assert float(incision[0, 0]) == pytest.approx(limit * drop, rel=1e-9)


def test_stream_power_validation() -> None:
    elev, ocean, direction = _north_ramp_setup(8, 16)
    area = _cell_area(8, 16)
    kwargs = dict(dt=0.1, cell_area=area, is_ocean=ocean)
    with pytest.raises(ValueError):
        le.stream_power_incision(elev, elev, direction, coefficient=-1.0, **kwargs)
    with pytest.raises(ValueError):
        le.stream_power_incision(elev, elev, direction, m=0.0, **kwargs)
    with pytest.raises(ValueError):
        le.stream_power_incision(elev, elev, direction, n=0.0, **kwargs)
    with pytest.raises(ValueError):
        le.stream_power_incision(elev, elev, direction, dt=0.0, cell_area=area, is_ocean=ocean)
    with pytest.raises(ValueError):
        le.stream_power_incision(elev, elev, direction, incision_limit=2.0, **kwargs)
    with pytest.raises(ValueError):
        le.stream_power_incision(elev, elev[:4], direction, **kwargs)


# ===== §2.2 景观演化方程 =====


def _blob(nlat: int = 20, nlon: int = 40, *, amp: float = 30.0, k: float = 4.0) -> np.ndarray:
    """南高北低的起伏地形，南侧两行为海。"""
    rows = np.arange(nlat)[:, None]
    base = 300.0 + (nlat - rows) * 6.0
    wave = amp * np.sin(2.0 * np.pi * rows * k / nlat)
    return np.asarray(base + wave, dtype=np.float64) * np.ones((1, nlon))


def test_landscape_evolve_uplift_only_raises_land() -> None:
    """只有 U 时：陆地整块抬升 U*dt*n_steps，海洋不动。"""
    elev = _blob()
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    result = le.landscape_evolve(
        elev, ocean, uplift=10.0, coefficient=0.0, kappa=0.0, dt_ma=1.0, n_steps=3
    )
    land = ~ocean
    assert np.allclose(result.elevation[land], elev[land] + 30.0)
    assert np.allclose(result.elevation[ocean], elev[ocean])
    assert np.allclose(result.uplift_total[ocean], 0.0)
    assert result.report["mean_uplift_m"] == pytest.approx(30.0)


def test_landscape_evolve_incision_only_lowers_land() -> None:
    """只有 K*A^m*S^n 时：陆地只降不升，且确实下降。"""
    elev = _blob()
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    result = le.landscape_evolve(
        elev, ocean, uplift=0.0, coefficient=0.05, kappa=0.0, dt_ma=0.1, n_steps=5
    )
    land = ~ocean
    assert np.all(result.elevation[land] <= elev[land] + 1e-9)
    assert float(result.incision_total[land].max()) > 0.0
    assert np.allclose(result.elevation[ocean], elev[ocean])


def test_landscape_evolve_diffusion_smooths_and_is_nonzero() -> None:
    """坡面扩散项被积分进来（方案 §2.2 视为不可省），使起伏变小。"""
    elev = _blob(amp=60.0, k=6.0)
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    result = le.landscape_evolve(
        elev, ocean, uplift=0.0, coefficient=0.0, kappa=1.0e5, dt_ma=0.1, n_steps=3
    )
    land = ~ocean
    assert float(np.abs(result.diffusion_total[land]).max()) > 0.0
    assert float(result.elevation[land].std()) < float(elev[land].std())


def test_landscape_evolve_bookkeeping_identity() -> None:
    """H_final = H_initial + 累计抬升 − 累计下切 + 累计扩散（逐步记账恒等式）。"""
    elev = _blob()
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    result = le.landscape_evolve(
        elev, ocean, uplift=8.0, coefficient=0.05, kappa=1.0e4, dt_ma=0.1, n_steps=4
    )
    reconstructed = elev + result.uplift_total - result.incision_total + result.diffusion_total
    assert np.allclose(result.elevation, reconstructed, rtol=0.0, atol=1e-9)


def test_landscape_evolve_from_layer1_uplift_rate() -> None:
    """跨层因果链：第一层的抬升速率场直接作为 §2.2 的 U（§2.2 的要求）。"""
    from virtual_world.terrain import plate_tectonics as pt

    tec = pt.generate_tectonic_field(nlat=24, nlon=48, n_major=6, seed=3, time_ma=100.0)
    elev = np.asarray(tec.elevation)
    ocean = np.asarray(tec.is_ocean)
    rate = np.asarray(tec.uplift_rate_m_per_ma)
    result = le.landscape_evolve(
        elev, ocean, uplift=rate, coefficient=0.05, kappa=le.DEFAULT_KAPPA_M2_PER_MA,
        dt_ma=0.5, n_steps=4,
    )
    assert result.elevation.shape == elev.shape
    assert np.all(np.isfinite(result.elevation))
    assert result.report["mean_uplift_m"] > 0.0
    assert np.all(result.uplift_total[ocean] == 0.0)
    # 抬升只发生在陆地
    assert np.allclose(result.elevation[ocean], elev[ocean])


def test_landscape_evolve_deterministic() -> None:
    elev = _blob()
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    kwargs = dict(uplift=5.0, coefficient=0.05, kappa=1.0e4, dt_ma=0.2, n_steps=3)
    a = le.landscape_evolve(elev, ocean, **kwargs)
    b = le.landscape_evolve(elev, ocean, **kwargs)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.incision_total, b.incision_total)


def test_landscape_evolve_update_flow_every_changes_cost_not_contract() -> None:
    """降低流向重算频率只影响精度，不影响输出契约。"""
    elev = _blob()
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[:2, :] = True
    result = le.landscape_evolve(
        elev, ocean, uplift=2.0, coefficient=0.05, kappa=0.0, dt_ma=0.1, n_steps=6,
        update_flow_every=3,
    )
    assert result.elevation.shape == elev.shape
    assert np.all(np.isfinite(result.elevation))
    assert result.report["update_flow_every"] == 3


def test_landscape_evolve_validation() -> None:
    elev = _blob(8, 16)
    ocean = np.zeros_like(elev, dtype=bool)
    with pytest.raises(ValueError):
        le.landscape_evolve(elev, ocean, uplift=-1.0)
    with pytest.raises(ValueError):
        le.landscape_evolve(elev, ocean, dt_ma=0.0)
    with pytest.raises(ValueError):
        le.landscape_evolve(elev, ocean, n_steps=0)
    with pytest.raises(ValueError):
        le.landscape_evolve(elev, ocean, kappa=-1.0)
    with pytest.raises(ValueError):
        le.landscape_evolve(elev, ocean, update_flow_every=0)
    with pytest.raises(ValueError):
        le.landscape_evolve(elev[0], ocean[0])
    with pytest.raises(ValueError):
        le.landscape_evolve(elev, ocean, nonlinear_diffusion=True, talus_angle_deg=0.0)
