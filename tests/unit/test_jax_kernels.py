"""§6.3/§6.4 JAX 内核测试（可选依赖，未安装时自动跳过）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import spherical
from virtual_world.terrain import erosion as ero
from virtual_world.terrain import hydraulic_erosion as he
from virtual_world.terrain import hydrology as hd
from virtual_world.terrain import isostasy, jax_kernels

HAS_JAX = jax_kernels.jax_available()

#: 地球年降水 1 m 折算的降水速率 (m/s)
P_EARTH = ero.DEFAULT_PRECIPITATION_M_PER_S
#: JAX 与 Numba 两条加速路径允许的最大绝对偏差 (m)。
#: 实测（本机 CPU，24x48 圆锥岛，60 步）：固定 dt 下 elevation/drop **逐位相同**、
#: water_depth ≤ 2.2e-19、sediment ≤ 1.4e-17；自适应 dt 下 elevation ≤ 2.8e-14
#: （相对 3000 m 地形量级 ≈ 9e-18）。偏差来自 Numba 内核的 ``fastmath=True``
#: （允许重排/近似，偏离严格 IEEE；已用严格 IEEE 的 NumPy 单步重放核对：JAX 逐位
#: 相同、Numba 差 ~2e-21），再经 ``speed - u_c`` 相减与自适应 dt 反馈放大。
TOL = 1e-12


def test_jax_available_returns_bool() -> None:
    assert isinstance(jax_kernels.jax_available(), bool)


def test_jax_kernel_validation_runs_without_dependency() -> None:
    """参数校验先于 JAX 导入，因此无 JAX 环境也能给出明确报错。"""
    crust = np.zeros((4, 6))
    with pytest.raises(ValueError):
        jax_kernels.integrate_crust_thickness_jax(crust, 0.0, 0.0, crust, dt=0.0, n_steps=1)
    with pytest.raises(ValueError):
        jax_kernels.integrate_crust_thickness_jax(crust, 0.0, 0.0, crust, dt=1.0, n_steps=0)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_jax_kernel_matches_analytic_relaxation() -> None:
    """与分裂格式的解析解一致（§4.4/§6.4）。

    分裂格式为"先增厚、再松弛"，故定点是 ``C_eq + T*(1-kappa*dt)/kappa``。
    """
    c0, c_eq, thick, thin, kappa, dt, n = 50.0, 30.0, 0.4, 0.0, 0.1, 1.0, 200
    crust = np.full((3, 4), c0)
    out = jax_kernels.integrate_crust_thickness_jax(
        crust, thick, thin, np.full((3, 4), c_eq), kappa=kappa, dt=dt, n_steps=n
    )
    decay = 1.0 - kappa * dt
    fixed_point = c_eq + thick * decay / kappa
    expected = fixed_point + (c0 - fixed_point) * decay**n
    assert np.allclose(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_jax_kernel_matches_numba_kernel() -> None:
    """与 Numba 内核数值一致（两条加速路径必须可互换）。"""
    rng = np.random.default_rng(11)
    crust = rng.uniform(25, 60, size=(12, 18))
    thickening = rng.uniform(0, 0.8, size=(12, 18))
    thinning = rng.uniform(0, 0.5, size=(12, 18))
    c_eq = np.full((12, 18), 30.0)
    expected = isostasy.integrate_crust_thickness(
        crust, thickening, thinning, c_eq, kappa=0.02, dt=2.0, n_steps=500
    )
    actual = jax_kernels.integrate_crust_thickness_jax(
        crust, thickening, thinning, c_eq, kappa=0.02, dt=2.0, n_steps=500
    )
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_jax_kernel_broadcasts_scalars() -> None:
    """标量 thickening/thinning 正确广播，且返回 NumPy 数组。"""
    crust = np.full((2, 3), 30.0)
    out = jax_kernels.integrate_crust_thickness_jax(
        crust, 0.5, 0.0, np.full((2, 3), 30.0), kappa=0.0, dt=1.0, n_steps=10
    )
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, 35.0)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_jax_kernel_long_integration_is_stable() -> None:
    """数千步长时积分不发散（方案 §6.4 的适用场景）。"""
    crust = np.full((6, 8), 30.0)
    out = jax_kernels.integrate_crust_thickness_jax(
        crust, 0.2, 0.0, np.full((6, 8), 30.0), kappa=0.04, dt=1.0, n_steps=5000
    )
    assert np.all(np.isfinite(out))
    assert float(out.max()) < 40.0


# ===== §6.3 水力侵蚀 JAX 内核（与 Numba 逐格等价）=====


def _island(
    nlat: int = 24,
    nlon: int = 48,
    *,
    peak: float = 1500.0,
    half_width_rad: float = 0.7,
    sea_floor: float = -3000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """赤道正圆锥岛 + 确定性正弦起伏（含海洋掩码，构造与地球尺度无关）。

    圆锥给出明确的高处→海岸排水方向，正弦起伏制造汇流不均——两者共同保证
    既有真实的水量输运，也有非零的悬移质（才能激活 MacCormack 分支）。
    """
    lat = spherical.lat_centers(nlat)
    lon = spherical.lon_centers(nlon)
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    cos_d = np.cos(phi) * np.cos(lam)
    dist = np.arccos(np.clip(cos_d, -1.0, 1.0))
    land = dist < half_width_rad
    elev = np.where(land, peak * (1.0 - dist / half_width_rad), sea_floor)
    rough = 100.0 * np.sin(3.0 * phi) * np.cos(5.0 * lam) + 40.0 * np.sin(
        11.0 * phi + 2.0
    ) * np.cos(7.0 * lam)
    elev = elev + np.where(land, rough, 0.0)
    return np.asarray(elev, dtype=np.float64), elev < 0.0


def _assert_equivalent(
    actual: he.HydraulicErosionResult, expected: he.HydraulicErosionResult, tol: float = TOL
) -> None:
    """逐格比较两个内核的四条输出场，最大绝对偏差必须在 ``tol`` 之内。"""
    deviations = {
        name: float(
            np.abs(
                np.asarray(getattr(actual, name), dtype=np.float64)
                - np.asarray(getattr(expected, name), dtype=np.float64)
            ).max()
        )
        for name in ("elevation", "drop", "water_depth", "sediment")
    }
    worst = max(deviations.values())
    detail = ", ".join(f"{name}={value:.3e}" for name, value in deviations.items())
    assert worst <= tol, f"最大绝对偏差超限: {detail}（上限 {tol:.1e}）"
    assert actual.report.keys() == expected.report.keys()
    for key, value in expected.report.items():
        got = actual.report[key]
        if isinstance(value, str):
            assert got == value
        else:
            assert float(got) == pytest.approx(float(value), rel=tol, abs=tol)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
@pytest.mark.parametrize("macormack", [False, True])
@pytest.mark.parametrize("fixed_dt", [None, 120.0])
def test_hydraulic_erode_jax_matches_numba(macormack: bool, fixed_dt: float | None) -> None:
    """四个组合（MacCormack 开/关 × 固定/自适应 dt）都与 Numba 逐格等价。

    ``fixed_dt=None`` 走 CFL 自适应步长（dt 作为 ``lax.scan`` 的 carry 随步更新），
    ``fixed_dt=120.0`` 走固定步长。
    """
    elev, ocean = _island()
    kwargs = dict(precipitation=P_EARTH, n_steps=60, macormack=macormack, dt=fixed_dt)
    expected = he.hydraulic_erode(elev, ocean, **kwargs)
    actual = jax_kernels.hydraulic_erode_jax(elev, ocean, **kwargs)
    assert expected.report["macormack"] == macormack
    assert expected.report["adaptive_dt"] == (fixed_dt is None)
    _assert_equivalent(actual, expected)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_exercises_sediment_transport() -> None:
    """前提检查：等价性测试的地形确实产生非零悬移质（否则 MacCormack 分支被架空）。"""
    elev, ocean = _island()
    result = he.hydraulic_erode(elev, ocean, precipitation=P_EARTH, n_steps=60)
    assert float(result.sediment.max()) > 0.0
    assert float(np.abs(result.drop).max()) > 0.0


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
@pytest.mark.parametrize("macormack", [False, True])
def test_hydraulic_erode_jax_equivalence_without_ocean(macormack: bool) -> None:
    """全陆网格（无海洋掩码外延）与含海洋掩码的网格都要等价。"""
    elev, _ = _island(12, 24)
    ocean = np.zeros_like(elev, dtype=bool)
    kwargs = dict(precipitation=P_EARTH, n_steps=40, macormack=macormack)
    _assert_equivalent(
        jax_kernels.hydraulic_erode_jax(elev, ocean, **kwargs),
        he.hydraulic_erode(elev, ocean, **kwargs),
    )


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_equivalence_on_irregular_mask() -> None:
    """非带状的不规则海洋掩码（含内陆湖与极区行）仍逐格等价。"""
    nlat, nlon = 16, 20
    rng = np.random.default_rng(5)
    elev = 500.0 + rng.normal(size=(nlat, nlon)) * 200.0
    ocean = rng.random((nlat, nlon)) < 0.25
    ocean[:2, :] = False  # 极区行留作陆地，检验"首行无南向邻居"的分支
    elev = np.where(ocean, -2500.0, elev)
    kwargs = dict(precipitation=P_EARTH, n_steps=50, dt=200.0)
    _assert_equivalent(
        jax_kernels.hydraulic_erode_jax(elev, ocean, **kwargs),
        he.hydraulic_erode(elev, ocean, **kwargs),
    )


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_accepts_precipitation_field() -> None:
    """逐格降水场（气候模块接入点）同样与 Numba 等价。"""
    elev, ocean = _island(12, 24)
    precip = np.where(ocean, 0.0, P_EARTH * 3.0)
    kwargs = dict(precipitation=precip, n_steps=30)
    _assert_equivalent(
        jax_kernels.hydraulic_erode_jax(elev, ocean, **kwargs),
        he.hydraulic_erode(elev, ocean, **kwargs),
    )


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_is_deterministic() -> None:
    """同一输入 ⇒ 逐位相同输出（无随机成分，与 Numba 内核同约定）。"""
    elev, ocean = _island(12, 24)
    kwargs = dict(precipitation=P_EARTH, n_steps=25)
    a = jax_kernels.hydraulic_erode_jax(elev, ocean, **kwargs)
    b = jax_kernels.hydraulic_erode_jax(elev, ocean, **kwargs)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.sediment, b.sediment)


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_physics_invariants() -> None:
    """水量非负、悬移质非负、海洋单元严格不变（与 Numba 内核同物理约束）。"""
    elev, ocean = _island(16, 32)
    result = jax_kernels.hydraulic_erode_jax(elev, ocean, precipitation=50.0 * P_EARTH, n_steps=40)
    assert np.all(result.water_depth >= 0.0)
    assert np.all(result.sediment >= 0.0)
    assert np.all(np.isfinite(result.elevation))
    assert np.array_equal(result.elevation[ocean], elev[ocean])
    assert np.all(result.sediment[ocean] == 0.0)
    assert result.report["max_water_depth_m"] > 0.0


def test_hydraulic_erode_jax_validation_matches_numba() -> None:
    """参数校验与 :func:`hydraulic_erode` 一致（且先于 JAX 导入，无 JAX 也能报错）。"""
    elev, ocean = _island(8, 16)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev, ocean, precipitation=-1.0, n_steps=5)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev, ocean, precipitation=P_EARTH, n_steps=0)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev, ocean, precipitation=np.zeros((4, 8)), n_steps=5)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev, ocean, precipitation=P_EARTH, dt=0.0)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev, ocean, precipitation=P_EARTH, cfl=1.5)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(
            elev, ocean, precipitation=P_EARTH, erosion_rate=1.5, n_steps=5
        )
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(
            elev, ocean, precipitation=P_EARTH, pipe_coefficient=0.0, n_steps=5
        )
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev.ndim * np.zeros(3), ocean, precipitation=P_EARTH)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(elev, ocean[:-1], precipitation=P_EARTH)
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(
            np.zeros((1, 8)), np.zeros((1, 8), dtype=bool), precipitation=P_EARTH
        )
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(
            elev, ocean, precipitation=P_EARTH, max_velocity=0.0, n_steps=5
        )
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(
            elev, ocean, precipitation=P_EARTH, critical_velocity=-1.0, n_steps=5
        )
    with pytest.raises(ValueError):
        jax_kernels.hydraulic_erode_jax(
            elev, ocean, precipitation=P_EARTH, capacity_constant=-1.0, n_steps=5
        )


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_defaults_match_numba_signature() -> None:
    """默认值与 :func:`hydraulic_erode` 共用同一组模块常量。"""
    import inspect

    jax_params = inspect.signature(jax_kernels.hydraulic_erode_jax).parameters
    numba_params = inspect.signature(he.hydraulic_erode).parameters
    assert list(jax_params) == list(numba_params)
    for name, param in numba_params.items():
        assert jax_params[name].default == param.default, name


@pytest.mark.skipif(not HAS_JAX, reason="未安装 jax（pip install jax）")
def test_hydraulic_erode_jax_passes_validate_erosion() -> None:
    """端到端：JAX 内核的输出可直接构造第三层结果并通过 :func:`validate_erosion`。"""
    elev, ocean = _island()
    bed = hd.fill_pits(elev, ocean)
    hydraulic = jax_kernels.hydraulic_erode_jax(
        bed, ocean, precipitation=P_EARTH, n_steps=60, macormack=True
    )
    final = hydraulic.elevation
    hydro = hd.analyze_hydrology(final, ocean)
    result = ero.ErosionResult(
        elevation=final,
        erosion_bed=bed,
        uplift=np.zeros_like(bed),
        filled_elevation=hydro.filled_elevation,
        hydraulic_drop=bed - final,
        thermal_drop=np.zeros_like(bed),
        flow_direction=hydro.flow_direction,
        flow_accumulation=hydro.flow_accumulation,
        river_network=hydro.river_network,
        river_width=hydro.river_width,
        report={},
        seed=0,
    )
    assert ero.validate_erosion(result) == []
    assert float(np.abs(hydraulic.drop).max()) > 0.0
