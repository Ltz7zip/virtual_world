"""Airy 均衡与地壳厚度演化测试（《第一层完善》§3–§4）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import isostasy


def test_airy_elevation_positive_crust() -> None:
    """§3.2：H = ((rho_m-rho_c)/rho_m) * (C - T)。"""
    rho_c, rho_m = 2800.0, 3300.0
    c_ref = 30.0  # km，对应于海平面
    # 地壳厚 70 km（青藏高原量级）→ 高海拔
    h = isostasy.airy_elevation(70.0, c_ref=c_ref, rho_crust=rho_c, rho_mantle=rho_m)
    assert h == pytest.approx((rho_m - rho_c) / rho_m * (70.0 - c_ref) * 1000.0, rel=1e-12)


def test_airy_elevation_reference_is_sea_level() -> None:
    """参考厚度 T 处高程为 0。"""
    h = isostasy.airy_elevation(30.0, c_ref=30.0)
    assert h == pytest.approx(0.0, abs=1e-9)


def test_airy_elevation_oceanic_crust_negative() -> None:
    """薄洋壳 → 负高程（海底）。"""
    # 洋壳 7 km：H = 残差为负
    h = isostasy.airy_elevation(7.0, c_ref=30.0)
    assert h < 0.0


def test_airy_elevation_validation() -> None:
    with pytest.raises(ValueError):
        isostasy.airy_elevation(-1.0)


def test_integrate_crust_thickening_only() -> None:
    """§4.4：纯汇聚增厚 C += eta*|v|*dt*n_steps。"""
    crust = np.full((4, 6), 30.0)
    out = isostasy.integrate_crust_thickness(
        crust,
        thickening=0.5,  # km/unit_time（已含 eta 与速率）
        thinning=0.0,
        c_eq=np.full((4, 6), 30.0),
        kappa=0.0,
        dt=1.0,
        n_steps=10,
    )
    assert np.allclose(out, 35.0)


def test_integrate_crust_thinning_only() -> None:
    """离散边界减薄 C -= delta*|v|*dt。"""
    crust = np.full((4, 6), 30.0)
    out = isostasy.integrate_crust_thickness(
        crust,
        thickening=0.0,
        thinning=0.2,
        c_eq=np.full((4, 6), 30.0),
        kappa=0.0,
        dt=1.0,
        n_steps=10,
    )
    assert np.allclose(out, 28.0)


def test_integrate_crust_gravity_relaxation_matches_analytic() -> None:
    """§4.4 重力松弛解析解：C(t) = C_eq + (C0-C_eq)*(1-kappa*dt)^n。"""
    crust = np.full((4, 6), 50.0)
    c_eq = np.full((4, 6), 30.0)
    kappa, dt, n = 0.1, 1.0, 50
    out = isostasy.integrate_crust_thickness(
        crust,
        thickening=0.0,
        thinning=0.0,
        c_eq=c_eq,
        kappa=kappa,
        dt=dt,
        n_steps=n,
    )
    expected = 30.0 + 20.0 * (1.0 - kappa) ** n
    assert np.allclose(out, expected, rtol=1e-9, atol=1e-9)


def test_integrate_crust_matches_numpy_reference() -> None:
    """Numba 内核与逐轮 NumPy 参考实现完全一致。"""
    rng = np.random.default_rng(7)
    crust = rng.uniform(25, 60, size=(12, 18))
    thickening = rng.uniform(0, 0.8, size=(12, 18))
    thinning = rng.uniform(0, 0.5, size=(12, 18))
    c_eq = np.full((12, 18), 30.0)
    out = isostasy.integrate_crust_thickness(
        crust, thickening, thinning, c_eq, kappa=0.02, dt=2.0, n_steps=30
    )
    ref = crust.copy()
    for _ in range(30):
        # 分裂格式（与 numba 内核一致，方案 §6.3）：
        # 先增厚/减薄，再用更新后的厚度做重力松弛
        ref = ref + (thickening - thinning) * 2.0
        ref = ref - 0.02 * (ref - c_eq) * 2.0
    assert np.allclose(out, ref, rtol=1e-7)


def test_integrate_crust_validation() -> None:
    with pytest.raises(ValueError):
        isostasy.integrate_crust_thickness(
            np.zeros((4, 6)), 0.0, 0.0, np.zeros((4, 6)), kappa=0.0, dt=-1.0, n_steps=1
        )
