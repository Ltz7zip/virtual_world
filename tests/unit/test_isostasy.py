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


# ===== §4.2 质量守恒与缩短因子 =====


def test_shortening_factor_round_trip() -> None:
    """beta = C/C_0 与 C = C_0*beta 互逆（§4.2）。"""
    c0, beta = 36.0, 2.34
    crust = isostasy.crust_from_shortening(c0, beta)
    assert crust == pytest.approx(c0 * beta)
    assert isostasy.shortening_factor(crust, c0) == pytest.approx(beta)


def test_shortening_factor_andes_reference() -> None:
    """§4.2 基准：安第斯中部 beta_0 = 2.34 对应 C = 30*2.34 km。"""
    c0 = 30.0
    beta = 2.34
    crust = isostasy.crust_from_shortening(c0, beta)
    assert crust == pytest.approx(70.2)
    assert isostasy.shortening_factor(crust, c0) == pytest.approx(beta)


def test_mass_conservation_identity() -> None:
    """L_0*C_0 = L*C：缩短量与增厚量严格互换（§4.2）。"""
    c0, l0, convergence = 36.0, 8000.0, 4000.0
    crust, beta = isostasy.mass_conserving_thickness(c0, l0, convergence)
    remaining = l0 - isostasy.shortening_distance_km(beta, l0)
    assert remaining == pytest.approx(l0 - convergence)
    assert remaining * crust == pytest.approx(l0 * c0, rel=1e-12)


def test_max_shortening_factor_limits() -> None:
    """无汇聚 -> beta=1；缩短一半 -> beta=2（§4.2）。"""
    assert isostasy.max_shortening_factor(1000.0, 0.0) == pytest.approx(1.0)
    assert isostasy.max_shortening_factor(1000.0, 500.0) == pytest.approx(2.0)


def test_shortening_validation() -> None:
    with pytest.raises(ValueError):
        isostasy.shortening_factor(30.0, 36.0)  # 增厚后反而更薄
    with pytest.raises(ValueError):
        isostasy.shortening_factor(36.0, 0.0)
    with pytest.raises(ValueError):
        isostasy.crust_from_shortening(36.0, 0.5)
    with pytest.raises(ValueError):
        isostasy.max_shortening_factor(1000.0, 1000.0)
    with pytest.raises(ValueError):
        isostasy.max_shortening_factor(1000.0, -1.0)
    with pytest.raises(ValueError):
        isostasy.shortening_distance_km(0.8, 1000.0)


# ===== §4.3 造山带高宽比 =====


def test_orogen_aspect_ratio_scaling() -> None:
    """§4.3：H/W ∝ (d_rho*g*C_0)/(eta_eff*|v_rel|)。"""
    base = isostasy.orogen_aspect_ratio(500.0, 36.0, 1.0)
    # 地壳越厚 -> 高宽比越大
    assert isostasy.orogen_aspect_ratio(500.0, 72.0, 1.0) > base
    # 汇聚越快 -> 按显式公式高宽比越小（见函数 docstring 的方案自相矛盾说明）
    assert isostasy.orogen_aspect_ratio(500.0, 36.0, 2.0) < base
    # 密度差越大 -> 高宽比越大
    assert isostasy.orogen_aspect_ratio(1000.0, 36.0, 1.0) > base
    # 增厚效率越高 -> 高宽比越小
    assert isostasy.orogen_aspect_ratio(500.0, 36.0, 1.0, eta_eff=0.4) < base


def test_orogen_aspect_ratio_is_calibrated() -> None:
    """参考条件下高宽比等于标定值，且随 C_0 线性变化（§4.3）。"""
    ref = isostasy.orogen_aspect_ratio(
        isostasy.REFERENCE_DELTA_RHO, isostasy.REFERENCE_C0_KM, isostasy.REFERENCE_V_REL
    )
    assert ref == pytest.approx(isostasy.REFERENCE_ASPECT_RATIO)
    doubled = isostasy.orogen_aspect_ratio(
        isostasy.REFERENCE_DELTA_RHO, 2.0 * isostasy.REFERENCE_C0_KM, isostasy.REFERENCE_V_REL
    )
    assert doubled == pytest.approx(2.0 * isostasy.REFERENCE_ASPECT_RATIO)
    # 参考高宽比 + 喜马拉雅量级高度 → 半宽 100 km 量级
    assert isostasy.ridge_half_width_km(5000.0, isostasy.REFERENCE_ASPECT_RATIO) == pytest.approx(
        100.0
    )
    # 归一化后正是方案的比例式：比值之比等于物理量之比
    ratio_a = isostasy.orogen_aspect_ratio(500.0, 36.0, 1.0)
    ratio_b = isostasy.orogen_aspect_ratio(500.0, 36.0, 3.0)
    assert ratio_a / ratio_b == pytest.approx(3.0)


def test_ridge_half_width_from_aspect_ratio() -> None:
    """W = H/(H/W)（§4.3）。"""
    assert isostasy.ridge_half_width_km(2000.0, 0.1) == pytest.approx(20.0)
    assert isostasy.ridge_half_width_km(-2000.0, 0.1) == pytest.approx(20.0)


def test_orogen_aspect_ratio_validation() -> None:
    with pytest.raises(ValueError):
        isostasy.orogen_aspect_ratio(500.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        isostasy.orogen_aspect_ratio(500.0, 36.0, 0.0)
    with pytest.raises(ValueError):
        isostasy.orogen_aspect_ratio(500.0, 36.0, 1.0, eta_eff=0.0)
    with pytest.raises(ValueError):
        isostasy.ridge_half_width_km(1000.0, 0.0)
