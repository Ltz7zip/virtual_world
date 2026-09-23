"""§6.4 JAX 长时积分内核测试（可选依赖，未安装时自动跳过）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import isostasy, jax_kernels

HAS_JAX = jax_kernels.jax_available()


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
