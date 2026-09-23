"""欧拉极点与板块运动学测试（《第一层完善》§2）。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.terrain import euler_poles as ep


def test_plate_velocity_solid_body_rotation() -> None:
    """单板块刚体旋转：v = omega x P，与欧拉公式一致（§2.1）。"""
    # 绕 z 轴角速度 1 rad/time：赤道 (1,0,0) 处速度 (0,1,0)
    omega = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    v = ep.plate_velocity(omega, p)
    assert np.allclose(v, [0.0, 1.0, 0.0])

    # 极点处速度为零
    pole = np.array([0.0, 0.0, 1.0])
    assert np.allclose(ep.plate_velocity(omega, pole), 0.0, atol=1e-12)


def test_plate_velocity_vectorized() -> None:
    omega = np.array([0.0, 0.0, 2.0])
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v = ep.plate_velocity(omega, p)
    assert v.shape == (2, 3)
    assert np.allclose(v, [[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0]])


def test_relative_velocity() -> None:
    """v_rel = (omega_A - omega_B) x P（§2.2）。"""
    omega_a = np.array([0.0, 0.0, 3.0])
    omega_b = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    v_rel = ep.relative_velocity(omega_a, omega_b, p)
    assert np.allclose(v_rel, [0.0, 2.0, 0.0])


def test_boundary_type_divergent_convergent_transform() -> None:
    """边界类型判定（§2.3）：v_rel 与法向同向 → 离散，反向 → 汇聚，垂直 → 转换。

    构造：omega_A=+z, omega_B=-z。边界点 P=(1,0,0) 处
    v_rel = (omega_A-omega_B)xP = 2z_hat x x_hat = +2 y_hat。
    """
    omega_a = np.array([0.0, 0.0, 1.0])
    omega_b = np.array([0.0, 0.0, -1.0])
    p = np.array([1.0, 0.0, 0.0])

    # 法向 = +y（与 v_rel 同向）→ 离散
    n = np.array([0.0, 1.0, 0.0])
    assert ep.boundary_type(omega_a, omega_b, p, n) == ep.BoundaryType.DIVERGENT

    # 法向 = z（与 v_rel 垂直）→ 转换
    n_perp = np.array([0.0, 0.0, 1.0])
    assert ep.boundary_type(omega_a, omega_b, p, n_perp) == ep.BoundaryType.TRANSFORM

    # 法向 = -y（与 v_rel 反向）→ 汇聚
    n_rev = np.array([0.0, -1.0, 0.0])
    assert ep.boundary_type(omega_a, omega_b, p, n_rev) == ep.BoundaryType.CONVERGENT


def test_boundary_type_vectorized_consistency() -> None:
    omega_a = np.array([0.0, 0.0, 1.0])
    omega_b = np.array([0.0, 0.0, -1.0])
    p = np.tile([1.0, 0.0, 0.0], (3, 1))
    n = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    btypes = ep.boundary_type(omega_a, omega_b, p, n)
    assert btypes[0] == ep.BoundaryType.DIVERGENT
    assert btypes[1] == ep.BoundaryType.DIVERGENT
    assert btypes[2] == ep.BoundaryType.TRANSFORM
    assert btypes.dtype == np.int32


def test_boundary_type_threshold_normalization() -> None:
    """阈值按相对速率归一化：纯切向运动始终判定为转换边界。"""
    omega_a = np.array([0.0, 0.0, 10.0])
    omega_b = np.array([0.0, 0.0, -10.0])
    p = np.array([0.0, 1.0, 0.0])
    # v_rel = 2*10 z_hat x y_hat = -20 x_hat；法向 z 与 v_rel 垂直 → 转换
    n = np.array([0.0, 0.0, 1.0])
    assert ep.boundary_type(omega_a, omega_b, p, n) == ep.BoundaryType.TRANSFORM


def test_pick_euler_poles_deterministic_and_positive_rate() -> None:
    """随机分配欧拉极点：确定性与角速度幅值可配置。"""
    omegas = ep.pick_euler_poles(n_plates=6, seed=42)
    assert omegas.shape == (6, 3)
    omegas2 = ep.pick_euler_poles(n_plates=6, seed=42)
    assert np.array_equal(omegas, omegas2)
    norms = np.linalg.norm(omegas, axis=1)
    assert np.all(norms > 0)


def test_pick_euler_poles_validation() -> None:
    with pytest.raises(ValueError):
        ep.pick_euler_poles(0)
    with pytest.raises(ValueError):
        ep.euler_pole_axis_angle(np.zeros(3))
