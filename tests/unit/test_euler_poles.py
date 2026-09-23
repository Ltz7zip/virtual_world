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


# ===== §2.4 单位四元数 =====


def test_quat_axis_angle_round_trip() -> None:
    """q = cos(theta/2) + sin(theta/2)*e 与其逆变换互逆（§2.4）。"""
    axis = np.array([1.0, 2.0, -2.0])
    angle = 1.234
    q = ep.quat_from_axis_angle(axis, angle)
    assert q.shape == (4,)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)

    back_axis, back_angle = ep.quat_to_axis_angle(q)
    assert back_angle == pytest.approx(angle, rel=1e-12)
    assert np.allclose(back_axis, axis / np.linalg.norm(axis), atol=1e-12)


def test_quat_from_axis_angle_zero_angle_is_identity() -> None:
    q = ep.quat_from_axis_angle(np.zeros(3), 0.0)
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])


def test_quat_rotate_matches_rodrigues() -> None:
    """四元数旋转 == 罗德里格斯公式（§2.4 与 §2.1 刚体旋转一致）。"""
    axis = np.array([0.0, 0.0, 1.0])
    angle = 0.7
    q = ep.quat_from_axis_angle(axis, angle)
    p = np.array([0.4, -1.3, 0.2])
    expected = (
        p * np.cos(angle)
        + np.cross(axis, p) * np.sin(angle)
        + axis * np.dot(axis, p) * (1.0 - np.cos(angle))
    )
    assert np.allclose(ep.quat_rotate(q, p), expected, atol=1e-12)


def test_quat_rotate_consistency_with_plate_velocity() -> None:
    """小角度旋转 P 得到 P + dt*(omega x P)，与 v = omega x P 一致（§2.1、§2.4）。"""
    omega = np.array([0.3, -0.7, 0.5])
    p = np.array([1.0, 0.0, 0.0])
    dt = 1.0e-6
    q = ep.quat_from_omega(omega, dt)
    moved = ep.quat_rotate(q, p)
    assert np.allclose(moved, p + dt * np.cross(omega, p), atol=1e-12)


def test_quat_multiply_matches_matrix_composition() -> None:
    """q_total = q_n ∘ ... ∘ q_1 与旋转矩阵连乘一致（§2.4）。"""
    q1 = ep.quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.4)
    q2 = ep.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -1.1)
    composed = ep.quat_multiply(q2, q1)
    expected = ep.quat_to_matrix(q2) @ ep.quat_to_matrix(q1)
    assert np.allclose(ep.quat_to_matrix(composed), expected, atol=1e-12)


def test_quat_composition_is_associative_and_inverse() -> None:
    q1 = ep.quat_from_axis_angle(np.array([1.0, 1.0, 0.0]), 0.9)
    q2 = ep.quat_from_axis_angle(np.array([0.0, 1.0, 1.0]), -0.3)
    q3 = ep.quat_from_axis_angle(np.array([1.0, 0.0, 1.0]), 2.0)
    left = ep.quat_multiply(ep.quat_multiply(q3, q2), q1)
    right = ep.quat_multiply(q3, ep.quat_multiply(q2, q1))
    assert np.allclose(left, right, atol=1e-12)
    # q ∘ q* = 单位四元数
    assert np.allclose(ep.quat_multiply(q1, ep.quat_conjugate(q1)), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_quat_omega_recovers_angular_velocity() -> None:
    """omega = 2 * q_dot ∘ q* 反解出角速度（§2.4）。"""
    axis = np.array([0.0, 0.0, 1.0])
    rate = 1.7
    t = 0.35
    dt = 1.0e-7
    q = ep.quat_from_axis_angle(axis, rate * t)
    q_next = ep.quat_from_axis_angle(axis, rate * (t + dt))
    qdot = (q_next - q) / dt
    omega = ep.quat_omega(q, qdot)
    assert np.allclose(omega, rate * axis, atol=1e-5)


def test_integrate_rotation_matches_analytic_constant_rate() -> None:
    """常角速度下，逐步四元数积分与解析解一致（§2.4 长时间积分）。"""
    omega = np.array([0.0, 0.0, 0.9])
    dt, n = 0.05, 400
    q0 = ep.quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.6)
    q = ep.integrate_rotation(q0, omega, dt, n)
    analytic = ep.quat_multiply(
        ep.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.9 * dt * n), q0
    )
    assert np.allclose(q, analytic, atol=1e-12)


def test_quaternion_integration_has_no_numerical_drift() -> None:
    """长时间积分下四元数保持单位模，旋转矩阵连乘则出现正交性漂移（§2.4）。

    这正是方案推荐四元数的理由：矩阵只能靠重新正交化清漂移，而四元数只需
    归一化（甚至不归一化时漂移也可忽略）。
    """
    omega = np.array([0.31, -0.47, 0.83])
    dt = 0.01
    n = 50_000
    q = ep.quat_from_omega(omega, dt)
    q_accum = np.array([1.0, 0.0, 0.0, 0.0])
    r_accum = np.eye(3)
    step_matrix = ep.quat_to_matrix(q)
    for _ in range(n):
        q_accum = ep.quat_multiply(q, q_accum)
        r_accum = step_matrix @ r_accum
    # 未归一化的四元数模长漂移
    quat_drift = abs(float(np.linalg.norm(q_accum)) - 1.0)
    # 旋转矩阵正交性漂移
    matrix_drift = float(np.abs(r_accum.T @ r_accum - np.eye(3)).max())
    assert quat_drift < 1.0e-9
    assert matrix_drift > quat_drift
    # 归一化后模长精确为 1
    assert np.linalg.norm(ep.quat_normalize(q_accum)) == pytest.approx(1.0, abs=1e-15)


def test_integrate_rotation_validation() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        ep.integrate_rotation(q0, omega, dt=0.0, n_steps=1)
    with pytest.raises(ValueError):
        ep.integrate_rotation(q0, omega, dt=1.0, n_steps=-1)


# ===== §2.5 板块角动量守恒 =====


def test_cap_angular_radius_full_sphere() -> None:
    radius = 6.371e6
    area = 4.0 * np.pi * radius**2
    assert ep.cap_angular_radius(area, radius) == pytest.approx(np.pi, rel=1e-12)


def test_cap_moment_of_inertia_full_sphere_thin_shell() -> None:
    """整球极限退化为薄壳公式 I = (2/3) m R^2（§2.5 转动惯量的物理校验）。"""
    radius, density, thickness = 6.371e6, 3000.0, 1.0e5
    inertia = ep.cap_moment_of_inertia(np.pi, radius, density, thickness)
    mass = 4.0 * np.pi * radius**2 * thickness * density
    assert inertia == pytest.approx(2.0 / 3.0 * mass * radius**2, rel=1e-12)


def test_cap_moment_of_inertia_monotonic_in_angular_radius() -> None:
    values = [ep.cap_moment_of_inertia(a, 6.371e6, 3000.0, 1.0e5) for a in (0.5, 1.0, 2.0, 3.0)]
    assert np.all(np.diff(values) > 0.0)


def test_merge_angular_momentum_conserves_angular_momentum() -> None:
    """L = sum(I_k w_k) 在合并前后守恒（§2.5）。"""
    omegas = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    inertias = np.array([3.0, 1.0])
    merged = ep.merge_angular_momentum(omegas, inertias)
    # 合并后 L 必须等于合并前 L
    assert np.allclose(inertias.sum() * merged, inertias @ omegas, atol=1e-15)
    assert np.allclose(merged, [0.25, 0.0, 0.75], atol=1e-15)


def test_merge_angular_momentum_equal_rates_unchanged() -> None:
    omegas = np.tile(np.array([0.2, -0.4, 0.1]), (4, 1))
    merged = ep.merge_angular_momentum(omegas, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(merged, [0.2, -0.4, 0.1], atol=1e-15)


def test_collision_kinetic_budget_nonnegative_and_zero_when_matched() -> None:
    """碰撞动能耗散 dE >= 0；速度一致时为零（§2.5）。"""
    inertias = np.array([2.0, 1.0])
    budget = ep.collision_kinetic_budget(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), inertias
    )
    assert budget > 0.0
    matched = ep.collision_kinetic_budget(
        np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]), inertias
    )
    assert matched == pytest.approx(0.0, abs=1e-15)


def test_merge_angular_momentum_validation() -> None:
    with pytest.raises(ValueError):
        ep.merge_angular_momentum(np.zeros((2, 4)), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        ep.merge_angular_momentum(np.zeros((2, 3)), np.array([1.0]))
    with pytest.raises(ValueError):
        ep.merge_angular_momentum(np.zeros((2, 3)), np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        ep.merge_angular_momentum(np.zeros((2, 3)), np.zeros(2))
    with pytest.raises(ValueError):
        ep.cap_angular_radius(0.0, 6.371e6)
