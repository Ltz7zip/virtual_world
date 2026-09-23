"""欧拉极点与板块运动学（《第一层完善》§2）。

刚体旋转模型：每个板块由其欧拉极点（旋转轴向）与角速度描述，
``v = omega x P``；相邻板块的相对速率 ``v_rel = (omega_A - omega_B) x P``，
与边界法向的内积符号决定边界类型（离散/汇聚/转换）。

本模块还提供两个长时间演化所需的结构：

- **单位四元数**（§2.4）：``q = cos(theta/2) + sin(theta/2) * e``，连续旋转用
  四元数乘法合成、瞬时角速度 ``omega = 2 * q_dot o q*``。相比旋转矩阵没有
  万向节死锁，且长时间积分的数值漂移显著更小（见 :func:`integrate_rotation`）。
- **板块角动量守恒**（§2.5）：碰撞合并时 ``L = I_A*w_A + I_B*w_B = (I_A+I_B)*w``，
  碰撞消耗的动能 ``dE_kinetic`` 转为地壳增厚的势能预算
  （见 :func:`merge_angular_momentum` / :func:`collision_kinetic_budget`）。

四元数分量顺序统一为 ``(w, x, y, z)``（实部在前），形状 ``(..., 4)``；
角速度矢量形状 ``(..., 3)``。
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

#: 归一化阈值：|v_rel · n̂| < threshold * |v_rel| 判定为转换边界
TRANSFORM_THRESHOLD = 0.2


class BoundaryType(IntEnum):
    """板块边界类型（§2.3）。

    ``CONVERGENT`` / ``DIVERGENT`` / ``TRANSFORM`` 描述真实板块边界；
    ``INTERIOR`` 表示板块内部（非边界）单元——与真正的转换边界区分开，
    避免下游误把整个板块内部当成走滑带。
    """

    CONVERGENT = 0  # 汇聚（碰撞/俯冲）
    DIVERGENT = 1  # 离散（张裂）
    TRANSFORM = 2  # 转换（走滑）
    INTERIOR = 3  # 板块内部（非边界）


def euler_pole_axis_angle(omega: np.ndarray) -> tuple[np.ndarray, float]:
    """分解角速度矢量 → (单位轴 ê, 角速度大小 |ω|)（§2.1）。"""
    omega = np.asarray(omega, dtype=np.float64)
    rate = float(np.linalg.norm(omega))
    if rate == 0.0:
        raise ValueError("角速度为零：无法确定欧拉极点方向")
    return omega / rate, rate


def plate_velocity(omega: np.ndarray, point: np.ndarray) -> np.ndarray:
    """板块上某点的刚体速度 ``v = omega x P``，点可为 (..., 3)。"""
    omega = np.asarray(omega, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    return np.cross(omega, point)


def relative_velocity(omega_a: np.ndarray, omega_b: np.ndarray, point: np.ndarray) -> np.ndarray:
    """相对角速度矢量差在边界点处的线速度（§2.2）。"""
    omega_a = np.asarray(omega_a, dtype=np.float64)
    omega_b = np.asarray(omega_b, dtype=np.float64)
    return plate_velocity(omega_a - omega_b, point)


def boundary_type(
    omega_a: np.ndarray,
    omega_b: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    threshold: float = TRANSFORM_THRESHOLD,
) -> np.ndarray:
    """判定边界类型（§2.3）。

    ``point`` / ``normal`` 形状 ``(..., 3)`` 时可向量化；标量输入返回 0 维
    int32 数组（:class:`BoundaryType` 像素值）。阈值按相对速率归一化：
    ``|v_rel·n̂| > threshold*|v_rel|`` 为汇聚/离散，否则为转换。
    """
    v_rel = relative_velocity(omega_a, omega_b, point)
    n = np.asarray(normal, dtype=np.float64)
    dots = np.einsum("...i,...i->...", v_rel, n)
    speeds = np.linalg.norm(v_rel, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(speeds > 0.0, dots / np.where(speeds == 0.0, 1.0, speeds), 0.0)
    codes = np.where(
        ratio > threshold,
        BoundaryType.DIVERGENT,
        np.where(ratio < -threshold, BoundaryType.CONVERGENT, BoundaryType.TRANSFORM),
    )
    return np.asarray(codes, dtype=np.int32).reshape(np.asarray(ratio).shape)


def pick_euler_poles(
    n_plates: int,
    seed: int = 0,
    rate_min: float = 0.5,
    rate_max: float = 3.0,
) -> np.ndarray:
    """随机分配 ``n_plates`` 个板块的欧拉角速度矢量，形状 ``(n, 3)``。

    极点方向在球面均匀采样，角速度幅值在 [rate_min, rate_max]（rad/Myr 量级，
    单位由调用方解释）。确定性由 ``seed`` 驱动。
    """
    if n_plates <= 0:
        raise ValueError(f"板块数量必须为正整数，实际为 {n_plates}")
    rng = np.random.default_rng(seed)
    # Marsaglia 球面均匀采样（避免极点聚集）
    raw = rng.normal(size=(n_plates, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    rates = rng.uniform(rate_min, rate_max, size=(n_plates, 1))
    return raw * rates


# ===== §2.4 单位四元数 =====


def quat_from_axis_angle(axis: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """轴角 → 单位四元数 ``q = cos(theta/2) + sin(theta/2) * e``（§2.4）。

    ``axis`` 形状 ``(..., 3)``（无须预先归一化），``angle`` 与之广播，
    返回 ``(..., 4)``。``angle == 0`` 时返回单位四元数。
    """
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    safe = np.where(norm > 0.0, norm, 1.0)
    unit = axis / safe
    half = angle[..., None] / 2.0
    vector = np.sin(half) * unit
    vector = np.where(norm > 0.0, vector, 0.0)
    return np.asarray(np.concatenate([np.cos(half), vector], axis=-1), dtype=np.float64)


def quat_from_omega(omega: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """角速度矢量 → 增量单位四元数（``theta = |omega| * dt``、``e = omega/|omega|``）。"""
    omega = np.asarray(omega, dtype=np.float64)
    rate = np.linalg.norm(omega, axis=-1, keepdims=True)
    safe = np.where(rate > 0.0, rate, 1.0)
    return quat_from_axis_angle(omega / safe, rate[..., 0] * float(dt))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """四元数乘法 ``a ∘ b``（先施加 ``b`` 的旋转，再施加 ``a``）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.asarray(
        np.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            axis=-1,
        ),
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭 ``q* = (w, -x, -y, -z)``。"""
    q = np.asarray(q, dtype=np.float64)
    return np.asarray(np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1), dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """归一化为单位四元数（长时间积分的数值漂移清理）。"""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.asarray(q / np.where(norm > 0.0, norm, 1.0), dtype=np.float64)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """单位四元数 → 3x3 旋转矩阵，形状 ``(..., 3, 3)``。"""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.asarray(
        np.stack(
            [
                np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
                np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
                np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
            ],
            axis=-2,
        ),
        dtype=np.float64,
    )


def quat_rotate(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """用单位四元数旋转矢量，``vectors`` 形状 ``(..., 3)``。"""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"vectors 的最后一维必须为 3，实际 {v.shape}")
    matrix = quat_to_matrix(q)
    return np.asarray(np.einsum("...ij,...j->...i", matrix, v), dtype=np.float64)


def quat_omega(q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """瞬时角速度 ``omega = 2 * q_dot ∘ q*``（§2.4），返回矢量部，形状 ``(..., 3)``。

    要求 ``q`` 为单位四元数（否则结果混入缩放因子）。
    """
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qdot, dtype=np.float64)
    product = quat_multiply(qd, quat_conjugate(q))
    return np.asarray(2.0 * product[..., 1:], dtype=np.float64)


def quat_advance(q: np.ndarray, omega: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """单步四元数推进 ``q(t+dt) = exp(omega*dt/2) ∘ q(t)``（§2.4 连续旋转合成）。"""
    return quat_multiply(quat_from_omega(omega, dt), np.asarray(q, dtype=np.float64))


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """单位四元数 → ``(单位轴, 角度)``，形状 ``(..., 3)`` 与 ``(...)``。"""
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    vector = q[..., 1:]
    # angle → 0 时轴向无定义，退化为 +z（与 quat_from_axis_angle 的可逆性测试一致）
    axis = np.where(
        sin_half[..., None] > 1e-12,
        vector / np.where(sin_half[..., None] > 1e-12, sin_half[..., None], 1.0),
        np.array([0.0, 0.0, 1.0]),
    )
    return np.asarray(axis, dtype=np.float64), np.asarray(angle, dtype=np.float64)


def integrate_rotation(q0: np.ndarray, omega: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """长时间四元数积分 ``q_n = exp(n*omega*dt/2) ∘ q0``（§2.4 推荐做法）。

    每步归一化抑制浮点漂移。对**常角速度**该式与解析解逐位等价，因此可直接
    用于数百万年尺度的板块漂移积分。
    """
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")
    if n_steps < 0:
        raise ValueError(f"n_steps 不能为负，实际为 {n_steps}")
    omega = np.asarray(omega, dtype=np.float64)
    step = quat_from_omega(omega, float(dt))
    q = quat_normalize(np.asarray(q0, dtype=np.float64))
    for _ in range(int(n_steps)):
        q = quat_normalize(quat_multiply(step, q))
    return q


# ===== §2.5 板块角动量守恒 =====


def cap_angular_radius(area_m2: float, radius: float) -> float:
    """球冠面积 → 角半径 (rad)：``A = 2*pi*R^2*(1-cos(alpha))``（§2.5）。"""
    if area_m2 <= 0.0:
        raise ValueError(f"面积必须为正，实际为 {area_m2}")
    if radius <= 0.0:
        raise ValueError(f"半径必须为正，实际为 {radius}")
    cos_alpha = 1.0 - area_m2 / (2.0 * np.pi * radius**2)
    return float(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))


def cap_moment_of_inertia(
    angular_radius_rad: float,
    radius: float,
    density: float,
    thickness: float,
) -> float:
    """球冠（薄壳）绕其对称轴的转动惯量 (kg m^2)（§2.5）。

    对薄壳球冠做精确积分 ``I = 2*pi*rho*h*R^4 * (cos^3(a)/3 - cos(a) + 2/3)``；
    ``a = pi`` 时退化为整球薄壳 ``I = (2/3) m R^2``（``m = 4*pi*R^2*h*rho``），
    与教科书结果一致。方案 §2.5 写作 ``I ∝ R^5 * 密度``——那是把厚度也取成
    ``∝ R`` 的量纲约定；这里按 ``I = rho * h * R^4``（面积 ∝ R^2、力臂 ∝ R^2）
    实现，对**比值**无影响（角动量守恒只用得到比值）。
    """
    cos_a = np.cos(float(angular_radius_rad))
    integral = cos_a**3 / 3.0 - cos_a + 2.0 / 3.0
    return float(2.0 * np.pi * density * thickness * radius**4 * integral)


def merge_angular_momentum(omegas: np.ndarray, inertias: np.ndarray) -> np.ndarray:
    """角动量守恒合并板块：``omega_merged = (sum I_k w_k) / (sum I_k)``（§2.5）。

    ``omegas`` 形状 ``(n, 3)``、``inertias`` 形状 ``(n,)``。碰撞合并不能简单
    让板块"停止"，必须守恒角动量——这是 §2.5 的核心约束。
    """
    omegas = np.asarray(omegas, dtype=np.float64)
    inertias = np.asarray(inertias, dtype=np.float64)
    if omegas.shape[-1] != 3:
        raise ValueError(f"omegas 的最后一维必须为 3，实际 {omegas.shape}")
    if inertias.shape != omegas.shape[:-1]:
        raise ValueError(f"inertias 形状 {inertias.shape} 与 omegas {omegas.shape[:-1]} 不匹配")
    if np.any(inertias < 0.0):
        raise ValueError("转动惯量不能为负")
    total = float(inertias.sum())
    if total <= 0.0:
        raise ValueError("转动惯量之和必须为正")
    angular_momentum = np.einsum("n,nk->k", inertias, omegas)
    return np.asarray(angular_momentum / total, dtype=np.float64)


def collision_kinetic_budget(omegas: np.ndarray, inertias: np.ndarray) -> float:
    """碰撞消耗的动能 (J)（§2.5）：``dE = sum(0.5*I_k*|w_k|^2) - 0.5*I_tot*|w_merged|^2``。

    该项非负（完全非弹性碰撞的能量耗散），在管线中转为碰撞带地壳增厚的
    "势能预算"。
    """
    omegas = np.asarray(omegas, dtype=np.float64)
    inertias = np.asarray(inertias, dtype=np.float64)
    merged = merge_angular_momentum(omegas, inertias)
    before = float(0.5 * np.sum(inertias * np.sum(omegas**2, axis=-1)))
    after = float(0.5 * float(inertias.sum()) * float(np.sum(merged**2)))
    return before - after
