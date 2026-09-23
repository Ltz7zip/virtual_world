"""欧拉极点与板块运动学（《第一层完善》§2）。

刚体旋转模型：每个板块由其欧拉极点（旋转轴向）与角速度描述，
``v = omega x P``；相邻板块的相对速率 ``v_rel = (omega_A - omega_B) x P``，
与边界法向的内积符号决定边界类型（离散/汇聚/转换）。
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

#: 归一化阈值：|v_rel · n̂| < threshold * |v_rel| 判定为转换边界
TRANSFORM_THRESHOLD = 0.2


class BoundaryType(IntEnum):
    """板块边界类型（§2.3）。"""

    CONVERGENT = 0  # 汇聚（碰撞/俯冲）
    DIVERGENT = 1  # 离散（张裂）
    TRANSFORM = 2  # 转换（走滑）


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
