"""切平面最小二乘算子测试（:mod:`virtual_world.core.operators`）。

测试网格用立方球 :class:`CubedSphere`：``z = sin(lat)`` 与刚体旋转场在球面上
有闭式梯度/散度/涡度/拉普拉斯，可用于检验二阶精度与网格加密的收敛行为。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from virtual_world.core import operators
from virtual_world.core.constants import EARTH_RADIUS
from virtual_world.core.cubed_sphere import CubedSphere

#: 共边（4 邻域）与共角（对角）格偏移
_EDGE_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL_OFFSETS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_R = EARTH_RADIUS
_OMEGA = 7.292e-5  # 刚体旋转角速度 (1/s)


@lru_cache(maxsize=8)
def _grid(n_side: int) -> CubedSphere:
    """按 ``n_side`` 缓存的立方球网格（几何构造较贵，测试内复用）。"""
    return CubedSphere(n_side)


@lru_cache(maxsize=8)
def _centers(n_side: int) -> np.ndarray:
    """单元中心单位向量，形状 ``(N, 3)``。"""
    return _grid(n_side).centers_xyz().reshape(-1, 3)


@lru_cache(maxsize=8)
def _lat_deg(n_side: int) -> np.ndarray:
    """单元中心纬度 (deg)，形状 ``(N,)``。"""
    return np.rad2deg(np.arcsin(np.clip(_centers(n_side)[:, 2], -1.0, 1.0)))


@lru_cache(maxsize=8)
def _stencil(n_side: int, diagonal: bool = False) -> operators.TangentStencil:
    """立方球上的切平面算子；``diagonal=True`` 时额外纳入 4 个共角邻居。"""
    sphere = _grid(n_side)
    offsets = _EDGE_OFFSETS + _DIAGONAL_OFFSETS if diagonal else _EDGE_OFFSETS
    table = [sphere.neighbor_index(di, dj).ravel() for di, dj in offsets]
    return operators.TangentStencil(_centers(n_side), np.stack(table, axis=0), _R)


def _rotation(n_side: int) -> tuple[np.ndarray, np.ndarray]:
    """刚体旋转的东/北分量：``u = Omega*R*cos(lat)``、``v = 0``。"""
    u = _OMEGA * _R * np.cos(np.deg2rad(_lat_deg(n_side)))
    return u, np.zeros_like(u)


def _gradient_error(n_side: int) -> np.ndarray:
    """``z`` 的梯度误差（乘回 R，无量纲）：``(0, cos(lat))`` 为解析值。"""
    gx, gy = _stencil(n_side).gradient(_centers(n_side)[:, 2])
    return np.hypot(gx * _R, gy * _R - np.cos(np.deg2rad(_lat_deg(n_side))))


def _query_points(count: int, seed: int = 20240923) -> np.ndarray:
    """球面随机查询点，形状 ``(count, 3)``。"""
    points = np.random.default_rng(seed).normal(size=(count, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


# ===== 线性场精确性 =====


@pytest.mark.parametrize("diagonal", [False, True])
def test_linear_field_in_tangent_plane_is_reproduced_exactly(diagonal: bool) -> None:
    """切平面内线性场被最小二乘精确重现（4 邻域与 8 邻域都成立）。

    用**同一套** ``tangent_offsets`` 构造邻居处的场值 ``f_k = G1*a_k + G2*b_k``，
    则参考单元处的全部约束相容，重建梯度应精确等于 ``(G1, G2)``（每弧度）。
    """
    stencil = _stencil(8, diagonal)
    a, b, _ = stencil.offsets
    for cell in (0, 100, stencil.size // 2, stencil.size - 1):
        field = np.zeros(stencil.size)
        field[stencil.neighbors[:, cell]] = 1.0e-6 * a[:, cell] - 2.0e-6 * b[:, cell]
        gx, gy = stencil.gradient(field)
        assert gx[cell] * _R == pytest.approx(1.0e-6, rel=1e-12, abs=0.0)
        assert gy[cell] * _R == pytest.approx(-2.0e-6, rel=1e-12, abs=0.0)


# ===== 解析标量场 =====


def test_gradient_of_z_matches_analytic_value() -> None:
    """n_side=32：``z = sin(lat)`` 的梯度在东/北方向为 ``(0, cos(lat)/R)``。"""
    stencil = _stencil(32)
    field = _centers(32)[:, 2]
    gx, gy = stencil.gradient(field)
    cos_lat = np.cos(np.deg2rad(_lat_deg(32)))
    assert np.abs(gy * _R - cos_lat).max() < 5.0e-3
    assert np.abs(gx * _R).max() < 5.0e-3
    assert np.sqrt(np.mean((gy * _R - cos_lat) ** 2 + (gx * _R) ** 2)) < 1.0e-3
    assert np.allclose(stencil.gradient_magnitude(field), np.hypot(gx, gy), rtol=1e-15)


def test_gradient_error_shrinks_second_order() -> None:
    """网格加密一倍，梯度 RMS 误差至少缩小 1.6 倍（内部二阶收敛）。"""
    coarse = float(np.sqrt(np.mean(_gradient_error(32) ** 2)))
    fine = float(np.sqrt(np.mean(_gradient_error(64) ** 2)))
    assert fine < coarse / 1.6


# ===== 散度与涡度 =====


def test_solid_body_rotation_divergence_and_vorticity() -> None:
    """刚体旋转无辐散（``|div|/Omega < 5e-2``）且涡度为 ``2*Omega*sin(lat)``。"""
    u, v = _rotation(32)
    stencil = _stencil(32)
    z = _centers(32)[:, 2]
    assert np.abs(stencil.divergence(u, v)).max() / _OMEGA < 5.0e-2
    assert np.abs(stencil.vorticity(u, v) - 2.0 * _OMEGA * z).max() / (2.0 * _OMEGA) < 1.0e-2


def test_divergence_and_vorticity_converge_on_refinement() -> None:
    """散度/涡度误差随网格加密单调减小（二阶算子）。"""
    error = []
    for n_side in (32, 64):
        u, v = _rotation(n_side)
        stencil = _stencil(n_side)
        error.append((np.abs(stencil.divergence(u, v)).max(),
                      np.abs(stencil.vorticity(u, v) - 2.0 * _OMEGA * _centers(n_side)[:, 2]).max()))
    (div_coarse, vort_coarse), (div_fine, vort_fine) = error
    assert div_fine < div_coarse / 1.5
    assert vort_fine < vort_coarse / 1.5


def test_laplacian_of_z_matches_analytic_value() -> None:
    """``laplacian(z) = -2z/R^2``（球面拉普拉斯含因子 2），且全球平均为 0。"""
    stencil = _stencil(32)
    z = _centers(32)[:, 2]
    lap = stencil.laplacian(z)
    assert np.abs(lap + 2.0 * z / _R**2).max() * _R**2 < 0.4
    areas = _grid(32).areas().ravel()
    assert abs(float(np.sum(lap * areas) / np.sum(areas))) * _R**2 < 1.0e-3


# ===== 缺失邻居与形状校验 =====


def test_missing_neighbor_entries_are_tolerated() -> None:
    """``-1`` 标记的缺失邻居按 0 权重处理，结果与完整邻居表几乎相同。"""
    n_side = 16
    neighbors = _grid(n_side).neighbors().copy()
    neighbors[-1, 0] = -1
    z = _centers(n_side)[:, 2]
    full = _stencil(n_side).gradient(z)
    patched = operators.TangentStencil(_centers(n_side), neighbors, _R)
    assert np.abs(patched.gradient(z)[0] - full[0]).max() < 1.0e-6
    assert np.abs(patched.gradient(z)[1] - full[1]).max() < 1.0e-6
    assert np.isfinite(patched.divergence(z, z)).all()
    assert np.isfinite(patched.laplacian(z)).all()


def test_stencil_rejects_invalid_configuration() -> None:
    """``centers``/``neighbors`` 形状不合法或半径非正时抛 ``ValueError``。"""
    centers = _centers(8)
    neighbors = _grid(8).neighbors()
    with pytest.raises(ValueError):
        operators.TangentStencil(centers[:, :2], neighbors)
    with pytest.raises(ValueError):
        operators.TangentStencil(centers, neighbors[:, :-1])
    with pytest.raises(ValueError):
        operators.TangentStencil(centers, neighbors.ravel())
    with pytest.raises(ValueError):
        operators.TangentStencil(centers, neighbors, radius=0.0)


def test_operators_reject_mismatched_field_lengths() -> None:
    """字段末维不等于单元数时抛 ``ValueError``；``jacobian`` 还校验前置维度。"""
    stencil = _stencil(8)
    with pytest.raises(ValueError):
        stencil.gradient(np.zeros(stencil.size + 1))
    with pytest.raises(ValueError):
        stencil.laplacian(np.zeros(stencil.size + 1))
    with pytest.raises(ValueError):
        stencil.divergence(np.zeros(stencil.size), np.zeros(stencil.size + 1))
    with pytest.raises(ValueError):
        stencil.jacobian(np.zeros((2, stencil.size)), np.zeros(stencil.size))


def test_operators_support_leading_dimensions() -> None:
    """字段可带前置维度：``(2, N) -> (2, N)``，常数分量梯度为 0。"""
    stencil = _stencil(8)
    z = _centers(8)[:, 2]
    field = np.stack([z, np.full(stencil.size, 7.0)], axis=0)
    gx, gy = stencil.gradient(field)
    assert gx.shape == field.shape and gy.shape == field.shape
    assert np.abs(gx[1]).max() == 0.0 and np.abs(gy[1]).max() == 0.0
    assert np.array_equal(gx[0], stencil.gradient(z)[0])


# ===== 局部标架与缓存 =====


def test_tangent_basis_and_offsets_are_consistent() -> None:
    """局部标架正交单位；``tangent_offsets`` 的角距等于大圆角距。"""
    centers = _centers(8)
    east, north = operators.tangent_basis(centers)
    assert np.abs(np.einsum("ij,ij->i", east, centers)).max() < 1e-15
    assert np.abs(np.einsum("ij,ij->i", north, centers)).max() < 1e-15
    assert np.abs(np.einsum("ij,ij->i", east, north)).max() < 1e-15
    assert np.allclose(np.linalg.norm(east, axis=1), 1.0)
    anchor = centers[0][None, :]
    target = centers[_grid(8).neighbors()[0, 0]][None, :]
    a, b, theta = operators.tangent_offsets(anchor, target)
    expected = np.arccos(np.clip(float(anchor.ravel() @ target.ravel()), -1.0, 1.0))
    assert theta[0] == pytest.approx(expected, abs=1e-12)
    assert np.hypot(a[0], b[0]) == pytest.approx(expected, rel=1e-12)
    a0, b0, theta0 = operators.tangent_offsets(anchor, anchor)
    assert max(np.abs(a0).max(), np.abs(b0).max(), np.abs(theta0).max()) == 0.0


def test_stencil_results_are_deterministic_and_symmetric() -> None:
    """同一几何构造的两个算子结果逐位相同，且正规方程逆矩阵对称。"""
    first = operators.TangentStencil(_centers(8).copy(), _grid(8).neighbors().copy())
    second = operators.TangentStencil(_centers(8), _grid(8).neighbors())
    z = _centers(8)[:, 2]
    assert np.array_equal(first.inverse_matrix, second.inverse_matrix)
    assert np.array_equal(first.gradient(z)[0], second.gradient(z)[0])
    inverse = first.inverse_matrix
    assert np.abs(inverse - inverse.transpose(0, 2, 1)).max() == 0.0
    assert operators.DEFAULT_K == 6


# ===== 局地线性插值 =====


def test_local_interpolator_nearest_index_is_identity_on_own_centers() -> None:
    """以网格自身中心为查询点时，最近邻索引是恒等排列。"""
    interpolator = operators.LocalInterpolator(_centers(16))
    assert np.array_equal(
        interpolator.nearest_index(_centers(16)), np.arange(interpolator.centers.shape[0])
    )


def test_local_interpolator_reproduces_smooth_linear_field() -> None:
    """``z``、``x`` 在格距尺度上近似切平面线性，插值误差应为二阶（< 1e-3）。"""
    centers = _centers(64)
    interpolator = operators.LocalInterpolator(centers, k=6)
    query = _query_points(400)
    assert np.abs(interpolator(centers[:, 2], query) - query[:, 2]).max() < 1.0e-3
    assert np.abs(interpolator(centers[:, 0], query) - query[:, 0]).max() < 1.0e-3


def test_local_interpolator_constant_and_leading_dimensions() -> None:
    """常数场插值回常数；``(2, N)`` 字段输出 ``(2, Q)``。"""
    centers = _centers(16)
    interpolator = operators.LocalInterpolator(centers)
    query = _query_points(50)
    assert np.abs(interpolator(np.full(centers.shape[0], 3.5), query) - 3.5).max() < 1.0e-10
    batch = np.stack([centers[:, 2], np.full(centers.shape[0], 2.0)], axis=0)
    out = interpolator(batch, query)
    assert out.shape == (2, query.shape[0])
    assert np.abs(out[1] - 2.0).max() < 1.0e-10


def test_local_interpolator_rejects_invalid_input() -> None:
    """``k < 3``、``centers`` 形状错误、字段末维不匹配时抛 ``ValueError``。"""
    centers = _centers(8)
    with pytest.raises(ValueError):
        operators.LocalInterpolator(centers, k=2)
    with pytest.raises(ValueError):
        operators.LocalInterpolator(centers.ravel())
    with pytest.raises(ValueError):
        operators.LocalInterpolator(centers)(np.zeros(centers.shape[0] + 1), _query_points(3))