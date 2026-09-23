"""非结构球面网格上的切平面算子（立方球与 HEALPix 共用）。

等经纬网格的差分算子（解析度规因子）在 :mod:`virtual_world.core.spherical`；
立方球与 HEALPix 这类**非正交/准规则**网格上，解析度规既繁琐又容易在面接缝与
菱形过渡处出错，因此本模块统一采用**切平面最小二乘重建**（least-squares
gradient，非结构网格数值方法的标准做法）：

1. 在每个单元中心建立局部正交标架：东 :math:`\\hat{e}`、北 :math:`\\hat{n}`；
2. 把邻居中心的位移投影到切平面，得到角位移分量 :math:`(a_k, b_k)`（弧度）与大圆角距；
3. 以反距离平方加权解 2x2 正规方程，重建标量梯度或矢量雅可比：

   .. math::
       \\min_{g}\\; \\sum_k w_k \\left(\\Delta f_k - a_k g_1 - b_k g_2\\right)^2 ,
       \\qquad w_k = 1/\\theta_k^2

   对切平面内**线性场**该重建精确（所有约束相容），故整体为二阶截断误差；
4. 由雅可比直接得到散度/涡度，由梯度的散度得到拉普拉斯算子。

单位与 :mod:`spherical` 一致：内部以弧度计算，输出再除以行星半径，因此
梯度/散度/涡度量纲分别为 ``1/m``、``1/s``、``1/s``。

同一套切平面重建也用于**局地线性插值**（:class:`LocalInterpolator`）：任意球面
查询点的插值、以及多分辨率的延长算子都基于它。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from . import backend
from .constants import EARTH_RADIUS

#: 局地线性重建默认使用的最邻近点数（3 个未知量：常数 + 两个梯度分量）
DEFAULT_K = 6
#: 角距下限，避免重合点导致权重发散
_MIN_THETA = 1e-9
#: 正规方程的正则化系数（相对量级），保证病态配置下仍可解
_RIDGE = 1e-12


# ===== 局部标架 =====


def tangent_basis(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """单元中心的东/北单位切向量，形状与 ``centers`` 一致 ``(..., 3)``。

    东向 :math:`\\hat{e} = \\hat{z} \\times \\hat{r}`；若中心恰好落在极点
    （单元中心一般不会）则退化为任意参考方向作数值兜底。
    """
    centers = np.asarray(centers, dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(z_axis, centers)
    norm = np.linalg.norm(east, axis=-1, keepdims=True)
    degenerate = norm[..., 0] < 1e-12
    if np.any(degenerate):
        fallback = np.cross(np.array([1.0, 0.0, 0.0]), centers)
        fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
        east = np.where(degenerate[..., None], fallback, east)
        norm = np.where(degenerate[..., None], fallback_norm, norm)
    east = east / np.where(norm > 0.0, norm, 1.0)
    north = np.cross(centers, east)
    return east, north


def tangent_offsets(
    anchors: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``targets`` 相对 ``anchors`` 的切平面位移。

    返回 ``(a, b, theta)``：切平面角位移分量（沿锚点东/北方向，弧度）与大圆角距。
    两个输入形状相同 ``(..., 3)``。
    """
    delta = targets - anchors
    chord = np.linalg.norm(delta, axis=-1)
    axial = np.einsum("...c,...c->...", delta, anchors)
    tangential = delta - axial[..., None] * anchors
    tnorm = np.linalg.norm(tangential, axis=-1)
    direction = tangential / np.where(tnorm > 0.0, tnorm, 1.0)[..., None]
    theta = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    east, north = tangent_basis(anchors)
    a = theta * np.einsum("...c,...c->...", direction, east)
    b = theta * np.einsum("...c,...c->...", direction, north)
    return a, b, theta


def order_by_azimuth(centers: np.ndarray, table: np.ndarray, invalid: int | None = None) -> np.ndarray:
    """把邻居索引表按**方位角递增**重排（自北起、向东为正）。

    ``table`` 形状 ``(K, N)``；``invalid`` 为缺失邻居的哨兵值（如 HEALPix 的
    ``npix``），这些条目排在各像素的末尾。方位角用切平面位移计算，因此对
    非正交网格同样适用。
    """
    centers = np.asarray(centers, dtype=np.float64)
    table = np.asarray(table, dtype=np.int64)
    size = centers.shape[0]
    degree = table.shape[0]
    if table.shape[1] != size:
        raise ValueError(f"邻居表第二维应为 {size}，实际 {table.shape}")
    valid = (table < size) if invalid is not None else np.ones_like(table, dtype=bool)
    anchor = np.broadcast_to(centers[:, None, :], (size, degree, 3))
    other = centers[np.where(valid, table, 0).T]
    a, b, _ = tangent_offsets(anchor, other)
    angle = np.where(valid.T, np.arctan2(b, a) % (2.0 * np.pi), np.inf)
    order = np.argsort(angle, axis=1, kind="stable")
    return np.ascontiguousarray(np.take_along_axis(table.T, order, axis=1).T)


@dataclass(frozen=True)
class TangentStencil:
    """切平面最小二乘算子：预计算 2x2 正规方程逆矩阵，逐场只需一次 gather。

    ``centers`` 形状 ``(N, 3)``（单位球面坐标）；``neighbors`` 形状 ``(K, N)``
    扁平索引，``-1`` 表示缺失（HEALPix 极点像素少一个对角邻居）；``radius``
    用于把角梯度/角速度换算为物理单位。
    """

    centers: np.ndarray
    neighbors: np.ndarray
    radius: float = EARTH_RADIUS

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        neighbors = np.asarray(self.neighbors, dtype=np.int64)
        if centers.ndim != 2 or centers.shape[-1] != 3:
            raise ValueError(f"centers 形状应为 (N, 3)，实际 {centers.shape}")
        if neighbors.ndim != 2 or neighbors.shape[-1] != centers.shape[0]:
            raise ValueError(
                f"neighbors 形状应为 (K, N) 且 N={centers.shape[0]}，实际 {neighbors.shape}"
            )
        if self.radius <= 0.0:
            raise ValueError(f"radius 必须为正，实际 {self.radius}")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "neighbors", neighbors)

    @property
    def size(self) -> int:
        """单元数 N。"""
        return int(self.centers.shape[0])

    @cached_property
    def offsets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """邻居切平面角位移 ``(a, b)`` 与有效性掩码，形状 ``(K, N)``。"""
        neighbors = self.neighbors
        valid = neighbors >= 0
        safe = np.where(valid, neighbors, 0)
        anchor = np.broadcast_to(self.centers[:, None, :], (self.size, neighbors.shape[0], 3))
        target = self.centers[safe].transpose(1, 0, 2)  # (N, K, 3)
        a, b, _ = tangent_offsets(anchor, target)
        valid_nk = valid.T
        a = np.where(valid_nk, a, 0.0).T
        b = np.where(valid_nk, b, 0.0).T
        return a, b, valid

    @cached_property
    def weights(self) -> np.ndarray:
        """反距离平方权重（缺失邻居权重为 0），形状 ``(K, N)``。"""
        a, b, valid = self.offsets
        theta2 = a * a + b * b
        return np.where(valid, 1.0 / np.maximum(theta2, _MIN_THETA), 0.0)

    @cached_property
    def inverse_matrix(self) -> np.ndarray:
        """正规方程矩阵的逆 ``A^{-1}``，形状 ``(N, 2, 2)``（对称 2x2 显式求逆）。"""
        a, b, _ = self.offsets
        w = self.weights
        saa = np.sum(w * a * a, axis=0)
        sbb = np.sum(w * b * b, axis=0)
        sab = np.sum(w * a * b, axis=0)
        det = saa * sbb - sab * sab
        if np.any(det <= _RIDGE):
            raise ValueError("正规方程奇异：邻居分布退化（检查 neighbors 自洽性）")
        inv = np.empty((self.size, 2, 2), dtype=np.float64)
        inv[:, 0, 0] = sbb / det
        inv[:, 1, 1] = saa / det
        inv[:, 0, 1] = -sab / det
        inv[:, 1, 0] = -sab / det
        return inv

    # ===== 内部工具 =====

    @staticmethod
    def _to_flat(values: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        field = np.asarray(backend.to_numpy(values), dtype=np.float64)
        if field.ndim == 0:
            raise ValueError("字段至少 1 维")
        return field.reshape(-1, field.shape[-1]), field.shape[:-1]

    @cached_property
    def transport(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """邻居标架到本单元标架的**平行移动**矩阵元素，形状均为 ``(K, N)``。

        矢量分量场 ``(u, v)`` 定义在各单元自己的东/北标架上，跨单元比较必须先把
        邻居矢量沿大圆平行移动到本单元（绕 ``x_c x x_k`` 轴转过大圆角距），否则
        会引入 O(1) 的标架旋转误差（散度/涡度无法收敛）。

        返回 ``(A, B, C, D)``，满足

        .. math::
            \\tilde u_k = A_k u_k + B_k v_k ,\\qquad
            \\tilde v_k = C_k u_k + D_k v_k .

        其中 ``A_k = e_k' \\cdot e_c``、``B_k = n_k' \\cdot e_c``、
        ``C_k = e_k' \\cdot n_c``、``D_k = n_k' \\cdot n_c``，撇号表示平行移动后的矢量。
        """
        neighbors = self.neighbors
        valid = neighbors >= 0
        safe = np.where(valid, neighbors, 0)
        centers = self.centers
        target = centers[safe].transpose(1, 0, 2)  # (N, K, 3)
        anchor = np.broadcast_to(centers[:, None, :], target.shape)
        delta = target - anchor
        theta = 2.0 * np.arcsin(np.clip(np.linalg.norm(delta, axis=-1) / 2.0, 0.0, 1.0))
        axis = np.cross(anchor, target)
        axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / np.where(axis_norm > 0.0, axis_norm, 1.0)
        east_k, north_k = tangent_basis(target)
        east_c, north_c = tangent_basis(anchor)

        def rotate(vector: np.ndarray) -> np.ndarray:
            """Rodrigues 旋转：绕 axis 转 theta（把 x_k 送到 x_c 的同一转动）。"""
            cos_t = np.cos(theta)[..., None]
            sin_t = np.sin(theta)[..., None]
            dot = np.sum(axis * vector, axis=-1, keepdims=True)
            return (
                vector * cos_t
                + np.cross(axis, vector) * sin_t
                + axis * dot * (1.0 - cos_t)
            )

        moved_east = rotate(east_k)
        moved_north = rotate(north_k)
        a = np.sum(moved_east * east_c, axis=-1)
        b = np.sum(moved_north * east_c, axis=-1)
        c = np.sum(moved_east * north_c, axis=-1)
        d = np.sum(moved_north * north_c, axis=-1)
        mask = valid.T
        zeros = np.zeros_like(a)

        def keep(mat: np.ndarray) -> np.ndarray:
            return np.where(mask, mat, zeros).T

        return keep(a), keep(b), keep(c), keep(d)

    def _neighbor_differences(self, values: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        """邻居减本单元 ``f_k - f``，形状 ``(K, M, N)``（``M`` 为前置维度展平）。"""
        flat, lead = self._to_flat(values)
        if flat.shape[-1] != self.size:
            raise ValueError(f"字段末维应为 {self.size}，实际 {values.shape}")
        safe = np.where(self.neighbors >= 0, self.neighbors, 0)
        gathered = flat[:, safe]  # (M, K, N)
        diffs = (gathered - flat[:, None, :]).transpose(1, 0, 2)
        return diffs, lead

    def _solve(self, diffs: np.ndarray) -> np.ndarray:
        """解最小二乘得 ``(g1, g2)``（每弧度），形状 ``(2, *lead, N)``。"""
        a, b, _ = self.offsets
        w = self.weights
        s1 = np.einsum("kn,kn,kmn->mn", w, a, diffs)
        s2 = np.einsum("kn,kn,kmn->mn", w, b, diffs)
        inv = self.inverse_matrix  # (N, 2, 2)
        g1 = inv[:, 0, 0][None, :] * s1 + inv[:, 0, 1][None, :] * s2
        g2 = inv[:, 1, 0][None, :] * s1 + inv[:, 1, 1][None, :] * s2
        return np.stack([g1, g2], axis=0)

    @staticmethod
    def _restore(reference: Any, values: np.ndarray, lead: tuple[int, ...]) -> Any:
        """把结果 reshape 回前置维度并搬回输入所在后端。"""
        out = values.reshape(2, *lead, values.shape[-1])
        xp = backend.backend_of(reference)
        if xp is np:
            return out
        return xp.array(out)

    # ===== 算子 =====

    def gradient(self, values: Any) -> tuple[Any, Any]:
        """标量场梯度 ``(df/dx, df/dy)``（东、北分量，单位 1/m）。"""
        diffs, lead = self._neighbor_differences(values)
        solved = self._solve(diffs) / self.radius
        both = self._restore(values, solved, lead)
        return both[0], both[1]

    def _vector_differences(self, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        """矢量分量场在邻居处的**平行移动后**差值 ``(du, dv)``，形状 ``(K, M, N)``。"""
        u_flat, lead_u = self._to_flat(u)
        v_flat, lead_v = self._to_flat(v)
        if lead_u != lead_v:
            raise ValueError(f"u 与 v 的前置维度不一致：{lead_u} vs {lead_v}")
        if u_flat.shape[-1] != self.size:
            raise ValueError(f"字段末维应为 {self.size}，实际 {u.shape}")
        a, b, c, d = self.transport
        safe = np.where(self.neighbors >= 0, self.neighbors, 0)
        u_nb = u_flat[:, safe]  # (L, K, N)
        v_nb = v_flat[:, safe]
        du = (u_nb * a + v_nb * b - u_flat[:, None, :]).transpose(1, 0, 2)
        dv = (u_nb * c + v_nb * d - v_flat[:, None, :]).transpose(1, 0, 2)
        return du, dv, lead_u

    def jacobian(self, u: Any, v: Any) -> tuple[Any, Any, Any, Any]:
        """矢量场（东/北分量）的局地雅可比 ``(du/dx, du/dy, dv/dx, dv/dy)``，单位 1/s。

        邻居矢量先沿大圆**平行移动**到本单元标架（见 :attr:`transport`），因此标架
        旋转不会污染结果；切平面内线性场仍可精确重现。
        """
        du, dv, lead = self._vector_differences(u, v)
        solved_u = self._restore(u, self._solve(du) / self.radius, lead)
        solved_v = self._restore(v, self._solve(dv) / self.radius, lead)
        return solved_u[0], solved_u[1], solved_v[0], solved_v[1]

    def divergence(self, u: Any, v: Any) -> Any:
        """水平散度 ``du/dx + dv/dy``（单位 1/s）。

        球面标架的度规项由**平行移动**隐式包含（见 :attr:`transport`）——邻居矢量先
        搬到本单元标架再做差分，因此不需要额外的解析度规项；这与
        :func:`~virtual_world.core.spherical.divergence` 的 ``cos(phi)`` 度规等价。
        注意常数**分量**场在球面上并非无散场：``div(0, v) = -v tan(phi)/R``。
        """
        du_dx, _, _, dv_dy = self.jacobian(u, v)
        return du_dx + dv_dy

    def vorticity(self, u: Any, v: Any) -> Any:
        """相对涡度 ``dv/dx - du/dy``（单位 1/s），度规项同样由平行移动包含。"""
        _, du_dy, dv_dx, _ = self.jacobian(u, v)
        return dv_dx - du_dy

    def laplacian(self, values: Any) -> Any:
        """拉普拉斯算子 ``div(grad f)``（单位 1/m^2）。"""
        gx, gy = self.gradient(values)
        return self.divergence(gx, gy)

    def gradient_magnitude(self, values: Any) -> Any:
        """梯度模 ``|grad f|``（单位 1/m）。"""
        gx, gy = self.gradient(values)
        xp = backend.backend_of(gx)
        return xp.sqrt(gx * gx + gy * gy)


class LocalInterpolator:
    """k 近邻切平面线性重建：把格点数据插值到任意球面查询点。

    对每个查询点取最近的 ``k`` 个源单元中心，在**最近源中心**的切平面内做线性
    最小二乘重建（常数 + 两个梯度分量），再在查询点处取值；权重取反距离平方，
    并加极小正则项以容忍退化配置。对光滑场为二阶插值，且不会产生新类别值
    （分类场请在外层改用最近邻）。
    """

    def __init__(self, centers: np.ndarray, k: int = DEFAULT_K, radius: float = 1.0) -> None:
        from scipy.spatial import (
            cKDTree,  # type: ignore[import-untyped]  # 延迟导入，保持 core 轻量
        )

        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[-1] != 3:
            raise ValueError(f"centers 形状应为 (N, 3)，实际 {centers.shape}")
        if k < 3:
            raise ValueError(f"k 至少为 3（常数 + 两个梯度分量），实际 {k}")
        self.centers = centers
        self.k = min(int(k), int(centers.shape[0]))
        self.radius = float(radius)
        self._tree = cKDTree(centers)

    def nearest_index(self, query: np.ndarray) -> np.ndarray:
        """查询点的最近源单元扁平索引，形状 ``(Q,)``。"""
        query = self._normalize(query)
        _, index = self._tree.query(query)
        return np.asarray(index, dtype=np.int64)

    @staticmethod
    def _normalize(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        norm = np.linalg.norm(query, axis=-1, keepdims=True)
        return query / np.where(norm > 0.0, norm, 1.0)

    def _design(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造 (设计矩阵, 查询点相对锚点的偏移, 最近索引) 供线性重建使用。

        设计矩阵形状 ``(Q, k, 3)``，行 = ``[1, a, b]``（常数 + 两个梯度分量）；
        偏移形状 ``(Q, 2)``，用于把拟合出的平面**在查询点处求值**（只返回常数项
        等于返回锚点值，会损失一阶精度）。权重取常数 1：**切平面内线性场在任意
        权重下都被精确重现**，而反距离权重会在查询点与源中心重合时发散。
        """
        query = self._normalize(query)
        _, index = self._tree.query(query, k=self.k)
        index = np.asarray(index, dtype=np.int64)
        safe = np.where(index >= 0, index, 0)
        anchor = self.centers[index[:, 0]]
        a, b, _ = tangent_offsets(anchor[:, None, :], self.centers[safe])
        design = np.stack([np.ones_like(a), a, b], axis=-1)
        qa, qb, _ = tangent_offsets(anchor, query)
        return design, np.stack([qa, qb], axis=-1), index

    def __call__(self, values: Any, query: np.ndarray) -> np.ndarray:
        """把 ``(..., N)`` 的格点场插值到 ``(Q, 3)`` 查询点，返回 ``(..., Q)`` NumPy 数组。"""
        field = np.asarray(backend.to_numpy(values), dtype=np.float64)
        if field.shape[-1] != self.centers.shape[0]:
            raise ValueError(f"字段末维应为 {self.centers.shape[0]}，实际 {field.shape}")
        lead = field.shape[:-1]
        flat = field.reshape(-1, field.shape[-1])

        design, query_offsets, index = self._design(query)
        normal = np.einsum("qki,qkj->qij", design, design)  # (Q, 3, 3)
        trace = np.einsum("qii->q", normal) / 3.0
        ridge = _RIDGE * np.maximum(trace, _MIN_THETA)
        normal = normal + ridge[:, None, None] * np.eye(3)[None, :, :]
        sampled = flat[:, index.reshape(-1)].reshape(-1, index.shape[0], index.shape[1])
        rhs = np.einsum("qki,mqk->mqi", design, sampled)  # (M, Q, 3)
        solution = np.linalg.solve(normal[None, :, :, :], rhs[..., None])[..., 0]  # (M, Q, 3)
        # 在查询点处求值：f(q) = c0 + a_q * g1 + b_q * g2
        out = solution[..., 0] + solution[..., 1] * query_offsets[:, 0] + solution[..., 2] * query_offsets[:, 1]
        return np.asarray(out.reshape(*lead, index.shape[0]), dtype=np.float64)