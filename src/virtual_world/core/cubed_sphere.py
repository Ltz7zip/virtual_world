"""立方球网格（Cubed-Sphere，方案《地形生成混合方案》§1.5）。

把球面投影到立方体的 6 个面上、每面用规则 ``n_side x n_side`` 网格离散化，
解决等经纬度网格的三个硬伤：

- 高纬度经线收敛导致单元面积趋近于零（数值不稳定、虚假极地聚集）；
- 极点奇点；
- 单元面积比无穷大。

本实现采用**等角投影**（equiangular / 广义 gnomonic）：面内坐标
``u = tan(xi)``、``xi = s * pi/4``、``s in [-1, 1]``，使面积比显著优于线性 gnomonic。
单元面积在球面上按球面三角形超量**精确**计算，因此 6 面求和严格等于 ``4*pi*R^2``。

拓扑：立方球上每个单元都恰好有 4 个邻居（无极点、无"域外"）。邻接**不靠硬编码的
12 条棱配对规则**，而是几何求解——把面内坐标外推一格得到球面点，再反查它落在哪个面
及其面内索引。这样接缝与角点（三个面交汇处）自动正确，不需要为 8 个角点写特例。

数据布局：字段形状 ``(6, n_side, n_side)``（方案 §2.2 的"face 索引在最外层"），
与 :mod:`virtual_world.core.grid` 的 ``(lat, lon)`` 布局通过 :meth:`~CubedSphere.to_latlon`
/ :meth:`~CubedSphere.from_latlon` 双向插值桥接。
"""

from __future__ import annotations

import dataclasses
import math
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from . import operators, spherical
from .constants import EARTH_RADIUS
from .spherical import latlon_to_xyz

#: 立方体 6 个面的编号顺序：+x, -x, +y, -y, +z, -z
FACE_CENTERS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
#: 各面的"东"方向（面内 u 轴）
FACE_EAST = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
#: 各面的"北"方向（面内 v 轴）；三者满足 east x north = center（右手系，无镜像）
FACE_NORTH = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

#: 邻接方向顺序：0=北(di+1)、1=南(di-1)、2=东(dj+1)、3=西(dj-1)
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIRECTION_INDEX = {d: k for k, d in enumerate(DIRECTIONS)}


def _spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray, radius: float) -> np.ndarray:
    """单位向量球面三角形的面积 (m^2)：``R^2 * E``，``E`` 为球面超量。"""
    numerator = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    denominator = 1.0 + np.einsum("...i,...i->...", a, b) + np.einsum("...i,...i->...", b, c) + np.einsum(
        "...i,...i->...", c, a
    )
    excess = 2.0 * np.arctan2(numerator, denominator)
    return radius**2 * excess


def _band_count(resolution_deg: float) -> int:
    """按名义分辨率确定纬向条带数（每条约一个单元高）。"""
    return max(1, int(round(180.0 / max(resolution_deg, 1e-6))))


@lru_cache(maxsize=32)
def _edge_table(n_side: int, equiangular: bool, legacy_order: bool) -> np.ndarray:
    """共边邻居表 ``(4, 6*n*n)``：``legacy_order`` 保持 :data:`DIRECTIONS` 顺序。"""
    sphere = CubedSphere(n_side, equiangular=equiangular)
    stacked = np.stack([sphere.neighbor_index(di, dj).ravel() for di, dj in DIRECTIONS], axis=0)
    if legacy_order:
        return stacked
    return operators.order_by_azimuth(sphere.centers_xyz().reshape(-1, 3), stacked)


@lru_cache(maxsize=32)
def _all_table(n_side: int, equiangular: bool) -> np.ndarray:
    """8-邻域表 ``(8, 6*n*n)``：4 个共边 + 4 个共角，按方位角递增排序。"""
    offsets = DIRECTIONS + ((1, 1), (1, -1), (-1, -1), (-1, 1))
    sphere = CubedSphere(n_side, equiangular=equiangular)
    stacked = np.stack([sphere.neighbor_index(di, dj).ravel() for di, dj in offsets], axis=0)
    return operators.order_by_azimuth(sphere.centers_xyz().reshape(-1, 3), stacked)


@lru_cache(maxsize=32)
def _stencil_for(n_side: int, radius: float, equiangular: bool) -> operators.TangentStencil:
    """按网格参数缓存的切平面算子（共边邻居）。"""
    sphere = CubedSphere(n_side, radius=radius, equiangular=equiangular)
    return operators.TangentStencil(
        centers=sphere.centers_xyz().reshape(-1, 3),
        neighbors=sphere.edge_neighbors(),
        radius=radius,
    )


@lru_cache(maxsize=32)
def _interpolator_for(
    n_side: int, radius: float, equiangular: bool, k: int
) -> operators.LocalInterpolator:
    """按网格参数缓存的局地线性插值器。"""
    sphere = CubedSphere(n_side, radius=radius, equiangular=equiangular)
    return operators.LocalInterpolator(sphere.centers_xyz().reshape(-1, 3), k=k, radius=radius)


@dataclasses.dataclass(frozen=True)
class CubedSphere:
    """立方球网格几何与拓扑（方案 §1.5）。

    ``n_side`` 为每个面的边长（单元数），总单元数 ``6 * n_side^2``；
    ``radius`` 为行星半径；``equiangular=True`` 使用等角投影（面积更均匀）。
    """

    n_side: int
    radius: float = EARTH_RADIUS
    equiangular: bool = True

    def __post_init__(self) -> None:
        if self.n_side < 1:
            raise ValueError(f"n_side 必须为正，实际为 {self.n_side}")
        if self.radius <= 0.0:
            raise ValueError(f"radius 必须为正，实际为 {self.radius}")

    # ===== 基本属性 =====

    @property
    def shape(self) -> tuple[int, int, int]:
        return (6, self.n_side, self.n_side)

    @property
    def size(self) -> int:
        return 6 * self.n_side * self.n_side

    @property
    def total_area(self) -> float:
        """球面总面积 (m^2)，等于 ``4*pi*R^2``。"""
        return 4.0 * math.pi * self.radius**2

    @property
    def resolution(self) -> float:
        """等面积等效名义角分辨率（度）：``sqrt(平均单元面积)/R``。

        与 :attr:`HealpixGrid.resolution` 采用同一口径，便于跨网格比较与按分辨率
        自动确定纬向条带数。等角投影下单元面积比约 1.5，故该值略小于面内格距
        ``90/n_side``。
        """
        return float(np.degrees(np.sqrt(self.total_area / self.size) / self.radius))

    # ===== 坐标 =====

    def _to_cube(self, s: np.ndarray) -> np.ndarray:
        """面内归一化坐标 ``s in [-1,1]`` → 立方体面坐标（等角时为 tan 变换）。"""
        s = np.asarray(s, dtype=np.float64)
        if not self.equiangular:
            return s
        return np.tan(s * (np.pi / 4.0))

    def _to_normalized(self, u: np.ndarray) -> np.ndarray:
        """立方体面坐标 → 面内归一化坐标（:meth:`_to_cube` 的逆）。"""
        u = np.asarray(u, dtype=np.float64)
        if not self.equiangular:
            return u
        return np.arctan(u) * (4.0 / np.pi)

    def centers_s(self) -> np.ndarray:
        """单元中心的归一化面内坐标 ``s``，形状 ``(n_side,)``。"""
        return -1.0 + 2.0 * (np.arange(self.n_side, dtype=np.float64) + 0.5) / self.n_side

    def centers_xyz(self) -> np.ndarray:
        """单元中心的单位球面坐标，形状 ``(6, n_side, n_side, 3)``。"""
        n = self.n_side
        s = self._to_cube(self.centers_s())
        # u 随面内 j 增大（向东），v 随面内 i 增大（向北）
        u = np.broadcast_to(s[None, None, :], (6, n, n))[..., None]
        v = np.broadcast_to(s[None, :, None], (6, n, n))[..., None]
        points = (
            FACE_CENTERS[:, None, None, :]
            + u * FACE_EAST[:, None, None, :]
            + v * FACE_NORTH[:, None, None, :]
        )
        norm = np.linalg.norm(points, axis=-1, keepdims=True)
        return np.asarray(points / norm, dtype=np.float64)

    def centers_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """单元中心经纬度 (deg)，各自形状 ``(6, n_side, n_side)``。"""
        xyz = self.centers_xyz()
        z = np.clip(xyz[..., 2], -1.0, 1.0)
        lat = np.rad2deg(np.arcsin(z))
        lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
        return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)

    # ===== 面积 =====

    def areas(self) -> np.ndarray:
        """单元面积 (m^2)，形状 ``(6, n_side, n_side)``（球面三角形超量，精确）。"""
        n = self.n_side
        edges = self._to_cube(-1.0 + 2.0 * np.arange(n + 1, dtype=np.float64) / n)
        # 单元 (i, j)：u 由 edges[j]→edges[j+1]（东），v 由 edges[i]→edges[i+1]（北）
        u_lo = np.broadcast_to(edges[None, :-1], (1, n, n))[..., None]
        u_hi = np.broadcast_to(edges[None, 1:], (1, n, n))[..., None]
        v_lo = np.broadcast_to(edges[:-1, None], (1, n, n))[..., None]
        v_hi = np.broadcast_to(edges[1:, None], (1, n, n))[..., None]

        def corner(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
            points = (
                FACE_CENTERS[:, None, None, :]
                + uu * FACE_EAST[:, None, None, :]
                + vv * FACE_NORTH[:, None, None, :]
            )
            norm = np.linalg.norm(points, axis=-1, keepdims=True)
            return np.asarray(points / norm, dtype=np.float64)

        p00 = corner(u_lo, v_lo)
        p01 = corner(u_hi, v_lo)
        p11 = corner(u_hi, v_hi)
        p10 = corner(u_lo, v_hi)
        first = _spherical_triangle_area(p00, p01, p11, self.radius)
        second = _spherical_triangle_area(p00, p11, p10, self.radius)
        return np.asarray(first + second, dtype=np.float64)

    def area_ratio(self) -> float:
        """最大/最小单元面积比（等经纬网格为无穷大，立方球应接近 1）。"""
        areas = self.areas()
        return float(areas.max() / areas.min())

    def spacing(self) -> np.ndarray:
        """单元特征格距 (m)，形状同 :meth:`areas`：``sqrt(单元面积)``。"""
        return np.asarray(np.sqrt(self.areas()), dtype=np.float64)

    # ===== 拓扑 =====

    def _lookup(self, face: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """面 ``face`` 上立方体坐标 ``(u, v)``（可在面外）→ ``(面, i, j)``。

        先由面心 + ``u*east + v*north`` 得到球面点，再用"与哪个面心点积最大"确定所属面，
        最后解出该面的面内坐标并取整。越出面上的点会自然落到相邻面（含角点三分面情形）。
        """
        face = np.asarray(face, dtype=np.int64)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        center = FACE_CENTERS[face]
        east = FACE_EAST[face]
        north = FACE_NORTH[face]
        point = center + u[..., None] * east + v[..., None] * north
        point = point / np.linalg.norm(point, axis=-1, keepdims=True)

        dots = point @ FACE_CENTERS.T
        target = np.argmax(dots, axis=-1)
        scale = np.take_along_axis(dots, target[..., None], axis=-1)[..., 0]
        target_east = FACE_EAST[target]
        target_north = FACE_NORTH[target]
        local_u = np.einsum("...i,...i->...", point, target_east) / scale
        local_v = np.einsum("...i,...i->...", point, target_north) / scale

        s_u = self._to_normalized(local_u)
        s_v = self._to_normalized(local_v)
        n = self.n_side
        j = np.clip(np.floor((s_u + 1.0) / 2.0 * n).astype(np.int64), 0, n - 1)
        i = np.clip(np.floor((s_v + 1.0) / 2.0 * n).astype(np.int64), 0, n - 1)
        return np.asarray(target, dtype=np.int64), i, j

    def neighbor_index(self, di: int, dj: int) -> np.ndarray:
        """偏移 ``(di, dj)``（南北/东西，单位：格）的邻居扁平索引，形状 ``(6, n, n)``。

        ``di = +1`` 向北、``dj = +1`` 向东；``|di|, |dj| <= 1``。索引与
        :meth:`centers_xyz` 的 ``(6, n, n)`` 布局一致，按 ``.ravel()`` 取用。
        """
        if abs(di) > 1 or abs(dj) > 1:
            raise ValueError(f"仅支持单格偏移，实际 ({di}, {dj})")
        n = self.n_side
        faces = np.arange(6, dtype=np.int64)[:, None, None] * np.ones((6, n, n), dtype=np.int64)
        s_j = self.centers_s()[None, None, :] + np.ones((6, n, 1)) * (2.0 * dj / n)
        s_i = self.centers_s()[None, :, None] + np.ones((6, 1, n)) * (2.0 * di / n)
        u = self._to_cube(s_j) * np.ones((6, n, n))
        v = self._to_cube(s_i) * np.ones((6, n, n))
        target, i, j = self._lookup(faces, u, v)
        return (target * (n * n) + i * n + j).astype(np.int64)

    def neighbors(self) -> np.ndarray:
        """4-邻域扁平索引，形状 ``(4, 6*n*n)``，顺序见 :data:`DIRECTIONS`（历史 API）。"""
        return _edge_table(self.n_side, self.equiangular, legacy_order=True)

    def edge_neighbors(self) -> np.ndarray:
        """共边邻居扁平索引，形状 ``(4, 6*n*n)``，按方位角递增排序。

        与 :class:`~virtual_world.core.healpix.HealpixGrid` 的同名方法语义一致
        （统一按方位角逆时针排列），便于跨网格统一处理；:meth:`neighbors` 保留
        方案早期的"南北东西"顺序供 terrain 层的既有代码使用。
        """
        return _edge_table(self.n_side, self.equiangular, legacy_order=False)

    def all_neighbors(self) -> np.ndarray:
        """含对角的 8-邻域扁平索引，形状 ``(8, 6*n*n)``，按方位角递增排序。

        含 4 个共边邻居与 4 个共角邻居（立方球上每个单元都有 8 个邻居）；
        排序只保证逆时针方位角顺序，共边/共角并不按位置分组。
        """
        return _all_table(self.n_side, self.equiangular)

    def shift(self, field: np.ndarray, di: int, dj: int) -> np.ndarray:
        """把 ``(..., 6, n, n)`` 字段按 ``(di, dj)`` 平移一格（接缝自动处理）。"""
        values = np.asarray(field)
        if values.shape[-3:] != self.shape:
            raise ValueError(f"字段末三维应为 {self.shape}，实际 {values.shape}")
        lead = values.shape[:-3]
        flat = np.asarray(values).reshape(*lead, self.size)
        shifted = flat[..., self.neighbor_index(di, dj).ravel()]
        return np.asarray(shifted.reshape(*lead, *self.shape), dtype=values.dtype)

    # ===== 层次化（多分辨率） =====

    def coarsen(self, field: np.ndarray, factor: int = 2) -> np.ndarray:
        """限制算子：每个面内按 ``factor x factor`` 块做**面积加权**平均（保守）。

        粗网格 ``n//factor`` 与细网格的块边界对齐，因此全球面积加权平均在粗化前后
        严格一致（单元面积不全相等，必须按面积加权）。
        """
        factor = self._check_factor(factor, require_divisor=True)
        values = np.asarray(field, dtype=np.float64)
        if values.shape[-3:] != self.shape:
            raise ValueError(f"字段末三维应为 {self.shape}，实际 {values.shape}")
        n = self.n_side // factor
        blocks = values.reshape(*values.shape[:-3], 6, n, factor, n, factor)
        weights = self.areas().reshape(6, n, factor, n, factor)
        # 下标必须互不相同：einsum 中同一字母重复出现 4 次属未定义行为
        numerator = np.einsum("...pifjg,pifjg->...pij", blocks, weights)
        denominator = weights.sum(axis=(2, 4))
        return np.asarray(numerator / denominator, dtype=np.float64)

    def upsample(self, field: np.ndarray, factor: int = 2, *, smooth: bool = True) -> np.ndarray:
        """延长算子：把粗网格 ``(6, n/factor, n/factor)`` 细化到本网格 ``(6, n, n)``。

        ``smooth=True`` 用切平面局地线性插值（光滑延长）；``smooth=False`` 用块复制
        （严格保守的延长，与 HEALPix 分层细化语义一致）。
        """
        factor = self._check_factor(factor, require_divisor=True)
        coarse_side = self.n_side // factor
        values = np.asarray(field, dtype=np.float64)
        if values.shape[-3:] != (6, coarse_side, coarse_side):
            raise ValueError(
                f"字段末三维应为 {(6, coarse_side, coarse_side)}（粗网格），实际 {values.shape}"
            )
        if not smooth:
            return np.asarray(
                np.repeat(np.repeat(values, factor, axis=-2), factor, axis=-1), dtype=np.float64
            )
        interpolator = _interpolator_for(coarse_side, self.radius, self.equiangular, 6)
        flat = values.reshape(-1, 6 * coarse_side * coarse_side)
        out = interpolator(flat, self.centers_xyz().reshape(-1, 3))
        return np.asarray(out.reshape(*values.shape[:-3], 6, self.n_side, self.n_side), dtype=np.float64)

    def _check_factor(self, factor: int, *, require_divisor: bool) -> int:
        factor = int(factor)
        if factor < 2:
            raise ValueError(f"factor 必须 >= 2，实际为 {factor}")
        if require_divisor and self.n_side % factor:
            raise ValueError(f"factor={factor} 必须整除 n_side={self.n_side}")
        return factor

    # ===== 统计与诊断 =====

    def _flatten(self, field: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        """把 ``(..., 6, n, n)`` 字段展平为 ``(M, size)``（算子与统计的统一入口）。"""
        values = np.asarray(field, dtype=np.float64)
        if values.shape[-3:] != self.shape:
            raise ValueError(f"字段末三维应为 {self.shape}，实际 {values.shape}")
        return values.reshape(-1, self.size), values.shape[:-3]

    def global_mean(self, field: Any) -> float:
        """面积加权全球平均（立方球单元面积不等，必须按面积加权）。"""
        flat, _ = self._flatten(field)
        return spherical.global_mean(flat, self.areas().ravel())

    def global_integral(self, field: Any) -> float:
        """面积加权全球积分。"""
        flat, _ = self._flatten(field)
        return spherical.global_integral(flat, self.areas().ravel())

    def zonal_mean(self, field: Any, nbands: int | None = None) -> np.ndarray:
        """按纬度条带的面积加权平均，返回 ``(..., nbands)`` 剖面。"""
        flat, lead = self._flatten(field)
        lat = self.centers_latlon()[0].ravel()
        bands = _band_count(self.resolution) if nbands is None else int(nbands)
        profile = spherical.zonal_mean_bands(flat, lat, self.areas().ravel(), bands)
        return np.asarray(profile.reshape(*lead, bands), dtype=np.float64)

    # ===== 微分算子（切平面最小二乘） =====

    def stencil(self) -> operators.TangentStencil:
        """基于共边邻居的切平面算子（按网格参数缓存，可反复复用）。"""
        return _stencil_for(self.n_side, self.radius, self.equiangular)

    def gradient(self, field: Any) -> tuple[Any, Any]:
        """标量场梯度 ``(df/dx, df/dy)``（东、北分量，1/m），形状与输入一致。"""
        flat, lead = self._flatten(field)
        gx, gy = self.stencil().gradient(flat)
        return (
            np.asarray(gx).reshape(*lead, *self.shape),
            np.asarray(gy).reshape(*lead, *self.shape),
        )

    def divergence(self, u: Any, v: Any) -> Any:
        """水平散度 ``du/dx + dv/dy``（1/s），形状与输入一致。"""
        flat_u, lead = self._flatten(u)
        flat_v, _ = self._flatten(v)
        out = self.stencil().divergence(flat_u, flat_v)
        return np.asarray(out).reshape(*lead, *self.shape)

    def vorticity(self, u: Any, v: Any) -> Any:
        """相对涡度 ``dv/dx - du/dy``（1/s），形状与输入一致。"""
        flat_u, lead = self._flatten(u)
        flat_v, _ = self._flatten(v)
        out = self.stencil().vorticity(flat_u, flat_v)
        return np.asarray(out).reshape(*lead, *self.shape)

    def laplacian(self, field: Any) -> Any:
        """拉普拉斯算子 ``div(grad f)``（1/m^2），形状与输入一致。"""
        flat, lead = self._flatten(field)
        out = self.stencil().laplacian(flat)
        return np.asarray(out).reshape(*lead, *self.shape)

    def interpolator(self, k: int = operators.DEFAULT_K) -> operators.LocalInterpolator:
        """格点场到任意球面点的局地线性插值器（按网格参数缓存）。"""
        return _interpolator_for(self.n_side, self.radius, self.equiangular, int(k))

    def pixel_of_latlon(self, lat: Any, lon: Any) -> np.ndarray:
        """任意经纬度查询点最近的单元扁平索引，形状与输入一致。"""
        query = latlon_to_xyz(lat, lon, 1.0).reshape(-1, 3)
        return self.interpolator().nearest_index(query).reshape(np.shape(lat))

    # ===== 与经纬网格互转 =====

    def to_latlon(self, field: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
        """立方球 ``(6, n, n)`` → 经纬 ``(nlat, nlon)``（面内双线性，接缝由几何解析）。"""
        values = np.asarray(field, dtype=np.float64)
        if values.shape[-3:] != self.shape:
            raise ValueError(f"字段末三维应为 {self.shape}，实际 {values.shape}")
        lat = -90.0 + (180.0 / nlat) * (np.arange(nlat, dtype=np.float64) + 0.5)
        lon = -180.0 + (360.0 / nlon) * (np.arange(nlon, dtype=np.float64) + 0.5)
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
        xyz = latlon_to_xyz(lat_2d, lon_2d, 1.0)

        dots = xyz.reshape(-1, 3) @ FACE_CENTERS.T
        face = np.argmax(dots, axis=1)
        scale = dots[np.arange(dots.shape[0]), face]
        point = xyz.reshape(-1, 3)
        local_u = np.einsum("ij,ij->i", point, FACE_EAST[face]) / scale
        local_v = np.einsum("ij,ij->i", point, FACE_NORTH[face]) / scale
        n = self.n_side
        frac_j = (self._to_normalized(local_u) + 1.0) / 2.0 * n - 0.5
        frac_i = (self._to_normalized(local_v) + 1.0) / 2.0 * n - 0.5

        i0 = np.floor(frac_i).astype(np.int64)
        j0 = np.floor(frac_j).astype(np.int64)
        wi = frac_i - i0
        wj = frac_j - j0

        def corner(ioff: np.ndarray, joff: np.ndarray) -> np.ndarray:
            # 用**单元中心**的面内坐标作几何查找（而不是格边）：越界的角点因此落在
            # 相邻面的边界单元中心上，与 :meth:`neighbor_index` 的构造完全一致。
            s_i = -1.0 + 2.0 * (i0 + ioff + 0.5) / n
            s_j = -1.0 + 2.0 * (j0 + joff + 0.5) / n
            target, ci, cj = self._lookup(face, self._to_cube(s_j), self._to_cube(s_i))
            gathered = values.reshape(-1, self.size)[:, target * (n * n) + ci * n + cj]
            return np.asarray(gathered, dtype=np.float64)

        zeros = np.zeros_like(i0)
        ones = np.ones_like(i0)
        v00 = corner(zeros, zeros)
        v01 = corner(zeros, ones)
        v10 = corner(ones, zeros)
        v11 = corner(ones, ones)
        top = v00 * (1.0 - wj) + v01 * wj
        bottom = v10 * (1.0 - wj) + v11 * wj
        merged = top * (1.0 - wi) + bottom * wi
        return np.asarray(merged.reshape(values.shape[:-3] + (nlat, nlon)), dtype=np.float64)

    def latlon_to_cell(self, nlat: int, nlon: int) -> np.ndarray:
        """每个经纬格心所在的立方球单元**扁平索引**，形状 ``(nlat, nlon)``。

        用单元中心的最近邻（KD 树）判定，因此编码/掩码类场重网格时不会造出新类别。
        """
        if nlat <= 0 or nlon <= 0:
            raise ValueError("nlat 与 nlon 必须为正")
        lat = -90.0 + (180.0 / nlat) * (np.arange(nlat, dtype=np.float64) + 0.5)
        lon = -180.0 + (360.0 / nlon) * (np.arange(nlon, dtype=np.float64) + 0.5)
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
        queries = latlon_to_xyz(lat_2d, lon_2d, 1.0).reshape(-1, 3)
        tree = cKDTree(self.centers_xyz().reshape(-1, 3))
        _, index = tree.query(queries)
        return np.asarray(index, dtype=np.int64).reshape(nlat, nlon)

    def from_latlon(self, field: np.ndarray, *, periodic_lon: bool = True) -> np.ndarray:
        """经纬 ``(..., nlat, nlon)`` → 立方球 ``(6, n, n)``（双线性，经向周期）。"""
        values = np.asarray(field, dtype=np.float64)
        if values.ndim < 2:
            raise ValueError(f"字段至少 2 维，实际 {values.shape}")
        nlat, nlon = values.shape[-2], values.shape[-1]
        lat, lon = self.centers_latlon()
        flat_lat = lat.ravel()
        flat_lon = lon.ravel()

        dlat = 180.0 / nlat
        dlon = 360.0 / nlon
        frac_i = (flat_lat + 90.0) / dlat - 0.5
        frac_j = (flat_lon + 180.0) / dlon - 0.5
        i0 = np.clip(np.floor(frac_i).astype(np.int64), 0, nlat - 1)
        i1 = np.clip(i0 + 1, 0, nlat - 1)
        wi = np.clip(frac_i - i0, 0.0, 1.0)
        j_floor = np.floor(frac_j).astype(np.int64)
        wj = frac_j - j_floor
        if periodic_lon:
            j0 = j_floor % nlon
            j1 = (j0 + 1) % nlon
        else:
            j0 = np.clip(j_floor, 0, nlon - 1)
            j1 = np.clip(j0 + 1, 0, nlon - 1)

        lead = values.shape[:-2]
        source = values.reshape(-1, nlat, nlon)
        v00 = source[:, i0, j0]
        v01 = source[:, i0, j1]
        v10 = source[:, i1, j0]
        v11 = source[:, i1, j1]
        top = v00 * (1.0 - wj) + v01 * wj
        bottom = v10 * (1.0 - wj) + v11 * wj
        merged = top * (1.0 - wi) + bottom * wi
        return np.asarray(merged.reshape(*lead, *self.shape), dtype=np.float64)


__all__ = ["DIRECTIONS", "FACE_CENTERS", "FACE_EAST", "FACE_NORTH", "CubedSphere"]
