"""HEALPix 等面积层次化网格（方案《地形生成混合方案》§1.5 备选方案）。

HEALPix（Hierarchical Equal Area isoLatitude Pixelization）把球面分成 **12 个等面积
基础像素**（菱面体十二面体的 12 个面），每个基础像素再按四叉树细分，因此：

- **严格等面积**：单元面积恒为 :math:`4\\pi R^2/(12\\,\\mathrm{nside}^2)`，
  面积比恒为 1（等经纬网格在高纬趋于无穷），面积相关统计无需数值积分；
- **无极点奇点**：极点落在 4 个单元的交点而非单元内部；
- **层次化嵌套**：``nside`` 层像素 ``p`` 在 ``2*nside`` 层的 4 个子像素为
  ``4p, 4p+1, 4p+2, 4p+3``，粗化（限制算子）与细化（延长算子）都能精确到
  机器精度，天然支持多分辨率与 LOD；
- **拓扑规则**：每个单元有 4 个共边邻居 + 4 个共角邻居（极点像素少一个对角邻居）。

本实现只使用 **NESTED（嵌套）排序**——层次化与等面积性都依赖它；基础像素与
投影公式经与参考实现 ``astropy-healpix`` 逐像素比对：单元中心一致到 1e-16，
邻居集合完全一致（见 ``tests/unit/test_healpix.py``）。

平面投影（面内归一化坐标 ``X, Y in [0, 1]``，``s = X + Y``、``t = 2 - s``）：

- 赤道带面（4 个）：``z = (2/3)s - 2/3``，``phi = phi_face + 45(X - Y)``；
- 北面（4 个）：``s <= 1`` 时 ``z = (2/3)s``，否则 ``z = 1 - t^2/3``，
  ``phi = phi_face + 45(X - Y)/t``；
- 南面（4 个）：镜像北面（``t`` 与 ``s`` 互换），``phi`` 用 ``45(X - Y)/min(s, 1)``。

网格单元是平面上的正方形，投到球面后成为 ``(phi, z)`` 里的菱形，这也是
邻居判定用"共角计数"而不是"最近距离"的原因（参考实现的邻居同样不满足
8-最近邻假设）。
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from . import backend, operators, spherical
from .constants import EARTH_RADIUS

Order = Literal["nearest", "bilinear"]

#: 12 个基础像素中心经度（度）：北面 45/135/225/315，赤道带 0/90/180/270，南面同上
FACE_LON = np.array([45, 135, 225, 315, 0, 90, 180, 270, 45, 135, 225, 315], dtype=np.float64)
#: 每个基础像素所在的"带"：0 = 北面、1 = 赤道带面、2 = 南面
FACE_KIND = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8)
#: 共角键的量化容差（弧度/度）：远大于浮点误差、远小于单元尺度
_CORNER_TOL = 1e-9


def _freeze(array: np.ndarray) -> np.ndarray:
    """把缓存数组标记为只读，防止调用方就地改写污染缓存。"""
    array.flags.writeable = False
    return array


def _deinterleave(sub: np.ndarray, nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Morton 码（Z 序）解交织：偶数位给 ``ix``、奇数位给 ``iy``。"""
    levels = int(np.log2(nside))
    ix = np.zeros_like(sub)
    iy = np.zeros_like(sub)
    for bit in range(levels):
        ix |= ((sub >> (2 * bit)) & 1) << bit
        iy |= ((sub >> (2 * bit + 1)) & 1) << bit
    return ix, iy


def _north_z(s: np.ndarray, nside: float) -> np.ndarray:
    """北半球 ``z`` 随面内 ``s = X + Y`` 的分段规律（赤道带线性、极区二次）。"""
    return np.where(s <= 1.0, (2.0 / 3.0) * s, 1.0 - (2.0 - s) ** 2 / 3.0)


@lru_cache(maxsize=32)
def _face_xy(nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每个嵌套索引的 ``(face, ix, iy)``（只读缓存）。"""
    ipix = np.arange(12 * nside * nside, dtype=np.int64)
    face = ipix >> (2 * int(np.log2(nside)))
    ix, iy = _deinterleave(ipix & (nside * nside - 1), nside)
    return _freeze(face), _freeze(ix), _freeze(iy)


def _project(face: np.ndarray, x: np.ndarray, y: np.ndarray, nside: float) -> tuple[np.ndarray, np.ndarray]:
    """面内归一化坐标 ``(X, Y)`` -> ``(z, 经度)``（度）。"""
    kind = FACE_KIND[face]
    face_lon = FACE_LON[face]
    s = x + y
    t = 2.0 - s
    z = np.empty_like(s)
    phi = np.empty_like(s)

    band = kind == 1
    z[band] = (2.0 / 3.0) * s[band] - 2.0 / 3.0
    phi[band] = face_lon[band] + 45.0 * (x[band] - y[band])

    north = kind == 0
    low = north & (s <= 1.0)
    high = north & (s > 1.0)
    z[low] = (2.0 / 3.0) * s[low]
    phi[low] = face_lon[low] + 45.0 * (x[low] - y[low])
    z[high] = 1.0 - t[high] ** 2 / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # 极点角点处 t -> 0，phi 无意义；只求不报错，极点在 _corners 里被同一化
        phi[high] = face_lon[high] + 45.0 * (x[high] - y[high]) / t[high]

    south = kind == 2
    near = south & (s < 1.0)
    far = south & (s >= 1.0)
    z[near] = -(1.0 - s[near] ** 2 / 3.0)
    z[far] = -(2.0 / 3.0) * t[far]
    with np.errstate(divide="ignore", invalid="ignore"):
        # 极点角点处 s -> 0，phi 无意义；此处只求不报错，极点在 _corners 里被同一化
        scale = 1.0 / np.where(s < 1.0, np.where(s > 0.0, s, 1.0), 1.0)
    phi[south] = face_lon[south] + 45.0 * (x[south] - y[south]) * scale[south]
    return z, np.mod(phi, 360.0)


@lru_cache(maxsize=32)
def _centers(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """全部单元中心的 ``(z, 经度)``（只读缓存）。"""
    face, ix, iy = _face_xy(nside)
    z, lon = _project(face, (ix + 0.5) / nside, (iy + 0.5) / nside, float(nside))
    return _freeze(z), _freeze(lon)


@lru_cache(maxsize=32)
def _corners(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """每个单元的 4 个角点 ``(z, 经度)``，形状 ``(npix, 4)``（只读缓存）。"""
    face, ix, iy = _face_xy(nside)
    half = 0.5 / nside
    x = (ix + 0.5) / nside
    y = (iy + 0.5) / nside
    zs: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        z, lon = _project(face, x + dx * half, y + dy * half, float(nside))
        zs.append(z)
        lons.append(lon)
    return _freeze(np.stack(zs, axis=1)), _freeze(np.stack(lons, axis=1))


@lru_cache(maxsize=32)
def _topology(nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """共边/共角邻居表 ``(edge(4,npix), all(8,npix), mask(8,npix))``（只读缓存）。

    判定方式为**共角计数**：两个单元共 2 个角 ⇒ 共边（边邻居）；共 1 个角 ⇒
    仅共顶点（对角邻居）。极点处 4 个单元的公共角做同一化处理，因此极点像素
    自然得到 7 个邻居（缺一个对角），与参考实现一致。
    """
    if nside < 2:
        raise ValueError("邻居表要求 nside >= 2（nside=1 时单元过少，参考实现同样给出 -1）")
    npix = 12 * nside * nside
    z, lon = _corners(nside)
    pole = np.abs(z) >= 1.0 - 1e-12
    z = np.where(pole, np.sign(z), z)
    lon = np.where(pole, 0.0, lon)

    quant = np.stack(
        [
            np.round(z.ravel() / _CORNER_TOL).astype(np.int64),
            np.round(lon.ravel() / _CORNER_TOL).astype(np.int64),
        ],
        axis=1,
    )
    pixel_of_corner = np.repeat(np.arange(npix, dtype=np.int64), 4)
    order = np.lexsort((quant[:, 1], quant[:, 0]))
    sorted_q = quant[order]
    sorted_pixel = pixel_of_corner[order]
    changes = np.any(np.diff(sorted_q, axis=0) != 0, axis=1)
    group = np.concatenate([[0], np.cumsum(changes)])

    max_size = int(np.bincount(group).max())
    pairs: list[np.ndarray] = []
    for offset in range(1, max_size):
        same = group[offset:] == group[:-offset]
        if not np.any(same):
            continue
        left = sorted_pixel[:-offset][same]
        right = sorted_pixel[offset:][same]
        pairs.append(np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1))
    shared = np.concatenate(pairs, axis=0) if pairs else np.zeros((0, 2), dtype=np.int64)
    unique_pairs, counts = np.unique(shared, axis=0, return_counts=True)

    edge_pairs = unique_pairs[counts >= 2]
    diag_pairs = unique_pairs[counts == 1]
    edge_unsorted = _assemble(edge_pairs, npix, 4, exact=True)
    all_unsorted = _assemble(np.concatenate([edge_pairs, diag_pairs]), npix, 8, exact=False)
    edge = _order_by_azimuth(edge_unsorted, nside)
    all_nb = _order_by_azimuth(all_unsorted, nside)
    mask = np.zeros((8, npix), dtype=bool)
    for k in range(8):
        mask[k] = all_nb[k] < npix
    return _freeze(edge), _freeze(np.where(mask, all_nb, 0)), _freeze(mask)


def _assemble(pairs: np.ndarray, npix: int, degree: int, *, exact: bool) -> np.ndarray:
    """把无向邻居对装配成 ``(degree, npix)`` 索引表（槽位顺序无方向含义）。

    每条无向边展开成两个有向条目，按首索引排序后同一像素的邻居连续出现，
    于是槽位号 = 该条目在所属块内的序号；缺失槽位填哨兵 ``npix``
    （``exact=False`` 时允许少一个邻居，即 HEALPix 极点像素少一个对角邻居）。
    """
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]
    counts = np.bincount(directed[:, 0], minlength=npix)
    if exact:
        if not np.all(counts == degree):
            bad = int(np.argmax(counts != degree))
            raise ValueError(f"像素 {bad} 的邻居数为 {int(counts[bad])}，期望 {degree}")
    elif np.any(counts > degree) or np.any(counts < degree - 1):
        bad = int(np.argmax((counts > degree) | (counts < degree - 1)))
        raise ValueError(f"像素 {bad} 的邻居数为 {int(counts[bad])}，期望 {degree} 或 {degree - 1}")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(directed.shape[0]) - np.repeat(starts, counts)
    table = np.full((degree, npix), npix, dtype=np.int64)
    table[slot, directed[:, 0]] = directed[:, 1]
    return table


def _order_by_azimuth(table: np.ndarray, nside: int) -> np.ndarray:
    """把邻居表按**方位角递增**重排（自北起、向东为正），缺失槽位排在最后。

    HEALPix 的单元是 ``(phi, z)`` 中的菱形，4 个共边邻居在赤道附近约位于
    45/135/225/315 度方向，因此"东/北"式罗盘命名并不适用；这里只保证邻居按
    方位角逆时针排列，语义与立方球网格的 :meth:`CubedSphere.edge_neighbors`
    保持一致（便于跨网格统一处理）。
    """
    return operators.order_by_azimuth(_center_xyz(nside), table, invalid=12 * nside * nside)


def _center_xyz(nside: int) -> np.ndarray:
    """单元中心单位球面坐标（内部用，避免每次构造 ``HealpixGrid``）。"""
    z, lon = _centers(nside)
    cos_lat = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    return np.stack(
        [cos_lat * np.cos(np.radians(lon)), cos_lat * np.sin(np.radians(lon)), z], axis=-1
    )


@dataclasses.dataclass(frozen=True)
class HealpixGrid:
    """HEALPix 网格几何、拓扑与多分辨率算子（方案 §1.5 备选方案）。

    ``nside`` 必须为 2 的幂（嵌套四叉树层次化与 LOD 的前提）；``radius`` 为行星半径。
    """

    nside: int
    radius: float = EARTH_RADIUS

    def __post_init__(self) -> None:
        nside = int(self.nside)
        if nside < 1 or (nside & (nside - 1)) != 0:
            raise ValueError(f"nside 必须是 2 的幂，实际为 {self.nside}")
        if self.radius <= 0.0:
            raise ValueError(f"radius 必须为正，实际 {self.radius}")
        object.__setattr__(self, "nside", nside)

    # ===== 基本属性 =====

    @property
    def npix(self) -> int:
        """单元总数 ``12 * nside^2``。"""
        return 12 * self.nside * self.nside

    @property
    def size(self) -> int:
        """单元总数（与 :attr:`shape` 一致）。"""
        return self.npix

    @property
    def shape(self) -> tuple[int, ...]:
        """数据布局形状 ``(npix,)``。"""
        return (self.npix,)

    @property
    def n_level(self) -> int:
        """层次深度 ``log2(nside)``。"""
        return int(np.log2(self.nside))

    @property
    def total_area(self) -> float:
        """球面总面积 ``4*pi*R^2``。"""
        return 4.0 * np.pi * self.radius**2

    @property
    def resolution(self) -> float:
        """名义角分辨率（度）：``sqrt(单元面积)/R`` 对应的角度。"""
        return float(np.degrees(np.sqrt(self.total_area / self.npix) / self.radius))

    def cell_area(self) -> float:
        """单个单元面积 ``4*pi*R^2/npix``（严格等面积）。"""
        return self.total_area / self.npix

    def areas(self) -> np.ndarray:
        """全部单元面积 ``(npix,)``（等面积，方差为 0）。"""
        return np.full(self.npix, self.cell_area(), dtype=np.float64)

    def spacing(self) -> np.ndarray:
        """单元特征格距 ``(npix,)``：``sqrt(单元面积)``。"""
        return np.full(self.npix, np.sqrt(self.cell_area()), dtype=np.float64)

    def area_ratio(self) -> float:
        """最大/最小单元面积比，恒为 1（等面积性）。"""
        return 1.0

    # ===== 坐标 =====

    def centers_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """单元中心经纬度（度），形状 ``(npix,)``。"""
        z, lon = _centers(self.nside)
        lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        return lat, lon

    def centers_xyz(self) -> np.ndarray:
        """单元中心单位球面坐标，形状 ``(npix, 3)``。"""
        z, lon = _centers(self.nside)
        return spherical.latlon_to_xyz(
            np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))), lon, 1.0
        )

    def face_xy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """每个单元的基础像素号与面内格号 ``(face, ix, iy)``。"""
        return _face_xy(self.nside)

    # ===== 拓扑 =====

    def edge_neighbors(self) -> np.ndarray:
        """共边邻居扁平索引，形状 ``(4, npix)``，按方位角递增排序。"""
        return _topology(self.nside)[0]

    def neighbors(self) -> np.ndarray:
        """兼容命名：等价于 :meth:`edge_neighbors`（4 邻居）。"""
        return self.edge_neighbors()

    def all_neighbors(self) -> np.ndarray:
        """全部邻居扁平索引，形状 ``(8, npix)``，按方位角递增排序。

        含 4 个共边邻居与 4 个共角邻居；极点像素缺少一个共角邻居，缺失项由
        :meth:`neighbor_mask` 标出（索引回填 0）。
        """
        return _topology(self.nside)[1]

    def neighbor_mask(self) -> np.ndarray:
        """邻居有效性掩码 ``(8, npix)``：``False`` 表示该方向无邻居（极点像素）。"""
        return _topology(self.nside)[2]

    def shift(self, field: np.ndarray, direction: int, *, diagonal: bool = False) -> np.ndarray:
        """把 ``(..., npix)`` 字段沿第 ``direction`` 个邻居方向平移一格。"""
        values = np.asarray(field)
        if values.shape[-1] != self.npix:
            raise ValueError(f"字段末维应为 {self.npix}，实际 {values.shape}")
        table = self.all_neighbors() if diagonal else self.edge_neighbors()
        mask = self.neighbor_mask() if diagonal else np.ones_like(table, dtype=bool)
        if not 0 <= direction < table.shape[0]:
            raise ValueError(f"方向索引应在 [0, {table.shape[0]}) 内，实际 {direction}")
        if not np.all(mask[direction]):
            raise ValueError(f"方向 {direction} 在极点单元处无邻居，无法整体平移")
        return np.asarray(values[..., table[direction]], dtype=values.dtype)

    # ===== 层次化（多分辨率） =====

    @staticmethod
    def children(pixel: np.ndarray | int) -> np.ndarray:
        """下一层的 4 个子像素嵌套索引（``4p`` 至 ``4p + 3``）。"""
        index = np.asarray(pixel, dtype=np.int64)
        return np.stack([4 * index + k for k in range(4)], axis=-1)

    @staticmethod
    def parent(pixel: np.ndarray | int) -> np.ndarray:
        """上一层父像素嵌套索引。"""
        return np.asarray(pixel, dtype=np.int64) // 4

    def coarsen(self, field: np.ndarray, factor: int = 2) -> np.ndarray:
        """限制算子：按 ``factor`` 粗化一层（``factor`` 需为 2 的幂）。

        嵌套排序下粗像素 ``p`` 的子孙恰为连续区间 ``[p*factor^2, (p+1)*factor^2)``，
        且各单元面积相等，因此**分块均值即面积加权均值**——粗化前后全球平均严格一致。
        """
        factor = self._check_factor(factor)
        values = np.asarray(field)
        if values.shape[-1] != self.npix:
            raise ValueError(f"字段末维应为 {self.npix}，实际 {values.shape}")
        block = factor * factor
        reshaped = values.reshape(*values.shape[:-1], self.npix // block, block)
        return np.asarray(reshaped.mean(axis=-1), dtype=np.float64)

    def upsample(self, field: np.ndarray, factor: int = 2, *, smooth: bool = True) -> np.ndarray:
        """延长算子：按 ``factor`` 细化一层（``factor`` 需为 2 的幂）。

        ``smooth=True`` 用切平面局地线性重建（与立方球网格的语义一致）；
        ``smooth=False`` 为分层复制（HEALPix 标准 ``ud_grade`` 的细化语义，严格保守）。
        """
        factor = self._check_factor(factor)
        values = np.asarray(field)
        coarse_npix = self.npix // (factor * factor)
        if values.shape[-1] != coarse_npix:
            raise ValueError(f"字段末维应为粗网格的 {coarse_npix}，实际 {values.shape}")
        if not smooth:
            return np.asarray(np.repeat(values, factor * factor, axis=-1), dtype=np.float64)
        coarse = HealpixGrid(self.nside // factor, radius=self.radius)
        interpolator = operators.LocalInterpolator(coarse.centers_xyz(), k=6, radius=self.radius)
        return interpolator(values, self.centers_xyz())

    def _check_factor(self, factor: int) -> int:
        factor = int(factor)
        if factor < 2 or (factor & (factor - 1)) != 0:
            raise ValueError(f"factor 必须是 >= 2 的 2 的幂，实际为 {factor}")
        if factor > self.nside:
            raise ValueError(f"factor={factor} 不能超过 nside={self.nside}")
        return factor

    # ===== 与经纬网格互转 =====

    def latlon_to_cell(self, nlat: int, nlon: int) -> np.ndarray:
        """每个经纬格心最近的 HEALPix 单元索引，形状 ``(nlat, nlon)``。

        用最近中心判定（KD 树），因此分类/掩码场重网格时不会造出新类别。
        """
        lat, lon = _latlon_axes(nlat, nlon)
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
        return self.pixel_of_latlon(lat_2d.ravel(), lon_2d.ravel()).reshape(nlat, nlon)

    def pixel_of_latlon(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """任意经纬度查询点最近的单元扁平索引，形状与输入一致。"""
        query = spherical.latlon_to_xyz(lat, lon, 1.0).reshape(-1, 3)
        return self.interpolator().nearest_index(query).reshape(np.shape(lat))

    def to_latlon(
        self, field: np.ndarray, nlat: int, nlon: int, order: Order = "bilinear"
    ) -> np.ndarray:
        """HEALPix ``(..., npix)`` -> 经纬 ``(..., nlat, nlon)``。

        ``order="bilinear"`` 用切平面局地线性重建（光滑场二阶精度）；
        ``"nearest"`` 取最近单元（分类/掩码场专用）。
        """
        values = np.asarray(field)
        if values.shape[-1] != self.npix:
            raise ValueError(f"字段末维应为 {self.npix}，实际 {values.shape}")
        lat, lon = _latlon_axes(nlat, nlon)
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
        query = spherical.latlon_to_xyz(lat_2d, lon_2d, 1.0).reshape(-1, 3)
        interpolator = self.interpolator()
        if order == "nearest":
            index = interpolator.nearest_index(query)
            out = values[..., index]
        elif order == "bilinear":
            out = interpolator(values, query)
        else:
            raise ValueError(f"未知插值阶数 {order!r}（可选 'nearest' / 'bilinear'）")
        return np.asarray(out, dtype=np.float64).reshape(*values.shape[:-1], nlat, nlon)

    def from_latlon(self, field: np.ndarray, order: Order = "bilinear") -> np.ndarray:
        """经纬 ``(..., nlat, nlon)`` -> HEALPix ``(..., npix)``。

        在单元中心处对经纬网格做双线性（``"bilinear"``）或最近邻（``"nearest"``）
        取值；经度方向周期、纬度方向零阶外推。
        """
        values = np.asarray(field)
        if values.ndim < 2:
            raise ValueError(f"字段至少 2 维，实际 {values.shape}")
        nlat, nlon = values.shape[-2], values.shape[-1]
        lat, lon = self.centers_latlon()
        lead = values.shape[:-2]
        flat = values.reshape(-1, nlat, nlon)
        if order == "nearest":
            i0, j0, _, _ = _latlon_weights(lat, lon, nlat, nlon)
            out = flat[:, i0, j0]
        elif order == "bilinear":
            i0, j0, wi, wj = _latlon_weights(lat, lon, nlat, nlon)
            i1 = np.clip(i0 + 1, 0, nlat - 1)
            j1 = (j0 + 1) % nlon
            v00 = flat[:, i0, j0]
            v01 = flat[:, i0, j1]
            v10 = flat[:, i1, j0]
            v11 = flat[:, i1, j1]
            top = v00 * (1.0 - wj) + v01 * wj
            bottom = v10 * (1.0 - wj) + v11 * wj
            out = top * (1.0 - wi) + bottom * wi
        else:
            raise ValueError(f"未知插值阶数 {order!r}（可选 'nearest' / 'bilinear'）")
        return np.asarray(out, dtype=np.float64).reshape(*lead, self.npix)

    # ===== 诊断 =====

    def global_mean(self, field: Any) -> float:
        """面积加权全球平均（等面积网格即算术平均）。"""
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        return float(values.mean())

    def global_integral(self, field: Any) -> float:
        """面积加权全球积分。"""
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        return float(values.sum() * self.cell_area())

    def zonal_mean(self, field: Any, nbands: int | None = None) -> np.ndarray:
        """纬向平均剖面，长度 ``nbands``（默认按名义分辨率取整数条带）。"""
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        lat, _ = self.centers_latlon()
        bands = _band_count(self.resolution) if nbands is None else int(nbands)
        return spherical.zonal_mean_bands(values, lat, self.areas(), bands)

    # ===== 微分算子（切平面最小二乘） =====

    def stencil(self) -> operators.TangentStencil:
        """基于共边邻居的切平面算子（按 ``nside``/``radius`` 缓存，可反复复用）。"""
        return _stencil_for(self.nside, self.radius)

    def gradient(self, field: Any) -> tuple[Any, Any]:
        """标量场梯度 ``(df/dx, df/dy)``（东、北分量，1/m）。"""
        return self.stencil().gradient(field)

    def divergence(self, u: Any, v: Any) -> Any:
        """水平散度 ``du/dx + dv/dy``（1/s）。"""
        return self.stencil().divergence(u, v)

    def vorticity(self, u: Any, v: Any) -> Any:
        """相对涡度 ``dv/dx - du/dy``（1/s）。"""
        return self.stencil().vorticity(u, v)

    def laplacian(self, field: Any) -> Any:
        """拉普拉斯算子 ``div(grad f)``（1/m^2）。"""
        return self.stencil().laplacian(field)

    def interpolator(self, k: int = operators.DEFAULT_K) -> operators.LocalInterpolator:
        """格点场到任意球面点的局地线性插值器（按 ``nside``/``k``/``radius`` 缓存）。"""
        return _interpolator_for(self.nside, int(k), self.radius)


# ===== 模块级工具 =====


@lru_cache(maxsize=32)
def _stencil_for(nside: int, radius: float) -> operators.TangentStencil:
    """按 ``(nside, radius)`` 缓存的切平面算子（共边邻居，4 邻域）。"""
    grid = HealpixGrid(nside, radius=radius)
    return operators.TangentStencil(
        centers=grid.centers_xyz(), neighbors=grid.edge_neighbors(), radius=radius
    )


@lru_cache(maxsize=32)
def _interpolator_for(nside: int, k: int, radius: float) -> operators.LocalInterpolator:
    """按 ``(nside, k, radius)`` 缓存的局地线性插值器。"""
    grid = HealpixGrid(nside, radius=radius)
    return operators.LocalInterpolator(grid.centers_xyz(), k=k, radius=radius)


def _latlon_axes(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    if nlat <= 0 or nlon <= 0:
        raise ValueError("nlat 与 nlon 必须为正")
    return spherical.lat_centers(nlat), spherical.lon_centers(nlon)


def _latlon_weights(
    lat: np.ndarray, lon: np.ndarray, nlat: int, nlon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """经纬网格双线性权重：``lat``/``lon`` 为任意查询点（度）。"""
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    frac_i = (np.asarray(lat, dtype=np.float64) + 90.0) / dlat - 0.5
    frac_j = (np.asarray(lon, dtype=np.float64) + 180.0) / dlon - 0.5
    i0 = np.clip(np.floor(frac_i).astype(np.int64), 0, nlat - 1)
    j_floor = np.floor(frac_j).astype(np.int64)
    wi = np.clip(frac_i - i0, 0.0, 1.0)
    wj = frac_j - j_floor
    j0 = j_floor % nlon
    return i0, j0, wi, wj


def _band_count(resolution_deg: float) -> int:
    """按名义分辨率确定纬向条带数（每条约一个单元高）。"""
    return max(1, int(round(180.0 / max(resolution_deg, 1e-6))))


__all__ = [
    "FACE_KIND",
    "FACE_LON",
    "HealpixGrid",
]