"""三种球面网格的统一几何抽象：等经纬度 / 立方球 / HEALPix。

:class:`~virtual_world.core.grid.GridState` 通过本模块的几何对象访问网格能力
（单元面积、单元坐标、邻域查找、多分辨率、跨网格重网格与微分算子），因此
"下游阶段在任一网格上跑"只需要换一个几何实现，数据模型与管线代码无需分支。

三个实现都把重活委托给各自的模块：

============  ====================  =========================
网格类型      几何与拓扑            重网格 / 多分辨率
============  ====================  =========================
``latlon``    :mod:`.spherical`     本模块（解析度规、保守块平均）
``cubed_sphere`` :class:`.CubedSphere`  面内块平均 / 平面线性插值
``healpix``   :class:`.HealpixGrid`  分层块平均 / 分层细化与线性插值
============  ====================  =========================

所有几何都实现同一接口，约定：

- ``shape`` 为**水平形状**（经纬 ``(nlat, nlon)``、立方球 ``(6, n, n)``、HEALPix ``(npix,)``），
  字段末尾若干维与之对齐（见 :attr:`GridGeometry.layout_ndim`）；
- ``areas()`` 与 ``shape`` 同形，单位 m²，``total_area()`` 为其全球和；
- ``to_latlon`` / ``from_latlon`` 用 ``order="bilinear"``（连续场）或 ``"nearest"``
  （分类/掩码场）在经纬网格与非结构网格之间转换，是三者互转的统一桥梁。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from . import backend, spherical
from .constants import EARTH_RADIUS
from .cubed_sphere import CubedSphere
from .healpix import HealpixGrid

Order = str
#: 非插值场（分类/掩码）在重网格与粗化时的取值方式
_NEAREST_ORDERS = {"nearest"}


def _nominal_bands(resolution_deg: float) -> int:
    """按名义分辨率确定纬向条带数（每条约一个单元高）。"""
    return max(1, int(round(180.0 / max(resolution_deg, 1e-6))))


class GridGeometry(ABC):
    """水平网格几何的统一接口（三种网格对等能力的汇聚点）。"""

    grid_type: ClassVar[str]
    #: 水平形状占用的末尾维度数（经纬 2、立方球 3、HEALPix 1）
    layout_ndim: ClassVar[int] = 2

    # ===== 元数据 =====

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """水平网格形状。"""

    @property
    @abstractmethod
    def radius(self) -> float:
        """行星半径 (m)。"""

    @property
    @abstractmethod
    def resolution(self) -> float:
        """等面积等效名义角分辨率（度）：``sqrt(平均单元面积)/R``。"""

    @property
    def size(self) -> int:
        """水平单元数。"""
        return int(np.prod(self.shape))

    @property
    def is_global(self) -> bool:
        """是否全球网格（三种网格的默认构造都是全球网格）。"""
        return True

    @abstractmethod
    def areas(self) -> np.ndarray:
        """单元面积 (m²)，形状同 :attr:`shape`。"""

    def total_area(self) -> float:
        """全球总面积 (m²)。"""
        return float(np.sum(self.areas()))

    @abstractmethod
    def cell_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """每个单元的经纬度（度），**展平**为长度 ``size`` 的一维数组。"""

    # ===== 布局转换与查找 =====

    @abstractmethod
    def lookup(self, lat: float, lon: float) -> int:
        """最近单元的**扁平**索引（按 ``shape`` 的 C 序展平）。"""

    def flat_index(self, indices: tuple[int, ...]) -> int:
        """把网格下标元组转换为扁平索引。"""
        if len(indices) != len(self.shape):
            raise ValueError(f"下标维度应为 {len(self.shape)}，实际 {indices}")
        return int(np.ravel_multi_index(indices, self.shape))

    # ===== 与经纬网格互转 =====

    @abstractmethod
    def to_latlon(
        self, field: Any, nlat: int, nlon: int, order: Order = "bilinear"
    ) -> np.ndarray:
        """网格布局 ``(..., *shape)`` -> 经纬 ``(..., nlat, nlon)``。"""

    @abstractmethod
    def from_latlon(self, field: Any, order: Order = "bilinear") -> np.ndarray:
        """经纬 ``(..., nlat, nlon)`` -> 网格布局 ``(..., *shape)``。"""

    # ===== 多分辨率 =====

    @abstractmethod
    def coarsen_geometry(self, factor: int) -> GridGeometry:
        """粗化后的网格几何（不含场数据，供构造新 :class:`GridState` 使用）。"""

    @abstractmethod
    def coarsen(self, field: Any, factor: int) -> tuple[GridGeometry, Any]:
        """限制算子：返回 ``(粗网格几何, 粗化后的场)``（保守，面积加权）。"""

    @abstractmethod
    def subsample(self, field: Any, factor: int) -> np.ndarray:
        """分类/布尔场的粗化：取每个粗单元的代表细单元值（不产生新类别）。"""

    # ===== 统计 =====

    def zonal_mean(self, field: Any, nbands: int | None = None) -> np.ndarray:
        """按纬度条带的面积加权平均，返回 ``(..., nbands)``。"""
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        lat, _ = self.cell_latlon()
        bands = _nominal_bands(self.resolution) if nbands is None else int(nbands)
        flat = values.reshape(-1, self.size)
        profile = spherical.zonal_mean_bands(flat, lat, self.areas().ravel(), bands)
        return np.asarray(profile.reshape(*values.shape[: -self.layout_ndim], bands))

    # ===== 微分算子 =====

    @abstractmethod
    def gradient(self, field: Any) -> tuple[Any, Any]:
        """标量场梯度 ``(df/dx, df/dy)``（东、北分量，1/m），形状与输入一致。"""

    @abstractmethod
    def divergence(self, u: Any, v: Any) -> Any:
        """水平散度 ``du/dx + dv/dy``（1/s）。"""

    @abstractmethod
    def vorticity(self, u: Any, v: Any) -> Any:
        """相对涡度 ``dv/dx - du/dy``（1/s）。"""

    @abstractmethod
    def laplacian(self, field: Any) -> Any:
        """拉普拉斯算子 ``div(grad f)``（1/m²）。"""

    # ===== 序列化 =====

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """重建该几何所需的形状参数（供 :meth:`GridState.to_dict` 使用）。

        不含 ``radius``——半径由所属网格状态的行星参数决定。
        """


# ===== 等经纬度网格 =====


@dataclass(frozen=True)
class LatLonGeometry(GridGeometry):
    """等经纬度网格几何（解析度规，见 :mod:`virtual_world.core.spherical`）。"""

    grid_type: ClassVar[str] = "latlon"
    layout_ndim: ClassVar[int] = 2

    nlat: int
    nlon: int
    #: 构造时可选传入的坐标轴；解析后的轴见 :attr:`lat` / :attr:`lon`
    lat_input: np.ndarray | None = None
    lon_input: np.ndarray | None = None
    radius: float = EARTH_RADIUS

    def __post_init__(self) -> None:
        if self.nlat <= 0 or self.nlon <= 0:
            raise ValueError("nlat 与 nlon 必须为正整数")
        lat = (
            spherical.lat_centers(self.nlat)
            if self.lat_input is None
            else np.asarray(self.lat_input, dtype=np.float64)
        )
        lon = (
            spherical.lon_centers(self.nlon)
            if self.lon_input is None
            else np.asarray(self.lon_input, dtype=np.float64)
        )
        if lat.shape != (self.nlat,) or lon.shape != (self.nlon,):
            raise ValueError("lat/lon 长度必须与 nlat/nlon 一致")
        object.__setattr__(self, "_lat", lat)
        object.__setattr__(self, "_lon", lon)

    @property
    def lat(self) -> np.ndarray:
        """纬度中心轴（度，升序）。"""
        return self._lat  # type: ignore[attr-defined]

    @property
    def lon(self) -> np.ndarray:
        """经度中心轴（度，升序）。"""
        return self._lon  # type: ignore[attr-defined]

    # ===== 元数据 =====

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.nlat, self.nlon)

    @property
    def resolution(self) -> float:
        """经纬向格距（度）：取自实际坐标轴，因此区域切片网格也是局部格距。"""
        lat = self.lat
        if lat is not None and lat.size >= 2:
            return float(abs(np.mean(np.diff(lat))))
        return 360.0 / self.nlon

    @property
    def lat_edges(self) -> np.ndarray:
        return spherical.lat_edges(self.nlat)

    @property
    def lon_edges(self) -> np.ndarray:
        return spherical.lon_edges(self.nlon)

    @property
    def cos_lat(self) -> np.ndarray:
        return spherical.cos_lat_weights(self.lat)

    @property
    def is_global(self) -> bool:
        """纬度覆盖 [-90, 90] 且经度覆盖 360 度。"""
        return bool(
            abs(float(self.lat[0]) + 90.0) < self.resolution
            and abs(float(self.lat[-1]) - 90.0) < self.resolution
            and self.nlon * self.resolution >= 360.0 - 1e-6
        )

    def areas(self) -> np.ndarray:
        return spherical.cell_areas(self.lat_edges, self.nlon, self.radius)

    def cell_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        lat_2d, lon_2d = np.meshgrid(self.lat, self.lon, indexing="ij")
        return np.asarray(lat_2d).ravel(), np.asarray(lon_2d).ravel()

    # ===== 查找与变换 =====

    def lookup(self, lat: float, lon: float) -> int:
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(((self.lon - lon + 180.0) % 360.0) - 180.0)))
        return int(np.ravel_multi_index((i, j), self.shape))

    def to_latlon(self, field: Any, nlat: int, nlon: int, order: Order = "bilinear") -> np.ndarray:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        if (nlat, nlon) == self.shape:
            return values
        from .interpolation import regrid

        return regrid(
            values,
            self.lat,
            self.lon,
            spherical.lat_centers(nlat),
            spherical.lon_centers(nlon),
            order="nearest" if order in _NEAREST_ORDERS else "bilinear",
            src_lat_edges=self.lat_edges,
            src_lon_edges=self.lon_edges,
        )

    def from_latlon(self, field: Any, order: Order = "bilinear") -> np.ndarray:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        src_lat = spherical.lat_centers(values.shape[-2])
        src_lon = spherical.lon_centers(values.shape[-1])
        if (values.shape[-2], values.shape[-1]) == self.shape:
            return values
        from .interpolation import regrid

        return regrid(
            values,
            src_lat,
            src_lon,
            self.lat,
            self.lon,
            order="nearest" if order in _NEAREST_ORDERS else "bilinear",
            src_lat_edges=spherical.lat_edges(values.shape[-2]),
            src_lon_edges=spherical.lon_edges(values.shape[-1]),
        )

    def coarsen_geometry(self, factor: int) -> GridGeometry:
        factor = int(factor)
        if factor < 1:
            raise ValueError("factor 必须 >= 1")
        if self.nlat % factor or self.nlon % factor:
            raise ValueError(
                f"factor={factor} 必须同时整除 nlat={self.nlat} 与 nlon={self.nlon}"
            )
        if factor == 1:
            return self
        return LatLonGeometry(
            nlat=self.nlat // factor,
            nlon=self.nlon // factor,
            lat_input=_coords_from_edges(self.lat_edges[::factor]),
            lon_input=_coords_from_edges(self.lon_edges[::factor]),
            radius=self.radius,
        )

    def coarsen(self, field: Any, factor: int) -> tuple[GridGeometry, Any]:
        coarse = self.coarsen_geometry(factor)
        if factor == 1:
            return coarse, np.asarray(field)
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        factor = int(factor)
        weights = self.areas().ravel()
        weights_4d = weights.reshape(
            coarse.nlat,  # type: ignore[attr-defined]
            factor,
            coarse.nlon,  # type: ignore[attr-defined]
            factor,
        )
        blocks = values.reshape(*values.shape[:-2], *weights_4d.shape)
        numerator = np.einsum("...abcd,abcd->...ac", blocks, weights_4d)
        denominator = weights_4d.sum(axis=(1, 3))
        return coarse, np.asarray(numerator / denominator, dtype=np.float64)

    def subsample(self, field: Any, factor: int) -> np.ndarray:
        values = np.asarray(backend.to_numpy(field))
        return np.asarray(values[..., ::factor, ::factor])

    # ===== 算子（解析度规中心差分） =====

    @staticmethod
    def _restore(reference: Any, values: np.ndarray) -> Any:
        xp = backend.backend_of(reference)
        return values if xp is np else xp.array(values)

    def gradient(self, field: Any) -> tuple[Any, Any]:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        gx, gy = spherical.gradient(values, self.lat, self.radius)
        return self._restore(field, gx), self._restore(field, gy)

    def divergence(self, u: Any, v: Any) -> Any:
        values = np.asarray(backend.to_numpy(u), dtype=np.float64)
        other = np.asarray(backend.to_numpy(v), dtype=np.float64)
        out = spherical.divergence(values, other, self.lat, self.radius)
        return self._restore(u, out)

    def vorticity(self, u: Any, v: Any) -> Any:
        values = np.asarray(backend.to_numpy(u), dtype=np.float64)
        other = np.asarray(backend.to_numpy(v), dtype=np.float64)
        out = spherical.vorticity(values, other, self.lat, self.radius)
        return self._restore(u, out)

    def laplacian(self, field: Any) -> Any:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        return self._restore(field, spherical.laplacian(values, self.lat, self.radius))

    def params(self) -> dict[str, Any]:
        # GridState 的 lat/lon 字段名与构造参数一致
        return {"nlat": self.nlat, "nlon": self.nlon, "lat": self.lat, "lon": self.lon}


def _coords_from_edges(edges: np.ndarray) -> np.ndarray:
    """由格边求格心（面积一致的分块中心）。"""
    return 0.5 * (edges[:-1] + edges[1:])


# ===== 立方球网格 =====


@dataclass(frozen=True)
class CubedSphereGeometry(GridGeometry):
    """立方球网格几何（方案《地形生成混合方案》§1.5 推荐方案）。"""

    grid_type: ClassVar[str] = "cubed_sphere"
    layout_ndim: ClassVar[int] = 3

    n_side: int
    radius: float = EARTH_RADIUS
    equiangular: bool = True

    @cached_property
    def sphere(self) -> CubedSphere:
        """底层立方球几何对象（按参数缓存）。"""
        return CubedSphere(self.n_side, radius=self.radius, equiangular=self.equiangular)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sphere.shape

    @property
    def resolution(self) -> float:
        return self.sphere.resolution

    def areas(self) -> np.ndarray:
        return self.sphere.areas()

    def cell_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = self.sphere.centers_latlon()
        return lat.ravel(), lon.ravel()

    def lookup(self, lat: float, lon: float) -> int:
        index = self.sphere.pixel_of_latlon(np.asarray(lat), np.asarray(lon))
        return int(np.asarray(index).reshape(-1)[0])

    def to_latlon(self, field: Any, nlat: int, nlon: int, order: Order = "bilinear") -> np.ndarray:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        if values.shape[-3:] != self.shape:
            raise ValueError(f"字段末三维应为 {self.shape}，实际 {values.shape}")
        if order in _NEAREST_ORDERS:
            cells = self.sphere.latlon_to_cell(nlat, nlon).ravel()
            out = values.reshape(-1, self.sphere.size)[:, cells]
            out = out.reshape(*values.shape[:-3], nlat, nlon)
        else:
            out = self.sphere.to_latlon(values, nlat, nlon)
        return np.asarray(out, dtype=np.float64)

    def from_latlon(self, field: Any, order: Order = "bilinear") -> np.ndarray:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        if order in _NEAREST_ORDERS:
            nlat, nlon = values.shape[-2:]
            lat, lon = self.sphere.centers_latlon()
            i = np.clip(np.floor((lat + 90.0) * nlat / 180.0).astype(np.int64), 0, nlat - 1)
            j = np.floor((lon + 180.0) * nlon / 360.0).astype(np.int64) % nlon
            flat = values.reshape(-1, nlat * nlon)[:, (i * nlon + j).ravel()]
            out = flat.reshape(*values.shape[:-2], *self.sphere.shape)
        else:
            out = self.sphere.from_latlon(values, periodic_lon=True)
        return np.asarray(out, dtype=np.float64)

    def coarsen_geometry(self, factor: int) -> GridGeometry:
        factor = int(factor)
        if factor < 2 or self.n_side % factor:
            raise ValueError(f"factor={factor} 必须是 >= 2 且整除 n_side={self.n_side} 的整数")
        return CubedSphereGeometry(
            n_side=self.n_side // factor, radius=self.radius, equiangular=self.equiangular
        )

    def coarsen(self, field: Any, factor: int) -> tuple[GridGeometry, Any]:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        coarse_field = self.sphere.coarsen(values, factor)
        return self.coarsen_geometry(factor), coarse_field

    def subsample(self, field: Any, factor: int) -> np.ndarray:
        """面内每 ``factor`` 格取一个代表（面轴不参与）。"""
        values = np.asarray(backend.to_numpy(field))
        return np.asarray(values[..., :, ::factor, ::factor])

    def gradient(self, field: Any) -> tuple[Any, Any]:
        return self.sphere.gradient(field)

    def divergence(self, u: Any, v: Any) -> Any:
        return self.sphere.divergence(u, v)

    def vorticity(self, u: Any, v: Any) -> Any:
        return self.sphere.vorticity(u, v)

    def laplacian(self, field: Any) -> Any:
        return self.sphere.laplacian(field)

    def params(self) -> dict[str, Any]:
        return {"n_side": self.n_side, "equiangular": self.equiangular}


# ===== HEALPix 网格 =====


@dataclass(frozen=True)
class HealpixGeometry(GridGeometry):
    """HEALPix 等面积层次化网格几何（方案 §1.5 备选方案）。"""

    grid_type: ClassVar[str] = "healpix"
    layout_ndim: ClassVar[int] = 1

    nside: int
    radius: float = EARTH_RADIUS

    @cached_property
    def grid(self) -> HealpixGrid:
        """底层 HEALPix 几何对象（按参数缓存）。"""
        return HealpixGrid(self.nside, radius=self.radius)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid.shape

    @property
    def resolution(self) -> float:
        return self.grid.resolution

    def areas(self) -> np.ndarray:
        return self.grid.areas()

    def cell_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        return self.grid.centers_latlon()

    def lookup(self, lat: float, lon: float) -> int:
        index = self.grid.pixel_of_latlon(np.asarray(lat), np.asarray(lon))
        return int(np.asarray(index).reshape(-1)[0])

    def to_latlon(self, field: Any, nlat: int, nlon: int, order: Order = "bilinear") -> np.ndarray:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        return self.grid.to_latlon(values, nlat, nlon, order="nearest" if order in _NEAREST_ORDERS else "bilinear")

    def from_latlon(self, field: Any, order: Order = "bilinear") -> np.ndarray:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        return self.grid.from_latlon(values, order="nearest" if order in _NEAREST_ORDERS else "bilinear")

    def coarsen_geometry(self, factor: int) -> GridGeometry:
        factor = int(factor)
        if factor < 2 or (factor & (factor - 1)):
            raise ValueError(f"factor={factor} 必须是 >= 2 的 2 的幂")
        if factor > self.nside:
            raise ValueError(f"factor={factor} 不能超过 nside={self.nside}")
        return HealpixGeometry(nside=self.nside // factor, radius=self.radius)

    def coarsen(self, field: Any, factor: int) -> tuple[GridGeometry, Any]:
        values = np.asarray(backend.to_numpy(field), dtype=np.float64)
        coarse_field = self.grid.coarsen(values, factor)
        return self.coarsen_geometry(factor), coarse_field

    def subsample(self, field: Any, factor: int) -> np.ndarray:
        """四叉树层次：每个粗像素取其第一个子像素（嵌套序下块首元素）。"""
        values = np.asarray(backend.to_numpy(field))
        return np.asarray(values[..., :: factor * factor])

    def gradient(self, field: Any) -> tuple[Any, Any]:
        return self.grid.gradient(field)

    def divergence(self, u: Any, v: Any) -> Any:
        return self.grid.divergence(u, v)

    def vorticity(self, u: Any, v: Any) -> Any:
        return self.grid.vorticity(u, v)

    def laplacian(self, field: Any) -> Any:
        return self.grid.laplacian(field)

    def params(self) -> dict[str, Any]:
        return {"nside": self.nside}


# ===== 构造入口 =====


def make_geometry(grid_type: str, **params: Any) -> GridGeometry:
    """按类型与参数构造几何对象（``radius`` 缺省为地球半径）。"""
    radius = float(params.pop("radius", EARTH_RADIUS)) if params.get("radius") else EARTH_RADIUS
    if grid_type == "latlon":
        return LatLonGeometry(
            nlat=int(params["nlat"]),
            nlon=int(params["nlon"]),
            lat_input=params.get("lat"),
            lon_input=params.get("lon"),
            radius=radius,
        )
    if grid_type == "cubed_sphere":
        if not params.get("n_side"):
            raise ValueError("cubed_sphere 网格需要 n_side 参数")
        return CubedSphereGeometry(
            n_side=int(params["n_side"]),
            radius=radius,
            equiangular=bool(params.get("equiangular", True)),
        )
    if grid_type == "healpix":
        if not params.get("nside"):
            raise ValueError("healpix 网格需要 nside 参数")
        return HealpixGeometry(nside=int(params["nside"]), radius=radius)
    raise ValueError(
        f"未知网格类型 {grid_type!r}（可选 'latlon' / 'cubed_sphere' / 'healpix'）"
    )


#: 支持的网格类型
GRID_TYPES: tuple[str, ...] = ("latlon", "cubed_sphere", "healpix")


__all__ = [
    "GRID_TYPES",
    "CubedSphereGeometry",
    "GridGeometry",
    "HealpixGeometry",
    "LatLonGeometry",
    "make_geometry",
]