"""网格数据模型。

``GridState`` 是整个系统的核心数据结构，承载所有物理场（见《项目结构与技术栈》§3.1）。
设计约束：

- **类型安全**：每个物理量的单位、有效范围、默认值由 :data:`FIELDS` 声明，
  写入时按 :class:`FieldSpec` 校验形状、dtype 与量级。
- **内存高效**：字段为 MLX / NumPy 定长数组，无 Python 对象开销。
- **可序列化**：:meth:`GridState.to_dict` / :meth:`GridState.from_dict` 保留网格元数据与全部字段。
- **可切片**：:meth:`GridState.slice_region` 区域提取（经纬网格）、:meth:`GridState.coarsen`
  限制算子、:meth:`GridState.resample` 重采样（延长算子，经纬网格）。

**三种网格对等**（方案《地形生成混合方案》§1.5）：字段水平形状由网格类型决定——

============   ===================  ===========================
``grid_type``  水平形状             说明
============   ===================  ===========================
``latlon``     ``(nlat, nlon)``     等经纬网格，高纬单元面积收缩
``cubed_sphere`` ``(6, n_side, n_side)`` 立方球网格，无极点奇点
``healpix``    ``(npix,)``          HEALPix 等面积层次化网格
============   ===================  ===========================

三者共享同一套字段声明、面积加权统计、微分算子（:meth:`GridState.gradient` 等）、
多分辨率（:meth:`GridState.coarsen`）与跨网格重网格（:meth:`GridState.regrid`），
几何能力统一由 :mod:`virtual_world.core.geometry` 提供。

经纬度网格的坐标约定见 :mod:`virtual_world.core.spherical`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml

from . import backend, spherical
from . import geometry as geometry_module
from .backend import DEFAULT_DTYPE
from .geometry import GRID_TYPES, GridGeometry
from .planet import PlanetParams, config_dir

# ===== 字段规范 =====


@dataclass(frozen=True)
class FieldSpec:
    """单个物理量的元数据：单位、dtype、有效范围与维度顺序。

    ``dims`` 中的 ``"lat"``/``"lon"`` 表示**水平维**（两种网格下都对应网格自身的
    水平形状），``"depth"`` 为垂直维。这样同一份字段声明可以直接用于三种网格。
    """

    name: str
    unit: str
    group: str
    dims: tuple[str, ...] = ("lat", "lon")
    dtype: str = DEFAULT_DTYPE
    valid_range: tuple[float, float] | None = None
    description: str = ""

    def expected_shape(self, horizontal: tuple[int, ...], nz: int = 1) -> tuple[int, ...]:
        """按水平形状与垂直层数计算期望形状。"""
        shape: list[int] = []
        index = 0
        while index < len(self.dims):
            dim = self.dims[index]
            if dim in ("lat", "lon"):
                if index + 1 >= len(self.dims) or self.dims[index + 1] not in ("lat", "lon"):
                    raise ValueError(f"{self.name}: 水平维必须成对出现，实际 dims={self.dims}")
                shape.extend(horizontal)
                index += 2
                continue
            if dim == "depth":
                shape.append(nz)
            else:
                raise ValueError(f"{self.name}: 未知维度 {dim!r}")
            index += 1
        return tuple(shape)

    def check(
        self,
        values: Any,
        horizontal: tuple[int, ...],
        nz: int = 1,
        check_range: bool = True,
    ) -> list[str]:
        """校验形状、有限性与取值范围，返回问题描述列表。"""
        arr = backend.to_numpy(values)
        expected = self.expected_shape(horizontal, nz)
        if arr.shape != expected:
            return [f"{self.name}: 形状 {arr.shape} 与期望 {expected} 不符"]

        problems: list[str] = []
        if arr.dtype.kind == "f":
            non_finite = int(np.count_nonzero(~np.isfinite(arr)))
            if non_finite:
                problems.append(f"{self.name}: 含 {non_finite} 个非有限值 (NaN/Inf)")
        if check_range and self.valid_range is not None:
            lo, hi = self.valid_range
            below = int(np.count_nonzero(arr < lo))
            above = int(np.count_nonzero(arr > hi))
            if below or above:
                problems.append(
                    f"{self.name}: {below} 个值 < {lo}，{above} 个值 > {hi}（单位 {self.unit}）"
                )
        return problems


def _spec(name: str, unit: str, group: str, lo: float | None, hi: float | None, desc: str, **kw: Any) -> FieldSpec:
    rng = None if lo is None or hi is None else (lo, hi)
    return FieldSpec(name=name, unit=unit, group=group, valid_range=rng, description=desc, **kw)


FIELDS: dict[str, FieldSpec] = {
    # ===== 地形层 =====
    "elevation": _spec("elevation", "m", "terrain", -12000, 9000, "海拔高度，海洋为海底高程"),
    "is_ocean": _spec("is_ocean", "bool", "terrain", None, None, "海洋掩码", dtype="bool"),
    "ocean_depth": _spec("ocean_depth", "m", "terrain", 0, 12000, "海洋深度，陆地为 0"),
    "terrain_type": _spec("terrain_type", "code", "terrain", 0, 127, "地形类型编码", dtype="int8"),
    # ===== 辐射层 =====
    "S_TOA_jan": _spec("S_TOA_jan", "W/m^2", "radiation", 0, 2000, "一月大气顶入射辐射"),
    "S_TOA_jul": _spec("S_TOA_jul", "W/m^2", "radiation", 0, 2000, "七月大气顶入射辐射"),
    "OLR": _spec("OLR", "W/m^2", "radiation", 0, 500, "向外长波辐射"),
    # ===== 温度层 =====
    "T_jan": _spec("T_jan", "degC", "temperature", -150, 150, "一月平均气温"),
    "T_jul": _spec("T_jul", "degC", "temperature", -150, 150, "七月平均气温"),
    "T_annual": _spec("T_annual", "degC", "temperature", -150, 150, "年平均气温"),
    "T_surface": _spec("T_surface", "degC", "temperature", -150, 150, "地表（皮温）温度"),
    "T_soil": _spec(
        "T_soil", "degC", "temperature", -150, 150, "土壤温度廓线", dims=("depth", "lat", "lon")
    ),
    # ===== 气压与风场 =====
    "pressure_jan": _spec("pressure_jan", "hPa", "atmosphere", 300, 1200, "一月海平面气压"),
    "pressure_jul": _spec("pressure_jul", "hPa", "atmosphere", 300, 1200, "七月海平面气压"),
    "u_jan": _spec("u_jan", "m/s", "atmosphere", -300, 300, "一月纬向风（向东为正）"),
    "v_jan": _spec("v_jan", "m/s", "atmosphere", -300, 300, "一月经向风（向北为正）"),
    "u_jul": _spec("u_jul", "m/s", "atmosphere", -300, 300, "七月纬向风"),
    "v_jul": _spec("v_jul", "m/s", "atmosphere", -300, 300, "七月经向风"),
    # ===== 海洋 =====
    "ocean_u": _spec("ocean_u", "m/s", "ocean", -10, 10, "洋流纬向分量"),
    "ocean_v": _spec("ocean_v", "m/s", "ocean", -10, 10, "洋流经向分量"),
    "ocean_temp": _spec("ocean_temp", "degC", "ocean", -5, 45, "海表温度 SST"),
    # ===== 降水与水循环 =====
    "P_jan": _spec("P_jan", "mm/month", "precipitation", 0, 5000, "一月降水量"),
    "P_jul": _spec("P_jul", "mm/month", "precipitation", 0, 5000, "七月降水量"),
    "P_annual": _spec("P_annual", "mm/year", "precipitation", 0, 20000, "年降水量"),
    "evaporation": _spec("evaporation", "mm/month", "precipitation", 0, 5000, "蒸发量"),
    # ===== 分类层 =====
    "climate_code": _spec("climate_code", "code", "classification", 0, 127, "Köppen 气候类型编码", dtype="int8"),
    "biome": _spec("biome", "code", "classification", 0, 127, "生物群系编码", dtype="int8"),
    "soil_type": _spec("soil_type", "code", "classification", 0, 127, "土壤类型编码（WRB）", dtype="int8"),
    "agriculture": _spec("agriculture", "code", "classification", 0, 127, "农业类型编码（FAO AEZ）", dtype="int8"),
    # ===== 派生层 =====
    "SOC": _spec("SOC", "kg/m^2", "derived", 0, 200, "土壤有机碳储量"),
    "pH": _spec("pH", "-", "derived", 1, 14, "土壤 pH"),
    "CEC": _spec("CEC", "cmol/kg", "derived", 0, 250, "阳离子交换量"),
    "NPP": _spec("NPP", "gC/m^2/yr", "derived", 0, 4000, "净初级生产力"),
}

#: 分类/布尔字段在粗化与重采样时不能做线性插值
NON_INTERPOLATED_DTYPES = frozenset({"bool", "int8", "int16", "int32", "int64", "uint8"})


def group_fields(group: str) -> list[str]:
    """列出某个分组的字段名。"""
    return [name for name, spec in FIELDS.items() if spec.group == group]


# ===== 分级分辨率 =====


def _field_order(spec: FieldSpec) -> str:
    """字段重网格时的插值阶数：分类/布尔场取最近邻，其余为双线性。"""
    return "nearest" if spec.dtype in NON_INTERPOLATED_DTYPES else "bilinear"


def _nside_for_resolution(resolution_deg: float) -> int:
    """把名义分辨率换算为最近的 2 的幂 ``nside``（HEALPix）。"""
    import math

    target = math.sqrt((4.0 * math.pi) / math.radians(max(resolution_deg, 1e-6)) ** 2 / 12.0)
    return max(1, 2 ** max(0, int(round(math.log2(max(target, 1.0))))))


@dataclass(frozen=True)
class ResolutionLevel:
    """分级分辨率配置项（见《精度与性能策略》§2.2）。"""

    name: str
    resolution: float
    nlat: int
    nlon: int
    purpose: str = ""
    expected_time: str = ""

    @property
    def cells(self) -> int:
        return self.nlat * self.nlon


def load_resolution_levels(path: str | Path | None = None) -> dict[str, ResolutionLevel]:
    """读取 ``configs/resolution_levels.yaml``。"""
    config_path = Path(path) if path else config_dir() / "resolution_levels.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    levels: dict[str, ResolutionLevel] = {}
    for name, item in (data.get("levels") or {}).items():
        levels[name] = ResolutionLevel(
            name=name,
            resolution=float(item["resolution"]),
            nlat=int(item["nlat"]),
            nlon=int(item["nlon"]),
            purpose=str(item.get("purpose", "")),
            expected_time=str(item.get("expected_time", "")),
        )
    return levels


def get_resolution_level(name: str, path: str | Path | None = None) -> ResolutionLevel:
    """按名字取分级分辨率配置。"""
    levels = load_resolution_levels(path)
    if name not in levels:
        raise KeyError(f"未知分辨率级别 {name!r}，可选: {', '.join(sorted(levels))}")
    return levels[name]


# ===== 网格状态 =====


@dataclass
class GridState:
    """虚拟世界的完整状态：网格元数据 + 全部物理场。

    网格参数按类型提供：``latlon`` 用 ``nlat``/``nlon``（可给自定义 ``lat``/``lon`` 轴），
    ``cubed_sphere`` 用 ``n_side``，``healpix`` 用 ``nside``。其余字段与网格类型无关。
    """

    # ===== 网格元数据 =====
    nlat: int = 0
    nlon: int = 0
    resolution: float = 0.0  # deg；为 0 时按网格类型推导
    grid_type: str = "latlon"  # latlon / cubed_sphere / healpix
    backend: str = "auto"  # auto / mlx / numpy
    dtype: str = DEFAULT_DTYPE
    nz: int = 1  # 垂直层数，三维字段（T_soil）的第二维
    seed: int = 0
    planet_params: PlanetParams | Mapping[str, Any] | None = None
    timestamp: str = ""

    # ===== 非经纬网格参数 =====
    n_side: int = 0  # cubed_sphere 每个面的边长
    nside: int = 0  # healpix 的 nside
    equiangular: bool = True  # 立方球是否使用等角投影

    # ===== 坐标（None 表示按 resolution 自动生成全球网格）=====
    lat: np.ndarray | None = field(default=None, repr=False)
    lon: np.ndarray | None = field(default=None, repr=False)

    # ===== 地形层 =====
    elevation: Any = None
    is_ocean: Any = None
    ocean_depth: Any = None
    terrain_type: Any = None

    # ===== 辐射层 =====
    S_TOA_jan: Any = None
    S_TOA_jul: Any = None
    OLR: Any = None

    # ===== 温度层 =====
    T_jan: Any = None
    T_jul: Any = None
    T_annual: Any = None
    T_surface: Any = None
    T_soil: Any = None

    # ===== 气压与风场 =====
    pressure_jan: Any = None
    pressure_jul: Any = None
    u_jan: Any = None
    v_jan: Any = None
    u_jul: Any = None
    v_jul: Any = None

    # ===== 洋流 =====
    ocean_u: Any = None
    ocean_v: Any = None
    ocean_temp: Any = None

    # ===== 降水 =====
    P_jan: Any = None
    P_jul: Any = None
    P_annual: Any = None
    evaporation: Any = None

    # ===== 分类层 =====
    climate_code: Any = None
    biome: Any = None
    soil_type: Any = None
    agriculture: Any = None

    # ===== 派生层 =====
    SOC: Any = None
    pH: Any = None
    CEC: Any = None
    NPP: Any = None

    # ===== 派生属性 =====
    geometry: GridGeometry = field(init=False, repr=False)
    lat_edges: np.ndarray | None = field(init=False, repr=False, default=None)
    lon_edges: np.ndarray | None = field(init=False, repr=False, default=None)
    cos_lat: np.ndarray | None = field(init=False, repr=False, default=None)
    cell_area: Any = field(init=False, repr=False, default=None)
    total_area: float = field(init=False, default=0.0)
    is_global: bool = field(init=False, default=True)
    _xp: Any = field(init=False, repr=False, compare=False, default=None)
    _backend_name: str = field(init=False, repr=False, compare=False, default="numpy")

    #: 判定全球网格的容差（度数）
    _GLOBAL_TOL: ClassVar[float] = 1e-6

    def __post_init__(self) -> None:
        self.nz = int(self.nz)
        if self.nz <= 0:
            raise ValueError("nz 必须为正整数")
        if self.grid_type not in GRID_TYPES:
            raise NotImplementedError(
                f"暂不支持 grid_type={self.grid_type!r}（可选 {', '.join(GRID_TYPES)}）"
            )

        if isinstance(self.planet_params, Mapping):
            self.planet_params = PlanetParams.from_dict(self.planet_params)

        self._xp = backend.resolve_backend(self.backend)
        self._backend_name = backend.backend_name(self._xp)

        self.geometry = self._build_geometry()
        self.resolution = float(self.geometry.resolution)

        if isinstance(self.geometry, geometry_module.LatLonGeometry):
            self.nlat = self.geometry.nlat
            self.nlon = self.geometry.nlon
            self.lat = self.geometry.lat
            self.lon = self.geometry.lon
            self.lat_edges = self.geometry.lat_edges
            self.lon_edges = self.geometry.lon_edges
            self.cos_lat = self.geometry.cos_lat
            self.is_global = self.geometry.is_global

        areas = self.geometry.areas()
        self.cell_area = backend.to_backend(areas, self._backend_name)
        self.total_area = float(np.sum(areas))
        if self.grid_type != "latlon":
            self.is_global = self.geometry.is_global

        for name in FIELDS:
            value = getattr(self, name)
            if value is not None:
                self.set(name, value)

    def _build_geometry(self) -> GridGeometry:
        """按网格类型与参数构造几何对象（经纬网格沿用原有校验语义）。"""
        radius = self.planet.radius if self.planet is not None else self._earth_radius()
        if self.grid_type == "latlon":
            if self.nlat <= 0 or self.nlon <= 0:
                raise ValueError("latlon 网格需要正整数 nlat 与 nlon")
            if self.resolution > 0 and (self.lat is None or self.lon is None):
                if abs(self.nlat * self.resolution - 180.0) > self._GLOBAL_TOL:
                    raise ValueError(
                        f"nlat={self.nlat} 与 resolution={self.resolution} 不一致："
                        f"nlat*resolution 应为 180"
                    )
                if abs(self.nlon * self.resolution - 360.0) > self._GLOBAL_TOL:
                    raise ValueError(
                        f"nlon={self.nlon} 与 resolution={self.resolution} 不一致："
                        f"nlon*resolution 应为 360"
                    )
            geom = geometry_module.make_geometry(
                "latlon",
                nlat=self.nlat,
                nlon=self.nlon,
                lat=self.lat,
                lon=self.lon,
                radius=radius,
            )
            if self.lat is not None and self.lon is not None and self.resolution > 0:
                # 区域提取的 lon 在接缝处经 360 度环绕，因此用模 360 检查间距
                self._check_uniform(np.asarray(self.lat), self.resolution)
                self._check_uniform(np.asarray(self.lon), self.resolution, modulo=360.0)
            return geom
        return geometry_module.make_geometry(
            self.grid_type,
            n_side=self.n_side,
            nside=self.nside,
            equiangular=self.equiangular,
            radius=radius,
        )

    # ===== 内部工具 =====

    @staticmethod
    def _earth_radius() -> float:
        from . import constants as const

        return const.EARTH_RADIUS

    @staticmethod
    def _check_uniform(values: np.ndarray, step: float, modulo: float | None = None) -> None:
        if values.size < 2:
            return
        diff = np.diff(values)
        if modulo is not None:
            diff = diff % modulo
        if not np.allclose(diff, step, atol=1e-6):
            raise ValueError(f"坐标间距不均匀：期望 {step}")

    @staticmethod
    def _edges_from_centers(centers: np.ndarray, step: float) -> np.ndarray:
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[:-1] = centers - step / 2.0
        edges[-1] = centers[-1] + step / 2.0
        return edges

    def _clone_array(self, values: Any, dtype: str | None = None) -> Any:
        """把数组搬到当前后端并复制一份（不共享底层缓冲）。"""
        arr = backend.to_backend(backend.to_numpy(values), self._backend_name)
        if dtype is not None:
            arr = arr.astype(backend.to_dtype(self._xp, dtype))
        return self._xp.array(arr)

    def _take(self, values: Any, indices: np.ndarray, axis: int) -> Any:
        """沿指定轴按索引取值（索引需转为后端支持的整数数组）。"""
        idx: Any = np.asarray(indices, dtype=np.int32)
        if self._backend_name == "mlx":
            idx = self._xp.array(idx)
        return self._xp.take(values, idx, axis=axis)

    @staticmethod
    def _coords_from_edges(edges: np.ndarray) -> np.ndarray:
        """由格边求格心（面积一致的分块中心）。"""
        return 0.5 * (edges[:-1] + edges[1:])

    def _like(self, **overrides: Any) -> GridState:
        """复制元数据构造一个新网格（网格参数可按需覆盖）。"""
        meta: dict[str, Any] = {
            "grid_type": self.grid_type,
            "backend": self.backend,
            "dtype": self.dtype,
            "nz": self.nz,
            "seed": self.seed,
            "planet_params": self.planet_params,
            "timestamp": self.timestamp,
            "n_side": self.n_side,
            "nside": self.nside,
            "equiangular": self.equiangular,
        }
        meta.update(overrides)
        return GridState(**meta)

    # ===== 基本属性 =====

    @property
    def xp(self) -> Any:
        """当前数组后端模块（``mlx.core`` 或 ``numpy``）。"""
        return self._xp

    @property
    def backend_name(self) -> str:
        """当前后端名（``"mlx"`` / ``"numpy"``）。"""
        return self._backend_name

    @property
    def planet(self) -> PlanetParams | None:
        """行星参数（未设置时为 None，此时面积计算退化为地球半径）。"""
        return self.planet_params if isinstance(self.planet_params, PlanetParams) else None

    @property
    def shape(self) -> tuple[int, ...]:
        """水平网格形状（经纬 ``(nlat, nlon)`` / 立方球 ``(6, n, n)`` / HEALPix ``(npix,)``）。"""
        return self.geometry.shape

    @property
    def size(self) -> int:
        """水平网格点数。"""
        return self.geometry.size

    @property
    def cell_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """每个单元的经纬度（度），展平为长度 ``size`` 的一维数组。"""
        return self.geometry.cell_latlon()

    @property
    def nbytes(self) -> int:
        """已填充字段占用的内存字节数。"""
        total = 0
        for name in FIELDS:
            value = getattr(self, name)
            if value is not None:
                total += backend.to_numpy(value).nbytes
        return total

    # ===== 构造 =====

    @classmethod
    def from_resolution(
        cls,
        resolution: float,
        planet: PlanetParams | None = None,
        seed: int = 0,
        backend_name: str = "auto",
        nz: int = 1,
        timestamp: str = "",
        grid_type: str = "latlon",
    ) -> GridState:
        """按角分辨率 (deg) 构造全球网格。

        ``grid_type="latlon"`` 时 ``resolution`` 必须整除 180 度；立方球取
        ``n_side = round(90/resolution)``，HEALPix 取与分辨率最近的 2 的幂 ``nside``。
        """
        if grid_type == "latlon":
            nlat_f = 180.0 / resolution
            nlat = int(round(nlat_f))
            if abs(nlat - nlat_f) > 1e-9:
                raise ValueError(f"分辨率 {resolution} 不能整除 180 度")
            return cls(
                nlat=nlat,
                nlon=2 * nlat,
                resolution=float(resolution),
                backend=backend_name,
                nz=nz,
                seed=seed,
                planet_params=planet,
                timestamp=timestamp,
            )
        if grid_type == "cubed_sphere":
            n_side = max(1, int(round(90.0 / float(resolution))))
            return cls.from_cubed_sphere(
                n_side, planet=planet, seed=seed, backend_name=backend_name, nz=nz
            )
        if grid_type == "healpix":
            # 目标 npix ≈ 4π/(res_rad)^2 ⇒ nside ≈ sqrt(npix/12)，再取最近的 2 的幂
            import math

            target = math.sqrt((4.0 * math.pi) / math.radians(float(resolution)) ** 2 / 12.0)
            nside = max(1, 2 ** max(0, int(round(math.log2(max(target, 1.0))))))
            return cls.from_healpix(
                nside, planet=planet, seed=seed, backend_name=backend_name, nz=nz
            )
        raise ValueError(f"未知网格类型 {grid_type!r}（可选 {', '.join(GRID_TYPES)}）")

    @classmethod
    def from_cubed_sphere(
        cls,
        n_side: int,
        planet: PlanetParams | None = None,
        seed: int = 0,
        backend_name: str = "auto",
        nz: int = 1,
        equiangular: bool = True,
    ) -> GridState:
        """构造立方球网格状态，字段水平形状 ``(6, n_side, n_side)``。"""
        return cls(
            grid_type="cubed_sphere",
            n_side=int(n_side),
            equiangular=bool(equiangular),
            backend=backend_name,
            nz=nz,
            seed=seed,
            planet_params=planet,
        )

    @classmethod
    def from_healpix(
        cls,
        nside: int,
        planet: PlanetParams | None = None,
        seed: int = 0,
        backend_name: str = "auto",
        nz: int = 1,
    ) -> GridState:
        """构造 HEALPix 网格状态，字段水平形状 ``(npix,)``（``npix = 12*nside^2``）。"""
        return cls(
            grid_type="healpix",
            nside=int(nside),
            backend=backend_name,
            nz=nz,
            seed=seed,
            planet_params=planet,
        )

    @classmethod
    def from_level(
        cls,
        level: str,
        planet: PlanetParams | None = None,
        seed: int = 0,
        backend_name: str = "auto",
        nz: int = 1,
        levels_path: str | Path | None = None,
        grid_type: str = "latlon",
    ) -> GridState:
        """按分级分辨率名（preview / coarse / medium / fine / high / ultra）构造网格。"""
        config = get_resolution_level(level, levels_path)
        if grid_type == "latlon":
            return cls(
                nlat=config.nlat,
                nlon=config.nlon,
                resolution=config.resolution,
                backend=backend_name,
                nz=nz,
                seed=seed,
                planet_params=planet,
            )
        return cls.from_resolution(
            config.resolution,
            planet=planet,
            seed=seed,
            backend_name=backend_name,
            nz=nz,
            grid_type=grid_type,
        )

    @classmethod
    def from_planet(
        cls, planet: PlanetParams, resolution: float, seed: int = 0, **kw: Any
    ) -> GridState:
        """按行星参数与分辨率构造网格（网格半径取自行星）。"""
        return cls.from_resolution(resolution, planet=planet, seed=seed, **kw)

    # ===== 字段访问 =====

    @staticmethod
    def spec(name: str) -> FieldSpec:
        """取字段规范，未定义字段直接报错。"""
        if name not in FIELDS:
            raise KeyError(f"未知字段 {name!r}，可用字段见 core.grid.FIELDS")
        return FIELDS[name]

    @classmethod
    def field_names(cls) -> list[str]:
        """全部字段名。"""
        return list(FIELDS)

    def zeros(self, name: str) -> Any:
        """按字段规范创建零数组。"""
        spec = self.spec(name)
        return backend.zeros(
            spec.expected_shape(self.geometry.shape, self.nz), self._backend_name, spec.dtype
        )

    def set(self, name: str, values: Any, check_range: bool = True) -> None:
        """写入字段：校验形状/量级并转换为规定的 dtype。

        校验在类型转换**之前**进行，避免越界值被 int8/float32 静默截断。
        """
        spec = self.spec(name)
        if not backend.is_array(values):
            raise TypeError(f"{name} 必须是数组，实际为 {type(values).__name__}")
        problems = spec.check(values, self.geometry.shape, self.nz, check_range=check_range)
        if problems:
            raise ValueError("字段校验失败:\n  - " + "\n  - ".join(problems))
        setattr(self, name, self._clone_array(values, dtype=spec.dtype))

    def get(self, name: str) -> Any:
        """读取字段，未填充时抛错。"""
        value = getattr(self, self.spec(name).name)
        if value is None:
            raise KeyError(f"字段 {name!r} 尚未填充")
        return value

    def has(self, name: str) -> bool:
        """字段是否已填充。"""
        return getattr(self, self.spec(name).name) is not None

    def filled_fields(self) -> list[str]:
        """已填充字段名列表。"""
        return [name for name in FIELDS if getattr(self, name) is not None]

    def as_numpy(self, name: str) -> np.ndarray:
        """读取字段并转为 NumPy（用于出图与磁盘 IO）。"""
        return backend.to_numpy(self.get(name))

    def to_numpy_dict(self) -> dict[str, np.ndarray]:
        """全部已填充字段的 NumPy 视图。"""
        return {name: backend.to_numpy(getattr(self, name)) for name in self.filled_fields()}

    def nearest_index(self, lat: float, lon: float) -> tuple[int, int]:
        """返回最接近给定经纬度的网格索引 ``(i, j)``（仅经纬网格）。"""
        if self.grid_type != "latlon":
            raise NotImplementedError(
                f"{self.grid_type} 网格没有 (i, j) 语义，请用 cell_index() 或 value_at()"
            )
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(((self.lon - lon + 180.0) % 360.0) - 180.0)))
        return i, j

    def cell_index(self, lat: float, lon: float) -> int:
        """给定经纬度最近的单元**扁平**索引（三种网格通用）。"""
        return self.geometry.lookup(float(lat), float(lon))

    def value_at(self, name: str, lat: float, lon: float) -> float:
        """查询距离给定经纬度最近单元上的字段值（三种网格通用）。"""
        if self.grid_type == "latlon":
            i, j = self.nearest_index(lat, lon)
            return float(backend.to_numpy(self.get(name))[..., i, j])
        index = self.cell_index(lat, lon)
        flat = backend.to_numpy(self.get(name)).reshape(-1, self.size)
        return float(np.asarray(flat[..., index]).reshape(-1)[0])

    # ===== 诊断与统计 =====

    def global_mean(self, name: str) -> float:
        """面积加权全球平均（三种网格通用，权重为单元面积）。"""
        flat = backend.to_numpy(self.get(name)).reshape(-1, self.size)
        return spherical.global_mean(flat, self._area_flat())

    def global_integral(self, name: str) -> float:
        """面积加权全球积分。"""
        flat = backend.to_numpy(self.get(name)).reshape(-1, self.size)
        return spherical.global_integral(flat, self._area_flat())

    def zonal_mean(self, name: str, nbands: int | None = None) -> np.ndarray:
        """纬向平均剖面。

        经纬网格默认按 ``nlat`` 条带（与原实现一致）；立方球 / HEALPix 按名义分辨率
        自动确定条带数（也可用 ``nbands`` 指定）。
        """
        if self.grid_type == "latlon" and nbands is None:
            return backend.to_numpy(spherical.zonal_mean(self.get(name)))
        values = backend.to_numpy(self.get(name))
        bands = nbands if nbands is not None else max(1, int(round(180.0 / self.resolution)))
        lat, _ = self.geometry.cell_latlon()
        flat = values.reshape(-1, self.size)
        profile = spherical.zonal_mean_bands(flat, lat, self._area_flat(), int(bands))
        return np.asarray(profile.reshape(*values.shape[: -self.geometry.layout_ndim], bands))

    def _area_flat(self) -> np.ndarray:
        """单元面积的展平视图（用于面积加权统计）。"""
        return np.asarray(backend.to_numpy(self.cell_area), dtype=np.float64).reshape(-1)

    # ===== 微分算子（三种网格通用） =====

    def gradient(self, name: str) -> tuple[Any, Any]:
        """标量场梯度 ``(df/dx, df/dy)``（东、北分量，1/m），形状与字段一致。"""
        return self.geometry.gradient(self.get(name))

    def divergence(self, u_name: str, v_name: str) -> Any:
        """水平散度 ``du/dx + dv/dy``（1/s）。"""
        return self.geometry.divergence(self.get(u_name), self.get(v_name))

    def vorticity(self, u_name: str, v_name: str) -> Any:
        """相对涡度 ``dv/dx - du/dy``（1/s）。"""
        return self.geometry.vorticity(self.get(u_name), self.get(v_name))

    def laplacian(self, name: str) -> Any:
        """拉普拉斯算子 ``div(grad f)``（1/m²）。"""
        return self.geometry.laplacian(self.get(name))

    # ===== 网格变换 =====

    def slice_region(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> GridState:
        """区域提取（含经度环绕），仅支持经纬网格。

        极点/日期线上的接缝处经度不再单调，区域网格不适合再做经向差分。
        """
        if self.grid_type != "latlon":
            raise NotImplementedError(
                f"{self.grid_type} 网格不支持矩形区域切片（无规则经纬索引）；"
                "可先用 regrid('latlon') 转换"
            )
        i0 = int(np.searchsorted(self.lat, lat_min, side="left"))
        i1 = int(np.searchsorted(self.lat, lat_max, side="right"))
        if i1 <= i0:
            raise ValueError(f"纬度范围 [{lat_min}, {lat_max}] 未覆盖任何网格点")

        step = 360.0 / self.nlon
        j_start = int(np.floor((lon_min + 180.0) / step))
        count = int(round(((lon_max - lon_min) % 360.0) / step))
        if count == 0:
            count = self.nlon
        j_idx = (j_start + np.arange(count)) % self.nlon

        lat_idx = np.arange(i0, i1)
        result = self._like(
            nlat=lat_idx.size,
            nlon=count,
            resolution=self.resolution,
            lat=self.lat[lat_idx],
            lon=self.lon[j_idx],
        )
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            sliced = self._take(value, lat_idx, axis=-2)
            sliced = self._take(sliced, j_idx, axis=-1)
            result.set(name, sliced)
        return result

    def coarsen(self, factor: int = 2) -> GridState:
        """限制算子：面积加权块平均（分类字段取最近邻），三种网格通用。

        经纬网格的粗网格格边取细网格格边的每 ``factor`` 个取值，立方球的块在面内
        对齐、HEALPix 的块在四叉树层次上对齐，因此三种网格的粗网格单元面积都精确
        等于细网格子块面积之和——**全球加权平均值在粗化前后严格一致**。
        """
        factor = int(factor)
        if factor < 1:
            raise ValueError("factor 必须 >= 1")
        if factor == 1:
            return self.copy()
        if self.grid_type == "latlon":
            return self._coarsen_latlon(factor)

        coarse_geometry = self.geometry.coarsen_geometry(factor)
        result = GridState(
            grid_type=self.grid_type,
            backend=self.backend,
            dtype=self.dtype,
            nz=self.nz,
            seed=self.seed,
            planet_params=self.planet_params,
            timestamp=self.timestamp,
            **coarse_geometry.params(),
        )
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            spec = FIELDS[name]
            if spec.dtype in NON_INTERPOLATED_DTYPES:
                coarse = self.geometry.subsample(backend.to_numpy(value), factor)
            else:
                _, coarse = self.geometry.coarsen(backend.to_numpy(value), factor)
            self._store_field(result, name, coarse)
        return result

    def _coarsen_latlon(self, factor: int) -> GridState:
        """经纬网格的限制算子（保守块平均，分类字段取最近邻）。"""
        if self.nlat % factor or self.nlon % factor:
            raise ValueError(f"factor={factor} 必须同时整除 nlat={self.nlat} 与 nlon={self.nlon}")
        coarse_geometry = self.geometry.coarsen_geometry(factor)
        result = self._like(
            nlat=coarse_geometry.nlat,  # type: ignore[attr-defined]
            nlon=coarse_geometry.nlon,  # type: ignore[attr-defined]
            resolution=self.resolution * factor,
            lat=coarse_geometry.lat,
            lon=coarse_geometry.lon,
        )
        lat_idx = np.arange(0, self.nlat, factor)
        lon_idx = np.arange(0, self.nlon, factor)
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            spec = FIELDS[name]
            if spec.dtype in NON_INTERPOLATED_DTYPES:
                coarse = self._take(self._take(value, lat_idx, axis=-2), lon_idx, axis=-1)
            else:
                _, coarse = self.geometry.coarsen(backend.to_numpy(value), factor)
            self._store_field(result, name, coarse)
        return result

    def _store_field(self, result: GridState, name: str, values: Any) -> None:
        """把（NumPy / MLX）场写入结果网格，按字段规范统一 dtype。"""
        spec = FIELDS[name]
        data = backend.to_numpy(values)
        if spec.dtype not in NON_INTERPOLATED_DTYPES:
            data = data.astype(np.float64)
        result.set(name, self._xp.array(data).astype(backend.to_dtype(self._xp, spec.dtype)))

    def resample(self, nlat: int, nlon: int) -> GridState:
        """延长算子：双线性重采样（分类字段取最近邻），经向循环，仅支持经纬网格。

        目标网格铺满当前网格的纬度与经度范围，因此全球网格重采样后仍为全球网格。
        其他网格请用 :meth:`regrid`（跨网格）或 :meth:`coarsen`（同网格粗化）。
        """
        if self.grid_type != "latlon":
            raise NotImplementedError(
                f"{self.grid_type} 网格请用 coarsen() 粗化或 regrid() 换网格，resample 仅支持经纬网格"
            )
        nlat, nlon = int(nlat), int(nlon)
        if nlat <= 0 or nlon <= 0:
            raise ValueError("目标网格尺寸必须为正")
        if (nlat, nlon) == (self.nlat, self.nlon):
            return self.copy()

        xp = self._xp
        lat_edges2 = np.linspace(self.lat_edges[0], self.lat_edges[-1], nlat + 1)
        lon_edges2 = np.linspace(self.lon_edges[0], self.lon_edges[-1], nlon + 1)
        new_lat = self._coords_from_edges(lat_edges2)
        new_lon = self._coords_from_edges(lon_edges2)
        resolution = float((self.lat_edges[-1] - self.lat_edges[0]) / nlat)

        fi = (new_lat - self.lat_edges[0]) / self.resolution - 0.5
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, self.nlat - 1)
        i1 = np.clip(i0 + 1, 0, self.nlat - 1)
        wi = np.clip(fi - i0, 0.0, 1.0)

        dlon = (self.lon_edges[-1] - self.lon_edges[0]) / self.nlon
        fj = (new_lon - self.lon_edges[0]) / dlon - 0.5
        fj_floor = np.floor(fj).astype(np.int64)
        j0 = fj_floor % self.nlon
        j1 = (j0 + 1) % self.nlon
        wj = fj - fj_floor

        result = self._like(
            nlat=nlat, nlon=nlon, resolution=resolution, lat=new_lat, lon=new_lon
        )
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            spec = FIELDS[name]
            if spec.dtype in NON_INTERPOLATED_DTYPES:
                i_near = np.clip(np.round(fi).astype(np.int64), 0, self.nlat - 1)
                j_near = np.round(fj).astype(np.int64) % self.nlon
                out = self._take(self._take(value, i_near, axis=-2), j_near, axis=-1)
            else:
                out = self._bilinear(value, i0, i1, wi, j0, j1, wj)
                out = out.astype(backend.to_dtype(xp, spec.dtype))
            result.set(name, out)
        return result

    def _bilinear(
        self,
        values: Any,
        i0: np.ndarray,
        i1: np.ndarray,
        wi: np.ndarray,
        j0: np.ndarray,
        j1: np.ndarray,
        wj: np.ndarray,
    ) -> Any:
        """双线性插值核心，纬度方向零阶外推、经度方向循环。"""
        rows0 = self._take(values, i0, axis=-2)
        rows1 = self._take(values, i1, axis=-2)
        f00 = self._take(rows0, j0, axis=-1)
        f01 = self._take(rows0, j1, axis=-1)
        f10 = self._take(rows1, j0, axis=-1)
        f11 = self._take(rows1, j1, axis=-1)
        wi_2d = backend.to_backend(wi[:, None], self._backend_name)
        wj_2d = backend.to_backend(wj[None, :], self._backend_name)
        top = f00 * (1.0 - wj_2d) + f01 * wj_2d
        bottom = f10 * (1.0 - wj_2d) + f11 * wj_2d
        return top * (1.0 - wi_2d) + bottom * wi_2d

    def copy(self) -> GridState:
        """深拷贝（数组独立）。"""
        return self.from_dict(self.to_dict())

    # ===== 跨网格重网格 =====

    def regrid(
        self,
        grid_type: str,
        *,
        nlat: int | None = None,
        nlon: int | None = None,
        n_side: int | None = None,
        nside: int | None = None,
    ) -> GridState:
        """把全部字段重网格到另一种网格（方案 §1.5："两者之间用插值转换"）。

        连续场用双线性/局地线性插值，分类与布尔场用最近邻（不会造出新类别）。
        经纬 ↔ 立方球 ↔ HEALPix 之间以经纬网格为桥梁互转；同类型但不同尺寸时同样
        走桥接路径。参数缺省时按当前分辨率换算目标网格尺寸。
        """
        if grid_type not in GRID_TYPES:
            raise ValueError(f"未知网格类型 {grid_type!r}（可选 {', '.join(GRID_TYPES)}）")
        target = self._target_state(grid_type, nlat, nlon, n_side, nside)
        if target.grid_type == self.grid_type and target.shape == self.shape:
            return self.copy()

        if target.grid_type == "latlon":
            return self._to_latlon_geometry(target.geometry)

        # 非经纬目标：以经纬网格为桥梁（尺寸取源与目标中更细的一方）
        bridge_resolution = min(self.resolution, target.resolution)
        bridge = self._like(
            grid_type="latlon",
            nlat=max(2, int(round(180.0 / bridge_resolution))),
            nlon=max(4, 2 * int(round(180.0 / bridge_resolution))),
            resolution=0.0,
            lat=None,
            lon=None,
            n_side=0,
            nside=0,
        )
        intermediate = self._to_latlon_geometry(bridge.geometry)
        return intermediate._from_latlon_geometry(target.geometry)

    def _to_latlon_geometry(self, target: GridGeometry) -> GridState:
        """把全部字段插值到目标经纬网格。"""
        return self._map_fields(
            target,
            lambda value, spec: self.geometry.to_latlon(
                backend.to_numpy(value),
                target.nlat,  # type: ignore[attr-defined]
                target.nlon,  # type: ignore[attr-defined]
                order=_field_order(spec),
            ),
        )

    def _from_latlon_geometry(self, target: GridGeometry) -> GridState:
        """把经纬网格上的全部字段插值到目标网格。"""
        if self.grid_type != "latlon":
            raise ValueError("_from_latlon_geometry 只接受经纬网格作为源")
        return self._map_fields(
            target,
            lambda value, spec: target.from_latlon(
                backend.to_numpy(value), order=_field_order(spec)
            ),
        )

    def _map_fields(self, target: GridGeometry, convert: Any) -> GridState:
        """按目标几何逐字段转换并组装新的网格状态。"""
        result = GridState(
            grid_type=target.grid_type,
            backend=self.backend,
            dtype=self.dtype,
            nz=self.nz,
            seed=self.seed,
            planet_params=self.planet_params,
            timestamp=self.timestamp,
            **target.params(),
        )
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            spec = FIELDS[name]
            converted = convert(value, spec)
            self._store_field(result, name, converted)
        return result

    def _target_state(
        self,
        grid_type: str,
        nlat: int | None,
        nlon: int | None,
        n_side: int | None,
        nside: int | None,
    ) -> GridState:
        """构造目标网格状态（不含字段，仅用于取几何与尺寸）。"""
        if grid_type == "latlon":
            rows = int(nlat) if nlat else max(2, int(round(180.0 / self.resolution)))
            cols = int(nlon) if nlon else 2 * rows
            return self._like(
                grid_type="latlon",
                nlat=rows,
                nlon=cols,
                resolution=0.0,
                lat=None,
                lon=None,
                n_side=0,
                nside=0,
            )
        if grid_type == "cubed_sphere":
            side = int(n_side) if n_side else max(1, int(round(90.0 / self.resolution)))
            return self._like(grid_type=grid_type, n_side=side, nside=0, nlat=0, nlon=0)
        target_nside = int(nside) if nside else _nside_for_resolution(self.resolution)
        return self._like(grid_type=grid_type, nside=target_nside, n_side=0, nlat=0, nlon=0)

    # ===== 校验与序列化 =====

    def validate(self, check_range: bool = True) -> list[str]:
        """校验网格元数据与全部已填充字段，返回问题列表。"""
        problems: list[str] = []
        if self.is_global and self.grid_type == "latlon":
            if abs(self.lat.min() + 90.0) > self.resolution:
                problems.append("全球网格的纬度范围未覆盖 [-90, 90]")
        if abs(self.total_area - 4.0 * np.pi * self.geometry.radius**2) > 1e-9 * self.total_area:
            problems.append(
                f"单元面积之和与球面积不符：{self.total_area:.6e} vs "
                f"{4.0 * np.pi * self.geometry.radius**2:.6e}"
            )
        for name in self.filled_fields():
            problems.extend(
                FIELDS[name].check(
                    getattr(self, name), self.geometry.shape, self.nz, check_range=check_range
                )
            )
        return problems

    def summary(self) -> dict[str, Any]:
        """网格概览，用于日志与生成报告。"""
        return {
            "shape": self.shape,
            "resolution": self.resolution,
            "grid_type": self.grid_type,
            "backend": self.backend_name,
            "dtype": self.dtype,
            "nz": self.nz,
            "is_global": self.is_global,
            "cells": self.size,
            "total_area_m2": self.total_area,
            "seed": self.seed,
            "planet": self.planet.name if self.planet is not None else None,
            "filled_fields": len(self.filled_fields()),
            "nbytes": self.nbytes,
        }

    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化字典（字段转 NumPy，便于 Zarr/NetCDF 落盘）。"""
        data: dict[str, Any] = {
            "grid": {
                "grid_type": self.grid_type,
                "resolution": self.resolution,
                "dtype": self.dtype,
                "backend": self.backend,
                "nz": self.nz,
                "seed": self.seed,
                "timestamp": self.timestamp,
                **self._params_for_serialization(),
            },
            "planet_params": self.planet.to_dict() if self.planet is not None else None,
            "fields": self.to_numpy_dict(),
        }
        return data

    def _params_for_serialization(self) -> dict[str, Any]:
        """网格专属参数（经纬：轴与尺寸；立方球 / HEALPix：层次参数）。"""
        if self.grid_type == "latlon":
            return {
                "nlat": self.nlat,
                "nlon": self.nlon,
                "lat": self.lat,
                "lon": self.lon,
            }
        if self.grid_type == "cubed_sphere":
            return {"n_side": self.n_side, "equiangular": self.equiangular}
        return {"nside": self.nside}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GridState:
        """从 :meth:`to_dict` 的输出重建网格状态。"""
        meta = dict(data.get("grid") or {})
        planet = data.get("planet_params")
        state = cls(
            planet_params=PlanetParams.from_dict(planet) if planet else None,
            **meta,
        )
        for name, values in (data.get("fields") or {}).items():
            state.set(name, values)
        return state

    def __repr__(self) -> str:
        return (
            f"GridState(shape={self.shape}, resolution={self.resolution}deg, "
            f"backend={self.backend_name}, fields={len(self.filled_fields())}/{len(FIELDS)})"
        )
