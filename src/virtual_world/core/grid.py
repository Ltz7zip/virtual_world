"""网格数据模型。

``GridState`` 是整个系统的核心数据结构，承载所有物理场（见《项目结构与技术栈》§3.1）。
设计约束：

- **类型安全**：每个物理量的单位、有效范围、默认值由 :data:`FIELDS` 声明，
  写入时按 :class:`FieldSpec` 校验形状、dtype 与量级。
- **内存高效**：字段为 MLX / NumPy 定长数组，无 Python 对象开销。
- **可序列化**：:meth:`GridState.to_dict` / :meth:`GridState.from_dict` 保留网格元数据与全部字段。
- **可切片**：:meth:`GridState.slice_region` 区域提取、:meth:`GridState.coarsen` 限制算子、
  :meth:`GridState.resample` 重采样（延长算子）。

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
from .backend import DEFAULT_DTYPE
from .planet import PlanetParams, config_dir

# ===== 字段规范 =====


@dataclass(frozen=True)
class FieldSpec:
    """单个物理量的元数据：单位、dtype、有效范围与维度顺序。"""

    name: str
    unit: str
    group: str
    dims: tuple[str, ...] = ("lat", "lon")
    dtype: str = DEFAULT_DTYPE
    valid_range: tuple[float, float] | None = None
    description: str = ""

    def expected_shape(self, nlat: int, nlon: int, nz: int = 1) -> tuple[int, ...]:
        """按网格尺寸计算期望形状。"""
        sizes = {"lat": nlat, "lon": nlon, "depth": nz}
        return tuple(sizes[dim] for dim in self.dims)

    def check(
        self,
        values: Any,
        nlat: int,
        nlon: int,
        nz: int = 1,
        check_range: bool = True,
    ) -> list[str]:
        """校验形状、有限性与取值范围，返回问题描述列表。"""
        arr = backend.to_numpy(values)
        expected = self.expected_shape(nlat, nlon, nz)
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
    """虚拟世界的完整状态：网格元数据 + 全部物理场。"""

    # ===== 网格元数据 =====
    nlat: int
    nlon: int
    resolution: float  # deg
    grid_type: str = "latlon"  # latlon / cubed_sphere / healpix
    backend: str = "auto"  # auto / mlx / numpy
    dtype: str = DEFAULT_DTYPE
    nz: int = 1  # 垂直层数，三维字段（T_soil）的第二维
    seed: int = 0
    planet_params: PlanetParams | Mapping[str, Any] | None = None
    timestamp: str = ""

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
    lat_edges: np.ndarray = field(init=False, repr=False)
    lon_edges: np.ndarray = field(init=False, repr=False)
    cos_lat: np.ndarray = field(init=False, repr=False)
    cell_area: Any = field(init=False, repr=False)
    total_area: float = field(init=False)
    is_global: bool = field(init=False)
    _xp: Any = field(init=False, repr=False, compare=False)
    _backend_name: str = field(init=False, repr=False, compare=False)

    #: 判定全球网格的容差（度数）
    _GLOBAL_TOL: ClassVar[float] = 1e-6

    def __post_init__(self) -> None:
        self.nlat = int(self.nlat)
        self.nlon = int(self.nlon)
        self.nz = int(self.nz)
        self.resolution = float(self.resolution)
        if self.nlat <= 0 or self.nlon <= 0:
            raise ValueError("nlat 与 nlon 必须为正整数")
        if self.nz <= 0:
            raise ValueError("nz 必须为正整数")
        if self.resolution <= 0:
            raise ValueError("resolution 必须为正")
        if self.grid_type != "latlon":
            raise NotImplementedError(f"暂不支持 grid_type={self.grid_type!r}（当前仅实现 latlon）")

        if isinstance(self.planet_params, Mapping):
            self.planet_params = PlanetParams.from_dict(self.planet_params)

        self._xp = backend.resolve_backend(self.backend)
        self._backend_name = backend.backend_name(self._xp)

        if self.lat is None or self.lon is None:
            self.lat = spherical.lat_centers(self.nlat)
            self.lon = spherical.lon_centers(self.nlon)
            self._check_resolution_consistency()
        else:
            self.lat = np.asarray(self.lat, dtype=np.float64)
            self.lon = np.asarray(self.lon, dtype=np.float64)
            if self.lat.shape != (self.nlat,) or self.lon.shape != (self.nlon,):
                raise ValueError("lat/lon 长度必须与 nlat/nlon 一致")
            # 区域提取的 lon 在接缝处经 360 度环绕，因此用模 360 检查间距
            self._check_uniform(self.lat, self.resolution)
            self._check_uniform(self.lon, self.resolution, modulo=360.0)

        self.lat_edges = self._edges_from_centers(self.lat, self.resolution)
        self.lon_edges = self._edges_from_centers(self.lon, self.resolution)
        self.cos_lat = spherical.cos_lat_weights(self.lat)

        radius = self.planet.radius if self.planet is not None else self._earth_radius()
        areas = spherical.cell_areas(self.lat_edges, self.nlon, radius)
        self.cell_area = backend.to_backend(areas, self._backend_name)
        self.total_area = float(np.sum(areas))
        self.is_global = (
            abs(self.nlat * self.resolution - 180.0) < self._GLOBAL_TOL
            and abs(self.nlon * self.resolution - 360.0) < self._GLOBAL_TOL
        )

        for name in FIELDS:
            value = getattr(self, name)
            if value is not None:
                self.set(name, value)

    # ===== 内部工具 =====

    @staticmethod
    def _earth_radius() -> float:
        from . import constants as const

        return const.EARTH_RADIUS

    def _check_resolution_consistency(self) -> None:
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
        """复制元数据构造一个新网格。"""
        meta: dict[str, Any] = {
            "grid_type": self.grid_type,
            "backend": self.backend,
            "dtype": self.dtype,
            "nz": self.nz,
            "seed": self.seed,
            "planet_params": self.planet_params,
            "timestamp": self.timestamp,
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
    def shape(self) -> tuple[int, int]:
        """水平网格形状 (nlat, nlon)。"""
        return (self.nlat, self.nlon)

    @property
    def size(self) -> int:
        """水平网格点数。"""
        return self.nlat * self.nlon

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
    ) -> GridState:
        """按角分辨率 (deg) 构造全球网格。"""
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

    @classmethod
    def from_level(
        cls,
        level: str,
        planet: PlanetParams | None = None,
        seed: int = 0,
        backend_name: str = "auto",
        nz: int = 1,
        levels_path: str | Path | None = None,
    ) -> GridState:
        """按分级分辨率名（preview / coarse / medium / fine / high / ultra）构造网格。"""
        config = get_resolution_level(level, levels_path)
        return cls(
            nlat=config.nlat,
            nlon=config.nlon,
            resolution=config.resolution,
            backend=backend_name,
            nz=nz,
            seed=seed,
            planet_params=planet,
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
            spec.expected_shape(self.nlat, self.nlon, self.nz), self._backend_name, spec.dtype
        )

    def set(self, name: str, values: Any, check_range: bool = True) -> None:
        """写入字段：校验形状/量级并转换为规定的 dtype。

        校验在类型转换**之前**进行，避免越界值被 int8/float32 静默截断。
        """
        spec = self.spec(name)
        if not backend.is_array(values):
            raise TypeError(f"{name} 必须是数组，实际为 {type(values).__name__}")
        problems = spec.check(values, self.nlat, self.nlon, self.nz, check_range=check_range)
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
        """返回最接近给定经纬度的网格索引 (i, j)。"""
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(((self.lon - lon + 180.0) % 360.0) - 180.0)))
        return i, j

    def value_at(self, name: str, lat: float, lon: float) -> float:
        """查询某个网格点上某字段的值。"""
        i, j = self.nearest_index(lat, lon)
        return float(backend.to_numpy(self.get(name))[..., i, j])

    # ===== 诊断与统计 =====

    def global_mean(self, name: str) -> float:
        """面积加权全球平均。"""
        return spherical.global_mean(self.get(name), self.cell_area)

    def global_integral(self, name: str) -> float:
        """面积加权全球积分。"""
        return spherical.global_integral(self.get(name), self.cell_area)

    def zonal_mean(self, name: str) -> np.ndarray:
        """纬向平均剖面（长度 nlat）。"""
        return backend.to_numpy(spherical.zonal_mean(self.get(name)))

    # ===== 网格变换 =====

    def slice_region(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> GridState:
        """区域提取（含经度环绕）。

        极点/日期线上的接缝处经度不再单调，区域网格不适合再做经向差分。
        """
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

    def coarsen(self, factor: int) -> GridState:
        """限制算子：按因子做面积加权块平均（分类字段取最近邻）。

        粗网格格边取细网格格边的每 factor 个取值，因此粗网格单元面积精确等于
        细网格子块面积之和，全球加权平均值在粗化前后严格一致。
        """
        factor = int(factor)
        if factor < 1:
            raise ValueError("factor 必须 >= 1")
        if factor == 1:
            return self.copy()
        if self.nlat % factor or self.nlon % factor:
            raise ValueError(f"factor={factor} 必须同时整除 nlat={self.nlat} 与 nlon={self.nlon}")

        nlat2, nlon2 = self.nlat // factor, self.nlon // factor
        lat_edges2 = self.lat_edges[::factor]
        lon_edges2 = self.lon_edges[::factor]
        result = self._like(
            nlat=nlat2,
            nlon=nlon2,
            resolution=self.resolution * factor,
            lat=self._coords_from_edges(lat_edges2),
            lon=self._coords_from_edges(lon_edges2),
        )

        lat_idx = np.arange(0, self.nlat, factor)
        lon_idx = np.arange(0, self.nlon, factor)
        weights = backend.to_numpy(self.cell_area)
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            spec = FIELDS[name]
            if spec.dtype in NON_INTERPOLATED_DTYPES:
                coarse = self._take(self._take(value, lat_idx, axis=-2), lon_idx, axis=-1)
            else:
                coarse = self._block_mean_weighted(value, weights, factor)
                coarse = coarse.astype(backend.to_dtype(self._xp, spec.dtype))
            result.set(name, coarse)
        return result

    def _block_mean_weighted(self, values: Any, weights: np.ndarray, factor: int) -> Any:
        """按 factor x factor 分块做权重平均（保守粗化）。"""
        xp = self._xp
        leading = values.shape[:-2]
        nlat2, nlon2 = self.nlat // factor, self.nlon // factor
        reshaped = values.reshape(leading + (nlat2, factor, nlon2, factor))
        weights_4d = backend.to_backend(
            weights.reshape(nlat2, factor, nlon2, factor), self._backend_name
        )
        numerator = xp.sum(xp.sum(reshaped * weights_4d, axis=-1), axis=-2)
        denominator = xp.sum(xp.sum(weights_4d, axis=-1), axis=-2)
        return numerator / denominator

    def resample(self, nlat: int, nlon: int) -> GridState:
        """延长算子：双线性重采样（分类字段取最近邻），经向循环。

        目标网格铺满当前网格的纬度与经度范围，因此全球网格重采样后仍为全球网格。
        """
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

    # ===== 校验与序列化 =====

    def validate(self, check_range: bool = True) -> list[str]:
        """校验网格元数据与全部已填充字段，返回问题列表。"""
        problems: list[str] = []
        if self.is_global:
            if abs(self.lat.min() + 90.0) > self.resolution:
                problems.append("全球网格的纬度范围未覆盖 [-90, 90]")
        for name in self.filled_fields():
            problems.extend(
                FIELDS[name].check(
                    getattr(self, name), self.nlat, self.nlon, self.nz, check_range=check_range
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
                "nlat": self.nlat,
                "nlon": self.nlon,
                "resolution": self.resolution,
                "grid_type": self.grid_type,
                "dtype": self.dtype,
                "backend": self.backend,
                "nz": self.nz,
                "seed": self.seed,
                "timestamp": self.timestamp,
                "lat": self.lat,
                "lon": self.lon,
            },
            "planet_params": self.planet.to_dict() if self.planet is not None else None,
            "fields": self.to_numpy_dict(),
        }
        return data

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
