"""核心数据模型与工具层。

- :mod:`virtual_world.core.grid`        GridState 网格状态对象
- :mod:`virtual_world.core.planet`      PlanetParams 行星参数
- :mod:`virtual_world.core.spherical`   球面几何工具（等经纬度网格）
- :mod:`virtual_world.core.cubed_sphere` 立方球网格（方案 §1.5 推荐）
- :mod:`virtual_world.core.healpix`     HEALPix 等面积层次化网格（方案 §1.5 备选）
- :mod:`virtual_world.core.geometry`    三种网格的统一几何接口
- :mod:`virtual_world.core.operators`   非结构网格切平面算子
- :mod:`virtual_world.core.rng`         确定性随机数
- :mod:`virtual_world.core.constants`   物理常数
- :mod:`virtual_world.core.units`       单位换算
- :mod:`virtual_world.core.backend`     数组后端（MLX / NumPy）
"""

from __future__ import annotations

from . import backend, constants, interpolation, operators, spherical, units
from .cubed_sphere import CubedSphere
from .geometry import (
    GRID_TYPES,
    CubedSphereGeometry,
    GridGeometry,
    HealpixGeometry,
    LatLonGeometry,
    make_geometry,
)
from .grid import (
    FIELDS,
    FieldSpec,
    GridState,
    ResolutionLevel,
    get_resolution_level,
    group_fields,
    load_resolution_levels,
)
from .healpix import HealpixGrid
from .operators import LocalInterpolator, TangentStencil
from .planet import PlanetParams, config_dir, list_presets
from .rng import DeterministicRNG, derive_seed

__all__ = [
    "FIELDS",
    "GRID_TYPES",
    "CubedSphere",
    "CubedSphereGeometry",
    "DeterministicRNG",
    "FieldSpec",
    "GridGeometry",
    "GridState",
    "HealpixGeometry",
    "HealpixGrid",
    "LatLonGeometry",
    "LocalInterpolator",
    "PlanetParams",
    "ResolutionLevel",
    "TangentStencil",
    "backend",
    "config_dir",
    "constants",
    "derive_seed",
    "get_resolution_level",
    "group_fields",
    "interpolation",
    "list_presets",
    "load_resolution_levels",
    "make_geometry",
    "operators",
    "spherical",
    "units",
]
