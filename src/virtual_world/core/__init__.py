"""核心数据模型与工具层。

- :mod:`virtual_world.core.grid`        GridState 网格状态对象
- :mod:`virtual_world.core.planet`      PlanetParams 行星参数
- :mod:`virtual_world.core.spherical`   球面几何工具
- :mod:`virtual_world.core.rng`         确定性随机数
- :mod:`virtual_world.core.constants`   物理常数
- :mod:`virtual_world.core.units`       单位换算
- :mod:`virtual_world.core.backend`     数组后端（MLX / NumPy）
"""

from __future__ import annotations

from . import backend, constants, interpolation, spherical, units
from .grid import (
    FIELDS,
    FieldSpec,
    GridState,
    ResolutionLevel,
    get_resolution_level,
    group_fields,
    load_resolution_levels,
)
from .planet import PlanetParams, config_dir, list_presets
from .rng import DeterministicRNG, derive_seed

__all__ = [
    "FIELDS",
    "DeterministicRNG",
    "FieldSpec",
    "GridState",
    "PlanetParams",
    "ResolutionLevel",
    "backend",
    "config_dir",
    "constants",
    "derive_seed",
    "get_resolution_level",
    "group_fields",
    "interpolation",
    "list_presets",
    "load_resolution_levels",
    "spherical",
    "units",
]
