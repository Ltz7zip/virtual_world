"""virtual_world：Python 虚拟 3D 世界生成器。

分层生成管线：行星参数 → 地形 → 气候 → 地表系统，所有阶段共享同一套
经纬度网格数据模型 :class:`~virtual_world.core.grid.GridState`。

模块划分见《项目结构与技术栈》§2：

- ``core``              核心数据模型（网格、行星参数、球面几何、确定性随机数）
- ``terrain``           阶段一：地形生成
- ``radiation``         阶段二：辐射强迫与辐射传输
- ``surface``           阶段三：地表能量平衡
- ``atmosphere``        阶段四：大气动力学
- ``ocean``             阶段四：海洋环流
- ``hydrology_cycle``   阶段五：水循环与降水
- ``classification``    阶段六：气候分类
- ``render``            阶段七：2D/3D 可视化
- ``validation``        阶段八：验证与保真度检验
- ``acceleration``      加速工具（MLX / Numba / 查找表 / 缓存）
- ``io``                数据 IO（Zarr / NetCDF / GeoJSON）
"""

from __future__ import annotations

from .core.grid import (
    FIELDS,
    FieldSpec,
    GridState,
    ResolutionLevel,
    get_resolution_level,
    load_resolution_levels,
)
from .core.planet import PlanetParams, list_presets
from .core.rng import DeterministicRNG

__version__ = "0.1.0"

__all__ = [
    "FIELDS",
    "DeterministicRNG",
    "FieldSpec",
    "GridState",
    "PlanetParams",
    "ResolutionLevel",
    "__version__",
    "get_resolution_level",
    "list_presets",
    "load_resolution_levels",
]
