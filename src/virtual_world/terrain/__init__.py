"""阶段一：地形生成。

板块构造模拟 → 噪声与域扭曲精修 → 水力/热力侵蚀 → 河流网络
（见《地形生成混合方案与程序加速策略》）。

当前实现范围：**第一层 板块构造模拟** 与 **第二层 噪声与扩散精修**。
- :mod:`virtual_world.terrain.voronoi`          球面域扭曲 Voronoi 板块划分
- :mod:`virtual_world.terrain.euler_poles`      欧拉极点与板块运动学
- :mod:`virtual_world.terrain.isostasy`         Airy 均衡与地壳厚度演化（Numba）
- :mod:`virtual_world.terrain.plate_tectonics`  板块构造完整管线编排
- :mod:`virtual_world.terrain.noise_refine`     噪声与扩散精修（fBm/域扭曲/残差融合）
"""

from __future__ import annotations

from . import euler_poles, isostasy, noise_refine, voronoi
from .euler_poles import BoundaryType, pick_euler_poles
from .isostasy import airy_elevation, integrate_crust_thickness
from .noise_refine import (
    DiffusionRefiner,
    Kernel,
    NoiseRefineResult,
    fbm_noise,
    refine_noise,
    ridged_noise,
    select_noise_kernel,
    simplex_noise,
    turbulence_noise,
    warped_noise,
    worley_noise,
)
from .plate_tectonics import (
    REF_CRUST_KM,
    RELAXATION_RATE,
    THICKENING_EFFICIENCY,
    THINNING_EFFICIENCY,
    TectonicFieldResult,
    generate_tectonic_field,
)
from .voronoi import assign_plates, fibonacci_sphere, merge_micro_plates

__all__ = [
    "BoundaryType",
    "DiffusionRefiner",
    "Kernel",
    "NoiseRefineResult",
    "REF_CRUST_KM",
    "RELAXATION_RATE",
    "THICKENING_EFFICIENCY",
    "THINNING_EFFICIENCY",
    "TectonicFieldResult",
    "airy_elevation",
    "assign_plates",
    "euler_poles",
    "fbm_noise",
    "fibonacci_sphere",
    "generate_tectonic_field",
    "integrate_crust_thickness",
    "isostasy",
    "merge_micro_plates",
    "noise_refine",
    "pick_euler_poles",
    "refine_noise",
    "ridged_noise",
    "select_noise_kernel",
    "simplex_noise",
    "turbulence_noise",
    "voronoi",
    "warped_noise",
    "worley_noise",
]
