"""行星参数系统。

``PlanetParams`` 是一个行星的全部物理参数，也是整条因果链的起点：

    行星参数 → 辐射强迫 → 地表能量 → 大气动力学 → 风应力 → 洋流
        ↓                                              ↓
        地形生成 → 气候场 → 降水 → 生物群系 → 土壤 → 农业

约定：外部接口单位统一为 m / deg / W m^-2 / Pa / day，内部按 SI 计算。
``radius`` 作为参考半径（地球配置取平均半径），扁率不为 0 时按旋转椭球
``a = radius``、``b = a(1-f)`` 处理。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml

from . import constants as const
from . import spherical
from .units import DEG_TO_RAD, angular_velocity_to_day_length

CONFIG_DIR_NAME = "configs"
PLANET_CONFIG_PREFIX = "planet_"
_CONFIG_ENV_VAR = "VIRTUAL_WORLD_CONFIGS"

_DEFAULT_COMPOSITION: dict[str, float] = {
    "N2": 0.78,
    "O2": 0.21,
    "CO2": 0.0004,
    "H2O": 0.01,
}


def config_dir() -> Path:
    """定位 ``configs/`` 目录。

    优先级：环境变量 ``VIRTUAL_WORLD_CONFIGS`` → 包所在项目的 configs/ → 当前工作目录。
    """
    env = os.environ.get(_CONFIG_ENV_VAR)
    if env:
        path = Path(env).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"{_CONFIG_ENV_VAR} 指向的目录不存在: {path}")
        return path
    for parent in Path(__file__).resolve().parents:
        candidate = parent / CONFIG_DIR_NAME
        if candidate.is_dir() and any(candidate.glob(f"{PLANET_CONFIG_PREFIX}*.yaml")):
            return candidate
    return Path.cwd() / CONFIG_DIR_NAME


@dataclass
class PlanetParams:
    """行星物理参数（见《项目结构与技术栈》§3.2）。"""

    # ===== 几何 =====
    radius: float = const.EARTH_RADIUS  # 参考半径 (m)
    flattening: float = 0.0  # 扁率 f = (a-b)/a
    gravity: float = const.EARTH_GRAVITY  # 表面重力加速度 (m/s^2)

    # ===== 自转与轨道 =====
    rotation_rate: float = const.EARTH_ROTATION_RATE  # 自转角速度 (rad/s)
    axial_tilt: float = const.EARTH_AXIAL_TILT  # 轴倾角 (deg)
    orbital_period: float = const.EARTH_ORBITAL_PERIOD  # 公转周期 (day)
    eccentricity: float = const.EARTH_ECCENTRICITY  # 轨道偏心率
    perihelion_arg: float = const.EARTH_PERIHELION_ARG  # 近日点幅角 (deg)
    semi_major_axis: float = const.ASTRONOMICAL_UNIT  # 轨道半长轴 (m)

    # ===== 恒星 =====
    solar_constant: float = const.SOLAR_CONSTANT  # 恒星常数 (W/m^2)
    stellar_mass: float = const.SOLAR_MASS  # 恒星质量 (kg)

    # ===== 大气 =====
    p_surface: float = 101325.0  # 表面气压 (Pa)
    atm_composition: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_COMPOSITION))
    greenhouse_factor: float = 1.0  # 温室效应因子

    # ===== 海洋 =====
    ocean_fraction: float = 0.71  # 海洋面积占比
    ocean_depth_mean: float = 3700.0  # 平均海洋深度 (m)

    # ===== 标识 =====
    name: str = "default"

    #: 允许的大气成分（其余成分名视为配置错误）
    KNOWN_GASES: ClassVar[frozenset[str]] = frozenset(const.MOLAR_MASS)

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "atm_composition":
                if not isinstance(value, Mapping):
                    raise TypeError("atm_composition 必须是 {气体名: 摩尔分数} 映射")
                object.__setattr__(
                    self, "atm_composition", {str(k): float(v) for k, v in value.items()}
                )
            elif f.name == "name":
                object.__setattr__(self, "name", str(value))
            else:
                object.__setattr__(self, f.name, float(value))

        problems = self.validate()
        if problems:
            raise ValueError("行星参数不合法:\n  - " + "\n  - ".join(problems))

    # ===== 校验 =====

    def validate(self) -> list[str]:
        """返回所有不合法参数的说明，合法时返回空列表。"""
        problems: list[str] = []
        if self.radius <= 0:
            problems.append("radius 必须 > 0")
        if not 0.0 <= self.flattening < 0.5:
            problems.append("flattening 必须落在 [0, 0.5)")
        if self.gravity <= 0:
            problems.append("gravity 必须 > 0")
        if self.rotation_rate <= 0:
            problems.append("rotation_rate 必须 > 0（自转周期由它导出）")
        if not 0.0 <= self.axial_tilt <= 180.0:
            problems.append("axial_tilt 必须落在 [0, 180] deg")
        if self.orbital_period <= 0:
            problems.append("orbital_period 必须 > 0")
        if not 0.0 <= self.eccentricity < 1.0:
            problems.append("eccentricity 必须落在 [0, 1)")
        if self.semi_major_axis <= 0:
            problems.append("semi_major_axis 必须 > 0")
        if self.solar_constant <= 0:
            problems.append("solar_constant 必须 > 0")
        if self.stellar_mass <= 0:
            problems.append("stellar_mass 必须 > 0")
        if self.p_surface <= 0:
            problems.append("p_surface 必须 > 0")
        if not 0.0 <= self.greenhouse_factor <= 10.0:
            problems.append("greenhouse_factor 必须落在 [0, 10]")
        if not 0.0 <= self.ocean_fraction <= 1.0:
            problems.append("ocean_fraction 必须落在 [0, 1]")
        if self.ocean_depth_mean < 0:
            problems.append("ocean_depth_mean 必须 >= 0")

        if not self.atm_composition:
            problems.append("atm_composition 不能为空")
        else:
            unknown = sorted(set(self.atm_composition) - self.KNOWN_GASES)
            if unknown:
                problems.append(f"atm_composition 含未知气体: {', '.join(unknown)}")
            bad = sorted(k for k, v in self.atm_composition.items() if not 0.0 <= v <= 1.0)
            if bad:
                problems.append(f"atm_composition 摩尔分数须落在 [0, 1]: {', '.join(bad)}")
            total = sum(self.atm_composition.values())
            # H2O 为可变量，允许总和略大于 1
            if not 0.0 < total <= 1.05:
                problems.append(f"atm_composition 摩尔分数总和 {total:.4f} 超出 (0, 1.05]")
        return problems

    # ===== 派生量 =====

    @property
    def day_length(self) -> float:
        """自转周期（小时）。"""
        return angular_velocity_to_day_length(self.rotation_rate)

    @property
    def equatorial_radius(self) -> float:
        """赤道半径 a (m)。"""
        return self.radius

    @property
    def polar_radius(self) -> float:
        """极半径 b = a(1-f) (m)。"""
        return self.radius * (1.0 - self.flattening)

    @property
    def volume(self) -> float:
        """旋转椭球体积 (m^3)。"""
        return 4.0 / 3.0 * np.pi * self.equatorial_radius**2 * self.polar_radius

    @property
    def surface_area(self) -> float:
        """表面积（球近似）(m^2)。"""
        return 4.0 * np.pi * self.radius**2

    @property
    def mass(self) -> float:
        """由表面重力反演的行星质量 (kg)。"""
        return self.gravity * self.radius**2 / const.GRAVITATIONAL_CONSTANT

    @property
    def mean_insolation(self) -> float:
        """全球年平均入射辐射 ``S0/4`` (W/m^2)，验证第三层行星能量平衡的基准。"""
        return self.solar_constant / 4.0

    @property
    def stellar_luminosity(self) -> float:
        """恒星光度 ``L = 4*pi*d^2*S0`` (W)。"""
        return 4.0 * np.pi * self.semi_major_axis**2 * self.solar_constant

    @property
    def water_vapour_fraction(self) -> float:
        """大气水汽摩尔分数。"""
        return float(self.atm_composition.get("H2O", 0.0))

    def molar_mass_air(self) -> float:
        """大气平均摩尔质量 (kg/mol)，按摩尔分数加权。"""
        total = sum(self.atm_composition.values())
        weighted = sum(const.MOLAR_MASS[gas] * frac for gas, frac in self.atm_composition.items())
        return float(weighted / total)

    def gas_constant_air(self) -> float:
        """大气气体常数 ``R = R_univ / M`` (J kg^-1 K^-1)。"""
        return float(const.UNIVERSAL_GAS_CONSTANT / self.molar_mass_air())

    def radius_at_latitude(self, lat_deg: Any) -> np.ndarray:
        """椭球体地心半径修正 (m)。

        ``r(phi) = a*b / sqrt(a^2 sin^2 phi + b^2 cos^2 phi)``
        赤道处为 a，极点处为 b，用于 3D 渲染与局地几何。
        """
        a = self.equatorial_radius
        b = self.polar_radius
        sin_phi = np.sin(DEG_TO_RAD * np.asarray(lat_deg, dtype=np.float64))
        cos_phi = np.cos(DEG_TO_RAD * np.asarray(lat_deg, dtype=np.float64))
        return a * b / np.sqrt(a**2 * sin_phi**2 + b**2 * cos_phi**2)

    def coriolis_parameter(self, lat_deg: Any) -> np.ndarray:
        """科里奥利参数 ``f = 2 Omega sin(phi)`` (1/s)。"""
        return spherical.coriolis_parameter(self.rotation_rate, lat_deg)

    def beta_parameter(self, lat_deg: Any) -> np.ndarray:
        """Rossby 参数 ``beta = 2 Omega cos(phi) / R`` (1/(m s))。"""
        return spherical.beta_parameter(self.rotation_rate, lat_deg, self.radius)

    # ===== 序列化 =====

    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanetParams:
        """由字典构造，未知字段直接报错以便尽早发现配置拼写问题。"""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"未知行星参数字段: {', '.join(unknown)}")
        return cls(**dict(data))

    def to_yaml(self, path: str | Path) -> Path:
        """写入 YAML 配置文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> PlanetParams:
        """从 YAML 配置文件读取。"""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, Mapping):
            raise ValueError(f"{path} 内容不是合法的行星参数映射")
        return cls.from_dict(data)

    @classmethod
    def from_preset(cls, name: str, directory: str | Path | None = None) -> PlanetParams:
        """按预设名读取 ``configs/planet_<name>.yaml``。"""
        base = Path(directory) if directory else config_dir()
        path = base / f"{PLANET_CONFIG_PREFIX}{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"未找到预设 {name!r}: {path}")
        return cls.from_yaml(path)

    def replace(self, **overrides: Any) -> PlanetParams:
        """基于当前参数派生一个新参数集（常用于参数扫描）。"""
        data = self.to_dict()
        data.update(overrides)
        return PlanetParams.from_dict(data)


def list_presets(directory: str | Path | None = None) -> list[str]:
    """列出可用的行星参数预设名。"""
    base = Path(directory) if directory else config_dir()
    return sorted(p.stem[len(PLANET_CONFIG_PREFIX) :] for p in base.glob(f"{PLANET_CONFIG_PREFIX}*.yaml"))
