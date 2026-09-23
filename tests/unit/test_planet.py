"""行星参数系统测试。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import constants as const
from virtual_world.core.planet import PlanetParams, list_presets


def test_default_params_are_earth_like() -> None:
    planet = PlanetParams()
    assert planet.radius == pytest.approx(const.EARTH_RADIUS)
    assert planet.rotation_rate == pytest.approx(const.EARTH_ROTATION_RATE)
    assert planet.axial_tilt == pytest.approx(const.EARTH_AXIAL_TILT)
    assert planet.solar_constant == pytest.approx(1361.0)
    assert planet.orbital_period == pytest.approx(const.EARTH_ORBITAL_PERIOD)
    assert planet.eccentricity == pytest.approx(const.EARTH_ECCENTRICITY)
    assert planet.p_surface == pytest.approx(101325.0)
    assert planet.ocean_fraction == pytest.approx(0.71)
    assert planet.ocean_depth_mean == pytest.approx(3700.0)
    assert planet.atm_composition["N2"] == pytest.approx(0.78)
    assert planet.flattening == pytest.approx(0.0)


def test_derived_quantities() -> None:
    planet = PlanetParams.from_preset("earth")
    # 自转周期约 23.93 小时（恒星日）
    assert planet.day_length == pytest.approx(23.9345, abs=1e-3)
    # 由表面重力反演的质量应与地球质量一致（1% 以内）
    assert planet.mass == pytest.approx(const.EARTH_MASS, rel=0.01)
    # 全球年平均入射辐射 S0/4
    assert planet.mean_insolation == pytest.approx(340.25)
    # 大气气体常数应接近干空气值 287 J/(kg K)
    assert planet.gas_constant_air() == pytest.approx(287.05, rel=0.02)
    assert planet.molar_mass_air() == pytest.approx(const.MOLAR_MASS_DRY_AIR, rel=0.05)
    assert planet.polar_radius < planet.equatorial_radius
    assert planet.surface_area == pytest.approx(4 * np.pi * planet.radius**2)


def test_coriolis_and_beta() -> None:
    planet = PlanetParams()
    # 30 度纬度的科里奥利参数约 7.29e-5 1/s
    assert planet.coriolis_parameter(30.0) == pytest.approx(7.292e-5, rel=0.01)
    assert planet.coriolis_parameter(0.0) == pytest.approx(0.0, abs=1e-12)
    beta = planet.beta_parameter(np.array([45.0]))
    assert beta[0] == pytest.approx(2 * 7.292e-5 * np.cos(np.deg2rad(45.0)) / 6.371e6, rel=1e-6)


def test_ellipsoid_radius_correction() -> None:
    planet = PlanetParams.from_preset("earth")
    radii = planet.radius_at_latitude(np.array([0.0, 90.0]))
    assert radii[0] > radii[1]
    assert radii[1] == pytest.approx(planet.polar_radius, rel=1e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"radius": -1.0}, "radius"),
        ({"flattening": 0.9}, "flattening"),
        ({"rotation_rate": 0.0}, "rotation_rate"),
        ({"axial_tilt": 200.0}, "axial_tilt"),
        ({"eccentricity": 1.5}, "eccentricity"),
        ({"solar_constant": 0.0}, "solar_constant"),
        ({"ocean_fraction": 1.5}, "ocean_fraction"),
        ({"ocean_depth_mean": -10.0}, "ocean_depth_mean"),
        ({"atm_composition": {"Xe": 1.0}}, "未知气体"),
        ({"atm_composition": {"N2": 2.0}}, "摩尔分数"),
        ({"atm_composition": {}}, "不能为空"),
    ],
)
def test_invalid_params_raise(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PlanetParams(**overrides)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="未知行星参数字段"):
        PlanetParams.from_dict({"radius": 1.0, "steller_mass": 1.0})


def test_replace_returns_new_instance() -> None:
    base = PlanetParams()
    tilted = base.replace(axial_tilt=45.0, name="high_tilt")
    assert tilted.axial_tilt == 45.0
    assert tilted.name == "high_tilt"
    assert base.axial_tilt == pytest.approx(const.EARTH_AXIAL_TILT)


def test_dict_roundtrip() -> None:
    planet = PlanetParams.from_preset("earth")
    assert PlanetParams.from_dict(planet.to_dict()) == planet


def test_yaml_roundtrip(tmp_path) -> None:
    planet = PlanetParams.from_preset("arid")
    path = planet.to_yaml(tmp_path / "planet_test.yaml")
    assert path.is_file()
    assert PlanetParams.from_yaml(path) == planet


def test_presets_available() -> None:
    presets = list_presets()
    assert {"default", "earth", "arid", "ocean"} <= set(presets)
    for name in presets:
        planet = PlanetParams.from_preset(name)
        assert planet.name == name
        assert planet.validate() == []


def test_missing_preset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        PlanetParams.from_preset("no_such_planet")
