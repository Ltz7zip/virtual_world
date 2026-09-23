"""单位换算工具。

约定：对外接口一律使用以下单位，模块内部若使用其他单位必须显式换算。
    长度 m / 深度 m / 海拔 m
    温度 degC（热力学计算用 K）
    气压 hPa（行星参数用 Pa）
    风速 m/s
    降水 mm/month 或 mm/day
    辐射 W/m^2
"""

from __future__ import annotations

import math
from typing import Any

from . import constants as const

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

SECONDS_PER_DAY = const.SECONDS_PER_DAY
SECONDS_PER_HOUR = const.SECONDS_PER_HOUR
DAYS_PER_YEAR = const.DAYS_PER_YEAR


def deg_to_rad(value: Any) -> Any:
    """度 → 弧度。"""
    return value * DEG_TO_RAD


def rad_to_deg(value: Any) -> Any:
    """弧度 → 度。"""
    return value * RAD_TO_DEG


def celsius_to_kelvin(value: Any) -> Any:
    """摄氏度 → 开尔文。"""
    return value + const.REFERENCE_TEMPERATURE


def kelvin_to_celsius(value: Any) -> Any:
    """开尔文 → 摄氏度。"""
    return value - const.REFERENCE_TEMPERATURE


def pa_to_hpa(value: Any) -> Any:
    """帕斯卡 → 百帕。"""
    return value / 100.0


def hpa_to_pa(value: Any) -> Any:
    """百帕 → 帕斯卡。"""
    return value * 100.0


def mm_to_kg_m2(value: Any) -> Any:
    """毫米水深 → 面密度 kg/m^2（数值相同）。"""
    return value


def kg_m2_s_to_mm_day(value: Any) -> Any:
    """kg/(m^2 s) 水汽通量 → mm/day。"""
    return value * SECONDS_PER_DAY


def mm_day_to_kg_m2_s(value: Any) -> Any:
    """mm/day → kg/(m^2 s)。"""
    return value / SECONDS_PER_DAY


def mm_day_to_mm_month(value: Any, days_in_month: float) -> Any:
    """mm/day → mm/month。"""
    return value * days_in_month


def mm_month_to_mm_day(value: Any, days_in_month: float) -> Any:
    """mm/month → mm/day。"""
    return value / days_in_month


def latent_flux_to_mm_day(flux: Any) -> Any:
    """潜热通量 W/m^2 → 等效水汽通量 mm/day。"""
    return flux / const.LATENT_HEAT_VAPORIZATION * SECONDS_PER_DAY


def mm_day_to_latent_flux(value: Any) -> Any:
    """等效水汽通量 mm/day → 潜热通量 W/m^2。"""
    return value / SECONDS_PER_DAY * const.LATENT_HEAT_VAPORIZATION


def hours_to_seconds(value: Any) -> Any:
    """小时 → 秒。"""
    return value * SECONDS_PER_HOUR


def seconds_to_hours(value: Any) -> Any:
    """秒 → 小时。"""
    return value / SECONDS_PER_HOUR


def days_to_seconds(value: Any) -> Any:
    """天 → 秒。"""
    return value * SECONDS_PER_DAY


def m_s_to_km_h(value: Any) -> Any:
    """m/s → km/h。"""
    return value * 3.6


def angular_velocity_to_day_length(rotation_rate: Any) -> Any:
    """自转角速度 (rad/s) → 自转周期（小时）。"""
    return 2.0 * math.pi / rotation_rate / SECONDS_PER_HOUR


def degrees_to_km(degrees: Any, radius: float) -> Any:
    """球面上的角度 → 弧长（km）。"""
    return deg_to_rad(degrees) * radius / 1000.0
