"""球面几何工具。

坐标约定（设计方案 §2.2）：
    纬度 -90（南极）.. +90（北极），升序；经度 -180 .. +180，升序。
    网格索引 ``(i, j)`` = (纬度索引, 经度索引)，字段形状 ``(nlat, nlon)``。
    经向为**循环边界**，纬向为**极点约束**（默认边界外推）。

梯度/散度/涡度函数默认输入为**全球网格**（经向循环）。区域切片网格上调用这些
函数会在经向接缝处产生误差，仅适用于诊断与出图。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import backend
from .units import DEG_TO_RAD

_TWO_PI = 2.0 * math.pi


# ===== 坐标 =====


def lat_centers(nlat: int) -> np.ndarray:
    """纬度中心点 (deg)，从南到北，-90+dlat/2 .. 90-dlat/2。"""
    dlat = 180.0 / nlat
    return -90.0 + dlat * (np.arange(nlat, dtype=np.float64) + 0.5)


def lon_centers(nlon: int) -> np.ndarray:
    """经度中心点 (deg)，从西到东，-180+dlon/2 .. 180-dlon/2。"""
    dlon = 360.0 / nlon
    return -180.0 + dlon * (np.arange(nlon, dtype=np.float64) + 0.5)


def lat_edges(nlat: int) -> np.ndarray:
    """纬度格边 (deg)，长度为 nlat+1，含两极。"""
    dlat = 180.0 / nlat
    return -90.0 + dlat * np.arange(nlat + 1, dtype=np.float64)


def lon_edges(nlon: int) -> np.ndarray:
    """经度格边 (deg)，长度为 nlon+1。"""
    dlon = 360.0 / nlon
    return -180.0 + dlon * np.arange(nlon + 1, dtype=np.float64)


def cos_lat_weights(lat_deg: np.ndarray) -> np.ndarray:
    """纬度权重 cos(phi)，用于面积加权。"""
    return np.cos(np.deg2rad(lat_deg))


def _align(array: Any, xp: Any) -> Any:
    """把坐标/权重数组对齐到字段所在后端。"""
    return backend.to_backend(array, backend.backend_name(xp))


# ===== 面积与积分 =====


def ring_areas(lat_edges_deg: np.ndarray, radius: float) -> np.ndarray:
    """每一条完整纬度环带的面积 (m^2)，形状 (nlat,)。

    ``A_i = 2*pi*R^2*(sin(phi_2) - sin(phi_1))``
    """
    edges = np.deg2rad(np.asarray(lat_edges_deg, dtype=np.float64))
    return _TWO_PI * radius**2 * (np.sin(edges[1:]) - np.sin(edges[:-1]))


def cell_areas(lat_edges_deg: np.ndarray, nlon: int, radius: float) -> np.ndarray:
    """网格单元面积 (m^2)，形状 (nlat, nlon)，每行相同。"""
    rings = ring_areas(lat_edges_deg, radius)
    return np.repeat((rings / nlon)[:, None], nlon, axis=1)


def total_area(lat_edges_deg: np.ndarray, radius: float) -> float:
    """全球总面积 (m^2)，球面上等于 4*pi*R^2。"""
    return float(np.sum(ring_areas(lat_edges_deg, radius)))


def global_mean(field: Any, area: Any) -> float:
    """面积加权全球平均。"""
    xp = backend.backend_of(field)
    a = _align(area, xp)
    return float(xp.sum(field * a) / xp.sum(a))


def global_integral(field: Any, area: Any) -> float:
    """面积加权全球积分。"""
    xp = backend.backend_of(field)
    a = _align(area, xp)
    return float(xp.sum(field * a))


def zonal_mean(field: Any) -> Any:
    """纬向平均（对经度取平均，形状 (nlat,)）。"""
    xp = backend.backend_of(field)
    return xp.mean(field, axis=-1)


def area_weighted_mean(field: Any, area: Any, axis: int = -1) -> Any:
    """沿给定轴做面积加权平均。"""
    xp = backend.backend_of(field)
    a = _align(area, xp)
    return xp.sum(field * a, axis=axis) / xp.sum(a, axis=axis)


# ===== 科里奥利参数 =====


def coriolis_parameter(rotation_rate: float, lat_deg: Any) -> np.ndarray:
    """科里奥利参数 ``f = 2*Omega*sin(phi)`` (1/s)。"""
    return 2.0 * rotation_rate * np.sin(DEG_TO_RAD * np.asarray(lat_deg, dtype=np.float64))


def beta_parameter(rotation_rate: float, lat_deg: Any, radius: float) -> np.ndarray:
    """Rossby 参数 ``beta = df/dy = 2*Omega*cos(phi)/R`` (1/(m s))。"""
    cos_phi = np.cos(DEG_TO_RAD * np.asarray(lat_deg, dtype=np.float64))
    return 2.0 * rotation_rate * cos_phi / radius


# ===== 差分算子（全球网格） =====


def zonal_gradient(field: Any, lat_deg: np.ndarray, radius: float) -> Any:
    """经向梯度 ``(1/(R cos phi)) d/dlambda``，经向循环，形状不变。"""
    xp = backend.backend_of(field)
    nlon = field.shape[-1]
    dlon = _TWO_PI / nlon
    east = xp.roll(field, -1, axis=-1)
    west = xp.roll(field, 1, axis=-1)
    cos_phi = _align(cos_lat_weights(lat_deg), xp)
    shape = (1,) * (field.ndim - 2) + (cos_phi.shape[0], 1)
    return (east - west) / (2.0 * dlon * radius * cos_phi.reshape(shape))


def meridional_gradient(field: Any, radius: float) -> Any:
    """经向梯度 ``(1/R) d/dphi``，极点使用边界外推，形状不变。"""
    xp = backend.backend_of(field)
    nlat = field.shape[-2]
    dlat = math.pi / nlat
    padded = xp.concatenate([field[..., :1, :], field, field[..., -1:, :]], axis=-2)
    return (padded[..., 2:, :] - padded[..., :-2, :]) / (2.0 * dlat * radius)


def divergence(u: Any, v: Any, lat_deg: np.ndarray, radius: float) -> Any:
    """水平散度 ``1/(R cos phi) [du/dlambda + d(v cos phi)/dphi]``。"""
    xp = backend.backend_of(u, v)
    cos_phi = _align(cos_lat_weights(lat_deg), xp)
    shape = (1,) * (u.ndim - 2) + (cos_phi.shape[0], 1)
    cos_2d = cos_phi.reshape(shape)
    term = meridional_gradient(v * cos_2d, radius)
    return zonal_gradient(u, lat_deg, radius) + term / cos_2d


def vorticity(u: Any, v: Any, lat_deg: np.ndarray, radius: float) -> Any:
    """相对涡度 ``zeta = 1/(R cos phi) [dv/dlambda - d(u cos phi)/dphi]``。"""
    xp = backend.backend_of(u, v)
    cos_phi = _align(cos_lat_weights(lat_deg), xp)
    shape = (1,) * (u.ndim - 2) + (cos_phi.shape[0], 1)
    cos_2d = cos_phi.reshape(shape)
    term = meridional_gradient(u * cos_2d, radius)
    return (zonal_gradient(v, lat_deg, radius) - term) / cos_2d


# ===== 坐标变换与距离 =====


def haversine_distance(
    lat1: Any, lon1: Any, lat2: Any, lon2: Any, radius: float
) -> np.ndarray:
    """大圆距离 (m)，支持广播。"""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def latlon_to_xyz(lat_deg: Any, lon_deg: Any, radius: float) -> np.ndarray:
    """经纬度 → 笛卡尔坐标 (m)，形状 (..., 3)。"""
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    cos_phi = np.cos(phi)
    x = radius * cos_phi * np.cos(lam)
    y = radius * cos_phi * np.sin(lam)
    z = radius * np.sin(phi)
    return np.stack([x, y, z], axis=-1)


def xyz_to_latlon(xyz: Any, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """笛卡尔坐标 → (纬度, 经度) 度数，输入形状 (..., 3)。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    norm = np.sqrt(x**2 + y**2 + z**2)
    lat = np.rad2deg(np.arcsin(np.clip(z / norm, -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(y, x))
    return lat, lon


def latitude_of_max(profile: Any, lat_deg: np.ndarray) -> float:
    """返回剖面最大值所在的纬度 (deg)，用于诊断 ITCZ、副热带高压等特征位置。"""
    values = backend.to_numpy(profile)
    if values.ndim != 1 or values.shape[0] != len(lat_deg):
        raise ValueError("剖面必须是一维且长度等于纬度点数")
    return float(np.asarray(lat_deg)[int(np.argmax(values))])
