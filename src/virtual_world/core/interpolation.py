"""插值与重采样工具（与网格对象解耦的数组级 API）。

约定（与 :mod:`virtual_world.core.spherical` 一致）：纬度升序、经度升序，
数组形状 ``(..., nlat, nlon)``；经度方向循环边界，纬度方向零阶外推。

``GridState.resample`` 内部的双线性/最近邻逻辑与此共享同一套坐标映射约定，
但为保持网格状态对象自洽其实现独立；本模块面向 render / 诊断 / 观测点插值。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Order = Literal["nearest", "bilinear"]


def _infer_edges(centers: np.ndarray, edges: np.ndarray | None) -> np.ndarray:
    """由等间距格心外推格边，或直接用给定格边。"""
    centers = np.asarray(centers, dtype=np.float64)
    if edges is not None:
        edges = np.asarray(edges, dtype=np.float64)
        if edges.size != centers.size + 1:
            raise ValueError("edges 长度必须等于 centers 长度 + 1")
        return edges
    if centers.size < 2:
        raise ValueError("centers 至少需要两个点才能推导格边")
    step = float(np.diff(centers)[0])
    if not np.allclose(np.diff(centers), step, atol=1e-9):
        raise ValueError("centers 必须严格等间距")
    return np.concatenate(([centers[0] - step / 2.0], centers + step / 2.0))


def _grid_bilinear_weights(
    src_edges: np.ndarray, dst_centers: np.ndarray, periodic: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把目标格心映射为源格索引：(lo, hi, w_lo, f)。

    非周期轴（纬度）clamp 到有效索引；周期轴（经度）取模。
    """
    n = src_edges.size - 1
    span = (src_edges[-1] - src_edges[0]) / n
    f = (np.asarray(dst_centers, dtype=np.float64) - src_edges[0]) / span - 0.5
    if periodic:
        lo = np.floor(f).astype(np.int64) % n
        hi = (lo + 1) % n
        return lo, hi, np.clip(f - np.floor(f), 0.0, 1.0), f
    lo = np.clip(np.floor(f).astype(np.int64), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    return lo, hi, np.clip(f - np.floor(f), 0.0, 1.0), f


def _bilinear_2d(
    values: np.ndarray,
    i_lo: np.ndarray,
    i_hi: np.ndarray,
    wi: np.ndarray,
    j_lo: np.ndarray,
    j_hi: np.ndarray,
    wj: np.ndarray,
) -> np.ndarray:
    """双线性插值核心（``i/j`` 索引为与目标同形的整数网格）。

    权重需按``values``的维度显式对齐：纬度权重为列向量 ``(..., nlat_out, 1)``、
    经度权重为行向量 ``(..., nlon_out)``，否则超过二维的输入会广播失败。
    """
    v00 = values[..., i_lo, :][..., :, j_lo]
    v01 = values[..., i_lo, :][..., :, j_hi]
    v10 = values[..., i_hi, :][..., :, j_lo]
    v11 = values[..., i_hi, :][..., :, j_hi]
    lead = (1,) * (values.ndim - 2)
    wi_col = wi.reshape(*lead, -1, 1)
    wj_row = wj.reshape(*lead, -1)
    top = v00 * (1.0 - wj_row) + v01 * wj_row
    bottom = v10 * (1.0 - wj_row) + v11 * wj_row
    return np.asarray(top * (1.0 - wi_col) + bottom * wi_col)


def regrid(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    order: Order = "bilinear",
    src_lat_edges: np.ndarray | None = None,
    src_lon_edges: np.ndarray | None = None,
) -> np.ndarray:
    """把 ``(..., nlat, nlon)`` 数组重采样到新网格（格心等间距）。

    ``order="bilinear"`` 用于连续物理量；``"nearest"`` 用于分类编码
    （不会产生新类别）。经度循环边界、纬度零阶外推。
    """
    values = np.asarray(values)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    if values.shape[-2:] != (src_lat.size, src_lon.size):
        raise ValueError(
            f"values 末尾两维应为 ({src_lat.size}, {src_lon.size})，实际 {values.shape}"
        )

    lat_edges = _infer_edges(src_lat, src_lat_edges)
    lon_edges = _infer_edges(src_lon, src_lon_edges)

    i_lo, i_hi, wi, _ = _grid_bilinear_weights(lat_edges, dst_lat, periodic=False)
    j_lo, j_hi, wj, _ = _grid_bilinear_weights(lon_edges, dst_lon, periodic=True)

    if order == "nearest":
        i_near = np.where(wi < 0.5, i_lo, i_hi)
        j_near = np.where(wj < 0.5, j_lo, j_hi)
        return values[..., i_near, :][..., :, j_near]
    if order != "bilinear":
        raise ValueError(f"未知插值阶数 {order!r}（可选 'nearest' / 'bilinear'）")
    return _bilinear_2d(values, i_lo, i_hi, wi, j_lo, j_hi, wj)


def value_at_coords(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    order: Order = "bilinear",
) -> np.ndarray:
    """在任意（不必等间距的）查询点上取值，返回与查询点同形的数组。"""
    values = np.asarray(values)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    q_lat = np.asarray(query_lat, dtype=np.float64)
    q_lon = np.asarray(query_lon, dtype=np.float64)
    if values.shape[-2:] != (src_lat.size, src_lon.size):
        raise ValueError(
            f"values 末尾两维应为 ({src_lat.size}, {src_lon.size})，实际 {values.shape}"
        )

    lat_edges = _infer_edges(src_lat, None)
    lon_edges = _infer_edges(src_lon, None)
    nlon = src_lon.size

    i_lo, i_hi, wi, f_i = _grid_bilinear_weights(lat_edges, q_lat, periodic=False)
    j_lo, j_hi, wj, f_j = _grid_bilinear_weights(lon_edges, q_lon, periodic=True)

    # 逐查询点收集：形状 (lead..., nq)
    lead = values.shape[:-2]
    nq = q_lat.size
    values2d = values.reshape(*lead, src_lat.size, nlon)

    def gather1d(index: np.ndarray, axis0_idx: np.ndarray | None = None) -> np.ndarray:
        # index: (nq,) 列索引；axis0_idx 为 (nq,) 行索引（None 表示整点收集）
        rows = values2d[..., axis0_idx, :] if axis0_idx is not None else values2d
        out = np.empty(lead + (nq,), dtype=values.dtype)
        for k in range(nq):
            out[..., k] = rows[..., :, index[k]]
        return out

    if order == "nearest":
        i_near = np.where(f_i < 0.5, i_lo, i_hi)
        j_near = np.where(f_j < 0.5, j_lo, j_hi)
        return gather1d(j_near, i_near)
    if order != "bilinear":
        raise ValueError(f"未知插值阶数 {order!r}（可选 'nearest' / 'bilinear'）")

    v00 = gather1d(j_lo, i_lo)
    v01 = gather1d(j_hi, i_lo)
    v10 = gather1d(j_lo, i_hi)
    v11 = gather1d(j_hi, i_hi)
    top = v00 * (1.0 - wj) + v01 * wj
    bottom = v10 * (1.0 - wj) + v11 * wj
    return np.asarray(top * (1.0 - wi) + bottom * wi)


__all__ = ["Order", "regrid", "value_at_coords"]