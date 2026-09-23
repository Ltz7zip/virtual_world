"""第三层·河流网络生成（《第三层完善：侵蚀模拟》§四、§5.2）。

管线：Priority-Flood 洼地填充 → D8 最陡下降流向（ESRI 1–128 编码）→
拓扑排序汇流累积 → 汇水面积 → 河流网络提取 → 水力几何河宽。

关键约束（§1.4.2、§4.2）：**D8 要求高程场无洼地**——否则闭合盆地会截留水流，
汇流累积停在半路，河网必然断裂。因此先做 Priority-Flood 填充
（``filled = max(elev, 出流侧水位)``，即方案 §4.2 的"设为当前最高水位"）。

填充会在盆地内形成**平地**（水面高于原地面）。平地没有严格下坡邻居，
D8 会给出二义方向甚至环。这里采用标准的 **Priority-Flood + ε** 变体：
水位抬升取 ``min_level + ε``，使填充面自出流口向外**严格单调**上升，于是
每个非出流单元都必然存在严格下坡邻居 ⇒ D8 无平地、**无环**（严格递减的
有向图不可能成环）。

ε 的代价是被填充区域整体抬高 ``路径长度 × ε``，因此取 ``1e-4 m``（0.1 mm）：
比浮点噪声与"准平地"的毫厘差异（实测 ~1e-6 m）高两个量级，又在 1e6 单元的
极端路径上仍不超过 100 m、常规路径上不足 1 cm，相对千米级地形起伏可忽略。
未被填充的单元高程**逐位不变**。

出流口（seed）取海洋单元与纬度首末行：本项目纬向非循环（极点约束），
纬度边界即"域外"，单元从那里离域。

汇流累积用 Kahn 拓扑排序（§4.3）：入度为 0 的单元先出队，逐层累加到下游。
若仍有陆地单元未出队则存在环——这是调用方构造非法流向时的输入错误，
直接报错而不是静默返回错值。

经纬网格的经向格距随 ``cos(lat)`` 收缩（:func:`core.spherical.cell_spacing`），
因此坡度按**度规距离**计算，汇水面积按真实单元面积累加。河道阈值用**上游单元数**
（§4.4"汇流累积量超过阈值"）：绝对面积阈值（1–10 km²）是 10–30 m 分辨率 DEM 的
经验值，在 1° 网格上一个单元就是 1e12 m² 量级，只能按单元数表达；
:func:`drainage_area` 同时给出物理汇水面积供报告与后续水文模块使用。
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

from ..core import spherical
from ..core.constants import EARTH_RADIUS

#: 形成河道的默认汇流累积阈值（上游单元数，含自身）
DEFAULT_RIVER_CELLS = 50
#: Priority-Flood 的 ε 抬升量 (m)：保证填充面严格单调 ⇒ D8 无环（见模块 docstring）
FILL_EPS_M = 1.0e-4
#: 水力几何 ``w = C·√A`` 的系数（SI 下 A 用 m^2、w 用 m）
RIVER_WIDTH_COEFFICIENT = 0.01
RIVER_WIDTH_MIN_M = 1.0
RIVER_WIDTH_MAX_M = 5000.0

#: ESRI D8 方向码（1 E / 2 SE / 4 S / 8 SW / 16 W / 32 NW / 64 N / 128 NE）
D8_CODES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
#: 各方向码对应的 (di, dj)：dj 向东为正、di 向北为正、经度循环
_D8_DI = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
_D8_DJ = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
_D8_CODE = np.array(D8_CODES, dtype=np.int64)
#: 方向码 → 网格偏移，供调用方/测试解算流向
D8_OFFSETS: dict[int, tuple[int, int]] = {
    code: (int(di), int(dj)) for code, di, dj in zip(D8_CODES, _D8_DI, _D8_DJ, strict=True)
}


# ===== 二元最小堆（Numba 内实现，避免依赖 numba 的 heapq 支持）=====


@njit(cache=True, inline="always")
def _heap_push(keys: np.ndarray, idxs: np.ndarray, n: int, key: float, idx: int) -> int:
    """压入 ``(key, idx)``，返回新的堆大小。"""
    i = n
    keys[i] = key
    idxs[i] = idx
    n += 1
    while i > 0:
        p = (i - 1) >> 1
        if keys[p] <= keys[i]:
            break
        tk = keys[p]
        keys[p] = keys[i]
        keys[i] = tk
        ti = idxs[p]
        idxs[p] = idxs[i]
        idxs[i] = ti
        i = p
    return n


@njit(cache=True, inline="always")
def _heap_pop(keys: np.ndarray, idxs: np.ndarray, n: int) -> tuple[float, int, int]:
    """弹出最小元素，返回 ``(key, idx, 新堆大小)``。"""
    key = keys[0]
    idx = idxs[0]
    n -= 1
    keys[0] = keys[n]
    idxs[0] = idxs[n]
    i = 0
    while True:
        left = 2 * i + 1
        right = left + 1
        small = i
        if left < n and keys[left] < keys[small]:
            small = left
        if right < n and keys[right] < keys[small]:
            small = right
        if small == i:
            break
        tk = keys[small]
        keys[small] = keys[i]
        keys[i] = tk
        ti = idxs[small]
        idxs[small] = idxs[i]
        idxs[i] = ti
        i = small
    return key, idx, n


# ===== 洼地填充（§4.2）=====


@njit(cache=True)
def _priority_flood(elev: np.ndarray, ocean: np.ndarray, eps: float) -> np.ndarray:
    """Priority-Flood + ε（Barnes 等）：把低于出流侧水位的单元抬到 ``水位 + ε``。"""
    nlat, nlon = elev.shape
    filled = elev.copy()
    visited = np.zeros((nlat, nlon), dtype=np.uint8)
    cap = 2 * nlat * nlon + 8
    keys = np.empty(cap, dtype=np.float64)
    idxs = np.empty(cap, dtype=np.int64)
    n = 0
    # 1：出流口入队（海洋 + 纬度边界行）
    for i in range(nlat):
        is_boundary = (i == 0) or (i == nlat - 1)
        for j in range(nlon):
            if ocean[i, j] or is_boundary:
                visited[i, j] = 1
                n = _heap_push(keys, idxs, n, elev[i, j], i * nlon + j)
    # 2–4：按水位从低到高扩展，低于水位的单元抬到 水位 + eps
    while n > 0:
        key, k, n = _heap_pop(keys, idxs, n)
        ci = k // nlon
        cj = k % nlon
        level = key + eps
        for di in range(-1, 2):
            ni = ci + di
            if ni < 0 or ni >= nlat:
                continue
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                nj = cj + dj
                if nj < 0:
                    nj += nlon
                elif nj >= nlon:
                    nj -= nlon
                if visited[ni, nj] != 0:
                    continue
                visited[ni, nj] = 1
                h = elev[ni, nj]
                if level > h:
                    h = level
                filled[ni, nj] = h
                n = _heap_push(keys, idxs, n, h, ni * nlon + nj)
    return filled


def fill_pits(
    elevation: np.ndarray, is_ocean: np.ndarray, *, epsilon: float = FILL_EPS_M
) -> np.ndarray:
    """Priority-Flood 洼地填充（§4.2），返回**无洼地**高程场。

    ``filled >= elevation`` 恒成立，且高于出流侧水位的单元逐位不变。
    ``epsilon`` 为严格单调抬升量：它保证填充面自出流口向外严格上升，从而
    :func:`flow_directions` 的 D8 既无平地歧义也**不可能成环**；代价是被填充
    区域沿路径累计抬高 ``路径长度 × epsilon``（见模块 docstring 的量级说明）。
    """
    elev, ocean = _as_arrays(elevation, is_ocean)
    if epsilon < 0.0:
        raise ValueError(f"epsilon 不能为负，实际为 {epsilon}")
    return np.asarray(
        _priority_flood(np.ascontiguousarray(elev), np.ascontiguousarray(ocean), float(epsilon)),
        dtype=np.float64,
    )


# ===== D8 流向（§4.1）=====


@njit(cache=True)
def _d8_kernel(
    filled: np.ndarray, ocean: np.ndarray, dlat_m: float, dlon_m: np.ndarray, out: np.ndarray
) -> None:
    """逐格取最陡下降方向（8 邻域、经度循环、纬向非循环）。

    :func:`fill_pits` 的 ε 抬升保证非出流单元必有严格下坡邻居，因此这里
    不需要平地/洼地特判——出流码 0 只可能出现在海洋与纬度边界行。
    """
    nlat, nlon = filled.shape
    for i in range(nlat):
        for j in range(nlon):
            if ocean[i, j]:
                out[i, j] = 0
                continue
            z = filled[i, j]
            best = 0.0
            code = 0
            for k in range(8):
                ni = i + _D8_DI[k]
                if ni < 0 or ni >= nlat:
                    continue
                nj = j + _D8_DJ[k]
                if nj < 0:
                    nj += nlon
                elif nj >= nlon:
                    nj -= nlon
                dz = z - filled[ni, nj]
                if dz <= 0.0:
                    continue
                if _D8_DI[k] == 0:
                    dist = dlon_m[i]
                elif _D8_DJ[k] == 0:
                    dist = dlat_m
                else:
                    dist = np.sqrt(dlat_m * dlat_m + dlon_m[i] * dlon_m[i])
                slope = dz / dist
                if slope > best:
                    best = slope
                    code = _D8_CODE[k]
            out[i, j] = np.int16(code)


def flow_directions(
    filled: np.ndarray, is_ocean: np.ndarray, *, radius: float = EARTH_RADIUS
) -> np.ndarray:
    """D8 最陡下降流向（§4.1），返回 ESRI 方向码场 ``int16``（海洋为 0）。

    传入 :func:`fill_pits` 的输出才能保证每个陆地单元都有出流（否则闭合
    洼地内的单元得到 0，即"无出流"）。坡度按度规距离计算，因此高纬度不会
    被经线收敛放大。填充产生的等高平地在内部做 BFS 导流（见模块 docstring）。
    """
    elev, ocean = _as_arrays(filled, is_ocean)
    nlat, nlon = elev.shape
    dlat_m, dlon_m = spherical.cell_spacing(spherical.lat_centers(nlat), nlon, radius)
    out = np.zeros((nlat, nlon), dtype=np.int16)
    _d8_kernel(
        np.ascontiguousarray(elev),
        np.ascontiguousarray(ocean),
        float(dlat_m),
        np.ascontiguousarray(dlon_m),
        out,
    )
    return out


# ===== 汇流累积（§4.3）=====


@njit(cache=True, inline="always")
def _code_index(code: int) -> int:
    """方向码（2 的幂）→ 方向表下标。"""
    k = 0
    while code > 1:
        code >>= 1
        k += 1
    return k


@njit(cache=True, inline="always")
def _receiver(i: int, j: int, code: int, nlat: int, nlon: int) -> tuple[int, int, bool]:
    """出流目标 ``(ni, nj, 有效)``；出域（越过纬度边界）视为出流口。"""
    k = _code_index(code)
    ni = i + _D8_DI[k]
    if ni < 0 or ni >= nlat:
        return -1, -1, False
    nj = j + _D8_DJ[k]
    if nj < 0:
        nj += nlon
    elif nj >= nlon:
        nj -= nlon
    return ni, nj, True


@njit(cache=True)
def _accumulate(
    direction: np.ndarray, ocean: np.ndarray, acc: np.ndarray, indeg: np.ndarray, queue: np.ndarray
) -> int:
    """Kahn 拓扑排序累积上游单元数，返回成功出队的陆地单元数。"""
    nlat, nlon = direction.shape
    for i in range(nlat):
        for j in range(nlon):
            if ocean[i, j]:
                acc[i, j] = 0.0
                indeg[i, j] = -1
            else:
                acc[i, j] = 1.0
                indeg[i, j] = 0
    for i in range(nlat):
        for j in range(nlon):
            if ocean[i, j]:
                continue
            code = direction[i, j]
            if code == 0:
                continue
            ni, nj, ok = _receiver(i, j, code, nlat, nlon)
            if not ok or ocean[ni, nj]:
                continue
            indeg[ni, nj] += 1
    head = 0
    tail = 0
    for i in range(nlat):
        for j in range(nlon):
            if ocean[i, j]:
                continue
            if indeg[i, j] == 0:
                queue[tail] = i * nlon + j
                tail += 1
    processed = 0
    while head < tail:
        kq = queue[head]
        head += 1
        processed += 1
        i = kq // nlon
        j = kq % nlon
        code = direction[i, j]
        if code == 0:
            continue
        ni, nj, ok = _receiver(i, j, code, nlat, nlon)
        if not ok or ocean[ni, nj]:
            continue
        acc[ni, nj] += acc[i, j]
        indeg[ni, nj] -= 1
        if indeg[ni, nj] == 0:
            queue[tail] = ni * nlon + nj
            tail += 1
    return processed


def flow_accumulation(direction: np.ndarray, is_ocean: np.ndarray) -> np.ndarray:
    """拓扑汇流累积（§4.3）：每个陆地单元计自身 1，沿流向逐级累加下游。

    流向必需**无环**（:func:`flow_directions` 的输出天然满足）；否则报错。
    海洋既不计汇流也不外流，累计到海岸即止。
    """
    dirs = np.asarray(direction, dtype=np.int16)
    ocean = np.asarray(is_ocean, dtype=bool)
    if dirs.ndim != 2:
        raise ValueError(f"direction 必须为 2D，实际 {dirs.shape}")
    if dirs.shape != ocean.shape:
        raise ValueError(f"direction 与 is_ocean 形状不一致: {dirs.shape} vs {ocean.shape}")
    valid = {0, *D8_CODES}
    unknown = set(np.unique(dirs).tolist()) - valid
    if unknown:
        raise ValueError(f"流向编码必须为 ESRI D8 码 {sorted(valid)}，出现非法值 {sorted(unknown)}")

    dirs = np.ascontiguousarray(dirs)
    ocean = np.ascontiguousarray(ocean)
    acc = np.zeros(dirs.shape, dtype=np.float64)
    indeg = np.zeros(dirs.shape, dtype=np.int64)
    queue = np.zeros(dirs.size, dtype=np.int64)
    processed = int(_accumulate(dirs, ocean, acc, indeg, queue))
    expected = int((~ocean).sum())
    if processed != expected:
        raise ValueError(f"流向存在环：仅 {processed}/{expected} 个陆地单元可拓扑处理")
    return acc


def drainage_area(accumulation: np.ndarray, cell_area: np.ndarray) -> np.ndarray:
    """汇水面积 (m^2) = 累积单元数 × 单元面积（§4.4 的水文量纲化）。"""
    acc = np.asarray(accumulation, dtype=np.float64)
    area = np.asarray(cell_area, dtype=np.float64)
    if acc.shape != area.shape:
        raise ValueError(f"accumulation 与 cell_area 形状不一致: {acc.shape} vs {area.shape}")
    return np.asarray(acc * area, dtype=np.float64)


def extract_river_network(
    accumulation: np.ndarray, is_ocean: np.ndarray, *, min_cells: int = DEFAULT_RIVER_CELLS
) -> np.ndarray:
    """河流网络掩码（§4.4）：汇流累积超过阈值处成河，海洋永不成河。"""
    acc = np.asarray(accumulation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    if acc.shape != ocean.shape:
        raise ValueError(f"accumulation 与 is_ocean 形状不一致: {acc.shape} vs {ocean.shape}")
    if min_cells < 1:
        raise ValueError(f"min_cells 必须 >= 1，实际为 {min_cells}")
    return np.asarray((acc >= float(min_cells)) & ~ocean, dtype=bool)


def river_width(
    drainage_area_m2: np.ndarray,
    river_network: np.ndarray,
    *,
    coefficient: float = RIVER_WIDTH_COEFFICIENT,
    min_width_m: float = RIVER_WIDTH_MIN_M,
    max_width_m: float = RIVER_WIDTH_MAX_M,
) -> np.ndarray:
    """水力几何河宽 ``w = C·√A``（§4.4，``w ∝ Q^0.5``），非河道为 0 (m)。"""
    area = np.asarray(drainage_area_m2, dtype=np.float64)
    river = np.asarray(river_network, dtype=bool)
    if area.shape != river.shape:
        raise ValueError(
            f"drainage_area 与 river_network 形状不一致: {area.shape} vs {river.shape}"
        )
    if coefficient <= 0.0:
        raise ValueError(f"coefficient 必须为正，实际为 {coefficient}")
    if min_width_m <= 0.0 or max_width_m < min_width_m:
        raise ValueError(f"河宽上下限非法: min={min_width_m}, max={max_width_m}")
    width = coefficient * np.sqrt(np.maximum(area, 0.0))
    width = np.clip(width, min_width_m, max_width_m)
    return np.asarray(np.where(river, width, 0.0), dtype=np.float64)


# ===== 顶层：河流网络分析 =====


@dataclasses.dataclass(frozen=True)
class HydrologyResult:
    """河流网络分析输出（第三层第 5–8 步，§七）。"""

    filled_elevation: np.ndarray  # 无洼地高程场（路由基底）
    flow_direction: np.ndarray  # ESRI D8 方向码 int16
    flow_accumulation: np.ndarray  # 上游单元数（含自身），海洋为 0
    drainage_area: np.ndarray  # 汇水面积 m^2
    river_network: np.ndarray  # 河流掩码 bool
    river_width: np.ndarray  # 河宽 m，非河道为 0
    report: dict[str, float | int | str]


def analyze_hydrology(
    elevation: np.ndarray,
    is_ocean: np.ndarray,
    *,
    river_min_cells: int = DEFAULT_RIVER_CELLS,
    radius: float = EARTH_RADIUS,
) -> HydrologyResult:
    """完整河流网络分析（§七 第 2、5–8 步）：填充 → D8 → 汇流 → 河网 → 河宽。"""
    elev, ocean = _as_arrays(elevation, is_ocean)
    nlat, nlon = elev.shape
    filled = fill_pits(elev, ocean)
    direction = flow_directions(filled, ocean, radius=radius)
    accumulation = flow_accumulation(direction, ocean)
    cell_area = spherical.cell_areas(spherical.lat_edges(nlat), nlon, radius)
    area = drainage_area(accumulation, cell_area)
    river = extract_river_network(accumulation, ocean, min_cells=river_min_cells)
    width = river_width(area, river)
    mean_cell_area = float(cell_area.mean())
    report: dict[str, float | int | str] = {
        "river_cells": int(river.sum()),
        "river_fraction_of_land": float(river.sum() / max(int((~ocean).sum()), 1)),
        "river_min_cells": int(river_min_cells),
        "river_min_area_equivalent_m2": float(river_min_cells * mean_cell_area),
        "max_flow_accumulation": float(accumulation.max()),
        "max_drainage_area_m2": float(area.max()),
        "max_river_width_m": float(width.max()),
    }
    return HydrologyResult(
        filled_elevation=filled,
        flow_direction=direction,
        flow_accumulation=accumulation,
        drainage_area=area,
        river_network=river,
        river_width=width,
        report=report,
    )


def _as_arrays(elevation: np.ndarray, is_ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """统一入参：高程转 float64、掩码转 bool，并校验 2D 与形状一致。"""
    elev = np.asarray(elevation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    if elev.shape != ocean.shape:
        raise ValueError(f"elevation 与 is_ocean 形状不一致: {elev.shape} vs {ocean.shape}")
    return elev, ocean


__all__ = [
    "D8_CODES",
    "D8_OFFSETS",
    "DEFAULT_RIVER_CELLS",
    "FILL_EPS_M",
    "RIVER_WIDTH_COEFFICIENT",
    "RIVER_WIDTH_MAX_M",
    "RIVER_WIDTH_MIN_M",
    "HydrologyResult",
    "analyze_hydrology",
    "drainage_area",
    "extract_river_network",
    "fill_pits",
    "flow_accumulation",
    "flow_directions",
    "river_width",
]
