"""球面域扭曲 Voronoi 板块划分（《第一层完善》§1）。

流程：Fibonacci 球面种子点 → 域扭曲 → 球面最近邻归属 → 微板块归并 → 连通性修复。
坐标约定：单元坐标经 :func:`virtual_world.core.spherical.latlon_to_xyz`
转为单位向量参与球面计算，结果保持 ``(nlat, nlon)`` 字段形状。

邻接约定（与 :mod:`virtual_world.core.spherical` 一致）：**经度循环、纬度非循环**
（极点约束），因此南北极行之间不产生伪邻接。

**网格无关性**（方案《地形生成混合方案》§1.5）：邻接相关的函数都接受可选的
``neighbours`` 参数（形状 ``(4, N)`` 的扁平邻接索引，顺序见
:func:`latlon_neighbours`）。默认 ``None`` 时按经纬网格构造，因此既有调用不受影响；
立方球网格传入 :meth:`virtual_world.core.cubed_sphere.CubedSphere.neighbors` 即可。
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]
from scipy.sparse import csgraph  # type: ignore[import-untyped]

from ..core.cubed_sphere import CubedSphere

#: 大板块数量范围（方案 §1.2.4：6–12）
MAJOR_PLATE_MIN = 6
MAJOR_PLATE_MAX = 12


def latlon_neighbours(nlat: int, nlon: int) -> np.ndarray:
    """经纬网格 4-邻接扁平索引 ``(4, nlat*nlon)``，``-1`` 表示域外（无邻居）。

    顺序与 :data:`virtual_world.core.cubed_sphere.DIRECTIONS` 一致：
    **北(i+1)、南(i-1)、东(j+1)、西(j-1)**。经度循环、纬度非循环（极点约束）。
    """
    nlat, nlon = int(nlat), int(nlon)
    if nlat <= 0 or nlon <= 0:
        raise ValueError("nlat 与 nlon 必须为正")
    idx = np.arange(nlat * nlon, dtype=np.int64).reshape(nlat, nlon)
    north = np.full((nlat, nlon), -1, dtype=np.int64)
    north[:-1, :] = idx[1:, :]
    south = np.full((nlat, nlon), -1, dtype=np.int64)
    south[1:, :] = idx[:-1, :]
    cols = np.arange(nlon)
    east = idx[:, (cols + 1) % nlon]
    west = idx[:, (cols - 1) % nlon]
    return np.ascontiguousarray(np.stack([north, south, east, west], axis=0).reshape(4, nlat * nlon))


def grid_neighbours(shape: tuple[int, ...], neighbours: np.ndarray | None = None) -> np.ndarray:
    """按网格形状给出 4-邻接扁平索引 ``(4, N)``，``-1`` 表示域外（无邻居）。

    - 显式给出 ``neighbours`` 时校验后原样返回；
    - 2D 形状按经纬网格（:func:`latlon_neighbours`：经度循环、纬度非循环）；
    - 3D ``(6, n, n)`` 按立方球网格（:class:`~virtual_world.core.cubed_sphere.CubedSphere`，
      每格都有 4 个邻居、无域外单元）。
    """
    expected = int(np.prod(shape))
    if neighbours is not None:
        nbr = np.asarray(neighbours, dtype=np.int64)
        if nbr.ndim != 2 or nbr.shape[0] != 4 or nbr.shape[1] != expected:
            raise ValueError(f"neighbours 形状应为 (4, {expected})，实际 {nbr.shape}")
        return nbr
    if len(shape) == 2:
        return latlon_neighbours(shape[0], shape[1])
    if len(shape) == 3 and shape[0] == 6 and shape[1] == shape[2]:
        return CubedSphere(shape[1]).neighbors()
    raise ValueError(f"无法推断该形状的邻接，请显式给出 neighbours（shape={shape}）")


def plate_angular_radius(n_seeds: int) -> float:
    """板块平均角半径 (rad)：由球面均分面积 ``4π/N`` 反解球冠角半径（§1.2）。"""
    if n_seeds <= 0:
        raise ValueError(f"种子点数量必须为正整数，实际为 {n_seeds}")
    return float(np.arccos(np.clip(1.0 - 2.0 / n_seeds, -1.0, 1.0)))


def fibonacci_sphere(n: int) -> np.ndarray:
    """生成 N 个单位球面上的 Fibonacci 分布种子点，形状 ``(n, 3)``。

    公式（《第一层完善》§1.2）：``phi_k = arccos(1 - 2(k+0.5)/n)``、
    ``theta_k = pi(1+sqrt(5)) k``。相比随机采样，避免极点聚集与局部空洞。
    """
    if n <= 0:
        raise ValueError(f"种子点数量必须为正整数，实际为 {n}")
    k = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * k
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


# ===== 域扭曲噪声（3D 值噪声 fBm，确定性、无外部状态）=====


def _hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray, salt: int) -> np.ndarray:
    """整数格点散列 → [0,1)，确定性伪随机。"""
    h = (
        np.asarray(x, dtype=np.int64) * 374761393
        + np.asarray(y, dtype=np.int64) * 668265263
        + np.asarray(z, dtype=np.int64) * 1442695041
        + salt * 22695477
    )
    h = (h ^ (h >> 13)) & np.int64(0x7FFFFFFF)
    return np.asarray((h * np.int64(1664525) & np.int64(0xFFFFFFFF)) / 4294967296.0)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return np.asarray(t * t * (3.0 - 2.0 * t))


def _value_noise(x: np.ndarray, y: np.ndarray, z: np.ndarray, salt: int) -> np.ndarray:
    """3D 值噪声，三次平滑插值，输出 [0,1]。"""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    tx = _smoothstep(x - x0)
    ty = _smoothstep(y - y0)
    tz = _smoothstep(z - z0)
    c000 = _hash3(x0, y0, z0, salt)
    c100 = _hash3(x0 + 1, y0, z0, salt)
    c010 = _hash3(x0, y0 + 1, z0, salt)
    c110 = _hash3(x0 + 1, y0 + 1, z0, salt)
    c001 = _hash3(x0, y0, z0 + 1, salt)
    c101 = _hash3(x0 + 1, y0, z0 + 1, salt)
    c011 = _hash3(x0, y0 + 1, z0 + 1, salt)
    c111 = _hash3(x0 + 1, y0 + 1, z0 + 1, salt)
    x00 = c000 + (c100 - c000) * tx
    x10 = c010 + (c110 - c010) * tx
    x01 = c001 + (c101 - c001) * tx
    x11 = c011 + (c111 - c011) * tx
    y0v = x00 + (x10 - x00) * ty
    y1v = x01 + (x11 - x01) * ty
    return np.asarray(y0v + (y1v - y0v) * tz)


def fbm_warp_offsets(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    seed: int,
    frequency: float = 3.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """两个独立的 fBm 域扭曲偏移场 ``(n1, n2)``，输出约 [-1,1]。

    方案 §1.3：``lam' = lam + A*n1, phi' = phi + A*n2``。实现用 3D 值噪声
    叠加多 octave，输入坐标为低频球面坐标（单位：个位数周期），确定性由
    ``seed`` 驱动（``n1`` 与 ``n2`` 使用不同 salt）。
    """
    xyz = np.stack(
        [
            np.cos(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg)),
            np.cos(np.deg2rad(lat_deg)) * np.sin(np.deg2rad(lon_deg)),
            np.sin(np.deg2rad(lat_deg)),
        ],
        axis=-1,
    )

    def _fbm(salt: int) -> np.ndarray:
        total = np.zeros_like(xyz[..., 0])
        amp = 1.0
        freq = frequency
        for _ in range(octaves):
            total += amp * _value_noise(
                xyz[..., 0] * freq, xyz[..., 1] * freq, xyz[..., 2] * freq, salt
            )
            amp *= persistence
            freq *= lacunarity
        norm = (1.0 - persistence**octaves) / (1.0 - persistence)
        return 2.0 * total / norm - 1.0

    return _fbm(seed * 2 + 1), _fbm(seed * 2 + 2)


def assign_plates(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    seeds: np.ndarray,
    seed: int = 0,
    warp_amp: float = 0.0,
    frequency: float = 3.0,
    octaves: int = 4,
) -> np.ndarray:
    """域扭曲球面最近邻归属：每个网格单元 → 最近的（扭曲）种子点。

    ``lat_deg`` / ``lon_deg`` 形状 ``(nlat, nlon)``，返回同形状 int32 数组
    （微板块 id ∈ [0, n_seeds)）。``warp_amp=0`` 退化为原始最近邻；
    ``warp_amp>0`` 时先对经纬度施加 fBm 噪声偏移（幅度 rad）再查询。
    """
    if warp_amp < 0.0:
        raise ValueError(f"warp_amp 不能为负，实际为 {warp_amp}")
    lat_a = np.asarray(lat_deg, dtype=np.float64)
    lon_a = np.asarray(lon_deg, dtype=np.float64)
    # 网格中心输入 (nlat,) x (nlon,) → (nlat, nlon)；标量 → (1,1)
    if lat_a.ndim == 0 and lon_a.ndim == 0:
        lat_2d = np.asarray([[lat_a]])
        lon_2d = np.asarray([[lon_a]])
    elif lat_a.ndim == 1 and lon_a.ndim == 1:
        lat_2d, lon_2d = np.meshgrid(lat_a, lon_a, indexing="ij")
    else:
        shape = np.broadcast_shapes(lat_a.shape, lon_a.shape)
        lat_2d = np.broadcast_to(lat_a, shape)
        lon_2d = np.broadcast_to(lon_a, shape)
    flat = np.stack(
        [
            np.cos(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(lon_2d)),
            np.cos(np.deg2rad(lat_2d)) * np.sin(np.deg2rad(lon_2d)),
            np.sin(np.deg2rad(lat_2d)),
        ],
        axis=-1,
    ).reshape(-1, 3)

    if warp_amp > 0.0:
        n1, n2 = fbm_warp_offsets(lat_2d, lon_2d, seed, frequency, octaves)
        flat_lat = lat_2d + np.rad2deg(warp_amp) * n1
        flat_lon = lon_2d + np.rad2deg(warp_amp) * n2
        flat = np.stack(
            [
                np.cos(np.deg2rad(flat_lat)) * np.cos(np.deg2rad(flat_lon)),
                np.cos(np.deg2rad(flat_lat)) * np.sin(np.deg2rad(flat_lon)),
                np.sin(np.deg2rad(flat_lat)),
            ],
            axis=-1,
        ).reshape(-1, 3)

    # 点积最大 = 角距离最近（种子为单位向量）
    nearest = seeds @ flat.T
    return np.asarray(nearest.argmax(axis=0).astype(np.int32).reshape(lat_2d.shape))


# ===== 邻接与连通性（经度循环、纬度非循环）=====


def connected_components(mask: np.ndarray, neighbours: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """4-邻连通分量标记：返回 ``(标签场, 分量数)``，0 为背景。

    标签按**光栅序**（首个单元出现的先后）从 1 开始编号，与 ``scipy.ndimage.label``
    的语义一致。经度方向循环、纬度方向非循环由 ``neighbours`` 决定（见
    :func:`latlon_neighbours` 与 :func:`grid_neighbours`）。
    """
    values = np.asarray(mask, dtype=bool)
    nbr = grid_neighbours(values.shape, neighbours)
    flat = values.ravel()
    n = flat.size
    if not flat.any():
        return np.zeros(values.shape, dtype=np.int32), 0

    source = np.arange(n, dtype=np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for d in range(4):
        nb = nbr[d]
        valid_nb = nb >= 0
        valid = valid_nb & flat & flat[np.where(valid_nb, nb, 0)]
        rows.append(source[valid])
        cols.append(nb[valid])
    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    graph = sparse.coo_matrix(
        (np.ones(rows_a.size, dtype=np.int8), (rows_a, cols_a)), shape=(n, n)
    ).tocsr()
    _, raw = csgraph.connected_components(graph, directed=False)
    raw = np.asarray(raw, dtype=np.int64)

    mask_idx = np.nonzero(flat)[0]
    comp = raw[mask_idx]
    uniq, inverse = np.unique(comp, return_inverse=True)
    first = np.full(uniq.size, mask_idx.size, dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(comp.size, dtype=np.int64))
    rank = np.argsort(first, kind="stable")
    remap = np.empty(uniq.size, dtype=np.int32)
    remap[rank] = np.arange(1, uniq.size + 1, dtype=np.int32)

    out = np.zeros(n, dtype=np.int32)
    out[mask_idx] = remap[inverse]
    return out.reshape(values.shape), int(uniq.size)


def micro_plate_adjacency(
    field: np.ndarray, neighbours: np.ndarray | None = None
) -> np.ndarray:
    """微板块邻接矩阵：``shared[a, b]`` = 两板块共享的网格边数（对称、对角为 0）。

    邻接由 ``neighbours`` 给出（缺省为经纬网格：经度循环、**纬度非循环**，
    南北极行之间不产生伪邻接）。每条无向边只计一次（以扁平索引较小者为起点去重）。
    """
    values = np.asarray(field)
    nbr = grid_neighbours(values.shape, neighbours)
    flat = values.ravel()
    n_ids = int(flat.max()) + 1
    shared = np.zeros((n_ids, n_ids), dtype=np.int64)
    source = np.arange(flat.size, dtype=np.int64)
    for d in range(4):
        nb = nbr[d]
        valid = nb >= 0
        src = source[valid]
        dst = nb[valid]
        keep = src < dst
        np.add.at(shared, (flat[src[keep]], flat[dst[keep]]), 1)
    shared = shared + shared.T
    np.fill_diagonal(shared, 0)
    return shared


def _neighbour_label_counts(
    labels: np.ndarray, mask: np.ndarray, neighbours: np.ndarray | None = None
) -> dict[int, int]:
    """统计 ``mask`` 区域四周的标签出现次数（含跨面接缝，域外不计）。"""
    values = np.asarray(labels)
    nbr = grid_neighbours(values.shape, neighbours)
    flat_labels = values.ravel()
    fragment = np.asarray(mask, dtype=bool).ravel()

    counts: dict[int, int] = {}
    for d in range(4):
        nb = nbr[d]
        valid = (nb >= 0) & fragment
        if not valid.any():
            continue
        neighbour_labels = flat_labels[np.where(valid, nb, 0)][valid]
        uniq, freq = np.unique(neighbour_labels, return_counts=True)
        for value, count in zip(uniq.tolist(), freq.tolist(), strict=True):
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    return counts


def ensure_connected(
    plate_map: np.ndarray, max_iter: int = 8, neighbours: np.ndarray | None = None
) -> np.ndarray:
    """保证每个板块在球面上连通（§1.4 关键约束）。

    每个板块保留其最大连通分量，其余碎片整体并入相邻板块中共享边界最长者。
    由于每个板块至少保留一个分量，板块数不会因修复而减少。
    """
    out = np.array(plate_map, dtype=np.int32, copy=True)
    nbr = grid_neighbours(out.shape, neighbours)
    for _ in range(max_iter):
        changed = False
        for lab in np.unique(out):
            labels, _ = connected_components(out == lab, nbr)
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            comp_ids = np.nonzero(sizes)[0]
            if comp_ids.size <= 1:
                continue
            keep = int(comp_ids[int(np.argmax(sizes[comp_ids]))])
            for comp_id in comp_ids:
                if comp_id == keep:
                    continue
                fragment = labels == comp_id
                counts = _neighbour_label_counts(out, fragment, nbr)
                counts.pop(int(lab), None)
                if not counts:
                    continue
                out[fragment] = max(counts, key=counts.__getitem__)
                changed = True
        if not changed:
            break
    return out


def _graph_distance(shared: np.ndarray, sources: list[int]) -> np.ndarray:
    """微板块邻接图上的 BFS 跳数距离；不可达节点记为 ``n_ids``（视作最远）。"""
    n_ids = shared.shape[0]
    dist = np.full(n_ids, np.inf)
    queue: deque[int] = deque()
    for source in sources:
        dist[source] = 0.0
        queue.append(int(source))
    while queue:
        node = queue.popleft()
        for nb in np.nonzero(shared[node] > 0)[0]:
            if np.isinf(dist[nb]):
                dist[nb] = dist[node] + 1.0
                queue.append(int(nb))
    return np.where(np.isinf(dist), float(n_ids), dist)


def _select_cores(shared: np.ndarray, n_major: int, seed: int) -> list[int]:
    """选 ``n_major`` 个核心微板块：最远点初始化（首核由 ``seed`` 决定）。

    每轮取到已有核心**图上跳数距离**最大的微板块作为新核心，使核心在全球均匀
    铺开——这是各板块面积均衡的前提（仅按共享边界长度选会令核心聚簇）。
    """
    rng = np.random.default_rng(seed)
    cores = [int(rng.integers(shared.shape[0]))]
    distance = _graph_distance(shared, cores)
    while len(cores) < n_major:
        next_core = int(np.argmax(distance))
        cores.append(next_core)
        distance = np.minimum(distance, _graph_distance(shared, [next_core]))
    return cores


def _grow_regions(shared: np.ndarray, cores: list[int], n_major: int) -> np.ndarray:
    """在微板块邻接图上按多源 Dijkstra（单位边长）生长出 ``n_major`` 个区域。

    代价函数取方案 §1.4 的 ``cost = α·dist + β/shared_length``：主导项为到核心的
    图距离 α·dist（保证各板块面积均衡、紧凑），等距时优先共享边界更长者
    （β 项，使边界更平滑）。按拓扑顺序认领节点，因此每个区域天然连通。
    """
    owner = np.full(shared.shape[0], -1, dtype=np.int32)
    heap: list[tuple[int, int, int, int]] = []
    for k, core in enumerate(cores):
        owner[core] = k
        heapq.heappush(heap, (0, 0, core, k))

    while heap:
        dist, neg_shared, node, k = heapq.heappop(heap)
        if owner[node] >= 0 and owner[node] != k:
            continue
        owner[node] = k
        for nb in np.nonzero(shared[node] > 0)[0]:
            nb = int(nb)
            if owner[nb] >= 0:
                continue
            heapq.heappush(heap, (dist + 1, -int(shared[nb, cores[k]]), nb, k))

    # 兜底：邻接图存在孤立分量时并入 0 号核心
    owner[owner < 0] = 0
    return owner


# ===== 微板块归并 =====


def merge_micro_plates(
    micro_plate: np.ndarray,
    n_major: int,
    seed: int = 0,
    neighbours: np.ndarray | None = None,
) -> np.ndarray:
    """把微板块归并为 ``n_major`` 个大板块，返回重编号为 0..n_major-1 的标签场。

    方案 §1.4 归并策略：
    1. :func:`_select_cores` 确定 ``n_major`` 个核心微板块（首核由 ``seed`` 决定，
       其余最大化空间分散）；
    2. :func:`_grow_regions` 按代价 ``cost = α·dist + β/shared_length`` 做多源区域
       生长，使各板块面积均衡且边界平滑；
    3. :func:`ensure_connected` 校验并修复碎片，保证大板块在球面上连通。

    微板块邻接由 ``neighbours`` 给出（缺省为经纬网格：经度循环、纬度非循环）。
    """
    if not MAJOR_PLATE_MIN <= n_major <= MAJOR_PLATE_MAX:
        raise ValueError(
            f"n_major 应在 {MAJOR_PLATE_MIN}–{MAJOR_PLATE_MAX} 之间（方案 §1.2.4），实际为 {n_major}"
        )

    unique_result = np.unique(micro_plate, return_inverse=True)
    dense: np.ndarray = unique_result[1]
    if dense.size == 0:
        raise ValueError("micro_plate 为空")
    n_ids = int(dense.max()) + 1
    if n_major > n_ids:
        raise ValueError(f"n_major={n_major} 超过微板块数 {n_ids}")

    if n_major == n_ids:
        return np.asarray(dense.reshape(micro_plate.shape).astype(np.int32))

    field = np.asarray(dense.reshape(micro_plate.shape))
    shared = micro_plate_adjacency(field, neighbours)
    cores = _select_cores(shared, n_major, seed)
    owner = _grow_regions(shared, cores, n_major)
    return ensure_connected(np.asarray(owner[field].astype(np.int32)), neighbours=neighbours)
