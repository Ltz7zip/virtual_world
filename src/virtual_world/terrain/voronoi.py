"""球面域扭曲 Voronoi 板块划分（《第一层完善》§1）。

流程：Fibonacci 球面种子点 → 域扭曲 → 球面最近邻归属 → 微板块归并。
坐标约定：单元坐标经 :func:`virtual_world.core.spherical.latlon_to_xyz`
转为单位向量参与球面计算，结果保持 ``(nlat, nlon)`` 字段形状。
"""

from __future__ import annotations

import numpy as np

#: 大板块数量范围（方案 §1.2.4：6–12）
MAJOR_PLATE_MIN = 6
MAJOR_PLATE_MAX = 12


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


# ===== 微板块归并 =====


def merge_micro_plates(micro_plate: np.ndarray, n_major: int, seed: int = 0) -> np.ndarray:
    """把微板块归并为 ``n_major`` 个大板块，返回重编号为 0..n_major-1 的标签场。

    方案 §1.4 归并策略：
    1. 用 k-means++ 风格（与已选核心共享边界最少）确定性选出 ``n_major`` 个核心微板块；
    2. 多源区域生长：从未分配微板块中，重复将"与任一已分配板块共享边界最长"的
       板块并入对应核心，直到全部归并完毕——天然保证大板块在球面上连续。

    微板块邻接按 4-邻计算（经度为循环边界）。
    """
    if n_major < 1:
        raise ValueError(f"n_major 必须 >= 1，实际为 {n_major}")

    unique_result = np.unique(micro_plate, return_inverse=True)
    dense: np.ndarray = unique_result[1]
    if dense.size == 0:
        raise ValueError("micro_plate 为空")
    n_ids = int(dense.max()) + 1
    if n_major > n_ids:
        raise ValueError(f"n_major={n_major} 超过微板块数 {n_ids}")

    if n_major == n_ids:
        return dense.reshape(micro_plate.shape).astype(np.int32)

    field = np.asarray(dense.reshape(micro_plate.shape))
    nlat, nlon = field.shape

    # 邻接矩阵：shared[a, b] = a、b 板块之间共享的网格边界数（对称）
    shared = np.zeros((n_ids, n_ids), dtype=np.int64)
    east = np.roll(field, -1, axis=1)
    south = np.roll(field, 1, axis=0)
    np.add.at(shared, (field.ravel(), east.ravel()), 1)
    np.add.at(shared, (field.ravel(), south.ravel()), 1)
    shared = shared + shared.T
    np.fill_diagonal(shared, 0)

    rng = np.random.default_rng(seed)
    remaining = set(range(n_ids))
    cores: list[int] = []
    if n_major >= 1:
        first = int(rng.integers(n_ids))
        cores.append(first)
        remaining.discard(first)
    while len(cores) < n_major and remaining:
        # 选与已选核心共享边界总和最小的板块作为新核心（最大化分散）
        next_core = min(remaining, key=lambda p: int(shared[p, cores].sum()))
        cores.append(next_core)
        remaining.discard(next_core)
    if len(cores) < n_major:
        raise ValueError(f"无法选出 {n_major} 个不重叠核心微板块")

    assign = np.full(n_ids, -1, dtype=np.int32)
    for k, c in enumerate(cores):
        assign[c] = k

    pending = list(remaining)
    while pending:
        newly = []
        for p in pending:
            nbrs = np.nonzero(shared[p] > 0)[0]
            assigned = nbrs[assign[nbrs] >= 0]
            core_len = np.zeros(n_major, dtype=np.int64)
            if assigned.size:
                np.add.at(core_len, assign[assigned], shared[p][assigned])
            best = int(core_len.argmax())
            if core_len[best] > 0:
                assign[p] = best
            else:
                newly.append(p)
        if len(newly) == len(pending):
            # 极端断开图：直接分配给离它最近的核心（按共享长度，全 0 则 0）
            for p in pending:
                if assign[p] < 0:
                    assign[p] = 0
            break
        pending = newly

    return np.asarray(assign[field].astype(np.int32))
