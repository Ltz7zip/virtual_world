"""扩散精修：条件生成、残差学习与三频带融合（《扩散精修方案》）。

设计定位（ADR）：**条件编排与约束强制归管线所有，细节合成走可插拔协议**。
这样做的直接后果是——无论精修器是内置确定性噪声还是外部学习模型，宏观骨架
（海陆边界、主要山脉位置）都不可能被破坏：约束由 :func:`enforce_residual_constraints`
在精修器输出**之后**施加，精修器无法绕过。

数学约束（§三）：

- **均值约束**：支撑集上 ``E[H_residual] = 0``
- **低频截断**：块尺度低频分量被严格扣除，``|H_tectonic| < ε`` 处不改动骨架
- **海岸线约束**：``|H_tectonic| < ε`` 的单元强制 ``H_residual = 0``

管线（§六 伪代码）：

    H_tectonic + 边界类型
      → 条件通道（低频构造 / 海陆掩码 / 边界距离 / 河流网络）
      → tile 划分 → 逐 tile 条件去噪 → 拼接
      → 约束强制（零均值 + 低频截断 + 海岸线）
      → 三频带融合（低=构造，中=噪声基底，高=扩散细节）
      → H_final + H_residual

加速（§四）：条件通道 4 通道中河流网络用 Numba ``@njit`` 的 D8 汇流累积
（唯一 O(n log n) 热点）；条件通道可落盘缓存（Zarr），静态通道不重复计算；
tile 划分支持批量合成；三频带融合在频域一次完成。

关于精度：本模块的场运算保持 ``float64``（与地形层其余部分一致）。扩散模型
推理若走 GPU 后端（MPS / MLX）则在其内部使用 float32/BF16，但**约束强制与
融合始终在 float64 完成**，避免低精度误差累积到骨架约束上。
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numba import njit

from . import noise_refine as nr

#: 三频带窗默认截断（单位：周期/格）——0.1 对应波长 10 格
#: （10 km 网格下即 100 km，与方案 §5.1 的频带划分一致）
K_LOW = 0.1
K_MID = 0.35
K_HIGH = 0.6
#: 块尺度低频分离默认块大小
BLOCK = 8
#: 海岸线判定阈值（m）：``|H_tectonic| < eps`` 视为海岸线
COASTLINE_EPS = 1.0
#: 条件通道球面坐标频率缩放
COORDS_SCALE = 6.0
#: 河流阈值占网格单元的比例（未显式指定时的默认值）
RIVER_FRACTION = 0.02
#: 分层堆叠中单层残差的默认幅度 = fraction × std(上一层地形)（§3.4）
LEVEL_RESIDUAL_FRACTION = 0.06


# ===== 精修器协议 =====


class DiffusionRefiner(Protocol):
    """扩散精修器协议（§3.3 残差学习）。

    实现者按 tile 接收条件通道，返回**残差** tile。返回值不必满足约束——
    低频截断、零均值与海岸线由 :func:`refine_diffusion` 在拼接后统一强制，
    因此任何实现都无法改变构造层建立的宏观格局。

    内置实现 :class:`StructuredDiffusionRefiner`（确定性噪声，无外部依赖）；
    外部模型走 :class:`TerrainDiffusionRefiner`。
    """

    def refine_tile(self, conditions: TileConditions, seed: int) -> np.ndarray:
        """返回形状等于 ``conditions.tectonic.shape`` 的残差 tile。"""
        ...


# ===== 条件通道（§3.2 四通道） =====


@dataclasses.dataclass(frozen=True)
class ConditionChannels:
    """扩散精修的输入条件信号（§3.2 的四路 + §1.3 的选核图）。

    ``lowpass`` / ``land_mask`` / ``boundary_distance`` / ``river_network`` 构成方案
    §3.2 的**四路模型输入通道**（见 :meth:`stack`）。``kernel_map`` 是按 §1.3 分配
    策略表逐单元选出的噪声核标签，供内置精修器保持与噪声基底一致的构造纹理；
    它不是扩散模型的输入通道，因此不参与 :meth:`stack`。
    """

    lowpass: np.ndarray  # 通道 1：构造高程低频分量
    land_mask: np.ndarray  # 通道 2：海陆掩码
    boundary_distance: np.ndarray  # 通道 3：到板块边界距离场
    river_network: np.ndarray  # 通道 4：河流网络掩码
    kernel_map: np.ndarray  # §1.3 选核图（辅助，不属于模型输入通道）

    def stack(self) -> np.ndarray:
        """堆叠为 ``(4, ...)``，供模型多通道输入（§3.2）。"""
        return np.stack(
            [
                np.asarray(self.lowpass, dtype=np.float64),
                np.asarray(self.land_mask, dtype=np.float64),
                np.asarray(self.boundary_distance, dtype=np.float64),
                np.asarray(self.river_network, dtype=np.float64),
            ],
            axis=0,
        )


def sphere_coords(nlat: int, nlon: int, *, scale: float = COORDS_SCALE) -> np.ndarray:
    """网格中心的单位球面坐标 × ``scale``，形状 ``(nlat, nlon, 3)``（§4.1）。

    用三维坐标而非经纬度采样，天然无极点奇点与经度接缝；也让跨 tile 的噪声
    只依赖全局位置，重叠区逐点一致（无缝）。
    """
    lat = -90.0 + (180.0 / nlat) * (np.arange(nlat) + 0.5)
    lon = -180.0 + (360.0 / nlon) * (np.arange(nlon) + 0.5)
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    cp = np.cos(phi)
    return np.stack(
        [
            cp * np.cos(lam) * scale,
            cp * np.sin(lam) * scale,
            np.sin(phi) * np.ones((nlat, nlon)) * scale,
        ],
        axis=-1,
    )


# ===== D8 汇流累积（§3.2 条件通道 4） =====


@njit(cache=True)
def _d8_kernel(
    elev: np.ndarray,
    min_elevation: float,
    acc: np.ndarray,
) -> None:
    """按高程降序单步累积（原地写入 ``acc``）。

    高程降序保证处理某单元时，所有汇入它的上游单元都已处理完毕，因此
    ``acc`` 一次遍历即为最终值。海洋（``elev <= min_elevation``）为汇点，
    既不外流也不计汇流。该累积有严格的顺序依赖，故不使用并行。
    """
    nlat, nlon = elev.shape
    order = np.argsort(-elev.ravel())
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    for k in range(order.size):
        idx = order[k]
        i = idx // nlon
        j = idx % nlon
        if elev[i, j] <= min_elevation:
            continue
        cos_lat = np.cos(np.deg2rad(-90.0 + dlat * (i + 0.5)))
        best_slope = 0.0
        bi = -1
        bj = -1
        for di in range(-1, 2):
            ni = i + di
            if ni < 0 or ni >= nlat:
                continue
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                nj = (j + dj) % nlon
                if elev[ni, nj] <= min_elevation:
                    continue
                drop = elev[i, j] - elev[ni, nj]
                if drop <= 0.0:
                    continue
                dist = np.sqrt((di * dlat) ** 2 + (dj * dlon * cos_lat) ** 2)
                slope = drop / dist
                if slope > best_slope:
                    best_slope = slope
                    bi = ni
                    bj = nj
        if bi >= 0:
            acc[bi, bj] += acc[i, j]


def d8_flow_accumulation(elevation: np.ndarray, min_elevation: float = 0.0) -> np.ndarray:
    """D8 汇流累积（单元数），海洋单元为 0（§1.4.2 的简化前置版）。

    每个陆地区域单元初始计 1，沿最陡下降方向（8 邻域、经度循环）逐级累加。
    这是条件通道 4 所需的河流网络来源；完整的河流网络与水流路径生成为
    第三层（侵蚀模拟）的职责。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError(f"elevation 必须为 2D，实际 {elev.shape}")
    acc = np.where(elev > min_elevation, 1.0, 0.0)
    _d8_kernel(elev, float(min_elevation), acc)
    return np.asarray(acc, dtype=np.float64)


# ===== 条件通道构建与缓存（§3.2 / §4.3） =====


def build_condition_channels(
    tectonic: np.ndarray,
    boundary_type: np.ndarray,
    *,
    block: int = BLOCK,
    river_threshold: float | None = None,
    river_network: np.ndarray | None = None,
) -> ConditionChannels:
    """构建四路条件通道（§3.2）。

    ``river_network`` 已给出时直接使用（不再做海陆掩码处理）；否则由
    :func:`d8_flow_accumulation` 计算，阈值为 ``river_threshold``，缺省为
    网格单元数的 :data:`RIVER_FRACTION`。
    """
    tec = np.asarray(tectonic, dtype=np.float64)
    bt = np.asarray(boundary_type, dtype=np.int32)
    if tec.ndim != 2:
        raise ValueError(f"tectonic 必须为 2D，实际 {tec.shape}")
    if tec.shape != bt.shape:
        raise ValueError(f"tectonic 与 boundary_type 形状不一致: {tec.shape} vs {bt.shape}")
    if block < 2:
        raise ValueError(f"block 必须 >= 2，实际为 {block}")

    land_mask = tec > 0.0
    lowpass = nr.expand_tile(nr.block_mean(tec, block), block, tec.shape)
    bdist = nr.boundary_distance_field(bt)

    if river_network is None:
        threshold = (
            float(river_threshold)
            if river_threshold is not None
            else max(2.0, RIVER_FRACTION * tec.size)
        )
        rivers = (d8_flow_accumulation(tec, min_elevation=0.0) >= threshold) & land_mask
    else:
        rivers = np.asarray(river_network, dtype=bool)
        if rivers.shape != tec.shape:
            raise ValueError(f"river_network 形状 {rivers.shape} 与构造场 {tec.shape} 不一致")

    return ConditionChannels(
        lowpass=lowpass,
        land_mask=land_mask,
        boundary_distance=bdist,
        river_network=rivers,
        kernel_map=nr.select_noise_kernel(bt, tec),
    )


def condition_cache_key(
    tectonic: np.ndarray,
    boundary_type: np.ndarray,
    *,
    block: int = BLOCK,
    river_threshold: float | None = None,
) -> str:
    """条件通道缓存键（§4.3）：由输入数据与参数的内容哈希决定。"""
    tec = np.ascontiguousarray(tectonic, dtype=np.float64)
    bt = np.ascontiguousarray(boundary_type, dtype=np.int32)
    hasher = hashlib.sha256()
    hasher.update(tec.tobytes())
    hasher.update(bt.tobytes())
    hasher.update(f"{tec.shape}|{block}|{river_threshold}".encode())
    return hasher.hexdigest()[:32]


def _save_conditions(path: Path, channels: ConditionChannels) -> None:
    import zarr

    group = zarr.open_group(str(path), mode="w")
    group.create_array("lowpass", data=np.asarray(channels.lowpass, dtype=np.float64))
    group.create_array("land_mask", data=np.asarray(channels.land_mask, dtype=bool))
    group.create_array(
        "boundary_distance", data=np.asarray(channels.boundary_distance, dtype=np.float64)
    )
    group.create_array("river_network", data=np.asarray(channels.river_network, dtype=bool))
    group.create_array("kernel_map", data=np.asarray(channels.kernel_map, dtype=np.int32))


def _load_conditions(path: Path) -> ConditionChannels | None:
    """读回缓存；缺任一字段（如旧版本写的缓存）时视为未命中。"""
    if not path.exists():
        return None
    import zarr

    group: Any = zarr.open_group(str(path), mode="r")
    if any(name not in group for name in ("lowpass", "land_mask", "boundary_distance", "river_network", "kernel_map")):
        return None
    return ConditionChannels(
        lowpass=np.asarray(group["lowpass"][:], dtype=np.float64),
        land_mask=np.asarray(group["land_mask"][:], dtype=bool),
        boundary_distance=np.asarray(group["boundary_distance"][:], dtype=np.float64),
        river_network=np.asarray(group["river_network"][:], dtype=bool),
        kernel_map=np.asarray(group["kernel_map"][:], dtype=np.int32),
    )


class ConditionsCache:
    """条件通道缓存（§4.3）：进程内命中，可选 Zarr 落盘跨进程复用。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._memory: dict[str, ConditionChannels] = {}

    def get_or_build(self, key: str, builder: Callable[[], ConditionChannels]) -> ConditionChannels:
        """按 key 取缓存，未命中则调用 ``builder`` 并写入缓存。"""
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        if self._path is not None:
            loaded = _load_conditions(self._path / key)
            if loaded is not None:
                self._memory[key] = loaded
                return loaded
        channels = builder()
        self._memory[key] = channels
        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)
            _save_conditions(self._path / key, channels)
        return channels


# ===== tile 划分与拼接（§4.2） =====


@dataclasses.dataclass(frozen=True)
class Tile:
    """网格上的一个矩形区块（半开区间 ``[i0, i1) x [j0, j1)``）。"""

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.i1 - self.i0, self.j1 - self.j0)


def _axis_starts(n: int, tile_size: int, step: int) -> list[int]:
    if n <= tile_size:
        return [0]
    starts = list(range(0, n - tile_size + 1, step))
    if starts[-1] != n - tile_size:
        starts.append(n - tile_size)
    return starts


def iter_tiles(nlat: int, nlon: int, tile_size: int, overlap: int = 0) -> list[Tile]:
    """按 ``tile_size`` 与 ``overlap`` 划分 tile，保证无遗漏覆盖（§4.2）。

    末块向内回退使边缘恰好落在网格边界上，因此边缘 tile 可能比 ``tile_size``
    略小或与邻块重叠更多。返回顺序为行优先（先纬度后经度）。
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size 必须为正，实际为 {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负，实际为 {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap={overlap} 必须小于 tile_size={tile_size}")
    step = tile_size - overlap
    i_starts = _axis_starts(nlat, tile_size, step)
    j_starts = _axis_starts(nlon, tile_size, step)
    return [
        Tile(i0, min(i0 + tile_size, nlat), j0, min(j0 + tile_size, nlon))
        for i0 in i_starts
        for j0 in j_starts
    ]


def tile_weight_map(tiles: list[Tile], shape: tuple[int, int]) -> np.ndarray:
    """每格的 tile 覆盖数（拼接归一化权重）。"""
    weight = np.zeros(shape, dtype=np.float64)
    for t in tiles:
        weight[t.i0 : t.i1, t.j0 : t.j1] += 1.0
    return weight


def stitch_tiles(tiles: list[Tile], pieces: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """按覆盖数归一化拼接 tile（重叠区取加权平均）。

    内置精修器由全局球面坐标决定输出，重叠区逐点一致，故拼接为恒等操作；
    对边缘有伪影的模型，覆盖平均也优于直接覆盖写。
    """
    if len(tiles) != len(pieces):
        raise ValueError(f"tiles({len(tiles)}) 与 pieces({len(pieces)}) 数量不一致")
    total = np.zeros(shape, dtype=np.float64)
    for t, piece in zip(tiles, pieces, strict=True):
        arr = np.asarray(piece, dtype=np.float64)
        if arr.shape != t.shape:
            raise ValueError(f"tile {t} 的残差形状 {arr.shape} 与期望 {t.shape} 不一致")
        total[t.i0 : t.i1, t.j0 : t.j1] += arr
    weight = tile_weight_map(tiles, shape)
    if np.any(weight == 0.0):
        raise ValueError("tile 未完整覆盖网格")
    return np.asarray(total / weight, dtype=np.float64)


# ===== 约束强制（§三） =====


def _masked_block_mean(field: np.ndarray, mask: np.ndarray, block: int) -> np.ndarray:
    """掩码块均值：每块内只对掩码单元求均值（块内无掩码时返回 0）。"""
    bl = max(int(block), 1)
    nlat, nlon = field.shape
    pad = [(0, (-nlat) % bl), (0, (-nlon) % bl)]
    f = np.asarray(np.pad(field, pad, mode="edge"), dtype=np.float64)
    m = np.asarray(np.pad(mask.astype(np.float64), pad, mode="edge"), dtype=np.float64)
    nl, no = f.shape[0] // bl, f.shape[1] // bl
    num = np.asarray((f * m).reshape(nl, bl, no, bl).sum(axis=(1, 3)), dtype=np.float64)
    den = np.asarray(m.reshape(nl, bl, no, bl).sum(axis=(1, 3)), dtype=np.float64)
    return np.asarray(num / np.maximum(den, 1.0), dtype=np.float64)


def enforce_residual_constraints(
    residual: np.ndarray,
    tectonic: np.ndarray,
    *,
    block: int = BLOCK,
    coastline_eps: float = COASTLINE_EPS,
) -> np.ndarray:
    """强制残差满足三条数学约束（§三、§3.3）。

    顺序即不变量：海岸线清零 → 掩码块均值高通（低频截断）→ 再次海岸线清零
    → 支撑集零均值。掩码块均值使块内陆地均值为零，因此随后的支撑集零均值
    只是扣除 ≈0 的全局常数，两者可同时精确成立。
    """
    tec = np.asarray(tectonic, dtype=np.float64)
    r = np.array(residual, dtype=np.float64, copy=True)
    if r.shape != tec.shape:
        raise ValueError(f"residual 形状 {r.shape} 与 tectonic {tec.shape} 不一致")
    if block < 2:
        raise ValueError(f"block 必须 >= 2，实际为 {block}")

    land = np.abs(tec) >= coastline_eps
    r[~land] = 0.0
    if land.any():
        r -= nr.expand_tile(_masked_block_mean(r, land, block), block, tec.shape)
        r[~land] = 0.0
        r[land] -= r[land].mean()
    return r


def validate_constraints(
    result: DiffusionRefineResult,
    *,
    block: int = BLOCK,
    coastline_eps: float = COASTLINE_EPS,
    mean_rtol: float = 1e-6,
    lowfreq_rtol: float = 0.05,
    coast_atol: float = 1e-6,
) -> list[str]:
    """校验扩散精修输出是否满足约束（§六 第 7 步），返回问题描述列表。

    容差相对于残差幅度：``scale = max|H_residual|``，因此不同量级的地形
    都使用同一套相对判据。
    """
    r = np.asarray(result.residual, dtype=np.float64)
    tec = np.asarray(result.tectonic, dtype=np.float64)
    elev = np.asarray(result.elevation, dtype=np.float64)
    problems: list[str] = []
    if not np.all(np.isfinite(r)):
        problems.append("残差含非有限值")
        return problems

    scale = float(np.abs(r).max())
    scale = scale if scale > 0.0 else 1.0
    land = np.abs(tec) >= coastline_eps
    coast = ~land

    if land.any():
        mean_land = abs(float(r[land].mean()))
        if mean_land > mean_rtol * scale:
            problems.append(f"支撑集均值约束违反: |mean|={mean_land:.3e} > {mean_rtol * scale:.3e}")
        lowfreq = np.abs(_masked_block_mean(r, land, block)).max()
        if lowfreq > lowfreq_rtol * scale:
            problems.append(
                f"低频截断约束违反: |块均值|max={lowfreq:.3e} > {lowfreq_rtol * scale:.3e}"
            )
    if coast.any():
        # 方案 §3.3：海岸线处约束的是最终高程 H_final，而非仅残差——海陆边界
        # 必须留在构造层设定的位置，否则细节层会推移岸线。
        coast_max = float(np.abs(elev[coast]).max())
        if coast_max > coast_atol:
            problems.append(f"海岸线约束违反: |H_final|max={coast_max:.3e} > {coast_atol:.3e}")
    return problems


# ===== 三频带融合（§5.1） =====


def _radial_wavenumber(shape: tuple[int, int]) -> np.ndarray:
    """归一化径向波数（周期/格），范围 ``[0, ~0.707]``，形状 ``shape``。"""
    kx = np.fft.fftfreq(shape[0])[:, None]
    ky = np.fft.fftfreq(shape[1])[None, :]
    return np.asarray(np.sqrt(kx**2 + ky**2), dtype=np.float64)


def _cosine_step(k: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """余弦平滑阶跃：``k <= lo`` 为 1，``k >= hi`` 为 0，之间余弦过渡。"""
    if hi <= lo:
        return np.where(k <= lo, 1.0, 0.0)
    t = np.clip((k - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(0.5 * (1.0 + np.cos(np.pi * t)))


def frequency_windows(
    shape: tuple[int, int],
    *,
    k_low: float = K_LOW,
    k_mid: float = K_MID,
    k_high: float = K_HIGH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三频带窗 ``(w_low, w_mid, w_high)``（§5.1）。

    ``w_low = s(k_low, k_mid)``、``w_high = 1 - s(k_mid, k_high)``、
    ``w_mid = 1 - w_low - w_high``，因此三者严格构成单位分解，过渡带为余弦窗。
    """
    if not (0.0 < k_low <= k_mid <= k_high):
        raise ValueError(
            f"频带截断必须满足 0 < k_low <= k_mid <= k_high，实际 {k_low}, {k_mid}, {k_high}"
        )
    k = _radial_wavenumber(shape)
    w_low = _cosine_step(k, k_low, k_mid)
    w_high = 1.0 - _cosine_step(k, k_mid, k_high)
    return w_low, 1.0 - w_low - w_high, w_high


def frequency_merge(
    tectonic: np.ndarray,
    noise: np.ndarray,
    diffusion: np.ndarray,
    *,
    k_low: float = K_LOW,
    k_mid: float = K_MID,
    k_high: float = K_HIGH,
) -> np.ndarray:
    """分频融合（§5.1）：低频=构造层，中频=噪声基底，高频=扩散细节。

    ``H_final = IFFT(w_low·Ĥ_tectonic + w_mid·Ĥ_noise + w_high·Ĥ_diffusion)``。
    窗函数为单位分解，故三个输入相同时输出等于该输入（空操作）。
    经纬网格的 FFT 不是严格的谱方法，但用于频带分离（平滑算子）是合适的。
    """
    tec = np.asarray(tectonic, dtype=np.float64)
    noi = np.asarray(noise, dtype=np.float64)
    dif = np.asarray(diffusion, dtype=np.float64)
    if not (tec.shape == noi.shape == dif.shape):
        raise ValueError(f"三个输入的形状必须一致: {tec.shape}, {noi.shape}, {dif.shape}")
    w_low, w_mid, w_high = frequency_windows(tec.shape, k_low=k_low, k_mid=k_mid, k_high=k_high)
    spectrum = w_low * np.fft.fft2(tec) + w_mid * np.fft.fft2(noi) + w_high * np.fft.fft2(dif)
    return np.asarray(np.fft.ifft2(spectrum).real, dtype=np.float64)


# ===== 精修器实现 =====


@dataclasses.dataclass(frozen=True)
class TileConditions:
    """单个 tile 的条件信号切片（含全局球面坐标，保证跨 tile 无缝）。"""

    tile: Tile
    tectonic: np.ndarray
    lowpass: np.ndarray
    land_mask: np.ndarray
    boundary_distance: np.ndarray
    river_network: np.ndarray
    kernel_map: np.ndarray  # §1.3 选核图（与噪声基底一致的构造纹理）
    init_noise: np.ndarray  # §5.2 第 2 步：扩散采样的初始噪声
    xyz: np.ndarray  # (h, w, 3) 全局球面坐标

    @classmethod
    def from_global(
        cls,
        tile: Tile,
        tectonic: np.ndarray,
        channels: ConditionChannels,
        coords: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        noise: np.ndarray | None = None,
    ) -> TileConditions:
        """从全局场切出该 tile 的条件（``coords`` 可为 ``(..., 3)`` 或三元组）。

        ``noise`` 为方案 §5.2 第 2 步的初始噪声（fBm + 域扭曲结果）；None 时为零场，
        表示精修器自行合成细节。
        """
        sl = (slice(tile.i0, tile.i1), slice(tile.j0, tile.j1))
        if isinstance(coords, tuple):
            coords = np.stack(coords, axis=-1)
        xyz = np.asarray(coords, dtype=np.float64)[sl]
        tectonic_tile = np.asarray(tectonic, dtype=np.float64)[sl]
        init_noise = (
            np.zeros_like(tectonic_tile)
            if noise is None
            else np.asarray(noise, dtype=np.float64)[sl]
        )
        return cls(
            tile=tile,
            tectonic=tectonic_tile,
            lowpass=np.asarray(channels.lowpass, dtype=np.float64)[sl],
            land_mask=np.asarray(channels.land_mask, dtype=bool)[sl],
            boundary_distance=np.asarray(channels.boundary_distance, dtype=np.float64)[sl],
            river_network=np.asarray(channels.river_network, dtype=bool)[sl],
            kernel_map=np.asarray(channels.kernel_map, dtype=np.int32)[sl],
            init_noise=init_noise,
            xyz=xyz,
        )


class StructuredDiffusionRefiner:
    """内置确定性结构化精修器（§5.2 可独立使用的精修层）。

    不依赖任何外部模型：用 3D 球面噪声合成高平细节，并按条件通道调制——
    造山带（靠近板块边界）振幅更大、河网处下切、**按 §1.3 分配策略表逐单元选核**
    （与噪声基底同源，因此两条第二层路径的纹理一致）。

    方案 §5.2 第 2 步要求"用 fBm + 域扭曲结果作为扩散采样的初始噪声"：给定
    ``conditions.init_noise`` 时直接以它作为细节基底（不再合成），否则回退到
    自身合成的 Ridged/Simplex 细节。两种情形都**只依赖逐点条件与全局坐标**，
    因此重叠 tile 逐点一致（不做 tile 内归一化，否则会撕出接缝）。
    """

    def __init__(
        self,
        *,
        amplitude: float = 260.0,
        freq_scale: float = 14.0,
        octaves: int = 5,
        boundary_gain: float = 0.8,
        boundary_decay: float = 6.0,
        river_carve: float = 0.6,
        warp_amp: float = nr.WARP_AMP,
    ) -> None:
        self.amplitude = float(amplitude)
        self.freq_scale = float(freq_scale)
        self.octaves = int(octaves)
        self.boundary_gain = float(boundary_gain)
        self.boundary_decay = float(boundary_decay)
        self.river_carve = float(river_carve)
        self.warp_amp = float(warp_amp)

    def refine_tile(self, conditions: TileConditions, seed: int) -> np.ndarray:
        xyz = conditions.xyz
        s = self.freq_scale
        x, y, z = xyz[..., 0] * s, xyz[..., 1] * s, xyz[..., 2] * s

        # §1.3 分配策略表：造山带 Ridged、洋中脊 Worley、海沟 Turbulence、其余 Simplex
        ridged = nr.build_kernel_fields(
            x,
            y,
            z,
            seed * 7 + 1,
            np.asarray(conditions.kernel_map, dtype=np.int32),
            octaves=self.octaves,
            warp_amp=self.warp_amp,
        )

        # §5.2 第 2 步：有初始噪声时以它作为细节基底（扩散采样从统计合理的起点开始）
        init_noise = np.asarray(conditions.init_noise, dtype=np.float64)
        if np.any(init_noise != 0.0):
            detail = init_noise
            amplitude = 1.0  # 初始噪声已由上游标定，不再乘振幅
        else:
            detail = ridged
            amplitude = self.amplitude

        # 条件通道 3：边界距离场控制振幅——造山带更陡峭
        gain = 1.0 + self.boundary_gain * np.exp(
            -np.asarray(conditions.boundary_distance) / self.boundary_decay
        )
        field = amplitude * detail * gain

        # 条件通道 4：河网下切（负向）
        field = field - self.river_carve * self.amplitude * conditions.river_network
        return np.asarray(field, dtype=np.float64)


class TerrainDiffusionRefiner:
    """Terrain Diffusion 模型适配器（§2.1、§4.1）。

    惰性加载 Hugging Face 权重：只有真正调用 :meth:`refine_tile` 时才导入
    推理栈，因此无模型/无网络环境下本项目其余部分照常运行。不可用时抛出
    含模型标识的 :class:`RuntimeError`，而不是静默退化为噪声——静默降级会让
    调用方误以为拿到了模型输出。

    ``allow_download=False`` 时只接受本地已缓存的权重，适合离线与 CI。
    """

    def __init__(
        self,
        model_id: str = "xandergos/terrain-diffusion-90m",
        *,
        device: str | None = None,
        dtype: str = "float32",
        steps: int = 4,
        allow_download: bool = True,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device
        self.dtype = str(dtype)
        self.steps = int(steps)
        self.allow_download = bool(allow_download)
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError(
                f"扩散模型 {self.model_id} 需要 diffusers 与 torch；"
                f"请先安装（pip install diffusers torch），或改用 StructuredDiffusionRefiner"
            ) from exc
        if not self.allow_download:
            raise RuntimeError(
                f"扩散模型 {self.model_id} 未在本地缓存，且 allow_download=False；"
                f"请先在线下载权重，或改用 StructuredDiffusionRefiner"
            )
        device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        pipe = DiffusionPipeline.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_id, torch_dtype=getattr(torch, self.dtype)
        )
        self._pipeline = pipe.to(device)
        return self._pipeline

    def refine_tile(self, conditions: TileConditions, seed: int) -> np.ndarray:
        """以条件通道为条件输入做去噪，返回残差 tile（§3.2、§3.3、§5.2）。

        方案 §5.2 第 2 步：把该 tile 的 fBm + 域扭曲结果作为扩散采样的**初始噪声**
        （``latents``），而不是纯高斯噪声——采样从统计合理的起点开始，可减少步数。
        """
        pipe = self._load()
        import torch

        cond = np.stack(
            [
                np.asarray(conditions.lowpass, dtype=np.float64),
                np.asarray(conditions.land_mask, dtype=np.float64),
                np.asarray(conditions.boundary_distance, dtype=np.float64),
                np.asarray(conditions.river_network, dtype=np.float64),
            ],
            axis=0,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        init_noise = np.asarray(conditions.init_noise, dtype=np.float64)
        kwargs: dict[str, Any] = {
            "image": torch.from_numpy(cond).unsqueeze(0).float(),
            "num_inference_steps": self.steps,
            "generator": generator,
        }
        if np.any(init_noise != 0.0):
            kwargs["latents"] = torch.from_numpy(init_noise)[None, None, ...].float()
        with torch.no_grad():
            out = pipe(**kwargs).images[0]
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        if arr.shape != conditions.tectonic.shape:
            raise RuntimeError(
                f"模型 {self.model_id} 返回形状 {arr.shape} 与 tile {conditions.tectonic.shape} 不一致"
            )
        return arr


# ===== 顶层管线（§6） =====


@dataclasses.dataclass(frozen=True)
class DiffusionRefineResult:
    """扩散精修层输出。"""

    elevation: np.ndarray  # H_final
    residual: np.ndarray  # H_residual（约束已强制）
    tectonic: np.ndarray  # 输入构造高程
    conditions: ConditionChannels
    noise: np.ndarray  # 中频噪声基底（未提供时为零场）
    report: dict[str, float | int | str]
    seed: int


def refine_diffusion(
    tectonic: np.ndarray,
    boundary_type: np.ndarray,
    *,
    seed: int = 0,
    noise: np.ndarray | None = None,
    refiner: DiffusionRefiner | None = None,
    tile_size: int = 32,
    overlap: int = 4,
    block: int = BLOCK,
    river_threshold: float | None = None,
    river_network: np.ndarray | None = None,
    coastline_eps: float = COASTLINE_EPS,
    k_low: float = K_LOW,
    k_mid: float = K_MID,
    k_high: float = K_HIGH,
    coords_scale: float = COORDS_SCALE,
    cache: ConditionsCache | None = None,
) -> DiffusionRefineResult:
    """扩散精修完整管线（§6 伪代码 1–8）。

    参数：
        tectonic: 构造层高程 H_tectonic（m），形状 ``(nlat, nlon)``
        boundary_type: 板块边界类型场（:class:`euler_poles.BoundaryType` 像素值）
        seed: 确定性种子（同时驱动精修器）
        noise: 中频噪声基底 H_noise；None 时该频带无贡献（§5.2 可省略）。
            同时作为 §5.2 第 2 步的**扩散采样初始噪声**传给精修器。
        refiner: 精修器；None 时用 :class:`StructuredDiffusionRefiner`
        tile_size / overlap: tile 划分（§4.2 批量推理）
        block: 低频截断的块尺度
        river_threshold / river_network: 条件通道 4 的来源（见 :func:`build_condition_channels`）
        coastline_eps: 海岸线判定阈值（m）
        k_low / k_mid / k_high: 三频带截断（周期/格）
        coords_scale: 球面坐标频率缩放
        cache: 条件通道缓存（§4.3）

    返回 :class:`DiffusionRefineResult`；约束由管线强制，可用
    :func:`validate_constraints` 独立复核。

    网格限制：本路径的 tile 划分与三频带融合都建立在**矩形周期网格**上
    （``np.fft.fft2`` 沿最后两轴做频带分离，文档字符串见 :func:`frequency_merge`），
    因此只接受 ``(nlat, nlon)``。立方球网格请走 :func:`noise_refine.refine_noise`
    （噪声路径已支持 ``(6, n, n)``），或用
    :func:`plate_tectonics.regrid_tectonic_result` 先插值回经纬网格。
    """
    if seed < 0:
        raise ValueError(f"seed 不能为负，实际为 {seed}")
    tec = np.asarray(tectonic, dtype=np.float64)
    bt = np.asarray(boundary_type, dtype=np.int32)
    if tec.shape != bt.shape:
        raise ValueError(f"tectonic 与 boundary_type 形状不一致: {tec.shape} vs {bt.shape}")
    if tec.ndim != 2:
        raise ValueError(
            f"扩散精修仅支持 (nlat, nlon) 经纬网格，实际 {tec.shape}；"
            "立方球网格请用 noise_refine.refine_noise，或先 regrid_tectonic_result 转回经纬网格"
        )
    nlat, nlon = tec.shape

    # 2–3：条件信号预计算（§6 第 3 步）
    def _build() -> ConditionChannels:
        return build_condition_channels(
            tec,
            bt,
            block=block,
            river_threshold=river_threshold,
            river_network=river_network,
        )

    if cache is None:
        conditions = _build()
    else:
        key = condition_cache_key(tec, bt, block=block, river_threshold=river_threshold)
        conditions = cache.get_or_build(key, _build)

    # 中频噪声基底（§5.1 的中频带 + §5.2 第 2 步的初始噪声，同一个场）
    noise_field = np.zeros_like(tec) if noise is None else np.asarray(noise, dtype=np.float64)
    if noise_field.shape != tec.shape:
        raise ValueError(f"noise 形状 {noise_field.shape} 与构造场 {tec.shape} 不一致")

    # 4：逐 tile 条件去噪（§6 第 4a–4d 步）
    active_refiner: DiffusionRefiner = (
        refiner if refiner is not None else StructuredDiffusionRefiner()
    )
    coords = sphere_coords(nlat, nlon, scale=coords_scale)
    tiles = iter_tiles(nlat, nlon, tile_size, overlap)
    pieces = [
        active_refiner.refine_tile(
            TileConditions.from_global(t, tec, conditions, coords, noise_field), seed=seed
        )
        for t in tiles
    ]

    # 5：拼接 + 约束强制（§三、§6 第 5–7 步）
    stitched = stitch_tiles(tiles, pieces, (nlat, nlon))
    residual = enforce_residual_constraints(stitched, tec, block=block, coastline_eps=coastline_eps)

    # 6：三频带融合（§5.1）
    elevation = frequency_merge(tec, noise_field, residual, k_low=k_low, k_mid=k_mid, k_high=k_high)

    # 7：海岸线钳制（§3.3）——方案约束的是 H_final 而非仅残差。频域融合是
    # 全局算子，会在岸线处引入中高频偏移；钳制把岸线钉回构造层设定位置，
    # 从而保证海陆边界不被细节层推移。
    coast = np.abs(tec) < coastline_eps
    if coast.any():
        elevation = elevation.copy()
        elevation[coast] = 0.0

    report: dict[str, float | int | str] = {
        "refiner": type(active_refiner).__name__,
        "n_tiles": len(tiles),
        "residual_std_m": float(np.std(residual)),
        "residual_max_abs_m": float(np.abs(residual).max()),
        "river_cells": int(conditions.river_network.sum()),
        "land_fraction": float(conditions.land_mask.mean()),
    }
    return DiffusionRefineResult(
        elevation=elevation,
        residual=residual,
        tectonic=tec,
        conditions=conditions,
        noise=noise_field,
        report=report,
        seed=seed,
    )


# ===== 分层扩散堆叠（§3.4）=====


@dataclasses.dataclass(frozen=True)
class DiffusionLevel:
    """分层扩散堆叠中的一层（§3.4）。

    方案 §3.4 要求"每一层以上一层的输出作为条件"：粗层建立大尺度结构，细层在被
    粗层更新后的地形上继续注入更细尺度的细节。本项目的网格分辨率固定，因此用
    ``freq_scale``（噪声特征频率）区分层级的**尺度**，而不是真的换网格——
    这与 §3.4 "全局结构由最粗层决定，细节由最细层补充"的作用等价。

    方案 §3.4 还指出本项目"可以跳过最粗层，直接从 1 km 分辨率的中层扩散模型开始"，
    因此 :func:`default_levels` 返回的中层起步序列即为方案默认。
    """

    #: 本层的球面坐标频率缩放（越大特征尺度越细）
    freq_scale: float
    #: 本层残差幅度 = ``residual_fraction × std(上一层地形)``
    residual_fraction: float = LEVEL_RESIDUAL_FRACTION
    #: 本层 tile 边长与重叠（§4.2 批量推理）
    tile_size: int = 32
    overlap: int = 4


def default_levels() -> list[DiffusionLevel]:
    """方案 §3.4 的默认分层序列：跳过最粗层，自中层起步精修到细层。"""
    return [
        DiffusionLevel(freq_scale=8.0, residual_fraction=0.06),
        DiffusionLevel(freq_scale=18.0, residual_fraction=0.03),
    ]


@dataclasses.dataclass(frozen=True)
class HierarchicalRefineResult:
    """分层扩散堆叠输出（§3.4）。"""

    elevation: np.ndarray  # H_final：逐层累加后的高程
    residual: np.ndarray  # 累计残差 = elevation − tectonic（低频受约束）
    tectonic: np.ndarray  # 输入构造高程
    level_residuals: list[np.ndarray]  # 各层各自的残差（自粗到细）
    conditions: list[ConditionChannels]  # 各层的条件通道（逐层以上一层输出重建）
    report: dict[str, float | int | str]
    seed: int


def refine_diffusion_hierarchical(
    tectonic: np.ndarray,
    boundary_type: np.ndarray,
    *,
    seed: int = 0,
    noise: np.ndarray | None = None,
    levels: list[DiffusionLevel] | None = None,
    refiners: list[DiffusionRefiner] | None = None,
    block: int = BLOCK,
    river_threshold: float | None = None,
    river_network: np.ndarray | None = None,
    coastline_eps: float = COASTLINE_EPS,
    coords_scale: float = COORDS_SCALE,
    caches: list[ConditionsCache] | None = None,
) -> HierarchicalRefineResult:
    """分层扩散堆叠精修（§3.4），逐层累加残差。

    与单级 :func:`refine_diffusion` 的关系：单级只做一层条件去噪；本函数把该过程
    串成自粗到细的级联，**每一层都用上一层的输出重建条件通道**（§3.4 的核心机制），
    从而让细层在被粗层更新过的地形上继续加细节。

    层内不做三频带融合（§5.1）——堆叠本身就是频率域上的逐层累加；每层残差仍由
    :func:`enforce_residual_constraints` 强制约束，因此**任何单层都无法移动海陆
    边界与主要山脉位置**，宏观格局始终由第一层决定。

    参数：
        tectonic: 构造层高程 H_tectonic（m），形状 ``(nlat, nlon)``
        boundary_type: 板块边界类型场（:class:`euler_poles.BoundaryType` 像素值）
        seed: 确定性种子；第 k 层用 ``seed + k``，保证层间去相关
        noise: 中频噪声基底（§5.1 中频带）；按 §5.2 第 2 步充当**最粗层**扩散采样的
            初始噪声（更细的层不再重复注入同一频带）
        levels: 分层序列；None 时取 :func:`default_levels`（自中层起步）
        refiners: 每层的精修器；None 时按层构造
            :class:`StructuredDiffusionRefiner`（用该层 ``freq_scale``）
        block / river_threshold / river_network / coastline_eps / coords_scale:
            同 :func:`refine_diffusion`
        caches: 每层的条件通道缓存；None 时不缓存

    返回 :class:`HierarchicalRefineResult`。
    """
    if seed < 0:
        raise ValueError(f"seed 不能为负，实际为 {seed}")
    tec = np.asarray(tectonic, dtype=np.float64)
    bt = np.asarray(boundary_type, dtype=np.int32)
    if tec.shape != bt.shape:
        raise ValueError(f"tectonic 与 boundary_type 形状不一致: {tec.shape} vs {bt.shape}")
    if tec.ndim != 2:
        raise ValueError(
            f"分层扩散精修仅支持 (nlat, nlon) 经纬网格，实际 {tec.shape}；"
            "立方球网格请用 noise_refine.refine_noise，或先 regrid_tectonic_result 转回经纬网格"
        )
    if block < 2:
        raise ValueError(f"block 必须 >= 2，实际为 {block}")

    level_list = list(default_levels() if levels is None else levels)
    if not level_list:
        raise ValueError("levels 不能为空")
    for level in level_list:
        if level.freq_scale <= 0.0:
            raise ValueError(f"freq_scale 必须为正，实际为 {level.freq_scale}")
        if level.residual_fraction < 0.0:
            raise ValueError(f"residual_fraction 不能为负，实际为 {level.residual_fraction}")
    if refiners is not None and len(refiners) != len(level_list):
        raise ValueError(f"refiners 长度 {len(refiners)} 与层数 {len(level_list)} 不一致")
    if caches is not None and len(caches) != len(level_list):
        raise ValueError(f"caches 长度 {len(caches)} 与层数 {len(level_list)} 不一致")

    nlat, nlon = tec.shape
    coords = sphere_coords(nlat, nlon, scale=coords_scale)
    noise_field = np.zeros_like(tec) if noise is None else np.asarray(noise, dtype=np.float64)
    if noise_field.shape != tec.shape:
        raise ValueError(f"noise 形状 {noise_field.shape} 与构造场 {tec.shape} 不一致")

    base = tec  # 当前层级的"上一层输出"，逐层被更新
    level_residuals: list[np.ndarray] = []
    level_conditions: list[ConditionChannels] = []
    level_stds: list[float] = []

    for index, level in enumerate(level_list):
        level_seed = seed + index

        # §3.4：本层的条件通道由**上一层的输出**重建（最粗层即构造场本身）
        def _build(source: np.ndarray = base) -> ConditionChannels:
            return build_condition_channels(
                source,
                bt,
                block=block,
                river_threshold=river_threshold,
                river_network=river_network,
            )

        if caches is None:
            channels = _build()
        else:
            key = condition_cache_key(base, bt, block=block, river_threshold=river_threshold)
            channels = caches[index].get_or_build(key, _build)
        level_conditions.append(channels)

        refiner: DiffusionRefiner = (
            refiners[index]
            if refiners is not None
            else StructuredDiffusionRefiner(freq_scale=level.freq_scale)
        )
        # §5.2 第 2 步：最粗层用中频噪声作初始噪声；细层不再重复注入同一频带
        init_noise = noise_field if index == 0 else None

        tiles = iter_tiles(nlat, nlon, level.tile_size, level.overlap)
        pieces = [
            refiner.refine_tile(
                TileConditions.from_global(t, base, channels, coords, init_noise),
                seed=level_seed,
            )
            for t in tiles
        ]
        stitched = stitch_tiles(tiles, pieces, (nlat, nlon))

        # 本层残差强制约束后，再按 residual_fraction 标定到相对上一层地形的幅度
        residual = enforce_residual_constraints(
            stitched, base, block=block, coastline_eps=coastline_eps
        )
        std = float(np.std(residual))
        if std > 1e-12 and level.residual_fraction > 0.0:
            residual = residual / std * (level.residual_fraction * float(np.std(base)))
        elif level.residual_fraction == 0.0:
            residual = np.zeros_like(residual)

        base = base + residual
        level_residuals.append(residual)
        level_stds.append(float(np.std(residual)))

    # 累计残差与累计低频都受各层约束约束，因此宏观格局仍由构造层决定
    residual_total = np.asarray(base - tec, dtype=np.float64)
    elevation = np.asarray(base, dtype=np.float64)
    # 海岸线钳制（§3.3）：把岸线钉回构造层设定的位置
    coast = np.abs(tec) < coastline_eps
    if coast.any():
        elevation = elevation.copy()
        elevation[coast] = 0.0

    report: dict[str, float | int | str] = {
        "n_levels": len(level_list),
        "freq_scales": ",".join(f"{level.freq_scale:g}" for level in level_list),
        "level_residual_std_m": ",".join(f"{value:.1f}" for value in level_stds),
        "residual_std_m": float(np.std(residual_total)),
        "residual_max_abs_m": float(np.abs(residual_total).max()),
    }
    return HierarchicalRefineResult(
        elevation=elevation,
        residual=residual_total,
        tectonic=tec,
        level_residuals=level_residuals,
        conditions=level_conditions,
        report=report,
        seed=seed,
    )


__all__ = [
    "BLOCK",
    "COASTLINE_EPS",
    "COORDS_SCALE",
    "ConditionChannels",
    "ConditionsCache",
    "DiffusionLevel",
    "DiffusionRefineResult",
    "DiffusionRefiner",
    "HierarchicalRefineResult",
    "K_HIGH",
    "K_LOW",
    "K_MID",
    "LEVEL_RESIDUAL_FRACTION",
    "RIVER_FRACTION",
    "StructuredDiffusionRefiner",
    "TerrainDiffusionRefiner",
    "Tile",
    "TileConditions",
    "build_condition_channels",
    "condition_cache_key",
    "d8_flow_accumulation",
    "default_levels",
    "enforce_residual_constraints",
    "frequency_merge",
    "frequency_windows",
    "iter_tiles",
    "refine_diffusion",
    "refine_diffusion_hierarchical",
    "sphere_coords",
    "stitch_tiles",
    "tile_weight_map",
    "validate_constraints",
]
