"""第二层：噪声与扩散精修（《第二层完善：噪声与扩散精修》）。

核心原则：**只生成残差，不重新生成骨架** —— 最终高程
``H_final = H_tectonic + H_residual``，其中 ``H_residual`` 均值为零，
低频分量严格受 ``H_tectonic`` 约束。

管线（§7 伪代码）：
    H_tectonic + 板块边界类型 → 3D 球面坐标 → 域扭曲（含边界距离引导）
    → 按构造环境选择噪声核 → 多核噪声基底 → 高通残差（去块均值） → 幅度标定
    →（可选）扩散精修器 → H_final

关键工程细节：

- **八度去相关**（§1.2）：lacunarity 取 2.03（非精确 2.0，避免频率对齐伪影），并且
  每个八度进入下一八度前把坐标乘一个固定旋转矩阵（约 36.87°），见
  :func:`octave_rotation_matrix`。
- **域扭曲 + 边界距离引导**（§2.3）：``p' = p + A_warp*fbm(p) + B*grad d_boundary``，
  使沟谷/脊线倾向于平行或垂直于板块边界——见 :func:`boundary_guidance_offset`。
- **球面 3D 噪声**（§4.1）：用球面点的笛卡尔坐标采样，天然无极点奇点与经度接缝。
- **立方球网格**（§4.2）：``refine_noise`` 直接接受 ``(6, n, n)`` 场，坐标取
  :meth:`virtual_world.core.cubed_sphere.CubedSphere.centers_xyz`；6 面之间无需接缝处理。
- **等面积校正**（§4.3）：``area_correction=True`` 时按
  ``s_corrected = s*sqrt(A_mean/A_cell)`` 施加逐格频率校正。

加速：噪声核均以 Numba ``@njit(parallel=True)`` 编译（§6.1），并提供与纯
Python 参考实现一致的验证入口（§6.4）。

噪声核与构造环境的对应关系（§1.3 分配策略表）：

| 构造环境 | 噪声核 |
|---------|--------|
| 造山带（汇聚边界） | Ridged + Simplex |
| 大陆内部（稳定地盾） | Simplex（低振幅） |
| 洋中脊（离散边界） | Worley F1 |
| 俯冲带海沟 | Turbulence |
| 大陆架 / 转换边界 | Simplex（低振幅） |
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum

import numpy as np
from numba import njit, prange
from numba.extending import register_jitable

from ..core import spherical
from ..core.cubed_sphere import CubedSphere
from . import voronoi
from .euler_poles import BoundaryType

# ===== 噪声核枚举 =====


class Kernel(IntEnum):
    """可用噪声核（§1.3）。"""

    SIMPLEX = 0
    WORLEY = 1
    RIDGED = 2
    TURBULENCE = 3


#: fBm 默认参数（§1.2）：persistence=0.5，lacunarity=2.03（避免晶格伪影）
PERSISTENCE = 0.5
LACUNARITY = 2.03
#: 八度去相关旋转角 (deg)（§1.2 的 ``mat2(0.80, 0.60, -0.60, 0.80)`` 即 36.87°）
OCTAVE_ROTATION_DEG = 36.87
#: 域扭曲默认幅度（§2.2 典型值 0.1–0.2）
WARP_AMP = 0.15
#: 域扭曲的边界距离引导强度（§2.3 的 B）：单位与 ``warp_amp`` 同量纲（采样坐标）
BOUNDARY_GUIDANCE = 0.1
#: 残差幅度相对构造场 std 的默认比例
RESIDUAL_FRACTION = 0.3


def octave_rotation_matrix(
    angle_deg: float = OCTAVE_ROTATION_DEG,
    axis: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """八度去相关用的固定旋转矩阵 ``(3, 3)``（§1.2，罗德里格斯公式）。

    方案给出的示例是二维旋转 ``mat2(0.80, 0.60, -0.60, 0.80)``（36.87°）；在球面的
    3D 噪声里把同一角度提升为绕 ``(1,1,1)`` 的旋转，使三个坐标轴都参与去相关
    （只转平面会留下沿第三轴的相关性）。
    """
    k = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(k))
    if norm <= 0.0:
        raise ValueError("旋转轴不能为零向量")
    k = k / norm
    theta = np.deg2rad(float(angle_deg))
    skew = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64
    )
    return np.asarray(
        np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew), dtype=np.float64
    )


#: 八度之间的固定旋转（模块级缓存，避免每个八度重复构造）
_OCTAVE_ROTATION = octave_rotation_matrix()


def _rotate_octave_coords(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对采样坐标施加八度去相关旋转（§1.2）。"""
    r = _OCTAVE_ROTATION
    return (
        r[0, 0] * x + r[0, 1] * y + r[0, 2] * z,
        r[1, 0] * x + r[1, 1] * y + r[1, 2] * z,
        r[2, 0] * x + r[2, 1] * y + r[2, 2] * z,
    )


#: 12 个归一化梯度方向（3D Simplex，无 perm 表的散列梯度版本）
_GRADIENTS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
)


# ===== 确定性散列（单一来源：Python 可调用，Numba 内联加速） =====


@register_jitable
def _hash3(i: int, j: int, k: int, seed: int) -> int:
    """整数格点散列 → [0, 0xFFF]，确定性伪随机。"""
    h = i * 374761393 + j * 668265263 + k * 1442695041 + seed * 22695477
    h = (h ^ (h >> 13)) % 2147483648
    h = (h * 1664525) % 4294967296
    return h % 4096


@register_jitable
def _simplex_corner(
    x: float,
    y: float,
    z: float,
    i: int,
    j: int,
    k: int,
    seed: int,
) -> float:
    """单个角点的衰减贡献 ``t^4 * (g · p)``。"""
    t = 0.6 - x * x - y * y - z * z
    if t <= 0.0:
        return 0.0
    t2 = t * t
    t4 = t2 * t2
    g = _GRADIENTS[_hash3(i, j, k, seed) % 12]
    return t4 * (g[0] * x + g[1] * y + g[2] * z)


@register_jitable
def _simplex3(point_x: float, point_y: float, point_z: float, seed: int) -> float:
    """单点 3D Simplex 噪声（Gustavson 算法，散列梯度），输出约 [-1, 1]。"""
    f3 = 1.0 / 3.0
    g3 = 1.0 / 6.0
    s = (point_x + point_y + point_z) * f3
    i = int(np.floor(point_x + s))
    j = int(np.floor(point_y + s))
    k = int(np.floor(point_z + s))
    t = (i + j + k) * g3
    x0 = point_x - (i - t)
    y0 = point_y - (j - t)
    z0 = point_z - (k - t)

    if x0 >= y0:
        if y0 >= z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
        elif x0 >= z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
        else:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
    else:
        if y0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
        elif x0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
        else:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

    x1 = x0 - i1 + g3
    y1 = y0 - j1 + g3
    z1 = z0 - k1 + g3
    x2 = x0 - i2 + 2.0 * g3
    y2 = y0 - j2 + 2.0 * g3
    z2 = z0 - k2 + 2.0 * g3
    x3 = x0 - 1.0 + 3.0 * g3
    y3 = y0 - 1.0 + 3.0 * g3
    z3 = z0 - 1.0 + 3.0 * g3

    n = _simplex_corner(x0, y0, z0, i, j, k, seed)
    n += _simplex_corner(x1, y1, z1, i + i1, j + j1, k + k1, seed)
    n += _simplex_corner(x2, y2, z2, i + i2, j + j2, k + k2, seed)
    n += _simplex_corner(x3, y3, z3, i + 1, j + 1, k + 1, seed)
    return 32.0 * n


@register_jitable
def _worley3(point_x: float, point_y: float, point_z: float, seed: int) -> float:
    """单点 Worley F1 距离（到最近特征点的距离），最小邻域搜索 3³。"""
    xi = int(np.floor(point_x))
    yi = int(np.floor(point_y))
    zi = int(np.floor(point_z))
    best = 1e9
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                cx = xi + dx
                cy = yi + dy
                cz = zi + dz
                h = _hash3(cx, cy, cz, seed)
                fx = cx + (h & 0x3FF) / 1024.0
                fy = cy + ((h >> 10) & 0x3FF) / 1024.0
                fz = cz + ((h >> 20) & 0x3FF) / 1024.0
                ddx = point_x - fx
                ddy = point_y - fy
                ddz = point_z - fz
                d2 = ddx * ddx + ddy * ddy + ddz * ddz
                if d2 < best:
                    best = d2
    return float(np.sqrt(best))


@njit(parallel=True, cache=True)
def _apply_simplex(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int, out: np.ndarray) -> None:
    """逐点 Simplex 噪声（并行），原地写入 out。"""
    n = out.size
    xx = x.ravel()
    yy = y.ravel()
    zz = z.ravel()
    for idx in prange(n):  # type: ignore[no-untyped-call, attr-defined]
        out.ravel()[idx] = _simplex3(xx[idx], yy[idx], zz[idx], seed)


@njit(parallel=True, cache=True)
def _apply_worley(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int, out: np.ndarray) -> None:
    """逐点 Worley F1（并行），原地写入 out。"""
    n = out.size
    xx = x.ravel()
    yy = y.ravel()
    zz = z.ravel()
    for idx in prange(n):  # type: ignore[no-untyped-call, attr-defined]
        out.ravel()[idx] = _worley3(xx[idx], yy[idx], zz[idx], seed)


def _check_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError(f"seed 不能为负，实际为 {seed}")


def _check_octaves(octaves: int) -> None:
    if octaves < 1:
        raise ValueError(f"octaves 必须 >= 1，实际为 {octaves}")


# ===== 基础噪声核（Numba 加速） =====


def simplex_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """3D Simplex 噪声（§1.3），输出约 [-1, 1]，形状与输入一致。

    用球面点的 3D 笛卡尔坐标采样（§4.1），天然避免极点奇点与经度接缝。
    """
    _check_seed(seed)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    np.broadcast_shapes(xa.shape, ya.shape, za.shape)
    out = np.empty(np.broadcast_shapes(xa.shape, ya.shape, za.shape), dtype=np.float64)
    _apply_simplex(xa, ya, za, seed, out)
    return out


def worley_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Worley F1 细胞噪声（§1.3），输出 F1 距离（≥ 0）。"""
    _check_seed(seed)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    out = np.empty(np.broadcast_shapes(xa.shape, ya.shape, za.shape), dtype=np.float64)
    _apply_worley(xa, ya, za, seed, out)
    return out


def _stack_octaves(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    octaves: int,
    persistence: float,
    lacunarity: float,
    mode: str,
) -> np.ndarray:
    """多八度叠加的公共实现（§1.2）：逐八度旋转去相关 + 按模式取核。

    ``mode`` ∈ ``{"fbm", "ridged", "turbulence"}``：
    ``fbm`` 取 ``n``、``ridged`` 取 ``|n|``、``turbulence`` 取 ``1-|n|``。
    所有八度共用同一 ``seed``——去相关由 :func:`_rotate_octave_coords` 的固定旋转
    提供（方案 §1.2 的做法），而不是靠换种子。
    """
    total = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=np.float64)
    px, py, pz = x, y, z
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(octaves):
        layer = simplex_noise(px * freq, py * freq, pz * freq, seed)
        if mode == "ridged":
            total += amp * np.abs(layer)
        elif mode == "turbulence":
            total += amp * (1.0 - np.abs(layer))
        else:
            total += amp * layer
        norm += amp
        px, py, pz = _rotate_octave_coords(px, py, pz)
        amp *= persistence
        freq *= lacunarity
    return total / norm if norm > 0.0 else total


def fbm_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    octaves: int = 6,
    persistence: float = PERSISTENCE,
    lacunarity: float = LACUNARITY,
) -> np.ndarray:
    """分形布朗运动：多 octave Simplex 噪声叠加（§1.2），输出约 [-1, 1]。

    ``A_k = persistence**k``，``f_k = lacunarity**k``。lacunarity 取 2.03 避免八度间
    频率对齐产生的晶格伪影；每个八度进入下一八度前把坐标乘固定旋转矩阵去相关。
    """
    _check_seed(seed)
    _check_octaves(octaves)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    return _stack_octaves(xa, ya, za, seed, octaves, persistence, lacunarity, "fbm")


def ridged_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    octaves: int = 6,
    persistence: float = PERSISTENCE,
    lacunarity: float = LACUNARITY,
) -> np.ndarray:
    """Ridged fBm（§1.3）：``Σ amp*|n|``，取绝对值生成尖锐山脊，输出约 [0, 1]。"""
    _check_seed(seed)
    _check_octaves(octaves)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    return _stack_octaves(xa, ya, za, seed, octaves, persistence, lacunarity, "ridged")


def turbulence_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    octaves: int = 6,
    persistence: float = PERSISTENCE,
    lacunarity: float = LACUNARITY,
) -> np.ndarray:
    """Turbulence（§1.3）：``Σ amp*(1-|n|)``，生成峡谷/沟槽，输出约 [0, 1]。"""
    _check_seed(seed)
    _check_octaves(octaves)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    return _stack_octaves(xa, ya, za, seed, octaves, persistence, lacunarity, "turbulence")



# ===== 域扭曲（§2） =====


def warped_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    octaves: int = 6,
    warp_amp: float = WARP_AMP,
    warp_frequency: float = 0.5,
    warp_octaves: int = 4,
    guidance: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """域扭曲噪声（§2.2）：``f(p + warp_amp * fbm(p) + guidance)``。

    扭曲噪声频率（``warp_frequency``）低于主噪声频率，使扭曲呈现
    "大尺度弯曲"而非"局部抖动"。``warp_amp=0`` 且无 ``guidance`` 时退化为普通 fBm。
    ``guidance`` 为 §2.3 的边界距离引导项（见 :func:`boundary_guidance_offset`）。
    """
    _check_seed(seed)
    _check_octaves(octaves)
    if warp_amp < 0.0:
        raise ValueError(f"warp_amp 不能为负，实际为 {warp_amp}")
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    xw, yw, zw = _warped_coords(xa, ya, za, seed, warp_amp, warp_frequency, warp_octaves, guidance)
    return fbm_noise(xw, yw, zw, seed, octaves=octaves)


# ===== 构造环境 → 噪声核选择（§1.3 分配策略表） =====


def select_noise_kernel(boundary_type: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """按构造环境选择噪声核，返回 :class:`Kernel` 像素值场（形状同输入）。

    判定优先级（海沟 > 造山带 > 洋中脊 > 其他）：
    - **海沟**：汇聚边界 + 深海洋壳（elev < -5000 m）→ TURBULENCE
    - **造山带**：汇聚边界 + 陆地（elev > 500 m）→ RIDGED
    - **洋中脊**：离散边界 → WORLEY
    - **其余**（稳定地盾 / 大陆架 / 转换边界）→ SIMPLEX
    """
    bt = np.asarray(boundary_type, dtype=np.int32)
    elev = np.asarray(elevation, dtype=np.float64)
    conv = bt == int(BoundaryType.CONVERGENT)
    div = bt == int(BoundaryType.DIVERGENT)
    trench = conv & (elev < -5000.0)
    orogeny = conv & (elev > 500.0)
    ridge = div
    kernel = np.where(
        trench,
        int(Kernel.TURBULENCE),
        np.where(
            orogeny, int(Kernel.RIDGED), np.where(ridge, int(Kernel.WORLEY), int(Kernel.SIMPLEX))
        ),
    )
    return np.asarray(kernel, dtype=np.int32)


# ===== 残差构建与频域融合（核心原则） =====


def block_mean(field: np.ndarray, block: int) -> np.ndarray:
    """块均值降采样（作用于最后两个轴，任意前导维均可）。

    非整数整除时按边缘填充到可整除尺寸，保证 :func:`expand_tile` 可精确截断回原形状。
    """
    values = np.asarray(field)
    if values.ndim < 2:
        raise ValueError(f"field 至少需要 2 维，实际 {values.shape}")
    lead = values.shape[:-2]
    nlat, nlon = values.shape[-2], values.shape[-1]
    bl = max(int(block), 1)
    pad_h = (-nlat) % bl
    pad_w = (-nlon) % bl
    if pad_h or pad_w:
        values = np.pad(values, ((0, 0),) * len(lead) + ((0, pad_h), (0, pad_w)), mode="edge")
    nl, no = values.shape[-2] // bl, values.shape[-1] // bl
    return values.reshape(*lead, nl, bl, no, bl).mean(axis=(-3, -1))


def expand_tile(coarse: np.ndarray, block: int, shape: tuple[int, ...]) -> np.ndarray:
    """将块均值场扩展回原分辨率（分块常数，作为低频分量近似）。"""
    values = np.asarray(coarse)
    bl = max(int(block), 1)
    lead = (1,) * (len(shape) - 2)
    return np.kron(values, np.ones(lead + (bl, bl)))[..., : shape[-2], : shape[-1]]


def _warped_coords(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    warp_amp: float,
    warp_frequency: float,
    warp_octaves: int,
    guidance: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算域扭曲后的采样坐标 ``p + A_warp * fbm(p) + guidance``（§2.1、§2.3）。

    三个独立的低频扭曲场（去相关 seed）；``warp_amp=0`` 时只保留 ``guidance``。
    """
    ox = np.zeros_like(x)
    oy = np.zeros_like(y)
    oz = np.zeros_like(z)
    if warp_amp > 0.0:
        ox = ox + warp_amp * fbm_noise(x * warp_frequency, y, z, seed * 3 + 1, octaves=warp_octaves)
        oy = oy + warp_amp * fbm_noise(x, y * warp_frequency, z, seed * 3 + 2, octaves=warp_octaves)
        oz = oz + warp_amp * fbm_noise(x, y, z * warp_frequency, seed * 3 + 3, octaves=warp_octaves)
    if guidance is not None:
        ox = ox + guidance[0]
        oy = oy + guidance[1]
        oz = oz + guidance[2]
    if warp_amp == 0.0 and guidance is None:
        return x, y, z
    return x + ox, y + oy, z + oz


def build_kernel_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    kernel: np.ndarray,
    octaves: int = 6,
    warp_amp: float = WARP_AMP,
    warp_frequency: float = 0.5,
    warp_octaves: int = 4,
    guidance: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """按核标签场组装多核噪声基底（§1.3 分配策略表、§5.2 第 2 步）。

    所有核先在**域扭曲后的坐标**上采样（§2 关键变换），再按标签逐单元取核：
    Simplex 用普通 fBm；Ridged / Turbulence 用对应变体；Worley 用 F1 距离场。
    返回形状与 ``kernel`` 相同的噪声基底，各核输出归一化到约 [-1, 1]。
    """
    kk = np.asarray(kernel, dtype=np.int32)
    xw, yw, zw = _warped_coords(
        x, y, z, seed, warp_amp, warp_frequency, warp_octaves, guidance
    )
    fields: list[np.ndarray] = []
    fields.append(fbm_noise(xw, yw, zw, seed=seed * 5 + 1, octaves=octaves))
    fields.append(worley_noise(xw * 2.0, yw * 2.0, zw * 2.0, seed=seed * 5 + 2) / 2.0 - 0.5)
    fields.append(ridged_noise(xw, yw, zw, seed=seed * 5 + 3, octaves=octaves) - 0.5)
    fields.append(turbulence_noise(xw, yw, zw, seed=seed * 5 + 4, octaves=octaves) - 0.5)
    stacked = np.stack(fields, axis=0)  # (4, nlat, nlon)
    base = np.take_along_axis(stacked, kk[None, ...], axis=0)[0]
    return np.asarray(base)


# ===== 扩散精修接口 =====
# 扩散精修器协议、条件通道与约束强制归 :mod:`virtual_world.terrain.diffusion`
# 所有；本模块只负责噪声基底与残差构建，是扩散层可选的降级路径（方案 §5.2）。


@dataclasses.dataclass(frozen=True)
class NoiseRefineResult:
    """噪声精修层输出。"""

    elevation: np.ndarray  # H_final = H_tectonic + H_residual
    residual: np.ndarray  # H_residual（零均值，中高频）
    tectonic: np.ndarray  # 输入构造高程
    kernel_map: np.ndarray  # 每单元选用的噪声核标签
    seed: int


def boundary_distance_field(
    boundary_type: np.ndarray, neighbours: np.ndarray | None = None
) -> np.ndarray:
    """到最近非转换板块边界的距离场（§3.2 条件通道 3）。

    多源 BFS 跳数（网格无关：邻接由 ``neighbours`` 给出，缺省按形状推断——经纬网格
    或立方球 ``(6, n, n)``，见 :func:`voronoi.grid_neighbours`）。非转换边界
    （汇聚/离散）为源、距离为 0；转换边界地形表现弱，板块内部（INTERIOR）不是边界。
    无任何边界时全 0（表示"无构造引导信息"）。
    """
    bt = np.asarray(boundary_type, dtype=np.int32)
    nbr = voronoi.grid_neighbours(bt.shape, neighbours)
    source = ((bt == int(BoundaryType.CONVERGENT)) | (bt == int(BoundaryType.DIVERGENT))).ravel()

    dist = np.full(source.size, np.inf, dtype=np.float64)
    dist[source] = 0.0
    frontier = source.copy()
    layer = 0
    while frontier.any():
        layer += 1
        reached = np.zeros_like(frontier)
        for d in range(4):
            nb = nbr[d]
            valid = nb >= 0
            sel = frontier & valid
            reached[nb[sel]] = True
        new = reached & np.isinf(dist)
        if not new.any():
            break
        dist[new] = layer
        frontier = new
    dist = np.where(np.isfinite(dist), dist, 0.0)
    return np.asarray(dist.reshape(bt.shape), dtype=np.float64)


def boundary_guidance_offset(
    boundary_type: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    strength: float = BOUNDARY_GUIDANCE,
    freq_scale: float = 1.0,
    neighbours: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§2.3 的边界距离引导项 ``B * grad d_boundary``（采样坐标空间的偏移）。

    方案形式：``p' = p + A_warp*fbm(p) + B*grad d_boundary(p)``——把"到最近板块边界
    的距离"的梯度叠加到域扭曲里，使沟谷/脊线倾向平行或垂直于板块边界，模拟基底
    构造对后期地貌的控制。

    实现：在球面切平面上对距离场做中心差分得到梯度方向（本单元指向远离边界的方向），
    幅度按全场最大值归一后乘 ``strength*freq_scale``，因此 ``strength`` 与
    ``warp_amp`` 同量纲（最大位移），且不会在远离边界处无限增长。无边界或梯度为零时
    返回全零偏移（等价于不加引导）。
    """
    bt = np.asarray(boundary_type, dtype=np.int32)
    if bt.shape != np.asarray(x).shape:
        raise ValueError(f"边界类型与坐标形状不一致: {bt.shape} vs {np.asarray(x).shape}")
    nbr = voronoi.grid_neighbours(bt.shape, neighbours)
    dist = boundary_distance_field(bt, nbr).reshape(-1)
    zero = np.zeros(bt.shape, dtype=np.float64)
    if dist.max() <= 0.0:
        return zero, zero, zero

    points = np.stack(
        [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)],
        axis=-1,
    )
    norm = np.linalg.norm(points, axis=-1, keepdims=True)
    unit = points / np.where(norm > 0.0, norm, 1.0)
    flat_unit = unit.reshape(-1, 3)

    def _tangent_step(direction: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nb = nbr[direction]
        valid = nb >= 0
        safe = np.where(valid, nb, 0)
        delta = flat_unit[safe] - flat_unit  # (N, 3)
        delta = delta - np.einsum("ij,ij->i", delta, flat_unit)[:, None] * flat_unit
        length = np.linalg.norm(delta, axis=-1)
        ok = valid & (length > 1e-12)
        return delta, np.where(ok, length, 1.0), ok

    # 东(+dj) 与北(+di) 方向的切向差分
    d_east, len_east, ok_east = _tangent_step(2)
    d_north, len_north, ok_north = _tangent_step(0)
    dist_east = dist[np.where(nbr[2] >= 0, nbr[2], 0)]
    dist_north = dist[np.where(nbr[0] >= 0, nbr[0], 0)]
    grad_east = np.where(ok_east, (dist_east - dist) / np.maximum(len_east, 1e-12), 0.0)
    grad_north = np.where(ok_north, (dist_north - dist) / np.maximum(len_north, 1e-12), 0.0)

    east_hat = d_east / np.maximum(len_east, 1e-12)[:, None]
    north_hat = d_north / np.maximum(len_north, 1e-12)[:, None]
    gradient = grad_east[:, None] * east_hat + grad_north[:, None] * north_hat
    # 按**矢量模长**（而非分量）归一，使最大位移严格不超过 strength*freq_scale
    magnitude = np.linalg.norm(gradient, axis=-1)
    scale = float(magnitude.max())
    if scale <= 0.0:
        return zero, zero, zero
    offset = gradient / scale * (float(strength) * float(freq_scale))
    shaped = offset.reshape(*bt.shape, 3)
    return (
        np.asarray(shaped[..., 0], dtype=np.float64),
        np.asarray(shaped[..., 1], dtype=np.float64),
        np.asarray(shaped[..., 2], dtype=np.float64),
    )



def refine_noise(
    tectonic: np.ndarray,
    boundary_type: np.ndarray,
    seed: int = 0,
    freq_scale: float = 4.0,
    octaves: int = 6,
    warp_amp: float = WARP_AMP,
    warp_frequency: float = 0.5,
    residual_fraction: float = RESIDUAL_FRACTION,
    block: int = 4,
    *,
    boundary_guidance: float = BOUNDARY_GUIDANCE,
    area_correction: bool = False,
    neighbours: np.ndarray | None = None,
) -> NoiseRefineResult:
    """第二层噪声精修完整管线（§7 伪代码 1–8）。

    参数：
        tectonic: 构造层高程 H_tectonic（m），形状 ``(nlat, nlon)`` 或立方球 ``(6, n, n)``
        boundary_type: 板块边界类型场（:class:`euler_poles.BoundaryType` 像素值）
        seed: 确定性种子
        freq_scale: 球面坐标频率缩放（3D 噪声在单位球面坐标上的采样尺度）
        octaves: 噪声叠加八度数
        warp_amp: 域扭曲幅度
        warp_frequency: 扭曲场频率（低于主噪声频率）
        residual_fraction: 残差幅度 = fraction × std(H_tectonic)
        block: 块均值尺度（低频分离），用于保证残差不含低频能量
        boundary_guidance: §2.3 的边界距离引导强度 ``B``（0 关闭；与 ``warp_amp`` 同量纲）
        area_correction: §4.3 的等面积校正（``s*sqrt(A_mean/A_cell)``）。适合立方球等
            近等面积网格；经纬网格极区单元面积极小、校正因子极大，一般不必开启。
        neighbours: 显式邻接表（缺省按形状推断，见 :func:`voronoi.grid_neighbours`）

    返回 :class:`NoiseRefineResult`。确定性由 ``seed`` 保证。

    需要扩散模型精修时用 :func:`virtual_world.terrain.diffusion.refine_diffusion`，
    本函数是方案 §5.2 的纯噪声降级路径。
    """
    _check_seed(seed)
    if boundary_guidance < 0.0:
        raise ValueError(f"boundary_guidance 不能为负，实际为 {boundary_guidance}")
    tec = np.asarray(tectonic, dtype=np.float64)
    bt = np.asarray(boundary_type, dtype=np.int32)
    if tec.shape != bt.shape:
        raise ValueError(f"tectonic 与 boundary_type 形状不一致: {tec.shape} vs {bt.shape}")

    # 3：3D 球面坐标（§4.1、§4.2）——频率缩放作用于坐标，特征尺度随 freq_scale 变化。
    # 经纬网格由行列中心构造；立方球直接用面中心笛卡尔坐标（6 面之间天然连续）。
    if tec.ndim == 2:
        nlat, nlon = tec.shape
        lat = -90.0 + (180.0 / nlat) * (np.arange(nlat) + 0.5)
        lon = -180.0 + (360.0 / nlon) * (np.arange(nlon) + 0.5)
        phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
        cp = np.cos(phi)
        x = cp * np.cos(lam) * freq_scale
        y = cp * np.sin(lam) * freq_scale
        z = np.sin(phi) * np.ones((nlat, nlon)) * freq_scale
        areas = spherical.cell_areas(spherical.lat_edges(nlat), nlon, 1.0)
    elif tec.ndim == 3 and tec.shape[0] == 6 and tec.shape[1] == tec.shape[2]:
        unit = CubedSphere(tec.shape[1]).centers_xyz()
        x = unit[..., 0] * freq_scale
        y = unit[..., 1] * freq_scale
        z = unit[..., 2] * freq_scale
        areas = CubedSphere(tec.shape[1]).areas()
    else:
        raise ValueError(f"不支持的网格形状 {tec.shape}：应为 (nlat, nlon) 或立方球 (6, n, n)")

    # §4.3 等面积校正：s_corrected = s*sqrt(A_mean/A_cell)
    if area_correction:
        factor = np.sqrt(float(np.mean(areas)) / areas)
        x, y, z = x * factor, y * factor, z * factor

    nbr = voronoi.grid_neighbours(tec.shape, neighbours)

    # §2.3 边界距离引导：p' = p + A_warp*fbm(p) + B*grad d_boundary(p)
    guidance = None
    if boundary_guidance > 0.0:
        guidance = boundary_guidance_offset(
            bt, x, y, z, strength=boundary_guidance, freq_scale=freq_scale, neighbours=nbr
        )

    # 4–5：域扭曲（含引导） + 按构造环境选核组装噪声基底
    kernel_map = select_noise_kernel(bt, tec)
    base = build_kernel_fields(
        x,
        y,
        z,
        seed,
        kernel_map,
        octaves=octaves,
        warp_amp=warp_amp,
        warp_frequency=warp_frequency,
        guidance=guidance,
    )
    base = base - base.mean()

    # 6：残差（纯噪声路径，§5.2 降级）
    low = expand_tile(block_mean(base, block), block, tec.shape)
    high = base - low  # 高通：去掉块尺度低频
    std = float(high.std())
    residual = high if std < 1e-12 else high / std * (residual_fraction * float(tec.std()))

    elevation = tec + residual
    return NoiseRefineResult(
        elevation=elevation,
        residual=residual,
        tectonic=tec,
        kernel_map=kernel_map,
        seed=seed,
    )


# ===== 纯 Python 参考实现（加速一致性验证） =====


def python_simplex_reference(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    """:func:`simplex_noise` 的纯 Python 逐点参考实现（§6.4 一致性验证）。

    与 Numba 加速内核共用同一份 :func:`_simplex3` 代码（``register_jitable``
    单一来源），但以纯 Python 逐点方式执行，仅用于测试比对，不追求速度。
    """
    _check_seed(seed)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    shape = np.broadcast_shapes(xa.shape, ya.shape, za.shape)
    out = np.empty(shape, dtype=np.float64)
    for idx in np.ndindex(shape):
        out[idx] = _simplex3(float(xa[idx]), float(ya[idx]), float(za[idx]), seed)
    return out


__all__ = [
    "Kernel",
    "NoiseRefineResult",
    "PERSISTENCE",
    "LACUNARITY",
    "WARP_AMP",
    "RESIDUAL_FRACTION",
    "block_mean",
    "boundary_distance_field",
    "build_kernel_fields",
    "expand_tile",
    "fbm_noise",
    "python_simplex_reference",
    "refine_noise",
    "ridged_noise",
    "select_noise_kernel",
    "simplex_noise",
    "turbulence_noise",
    "warped_noise",
    "worley_noise",
]
