"""地形阶段：把三层混合方案接入生产管线（《地形生成混合方案与程序加速策略》）。

三层此前都是"可独立调用的模块"，本模块把它们接成
:class:`~virtual_world.pipeline.PipelineStage`，形成方案 §1.1 的因果链：

    第一层 板块构造（:mod:`plate_tectonics`）
      → 第二层 噪声/扩散精修（:mod:`noise_refine` / :mod:`diffusion`）
      → 第三层 侵蚀雕刻（:mod:`erosion`，内部已含 :mod:`hydrology` 河网）

产出写入 :class:`~virtual_world.core.grid.GridState` 的 terrain 字段
（``elevation`` / ``is_ocean`` / ``ocean_depth``），供下游辐射与气候阶段消费。

**工作网格**：第三层侵蚀只接受经纬网格（``(nlat, nlon)`` 2D），因此整条链统一在
经纬网格上执行；非经纬网格（立方球 / HEALPix）在链路结束后用
:meth:`~virtual_world.core.grid.GridState.regrid` 回填到本网格形状，即方案 §1.5 的
"两者之间用插值转换"。需要立方球**原生**的第一、二层结果时可直接调用
:func:`plate_tectonics.generate_tectonic_field`（``grid="cubed_sphere"``）与
:func:`noise_refine.refine_noise`。

**跨层约束**（方案 §1.1）：第二、三层只加细节，**不改变第一层建立的宏观格局**。
实现上表现为海陆掩码始终取第一层结果，第二层精修后把海岸线钉回该划分。
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np

from ..core.constants import EARTH_RADIUS
from ..core.grid import GridState
from . import diffusion as df
from . import erosion as ero
from . import noise_refine as nr
from . import plate_tectonics as pt

#: 本阶段覆盖的规划阶段键（:data:`virtual_world.pipeline.PLANNED_STAGES` 中的条目）。
#: ``diffusion`` 指第二层的扩散精修路径（:meth:`TerrainStage._layer2` 的
#: ``use_diffusion`` / ``diffusion_levels`` 两条分支）；``landscape_evolution``
#: （Ma 尺度长期演化）**不在**此列——它未接入本阶段，仍应报告为待实现。
COVERS: tuple[tuple[str, str], ...] = (
    ("terrain", "plate_tectonics"),
    ("terrain", "noise_refine"),
    ("terrain", "diffusion"),
    ("terrain", "erosion"),
    ("terrain", "hydrology"),
)

#: 本阶段写入 GridState 的字段
OUTPUT_FIELDS: tuple[str, ...] = ("elevation", "is_ocean", "ocean_depth")

#: 海陆钉扎的岸线缩进 (m)：保证 ``sign(elevation)`` 与第一层海陆划分一致
COASTLINE_EPS = 1.0


def _pin_coastline(
    elevation: np.ndarray, is_ocean: np.ndarray, *, eps: float = COASTLINE_EPS
) -> np.ndarray:
    """把海岸线钉回第一层设定的海陆划分（方案 §1.1、第二层方案 §二）。

    第二层精修给出的是零均值残差，岸线附近仍可能把高程推过 0；这里只做 ``eps``
    量级的钳制，使海陆边界与掩码严格一致，不改变任何宏观格局。
    """
    elev = np.asarray(elevation, dtype=np.float64)
    ocean = np.asarray(is_ocean, dtype=bool)
    return np.asarray(np.where(ocean, np.minimum(elev, -eps), np.maximum(elev, eps)))


@dataclasses.dataclass
class TerrainStage:
    """三层地形生成阶段（:class:`~virtual_world.pipeline.PipelineStage` 实现）。

    参数按层分组；默认值取各模块的模块级默认，保证与单独调用一致。
    """

    name: str = "terrain"

    #: 本阶段覆盖的规划阶段键（供 ``pipeline.create_world`` 剔除未实现报告）
    covers: ClassVar[tuple[tuple[str, str], ...]] = COVERS

    # ===== 第一层：板块构造（方案第一层 §一–§七）=====
    n_seeds: int = 300
    n_major: int = 7
    time_ma: float = 200.0
    dt_ma: float = 1.0
    ocean_fraction: float = pt.DEFAULT_OCEAN_FRACTION
    subduction: bool = True
    #: 第一层地壳厚度时间积分改用 JAX（方案 §6.4，``lax.scan``）；缺省 Numba 内核。
    #: 两者数值一致；长时积分（数千步）在 GPU 上收益显著。JAX 为可选依赖
    jax_crust: bool = False

    # ===== 第二层：噪声与扩散精修（方案第二层 §一–§七）=====
    refine: bool = True
    freq_scale: float = 4.0
    octaves: int = 6
    residual_fraction: float = nr.RESIDUAL_FRACTION
    #: 用扩散模型精修（§三）；需可选依赖 ``diffusers`` 与模型权重，缺省走纯噪声路径
    use_diffusion: bool = False
    #: 分层扩散堆叠的分层序列（§3.4）；None 时用单级精修（§3.2/§3.3）。
    #: 给定多个层时改用 :func:`diffusion.refine_diffusion_hierarchical` 逐层累加
    diffusion_levels: list[df.DiffusionLevel] | None = None

    # ===== 第三层：侵蚀模拟（方案第三层 §一–§七）=====
    erode: bool = True
    hydraulic_steps: int = ero.DEFAULT_HYDRAULIC_STEPS
    #: 第三层水力侵蚀内核：``"numba"``（缺省，方案 §2.3）或 ``"jax"``
    #: （方案 §6.3，``jnp.roll`` + ``lax.scan``，高分辨率网格上编译为 XLA）。两者数值等价
    hydraulic_backend: str = "numba"
    #: 侵蚀演化期间持续构造抬升的时间跨度 (Ma)，用于叠加第三层方案 §2.2 的 ``U`` 项。
    #: 缺省 0：本层是 100 m–1 km 尺度的**微观刻蚀**，第一层给出的均衡高程已含造山结果，
    #: 直接再叠加会重复计入抬升。Ma 尺度的抬升-下切长期耦合请用
    #: :func:`landscape_evolution.landscape_evolve`（方案第三层 §2.2 的正解）
    uplift_period_ma: float = 0.0

    #: 最近一次 ``run`` 的摘要（供 ``pipeline`` 汇报；非数据字段）
    last_message: str = dataclasses.field(default="", init=False, repr=False)

    # ===== 网格 =====

    @staticmethod
    def work_shape(state: GridState) -> tuple[int, int]:
        """地形链的工作网格尺寸（经纬网格）：经纬网格取其自身，其余按分辨率换算。"""
        if state.grid_type == "latlon":
            return state.nlat, state.nlon
        nlat = max(2, int(round(180.0 / float(state.resolution))))
        return nlat, 2 * nlat

    def _regrid_kwargs(self, state: GridState) -> dict[str, int]:
        """回填到本网格形状时所需的尺寸参数。"""
        if state.grid_type == "cubed_sphere":
            return {"n_side": state.n_side}
        if state.grid_type == "healpix":
            return {"nside": state.nside}
        return {}

    # ===== 各层 =====

    def _layer1(self, nlat: int, nlon: int, seed: int, radius: float) -> pt.TectonicFieldResult:
        """第一层：板块构造模拟（方案第一层 §七 伪代码 1–12）。"""
        return pt.generate_tectonic_field(
            nlat=nlat,
            nlon=nlon,
            n_seeds=self.n_seeds,
            n_major=self.n_major,
            seed=seed,
            time_ma=self.time_ma,
            dt_ma=self.dt_ma,
            ocean_fraction=self.ocean_fraction,
            subduction=self.subduction,
            radius=radius,
            use_jax=self.jax_crust,
        )

    def _layer2(
        self, tectonic: pt.TectonicFieldResult, seed: int
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        """第二层：噪声基底（+ 可选的扩散残差精修），返回精修高程与摘要。"""
        noise = nr.refine_noise(
            tectonic.elevation,
            tectonic.boundary_type,
            seed=seed,
            freq_scale=self.freq_scale,
            octaves=self.octaves,
            residual_fraction=self.residual_fraction,
        )
        info: dict[str, float | str] = {
            "layer2_path": "diffusion" if self.use_diffusion else "noise",
            "noise_residual_std_m": float(np.std(noise.residual)),
        }
        if not self.use_diffusion:
            return noise.elevation, info
        # §5.1/§5.2：噪声基底既是中频带，也作为扩散采样的初始噪声
        if self.diffusion_levels is not None:
            # §3.4 分层扩散堆叠：逐层以上一层输出为条件，自粗到细累加
            hierarchical = df.refine_diffusion_hierarchical(
                tectonic.elevation,
                tectonic.boundary_type,
                seed=seed,
                noise=noise.residual,
                levels=self.diffusion_levels,
            )
            info["layer2_path"] = "diffusion-hierarchical"
            info["diffusion_levels"] = hierarchical.report["n_levels"]
            info["diffusion_residual_std_m"] = float(np.std(hierarchical.residual))
            return hierarchical.elevation, info
        diff = df.refine_diffusion(
            tectonic.elevation,
            tectonic.boundary_type,
            seed=seed,
            noise=noise.residual,
        )
        info["diffusion_residual_std_m"] = float(np.std(diff.residual))
        return diff.elevation, info

    def _layer3(
        self,
        elevation: np.ndarray,
        is_ocean: np.ndarray,
        uplift_rate: np.ndarray,
        seed: int,
        radius: float,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        """第三层：侵蚀雕刻（第三层方案 §七 伪代码 1–9，含 D8 河网）。"""
        erosion = ero.simulate_erosion(
            elevation,
            is_ocean,
            uplift=np.asarray(uplift_rate, dtype=np.float64) * self.uplift_period_ma,
            seed=seed,
            hydraulic_steps=self.hydraulic_steps,
            hydraulic_backend=self.hydraulic_backend,
            radius=radius,
        )
        report = erosion.report
        return erosion.elevation, {
            "river_cells": int(np.count_nonzero(erosion.river_network)),
            # 本层是秒级时间步的微观刻蚀，降幅在毫米量级，用 mm 汇报才有区分度
            "mean_land_drop_mm": float(report["mean_land_drop_m"]) * 1.0e3,
            "max_total_drop_m": float(report["max_total_drop_m"]),
            "hydraulic_backend": str(report["hydraulic_backend"]),
        }

    # ===== 执行 =====

    def run(self, state: GridState) -> GridState:
        """执行三层地形生成，把结果写入 ``state`` 的 terrain 字段并返回之。

        确定性：同一 ``state.seed`` 与同一组参数 ⇒ 逐位相同的输出场。
        """
        seed = int(state.seed)
        radius = float(state.planet.radius) if state.planet is not None else EARTH_RADIUS
        nlat, nlon = self.work_shape(state)

        # L1 构造层：海洋占比可配置（方案 §1.6-1）
        tectonic = self._layer1(nlat, nlon, seed, radius)

        # L2 精修层：只加残差，宏观格局（海陆边界、主要山脉位置）保持 L1 的设定
        if self.refine:
            surface, l2_info = self._layer2(tectonic, seed)
        else:
            surface, l2_info = (
                np.asarray(tectonic.elevation, dtype=np.float64),
                {"layer2_path": "off"},
            )
        is_ocean = np.asarray(tectonic.is_ocean, dtype=bool)
        surface = _pin_coastline(surface, is_ocean)

        # L3 雕刻层：河流切割、坡面冲刷（内部完成洼地填充 → D8 → 汇流 → 河网）
        l3_info: dict[str, float | int | str] = {}
        if self.erode:
            surface, l3_info = self._layer3(
                surface, is_ocean, tectonic.uplift_rate_m_per_ma, seed, radius
            )

        self._write_back(state, surface, is_ocean, nlat=nlat, nlon=nlon)
        self._summarize(tectonic, l2_info, l3_info, surface, is_ocean)
        return state

    def _write_back(
        self,
        state: GridState,
        surface: np.ndarray,
        is_ocean: np.ndarray,
        *,
        nlat: int,
        nlon: int,
    ) -> None:
        """把经纬网格上的三层结果写回 ``state``（非经纬网格先重网格回本形状）。"""
        work = GridState(
            nlat=nlat,
            nlon=nlon,
            resolution=180.0 / nlat,
            planet_params=state.planet,
            backend=state.backend_name,
            seed=state.seed,
        )
        work.set("elevation", np.asarray(surface, dtype=np.float64))
        work.set("is_ocean", np.asarray(is_ocean, dtype=bool))
        work.set("ocean_depth", np.where(is_ocean, np.maximum(-np.asarray(surface), 0.0), 0.0))

        if (work.grid_type, work.shape) != (state.grid_type, state.shape):
            work = work.regrid(state.grid_type, **self._regrid_kwargs(state))
        for name in OUTPUT_FIELDS:
            state.set(name, np.asarray(work.get(name)))

    def _summarize(
        self,
        tectonic: pt.TectonicFieldResult,
        l2_info: dict[str, float | str],
        l3_info: dict[str, float | int | str],
        surface: np.ndarray,
        is_ocean: np.ndarray,
    ) -> None:
        """拼装一行摘要，供 ``pipeline`` 汇报阶段产出。"""
        parts = [
            f"L1 {self.n_major}板块 海陆比{float(np.mean(is_ocean)):.2f} "
            f"H∈[{float(surface.min()):.0f},{float(surface.max()):.0f}]m"
        ]
        if l2_info.get("layer2_path") == "diffusion-hierarchical":
            parts.append(
                f"L2 分层扩散{int(l2_info.get('diffusion_levels', 0))}层 "
                f"残差σ={float(l2_info.get('diffusion_residual_std_m', 0.0)):.1f}m"
            )
        elif l2_info.get("layer2_path") == "diffusion":
            parts.append(f"L2 扩散残差σ={float(l2_info.get('diffusion_residual_std_m', 0.0)):.1f}m")
        elif l2_info.get("layer2_path") == "noise":
            parts.append(f"L2 噪声残差σ={float(l2_info.get('noise_residual_std_m', 0.0)):.1f}m")
        else:
            parts.append("L2 关闭")
        if l3_info:
            parts.append(
                f"L3 河道{int(l3_info.get('river_cells', 0))}格 "
                f"陆地均降{float(l3_info.get('mean_land_drop_mm', 0.0)):.2f}mm "
                f"最大{float(l3_info.get('max_total_drop_m', 0.0)):.1f}m"
            )
        else:
            parts.append("L3 关闭")
        parts.append(f"造山带高宽比{tectonic.orogen_aspect_ratio:.2f}")
        # 内核后端可观测：明确报告哪些路径用 JAX（§6.3/§6.4），避免"以为跑了 GPU"
        if self.jax_crust or self.hydraulic_backend == "jax":
            parts.append(
                f"内核[L1={'jax' if self.jax_crust else 'numba'} "
                f"L3={self.hydraulic_backend}]"
            )
        self.last_message = "; ".join(parts)


__all__ = ["COASTLINE_EPS", "COVERS", "OUTPUT_FIELDS", "TerrainStage"]
