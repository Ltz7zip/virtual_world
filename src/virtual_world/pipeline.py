"""主管线编排：物理因果链的逐阶段执行。

因果链（见《项目结构与技术栈》§5.1）：

    行星参数 → 地形 → 辐射 → 地表能量 → 大气动力学 → 风应力 → 海洋
        ↓                                              ↓
    气候场 → 降水 → 生物群系 → 土壤 → 农业

每一阶段以 :class:`GridState` 为输入输出，通过 :class:`PipelineStage` 协议解耦。
当前仅核心数据模型已实现；物理阶段会随实施路线图逐个接入并在此注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .core.grid import GridState
from .core.planet import PlanetParams
from .core.rng import DeterministicRNG


class StageStatus(StrEnum):
    """阶段状态。"""

    NOT_IMPLEMENTED = "not_implemented"
    READY = "ready"
    FAILED = "failed"


@dataclass
class StageResult:
    """单阶段的执行结果。"""

    name: str
    status: StageStatus
    elapsed_s: float = 0.0
    message: str = ""
    output_fields: list[str] = field(default_factory=list)


class PipelineStage(Protocol):
    """物理阶段的接口：以 GridState 进、以 GridState 出。"""

    name: str

    def run(self, state: GridState) -> GridState:  # pragma: no cover - 协议
        ...


@dataclass
class RuntimeConfig:
    """一次世界生成的全部运行时参数。"""

    seed: int = 0
    planet: PlanetParams = field(default_factory=PlanetParams)
    resolution: float = 2.0  # deg
    backend: str = "auto"
    nz: int = 1
    output_dir: str = "data/output"
    cache_dir: str = "data/cache"
    verbose: bool = True
    #: 逐阶段调用的实现；空表则仅初始化网格（核心阶段）
    stages: list[PipelineStage] = field(default_factory=list)

    def make_rng(self, name: str = "world") -> DeterministicRNG:
        """为本次生成派生确定性随机源。"""
        return DeterministicRNG(self.seed).spawn(name)


#: 文档规划的阶段顺序（物理因果链）。实现接入后直接追加到该表。
#: 每一项为 (模块, 模块内阶段名)
PLANNED_STAGES: list[tuple[str, str]] = [
    ("terrain", "plate_tectonics"),
    ("terrain", "noise_refine"),
    ("terrain", "erosion"),
    ("terrain", "hydrology"),
    ("radiation", "insolation"),
    ("radiation", "radiative_transfer"),
    ("surface", "energy_balance"),
    ("atmosphere", "dynamics"),
    ("ocean", "circulation"),
    ("hydrology_cycle", "precipitation"),
    ("classification", "koppen"),
    ("classification", "biome_soil_agro"),
    ("render", "map_2d"),
    ("render", "view_3d"),
    ("validation", "validate"),
]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[pipeline] {message}")


def create_world(config: RuntimeConfig) -> tuple[GridState, list[StageResult]]:
    """从头生成一个世界：初始化网格 → 依次执行已注册阶段。

    返回 (最终的 GridState, 各阶段结果)。阶段未实现时返回
    ``StageStatus.NOT_IMPLEMENTED`` 而不中断，便于先粗后细地逐步接入。
    """
    state = GridState.from_resolution(
        resolution=config.resolution,
        planet=config.planet,
        seed=config.seed,
        backend_name=config.backend,
        nz=config.nz,
    )
    _log(config.verbose, f"网格初始化完成 {state.shape} @ {state.resolution}\u00b0")

    results: list[StageResult] = []
    for stage in config.stages:
        result = _run_stage(stage, state, config)
        results.append(result)
        _log(config.verbose, f"  {result.status.value:<16} {result.name}")
        if result.status is StageStatus.FAILED:
            _log(config.verbose, f"    原因: {result.message}")
            break
    if not config.stages:
        for module, stage_name in PLANNED_STAGES:
            results.append(
                StageResult(name=stage_name, status=StageStatus.NOT_IMPLEMENTED, message=f"待实现: {module}/{stage_name}")
            )
    return state, results


def _run_stage(stage: PipelineStage, state: GridState, config: RuntimeConfig) -> StageResult:
    import time

    t0 = time.perf_counter()
    try:
        outputs_before = set(state.filled_fields())
        stage.run(state)
        new_fields = set(state.filled_fields()) - outputs_before
        return StageResult(
            name=stage.name,
            status=StageStatus.READY,
            elapsed_s=time.perf_counter() - t0,
            output_fields=sorted(new_fields),
        )
    except Exception as exc:  # noqa: BLE001 - 阶段失败须记录而非中断整个管线
        return StageResult(
            name=stage.name,
            status=StageStatus.FAILED,
            elapsed_s=time.perf_counter() - t0,
            message=f"{type(exc).__name__}: {exc}",
        )


def world_summary(state: GridState, results: list[StageResult]) -> str:
    """生成人可读的世界概览文本。"""
    summary = state.summary()
    lines = [
        f"世界 #{summary['seed']}  行星={summary['planet']}  "
        f"分辨率={summary['resolution']}° 后端={summary['backend']}",
        f"网格: {summary['shape'][0]}×{summary['shape'][1]}  面积={summary['total_area_m2']:.3e} m²",
        f"已填充字段 {len(state.filled_fields())} 个: {', '.join(state.filled_fields()) or '无'}",
        "阶段:",
    ]
    for r in results:
        extra = ""
        if r.status is StageStatus.READY:
            extra = f" ({r.elapsed_s:.2f}s) 产出[{', '.join(r.output_fields) or '无'}]"
        elif r.status is StageStatus.FAILED:
            extra = f" -> {r.message}"
        lines.append(f"  {r.status.value:<16} {r.name}{extra}")
    return "\n".join(lines)


__all__ = [
    "PLANNED_STAGES",
    "PipelineStage",
    "RuntimeConfig",
    "StageResult",
    "StageStatus",
    "create_world",
    "world_summary",
]