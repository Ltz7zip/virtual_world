"""端到端集成测试：地形三层接入生产管线（《地形生成混合方案与程序加速策略》）。

覆盖 P0 缺口：
- 三层（板块构造 → 噪声/扩散精修 → 侵蚀）被 :class:`TerrainStage` 串成一条链；
- :func:`pipeline.create_world` 默认执行该阶段，产出写入 ``GridState`` 的 terrain 字段；
- 三种网格（经纬 / 立方球 / HEALPix）均能端到端跑通。

分辨率取很粗（10°）以保持测试快速；物理正确性由各层单元测试保证，这里只验证串联。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from virtual_world.core.grid import GridState
from virtual_world.core.planet import PlanetParams
from virtual_world.pipeline import (
    PLANNED_STAGES,
    RuntimeConfig,
    StageStatus,
    create_world,
    default_stages,
)
from virtual_world.terrain.stage import OUTPUT_FIELDS, TerrainStage

#: 粗分辨率（10° → 18x36 网格），并在每层用较小工作量
RESOLUTION = 10.0


def _stage() -> TerrainStage:
    """小工作量的地形阶段，用于快速集成验证。"""
    return TerrainStage(n_major=6, hydraulic_steps=40)


def _state(grid_type: str = "latlon") -> GridState:
    return GridState.from_resolution(
        resolution=RESOLUTION,
        planet=PlanetParams.from_preset("earth"),
        seed=42,
        grid_type=grid_type,
    )


# ===== 阶段契约 =====


def test_default_stages_cover_the_terrain_planned_stages():
    """默认阶段必须声明覆盖 PLANNED_STAGES 中全部 terrain 条目。"""
    terrain_keys = {key for key in PLANNED_STAGES if key[0] == "terrain"}
    covered = {key for stage in default_stages() for key in stage.covers}
    assert terrain_keys, "PLANNED_STAGES 应包含 terrain 条目"
    assert terrain_keys <= covered


def test_default_stages_satisfy_pipeline_stage_protocol():
    """TerrainStage 必须可被 create_world 当作 PipelineStage 使用。"""
    for stage in default_stages():
        assert isinstance(stage.name, str) and stage.name
        assert callable(stage.run)


# ===== 经纬网格：三层串联与产出 =====


def test_terrain_stage_fills_terrain_fields():
    """run() 必须填充 elevation / is_ocean / ocean_depth 三个 terrain 字段。"""
    state = _state()
    returned = _stage().run(state)

    assert returned is state, "run 应原地写入并返回同一 GridState"
    for name in OUTPUT_FIELDS:
        assert state.has(name), f"字段 {name} 未填充"
        assert state.as_numpy(name).shape == state.shape


def test_terrain_stage_outputs_are_valid_and_finite():
    """产出必须有限、落在字段量程内，且海陆掩码与高程符号一致。"""
    state = _state()
    _stage().run(state)

    elevation = state.as_numpy("elevation")
    is_ocean = state.as_numpy("is_ocean")
    ocean_depth = state.as_numpy("ocean_depth")

    assert np.all(np.isfinite(elevation))
    lo, hi = state.spec("elevation").valid_range
    assert elevation.min() >= lo and elevation.max() <= hi
    assert np.all(ocean_depth >= 0.0)

    # 方案 §1.1：第二、三层不改变第一层的宏观海陆格局 —— 掩码与符号必须一致
    assert np.all(elevation[is_ocean] < 0.0)
    assert np.all(elevation[~is_ocean] > 0.0)
    assert np.all(ocean_depth[~is_ocean] == 0.0)


def test_terrain_stage_both_land_and_ocean_exist():
    """海陆比可配置，但不应退化为全陆或全海。"""
    state = _state()
    _stage().run(state)
    ocean_fraction = float(np.mean(state.as_numpy("is_ocean")))
    assert 0.05 < ocean_fraction < 0.95


def test_terrain_stage_is_deterministic():
    """同一种子 + 同一参数 ⇒ 逐位相同（方案允诺的种子一致性）。"""
    first = _state()
    second = _state()
    _stage().run(first)
    _stage().run(second)
    np.testing.assert_array_equal(first.as_numpy("elevation"), second.as_numpy("elevation"))


def test_terrain_stage_layer3_actually_sculpts_terrain():
    """第三层必须真正改变地形（守住"侵蚀在串联中被静默跳过"这类退化）。"""
    stage = _stage()
    sculpted = _state()
    intact = _state()
    stage.run(sculpted)
    dataclasses.replace(stage, erode=False).run(intact)

    drop = intact.as_numpy("elevation") - sculpted.as_numpy("elevation")
    assert np.any(drop > 0.0), "侵蚀未降低任何单元的高程"
    assert drop.max() > 0.0


def test_terrain_stage_preserves_layer1_macro_layout():
    """方案 §1.1 跨层约束：第二、三层只加细节，海陆格局与第一层逐格一致。"""
    from virtual_world.terrain import plate_tectonics as pt

    stage = _stage()
    state = _state()
    stage.run(state)

    nlat, nlon = stage.work_shape(state)
    tectonic = pt.generate_tectonic_field(
        nlat=nlat,
        nlon=nlon,
        n_seeds=stage.n_seeds,
        n_major=stage.n_major,
        seed=state.seed,
        time_ma=stage.time_ma,
        dt_ma=stage.dt_ma,
        ocean_fraction=stage.ocean_fraction,
        subduction=stage.subduction,
        radius=state.planet.radius,
    )

    # 海陆划分由第一层决定，精修与侵蚀都不得移动它
    np.testing.assert_array_equal(state.as_numpy("is_ocean"), tectonic.is_ocean)
    # 但高程必须已经不再是纯构造场（第二层确实注入了残差）
    assert not np.allclose(state.as_numpy("elevation"), tectonic.elevation)


# ===== 三种网格对等（方案 §1.5）=====


@pytest.mark.parametrize("grid_type", ["latlon", "cubed_sphere", "healpix"])
def test_terrain_stage_runs_on_all_grid_types(grid_type: str):
    """非经纬网格先走经纬工作网格、末尾重网格回本形状，字段形状须与本网格一致。"""
    state = _state(grid_type)
    _stage().run(state)

    for name in OUTPUT_FIELDS:
        assert state.has(name)
        assert state.as_numpy(name).shape == state.shape
    assert np.all(np.isfinite(state.as_numpy("elevation")))


# ===== 经 create_world 的生产路径 =====


def test_create_world_runs_explicitly_registered_terrain_stage():
    """显式注册阶段时，create_world 执行该阶段并把产出写入 state。"""
    config = RuntimeConfig(
        seed=7,
        planet=PlanetParams.from_preset("earth"),
        resolution=RESOLUTION,
        verbose=False,
        stages=[_stage()],
    )
    state, results = create_world(config)

    assert [r.name for r in results] == ["terrain"]
    assert results[0].status is StageStatus.READY
    assert state.has("elevation")


def test_create_world_without_explicit_stages_uses_defaults():
    """未显式注册阶段时，地形三层应被默认执行而不再报 not_implemented。"""
    config = RuntimeConfig(
        seed=3,
        planet=PlanetParams.from_preset("earth"),
        resolution=RESOLUTION,
        verbose=False,
    )
    state, results = create_world(config)

    ready = {r.name for r in results if r.status is StageStatus.READY}
    not_implemented = {r.name for r in results if r.status is StageStatus.NOT_IMPLEMENTED}
    assert "terrain" in ready, f"地形阶段未执行: {results}"
    assert state.has("elevation")
    # 地形覆盖的规划条目不应再出现在未实现清单中
    for _, stage_name in TerrainStage.covers:
        assert stage_name not in not_implemented
    # 后续物理阶段仍待实现（辐射起）
    assert "insolation" in not_implemented
