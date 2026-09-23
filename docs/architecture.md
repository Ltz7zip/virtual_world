# 架构文档

本文件是项目结构导读；完整的设计方案（物理模型、数学形式、目录树）见
`doc/方案/项目结构与技术栈/项目结构与技术栈.md`。

## 分层架构

```
configs/ (YAML 参数)
   ↓
src/virtual_world/
├── core/              数据模型层（网格 / 行星参数 / 球面几何 / 插值 / RNG / 常数 / 单位）
├── terrain/           阶段一 地形生成（板块构造 → 噪声精修 → 侵蚀 → 水文）
├── radiation/         阶段二 辐射强迫与辐射传输
├── surface/           阶段三 地表能量平衡
├── atmosphere/        阶段四 大气动力学
├── ocean/             阶段四 海洋环流
├── hydrology_cycle/   阶段五 水循环与降水
├── classification/    阶段六 气候分类（Köppen / 生物群系 / 土壤 / 农业）
├── render/            阶段七 2D / 3D 可视化
├── validation/        阶段八 验证与保真度检验
├── acceleration/      MLX / Numba / 查找表 / 缓存
├── io/                Zarr / NetCDF / GeoJSON
├── pipeline.py        主管线编排（物理因果链）
└── cli.py             命令行入口
```

## 核心约定

- **数据流**：各阶段通过 `GridState` 传递数据（类型安全、可序列化、可切片），不直接用松散 dict。
- **种子确定性**：唯一 `DeterministicRNG`，blake2b 派生子流，状态可序列化恢复。
- **后端透明**：`core.backend` 统一 MLX/NumPy，默认 Float32。
- **分级分辨率**：`configs/resolution_levels.yaml` 定义 10°~0.25° 六级。

## 球面网格（方案《地形生成混合方案》§1.5）

三种网格在 `GridState` 中**对等可用**，字段水平形状由 `grid_type` 决定：

| `grid_type` | 水平形状 | 几何模块 | 单元面积比 | 特点 |
|-------------|----------|----------|-----------|------|
| `latlon` | `(nlat, nlon)` | `core.spherical` | ∞（高纬收缩） | 解析度规、最简单的 IO 与出图 |
| `cubed_sphere` | `(6, n_side, n_side)` | `core.cubed_sphere` | ≈1.5 | 无极点奇点，推荐用于地形生成 |
| `healpix` | `(npix,)` | `core.healpix` | 1（严格等面积） | 层次化嵌套，天然多分辨率 |

统一接口由 `core.geometry`（`GridGeometry`）提供，因此下游阶段只需换网格类型，不必改代码：

- **单元面积与统计**：`cell_area` / `total_area` / `global_mean` / `global_integral` / `zonal_mean`（面积加权）
- **微分算子**：`gradient` / `divergence` / `vorticity` / `laplacian`
  （等经纬度用解析度规中心差分；立方球与 HEALPix 用 `core.operators` 的切平面最小二乘重建 +
  平行移动，后者同时提供局地线性插值）
- **多分辨率**：`coarsen`（限制算子，保守，全球加权平均严格不变）
- **跨网格重网格**：`GridState.regrid(grid_type, ...)`，经纬网格作为桥梁，
  分类/布尔场自动取最近邻
- **拓扑**：立方球 `edge_neighbors/all_neighbors/shift`；HEALPix `edge_neighbors/all_neighbors`
  （共角计数判定，含极点像素少一个对角邻居的处理）

命令行可用 `--grid {latlon,cubed_sphere,healpix}` 选择网格；`healpix` 的 `nside` 必须是
2 的幂（层次化与 LOD 的前提）。

## 运行

```bash
python -m virtual_world.cli generate -p earth -r 2 -s 42
python -m virtual_world.cli generate -p earth -r 5 --grid healpix
python -m virtual_world.cli generate -p earth -r 5 --grid cubed_sphere
python scripts/batch_generate.py -n 4
python scripts/validate_world.py
python scripts/benchmark.py
```

## 说明

- 阶段模块 `terrain/ ... validation/` 内的实现文件随实施路线图逐阶段创建；
  未实现阶段在 `pipeline.PLANNED_STAGES` 中登记，生成时标记 `not_implemented`。
- 第三方源码依赖（PyAEZ、gdal 兼容层）位于 `vendor/`，安装方式见 README。