# Virtual World — Python 虚拟 3D 世界生成器

物理自洽的虚拟行星生成管线：**地形 → 气候 → 地表系统**，所有阶段共享同一套
经纬度网格数据模型。核心设计见 `doc/方案/`。

## 特性

- **数据驱动分层生成**：行星参数 → 地形 → 辐射 → 大气 → 海洋 → 降水 → 气候分类 → 可视化，逐层因果链
- **种子确定性**：相同种子 + 相同参数 = 完全相同的结果（`DeterministicRNG`，blake2b 派生子流）
- **分级分辨率**：10°（预览）到 0.25°（局地），先粗后细
- **Apple Silicon 原生**：MLX（Metal GPU）核心求解器 + Numba（CPU 热点）+ Zarr 缓存 + VTK Metal 渲染
- **分层验证**：守恒律 → 动力学一致性 → 基准测试 → 地球对比（阈值见 `configs/validation_thresholds.yaml`）

## 安装

Python ≥ 3.11（Apple Silicon 上 MLX 自动启用；无 MLX 时回退 NumPy）。

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

GDAL/KoppenClimate/PyAEZ 等可选 science 依赖的安装方式见 `docs/architecture.md` 或
`doc/方案/项目结构与技术栈/`。

## 快速开始

```python
from virtual_world import GridState, PlanetParams

planet = PlanetParams.from_preset("earth")
grid = GridState.from_resolution(2.0, planet=planet, seed=42)
print(grid.summary())
```

或使用命令行入口：

```bash
python -m virtual_world.cli presets            # 列出行星参数预设
python -m virtual_world.cli generate -p earth -r 2 -s 42   # 生成（按已实现阶段）
python scripts/validate_world.py               # 校验与验证
```

## 项目结构

```
virtual_world/
├── configs/            # 行星参数 / 分辨率 / 验证阈值 YAML
├── src/virtual_world/  # 源码包（core 数据模型 + 8 个物理阶段 + 加速 / IO / render / validation）
├── scripts/            # 生成 / 批量生成 / 验证 / 基准 脚本
├── tests/              # unit / integration / validation / benchmarks
├── notebooks/          # Jupyter 探索
├── data/               # input / cache / output
├── vendor/             # 第三方源码依赖（PyAEZ、gdal 兼容层）
└── docs/               # 架构 / 物理 / 加速 / 验证 文档
```

完整目录树与每个模块的文件规划见
`doc/方案/项目结构与技术栈/项目结构与技术栈.md`。

## 测试

```bash
.venv/bin/python -m pytest
```

## License

[MIT](LICENSE)