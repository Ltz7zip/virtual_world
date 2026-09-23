# 加速文档

完整策略见 `doc/方案/精度与性能策略/精度与性能策略.md`。

## 加速金字塔

| 层次 | 工具 | 适用 |
|------|------|------|
| 算法 | NumPy 矢量化 / 谱方法 + FFT / 多重网格 / 查找表 | 所有模块 |
| JIT | Numba `@njit(parallel=True, fastmath=True)` | 地形、侵蚀、降水、流线 |
| GPU | MLX（Metal），默认 Float32 | 大气、海洋、辐射、分类 |
| 并行 | joblib 多进程（16 核多世界） | 批量生成 |
| IO | Zarr 分块 + 缓存 | 静态场（地形 / 辐射 / 气候） |

## 路线图

1. NumPy 矢量化消灭 Python 循环（基础 10–100×）
2. Numba JIT 编译地形与侵蚀热点（额外 10–20×）
3. MLX GPU 加速大气与海洋（50–100×）
4. Zarr 缓存减少重复计算
5. joblib 参数扫描（线性加速）
6. mpi4py 域分解（仅网格 > 10⁶ 单元）

## 已验证

```bash
python scripts/benchmark.py          # 核心数据模型操作耗时
scalene --gpu --memory --profile-all scripts/generate_world.py
```

## Apple Silicon 要点

- 统一内存：MLX 与 NumPy 互转开销极小，全程尽量保持 MLX 数组
- GPU 路径统一 Float32（M4 吞吐远高于 Float64）
- 3D 渲染：VTK 9.7 Metal 后端 + PyVista（已安装验证）