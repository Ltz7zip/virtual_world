"""数组后端：优先 MLX（Apple Silicon Metal GPU），回退 NumPy。

系统遵循 Apple Silicon 原生原则：核心计算的数据尽量保留在 MLX 数组中，
仅在磁盘 IO（Zarr）或与外部库（Matplotlib / PyVista）交互时转回 NumPy。
NumPy 后端可让核心层在非 Apple 平台与 CI 上运行。
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

#: MLX 模块（不可用时为 ``None``）；显式标注使导入失败分支可赋 None
mx: Any

try:  # pragma: no cover - 取决于平台
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - 非 Apple Silicon
    mx = None
    MLX_AVAILABLE = False

#: GPU 路径统一使用 Float32（见《精度与性能策略》§2.3）
DEFAULT_DTYPE = "float32"
#: 分类/掩码字段使用 int8 或 bool
CATEGORY_DTYPE = "int8"

#: NumPy 与 MLX 的 dtype 命名差异
_DTYPE_ALIASES = {"bool": "bool_"}


def mlx_available() -> bool:
    """当前环境是否可用 MLX 后端。"""
    return MLX_AVAILABLE


@cache
def resolve_backend(name: str | None = None) -> Any:
    """解析数组后端模块。

    ``name`` 为 ``None`` / ``"auto"`` 时优先 MLX，不可用则回退 NumPy。
    """
    if name in (None, "auto"):
        return mx if MLX_AVAILABLE else np
    if name == "mlx":
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX 后端不可用：mlx 仅在 Apple Silicon (macOS >= 13.5) 上可用")
        return mx
    if name == "numpy":
        return np
    raise ValueError(f"未知后端: {name!r}（可选 'mlx' / 'numpy' / 'auto'）")


def backend_name(xp: Any) -> str:
    """返回后端模块的名字。"""
    return "mlx" if MLX_AVAILABLE and xp is mx else "numpy"


def to_dtype(xp: Any, dtype: Any) -> Any:
    """把 NumPy 风格的 dtype（字符串或 dtype 对象）转换为指定后端的 dtype。"""
    name = np.dtype(dtype).name
    return getattr(xp, _DTYPE_ALIASES.get(name, name))


def is_array(value: Any) -> bool:
    """判断对象是否为受支持的数组类型。"""
    if isinstance(value, np.ndarray):
        return True
    return bool(MLX_AVAILABLE and isinstance(value, mx.array))


def backend_of(*arrays: Any) -> Any:
    """按输入数组推断后端：任一输入为 MLX 数组则使用 MLX，否则使用 NumPy。"""
    if MLX_AVAILABLE:
        for arr in arrays:
            if isinstance(arr, mx.array):
                return mx
    return np


def asarray(data: Any, backend: str | None = None, dtype: Any = DEFAULT_DTYPE) -> Any:
    """把输入转换为指定后端的数组，默认 Float32。"""
    xp = resolve_backend(backend)
    if dtype is None:
        return xp.array(data)
    return xp.array(data, dtype=to_dtype(xp, dtype))


def zeros(shape: tuple[int, ...], backend: str | None = None, dtype: Any = DEFAULT_DTYPE) -> Any:
    """创建指定后端的零数组。"""
    xp = resolve_backend(backend)
    return xp.zeros(shape, dtype=to_dtype(xp, dtype))


def to_numpy(data: Any) -> np.ndarray:
    """转为 NumPy 数组（用于磁盘 IO 与外部可视化库）。"""
    if MLX_AVAILABLE and isinstance(data, mx.array):
        return np.array(data)
    return np.asarray(data)


def to_backend(data: Any, backend: str | None = None) -> Any:
    """把 NumPy 数组（或可转换对象）搬到指定后端。"""
    xp = resolve_backend(backend)
    if xp is np:
        return np.asarray(data)
    if isinstance(data, mx.array):
        return data
    return mx.array(np.asarray(data))
