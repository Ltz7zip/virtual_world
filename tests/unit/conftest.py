"""测试公共配置：可用数组后端列表。"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_world.core import backend

#: 可用后端：NumPy 始终可用，MLX 仅在 Apple Silicon 环境可用
BACKENDS: list[str] = ["numpy"] + (["mlx"] if backend.mlx_available() else [])


@pytest.fixture
def rng() -> np.random.Generator:
    """固定种子的 NumPy 生成器，用于构造确定性的测试数据。"""
    return np.random.default_rng(20240923)


@pytest.fixture(params=BACKENDS, ids=str)
def backend_name(request: pytest.FixtureRequest) -> str:
    """在全部可用后端上参数化测试（NumPy，以及 Apple Silicon 上的 MLX）。"""
    return str(request.param)
