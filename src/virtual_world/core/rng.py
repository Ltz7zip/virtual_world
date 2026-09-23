"""确定性随机数生成。

整个系统由**单一随机种子**驱动：相同种子 + 相同行星参数 = 完全相同的结果。
所有随机数必须来自 :class:`DeterministicRNG`，禁止直接调用 ``numpy.random``
的全局接口（全局状态不可复现）。

每个模块通过 :meth:`DeterministicRNG.spawn` 派生自己的独立流：子流的种子为
``blake2b(f"{seed}:{name}")`` 的哈希值，与推导顺序无关（使用哈希而非
``hash()``，因为后者带进程随机盐，不可复现）。
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

#: 子流种子的位宽（blake2b 摘要长度，字节）
_SEED_BYTES = 8


def derive_seed(seed: int, name: str) -> int:
    """由父种子与逻辑名派生确定性子种子。"""
    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=_SEED_BYTES).digest()
    return int.from_bytes(digest, "big")


class DeterministicRNG:
    """可序列化与恢复的确定性随机数生成器。"""

    def __init__(self, seed: int, name: str = "root") -> None:
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed 必须是整数")
        self.seed = int(seed)
        self.name = name
        self.rng = np.random.default_rng(self.seed)

    def spawn(self, name: str) -> DeterministicRNG:
        """派生一个独立子流，用于某个模块或某个生成阶段。"""
        return DeterministicRNG(derive_seed(self.seed, name), name=f"{self.name}/{name}")

    def get_state(self) -> dict[str, Any]:
        """导出可序列化状态（种子 + 生成器内部状态）。"""
        return {
            "seed": self.seed,
            "name": self.name,
            "bit_generator": self.rng.bit_generator.state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """从 :meth:`get_state` 的输出恢复状态，保证结果可复现。"""
        self.seed = int(state["seed"])
        self.name = str(state.get("name", self.name))
        self.rng.bit_generator.state = state["bit_generator"]

    # ---- 常用分布的便捷接口 ----

    def random(self, size: int | tuple[int, ...] | None = None) -> np.ndarray:
        """[0, 1) 均匀分布。"""
        return self.rng.random(size)

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any = None) -> np.ndarray:
        """均匀分布。"""
        return self.rng.uniform(low, high, size)

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> np.ndarray:
        """正态分布。"""
        return self.rng.normal(loc, scale, size)

    def integers(self, low: Any, high: Any = None, size: Any = None) -> np.ndarray:
        """整数分布。"""
        return self.rng.integers(low, high, size)

    def choice(self, a: Any, size: Any = None, replace: bool = True, p: Any = None) -> np.ndarray:
        """离散抽样。"""
        return self.rng.choice(a, size=size, replace=replace, p=p)

    def __repr__(self) -> str:
        return f"DeterministicRNG(seed={self.seed}, name={self.name!r})"
