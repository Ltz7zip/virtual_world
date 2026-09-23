"""§6.4 JAX 长时间积分内核（可选依赖）。

方案 §6.4：当模拟时间尺度达到数百万年、需要数千个时间步时，JAX 的 ``jit`` +
``lax.scan`` 会把整个时间循环编译成 XLA 计算图，消除 Python 循环开销，并在 GPU
上获得极高吞吐（对比见《第一层完善》§6.5 加速分工表）。

本模块提供与 Numba 内核 :func:`virtual_world.terrain.isostasy.integrate_crust_thickness`
**数值一致**的实现，采用同一分裂格式（先增厚/减薄、再用更新后的厚度做重力松弛）。

JAX 是可选依赖：本模块顶层不导入 jax，缺失时 :func:`integrate_crust_thickness_jax`
抛出含安装指引的 :class:`RuntimeError`，而不是静默退化为 Numba——静默降级会让调用方
以为拿到了 GPU 加速的结果。

精度：JAX 默认关闭 64 位浮点（float32），与本项目地形层的 float64 约定不符，因此
:func:`integrate_crust_thickness_jax` 会先调用 :func:`enable_x64` 打开 x64
（该开关幂等，是 JAX 的既定用法）。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def jax_available() -> bool:
    """JAX 是否可用（供调用方选择内核，避免依赖异常做流程控制）。"""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


def _require_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - 取决于可选依赖
        raise RuntimeError(
            "JAX 内核需要安装 jax（pip install jax）；"
            "也可改用 Numba 内核 isostasy.integrate_crust_thickness"
        ) from exc
    return jax, jnp


def enable_x64() -> bool:
    """打开 JAX 的 64 位浮点（幂等）；JAX 不可用时返回 ``False``。"""
    if not jax_available():
        return False
    import jax

    jax.config.update("jax_enable_x64", True)
    return True


def integrate_crust_thickness_jax(
    crust: np.ndarray,
    thickening: np.ndarray | float,
    thinning: np.ndarray | float,
    c_eq: np.ndarray,
    *,
    kappa: float = 0.0,
    dt: float = 1.0,
    n_steps: int = 1,
) -> np.ndarray:
    """地壳厚度显式时间积分的 JAX 实现（方案 §6.4），返回 NumPy 数组。

    参数与 :func:`virtual_world.terrain.isostasy.integrate_crust_thickness` 一致，
    数值结果相同；差别只在于时间循环由 ``jax.lax.scan`` 编译为 XLA 计算图
    （可放到 GPU 上跑数千步）。

    分裂格式（与 Numba 内核一致）：``C += (T - S)*dt`` 之后再
    ``C -= kappa*(C - C_eq)*dt``，因此该格式的定点为 ``C_eq + T*(1-kappa*dt)/kappa``
    （松弛作用在增厚之后的厚度上），而非 ``C_eq + T/kappa``。
    """
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，实际为 {dt}")
    if n_steps <= 0:
        raise ValueError(f"n_steps 必须为正，实际为 {n_steps}")

    jax, jnp = _require_jax()
    enable_x64()
    base = np.asarray(crust, dtype=np.float64)
    if base.ndim < 1:
        raise ValueError(f"crust 至少需要 1 维，实际 {base.shape}")
    thick = jnp.asarray(np.broadcast_to(np.asarray(thickening, dtype=np.float64), base.shape))
    thin = jnp.asarray(np.broadcast_to(np.asarray(thinning, dtype=np.float64), base.shape))
    equilibrium = jnp.asarray(np.broadcast_to(np.asarray(c_eq, dtype=np.float64), base.shape))
    state = jnp.asarray(base)
    step_dt = float(dt)
    relax = float(kappa)

    def step(current: Any, _: Any) -> tuple[Any, None]:
        current = current + (thick - thin) * step_dt
        current = current - relax * (current - equilibrium) * step_dt
        return current, None

    result, _ = jax.lax.scan(step, state, None, length=int(n_steps))
    return np.asarray(jax.block_until_ready(result), dtype=np.float64)


__all__ = ["enable_x64", "integrate_crust_thickness_jax", "jax_available"]

