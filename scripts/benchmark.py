#!/usr/bin/env python3
"""性能基准脚本：核心数据模型操作的耗时基准（Numba/MLX 阶段基准稍后接入）。

等价于 ``python -m virtual_world.cli benchmark``。

用法：
    python scripts/benchmark.py -p earth -r 2.0
"""

from __future__ import annotations

import sys
import time

import numpy as np

from virtual_world.core.grid import GridState
from virtual_world.core.planet import PlanetParams


def bench_one(name: str, fn, iters: int = 5) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="性能基准")
    parser.add_argument("-p", "--planet", default="earth")
    parser.add_argument("-r", "--resolution", type=float, default=2.0)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args(argv)

    planet = PlanetParams.from_preset(args.planet)
    grid = GridState.from_resolution(args.resolution, planet=planet, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    grid.set("elevation", rng.normal(size=grid.shape))

    cases = {
        "网格初始化": lambda: GridState.from_resolution(grid.resolution, planet=planet),
        "面积加权平均": lambda: grid.global_mean("elevation"),
        "纬向平均": lambda: grid.zonal_mean("elevation"),
        "粗化 x3": lambda: grid.coarsen(3),
        "重采样 1°": lambda: grid.resample(180, 360),
        "序列化往返": lambda: GridState.from_dict(grid.to_dict()),
    }
    print(f'{grid} | {"操作":<18}{"耗时 (ms)":>12}')
    for name, fn in cases.items():
        ms = bench_one(name, fn) * 1000
        print(f'{"":<24}{name:<18}{ms:>12.3f}')
    return 0


if __name__ == "__main__":
    sys.exit(main())