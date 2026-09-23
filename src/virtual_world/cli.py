"""命令行入口：``python -m virtual_world.cli``。

子命令（对应 `scripts/`）：
    presets            列出行星参数预设
    generate           按预设 + 分辨率 + 种子生成世界（执行已注册阶段）
    validate           校验一个网格状态（文件或仅初始化后校验）
    benchmark          对核心操作做简单基准，输出耗时表
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable

from .core.grid import GridState, get_resolution_level
from .core.planet import PlanetParams, list_presets
from .pipeline import RuntimeConfig, create_world, world_summary

__version__ = "0.1.0"


def _planet(args: argparse.Namespace) -> PlanetParams:
    return PlanetParams.from_preset(args.planet)


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--planet", default="default", help="行星参数预设名（list presets 查看）")
    parser.add_argument("-r", "--resolution", type=float, help="分辨率（度），如 2.0；与 --level 二选一")
    parser.add_argument("-l", "--level", default="medium", help="分级分辨率名：preview/coarse/medium/fine/high/ultra")
    parser.add_argument("-s", "--seed", type=int, default=0, help="世界种子（默认 0）")
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "numpy"], help="数组后端")
    parser.add_argument(
        "--grid",
        default="latlon",
        choices=["latlon", "cubed_sphere", "healpix"],
        help="网格类型（方案《地形生成混合方案》§1.5）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出概览")


def cmd_presets(args: argparse.Namespace) -> int:
    print("可用行星参数预设:")
    for name in list_presets():
        print(f"  - {name}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    config = RuntimeConfig(
        seed=args.seed, planet=_planet(args), backend=args.backend, grid_type=args.grid
    )
    if args.resolution is not None:
        config.resolution = args.resolution
    else:
        config.resolution = get_resolution_level(args.level).resolution
    t0 = time.perf_counter()
    state, results = create_world(config)
    text = world_summary(state, results)
    elapsed = time.perf_counter() - t0
    if args.json:
        payload = {
            "elapsed_s": round(elapsed, 4),
            **state.summary(),
            "stages": [
                {"name": r.name, "status": r.status.value, "elapsed_s": r.elapsed_s, "message": r.message}
                for r in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)
        print(f"总耗时 {elapsed:.2f}s")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    state = GridState.from_resolution(
        resolution=get_resolution_level(args.level).resolution,
        planet=PlanetParams.from_preset(args.planet),
        seed=args.seed,
        backend_name=args.backend,
        grid_type=args.grid,
    )
    problems = state.validate()
    if problems:
        print("校验未通过:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"网格校验通过：{state}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    import numpy as np

    planet = PlanetParams.from_preset(args.planet)
    rows: list[tuple[str, float]] = []

    def bench(name: str, fn: Callable[[], object], iters: int = 5) -> None:
        fn()  # 预热
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        rows.append((name, (time.perf_counter() - t0) / iters))

    grid = GridState.from_resolution(args.resolution or 2.0, planet=planet, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    grid.set("elevation", rng.normal(size=grid.shape))

    bench("网格初始化", lambda: GridState.from_resolution(grid.resolution, planet=planet))
    bench("面积加权平均", lambda: grid.global_mean("elevation"))
    bench("粗化 x3", lambda: grid.coarsen(3))
    bench("重采样 1°", lambda: grid.resample(180, 360))
    bench("序列化往返", lambda: GridState.from_dict(grid.to_dict()))

    print(f"{'操作':<18}{'耗时 (ms)':>12}")
    for name, ms in rows:
        print(f"{name:<18}{ms * 1000:>12.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="virtual_world", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("presets", help="列出行星参数预设")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("generate", help="生成世界")
    _add_generate_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="校验网格")
    _add_generate_args(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("benchmark", help="核心操作基准")
    p.add_argument("-p", "--planet", default="earth")
    p.add_argument("-r", "--resolution", type=float, default=2.0)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    sys.exit(main())