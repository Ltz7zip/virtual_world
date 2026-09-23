#!/usr/bin/env python3
"""批量生成：用 joblib 在多个进程并行生成多个世界（16 核加速，见
《精度与性能策略》§7.3）。每个进程一个独立子种子，结果写入 ``data/output/reports/``。

用法：
    python scripts/batch_generate.py -p earth -r 2 -s 0 --n-worlds 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _generate_one(seed: int, planet: str, resolution: float, backend: str) -> dict:
    from virtual_world.core import backend as xb
    from virtual_world.core.planet import PlanetParams
    from virtual_world.pipeline import RuntimeConfig, create_world, world_summary

    config = RuntimeConfig(
        seed=seed, planet=PlanetParams.from_preset(planet), resolution=resolution, backend=backend
    )
    state, results = create_world(config)
    return {
        "seed": seed,
        "planet": planet,
        "resolution": resolution,
        "backend": xb.backend_name(state.xp),
        "shape": list(state.shape),
        "summary": world_summary(state, results),
        "stages": [{"name": r.name, "status": r.status.value} for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量生成多个世界")
    parser.add_argument("-p", "--planet", default="earth")
    parser.add_argument("-r", "--resolution", type=float, default=2.0)
    parser.add_argument("-s", "--seed0", type=int, default=0, help="起始种子，世界种子依次 +1")
    parser.add_argument("-n", "--n-worlds", type=int, default=4)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--n-jobs", type=int, default=4, help="并行进程数")
    parser.add_argument("-o", "--output", default="data/output/reports/batch_worlds.json")
    args = parser.parse_args(argv)

    from joblib import Parallel, delayed

    seeds = [args.seed0 + i for i in range(args.n_worlds)]
    worlds = Parallel(n_jobs=args.n_jobs)(
        delayed(_generate_one)(seed, args.planet, args.resolution, args.backend) for seed in seeds
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(worlds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(worlds)} 个世界，报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())