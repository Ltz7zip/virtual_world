#!/usr/bin/env python3
"""世界校验与保真度脚本：网格校验 + 守恒/一致性诊断骨架。

完整八层验证体系见 `doc/方案/项目结构与技术栈/完善验证与保真度检验.md`；
各物理量阈值见 `configs/validation_thresholds.yaml`。

用法：
    python scripts/validate_world.py -p earth -r 2 -s 0
"""

from __future__ import annotations

import argparse
import sys

import yaml


def _load_thresholds() -> dict:
    from virtual_world.core.planet import config_dir

    path = config_dir() / "validation_thresholds.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验/保真度检查")
    parser.add_argument("-p", "--planet", default="earth")
    parser.add_argument("-r", "--resolution", type=float, default=2.0)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args(argv)

    from virtual_world.core.grid import GridState
    from virtual_world.core.planet import PlanetParams

    grid = GridState.from_resolution(
        args.resolution, planet=PlanetParams.from_preset(args.planet), seed=args.seed, backend_name=args.backend
    )
    problems = grid.validate()
    if problems:
        print("== 网格校验失败 ==")
        for p in problems:
            print("  -", p)
        return 1

    thresholds = _load_thresholds()
    print(f"== 网格校验通过：{grid} ==")
    print(f"验证阈值已加载：conservation = {thresholds['conservation']}")
    print("提示：守恒律/动力学/辐射/水文/生态一致性诊断将随实现阶段接入。")
    return 0


if __name__ == "__main__":
    sys.exit(main())