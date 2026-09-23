#!/usr/bin/env python3
"""单世界生成脚本。

等价于 ``python -m virtual_world.cli generate``，作为独立脚本入口保留
（对应 docs/architecture.md 中的 scripts 用途）：

    python scripts/generate_world.py -p earth -l preview -s 42
"""

from __future__ import annotations

import sys

from virtual_world.cli import main

if __name__ == "__main__":
    # 转发默认子命令 generate
    argv = ["generate", *sys.argv[1:]]
    sys.exit(main(argv))