"""顶层 ``import gdal`` 兼容层。

自 GDAL 3.10+ 起，官方 Python 绑定移除了顶层 ``gdal`` 模块（须用
``from osgeo import gdal``）。PyAEZ 等老代码仍写 ``import gdal``，因此本
模块在导入时直接把模块对象替换为 ``osgeo.gdal``，后续所有引用均为真实 GDAL。

安装顺序：先 ``brew install gdal`` 再 ``pip install GDAL``，配好本兼容层，
``import gdal`` 即获得完整读写能力（``gdal.Open``、``GetDriverByName`` 等）。
"""

from __future__ import annotations

import sys

import osgeo.gdal as _real_gdal

# 用真实 GDAL 模块替换本模块（sys.modules['gdal'] 指向 osgeo.gdal）
sys.modules[__name__] = _real_gdal  # type: ignore[assignment]