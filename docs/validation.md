# 验证文档

完整八层验证体系与阈值见
`doc/方案/项目结构与技术栈/完善验证与保真度检验.md`；阈值配置在
`configs/validation_thresholds.yaml`。

## 八层验证

| 层 | 内容 | 阈值（可接受偏差） |
|----|------|--------------------|
| 1 守恒律 | 能量 / 质量 / 角动量 / 位涡 | 能量 < 0.1%/天，质量 < 10⁻⁶/天 |
| 2 动力学 | 地转 / 热成风 / Rossby / 动能谱 | AGD < 20% |
| 3 辐射 | `S(1-α)/4 = OLR`、辐射-对流平衡 | < 1% |
| 4 水文 | `∫E = ∫P`、`P = E + R` | < 1% |
| 5 海洋 | Sverdrup / Ekman / 西边界流 | 符合度 > 70% |
| 6 生态 | Köppen / 生物群系 / 土壤 / 农业一致性 | 分类精度 > 70% |
| 7 基准 | Held-Suarez / Aquaplanet / MITC | 与参考解偏差 < 10% |
| 8 地球对比 | 全球均温 14–15°C、降水 1000 mm/年 等 | 逐项见配置 |

## 已实现

- `GridState.validate()`：字段形状 / 有限性 / 取值范围校验
- `scripts/validate_world.py`：网格 + 阈值配置加载
- 守恒律积分与动力学一致性诊断函数规划于 `validation/` 模块

## 运行

```bash
python scripts/validate_world.py -p earth -r 2 -s 0
python -m pytest tests/unit tests/integration tests/validation
```