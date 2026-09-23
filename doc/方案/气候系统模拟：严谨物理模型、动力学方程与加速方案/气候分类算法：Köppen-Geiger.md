# 气候分类算法：Köppen-Geiger

Köppen-Geiger分类是气候系统模拟的**最终诊断层**，将连续的温度场和降水场映射为离散的、具有生态意义的气候类型。这一模块不涉及时间积分，但分类逻辑的精确性和计算效率直接影响最终输出质量。


## 一、分类体系的数学定义

### 1.1 分类变量的定义

Köppen-Geiger分类基于**月平均气温**和**月降水量**的长期统计量。所有分类变量的严格定义如下：

| 变量 | 符号 | 定义 | 单位 |
|------|------|------|------|
| 年均温 | MAT | 12个月平均气温的均值 | °C |
| 最冷月均温 | T_cold | 12个月中最低的月均温 | °C |
| 最暖月均温 | T_hot | 12个月中最高的月均温 | °C |
| 最热月数 | T_mon10 | 月均温 > 10°C 的月份数 | 月 |
| 年降水量 | MAP | 12个月降水量之和 | mm |
| 最干月降水 | P_dry | 12个月中最低的月降水量 | mm |
| 夏季最干月降水 | P_sdry | 夏季月份中最低的月降水量 | mm |
| 冬季最干月降水 | P_wdry | 冬季月份中最低的月降水量 | mm |
| 夏季最湿月降水 | P_swet | 夏季月份中最高的月降水量 | mm |
| 冬季最湿月降水 | P_wwet | 冬季月份中最高的月降水量 | mm |

**夏季/冬季的划分**：夏季定义为 4–9 月（北半球）或 10–3 月（南半球）中较暖的半年，冬季为较冷的半年。


## 二、分类逻辑的完整数学表达

Köppen-Geiger分类采用**三级层次结构**：第一级（主要气候带）、第二级（季节性）、第三级（温度或干旱程度细分）。

### 2.1 干旱阈值（B类判定）

在判定任何非干旱气候之前，首先计算**干旱阈值** \( P_{\text{threshold}} \)：

\[
P_{\text{threshold}} = \begin{cases}
2 \times \text{MAT} & \text{如果 } >70\% \text{ 的降水在冬季} \\
2 \times \text{MAT} + 28 & \text{如果 } >70\% \text{ 的降水在夏季} \\
2 \times \text{MAT} + 14 & \text{其他情况（降水均匀分布）}
\end{cases}
\]

**B类（干旱气候）判定**：如果 \( \text{MAP} < 10 \times P_{\text{threshold}} \)，则属于 B 类。

**注意**：MAT 的单位为 °C，MAP 和 \( P_{\text{threshold}} \) 的单位为 **mm**（原始 Köppen 定义中为 cm，此处已转换为 mm）。

### 2.2 第一级：五大气候带

**A 类（热带气候）** ：非 B 类，且 \( T_{\text{cold}} \geq 18°\text{C} \)。

**B 类（干旱气候）** ：\( \text{MAP} < 10 \times P_{\text{threshold}} \)。

**C 类（温带气候）** ：非 B 类，\( T_{\text{hot}} > 10°\text{C} \)，且 \( -3°\text{C} < T_{\text{cold}} < 18°\text{C} \)。

**D 类（大陆性气候）** ：非 B 类，\( T_{\text{hot}} > 10°\text{C} \)，且 \( T_{\text{cold}} \leq -3°\text{C} \)。

**E 类（极地气候）** ：\( T_{\text{hot}} < 10°\text{C} \)。

### 2.3 第二级和第三级细分

**A 类细分**（基于降水季节性）：

- **Af（热带雨林）** ：\( P_{\text{dry}} \geq 60 \) mm
- **Am（热带季风）** ：非 Af，且 \( P_{\text{dry}} \geq 100 - \text{MAP}/25 \)
- **Aw（热带草原）** ：非 Af，且 \( P_{\text{dry}} < 100 - \text{MAP}/25 \)

**B 类细分**（基于干旱程度和温度）：

- **BW（沙漠）** ：\( \text{MAP} < 5 \times P_{\text{threshold}} \)
- **BS（半干旱/草原）** ：\( \text{MAP} \geq 5 \times P_{\text{threshold}} \)
- **h（炎热）** ：\( \text{MAT} \geq 18°\text{C} \)
- **k（寒冷）** ：\( \text{MAT} < 18°\text{C} \)

**C 类细分**（基于降水季节性和夏季温度）：

- **Cw（冬季干燥）** ：\( P_{\text{wdry}} < P_{\text{swet}}/10 \)
- **Cs（夏季干燥，地中海气候）** ：非 Cw，且 \( P_{\text{sdry}} < 40 \) mm 且 \( P_{\text{sdry}} < P_{\text{wwet}}/3 \)
- **Cf（无干季）** ：非 Cs 且非 Cw
- **a（炎热夏季）** ：\( T_{\text{hot}} \geq 22°\text{C} \)
- **b（温暖夏季）** ：非 a，且 \( T_{\text{mon10}} \geq 4 \)
- **c（寒冷夏季）** ：非 a 且非 b

**D 类细分**（与 C 类类似，但温度阈值不同）：

- **Dw、Ds、Df**：与 Cw、Cs、Cf 相同的降水季节性判定
- **a、b、c、d**：基于夏季温度和冬季严寒程度细分

**E 类细分**：

- **ET（苔原）** ：\( 0°\text{C} < T_{\text{hot}} < 10°\text{C} \)
- **EF（冰盖）** ：\( T_{\text{hot}} \leq 0°\text{C} \)


## 三、物理基础与地理意义

### 3.1 分类的物理依据

Köppen-Geiger分类虽然是**经验性**的，但其阈值背后有明确的物理和生态依据：

- **\( T_{\text{cold}} = 18°\text{C} \)**（A/C 边界）：对应热带雨林的温度下限，低于此温度热带作物无法生存。
- **\( T_{\text{cold}} = -3°\text{C} \)**（C/D 边界）：对应土壤冻结的持久性阈值，低于此温度冬季积雪持续。
- **\( T_{\text{hot}} = 10°\text{C} \)**（E 类边界）：对应树木生长的温度下限（最暖月均温低于 10°C 无法支持森林）。
- **干旱阈值 \( 2 \times \text{MAT} \)**：基于**Thornthwaite 可能蒸散量**的简化。干旱气候的降水量不足以补偿蒸发，这是植被从森林向草原/沙漠转变的物理临界点。

### 3.2 地理分布约束

Köppen-Geiger分类的地理分布受**纬度、海陆位置、地形和洋流**的联合控制：

- **A 类**集中在赤道附近（\( |\phi| < 15° \)），受 ITCZ 上升气流控制。
- **B 类**集中在副热带高压带（\( 20°-35° \)）和大陆内部，受下沉气流和远离水汽源控制。
- **C 类**集中在中纬度西岸（地中海气候）和中纬度东岸（湿润副热带）。
- **D 类**集中在北半球中高纬度大陆内部（南半球缺乏此纬度带的陆地）。
- **E 类**集中在极地和高山，受极低太阳辐射控制。


## 四、算法实现

### 4.1 矢量化的分类逻辑

Köppen-Geiger分类本质上是**逐网格的规则判定**，所有判定条件都是对月温度和月降水数组的阈值操作，天然适合**向量化实现**。核心策略是将三级判定展开为**嵌套的 `np.where`**，一次性处理所有网格点：

```python
def koppen_geiger(T_monthly, P_monthly, lat):
    """
    T_monthly: (12, nlat, nlon) 月均温 (°C)
    P_monthly: (12, nlat, nlon) 月降水 (mm)
    lat: (nlat,) 纬度
    返回: (nlat, nlon) 整数分类代码
    """
    # 统计量（沿月份轴运算）
    MAT = T_monthly.mean(axis=0)
    T_cold = T_monthly.min(axis=0)
    T_hot = T_monthly.max(axis=0)
    T_mon10 = (T_monthly > 10).sum(axis=0)
    MAP = P_monthly.sum(axis=0)
    P_dry = P_monthly.min(axis=0)

    # 夏/冬月份索引（北半球：4-9月为夏）
    summer_idx = np.arange(4, 10) if lat[0] > 0 else np.arange(0, 3).tolist() + np.arange(9, 12).tolist()
    winter_idx = np.setdiff1d(np.arange(12), summer_idx)

    P_sdry = P_monthly[summer_idx].min(axis=0)
    P_wdry = P_monthly[winter_idx].min(axis=0)
    P_swet = P_monthly[summer_idx].max(axis=0)
    P_wwet = P_monthly[winter_idx].max(axis=0)

    # 干旱阈值
    winter_frac = P_monthly[winter_idx].sum(axis=0) / MAP
    summer_frac = P_monthly[summer_idx].sum(axis=0) / MAP
    P_threshold = np.where(winter_frac > 0.7, 2 * MAT,
                  np.where(summer_frac > 0.7, 2 * MAT + 28, 2 * MAT + 14))

    # 第一级判定（向量化）
    is_B = MAP < 10 * P_threshold
    is_A = (~is_B) & (T_cold >= 18)
    is_E = (~is_B) & (T_hot < 10)
    is_C = (~is_B) & (~is_A) & (~is_E) & (T_cold > -3) & (T_cold < 18)
    is_D = (~is_B) & (~is_A) & (~is_E) & (~is_C)

    # 使用整数编码（11-30）
    code = np.zeros_like(MAT, dtype=np.int8)

    # A 类细分
    is_Af = is_A & (P_dry >= 60)
    is_Am = is_A & (~is_Af) & (P_dry >= 100 - MAP / 25)
    is_Aw = is_A & (~is_Af) & (~is_Am)
    code = np.where(is_Af, 11, code)
    code = np.where(is_Am, 12, code)
    code = np.where(is_Aw, 13, code)

    # B 类细分
    is_BW = is_B & (MAP < 5 * P_threshold)
    is_BS = is_B & (MAP >= 5 * P_threshold)
    is_h = is_B & (MAT >= 18)
    is_k = is_B & (MAT < 18)
    code = np.where(is_BW & is_h, 21, code)  # BWh
    code = np.where(is_BW & is_k, 22, code)  # BWk
    code = np.where(is_BS & is_h, 23, code)  # BSh
    code = np.where(is_BS & is_k, 24, code)  # BSk

    # C 类细分
    is_Cw = is_C & (P_wdry < P_swet / 10)
    is_Cs = is_C & (~is_Cw) & (P_sdry < 40) & (P_sdry < P_wwet / 3)
    is_Cf = is_C & (~is_Cw) & (~is_Cs)
    is_a = (is_C | is_D) & (T_hot >= 22)
    is_b = (is_C | is_D) & (~is_a) & (T_mon10 >= 4)
    is_c = (is_C | is_D) & (~is_a) & (~is_b)

    # C 类编码（30-38）
    code = np.where(is_Cw & is_a, 30, code)  # Cwa
    code = np.where(is_Cw & is_b, 31, code)  # Cwb
    code = np.where(is_Cw & is_c, 32, code)  # Cwc
    code = np.where(is_Cs & is_a, 33, code)  # Csa
    code = np.where(is_Cs & is_b, 34, code)  # Csb
    code = np.where(is_Cs & is_c, 35, code)  # Csc
    code = np.where(is_Cf & is_a, 36, code)  # Cfa
    code = np.where(is_Cf & is_b, 37, code)  # Cfb
    code = np.where(is_Cf & is_c, 38, code)  # Cfc

    # D 类和 E 类类似处理（省略以节省篇幅）

    return code
```

### 4.2 现有Python库

**KoppenClimate**（AtmosSciTools）是专门用于 Köppen-Geiger 分类的 Python 库，提供从月温度和降水数据计算气候分类的功能，并支持绘制月温度-降水图（hythergraph）。

**koppengeiger**（JPL）提供加载和生成 Köppen-Geiger 土地覆盖分类栅格的功能，基于 Beck et al. (2018) 的 1 km 分辨率数据集。

**geekgcc** 提供在 Google Earth Engine 中创建 Köppen-Geiger 分类图的功能。


## 五、加速方案

### 5.1 计算热点分析

Köppen-Geiger分类的计算量相对较小，因为**不涉及时间积分**。对于 1° 分辨率（65,000 网格点），计算量约为 \( 10^5 \) 次阈值判定操作。但在高分辨率（0.1° 或更高）下，网格点数达到 \( 10^7 \) 量级，**逐网格的分支判断**成为瓶颈。

### 5.2 NumPy 向量化

上述实现已经充分利用了 NumPy 的向量化能力。核心操作（`mean`、`min`、`max`、`sum`、`where`）都是 C 级速度。对于 1° 分辨率，整个分类过程在**毫秒级**完成。

**关键优化**：避免在分类循环中使用 Python 的 `if/elif/else`。将三级判定展开为**嵌套的 `np.where`**，使所有网格点并行处理。

### 5.3 Numba JIT 加速

对于需要**逐网格循环**的变体（如需要动态调整阈值或处理不规则网格），Numba 可以显著加速：

```python
@njit(parallel=True, fastmath=True, cache=True)
def koppen_geiger_numba(T_monthly, P_monthly, summer_idx, winter_idx):
    nlat, nlon = T_monthly.shape[1], T_monthly.shape[2]
    code = np.zeros((nlat, nlon), dtype=np.int8)
    
    for i in prange(nlat):
        for j in range(nlon):
            # 统计量
            MAT = 0.0; T_cold = 1e6; T_hot = -1e6; MAP = 0.0
            for m in range(12):
                t = T_monthly[m, i, j]
                MAT += t
                if t < T_cold: T_cold = t
                if t > T_hot: T_hot = t
                MAP += P_monthly[m, i, j]
            MAT /= 12.0
            
            # 干旱阈值（简化：假设降水均匀）
            P_threshold = 2.0 * MAT + 14.0
            
            # 第一级判定
            if MAP < 10.0 * P_threshold:
                # B 类
                if MAT >= 18.0:
                    code[i, j] = 21 if MAP < 5*P_threshold else 23
                else:
                    code[i, j] = 22 if MAP < 5*P_threshold else 24
            elif T_cold >= 18.0:
                # A 类
                ...
            elif T_hot < 10.0:
                code[i, j] = 40 if T_hot > 0 else 41  # ET/EF
            elif T_cold > -3.0:
                # C 类
                ...
            else:
                # D 类
                ...
    return code
```

Numba 的 `@njit(parallel=True)` 对逐网格循环的加速效果显著，尤其是在处理不规则网格或需要动态调整阈值时。

### 5.4 JAX 加速与自动微分

JAX 的 `jit` 和 `vmap` 可以将整个分类逻辑编译为 XLA 计算图，在 GPU 上批量执行。对于需要**参数优化**的场景（如“调整干旱阈值使沙漠面积达到 X%”），JAX 的自动微分可以高效计算分类结果对阈值的梯度：

```python
import jax.numpy as jnp
from jax import jit, vmap, grad

@jit
def koppen_classify(T_monthly, P_monthly, params):
    MAT = T_monthly.mean(axis=0)
    T_cold = T_monthly.min(axis=0)
    MAP = P_monthly.sum(axis=0)
    P_threshold = 2 * MAT + 14  # 简化
    is_B = MAP < 10 * P_threshold
    is_A = (~is_B) & (T_cold >= 18)
    # ... 返回整数代码
    return code

# 自动微分：分类结果对干旱阈值的敏感性
dcode_dthreshold = grad(lambda t: koppen_classify(T, P, {'threshold': t}).sum(), 
                         argnums=0)
```

### 5.5 加速分工

| 子模块 | 推荐工具 | 预期效果 |
|--------|---------|---------|
| 统计量计算 | NumPy 向量化 | 毫秒级（1° 分辨率） |
| 分类判定（规则） | NumPy `where` 嵌套 | 批量并行 |
| 分类判定（循环变体） | Numba `@njit(parallel=True)` | 10–50× vs Python |
| 高分辨率（>10⁷ 网格） | JAX `jit` + `vmap` | GPU 上 50–100× |
| 参数优化 | JAX `grad` | 精确梯度，无有限差分 |


## 六、完整分类管线

```
输入：T_monthly(12, λ, φ), P_monthly(12, λ, φ), 纬度 φ

=== 统计量计算 ===
1. MAT = mean(T_monthly, axis=month)
2. T_cold = min(T_monthly, axis=month)
3. T_hot = max(T_monthly, axis=month)
4. T_mon10 = count(T_monthly > 10, axis=month)
5. MAP = sum(P_monthly, axis=month)
6. P_dry = min(P_monthly, axis=month)

=== 夏/冬划分 ===
7. 根据纬度确定夏季月份（北半球：4-9月；南半球：10-3月）
8. 计算 P_sdry, P_wdry, P_swet, P_wwet

=== 干旱阈值 ===
9. 计算冬季降水占比 winter_frac 和夏季降水占比 summer_frac
10. P_threshold = where(winter_frac>0.7, 2*MAT,
                    where(summer_frac>0.7, 2*MAT+28, 2*MAT+14))

=== 第一级分类 ===
11. is_B = MAP < 10 * P_threshold
12. is_A = ~is_B & T_cold >= 18
13. is_E = ~is_B & T_hot < 10
14. is_C = ~is_B & ~is_A & ~is_E & T_cold > -3 & T_cold < 18
15. is_D = ~is_B & ~is_A & ~is_E & ~is_C

=== 第二/三级细分 ===
16. A 类：Af(P_dry>=60), Am, Aw
17. B 类：BW(MAP<5*P_thr), BS; h(MAT>=18), k(MAT<18)
18. C 类：Cw(P_wdry<P_swet/10), Cs(P_sdry<40 & P_sdry<P_wwet/3), Cf
        a(T_hot>=22), b(T_mon10>=4), c
19. D 类：同 C 类降水季节性，温度细分
20. E 类：ET(0<T_hot<10), EF(T_hot<=0)

=== 编码输出 ===
21. 将三级分类映射为整数代码（11-30）
22. 输出：climate_code(λ, φ)
```

这个管线的输出是**具有生态意义的气候分类图**。气候分类是生物群系、土壤类型和农业类型判定的直接输入，闭合了从行星辐射强迫到地表生态系统的完整因果链。