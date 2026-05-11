# ANOVA Matern 核函数优化设计

## 1. 问题陈述

### 1.1 当前核函数

当前 GPR 使用 `ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel`，长度尺度约束在 `(0.3, 5.0)`。

### 1.2 存在的问题

| 问题 | 表现 | 根因 |
|---|---|---|
| LOOCV 预测误差大 | 留一交叉验证中产率预测偏差较大 | 核函数未捕捉初始状态×工艺的交互效应 |
| ARD 长度尺度不合理 | 某些本应重要的特征被忽略，不重要的特征长度尺度偏小 | 平稳核假设所有区域平滑度相同 |
| 不同样品预测表现差异大 | 对某些样品预测很准，对另一些偏差大 | 工艺效果依赖于初始状态（非平稳行为） |
| 不确定度 σ 不可靠 | σ 值与实际误差不匹配 | 超参数优化陷入局部最优，σ 失真 |

### 1.3 核心物理假设

所有样品均为相同材质的镍箔，经过不同初始加工状态（冷轧等）后，进行不同工艺条件（温度、时间、H₂、Ar）的退火。初始状态通过 EBSD 表征（Pre_ 特征），退火结果通过目标晶向产率衡量。

**关键物理机制**：同一退火工艺对不同初始状态的镍箔产生不同效果——温度对粗晶和细晶材料的再结晶行为有不同的影响路径。这是非平稳性的来源。

---

## 2. 设计目标

1. **预测精度**：LOOCV RMSE 显著低于当前平稳 Matern 核
2. **可解释性**：ARD 长度尺度按特征组可解释，方差成分可分解为主效应和交互效应
3. **不确定度可靠性**：σ 值应与实际预测误差匹配
4. **小样本兼容**：在 ~176 个训练样本（44 实验 × 4 scheme）上稳定运行

---

## 3. 核函数设计

### 3.1 数学结构（ANOVA 分解）

```
K = σ²_pre × K_pre(θ_pre)
  + σ²_target × K_target(θ_target)
  + σ²_proc × K_proc(θ_proc)
  + σ²_inter × (K_pre(θ_pre) × K_proc(θ_proc))
  + σ²_noise
```

其中：
- `K_pre`：Matern 5/2，作用于 Pre_ 特征（28 维）
- `K_target`：Matern 5/2，作用于 Target_ 特征（9 维）
- `K_proc`：Matern 3/2，作用于 Process_ 特征（4 维）
- `K_pre × K_proc`：乘积核，捕捉初始状态×工艺的交互效应
- 白噪声核：捕捉实验测量误差

### 3.2 物理含义

| 组件 | 核类型 | nu | 物理含义 |
|---|---|---|---|
| K_pre | Matern 5/2 | 2.5 | EBSD 初始状态对产率的直接影响（平滑） |
| K_target | Matern 5/2 | 2.5 | 目标晶向选择对产率的影响（平滑） |
| K_proc | Matern 3/2 | 1.5 | 工艺参数对产率的影响（允许突变/阈值效应） |
| K_pre × K_proc | Matern 5/2 × Matern 3/2 | 2.5 × 1.5 | 工艺效果依赖于初始状态（交互） |
| White | — | — | 实验测量噪声 |

### 3.3 混合 nu 的理由

- **Matern 5/2 (nu=2.5)**：假设函数二阶可导，适用于平滑的物理响应（初始状态影响、晶向选择）
- **Matern 3/2 (nu=1.5)**：仅一阶可导，允许更尖锐的响应（温度阈值、再结晶临界点）
- 材料科学中的再结晶、相变等过程存在阈值效应，Process_ 使用 nu=1.5 更符合物理实际

### 3.4 参数量

| 参数 | 数量 | 说明 |
|---|---|---|
| K_pre lengthscale | 28 | Pre_ 特征 ARD |
| K_target lengthscale | 9 | Target_ 特征 ARD |
| K_proc lengthscale | 4 | Process_ 特征 ARD（与交互项共享） |
| σ²_pre | 1 | 主效应振幅 |
| σ²_target | 1 | 主效应振幅 |
| σ²_proc | 1 | 主效应振幅 |
| σ²_inter | 1 | 交互效应振幅 |
| σ²_noise | 1 | 噪声方差 |
| **总计** | **~46** | 样本/参数比 ≈ 3.8:1 |

### 3.5 参数共享策略

**长度尺度（θ）共享**：K_pre 和交互项中的 K_pre 共享同一套 θ_pre（28 个参数）；K_proc 和交互项中的 K_proc 共享同一套 θ_proc（4 个参数）。

**方差（σ²）独立**：每个组件有独立的振幅参数。交互项通过固定一个子核方差为 1.0 实现独立 σ²_inter。

---

## 4. GPy 实现细节

### 4.1 依赖

```
GPy >= 1.10
```

### 4.2 数据标准化

所有特征和目标值必须标准化：

```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler().fit(X_df)
scaler_y = StandardScaler().fit(y.reshape(-1, 1))

X_scaled = scaler_X.transform(X_df)
y_scaled = scaler_y.transform(y.reshape(-1, 1))
```

**理由**：
- Matern 核基于欧氏距离，StandardScaler 使 "1 单位距离 = 1 标准差"
- 长度尺度初始化在 1.0 附近具有统计意义
- 目标值标准化是 GPy 稳定拟合白噪声方差的前提

### 4.3 核函数构建

```python
import GPy

# 特征索引（按列位置）
pre_idx = list(range(0, 28))       # Pre_ 特征
target_idx = list(range(28, 37))   # Target_ 特征
proc_idx = list(range(37, 41))     # Process_ 特征

# --- 主效应核 ---
k_pre_main = GPy.kern.Matern52(input_dim=28, active_dims=pre_idx, ARD=True, name='pre_main')
k_target = GPy.kern.Matern52(input_dim=9, active_dims=target_idx, ARD=True, name='target')
k_proc_main = GPy.kern.Matern32(input_dim=4, active_dims=proc_idx, ARD=True, name='proc_main')

# --- 交互核（独立实例，固定 pre_inter 方差为 1.0）---
k_pre_inter = GPy.kern.Matern52(input_dim=28, active_dims=pre_idx, ARD=True, name='pre_inter')
k_proc_inter = GPy.kern.Matern32(input_dim=4, active_dims=proc_idx, ARD=True, name='proc_inter')

# 固定 pre_inter 的方差为 1.0 → 交互项的振幅完全由 proc_inter.variance 决定
k_pre_inter.variance.fix(1.0)

# --- 白噪声 ---
k_noise = GPy.kern.White(input_dim=1, name='noise')

# --- 组合 ---
K_total = k_pre_main + k_target + k_proc_main + k_pre_inter * k_proc_inter + k_noise
```

### 4.4 参数绑定

```python
model = GPy.models.GPRegression(X_scaled, y_scaled, K_total)

# 绑定长度尺度（仅长度尺度，不绑定方差）
# 注意：实际路径取决于 GPy 的参数命名，可能需要先 print(model) 确认路径格式
# 以下为预期路径，若不对则根据 model 参数列表调整正则
model.tie_params('pre_main.lengthscale', 'pre_inter.lengthscale')
model.tie_params('proc_main.lengthscale', 'proc_inter.lengthscale')
model.tie_params()  # 应用绑定
```

### 4.5 先验设置

```python
# 交互项方差加弱 Gamma 先验（偏向 0，防止过拟合）
# σ²_inter 的路径：proc_inter.variance（因为 pre_inter.variance 被固定为 1.0）
model['proc_inter.variance'].set_prior(GPy.priors.Gamma(a=1.5, b=1.0))
```

### 4.6 分组长度尺度边界

```python
# Pre_ 长度尺度：允许短程和长程相关性
model['pre_main.lengthscale'].constrain_bounded(0.3, 5.0)

# Target_ 长度尺度：二值编码，防止过度平滑
model['target.lengthscale'].constrain_bounded(0.5, 3.0)

# Process_ 长度尺度：工艺参数应有明确响应
model['proc_main.lengthscale'].constrain_bounded(0.3, 3.0)
```

### 4.7 多起点优化

```python
model.optimize_restarts(num_restarts=20, robust=True)
```

`robust=True` 确保单次优化失败不会中断整个流程。

---

## 5. 长度尺度分组边界设计

| 特征组 | 维度 | 下界 | 上界 | 理由 |
|---|---|---|---|---|
| Pre_ | 28 | 0.3 | 5.0 | EBSD 状态变化范围大，允许短程和长程相关性 |
| Target_ | 9 | 0.5 | 3.0 | 二值 one-hot，过度平滑会抹掉目标差异 |
| Process_ | 4 | 0.3 | 3.0 | 工艺参数应有明确响应，防止被忽略 |

---

## 6. 与当前实现的差异

| 方面 | 当前实现 | 新设计 |
|---|---|---|
| 核函数 | ConstantKernel × Matern(5/2) + White | ANOVA: 3 主效应 + 1 交互 + White |
| nu | 统一 2.5 | Process_ 用 1.5，其余用 2.5 |
| 交互效应 | 无（隐式，被主核吸收） | 显式建模，独立振幅 |
| 参数绑定 | 无 | 长度尺度共享，方差独立 |
| 方差组件 | 1 个全局 ConstantKernel | 4 个独立 σ²（pre, target, proc, inter） |
| 超参数优化 | sklearn 默认（单次 L-BFGS-B） | GPy 多起点（20 次重启） |
| 先验 | 无 | 交互项方差弱 Gamma 先验 |
| 依赖 | sklearn | GPy |

---

## 7. 评估计划

### 7.1 LOOCV 对比

用新核函数重训模型后，与当前平稳 Matern 核的 LOOCV RMSE 做直接对比。

### 7.2 ARD 长度尺度分析

检查新核的长度尺度是否比当前核更合理：
- Process_ 特征（尤其温度）的长度尺度应偏小（重要）
- 不重要的 Pre_ 特征长度尺度应偏大

### 7.3 方差成分分析

分解总预测方差中各组件的贡献：
- σ²_pre 占比多少（初始状态的主效应）
- σ²_proc 占比多少（工艺的主效应）
- σ²_inter 占比多少（交互效应）

如果 σ²_inter 显著大于 0，说明交互效应确实存在，验证了非平稳性假设。

### 7.4 不确定度校准

用 LOOCV 残差检查 σ 是否可靠：理想情况下，~68% 的真实值应落在 μ ± σ 范围内。

---

## 8. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|---|---|---|
| 176 样本支撑 ~46 参数过紧 | 中 | 弱 Gamma 先验约束交互项；若过拟合严重，可降级回纯加法（去掉交互项） |
| GPy 优化收敛困难 | 中 | 20 次多起点重启；robust=True 容错 |
| 交互项被先验完全压制 | 低 | 使用弱先验 Gamma(1.5, 1.0)，让数据主导 |
| 长度尺度绑定导致优化路径错误 | 低 | tie_params 是 GPy 的标准功能，有充分的社区验证 |

---

## 9. 回退方案

如果新核函数表现不如预期：
1. **去掉交互项**：退回纯 ANOVA 加法模型（3 个主效应 + 白噪声）
2. **换回平稳核**：退回 sklearn 的 Matern(5/2) + ARD，仅优化长度尺度边界
3. **增加数据**：通过文献挖掘补充更多训练样本，扩大样本/参数比
