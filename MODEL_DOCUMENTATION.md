# 多任务贝叶斯优化模型技术文档

## 1. 模型概述

### 1.1 核心功能
本模型实现**多任务贝叶斯优化（Multi-Task Bayesian Optimization）**，用于镍基合金退火工艺参数优化，目标是在给定初始微观结构（EBSD数据）的情况下，推荐最优退火工艺参数，以最大化特定目标晶向的产率。

### 1.2 技术架构
```
输入层: EBSD预处理特征(21维) + 晶向Multi-Hot(9维) + 方案标识(1维) + 工艺参数(4维) = 35维
    ↓
代理模型: 高斯过程回归(GPR) + ARD Matern核函数
    ↓
采集函数: 期望提升(Expected Improvement, EI)
    ↓
输出: 最优工艺参数推荐 + 不确定性量化
```

---

## 2. 特征工程

### 2.1 EBSD预处理特征 (21维)
从退火前的晶粒取向分布图像提取：

| 特征类别 | 特征名称 | 维度 | 物理意义 |
|---------|---------|------|---------|
| IPF颜色均值 | Pre_R_Mean, Pre_G_Mean, Pre_B_Mean | 3 | 晶粒取向的平均颜色分布 |
| IPF颜色标准差 | Pre_R_Std, Pre_G_Std, Pre_B_Std | 3 | 晶粒取向的离散程度 |
| GND统计特征 | Pre_GND_Mean, Pre_GND_Std | 2 | 几何必需位错密度均值与波动 |
| GND分位数 | Pre_GND_Q25, Q50, Q75, Q90, Q95, Q99 | 6 | 位错密度的分布形态 |
| GND分布特征 | Pre_GND_IQR, Peak, Skewness, Kurtosis | 4 | 分布的偏态和峰态 |
| GND变异系数 | Pre_GND_CV | 1 | 位错密度的相对波动 |
| GND高低比 | Pre_GND_HighRatio, LowRatio | 2 | 高位错/低位错区域占比 |

### 2.2 目标晶向上下文编码 (10维)
将目标晶向方案编码为上下文特征，由两部分组成：

#### 2.2.1 晶向Multi-Hot编码 (9维)
```python
# 4个预定义方案共涉及9个独特晶向
Scheme 1: <103>, <102>, <301>
Scheme 2: <114>, <115>, <105>
Scheme 3: <124>, <125>, <214>
Scheme 4: <103>, <114>, <124>

# Multi-Hot编码示例（方案1）
Target_103 = 1, Target_102 = 1, Target_301 = 1  # 激活的晶向
Target_114 = 0, Target_115 = 0, ...              # 未激活的晶向
```

| 维度 | 特征名 | 对应晶向 |
|------|--------|---------|
| 1 | Target_103 | <103> |
| 2 | Target_102 | <102> |
| 3 | Target_301 | <301> |
| 4 | Target_114 | <114> |
| 5 | Target_115 | <115> |
| 6 | Target_105 | <105> |
| 7 | Target_124 | <124> |
| 8 | Target_125 | <125> |
| 9 | Target_214 | <214> |

**关键设计**：一个方案可同时激活多个晶向（Multi-Hot），实现多目标优化。

#### 2.2.2 方案标识 (1维)
```python
Target_Scheme = {1, 2, 3, 4}  # 标识当前使用哪个预定义方案
```

**作用**：区分不同方案的数据分布，帮助模型学习方案级别的差异。

### 2.3 工艺参数 (4维)
| 参数 | 符号 | 搜索范围 | 单位 |
|-----|------|---------|------|
| 退火温度 | Process_Temp | [1000, 1500] | °C |
| 退火时间 | Process_Time | [1, 30] | h |
| 氢气流量 | Process_H2 | [0, 160] | sccm |
| 氩气流量 | Process_Ar | [0, 800] | sccm |

---

## 3. 代理模型

### 3.1 高斯过程回归 (GPR)
**核函数配置**：
```
Kernel = ConstantKernel(1.0) × Matern(length_scale=[...], nu=2.5) + WhiteKernel(noise_level=1e-4)
```

| 组件 | 功能 |
|-----|------|
| ConstantKernel | 控制输出幅值 |
| Matern(nu=2.5) | 建模二阶可导的物理过程 |
| ARD (长度尺度向量) | 为35个特征各学习独立敏感度 |
| WhiteKernel | 吸收实验测量噪声 |

### 3.2 ARD (自动相关性确定)
**核心思想**：通过优化每个特征的长度尺度，自动识别重要特征。

```python
# 长度尺度的解读
length_scale < 1    → 高重要性（模型对该特征敏感）
length_scale ≈ 1-10 → 中等重要性
length_scale > 100  → 低重要性（模型对该特征不敏感）
```

**训练后典型结果**：
- Target_103: 0.15 (★★★ 极重要)
- Pre_GND_Q25: 0.31 (★★★ 高敏感)
- Pre_R_Mean: 100000 (★ 几乎无关)

### 3.3 数据标准化
```python
# 输入标准化（Z-score）
X_scaled = (X - μ) / σ

# 输出归一化（内置normalize_y=True）
y_normalized ~ N(0, 1)
```

---

## 4. 采集函数

### 4.1 期望提升 (Expected Improvement)
**数学定义**：
```
EI(x) = E[max(0, f(x) - y_best)]
```

其中：
- `f(x)`：GPR预测的产率分布（高斯分布）
- `y_best`：当前目标晶向方案的历史最优产率
- `EI`：选择该参数组合的预期改进量

### 4.2 局部最优策略
**关键修正**：EI计算使用**目标晶向相关的局部最优**，而非全局最优。

```python
# 错误做法（全局最优）
y_best = max(all_training_yields)

# 正确做法（局部最优）
y_best = max(yields_for_same_target_orientation)
```

这确保推荐结果针对特定目标晶向优化。

### 4.3 多起点优化
```python
# 拉丁超立方采样 + L-BFGS-B优化
n_random_starts = 100000  # 随机采样点数
optimizer = 'fmin_l_bfgs_b'  # 梯度优化
```

---

## 5. 数据构建策略

### 5.1 产率计算
从EBSD图像统计目标晶向的像素占比：
```python
Yield = (目标晶向像素数 / 总像素数) × 100%
```

### 5.2 颜色匹配容忍度
```python
# IPF颜色匹配（RGB空间欧氏距离）
tolerance = 80  # 像素值差异容忍度
match = ||RGB_pixel - RGB_target|| < tolerance
```

### 5.3 多目标数据扩展
**核心创新**：一个实验扩展为4行数据。

```python
# 原始数据：1个实验 → 1个产率
Experiment_001: Condition + EBSD → Yield_for_<103>

# 多任务扩展：1个实验 → 4个产率（不同目标晶向）
Experiment_001_Scheme1: ..., Target_103=1, Target_102=1, ... → Yield_103
Experiment_001_Scheme2: ..., Target_114=1, Target_115=1, ... → Yield_114
Experiment_001_Scheme3: ..., Target_124=1, Target_125=1, ... → Yield_124
Experiment_001_Scheme4: ..., Target_103=1, Target_114=1, ... → Yield_103_114_124
```

**效果**：样本量扩大4倍，支持多任务学习。

---

## 6. 模型验证

### 6.1 留一法交叉验证 (LOOCV)
适用于小样本场景（N=44）：
```python
# 每次留出一个样本，用其余样本训练
for i in range(N):
    train = data[all except i]
    test = data[i]
    model.fit(train)
    pred = model.predict(test)
```

**指标**：MAE（平均绝对误差）、RMSE（均方根误差）

### 6.2 认知收敛分析
```
全域认知不确定度 = mean(σ(x)), x ∈ 整个参数空间
```

随着实验数据增加，全域不确定度应单调下降，表明模型认知提升。

---

## 7. 可视化输出

### 7.1 ARD特征重要性分析
- **左图**：特征重要性排序（水平条形图）
- **右图**：各类特征长度尺度分布（小提琴图+散点+智能标注）

### 7.2 响应面可视化
- **3D曲面图**：产率随工艺参数变化
- **2D等高线图**：工艺参数空间的最优区域
- **不确定性图**：模型预测置信度分布

### 7.3 收敛性分析
- Max EI衰减曲线
- Simple Regret曲线
- 累积最优产率曲线
- LOOCV误差曲线

---

## 8. 使用流程

### 8.1 训练模型
```bash
python data_builder.py  # 构建训练数据
python model_visualization.py  # 训练并可视化
```

### 8.2 预测新样品
```bash
python predict_new_sample.py
# 输入：新样品的EBSD预处理文件路径
# 输出：推荐的最优工艺参数
```

### 8.3 分析特征重要性
```bash
python ard_feature_importance.py
# 输出：各特征的ARD长度尺度及重要性排序
```

---

## 9. 关键设计决策

| 决策 | 选择 | 理由 |
|-----|------|------|
| 核函数 | Matern(nu=2.5) | 物理过程通常二阶可导 |
| ARD | 启用 | 自动特征选择，提升模型解释性 |
| EI基准 | 局部最优 | 避免跨晶向干扰，确保针对性优化 |
| 验证方法 | LOOCV | 小样本场景下最可靠的验证 |
| 数据扩展 | 1→4 | 支持多任务学习，提升样本效率 |

---

## 10. 文件清单

| 文件 | 功能 |
|-----|------|
| `contextual_bo_model.py` | 核心贝叶斯优化类 |
| `data_builder.py` | 训练数据构建 |
| `model_visualization.py` | 模型训练与可视化 |
| `predict_new_sample.py` | 新样品工艺推荐 |
| `ard_feature_importance.py` | ARD特征重要性分析 |
| `cognitive_convergence.py` | 认知收敛性分析 |

---

## 11. 参考文献

1. Rasmussen & Williams, "Gaussian Processes for Machine Learning", 2006
2. Snoek et al., "Practical Bayesian Optimization of Machine Learning Algorithms", 2012
3. Swersky et al., "Multi-Task Bayesian Optimization", 2013
4. Frazier, "A Tutorial on Bayesian Optimization", 2018

---

**文档版本**：v1.0  
**最后更新**：2025年4月1日  
**作者**：BFNi项目团队
