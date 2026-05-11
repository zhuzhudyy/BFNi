# 模型健康状态仪表盘设计

## 1. 目标

每次新实验完成后，运行一次即可看到模型的全局健康状态。输出为一张 2×2 的仪表盘图。

## 2. 文件

新建 `bo_optimization/model_health_dashboard.py`，与 `honest_visualization.py` 职责分离：
- `honest_visualization.py`：物理因果链分析（PDP、ICE、ARD）
- `model_health_dashboard.py`：模型健康/进度追踪（RMSE、覆盖率、方差分解）

## 3. 仪表盘布局

```
┌──────────────────────┬──────────────────────┐
│ ① RMSE vs 数据量     │ ② Process_ pairplot   │
│   Bootstrap 子采样曲线 │   训练点(蓝) + LHS(橙)│
├──────────────────────┼──────────────────────┤
│ ③ 方差成分饼图        │ ④ 覆盖率指标          │
│   σ² 分解            │   进度条 + 数字        │
└──────────────────────┴──────────────────────┘
```

## 4. 各面板详细设计

### 4.1 面板 ①：RMSE vs 数据量（Train/Test Split）

**方法**：
- 对训练数据做 6 次 train/test split：50/50, 60/40, 70/30, 80/20, 90/10, 全量 LOOCV
- 每次用训练集训练 ANOVA Matern GPR（5 次多起点，减少耗时）
- 在测试集上计算 RMSE
- 全量数据用 LOOCV 计算 RMSE
- 重复 3 次（不同随机划分）取均值 ± 标准差

**X 轴**：训练集大小（样本数）
**Y 轴**：测试集 RMSE
**样式**：蓝色实线 + 浅蓝色置信带，当前数据量用红色虚线标注

**预期输出**：曲线应随数据量增加而下降。如果曲线趋于平坦，说明继续采样的边际收益递减。

### 4.2 面板 ②：Process_ 空间 Pairplot

**方法**：
- 6 个子图展示 Process_ 4 维的所有两两组合（Temp×Time, Temp×H₂, Temp×Ar, Time×H₂, Time×Ar, H₂×Ar）
- 蓝色点：现有训练数据
- 橙色点：LHS 推荐的下一批采样点（如果有的话）
- 每个子图标题显示该维度对的相关系数

**约束线**：在 H₂×Ar 子图中画 Ar = 2×H₂ 的虚线

**样式**：训练点用小圆点（alpha=0.5），LHS 推荐点用大三角形（红色边框）

### 4.3 面板 ③：方差成分饼图

**方法**：
- 从训练好的 ANOVA Matern 核中提取 σ²_pre, σ²_target, σ²_proc, σ²_inter
- 绘制饼图

**标签**：
- σ²_pre → "初始状态"
- σ²_target → "目标晶向"
- σ²_proc → "工艺参数"
- σ²_inter → "交互效应"

**样式**：4 色饼图，交互效应用斜线填充（强调其特殊性）

### 4.4 面板 ④：覆盖率指标

**方法**：
- 调用 `compute_space_coverage()` 获取覆盖率、中位距离、P90 距离
- 调用 `estimate_experiments_for_coverage()` 获取达到 30% 需要的实验数
- 用进度条 + 数字展示

**内容**：
```
Process_ 空间覆盖率
━━━━━━━━━━━━━━━━━━━━ 14.9%
当前: 76 样本 | 中位距离: 0.468
达 30% 覆盖还需: ~41 组实验
```

**样式**：水平进度条（绿色填充），下方显示关键数字

## 5. 运行方式

```bash
python bo_optimization/model_health_dashboard.py
```

交互式：选择方案 → 加载/训练模型 → 生成仪表盘 → 保存为 PNG

## 6. 依赖

- matplotlib, numpy, pandas, sklearn（已有）
- `contextual_bo_model.py` 中的 `ANOVAMaternKernel`, `GPRWithPriors`, `ContextualBayesianOptimizer`
- `compute_space_coverage()`, `estimate_experiments_for_coverage()`（已实现）
