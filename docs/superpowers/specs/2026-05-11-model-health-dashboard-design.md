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

### 4.1 面板 ①：RMSE vs 数据量（K-Fold CV 学习曲线）

**方法**：
- 在 4 个数据子集上分别执行 5-Fold CV：20, 40, 60, 76（全量）个样本
- 每次用训练折训练 ANOVA Matern GPR（3 次多起点，用全量模型参数作 warm start）
- 在验证折上计算 RMSE，5 折取均值 ± 标准差
- 全量数据额外做 LOOCV 作为最终指标

**X 轴**：训练集大小（样本数）
**Y 轴**：5-Fold CV RMSE（均值 ± 标准差）
**样式**：蓝色实线 + 浅蓝色置信带，全量 LOOCV 用红色星标

**预期输出**：曲线应随数据量增加而下降。如果曲线趋于平坦，说明继续采样的边际收益递减。

**性能优化**：3 次多起点（非 20 次）+ 全量模型超参数作 warm start，减少计算耗时。

### 4.2 面板 ②：Process_ 关键子图

**方法**：
- 1×2 网格展示两个最关键的工艺维度对：
  - 左图：Temp × Time（退火温度-时间，主效应）
  - 右图：H₂ × Ar（气体流量，含物理约束）
- 蓝色点：现有训练数据
- 绿色方块：已分配的空间填充采样点（已做过实验）
- 橙色三角：待分配的空间填充采样点（未做实验）
- 数据来源：读取 `space_filling_state_方案X.json` 状态文件（由 `space_filling_plan.py` 维护）
- 每个子图标题显示维度对名称 + 相关系数

**约束线**：在 H₂×Ar 子图中画 Ar = 2×H₂ 的虚线（不可行区域灰色填充）。LHS 推荐点已通过拒绝采样确保满足约束，不会落在不可行区域。

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

**预计耗时**：面板 ① 的 K-Fold CV 学习曲线最耗时（~4 个数据量 × 5 折 × 3 次重启 = 60 次 GPR 拟合），其余面板秒级。总耗时约 3-5 分钟。

## 6. 依赖

- matplotlib, numpy, pandas, sklearn（已有）
- `contextual_bo_model.py` 中的 `ANOVAMaternKernel`, `GPRWithPriors`, `ContextualBayesianOptimizer`
- `compute_space_coverage()`, `estimate_experiments_for_coverage()`（已实现）
