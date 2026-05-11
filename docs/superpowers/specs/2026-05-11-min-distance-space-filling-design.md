# 空间填充最小物理距离约束

## 1. 目标

在空间填充采样的贪心选择中增加 per-dimension 最小物理距离约束，避免生成与已有数据点过近的采样点。物理依据：退火实验中微小参数变化（如 ±10°C）不导致产率显著变化，过近的采样点浪费实验资源。

## 2. 设计参数

### 2.1 默认最小物理距离

```python
DEFAULT_MIN_PHYSICAL_DISTANCES = {
    'Process_Temp': 30,   # ±30°C
    'Process_Time': 3,    # ±3h
    'Process_H2': 15,     # ±15sccm
    'Process_Ar': 50,     # ±50sccm
}
```

### 2.2 距离判定逻辑

候选点 c 被拒绝的条件（高维矩形禁区）：

```
∃ 参考点 r ∈ X_virtual: |c_i - r_i| < threshold_i  ∀ i ∈ 所有维度
```

即候选点落入某个参考点的**全维度矩形禁区**才被剔除。单一维度接近但其他维度远离 → 不拒绝。

### 2.3 空间选择

距离计算使用**固定边界归一化**（与 `_maximin_distance` 的数据依赖 MinMaxScaler 不同）。物理阈值根据 `self.bounds` 转换为归一化阈值：

```
norm_threshold_i = physical_threshold_i / (bounds_i_max - bounds_i_min)
```

候选点和参考点同样用固定边界归一化到 [0,1]，然后在归一化空间中比较 per-dimension 距离。这样阈值不随当前数据分布变化，物理意义稳定。

## 3. 多轮自适应采样

### 3.1 机制

完成一轮高稀疏度采样后，自动缩小间距约束进入下一轮：

| 轮次 | sparsity_multiplier | 有效间距（以 Temp 为例） | 说明 |
|------|---------------------|------------------------|------|
| Round 1 | 1.0 | ±30°C | 粗粒度探索 |
| Round 2 | 0.5 | ±15°C | 中等细化 |
| Round 3 | 0.25 | ±7.5°C | 精细采样 |

### 3.2 状态文件扩展

```json
{
  "round": 1,
  "sparsity_multiplier": 1.0,
  "points": [...],
  "allocations": [...]
}
```

- `round`: 当前轮次（从 1 开始）
- `sparsity_multiplier`: 当前轮次的间距缩放因子

### 3.3 触发逻辑

在 `space_filling_plan.py` 的 `run_space_filling_plan` 中：

1. 检测当前轮次所有点是否已分配（`n_remaining == 0`）
2. 若已全部分配 → 提示用户"当前轮次已完成，是否进入下一轮？"
3. 用户确认 → `round += 1`, `sparsity_multiplier *= 0.5`
4. 用新的 `sparsity_multiplier * DEFAULT_MIN_PHYSICAL_DISTANCES` 生成下一批点
5. 新点追加到同一 `points` 数组，带 `round` 标记

### 3.4 用户可配置

`space_filling_plan.py` 允许用户自定义：
- `SPARSITY_DECAY = 0.5`（每轮缩小比例）
- 最小轮次阈值（如 multiplier < 0.1 时停止）

## 4. 修改文件

### 4.1 `bo_optimization/contextual_bo_model.py`

**4.1.1 新增常量**（文件顶部，类定义之前）

```python
DEFAULT_MIN_PHYSICAL_DISTANCES = {
    'Process_Temp': 30,
    'Process_Time': 3,
    'Process_H2': 15,
    'Process_Ar': 50,
}
```

**4.1.2 修改 `_greedy_select_mixed_score`**

新增参数 `min_physical_distances`（dict 或 None）。

在主循环**之前**预计算归一化坐标和阈值，在循环中过滤 `remaining`：

```python
if min_physical_distances is not None:
    # 预计算：物理阈值 → 固定边界归一化阈值（循环外只算一次）
    bounds_min = np.array([self.bounds[col][0] for col in self.process_cols])
    bounds_range = np.array([self.bounds[col][1] - self.bounds[col][0] for col in self.process_cols])
    norm_thresholds = np.array([
        min_physical_distances.get(col, 0) / bounds_range[i]
        for i, col in enumerate(self.process_cols)
    ])
    # 候选点归一化（循环外只算一次）
    X_cand_norm = (X_candidates - bounds_min) / bounds_range

for _ in range(min(n_select, len(remaining))):
    # 过滤最小距离约束
    if min_physical_distances is not None and remaining:
        if len(X_virtual) == 0:
            pass  # 冷启动：无历史点，跳过过滤
        else:
            X_ref_norm = (X_virtual - bounds_min) / bounds_range
            feasible = []
            for local_i, global_idx in enumerate(remaining):
                c = X_cand_norm[global_idx]
                diffs = np.abs(X_ref_norm - c)
                in_exclusion = np.all(diffs < norm_thresholds, axis=1)
                if not np.any(in_exclusion):
                    feasible.append(global_idx)
            if not feasible:
                # 兜底：原地缩小阈值 50%，后续迭代继续生效
                norm_thresholds *= 0.5
                for local_i, global_idx in enumerate(remaining):
                    c = X_cand_norm[global_idx]
                    diffs = np.abs(X_ref_norm - c)
                    in_exclusion = np.all(diffs < norm_thresholds, axis=1)
                    if not np.any(in_exclusion):
                        feasible.append(global_idx)
                if feasible:
                    import warnings
                    warnings.warn(f"[空间填充] 候选空间耗尽，阈值已缩小 50%，后续迭代沿用宽松阈值")
                else:
                    import warnings
                    warnings.warn(f"[空间填充] 即使缩小阈值仍无可用候选，已选 {len(selected_indices)} 个")
                    break
            remaining = feasible

    # 计算当前候选点到虚拟参考集的 maximin distance
    X_cand_remaining = X_candidates[remaining]
    d_min = self._maximin_distance(X_cand_remaining, X_virtual, self.process_cols)
    # ... 后续 EI + maximin 混合得分逻辑不变
```

**4.1.3 修改 `suggest_space_filling`**

新增参数 `min_physical_distances=None`，透传给 `_greedy_select_mixed_score`。

### 4.2 `bo_optimization/space_filling_plan.py`

**4.2.1 修改 `_init_state`**

扩展状态文件格式，新增 `round` 和 `sparsity_multiplier` 字段：

```python
def _init_state(scheme, recommendations, alpha, round=1, sparsity_multiplier=1.0):
    state = {
        'scheme': scheme,
        'round': round,
        'sparsity_multiplier': sparsity_multiplier,
        'alpha': alpha,
        'points': [...],
        'allocations': []
    }
```

**4.2.2 修改 `run_space_filling_plan`**

- 导入 `DEFAULT_MIN_PHYSICAL_DISTANCES`, `SPARSITY_DECAY`
- 初始化时设置 `round=1`, `sparsity_multiplier=1.0`
- 计算当前轮次的间距：`effective_dist = {k: v * sparsity_multiplier for k, v in DEFAULT_MIN_PHYSICAL_DISTANCES.items()}`
- 调用 `suggest_space_filling` 时传入 `min_physical_distances=effective_dist`
- 打印实际选出数量（可能 < N）
- 当 `n_remaining == 0` 且用户确认进入下一轮时：
  - `round += 1`
  - `sparsity_multiplier *= SPARSITY_DECAY`
  - 用新间距生成下一批点，追加到 `state['points']`

**4.2.3 新增常量**

```python
SPARSITY_DECAY = 0.5  # 每轮间距缩小比例
```

## 5. 行为变化

| 场景 | 之前 | 之后 |
|------|------|------|
| N=10，训练数据稀疏 | 选出 10 个点 | 选出 10 个点（间距约束不影响） |
| N=10，部分区域密集 | 选出 10 个点（可能有冗余） | 选出 6-8 个点（自动过滤冗余） |
| N=10，全部区域密集 | 选出 10 个点（高度冗余） | 选出 2-3 个点 + Warning |
| 阈值过大，第 1 个点后清空 | N/A | 阈值缩小 50% 重试 + Warning |
| Round 1 全部分配完 | 无后续 | 自动提示进入 Round 2（间距 ×0.5） |
| Round 2 全部分配完 | 无后续 | 自动提示进入 Round 3（间距 ×0.25） |

## 6. 不修改的部分

- `N` 的默认值（仍为 10），间距约束自然控制有效密度
- dashboard 的读取逻辑（自动适配实际点数，多轮点数叠加显示）
- LHS 候选池生成逻辑（不预过滤，保持候选多样性）
- `contextual_bo_model.py` 中 `suggest_space_filling` 的聚类逻辑

## 7. 验证

```bash
python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('bo_optimization/space_filling_plan.py', encoding='utf-8').read()); print('OK')"
```

运行 `python bo_optimization/space_filling_plan.py`，确认：
- 实际选出点数 ≤ N
- 任意两点之间满足 per-dimension 间距约束
- 打印 Warning（如果点数减少）
