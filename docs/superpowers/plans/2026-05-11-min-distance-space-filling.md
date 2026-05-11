# 空间填充最小物理距离约束 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-dimension minimum physical distance constraints to space-filling greedy selection, with multi-round adaptive sampling.

**Architecture:** Modify `_greedy_select_mixed_score` in `contextual_bo_model.py` to filter candidates falling within hyperrectangle exclusion zones around reference points. Extend `space_filling_plan.py` state file with round/sparsity tracking for multi-round sampling.

**Tech Stack:** numpy, json, Python warnings

---

### Task 1: Add constant and modify `_greedy_select_mixed_score`

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py:14-16` (add constant after imports)
- Modify: `bo_optimization/contextual_bo_model.py:979-1018` (modify greedy function)

- [ ] **Step 1: Add `DEFAULT_MIN_PHYSICAL_DISTANCES` constant**

In `bo_optimization/contextual_bo_model.py`, add after line 16 (`warnings.filterwarnings("ignore")`):

```python
# 空间填充：默认最小物理距离（per-dimension 半宽）
DEFAULT_MIN_PHYSICAL_DISTANCES = {
    'Process_Temp': 30,   # ±30°C
    'Process_Time': 3,    # ±3h
    'Process_H2': 15,     # ±15sccm
    'Process_Ar': 50,     # ±50sccm
}
```

- [ ] **Step 2: Modify `_greedy_select_mixed_score` signature and body**

Replace the entire `_greedy_select_mixed_score` method (lines 979-1018) with:

```python
    def _greedy_select_mixed_score(self, X_candidates, ei_values, X_train_proc, n_select, alpha=0.5, min_physical_distances=None):
        """
        贪心选择：混合 EI + maximin distance，每选一个点后动态更新距离基准

        Args:
            X_candidates: (n_cand, d_proc) 候选点（Process_ 空间）
            ei_values: (n_cand,) 每个候选点的 EI 值
            X_train_proc: (n_train, d_proc) 已训练样本的 Process_ 特征
            n_select: 需要选择的点数
            alpha: EI 权重（0~1），(1-alpha) 为 distance 权重
            min_physical_distances: dict 或 None，per-dimension 最小物理距离

        Returns:
            selected_indices: 选中的候选点索引
            selected_points: 选中的候选点坐标
        """
        remaining = list(range(len(X_candidates)))
        X_virtual = X_train_proc.copy()
        selected_indices = []

        # 预计算：固定边界归一化（循环外只算一次）
        if min_physical_distances is not None:
            bounds_min = np.array([self.bounds[col][0] for col in self.process_cols])
            bounds_range = np.array([self.bounds[col][1] - self.bounds[col][0] for col in self.process_cols])
            norm_thresholds = np.array([
                min_physical_distances.get(col, 0) / bounds_range[i]
                for i, col in enumerate(self.process_cols)
            ])
            X_cand_norm = (X_candidates - bounds_min) / bounds_range

        for _ in range(min(n_select, len(remaining))):
            # 过滤最小距离约束
            if min_physical_distances is not None and remaining:
                if len(X_virtual) == 0:
                    pass  # 冷启动：无历史点，跳过过滤
                else:
                    X_ref_norm = (X_virtual - bounds_min) / bounds_range
                    feasible = []
                    for global_idx in remaining:
                        c = X_cand_norm[global_idx]
                        diffs = np.abs(X_ref_norm - c)
                        in_exclusion = np.all(diffs < norm_thresholds, axis=1)
                        if not np.any(in_exclusion):
                            feasible.append(global_idx)
                    if not feasible:
                        # 兜底：原地缩小阈值 50%，后续迭代继续生效
                        norm_thresholds *= 0.5
                        for global_idx in remaining:
                            c = X_cand_norm[global_idx]
                            diffs = np.abs(X_ref_norm - c)
                            in_exclusion = np.all(diffs < norm_thresholds, axis=1)
                            if not np.any(in_exclusion):
                                feasible.append(global_idx)
                        if feasible:
                            warnings.warn("[空间填充] 候选空间耗尽，阈值已缩小 50%，后续迭代沿用宽松阈值")
                        else:
                            warnings.warn(f"[空间填充] 即使缩小阈值仍无可用候选，已选 {len(selected_indices)} 个")
                            break
                    remaining = feasible

            if not remaining:
                break

            # 计算当前候选点到虚拟参考集的 maximin distance
            X_cand_remaining = X_candidates[remaining]
            d_min = self._maximin_distance(X_cand_remaining, X_virtual, self.process_cols)

            # 计算 rank 百分位
            ei_remaining = ei_values[remaining]
            ei_ranks = np.argsort(np.argsort(ei_remaining)) / max(len(ei_remaining) - 1, 1)
            dist_ranks = np.argsort(np.argsort(d_min)) / max(len(d_min) - 1, 1)

            # 混合得分
            scores = alpha * ei_ranks + (1 - alpha) * dist_ranks
            best_local_idx = np.argmax(scores)
            best_global_idx = remaining[best_local_idx]

            # 选中，虚拟加入参考集
            selected_indices.append(best_global_idx)
            X_virtual = np.vstack([X_virtual, X_candidates[best_global_idx:best_global_idx + 1]])
            remaining.pop(best_local_idx)

        return np.array(selected_indices), X_candidates[selected_indices]
```

- [ ] **Step 3: Syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "feat: add min physical distance constraint to greedy selection"
```

---

### Task 2: Pass `min_physical_distances` through `suggest_space_filling`

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py:1020-1133` (suggest_space_filling signature + call)

- [ ] **Step 1: Modify `suggest_space_filling` signature**

Change line 1020 from:
```python
    def suggest_space_filling(self, n_total_points, alpha=0.5, n_candidates_per_cluster=1000):
```
to:
```python
    def suggest_space_filling(self, n_total_points, alpha=0.5, n_candidates_per_cluster=1000, min_physical_distances=None):
```

- [ ] **Step 2: Pass to `_greedy_select_mixed_score` call**

Change lines 1131-1133 from:
```python
            selected_idx, selected_points = self._greedy_select_mixed_score(
                X_feasible, ei_vals, X_train_proc, n_select, alpha=alpha
            )
```
to:
```python
            selected_idx, selected_points = self._greedy_select_mixed_score(
                X_feasible, ei_vals, X_train_proc, n_select, alpha=alpha,
                min_physical_distances=min_physical_distances
            )
```

- [ ] **Step 3: Syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "feat: pass min_physical_distances through suggest_space_filling"
```

---

### Task 3: Extend state file and add constants in `space_filling_plan.py`

**Files:**
- Modify: `bo_optimization/space_filling_plan.py:32-35` (imports)
- Modify: `bo_optimization/space_filling_plan.py:63-80` (_init_state)

- [ ] **Step 1: Add imports and constant**

Change lines 32-35 from:
```python
from bo_optimization.contextual_bo_model import (
    ContextualBayesianOptimizer, select_scheme, SCHEME_TARGETS, SCHEME_NAMES,
    DEFAULT_PROCESS_BOUNDS, PROCESS_LABELS_CN
)
```
to:
```python
from bo_optimization.contextual_bo_model import (
    ContextualBayesianOptimizer, select_scheme, SCHEME_TARGETS, SCHEME_NAMES,
    DEFAULT_PROCESS_BOUNDS, PROCESS_LABELS_CN, DEFAULT_MIN_PHYSICAL_DISTANCES
)

SPARSITY_DECAY = 0.5  # 每轮间距缩小比例
```

- [ ] **Step 2: Modify `_init_state` to include round/sparsity fields**

Replace lines 63-80 (`_init_state` function) with:

```python
def _init_state(scheme, recommendations, alpha, round=1, sparsity_multiplier=1.0):
    """首次运行：从 suggest_space_filling 结果初始化状态"""
    points = []
    for rec in recommendations:
        point = {k: v for k, v in rec.items() if k.startswith('Process_')}
        point['cluster'] = rec.get('cluster', 0)
        point['ei'] = rec.get('ei', 0.0)
        point['round'] = round
        points.append(point)

    state = {
        'scheme': scheme,
        'created_at': datetime.now().isoformat(),
        'round': round,
        'sparsity_multiplier': sparsity_multiplier,
        'alpha': alpha,
        'points': points,
        'allocations': [],  # [{index, sample_name, assigned_at}]
    }
    _save_state(scheme, state)
    return state
```

- [ ] **Step 3: Syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('bo_optimization/space_filling_plan.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add bo_optimization/space_filling_plan.py
git commit -m "feat: extend state file with round/sparsity tracking"
```

---

### Task 4: Modify `run_space_filling_plan` for min-distance and multi-round

**Files:**
- Modify: `bo_optimization/space_filling_plan.py:187-239` (run_space_filling_plan)

- [ ] **Step 1: Modify initialization block (lines 209-218)**

Replace lines 209-218 with:

```python
    # 4. 加载或初始化状态
    state, err = _load_state(scheme)
    if state is None:
        print(f"\n[状态] {err}，需要初始化空间填充计划")
        n_total = int(input("计划生成多少个空间填充采样点？(默认 10): ").strip() or "10")

        # 计算当前轮次的间距约束
        sparsity_multiplier = 1.0
        effective_dist = {k: v * sparsity_multiplier for k, v in DEFAULT_MIN_PHYSICAL_DISTANCES.items()}
        print(f"\n正在生成 {n_total} 个空间填充采样点（间距约束: Temp±{effective_dist['Process_Temp']:.0f}°C, "
              f"Time±{effective_dist['Process_Time']:.0f}h, H₂±{effective_dist['Process_H2']:.0f}sccm, "
              f"Ar±{effective_dist['Process_Ar']:.0f}sccm）...")
        recommendations = optimizer.suggest_space_filling(
            n_total_points=n_total, alpha=0.5, min_physical_distances=effective_dist
        )
        print(f"实际选出: {len(recommendations)} 个采样点")
        state = _init_state(scheme, recommendations, alpha=0.5, round=1, sparsity_multiplier=sparsity_multiplier)
        print(f"状态已保存至: {_get_state_path(scheme)}")
```

- [ ] **Step 2: Modify "all allocated" block (lines 223-239)**

Replace lines 223-239 with:

```python
    if n_remaining == 0:
        round_num = state.get('round', 1)
        sparsity_multiplier = state.get('sparsity_multiplier', 1.0)
        print(f"\n第 {round_num} 轮采样点已全部分配完毕！（当前间距倍率: {sparsity_multiplier}）")

        # 检查是否还能继续缩小间距
        next_multiplier = sparsity_multiplier * SPARSITY_DECAY
        if next_multiplier < 0.1:
            print("间距倍率已低于 0.1，不再继续细化。退出。")
            return

        choice = input(f"是否进入第 {round_num + 1} 轮（间距 ×{next_multiplier}）？(y/n): ").strip().lower()
        if choice == 'y':
            n_new = int(input("新一批采样点数 (默认 10): ").strip() or "10")
            sparsity_multiplier = next_multiplier
            effective_dist = {k: v * sparsity_multiplier for k, v in DEFAULT_MIN_PHYSICAL_DISTANCES.items()}
            print(f"\n正在生成 {n_new} 个新的空间填充采样点（间距约束: Temp±{effective_dist['Process_Temp']:.0f}°C, "
                  f"Time±{effective_dist['Process_Time']:.0f}h, H₂±{effective_dist['Process_H2']:.0f}sccm, "
                  f"Ar±{effective_dist['Process_Ar']:.0f}sccm）...")
            recommendations = optimizer.suggest_space_filling(
                n_total_points=n_new, alpha=0.5, min_physical_distances=effective_dist
            )
            print(f"实际选出: {len(recommendations)} 个采样点")
            for rec in recommendations:
                point = {k: v for k, v in rec.items() if k.startswith('Process_')}
                point['cluster'] = rec.get('cluster', 0)
                point['ei'] = rec.get('ei', 0.0)
                point['round'] = round_num + 1
                state['points'].append(point)
            state['round'] = round_num + 1
            state['sparsity_multiplier'] = sparsity_multiplier
            _save_state(scheme, state)
            n_remaining = _print_state_summary(state)
        else:
            print("退出。")
            return
```

- [ ] **Step 3: Syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('bo_optimization/space_filling_plan.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add bo_optimization/space_filling_plan.py
git commit -m "feat: add multi-round adaptive sampling with min-distance constraints"
```

---

### Task 5: Verify end-to-end

**Files:**
- Test: `bo_optimization/contextual_bo_model.py`
- Test: `bo_optimization/space_filling_plan.py`

- [ ] **Step 1: Syntax check both files**

Run:
```bash
python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py', encoding='utf-8').read()); print('contextual_bo_model OK')"
python -c "import ast; ast.parse(open('bo_optimization/space_filling_plan.py', encoding='utf-8').read()); print('space_filling_plan OK')"
```
Expected: both print `OK`

- [ ] **Step 2: Import check**

Run:
```bash
python -c "from bo_optimization.contextual_bo_model import DEFAULT_MIN_PHYSICAL_DISTANCES; print(DEFAULT_MIN_PHYSICAL_DISTANCES)"
```
Expected: `{'Process_Temp': 30, 'Process_Time': 3, 'Process_H2': 15, 'Process_Ar': 50}`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: space-filling min-distance constraint with multi-round adaptive sampling"
```
