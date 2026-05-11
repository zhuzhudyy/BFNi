# 空间填充实验设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a space-filling sampling strategy to `ContextualBayesianOptimizer` that uses LHS candidate pools, maximin distance in normalized space, and greedy mixed-score selection (EI + distance) to improve global surrogate model accuracy.

**Architecture:** All new code goes into `bo_optimization/contextual_bo_model.py` as new methods on `ContextualBayesianOptimizer`. No new files needed. The space-filling method clusters Pre_ features, generates LHS candidates per cluster with rejection sampling for Ar≥2H₂, and selects points via a greedy loop that dynamically updates distance baselines.

**Tech Stack:** Python, numpy, scipy (LHS via `scipy.stats.qmc`), sklearn (KMeans, silhouette_score, MinMaxScaler)

---

### Task 1: Add `_maximin_distance` helper method

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py` (after `extract_ard_importance`, ~line 945)

This method computes the maximin distance from each candidate point to a reference set, in Min-Max normalized space.

- [ ] **Step 1: Add the method**

Add after `extract_ard_importance` method (before the `add_new_data_to_training` function):

```python
def _maximin_distance(self, X_candidates, X_reference, feature_cols):
    """
    计算候选点到参考集的 maximin distance（归一化空间）

    Args:
        X_candidates: (n_candidates, d) 候选点矩阵
        X_reference: (n_ref, d) 参考点矩阵（已训练样本 + 已选点）
        feature_cols: 特征列名列表，用于确定归一化范围

    Returns:
        d_min: (n_candidates,) 每个候选点到参考集的最小距离
    """
    from sklearn.preprocessing import MinMaxScaler

    # 合并候选点和参考点，统一归一化
    X_all = np.vstack([X_candidates, X_reference])
    scaler = MinMaxScaler()
    X_all_norm = scaler.fit_transform(X_all)

    X_cand_norm = X_all_norm[:len(X_candidates)]
    X_ref_norm = X_all_norm[len(X_candidates):]

    # 计算欧氏距离矩阵 (n_candidates, n_ref)
    # 使用 ||a-b||² = ||a||² + ||b||² - 2a·b
    sq_cand = np.sum(X_cand_norm ** 2, axis=1, keepdims=True)
    sq_ref = np.sum(X_ref_norm ** 2, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(sq_cand + sq_ref.T - 2 * X_cand_norm @ X_ref_norm.T, 0))

    # 每个候选点到参考集的最小距离
    d_min = dists.min(axis=1)
    return d_min
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "feat: add maximin distance helper for space-filling design"
```

---

### Task 2: Add `_greedy_select_mixed_score` method

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py` (after `_maximin_distance`)

This method implements the greedy selection loop: pick top-1, virtual-add to X_train, recompute distances, repeat.

- [ ] **Step 1: Add the method**

```python
def _greedy_select_mixed_score(self, X_candidates, ei_values, X_train_proc, n_select, alpha=0.5):
    """
    贪心选择：混合 EI + maximin distance，每选一个点后动态更新距离基准

    Args:
        X_candidates: (n_cand, d_proc) 候选点（Process_ 空间）
        ei_values: (n_cand,) 每个候选点的 EI 值
        X_train_proc: (n_train, d_proc) 已训练样本的 Process_ 特征
        n_select: 需要选择的点数
        alpha: EI 权重（0~1），(1-alpha) 为 distance 权重

    Returns:
        selected_indices: 选中的候选点索引
        selected_points: 选中的候选点坐标
    """
    remaining = list(range(len(X_candidates)))
    X_virtual = X_train_proc.copy()
    selected_indices = []

    for _ in range(min(n_select, len(remaining))):
        # 计算当前候选点到虚拟参考集的 maximin distance
        X_cand_remaining = X_candidates[remaining]
        d_min = self._maximin_distance(X_cand_remaining, X_virtual,
                                        self.process_cols[:X_candidates.shape[1]])

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

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "feat: add greedy mixed-score selection with dynamic distance update"
```

---

### Task 3: Add `suggest_space_filling` method

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py` (after `_greedy_select_mixed_score`)

This is the main entry point. It clusters Pre_ features, generates LHS candidates per cluster with rejection sampling, computes EI, and calls the greedy selector.

- [ ] **Step 1: Add the method**

```python
def suggest_space_filling(self, n_total_points, alpha=0.5, n_candidates_per_cluster=1000):
    """
    空间填充采样建议：混合 EI + maximin distance

    Args:
        n_total_points: 总共需要推荐的采样点数
        alpha: EI 权重（0~1），默认 0.5
        n_candidates_per_cluster: 每个 cluster 的 LHS 候选点数

    Returns:
        recommendations: list of dict，每个包含 process params + cluster info
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.stats import qmc

    if not hasattr(self, 'training_df') or self.training_df is None:
        raise ValueError("请先调用 train() 训练模型")

    # --- 1. 聚类 Pre_ 特征 ---
    X_pre_train = self.training_df[self.pre_feature_cols].values

    if len(X_pre_train) < 15:
        k_best = 1
        labels = np.zeros(len(X_pre_train), dtype=int)
        best_score = 0.0
        print(f"\n[空间填充] 样本量 {len(X_pre_train)} < 15，跳过聚类（单一 cluster）")
    else:
        best_score = -1
        k_best = 3
        labels = np.zeros(len(X_pre_train), dtype=int)
        for k in [3, 4, 5]:
            if k >= len(X_pre_train):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_k = km.fit_predict(X_pre_train)
            score = silhouette_score(X_pre_train, labels_k)
            if score > best_score:
                best_score = score
                k_best = k
                labels = labels_k
        print(f"\n[空间填充] 最优聚类 k={k_best}（silhouette={best_score:.3f}）")

    # --- 2. 按 cluster 大小比例分配采样点数 ---
    n_per_cluster = {}
    for c in range(k_best):
        count = int(np.sum(labels == c))
        n_per_cluster[c] = max(1, round(n_total_points * count / len(X_pre_train)))
    # 调整总数（round 可能导致偏差）
    total_assigned = sum(n_per_cluster.values())
    if total_assigned != n_total_points:
        max_c = max(n_per_cluster, key=n_per_cluster.get)
        n_per_cluster[max_c] += n_total_points - total_assigned

    print(f"[空间填充] 各 cluster 分配: {n_per_cluster}")

    # --- 3. 获取训练数据的 Process_ 特征（用于距离计算）---
    X_train_proc = self.training_df[self.process_cols].values

    # --- 4. 对每个 cluster 生成候选点并选择 ---
    h2_idx = self.process_cols.index('Process_H2')
    ar_idx = self.process_cols.index('Process_Ar')
    all_recommendations = []

    for c in range(k_best):
        n_select = n_per_cluster[c]
        if n_select <= 0:
            continue

        # LHS 生成候选点（Process_ 空间）
        d_proc = len(self.process_cols)
        sampler = qmc.LatinHypercube(d=d_proc, seed=42 + c)
        X_lhs_unit = sampler.random(n=n_candidates_per_cluster)

        # 缩放到物理范围
        bounds_array = np.array([[self.bounds[col][0], self.bounds[col][1]]
                                  for col in self.process_cols])
        X_lhs = qmc.scale(X_lhs_unit, bounds_array[:, 0], bounds_array[:, 1])

        # 拒绝采样：Ar >= 2 * H2
        feasible_mask = X_lhs[:, ar_idx] >= 2 * X_lhs[:, h2_idx]
        X_feasible = X_lhs[feasible_mask]
        print(f"  Cluster {c}: LHS {n_candidates_per_cluster} → 可行 {len(X_feasible)} "
              f"（拒绝率 {1 - feasible_mask.mean():.1%}）")

        if len(X_feasible) < n_select:
            print(f"  [警告] 可行点不足，仅选 {len(X_feasible)} 个")
            n_select = len(X_feasible)

        # 计算 EI
        # 使用 cluster 中心的 Pre_ 特征作为代表
        cluster_mask = labels == c
        x_pre_center = X_pre_train[cluster_mask].mean(axis=0)

        # 使用该 cluster 内样本的 y_best
        cluster_yields = self.training_df.loc[cluster_mask, 'TARGET_Yield'].values
        y_best_cluster = np.percentile(cluster_yields, 95) if len(cluster_yields) > 0 else 0.5

        # 目标晶向：使用训练数据中出现的目标
        target_onehot = np.zeros(len(self.target_cols))
        if self.target_cols:
            # 用训练数据中最常见的目标组合
            for col in self.target_cols:
                if col in self.training_df.columns:
                    if self.training_df[col].mean() > 0.5:
                        target_onehot[self.target_cols.index(col)] = 1.0

        ei_vals, _, _ = self.expected_improvement(
            X_feasible, x_pre_center, y_best_cluster, target_onehot
        )

        # 贪心选择
        selected_idx, selected_points = self._greedy_select_mixed_score(
            X_feasible, ei_vals, X_train_proc, n_select, alpha=alpha
        )

        for idx, point in zip(selected_idx, selected_points):
            rec = {col: point[i] for i, col in enumerate(self.process_cols)}
            rec['cluster'] = c
            rec['ei'] = ei_vals[idx]
            all_recommendations.append(rec)

        # 将选中的点加入虚拟参考集（跨 cluster 也要保持距离）
        X_train_proc = np.vstack([X_train_proc, selected_points])

    # --- 5. 输出 ---
    print(f"\n[空间填充] 共推荐 {len(all_recommendations)} 个采样点:")
    for i, rec in enumerate(all_recommendations):
        proc_str = ", ".join([f"{k}={v:.1f}" for k, v in rec.items()
                              if k.startswith('Process_')])
        print(f"  #{i+1} [Cluster {rec['cluster']}] {proc_str}  (EI={rec['ei']:.6f})")

    return all_recommendations
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/contextual_bo_model.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "feat: add suggest_space_filling with LHS + rejection sampling + greedy mixed-score"
```

---

### Task 4: Test with existing training data

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py` (`__main__` block)

Add a quick integration test in the main block to verify the space-filling method works end-to-end.

- [ ] **Step 1: Add test call in `__main__`**

Add after the `optimizer.train(data_file)` line and before the final print statements:

```python
    # 测试空间填充采样
    print("\n" + "="*60)
    print("测试空间填充采样建议:")
    try:
        recommendations = optimizer.suggest_space_filling(n_total_points=8, alpha=0.5)
    except Exception as e:
        print(f"空间填充测试失败: {e}")
```

- [ ] **Step 2: Run the script with scheme 1**

Run: `cd "C:/学习/超滑所/毕设/优化模拟/bo_optimization" && source ../.venv/Scripts/activate && echo "1" | python contextual_bo_model.py`
Expected: Script trains model, then prints space-filling recommendations with cluster assignments and EI values. No errors.

- [ ] **Step 3: Fix any runtime errors**

If there are errors (e.g., missing scipy.stats.qmc, wrong column names), fix them.

- [ ] **Step 4: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "test: add space-filling integration test in main block"
```

---

### Task 5: Clean up test code and finalize

**Files:**
- Modify: `bo_optimization/contextual_bo_model.py`

Remove or comment out the test code from `__main__`, keep the method.

- [ ] **Step 1: Comment out the test block**

Change the test block to a comment showing usage:

```python
    # 空间填充采样用法:
    # recommendations = optimizer.suggest_space_filling(n_total_points=8, alpha=0.5)
```

- [ ] **Step 2: Commit**

```bash
git add bo_optimization/contextual_bo_model.py
git commit -m "chore: clean up space-filling test code, keep method"
```
