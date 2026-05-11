# Model Health Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a 2×2 dashboard figure showing model accuracy, Process_ space coverage, variance decomposition, and coverage metrics — run once after each experiment batch.

**Architecture:** Single new file `bo_optimization/model_health_dashboard.py` with 4 plot functions + 1 assembly function + `__main__` block. Imports `ContextualBayesianOptimizer` for model access. Follows `honest_visualization.py` patterns for output directory and font config.

**Tech Stack:** matplotlib, numpy, pandas, sklearn (KMeans, KFold, MinMaxScaler)

---

### Task 1: Create file skeleton + panel ③ (variance pie chart)

Start with the simplest panel to establish the file structure.

**Files:**
- Create: `bo_optimization/model_health_dashboard.py`

- [ ] **Step 1: Create the file with imports, font config, and `plot_variance_pie`**

```python
"""
模型健康状态仪表盘

每次新实验完成后运行一次，查看模型全局健康状态。
输出 2×2 仪表盘：RMSE 学习曲线 | Process_ pairplot | 方差成分饼图 | 覆盖率指标

使用方法:
    python bo_optimization/model_health_dashboard.py
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from bo_optimization.contextual_bo_model import (
    ContextualBayesianOptimizer, SCHEME_NAMES, DEFAULT_PROCESS_BOUNDS,
    PROCESS_LABELS_CN, select_scheme
)

# 颜色方案
COLORS = {
    'primary': '#377eb8',
    'secondary': '#e41a1c',
    'accent': '#ff7f00',
    'success': '#4daf4a',
    'gray': '#999999',
    'pre': '#1f77b4',
    'target': '#2ca02c',
    'proc': '#d62728',
    'inter': '#9467bd',
}


def create_output_dir():
    """创建输出目录"""
    if os.path.exists(r"D:\毕业设计\织构数据"):
        base = r"D:\毕业设计\织构数据\visualization\output"
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "output")
    os.makedirs(base, exist_ok=True)
    return base


def plot_variance_pie(ax, optimizer):
    """
    面板 ③：方差成分饼图

    从 ANOVA Matern 核提取 σ²_pre, σ²_target, σ²_proc, σ²_inter，
    绘制饼图展示各成分对总方差的贡献。
    """
    kernel = optimizer.gpr.kernel_
    var_pre = np.exp(kernel.log_var_pre) if hasattr(kernel, 'log_var_pre') else 0
    var_target = np.exp(kernel.log_var_target) if hasattr(kernel, 'log_var_target') else 0
    var_proc = np.exp(kernel.log_var_proc) if hasattr(kernel, 'log_var_proc') else 0
    var_inter = np.exp(kernel.log_var_inter) if hasattr(kernel, 'log_var_inter') else 0

    values = [var_pre, var_target, var_proc, var_inter]
    labels = ['初始状态\n(Pre_)', '目标晶向\n(Target_)', '工艺参数\n(Process_)', '交互效应\n(Pre_×Proc_)']
    colors = [COLORS['pre'], COLORS['target'], COLORS['proc'], COLORS['inter']]
    explode = (0, 0, 0, 0.08)  # 交互项稍微突出

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        textprops={'fontsize': 9}
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight('bold')

    total = sum(values)
    ax.set_title(f'方差成分分解 (σ²_total = {total:.4f})', fontsize=11, fontweight='bold')
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/model_health_dashboard.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/model_health_dashboard.py
git commit -m "feat: create model_health_dashboard.py with variance pie chart"
```

---

### Task 2: Implement panel ④ (coverage progress bar)

**Files:**
- Modify: `bo_optimization/model_health_dashboard.py`

- [ ] **Step 1: Add `plot_coverage_bar` function after `plot_variance_pie`**

```python
def plot_coverage_bar(ax, optimizer):
    """
    面板 ④：覆盖率指标（进度条 + 数字）

    调用 compute_space_coverage() 和 estimate_experiments_for_coverage()
    显示当前覆盖率、中位距离、达到 30% 还需多少实验。
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    coverage, median_dist, p90_dist = optimizer.compute_space_coverage()
    n_needed, _, curve = optimizer.estimate_experiments_for_coverage(target_coverage=0.30)
    n_samples = len(optimizer.training_df)

    # 进度条
    bar_y = 7.0
    bar_width = 8.0
    bar_height = 0.8
    bar_x = 1.0

    # 背景条
    ax.barh(bar_y, bar_width, bar_height, left=bar_x, color='#e0e0e0', edgecolor='none')
    # 填充条
    fill_width = bar_width * min(coverage, 1.0)
    bar_color = COLORS['success'] if coverage >= 0.30 else (COLORS['accent'] if coverage >= 0.15 else COLORS['secondary'])
    ax.barh(bar_y, fill_width, bar_height, left=bar_x, color=bar_color, edgecolor='none')
    # 百分比文字
    ax.text(bar_x + bar_width / 2, bar_y, f'{coverage:.1%}', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white' if coverage > 0.1 else 'black')

    # 标题
    ax.text(5.0, 9.2, 'Process_ 空间覆盖率', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # 指标文字
    info_lines = [
        f'当前样本: {n_samples}',
        f'中位距离: {median_dist:.3f} (归一化)',
        f'P90 距离: {p90_dist:.3f}',
        f'达 30% 还需: ~{max(0, n_needed - n_samples)} 组实验',
    ]
    for i, line in enumerate(info_lines):
        ax.text(5.0, 5.2 - i * 0.9, line, ha='center', va='center', fontsize=10, color='#333333')

    # 状态标签
    if coverage >= 0.30:
        status = '✓ 覆盖率良好'
        status_color = COLORS['success']
    elif coverage >= 0.15:
        status = '△ 覆盖率中等'
        status_color = COLORS['accent']
    else:
        status = '✗ 覆盖率不足'
        status_color = COLORS['secondary']
    ax.text(5.0, 1.0, status, ha='center', va='center', fontsize=12,
            fontweight='bold', color=status_color)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/model_health_dashboard.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/model_health_dashboard.py
git commit -m "feat: add coverage progress bar panel"
```

---

### Task 3: Implement panel ① (RMSE learning curve with K-Fold CV)

**Files:**
- Modify: `bo_optimization/model_health_dashboard.py`

- [ ] **Step 1: Add `plot_rmse_curve` function after `plot_coverage_bar`**

```python
def plot_rmse_curve(ax, optimizer, n_repeats=3):
    """
    面板 ①：RMSE vs 数据量（K-Fold CV 学习曲线）

    在 20, 40, 60, 全量样本上分别执行 5-Fold CV，
    绘制 RMSE 均值 ± 标准差曲线。
    使用 3 次多起点 + warm start 加速。
    """
    from sklearn.model_selection import KFold

    df = optimizer.training_df
    pre_cols = optimizer.pre_feature_cols
    target_cols = optimizer.target_cols
    process_cols = optimizer.process_cols
    all_cols = pre_cols + target_cols + process_cols

    X_full = df[all_cols].values
    y_full = df['TARGET_Yield'].values
    n_full = len(df)

    # 数据量梯度
    sizes = sorted(set([20, 40, 60, n_full]))
    sizes = [s for s in sizes if s <= n_full]

    # 保存全量模型的超参数作为 warm start
    warm_theta = optimizer.gpr.kernel_.theta.copy() if hasattr(optimizer.gpr, 'kernel_') else None

    results = {}  # size -> list of rmse values

    for n_sub in sizes:
        rmses = []
        for rep in range(n_repeats):
            # 子采样
            rng = np.random.RandomState(rep * 100 + n_sub)
            if n_sub < n_full:
                idx = rng.choice(n_full, size=n_sub, replace=False)
            else:
                idx = np.arange(n_full)

            X_sub = X_full[idx]
            y_sub = y_full[idx]

            # 5-Fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=rep)
            fold_rmses = []

            for train_idx, test_idx in kf.split(X_sub):
                X_train, X_test = X_sub[train_idx], X_sub[test_idx]
                y_train, y_test = y_sub[train_idx], y_sub[test_idx]

                # 训练 GPR（3 次多起点 + warm start）
                from bo_optimization.contextual_bo_model import ANOVAMaternKernel, GPRWithPriors
                kernel = ANOVAMaternKernel(len(pre_cols), len(target_cols), len(process_cols))
                gpr = GPRWithPriors(
                    kernel=kernel, alpha=1e-10, normalize_y=True,
                    inter_var_prior=(1.5, 1.0), num_restarts=3, random_state=42
                )

                # Warm start
                if warm_theta is not None and len(warm_theta) == len(kernel.theta):
                    try:
                        kernel.theta = warm_theta
                    except Exception:
                        pass

                try:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    gpr.fit(X_train_s, y_train)
                    y_pred = gpr.predict(X_test_s)
                    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
                    fold_rmses.append(rmse)
                except Exception:
                    continue

            if fold_rmses:
                rmses.append(np.mean(fold_rmses))

        results[n_sub] = rmses

    # 绘制
    x_vals = []
    y_means = []
    y_stds = []

    for size in sizes:
        rmses = results.get(size, [])
        if rmses:
            x_vals.append(size)
            y_means.append(np.mean(rmses))
            y_stds.append(np.std(rmses) if len(rmses) > 1 else 0)

    if not x_vals:
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
        return

    y_means = np.array(y_means)
    y_stds = np.array(y_stds)

    ax.plot(x_vals, y_means, '-o', color=COLORS['primary'], linewidth=2, markersize=6, label='5-Fold CV RMSE')
    ax.fill_between(x_vals, y_means - y_stds, y_means + y_stds,
                     alpha=0.2, color=COLORS['primary'])

    # 标注全量 LOOCV（如果可用）
    if n_full in results and results[n_full]:
        ax.plot(n_full, np.mean(results[n_full]), '*', color=COLORS['secondary'],
                markersize=15, label=f'全量 RMSE = {np.mean(results[n_full]):.4f}', zorder=5)

    ax.set_xlabel('训练集大小', fontsize=10)
    ax.set_ylabel('RMSE', fontsize=10)
    ax.set_title('模型精度 vs 数据量', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/model_health_dashboard.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/model_health_dashboard.py
git commit -m "feat: add RMSE learning curve panel (K-Fold CV)"
```

---

### Task 4: Implement panel ② (Process_ pairplot)

**Files:**
- Modify: `bo_optimization/model_health_dashboard.py`

- [ ] **Step 1: Add `plot_process_pairplot` function after `plot_rmse_curve`**

```python
def plot_process_pairplot(ax_grid, optimizer, recommendations=None):
    """
    面板 ②：Process_ 空间 Pairplot（2×3 网格）

    展示 Process_ 4 维的所有两两组合。
    蓝色点：训练数据。橙色三角：LHS 推荐点。
    在 H₂×Ar 子图中画约束线 Ar = 2×H₂。

    Args:
        ax_grid: 2×3 的 Axes 数组
        optimizer: 训练好的 ContextualBayesianOptimizer
        recommendations: suggest_space_filling() 的返回值（可选）
    """
    proc_cols = optimizer.process_cols
    n_proc = len(proc_cols)
    X_train = optimizer.training_df[proc_cols].values

    # 生成所有两两组合
    pairs = []
    for i in range(n_proc):
        for j in range(i + 1, n_proc):
            pairs.append((i, j))

    # 提取 LHS 推荐点
    X_lhs = None
    if recommendations:
        X_lhs = np.array([[r[col] for col in proc_cols] for r in recommendations])

    # 约束信息
    h2_idx = proc_cols.index('Process_H2') if 'Process_H2' in proc_cols else None
    ar_idx = proc_cols.index('Process_Ar') if 'Process_Ar' in proc_cols else None

    for idx, (i, j) in enumerate(pairs):
        row, col = divmod(idx, 3)
        ax = ax_grid[row, col]

        # 训练数据
        ax.scatter(X_train[:, j], X_train[:, i], c=COLORS['primary'],
                   s=20, alpha=0.5, label='训练数据', zorder=2)

        # LHS 推荐点
        if X_lhs is not None:
            ax.scatter(X_lhs[:, j], X_lhs[:, i], c=COLORS['accent'],
                       s=80, marker='^', edgecolors='red', linewidths=1.5,
                       label='LHS 推荐', zorder=3)

        # 约束线 (H₂ × Ar)
        if h2_idx is not None and ar_idx is not None:
            if i == h2_idx and j == ar_idx:
                x_range = np.linspace(optimizer.bounds[proc_cols[j]][0],
                                       optimizer.bounds[proc_cols[j]][1], 100)
                ax.plot(x_range, x_range / 2, '--', color=COLORS['gray'],
                        linewidth=1, label='Ar = 2×H₂')
                ax.fill_between(x_range, 0, x_range / 2, alpha=0.05, color='red')

        # 标签（去掉 Process_ 前缀，缩短显示）
        xlabel = proc_cols[j].replace('Process_', '')
        ylabel = proc_cols[i].replace('Process_', '')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

        # 相关系数
        corr = np.corrcoef(X_train[:, i], X_train[:, j])[0, 1]
        ax.set_title(f'r = {corr:.2f}', fontsize=8, color=COLORS['gray'])

        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    # 隐藏多余的子图
    for idx in range(len(pairs), 6):
        row, col = divmod(idx, 3)
        ax_grid[row, col].axis('off')
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/model_health_dashboard.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/model_health_dashboard.py
git commit -m "feat: add Process_ pairplot panel with constraint line"
```

---

### Task 5: Add dashboard assembly + `__main__` block

**Files:**
- Modify: `bo_optimization/model_health_dashboard.py`

- [ ] **Step 1: Add `create_dashboard` function and `__main__` block**

```python
def create_dashboard(optimizer, recommendations=None, save_path=None):
    """
    组装 2×2 仪表盘

    Args:
        optimizer: 训练好的 ContextualBayesianOptimizer
        recommendations: suggest_space_filling() 的返回值（可选）
        save_path: 保存路径（None 则自动生成）
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 面板 ①：RMSE 学习曲线
    ax1 = fig.add_subplot(gs[0, 0])
    print("[*] 计算 RMSE 学习曲线（K-Fold CV）...")
    plot_rmse_curve(ax1, optimizer)

    # 面板 ②：Process_ pairplot
    gs_right = gs[0, 1].subgridspec(2, 3, hspace=0.4, wspace=0.35)
    ax_pair = np.array([[fig.add_subplot(gs_right[r, c]) for c in range(3)] for r in range(2)])
    print("[*] 绘制 Process_ pairplot...")
    plot_process_pairplot(ax_pair, optimizer, recommendations)

    # 面板 ③：方差饼图
    ax3 = fig.add_subplot(gs[1, 0])
    plot_variance_pie(ax3, optimizer)

    # 面板 ④：覆盖率指标
    ax4 = fig.add_subplot(gs[1, 1])
    plot_coverage_bar(ax4, optimizer)

    # 总标题
    n_samples = len(optimizer.training_df)
    fig.suptitle(f'模型健康状态仪表盘 (N = {n_samples})', fontsize=14, fontweight='bold', y=0.98)

    # 保存
    if save_path is None:
        output_dir = create_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f'model_health_{timestamp}.png')

    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[*] 仪表盘已保存至: {save_path}")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    scheme = select_scheme()

    # 加载模型
    model_path = f"trained_models/model_scheme{scheme}.pkl"
    if not os.path.exists(model_path):
        model_path = "trained_models/model_scheme1.pkl"

    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    if not optimizer.load_model(model_path):
        data_file = f"Optimized_Training_Data_方案{scheme}.csv"
        if not os.path.exists(data_file):
            data_file = "Optimized_Training_Data.csv"
        optimizer.train(data_file)

    # 可选：生成 LHS 推荐点
    print("\n是否生成空间填充推荐？(y/n): ", end='')
    do_lhs = input().strip().lower() == 'y'
    recommendations = None
    if do_lhs:
        n_points = int(input("推荐点数 (默认 6): ").strip() or "6")
        recommendations = optimizer.suggest_space_filling(n_total_points=n_points)

    # 生成仪表盘
    save_path = create_dashboard(optimizer, recommendations=recommendations)
    print(f"\n完成！仪表盘: {save_path}")
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bo_optimization/model_health_dashboard.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bo_optimization/model_health_dashboard.py
git commit -m "feat: add dashboard assembly and main block"
```

---

### Task 6: End-to-end test

**Files:**
- Modify: `bo_optimization/model_health_dashboard.py` (if bugs found)

- [ ] **Step 1: Run the dashboard with scheme 1**

Run: `cd "C:/学习/超滑所/毕设/优化模拟/bo_optimization" && source ../.venv/Scripts/activate && echo -e "1\nn" | python model_health_dashboard.py`
Expected: Script trains/loads model, generates dashboard PNG, prints save path. No errors.

- [ ] **Step 2: Fix any runtime errors**

If there are errors (import failures, attribute errors, dimension mismatches), fix them.

- [ ] **Step 3: Verify the output PNG exists and is non-empty**

Run: `ls -la bo_optimization/visualization/output/model_health_*.png`
Expected: File exists with size > 0.

- [ ] **Step 4: Commit fixes (if any)**

```bash
git add bo_optimization/model_health_dashboard.py
git commit -m "fix: resolve runtime errors in model health dashboard"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the dashboard to the pipeline section**

Add after the `space_filling_plan.py` line:

```
python bo_optimization/model_health_dashboard.py  # 2x2 model health dashboard (RMSE, pairplot, variance, coverage)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add model_health_dashboard to CLAUDE.md pipeline"
```
