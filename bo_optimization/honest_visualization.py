"""
物理诚实的贝叶斯优化可视化工具 (Physically Honest Visualization)

替代 model_visualization.py 中固定 Pre_ 均值扫工艺参数的 3D 响应曲面，
提供基于实际数据分布和模型不确定性的可视化方法。

因果链: 工艺参数(Process) → 微观结构(Pre_) → 产率(Yield)
GP 模型将 Pre_ + Process_ 同时作为输入，固定 Pre_=均值违反了物理因果。

可视化套件:
    Tier 1 (核心替代):
        plot_yield_by_experiment   — 实验产率分组柱状图
        plot_partial_dependence    — 部分依赖图 (PDP): 在 Pre_ 分布上积分
        plot_raw_data_scatter      — 原始数据: 工艺参数 vs 产率散点
    Tier 2 (物理叙事):
        plot_pre_feature_space     — 微观结构特征空间: 分布与产率
        plot_ice_curves            — 个体条件期望 (ICE): 逐样本轨迹
        plot_ard_grouped           — ARD 分组对比: 工艺 vs 微观结构
    Tier 3 (辅助分析):
        plot_uncertainty_map       — 模型不确定性热力图
        plot_process_pre_mediation — 探索性: 工艺→微观结构因果链
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from bo_optimization.contextual_bo_model import (
    ContextualBayesianOptimizer, SCHEME_TARGETS, SCHEME_NAMES, DEFAULT_PROCESS_BOUNDS
)
from bo_optimization.model_visualization import create_output_dir

SCHEME_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
PROCESS_LABELS_CN = {
    'Process_Temp': '退火温度 (°C)',
    'Process_Time': '保温时间 (h)',
    'Process_H2': 'H$_2$ 流量 (sccm)',
    'Process_Ar': 'Ar 流量 (sccm)',
}


# ============================================================================
# Tier 1 — 核心替代
# ============================================================================

def plot_yield_by_experiment(optimizer, output_dir, target_scheme=None):
    """
    实验产率分组柱状图 — 最诚实的原始数据展示。
    每个实验的 4 个方案产率并排显示，按平均产率降序排列。
    """
    df = optimizer.training_df.copy()
    target_cols = optimizer.target_cols

    # 推断实验 ID：用 Pre_ 特征的行签名做分组
    pre_cols = optimizer.pre_feature_cols
    # 每个实验=4行（4方案），用 Pre_ 行的 tuple 做 key
    exp_keys = [tuple(row[pre_cols].values.round(3)) for _, row in df.iterrows()]
    unique_keys = list(dict.fromkeys(exp_keys))  # preserve order
    n_exp = len(unique_keys)

    # 为每个实验计算各方案的产率
    scheme_ids = list(range(1, 5))
    exp_data = {eid: {} for eid in range(n_exp)}
    exp_mean = np.zeros(n_exp)

    for eid, key in enumerate(unique_keys):
        rows = df[[k == key for k in exp_keys]]
        for sid in scheme_ids:
            target_keys = [f"Target_{h}{k}{l}" for h, k, l in SCHEME_TARGETS[sid]]
            mask = np.ones(len(rows), dtype=bool)
            for tk in target_keys:
                if tk in rows.columns:
                    mask &= rows[tk] == 1.0
            if mask.any():
                exp_data[eid][sid] = rows[mask]['TARGET_Yield'].values[0]
            else:
                exp_data[eid][sid] = np.nan
        exp_mean[eid] = np.nanmean(list(exp_data[eid].values()))

    # 按均值降序
    sort_order = np.argsort(-exp_mean)
    global_mean = df['TARGET_Yield'].mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(n_exp)
    bar_w = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for sidx, sid in enumerate(scheme_ids):
        vals = [exp_data[sort_order[i]].get(sid, 0) for i in range(n_exp)]
        ax.bar(x + offsets[sidx] * bar_w, vals, bar_w,
               color=SCHEME_COLORS[sidx], alpha=0.85,
               label=SCHEME_NAMES.get(sid, f'方案{sid}'))

    ax.axhline(global_mean, color='gray', linestyle='--', lw=1.2,
               label=f'全局均值 = {global_mean:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{sort_order[i]+1}' for i in range(n_exp)], fontsize=8)
    ax.set_xlabel('实验编号 (按平均产率降序排列)', fontsize=11)
    ax.set_ylabel('目标产率 TARGET_Yield', fontsize=11)
    ax.set_title('各实验在不同目标晶向方案下的产率', fontsize=14)
    ax.legend(fontsize=8, ncol=6, loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "1_Experimental_Yield_by_Sample.png")
    plt.savefig(save_path, dpi=300)
    print(f"[*] 实验产率柱状图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_partial_dependence(optimizer, output_dir, target_scheme=None,
                            n_grid=30, n_samples=500):
    """
    部分依赖图 (PDP) — 对每个工艺参数在 Pre_ 实际分布上积分。

    与固定 Pre_ 均值的做法不同，PDP 从训练数据的 Pre_ 行中随机抽取，
    每次预测同时变动目标工艺参数，取均值作为边际效应估计。
    """
    df = optimizer.training_df.copy()
    pre_cols = optimizer.pre_feature_cols
    process_cols = optimizer.process_cols
    target_cols_list = optimizer.target_cols

    # 目标编码
    if target_scheme is not None and target_cols_list:
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme]
        target_enc = np.array([1.0 if c in target_keys else 0.0 for c in target_cols_list])
        # 筛选同方案数据用于 Pre_ 采样
        mask = np.ones(len(df), dtype=bool)
        for tk in target_keys:
            if tk in df.columns:
                mask &= df[tk] == 1.0
        pre_pool = df.loc[mask, pre_cols].values
    else:
        target_enc = np.zeros(len(target_cols_list)) if target_cols_list else np.array([])
        pre_pool = df[pre_cols].values

    scheme_str = f" — {target_scheme}" if target_scheme else ""

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, col in enumerate(process_cols):
        lo, hi = optimizer.bounds[col]
        grid = np.linspace(lo, hi, n_grid)
        means = np.zeros(n_grid)
        stds = np.zeros(n_grid)

        for j, g in enumerate(grid):
            idxs = np.random.choice(len(pre_pool), size=n_samples, replace=True)
            pre_batch = pre_pool[idxs]
            # 其他工艺参数均匀采样
            proc_batch = np.zeros((n_samples, len(process_cols)))
            for k, pc in enumerate(process_cols):
                pl, ph = optimizer.bounds[pc]
                proc_batch[:, k] = np.random.uniform(pl, ph, n_samples)
            proc_batch[:, i] = g  # 锁死目标参数

            X_full = np.hstack([
                pre_batch,
                np.tile(target_enc, (n_samples, 1)),
                proc_batch
            ])
            Xs = optimizer.scaler_X.transform(X_full)
            mu, _ = optimizer.gpr.predict(Xs, return_std=True)
            means[j] = np.mean(mu)
            stds[j] = np.std(mu)

        axes[i].fill_between(grid, means - 2 * stds, means + 2 * stds,
                             alpha=0.25, color='steelblue')
        axes[i].plot(grid, means, color='steelblue', lw=2)
        axes[i].axhline(df['TARGET_Yield'].mean(), color='gray',
                        linestyle='--', lw=1, alpha=0.6, label='全局均值产率')
        axes[i].set_xlabel(PROCESS_LABELS_CN.get(col, col), fontsize=11)
        axes[i].set_ylabel('PDP 预测产率', fontsize=11)
        axes[i].set_title(f'{col.replace("Process_", "")}{scheme_str}', fontsize=12)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('部分依赖图 (PDP): 工艺参数在微观结构分布上的边际效应',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "2_Partial_Dependence_Plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 部分依赖图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_raw_data_scatter(optimizer, output_dir):
    """
    原始数据散点图 — 产率 vs 各工艺参数，按方案着色，叠加箱线图。
    不做模型插值，纯数据。
    """
    df = optimizer.training_df.copy()
    process_cols = optimizer.process_cols

    # 给每行分配方案 ID (列式显示)
    scheme_id = np.zeros(len(df), dtype=int)
    for sid in range(1, 5):
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in SCHEME_TARGETS[sid]]
        mask = np.ones(len(df), dtype=bool)
        for tk in target_keys:
            if tk in df.columns:
                mask &= df[tk] == 1.0
        scheme_id[mask.values] = sid

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, col in enumerate(process_cols):
        ax = axes[i]
        vals = df[col].values
        unique_vals = np.sort(df[col].unique())
        jitter = 0.02 * (unique_vals.max() - unique_vals.min())

        for sid in range(1, 5):
            m = scheme_id == sid
            ax.scatter(vals[m] + np.random.uniform(-jitter, jitter, m.sum()),
                      df.loc[m, 'TARGET_Yield'],
                      c=SCHEME_COLORS[sid - 1], s=50, alpha=0.7,
                      edgecolor='white', linewidth=0.5, zorder=3,
                      label=SCHEME_NAMES.get(sid, f'方案{sid}'))

        # 箱线图叠加
        positions = unique_vals
        box_data = []
        for uv in unique_vals:
            box_data.append(df.loc[np.abs(df[col] - uv) < 1e-6, 'TARGET_Yield'].values)
        bp = ax.boxplot(box_data, positions=positions, widths=jitter * 1.5,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='white', alpha=0.4, edgecolor='gray'),
                        whiskerprops=dict(color='gray', alpha=0.5),
                        capprops=dict(color='gray', alpha=0.5),
                        medianprops=dict(color='black', lw=1.5),
                        zorder=1)

        ax.set_xlabel(PROCESS_LABELS_CN.get(col, col), fontsize=11)
        ax.set_ylabel('TARGET_Yield', fontsize=11)
        ax.set_title(col.replace('Process_', ''), fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    fig.suptitle('原始实验数据: 产率 vs 工艺参数 (按目标晶向方案着色)',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "3_Raw_Data_Scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 原始数据散点图已保存至: {save_path}")
    plt.show()
    plt.close()


# ============================================================================
# Tier 2 — 物理叙事
# ============================================================================

def plot_pre_feature_space(optimizer, output_dir, target_scheme=None):
    """
    微观结构特征空间 — Top-2 ARD Pre_ 特征散点 + 所有 Pre_ 特征与产率的相关系数。
    强调 Pre_ 方差很大，"均值 Pre_" 不是一个真实存在的材料。
    """
    df = optimizer.training_df.copy()
    pre_cols = optimizer.pre_feature_cols

    # 获取 ARD 排序
    ard_df = optimizer.extract_ard_importance()
    ard_pre = ard_df[ard_df['category'] == 'EBSD预处理'].sort_values('length_scale')
    top2 = ard_pre['feature'].values[:2]

    # 筛选目标方案
    if target_scheme is not None:
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme]
        mask = np.ones(len(df), dtype=bool)
        for tk in target_keys:
            if tk in df.columns:
                mask &= df[tk] == 1.0
        df_plot = df[mask].copy()
        scheme_str = f" — {target_scheme}"
    else:
        df_plot = df
        scheme_str = ""

    fig = plt.figure(figsize=(16, 6))

    # Panel A: Top-2 ARD 特征散点图
    ax_a = fig.add_subplot(121)
    sc = ax_a.scatter(df_plot[top2[0]], df_plot[top2[1]],
                      c=df_plot['TARGET_Yield'], cmap='viridis',
                      s=70, edgecolor='white', linewidth=0.6, alpha=0.85)
    plt.colorbar(sc, ax=ax_a, label='TARGET_Yield')

    # 标注 1σ / 2σ 椭圆
    from matplotlib.patches import Ellipse
    mean_xy = df_plot[[top2[0], top2[1]]].mean().values
    cov = df_plot[[top2[0], top2[1]]].cov().values
    for n_std, ls, alpha_val in [(1, '--', 0.4), (2, ':', 0.2)]:
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(vals)
        ell = Ellipse(xy=mean_xy, width=w, height=h, angle=angle,
                      edgecolor='red', facecolor='none', lw=1.2, linestyle=ls)
        ax_a.add_patch(ell)
        ax_a.annotate(f'{n_std}σ', xy=(mean_xy[0] + w/2 * 0.8, mean_xy[1]),
                     fontsize=9, color='red', alpha=0.7)

    ax_a.scatter([mean_xy[0]], [mean_xy[1]], c='red', s=120, marker='X',
                 edgecolor='white', linewidth=1.5, zorder=5,
                 label='Pre_ 均值点 (人造的)')
    ax_a.set_xlabel(top2[0], fontsize=11)
    ax_a.set_ylabel(top2[1], fontsize=11)
    ax_a.set_title(f'Top-2 ARD 微观结构特征{scheme_str}', fontsize=13)
    ax_a.legend(fontsize=8)
    ax_a.grid(True, linestyle='--', alpha=0.3)

    # Panel B: 相关系数
    ax_b = fig.add_subplot(122)
    corrs = []
    for col in pre_cols:
        r, p = spearmanr(df_plot[col], df_plot['TARGET_Yield'], nan_policy='omit')
        corrs.append({'feature': col, 'rho': r, 'p': p})
    corr_df = pd.DataFrame(corrs).sort_values('rho', key=abs, ascending=True)

    colors_bar = ['#2166ac' if v < 0 else '#b2182b' for v in corr_df['rho']]
    bars = ax_b.barh(range(len(corr_df)), corr_df['rho'], color=colors_bar, alpha=0.8)
    ax_b.set_yticks(range(len(corr_df)))
    ax_b.set_yticklabels(corr_df['feature'], fontsize=9)
    ax_b.set_xlabel("Spearman ρ (与产率的相关性)", fontsize=11)
    ax_b.set_title('各 Pre_ 特征与产率的相关性', fontsize=13)
    ax_b.axvline(0, color='black', lw=0.8)
    ax_b.grid(axis='x', linestyle='--', alpha=0.3)

    # 标注显著性
    for idx, row in corr_df.iterrows():
        if row['p'] < 0.05:
            ax_b.annotate('*' if row['p'] > 0.01 else '**',
                         xy=(row['rho'] + 0.02 * np.sign(row['rho']), corr_df.index.get_loc(idx)),
                         fontsize=11, va='center')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "4_Pre_Feature_Space.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 微观结构特征空间图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_ice_curves(optimizer, output_dir, target_scheme=None,
                    param='Process_Temp', n_curves=12, n_grid=50):
    """
    个体条件期望 (ICE) 曲线 — 固定每个实验自身的 Pre_ + Target_ 编码，
    扫描 Temp，观察每个实验的预测产率轨迹。
    """
    df = optimizer.training_df.copy()
    pre_cols = optimizer.pre_feature_cols
    process_cols = optimizer.process_cols

    # 目标编码 + 过滤
    if target_scheme is not None and optimizer.target_cols:
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme]
        target_enc = np.array([1.0 if c in target_keys else 0.0 for c in optimizer.target_cols])
        mask = np.ones(len(df), dtype=bool)
        for tk in target_keys:
            if tk in df.columns:
                mask &= df[tk] == 1.0
        df_plot = df[mask].copy()
        scheme_str = f" — {target_scheme}"
    else:
        df_plot = df
        target_enc = np.zeros(len(optimizer.target_cols))
        scheme_str = ""

    # 按产率四分位选代表性实验
    if len(df_plot) <= n_curves:
        selected = df_plot
    else:
        df_plot = df_plot.copy()
        df_plot['yield_quartile'] = pd.qcut(df_plot['TARGET_Yield'], 4,
                                            labels=False, duplicates='drop')
        n_per = n_curves // 4
        selected_idx = []
        for q in range(4):
            q_data = df_plot[df_plot['yield_quartile'] == q]
            n_pick = min(n_per, len(q_data))
            if n_pick > 0:
                selected_idx.extend(np.random.choice(q_data.index, n_pick, replace=False))
        selected = df_plot.loc[selected_idx]

    lo, hi = optimizer.bounds[param]
    grid = np.linspace(lo, hi, n_grid)
    param_idx = process_cols.index(param)

    fig, ax = plt.subplots(figsize=(10, 6))
    all_curves = np.zeros((len(selected), n_grid))

    for i, (_, row) in enumerate(selected.iterrows()):
        pre_i = row[pre_cols].values.astype(float)
        proc_i = row[process_cols].values.astype(float)

        X_proc = np.tile(proc_i, (n_grid, 1))
        X_proc[:, param_idx] = grid

        X_full = np.hstack([
            np.tile(pre_i, (n_grid, 1)),
            np.tile(target_enc, (n_grid, 1)),
            X_proc
        ])
        Xs = optimizer.scaler_X.transform(X_full)
        mu, _ = optimizer.gpr.predict(Xs, return_std=True)
        all_curves[i] = mu

        ax.plot(grid, mu, color='gray', alpha=0.3, lw=0.8)

    # PDP 均值线
    pdp_mean = all_curves.mean(axis=0)
    ax.plot(grid, pdp_mean, color='red', lw=2.5, label='PDP 均值')

    # 真实点标注
    for i, (_, row) in enumerate(selected.iterrows()):
        orig_x = row[param]
        ax.scatter([orig_x], [all_curves[i, np.argmin(np.abs(grid - orig_x))]],
                  c='black', s=30, zorder=5, edgecolor='white', linewidth=0.5)

    ax.set_xlabel(PROCESS_LABELS_CN.get(param, param), fontsize=11)
    ax.set_ylabel('模型预测产率', fontsize=11)
    ax.set_title(f'ICE 曲线: 逐实验预测产率 vs {param.replace("Process_", "")}{scheme_str}',
                 fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 文本说明
    ax.text(0.02, 0.98,
            f'N={len(selected)} 个代表性实验\n灰线 = 单实验轨迹\n黑点 = 真实实验条件',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "5_ICE_Curves.png")
    plt.savefig(save_path, dpi=300)
    print(f"[*] ICE 曲线图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_ard_grouped(optimizer, output_dir):
    """
    ARD 长度尺度分组对比 — Pre_ vs Target_ vs Process_ 三组水平条形图。
    直观展示哪些特征在核函数中实际起效。
    """
    ard_df = optimizer.extract_ard_importance()
    # 按类别 + length_scale 排序
    cat_order = {'EBSD预处理': 0, '目标晶向': 1, '工艺参数': 2}
    cat_colors = {'EBSD预处理': '#2166ac', '目标晶向': '#f4a582', '工艺参数': '#4daf4a'}
    ard_df['cat_order'] = ard_df['category'].map(cat_order)
    ard_df = ard_df.sort_values(['cat_order', 'length_scale'], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [cat_colors.get(c, 'gray') for c in ard_df['category']]
    bars = ax.barh(range(len(ard_df)), ard_df['length_scale'], color=colors, alpha=0.85)

    ax.set_yticks(range(len(ard_df)))
    # 简短标签
    short_labels = [f.replace('Pre_', '').replace('Target_', '')
                    for f in ard_df['feature']]
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel('ARD 长度尺度 (越小越重要)', fontsize=11)
    ax.set_title('ARD 自动相关性判定: 长度尺度分组对比', fontsize=14)

    # 上界标注线
    ax.axvline(2.0, color='red', linestyle='--', lw=1.2, alpha=0.6)
    ax.annotate('接近上界 (5.0):\n核函数近似平坦',
                xy=(4.5, len(ard_df) * 0.85), fontsize=9, color='red',
                ha='center')

    # 图例
    legend_patches = [
        mpatches.Patch(color=cat_colors['EBSD预处理'], label='EBSD 微观结构'),
        mpatches.Patch(color=cat_colors['目标晶向'], label='目标晶向'),
        mpatches.Patch(color=cat_colors['工艺参数'], label='工艺参数'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "6_ARD_Grouped_Comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] ARD 分组对比图已保存至: {save_path}")
    plt.show()
    plt.close()


# ============================================================================
# Tier 3 — 辅助分析
# ============================================================================

def plot_uncertainty_map(optimizer, output_dir, target_scheme=None,
                         grid_size=80):
    """
    模型不确定性热力图 — Temp-Time 平面上的 σ 分布。
    展示在哪些工艺参数区域模型有/没有数据支撑。
    """
    df = optimizer.training_df.copy()
    pre_cols = optimizer.pre_feature_cols
    process_cols = optimizer.process_cols

    if target_scheme is not None and optimizer.target_cols:
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme]
        target_enc = np.array([1.0 if c in target_keys else 0.0 for c in optimizer.target_cols])
        mask = np.ones(len(df), dtype=bool)
        for tk in target_keys:
            if tk in df.columns:
                mask &= df[tk] == 1.0
        df_plot = df[mask].copy()
        scheme_str = f" — {target_scheme}"
    else:
        df_plot = df
        target_enc = np.zeros(len(optimizer.target_cols))
        scheme_str = ""

    pre_mean = df_plot[pre_cols].mean().values
    process_mean = df_plot[process_cols].mean().values

    # Temp-Time 网格
    bounds_t = optimizer.bounds['Process_Temp']
    bounds_time = optimizer.bounds['Process_Time']
    t_vals = np.linspace(bounds_t[0], bounds_t[1], grid_size)
    time_vals = np.linspace(bounds_time[0], bounds_time[1], grid_size)
    T_grid, Time_grid = np.meshgrid(t_vals, time_vals)

    idx_t = process_cols.index('Process_Temp')
    idx_time = process_cols.index('Process_Time')
    n = grid_size * grid_size
    X_proc = np.tile(process_mean, (n, 1))
    X_proc[:, idx_t] = T_grid.ravel()
    X_proc[:, idx_time] = Time_grid.ravel()

    X_full = np.hstack([
        np.tile(pre_mean, (n, 1)),
        np.tile(target_enc, (n, 1)),
        X_proc
    ])
    Xs = optimizer.scaler_X.transform(X_full)
    _, sigma = optimizer.gpr.predict(Xs, return_std=True)
    Sigma_grid = sigma.reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: sigma heatmap
    c1 = axes[0].contourf(T_grid, Time_grid, Sigma_grid, levels=40,
                          cmap='YlOrRd')
    plt.colorbar(c1, ax=axes[0], label='预测标准差 σ')
    axes[0].scatter(df_plot['Process_Temp'], df_plot['Process_Time'],
                   c='black', s=40, edgecolor='white', linewidth=0.8,
                   alpha=0.8, label='训练数据点')
    axes[0].set_xlabel('退火温度 (°C)', fontsize=11)
    axes[0].set_ylabel('保温时间 (h)', fontsize=11)
    axes[0].set_title(f'模型不确定性 σ (Temp × Time){scheme_str}', fontsize=13)
    axes[0].legend(fontsize=8)

    # Panel B: H2-Ar
    bounds_h2 = optimizer.bounds['Process_H2']
    bounds_ar = optimizer.bounds['Process_Ar']
    h2_vals = np.linspace(bounds_h2[0], bounds_h2[1], grid_size)
    ar_vals = np.linspace(bounds_ar[0], bounds_ar[1], grid_size)
    H2_grid, Ar_grid = np.meshgrid(h2_vals, ar_vals)

    idx_h2 = process_cols.index('Process_H2')
    idx_ar = process_cols.index('Process_Ar')
    X_proc2 = np.tile(process_mean, (n, 1))
    X_proc2[:, idx_h2] = H2_grid.ravel()
    X_proc2[:, idx_ar] = Ar_grid.ravel()

    X_full2 = np.hstack([
        np.tile(pre_mean, (n, 1)),
        np.tile(target_enc, (n, 1)),
        X_proc2
    ])
    Xs2 = optimizer.scaler_X.transform(X_full2)
    _, sigma2 = optimizer.gpr.predict(Xs2, return_std=True)
    Sigma2_grid = sigma2.reshape(grid_size, grid_size)

    c2 = axes[1].contourf(H2_grid, Ar_grid, Sigma2_grid, levels=40,
                          cmap='YlOrRd')
    plt.colorbar(c2, ax=axes[1], label='预测标准差 σ')
    axes[1].scatter(df_plot['Process_H2'], df_plot['Process_Ar'],
                   c='black', s=40, edgecolor='white', linewidth=0.8,
                   alpha=0.8, label='训练数据点')
    axes[1].set_xlabel('H$_2$ 流量 (sccm)', fontsize=11)
    axes[1].set_ylabel('Ar 流量 (sccm)', fontsize=11)
    axes[1].set_title(f'模型不确定性 σ (H2 × Ar){scheme_str}', fontsize=13)
    axes[1].legend(fontsize=8)

    fig.suptitle('模型认知不确定性: 标准差 σ 在工艺参数空间中的分布',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "7_Uncertainty_Map.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 不确定性热力图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_process_pre_mediation(optimizer, output_dir):
    """
    探索性因果分析 — 工艺参数 vs 关键 Pre_ 特征的散点图 + LOESS 平滑。
    N=18 独立实验，统计功效不足，标记为"探索性"。
    """
    df = optimizer.training_df.copy()
    pre_cols = optimizer.pre_feature_cols
    process_cols = optimizer.process_cols

    # 识别关键 Pre_ 特征：与产率相关系数最高的那个
    best_pre = None
    best_rho = 0
    for col in pre_cols:
        r, _ = spearmanr(df[col], df['TARGET_Yield'], nan_policy='omit')
        if abs(r) > abs(best_rho):
            best_rho = r
            best_pre = col

    # 只取每个实验的一行（去重，因为每个实验×4方案）
    exp_rows = []
    seen = set()
    for i, row in df.iterrows():
        key = tuple(row[process_cols].values.round(1))
        if key not in seen:
            seen.add(key)
            exp_rows.append(i)
    df_unique = df.iloc[exp_rows].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, col in enumerate(process_cols):
        ax = axes[i]
        x = df_unique[col].values
        y = df_unique[best_pre].values

        ax.scatter(x, y, c='steelblue', s=70, edgecolor='white',
                  linewidth=0.8, alpha=0.85)

        # 如果有足够的唯一 x 值，添加 LOESS 平滑
        if len(np.unique(x)) >= 3:
            try:
                from scipy.interpolate import make_interp_spline
                sort_idx = np.argsort(x)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                # LOWESS via statsmodels if available
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smoothed = lowess(y_sorted, x_sorted, frac=0.7, return_sorted=True)
                    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', lw=2,
                           alpha=0.7, label='LOWESS 平滑')
                except ImportError:
                    pass
            except Exception:
                pass

        rho, pval = spearmanr(x, y, nan_policy='omit')
        ax.set_xlabel(PROCESS_LABELS_CN.get(col, col), fontsize=11)
        ax.set_ylabel(best_pre, fontsize=11)
        ax.set_title(f'{col.replace("Process_", "")} vs {best_pre}', fontsize=12)

        sig_str = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
        ax.annotate(f'Spearman ρ = {rho:.3f} {sig_str}\np = {pval:.4f}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle(f'探索性分析: 工艺参数 → 微观结构 ({best_pre}) 的因果链',
                 fontsize=14, y=1.01)

    # 底部总说明
    fig.text(0.5, 0.01,
             '注意: N = 18 个独立实验, 统计功效不足。'
             '若所有相关性均不显著，需设计系统性的工艺参数梯度实验来建立因果链。',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(output_dir, "8_Exploratory_Process_Pre_Mediation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 探索性因果链图已保存至: {save_path}")
    plt.show()
    plt.close()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    default_file = os.path.join(_PROJECT_ROOT, "Optimized_Training_Data.csv")

    if not os.path.exists(default_file):
        print(f"[错误] 未找到数据文件: {default_file}")
        sys.exit(1)

    print(f"[*] 使用数据文件: {default_file}")
    print("\n请选择目标晶向方案 (用于 Pre_ 特征空间筛选):")
    print("  方案 1: <103>, <102>, <301>")
    print("  方案 2: <114>, <115>, <105>")
    print("  方案 3: <124>, <125>, <214>")
    print("  方案 4: <103>, <114>, <124>")

    while True:
        try:
            inp = input("\n请输入方案编号 (1-4，直接回车默认方案1): ").strip()
            if inp == "":
                current_scheme = 1
                break
            current_scheme = int(inp)
            if current_scheme in [1, 2, 3, 4]:
                break
            print("[!] 请输入 1-4 之间的数字")
        except ValueError:
            print("[!] 请输入有效的数字")

    target_scheme = SCHEME_TARGETS[current_scheme]
    print(f"[*] 已选择方案 {current_scheme}: {target_scheme}")

    # 训练模型
    print("\n正在训练 GP 代理模型...")
    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    optimizer.train(default_file)

    df = pd.read_csv(default_file)
    n_samples = len(df)

    # 输出目录
    output_dir = create_output_dir(scheme_id=current_scheme, n_samples=n_samples)

    # Tier 1
    print("\n" + "=" * 50)
    print("  Tier 1: 核心替代可视化")
    print("=" * 50)
    print("\n[1/8] 实验产率分组柱状图...")
    plot_yield_by_experiment(optimizer, output_dir, target_scheme=target_scheme)

    print("\n[2/8] 部分依赖图 (PDP)...")
    plot_partial_dependence(optimizer, output_dir, target_scheme=target_scheme)

    print("\n[3/8] 原始数据散点图...")
    plot_raw_data_scatter(optimizer, output_dir)

    # Tier 2
    print("\n[4/8] 微观结构特征空间...")
    plot_pre_feature_space(optimizer, output_dir, target_scheme=target_scheme)

    print("\n[5/8] ICE 曲线...")
    plot_ice_curves(optimizer, output_dir, target_scheme=target_scheme)

    print("\n[6/8] ARD 分组对比...")
    plot_ard_grouped(optimizer, output_dir)

    # Tier 3
    print("\n[7/8] 模型不确定性热力图...")
    plot_uncertainty_map(optimizer, output_dir, target_scheme=target_scheme)

    print("\n[8/8] 探索性因果链...")
    plot_process_pre_mediation(optimizer, output_dir)

    print(f"\n{'=' * 50}")
    print(f"  全部 8 张可视化图已保存至:")
    print(f"  {output_dir}")
    print(f"{'=' * 50}")
