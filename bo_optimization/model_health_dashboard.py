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
    explode = (0, 0, 0, 0.08)

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
