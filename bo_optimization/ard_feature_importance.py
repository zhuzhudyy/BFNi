# -*- coding: utf-8 -*-
"""
ARD (Automatic Relevance Determination) 特征重要性可视化

功能:
    1. 训练高斯过程模型并启用 ARD
    2. 提取每个特征的长度尺度参数
    3. 可视化特征重要性排序（水平条形图）
    4. 按特征类别分组展示

输出:
    保存路径: D:\毕业设计\织构数据\visualization\ard_importance
    文件名: Scheme{ID}_N{样本量}_{时间戳}.png
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import io

# 修复 VS Code / Windows 终端中文乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import logging

# 完全屏蔽所有警告
warnings.filterwarnings("ignore")

# 屏蔽 Matplotlib 的字体日志警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 设置中文字体
# 使用 SimHei 显示中文，DejaVu Sans 显示负号等特殊字符
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True  # 使用 Unicode 负号，确保正确显示

# 启用 Matplotlib 内置数学文本渲染（无需安装 LaTeX）
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.formatter.use_mathtext'] = True

from bo_optimization.contextual_bo_model import (
    ContextualBayesianOptimizer, SCHEME_TARGETS, SCHEME_NAMES, DEFAULT_PROCESS_BOUNDS
)


def train_model_with_ard(data_file):
    """训练模型并返回优化器实例"""
    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    optimizer.train(data_file)
    return optimizer


def permutation_importance(optimizer, X, y, feature_cols, n_repeats=10):
    """
    留一法置换重要性（Permutation Importance）
    随机打乱某个特征的数值顺序，观察LOOCV中预测误差（RMSE）的增加量
    
    Returns:
        perm_importance_df: 置换重要性数据框
    """
    print("\n" + "="*60)
    print("【置换重要性分析】Permutation Importance")
    print("="*60)
    print(f"进行 {n_repeats} 次置换重复...")
    
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import mean_squared_error
    
    # 基准LOOCV误差（不打乱）
    print("\n计算基准LOOCV误差...")
    loo = LeaveOneOut()
    baseline_preds = []
    baseline_trues = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练临时模型
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
        gpr.fit(X_train, y_train)
        
        y_pred = gpr.predict(X_test)
        baseline_preds.append(y_pred[0])
        baseline_trues.append(y_test[0])
    
    baseline_rmse = np.sqrt(mean_squared_error(baseline_trues, baseline_preds))
    print(f"  基准RMSE: {baseline_rmse:.4f}")
    
    # 对每个特征进行置换重要性计算
    perm_importance = []
    n_features = len(feature_cols)
    n_loocv = len(X)  # LOOCV的折数
    
    print(f"\n  总特征数: {n_features}, LOOCV折数: {n_loocv}, 置换重复: {n_repeats}")
    print(f"  预计总模型训练次数: {n_features} x {n_repeats} x {n_loocv} = {n_features * n_repeats * n_loocv}")
    print("  " + "-" * 60)
    
    # 计算总任务数用于进度条
    total_tasks = n_features * n_repeats * n_loocv
    current_task = 0
    
    for feat_idx, feat_name in enumerate(feature_cols):
        print(f"\n  [{feat_idx+1}/{n_features}] 分析特征: {feat_name}")
        rmse_increases = []
        
        for repeat in range(n_repeats):
            # 置换该特征
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feat_idx])
            
            # LOOCV评估
            perm_preds = []
            perm_trues = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X_permuted)):
                # 更新进度
                current_task += 1
                progress = current_task / total_tasks * 100
                bar_length = 30
                filled = int(bar_length * current_task / total_tasks)
                bar = '=' * filled + '>' + '-' * (bar_length - filled - 1) if filled < bar_length else '=' * bar_length
                
                # 显示进度条（在同一行更新）
                print(f"\r    重复{repeat+1}/{n_repeats} 折{fold_idx+1}/{n_loocv} [{bar}] {progress:.1f}%", end='', flush=True)
                
                X_train, X_test = X_permuted[train_idx], X_permuted[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
                gpr.fit(X_train, y_train)
                
                y_pred = gpr.predict(X_test)
                perm_preds.append(y_pred[0])
                perm_trues.append(y_test[0])
            
            print()  # 换行
            
            perm_rmse = np.sqrt(mean_squared_error(perm_trues, perm_preds))
            rmse_increase = perm_rmse - baseline_rmse
            rmse_increases.append(rmse_increase)
        
        mean_increase = np.mean(rmse_increases)
        std_increase = np.std(rmse_increases)
        
        perm_importance.append({
            'feature': feat_name,
            'rmse_increase_mean': mean_increase,
            'rmse_increase_std': std_increase,
            'importance_score': mean_increase / (baseline_rmse + 1e-10)  # 归一化重要性
        })
        print(f"    -> RMSE增加: {mean_increase:.4f} ± {std_increase:.4f}")
    
    perm_df = pd.DataFrame(perm_importance).sort_values('importance_score', ascending=False)
    
    print("\n置换重要性排名（Top 10）:")
    print("-" * 60)
    for idx, row in perm_df.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance_score']:.4f} "
              f"(+{row['rmse_increase_mean']:.4f} ± {row['rmse_increase_std']:.4f})")
    
    return perm_df, baseline_rmse


def plot_loocv_permutation_importance(perm_df, scheme_id, n_samples, output_dir, baseline_rmse):
    """
    绘制 LOOCV 置换重要性可视化（独立保存）
    
    Args:
        perm_df: 置换重要性数据框
        scheme_id: 方案ID
        n_samples: 样本数量
        output_dir: 输出目录
        baseline_rmse: 基准LOOCV RMSE
    """
    from datetime import datetime
    
    # 按重要性排序（降序）
    df_sorted = perm_df.sort_values('importance_score', ascending=True)
    
    # 创建颜色映射
    color_map = {
        'EBSD预处理': '#3498db',  # 蓝色
        '目标晶向': '#e74c3c',     # 红色
        '工艺参数': '#2ecc71'      # 绿色
    }
    
    # 识别特征类别
    def get_category(feature_name):
        if feature_name.startswith('Pre_'):
            return 'EBSD预处理'
        elif feature_name.startswith('Target_'):
            return '目标晶向'
        elif feature_name.startswith('Process_'):
            return '工艺参数'
        else:
            return '其他'
    
    df_sorted['category'] = df_sorted['feature'].apply(get_category)
    colors = [color_map[cat] for cat in df_sorted['category']]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # ========== 左图：置换重要性排序（水平条形图）==========
    y_pos = np.arange(len(df_sorted))
    bars = ax1.barh(y_pos, df_sorted['importance_score'], color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    # 添加误差线
    ax1.errorbar(df_sorted['importance_score'], y_pos, 
                 xerr=df_sorted['rmse_increase_std'] / (baseline_rmse + 1e-10),
                 fmt='none', color='black', alpha=0.5, capsize=2)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['feature'], fontsize=9)
    ax1.set_xlabel('置换重要性分数 (RMSE增加倍数)', fontsize=12, fontweight='bold')
    ax1.set_title(f'LOOCV 置换重要性排序\n方案 {scheme_id} | 样本量 N={n_samples} | 基准RMSE={baseline_rmse:.4f}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax1.text(row['importance_score'] + 0.02, i, 
                f"+{row['rmse_increase_mean']:.3f}", 
                va='center', fontsize=8, color='darkred')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat, alpha=0.8) 
                      for cat in ['EBSD预处理', '目标晶向', '工艺参数']]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # ========== 右图：RMSE增加量分布（小提琴图+散点）==========
    categories = ['EBSD预处理', '目标晶向', '工艺参数']
    
    # 为每个类别准备数据
    cat_data = []
    cat_positions = []
    position = 1
    
    for cat in categories:
        data = df_sorted[df_sorted['category'] == cat]['rmse_increase_mean'].values
        if len(data) > 0:
            cat_data.append(data)
            cat_positions.append(position)
            position += 1
    
    # 绘制小提琴图
    if cat_data:
        try:
            parts = ax2.violinplot(cat_data, positions=cat_positions, widths=0.6,
                                   showmeans=True, showmedians=True, showextrema=False)
            
            # 设置小提琴图颜色
            active_cats = [cat for cat in categories 
                          if len(df_sorted[df_sorted['category'] == cat]) > 0]
            for i, (pc, cat) in enumerate(zip(parts['bodies'], active_cats)):
                pc.set_facecolor(color_map[cat])
                pc.set_alpha(0.4)
                pc.set_edgecolor(color_map[cat])
                pc.set_linewidth(2)
        except:
            # 如果小提琴图失败，回退到箱线图
            bp = ax2.boxplot(cat_data, positions=cat_positions, widths=0.6,
                            patch_artist=True, showmeans=True)
            active_cats = [cat for cat in categories 
                          if len(df_sorted[df_sorted['category'] == cat]) > 0]
            for patch, cat in zip(bp['boxes'], active_cats):
                patch.set_facecolor(color_map[cat])
                patch.set_alpha(0.4)
    
    # 添加散点
    np.random.seed(42)
    for i, (cat, pos) in enumerate(zip(active_cats, cat_positions)):
        cat_df = df_sorted[df_sorted['category'] == cat]
        y_values = cat_df['rmse_increase_mean'].values
        x_jitter = np.random.normal(pos, 0.08, len(y_values))
        ax2.scatter(x_jitter, y_values, c=color_map[cat], s=50, alpha=0.7, 
                   edgecolors='black', linewidth=0.5, zorder=5)
        
        # 为重要特征添加标签
        for idx, row in cat_df.iterrows():
            if row['importance_score'] > 0.5:  # 重要性较高的特征
                ax2.annotate(row['feature'], 
                           xy=(pos + np.random.normal(0, 0.1), row['rmse_increase_mean']),
                           xytext=(10, 0), textcoords='offset points',
                           fontsize=7, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax2.set_xticks(cat_positions)
    ax2.set_xticklabels(active_cats, fontsize=11)
    ax2.set_ylabel('RMSE 增加量', fontsize=12, fontweight='bold')
    ax2.set_title('各类特征置换重要性分布\n(打乱后RMSE增加量)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加基准线
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='基准线')
    
    plt.tight_layout()
    
    # 保存图片（独立文件）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"LOOCV_PermImportance_Scheme{scheme_id}_N{n_samples}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] LOOCV置换重要性图已保存至: {save_path}")
    
    plt.show()
    
    return save_path



def plot_ard_importance(df_importance, scheme_id, n_samples, output_dir):
    """绘制 ARD 特征重要性可视化"""
    
    # 按重要性排序
    df_sorted = df_importance.sort_values('importance', ascending=True)
    
    # 创建颜色映射
    color_map = {
        'EBSD预处理': '#3498db',  # 蓝色
        '目标晶向': '#e74c3c',     # 红色
        '工艺参数': '#2ecc71'      # 绿色
    }
    colors = [color_map[cat] for cat in df_sorted['category']]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # ========== 左图：特征重要性排序（水平条形图）==========
    y_pos = np.arange(len(df_sorted))
    bars = ax1.barh(y_pos, df_sorted['importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['feature'], fontsize=9)
    ax1.set_xlabel('重要性分数 (1/Length Scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'ARD 特征重要性排序\n方案 {scheme_id} | 样本量 N={n_samples}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax1.text(row['importance'] + 0.01, i, f"{row['length_scale']:.3f}", 
                va='center', fontsize=8, color='darkred')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat, alpha=0.8) 
                      for cat in ['EBSD预处理', '目标晶向', '工艺参数']]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # ========== 右图：各类特征长度尺度对比（优化版）==========
    categories = ['EBSD预处理', '目标晶向', '工艺参数']
    
    # 为每个类别准备数据
    cat_data = []
    cat_positions = []
    position = 1
    
    for cat in categories:
        data = df_importance[df_importance['category'] == cat]['length_scale'].values
        cat_data.append(data)
        cat_positions.append(position)
        position += 1
    
    # 绘制小提琴图展示分布
    from matplotlib.patches import Rectangle
    try:
        parts = ax2.violinplot(cat_data, positions=cat_positions, widths=0.6,
                               showmeans=True, showmedians=True, showextrema=False)
        
        # 设置小提琴图颜色
        for i, (pc, cat) in enumerate(zip(parts['bodies'], categories)):
            pc.set_facecolor(color_map[cat])
            pc.set_alpha(0.4)
            pc.set_edgecolor(color_map[cat])
            pc.set_linewidth(2)
    except:
        # 如果小提琴图失败，回退到箱线图
        bp = ax2.boxplot(cat_data, positions=cat_positions, widths=0.6,
                        patch_artist=True, showmeans=True)
        for patch, cat in zip(bp['boxes'], categories):
            patch.set_facecolor(color_map[cat])
            patch.set_alpha(0.4)
    
    # 叠加散点：显示每个特征的具体位置
    np.random.seed(42)  # 固定随机种子，保证可重复
    
    # 为每个点准备标注信息
    annotations = []
    
    for i, (data, cat) in enumerate(zip(cat_data, categories)):
        # 添加抖动，避免重叠
        jitter = np.random.normal(0, 0.08, len(data))
        x_positions = np.array([cat_positions[i]] * len(data)) + jitter
        
        # 根据重要性设置点的大小和颜色深浅
        sizes = [100 if ls < 1 else 50 for ls in data]
        alphas = [0.9 if ls < 1 else 0.5 for ls in data]
        
        scatter = ax2.scatter(x_positions, data, c=color_map[cat], s=sizes, 
                   alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
        
        # 收集所有点的标注信息
        cat_df = df_importance[df_importance['category'] == cat].reset_index(drop=True)
        for j, (x, y) in enumerate(zip(x_positions, data)):
            if j < len(cat_df):
                feature_name = cat_df.iloc[j]['feature']
                # 简化特征名显示
                short_name = feature_name.replace('Pre_', '').replace('Target_', '').replace('Process_', '')
                importance_level = "★★★" if y < 0.5 else ("★★" if y < 1 else ("★" if y < 10 else "☆"))
                annotations.append({
                    'x': x, 'y': y, 'name': short_name, 
                    'category': cat, 'importance': importance_level,
                    'ls': y
                })
    
    # 智能布局标注：按Y值分组，避免重叠
    # 将点按重要性分层，高重要性的优先标注且更靠近点
    high_importance = [a for a in annotations if a['ls'] < 1]  # 高重要性
    medium_importance = [a for a in annotations if 1 <= a['ls'] < 100]  # 中等重要性
    low_importance = [a for a in annotations if a['ls'] >= 100]  # 低重要性
    
    # 标注高重要性特征（左侧标注，避免遮挡）
    for idx, ann in enumerate(high_importance):
        offset_x = -60 if idx % 2 == 0 else -100  # 交替左右偏移
        offset_y = 15 if idx % 3 == 0 else (-15 if idx % 3 == 1 else 0)
        
        ax2.annotate(f"{ann['name']}\n({ann['importance']})", 
                    (ann['x'], ann['y']), 
                    xytext=(offset_x, offset_y),
                    textcoords='offset points', 
                    fontsize=8,
                    ha='right' if offset_x < 0 else 'left',
                    fontweight='bold',
                    color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=color_map[ann['category']], 
                             alpha=0.3, edgecolor='darkred', linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1,
                                  connectionstyle='arc3,rad=0.1'))
    
    # 标注中等重要性特征（右侧标注）
    for idx, ann in enumerate(medium_importance[:5]):  # 只显示前5个避免拥挤
        offset_x = 40 + (idx % 2) * 30
        offset_y = (idx % 3 - 1) * 20
        
        ax2.annotate(ann['name'], 
                    (ann['x'], ann['y']), 
                    xytext=(offset_x, offset_y),
                    textcoords='offset points', 
                    fontsize=7,
                    ha='left',
                    color='darkblue',
                    bbox=dict(boxstyle='round,pad=0.25', 
                             facecolor='white', 
                             alpha=0.6, edgecolor=color_map[ann['category']]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    
    # 标注低重要性特征（仅标注极端值）
    if low_importance:
        # 只标注最大值和最小值
        low_importance_sorted = sorted(low_importance, key=lambda x: x['ls'])
        for ann in [low_importance_sorted[0], low_importance_sorted[-1]]:
            ax2.annotate(ann['name'], 
                        (ann['x'], ann['y']), 
                        xytext=(50, 0),
                        textcoords='offset points', 
                        fontsize=7,
                        ha='left',
                        style='italic',
                        color='gray',
                        alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.2', 
                                 facecolor='lightgray', 
                                 alpha=0.3, edgecolor='gray'))
    
    # 添加重要性阈值线
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='重要性阈值 (ls=1)')
    ax2.axhline(y=100, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='低重要性阈值 (ls=100)')
    
    # 添加重要性区域标注
    ax2.axhspan(0.001, 1, alpha=0.1, color='green', label='高重要性区域')
    ax2.axhspan(100, 1000000, alpha=0.1, color='red', label='低重要性区域')
    
    ax2.set_xticks(cat_positions)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel('Length Scale (对数刻度)', fontsize=12, fontweight='bold')
    ax2.set_title('各类特征长度尺度分布\n(散点=单个特征, 小提琴=分布形状)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 200000)
    
    # 自定义Y轴刻度标签，使用 LaTeX 格式避免 Unicode 上标乱码
    ax2.set_yticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    ax2.set_yticklabels([
        r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$',
        r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$'
    ], fontsize=10)
    
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)
    
    # 添加右侧重要性说明
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylim(ax2.get_ylim())
    ax2_twin.set_yscale('log')
    ax2_twin.set_yticks([0.1, 1, 10, 100, 10000])
    ax2_twin.set_yticklabels(['★★★\n极重要', '★★\n重要', '★\n一般', '☆\n次要', '☆☆\n无关'], fontsize=9)
    ax2_twin.set_ylabel('重要性评级', fontsize=11, fontweight='bold', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"Scheme{scheme_id}_N{n_samples}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] ARD 特征重要性图已保存至: {save_path}")
    
    plt.show()
    
    return save_path


def print_top_features(df_importance, top_n=5):
    """打印最重要的特征"""
    print("\n" + "="*60)
    print(f"           最重要的 {top_n} 个特征 (按 ARD 长度尺度)")
    print("="*60)
    
    df_sorted = df_importance.sort_values('length_scale', ascending=True)
    
    print(f"\n{'排名':<4} {'特征名':<25} {'类别':<12} {'长度尺度':<12} {'重要性'}")
    print("-" * 70)
    
    for i, (idx, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        importance_str = "★★★" if row['length_scale'] < 0.5 else ("★★" if row['length_scale'] < 1.0 else "★")
        print(f"{i:<4} {row['feature']:<25} {row['category']:<12} {row['length_scale']:<12.4f} {importance_str}")
    
    print("="*60)


if __name__ == "__main__":
    # 检测数据文件
    default_file = "Optimized_Training_Data.csv"
    
    if not os.path.exists(default_file):
        print("[!] 未找到数据文件，请确保 Optimized_Training_Data.csv 存在")
        sys.exit(1)
    
    data_file = default_file
    print(f"[*] 使用数据文件: {data_file}")
    
    print("\n请选择要分析的目标晶向方案:")
    print("  方案 1: <103>, <102>, <301>")
    print("  方案 2: <114>, <115>, <105>")
    print("  方案 3: <124>, <125>, <214>")
    print("  方案 4: <103>, <114>, <124>")
    
    while True:
        try:
            scheme_input = input("\n请输入方案编号 (1-4，直接回车默认方案1): ").strip()
            if scheme_input == "":
                scheme_id = 1
            else:
                scheme_id = int(scheme_input)
            if scheme_id in [1, 2, 3, 4]:
                break
            else:
                print("[!] 请输入 1-4 之间的数字")
        except ValueError:
            print("[!] 请输入有效的数字")
    
    print(f"[*] 已选择方案 {scheme_id}: {SCHEME_TARGETS[scheme_id]}")
    
    # 加载数据
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    # 训练模型
    print("\n正在训练高斯过程模型 (启用 ARD)...")
    optimizer = train_model_with_ard(data_file)
    
    # 提取 ARD 重要性
    print("\n正在提取 ARD 特征重要性...")
    df_importance = optimizer.extract_ard_importance()
    
    # 打印最重要的特征
    print_top_features(df_importance, top_n=10)
    
    # 创建输出目录（提前定义，供置换重要性可视化使用）
    # 优先使用 D:\毕业设计\织构数据\visualization，如果不存在则使用当前目录下的 visualization
    default_path = r"D:\毕业设计\织构数据\visualization\ard_importance"
    if os.path.exists(r"D:\毕业设计\织构数据"):
        output_dir = default_path
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "ard_importance")
        print(f"[!] 未检测到 D:\毕业设计\织构数据，使用本地路径: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 交叉验证 2: 置换重要性分析 ==========
    # 准备数据
    all_feature_cols = optimizer.pre_feature_cols + optimizer.target_cols + optimizer.process_cols
    X = df[all_feature_cols].values
    y = df['TARGET_Yield'].values
    
    # 询问是否进行置换重要性（计算量较大）
    print("\n" + "="*60)
    print("【可选分析】置换重要性 (Permutation Importance)")
    print("="*60)
    print("说明: 该分析通过LOOCV计算，需要约5-10分钟")
    print("      可以作为ARD排名的对照验证")
    print("      输入 'y' 进行分析，直接回车跳过")
    
    try:
        user_input = input("是否进行置换重要性分析? (y/n, 默认n): ").strip().lower()
        if user_input == 'y':
            perm_importance_df, baseline_rmse = permutation_importance(
                optimizer, X, y, all_feature_cols, n_repeats=5
            )
            
            # 对比ARD和置换重要性
            print("\n" + "="*60)
            print("【ARD vs 置换重要性 对比】")
            print("="*60)
            print(f"{'特征名':<25} {'ARD排名':<10} {'置换排名':<10} {'一致性':<10}")
            print("-" * 60)
            
            ard_rank = df_importance.sort_values('importance', ascending=False).reset_index(drop=True)
            perm_rank = perm_importance_df.reset_index(drop=True)
            
            for feat in ard_rank['feature'].head(10):
                ard_pos = ard_rank[ard_rank['feature'] == feat].index[0] + 1
                perm_match = perm_rank[perm_rank['feature'] == feat]
                perm_pos = perm_match.index[0] + 1 if len(perm_match) > 0 else '-'
                
                # 判断一致性（排名差异<=3认为一致）
                if perm_pos != '-' and abs(ard_pos - perm_pos) <= 3:
                    consistency = "一致"
                elif perm_pos != '-':
                    consistency = "差异"
                else:
                    consistency = "-"
                
                print(f"{feat:<25} {ard_pos:<10} {perm_pos:<10} {consistency:<10}")
            
            # 绘制LOOCV置换重要性可视化（独立保存）
            print("\n" + "="*60)
            print("【生成LOOCV置换重要性可视化】")
            print("="*60)
            plot_loocv_permutation_importance(
                perm_importance_df, scheme_id, n_samples, output_dir, baseline_rmse
            )
        else:
            print("\n跳过置换重要性分析")
    except EOFError:
        # 非交互式运行时的处理
        print("\n检测到非交互式运行，跳过置换重要性分析")
    except Exception as e:
        print(f"\n置换重要性分析出错: {e}")
        print("跳过此分析")
    
    # 绘制可视化
    print("\n正在绘制 ARD 特征重要性可视化...")
    plot_ard_importance(df_importance, scheme_id, n_samples, output_dir)
    
    print(f"\n分析完成！结果已保存至: {output_dir}/")
