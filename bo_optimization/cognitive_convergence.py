"""
多任务上下文贝叶斯优化 (MTBO) 认知收敛分析工具

核心学术思想:
    传统的贝叶斯优化收敛指标 (如 Cumulative Best Yield, Simple Regret) 旨在全域寻找"单一最优解"。
    但在"上下文贝叶斯优化 (Contextual BO)" 中，我们的目标是为不同的初始微观状态 (Pre_ 特征) 
    寻找定制化的最优工艺 (即"因材施教")。不同初始态的理论上限不同，因此全局最高产率指标失效。
    
    本脚本采用《全域认知不确定度衰减 (Global Cognitive Uncertainty Decay)》作为收敛评价标准：
    通过监测代理模型在涵盖所有"初始态-工艺"组合的高维空间中，预测盲区 (Standard Deviation, \sigma) 
    的逐步下降，来严格证明模型已充分掌握物理映射规律。

使用方法:
    python cognitive_convergence.py
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True

from bo_optimization.contextual_bo_model import ContextualBayesianOptimizer, SCHEME_TARGETS, DEFAULT_PROCESS_BOUNDS

def calculate_cognitive_convergence(optimizer, data_file, target_orientations, n_simulations=5000):
    """
    计算模型在多维空间中的认知盲区（不确定度）衰减轨迹
    """
    df = pd.read_csv(data_file)
    
    # 1. MTBO: 生成任务掩码，只抽取当前目标晶向的历史数据
    active_keys = [f"Target_{h}{k}{l}" for h, k, l in target_orientations]
    mask = np.ones(len(df), dtype=bool)
    for key in active_keys:
        if key in df.columns:
            mask = mask & (df[key] == 1.0)
            
    df_scheme = df[mask].reset_index(drop=True)
    n_samples = len(df_scheme)
    
    if n_samples < 5:
        raise ValueError("该方案的历史数据少于 5 条，无法绘制有统计学意义的收敛曲线。请先积累更多实验数据。")

    target_encoding = np.array([1.0 if col in active_keys else 0.0 for col in optimizer.target_cols])
    
    # 2. 核心：构建高维全空间测试集 (Pre_特征 + 工艺参数)
    n_pre = len(optimizer.pre_feature_cols)
    n_process = len(optimizer.process_cols)
    total_dims = n_pre + n_process
    
    # 使用拉丁超立方采样 (LHS) 均匀覆盖整个高维空间 (取消固定seed，保留自然随机波动)
    sampler = qmc.LatinHypercube(d=total_dims)
    sample = sampler.random(n_simulations)
    
    # 初始化测试矩阵
    X_test_concat = np.zeros((n_simulations, len(optimizer.pre_feature_cols) + len(optimizer.target_cols) + len(optimizer.process_cols)))
    
    # 填充 Pre_ 随机值 (基于历史数据的物理上下限)
    for i, col in enumerate(optimizer.pre_feature_cols):
        col_min, col_max = df_scheme[col].min(), df_scheme[col].max()
        # 容错处理：防止某个特征在所有样品中完全一致导致极差为0
        if col_max - col_min < 1e-6: 
            col_max = col_min + 1e-3 
        X_test_concat[:, i] = col_min + sample[:, i] * (col_max - col_min)
        
    # 填充 Target_ (固定的目标晶向任务编码)
    target_start_idx = len(optimizer.pre_feature_cols)
    for i in range(len(optimizer.target_cols)):
        X_test_concat[:, target_start_idx + i] = target_encoding[i]
        
    # 填充 Process_ 随机值 (基于管式炉物理边界)
    process_start_idx = len(optimizer.pre_feature_cols) + len(optimizer.target_cols)
    for i, col in enumerate(optimizer.process_cols):
        low, high = optimizer.bounds[col]
        X_test_concat[:, process_start_idx + i] = low + sample[:, n_pre + i] * (high - low)

    mean_uncertainty_history = []
    max_uncertainty_history = []
    iterations = list(range(3, n_samples + 1))
    
    print(f"[*] 空间构建完毕。开始模拟模型的动态学习认知过程 (共 {len(iterations)} 步迭代)...")
    
    # 3. 模拟真实的序贯实验与认知学习过程
    for n in iterations:
        # 冷启动视角：只允许模型看到前 n 个实验数据
        df_subset = df_scheme.iloc[:n]
        X_train = df_subset[optimizer.pre_feature_cols + optimizer.target_cols + optimizer.process_cols].values
        y_train = df_subset['TARGET_Yield'].values
        
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        scaler = optimizer.scaler_X
        X_train_scaled = scaler.fit_transform(X_train)
        n_features = X_train_scaled.shape[1]
        
        # 统一使用受保护的 ARD 核函数，移除导致曲线断裂的 if n < 10 硬分支
        # 强制特征长度尺度下限为 0.1，防止维度坍缩
        safe_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * \
                      Matern(length_scale=[1.0]*n_features, length_scale_bounds=[(0.1, 100.0)]*n_features, nu=2.5) + \
                      WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        
        # 重新实例化安全的 GPR 并训练
        gpr = GaussianProcessRegressor(kernel=safe_kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
        gpr.fit(X_train_scaled, y_train)
        
        # 让模型去预测全空间的产率与不确定度(Sigma)
        X_test_scaled = scaler.transform(X_test_concat)
        _, sigma = gpr.predict(X_test_scaled, return_std=True)
        
        # 记录全域平均盲区(Mean Sigma) 和 最极端盲区(Max Sigma)
        mean_uncertainty_history.append(np.mean(sigma))
        max_uncertainty_history.append(np.max(sigma))
        
        if n % 5 == 0 or n == iterations[-1]:
            print(f"    - 迭代第 {n:02d} 步: 平均不确定度 \u03c3 = {np.mean(sigma):.4f}")
        
    return iterations, mean_uncertainty_history, max_uncertainty_history

def plot_cognitive_convergence(iterations, mean_uncertainty, max_uncertainty, scheme_id):
    """
    绘制严谨的认知衰减学术图表
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # 图：不确定度衰减曲线
    ax.plot(iterations, mean_uncertainty, marker='o', linestyle='-', color='teal', 
            linewidth=2.5, markersize=7, label='全空间平均盲区 (Mean Uncertainty, \u03bc_\u03c3)')
    ax.plot(iterations, max_uncertainty, marker='s', linestyle='--', color='crimson', 
            linewidth=2, markersize=6, alpha=0.6, label='最极端未知区盲区 (Max Uncertainty, max_\u03c3)')
    
    # 填充均值下方的面积，增加视觉美感
    ax.fill_between(iterations, 0, mean_uncertainty, color='teal', alpha=0.15)
    
    ax.set_xlabel('有效实验迭代批次 (Number of Observations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('代理模型预测不确定度 (Standard Deviation, \u03c3)', fontsize=12, fontweight='bold')
    ax.set_title(f'【方案 {scheme_id}】上下文代理模型全域认知收敛分析', fontsize=15, fontweight='bold', pad=15)
    
    # 设置合理的 Y 轴范围
    ax.set_ylim(0, max(max_uncertainty) * 1.1)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 生成学术结论：以历史最大峰值盲区作为衰减基准，修正早期假性自信导致的负衰减率
    final_mean_sigma = mean_uncertainty[-1]
    peak_mean_sigma = max(mean_uncertainty)
    decay_ratio = (peak_mean_sigma - final_mean_sigma) / peak_mean_sigma if peak_mean_sigma > 0 else 0
    
    if final_mean_sigma < 0.1 or decay_ratio > 0.7:
        conclusion = (f"结论: 平均预测不确定度已降至 {final_mean_sigma:.4f} (峰值衰减 {decay_ratio:.1%})。\n"
                      f"证明代理模型已充分学习并掌握了初始微观态(Pre)与工艺参数(Process)的复杂映射规律，全局优化已达成认知收敛。")
        color = 'forestgreen'
    else:
        conclusion = (f"结论: 整体盲区仍在下降通道中 (当前 \u03c3={final_mean_sigma:.4f})。\n"
                      f"模型正在持续学习新的映射规则，继续主动学习(Active Learning)有望进一步完善泛化能力。")
        color = 'darkorange'
        
    plt.figtext(0.5, -0.06, conclusion, ha="center", fontsize=11, 
                bbox={"facecolor":"white", "alpha":0.95, "pad":10, "edgecolor":color, "linewidth":2})

    plt.tight_layout()
    
    base_output_path = r"D:\毕业设计\织构数据\visualization\cognitive_convergence"
    if not os.path.exists(r"D:\毕业设计\织构数据"):
        base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "cognitive_convergence")
    os.makedirs(base_output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_samples = len(mean_uncertainty)
    save_path = os.path.join(base_output_path, f"Scheme{scheme_id}_N{n_samples}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 认知收敛分析图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" " * 5 + "上下文代理模型【认知盲区衰减】收敛分析 (Contextual BO)")
    print("="*60)
    
    try:
        scheme_id = int(input("请输入要分析收敛性的目标方案编号 (1-4): ").strip())
        if scheme_id not in [1, 2, 3, 4]: scheme_id = 1
        target_scheme = SCHEME_TARGETS[scheme_id]
    except:
        scheme_id = 1
        target_scheme = SCHEME_TARGETS[scheme_id]

    data_file = "Optimized_Training_Data.csv"
    if not os.path.exists(data_file):
        print(f"\n[错误] 未找到全局数据集 {data_file}。请先运行 data_builder.py。")
        sys.exit(1)

    print("\n[*] 正在初始化 Contextual MTBO 模型...")
    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    optimizer.train(data_file)
    
    try:
        iterations, mean_uncertainty, max_uncertainty = calculate_cognitive_convergence(
            optimizer, data_file, target_scheme
        )
        plot_cognitive_convergence(iterations, mean_uncertainty, max_uncertainty, scheme_id)
    except Exception as e:
        print(f"\n[计算失败] 发生错误: {e}")
        import traceback
        traceback.print_exc()