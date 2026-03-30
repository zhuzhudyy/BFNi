"""
贝叶斯优化收敛性可视化工具 (Convergence Visualization)

功能:
    1. Max EI 衰减曲线 - 判断优化是否收敛
    2. Simple Regret 曲线 - 评估优化效率
    3. 累积最优产率曲线 - 展示优化进展
    4. 模型预测误差随样本变化 - 评估模型拟合收敛

使用方法:
    python convergence_visualization.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import qmc, norm
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contextual_bo_model import ContextualBayesianOptimizer


def create_output_dir():
    """创建带时间戳的输出文件夹"""
    base_dir = r"D:\毕业设计\织构数据\convergence"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"conv_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] 创建收敛性可视化输出文件夹: {output_dir}")
    return output_dir


def calculate_convergence_metrics(data_file, optimizer, n_simulations=2000):
    """
    计算收敛性指标
    
    返回:
        metrics: 包含各种收敛指标的字典
    """
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    if n_samples < 5:
        print("[!] 样本数量不足 (需要 >= 5)，无法进行收敛性分析")
        return None
    
    # 准备特征
    feature_cols = optimizer.pre_feature_cols + optimizer.target_cols + optimizer.process_cols
    
    # 拉丁超立方采样生成候选点
    n_params = len(optimizer.process_cols)
    sampler = qmc.LatinHypercube(d=n_params)
    sample = sampler.random(n_simulations)
    
    sampled_process = np.zeros((n_simulations, n_params))
    for i, col in enumerate(optimizer.process_cols):
        low, high = optimizer.bounds[col]
        sampled_process[:, i] = low + sample[:, i] * (high - low)
    
    # 预处理特征均值
    x_pre_mean = df[optimizer.pre_feature_cols].mean().values
    
    # 目标晶向（使用Multi-Hot编码）
    if optimizer.target_cols:
        target_mean = df[optimizer.target_cols].mean().values
    else:
        target_mean = np.array([])
    
    # 逐步增加样本量，计算各项指标
    max_ei_history = []
    simple_regret_history = []
    cumulative_best = []
    loocv_errors = []
    
    print(f"    计算收敛指标，样本量从 3 到 {n_samples}...")
    
    for n in range(3, n_samples + 1):
        df_subset = df.iloc[:n]
        X_subset = df_subset[feature_cols].values
        y_subset = df_subset['TARGET_Yield'].values
        
        # 训练模型
        scaler = optimizer.scaler_X
        X_scaled = scaler.fit_transform(X_subset)
        optimizer.gpr.fit(X_scaled, y_subset)
        
        y_best = np.max(y_subset)
        cumulative_best.append(y_best)
        
        # 计算 Simple Regret (与全局最优的差距)
        # 使用当前已观测到的最优作为近似全局最优
        simple_regret = 1.0 - y_best  # 假设最大产率为1.0
        simple_regret_history.append(simple_regret)
        
        # 计算 Max EI
        X_concat = np.hstack([
            np.tile(x_pre_mean, (n_simulations, 1)),
            np.tile(target_mean, (n_simulations, 1)) if len(target_mean) > 0 else np.zeros((n_simulations, 0)),
            sampled_process
        ])
        X_concat_scaled = scaler.transform(X_concat)
        mu, sigma = optimizer.gpr.predict(X_concat_scaled, return_std=True)
        
        imp = mu - y_best
        Z = np.zeros_like(imp)
        mask = sigma > 1e-10
        Z[mask] = imp[mask] / sigma[mask]
        ei = np.zeros_like(imp)
        ei[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])
        
        max_ei = np.max(ei)
        max_ei_history.append(max_ei)
        
        # LOOCV 误差（每5个样本计算一次，节省计算）
        if n % 5 == 0 or n == n_samples:
            loo = LeaveOneOut()
            errors = []
            X_n = X_subset
            y_n = y_subset
            for train_idx, val_idx in loo.split(X_n):
                X_train, X_val = X_n[train_idx], X_n[val_idx]
                y_train, y_val = y_n[train_idx], y_n[val_idx]
                
                X_train_scaled = scaler.fit_transform(X_train)
                optimizer.gpr.fit(X_train_scaled, y_train)
                
                X_val_scaled = scaler.transform(X_val)
                y_pred = optimizer.gpr.predict(X_val_scaled)[0]
                errors.append(abs(y_val[0] - y_pred))
            
            loocv_errors.append({'n': n, 'mae': np.mean(errors), 'std': np.std(errors)})
    
    return {
        'n_samples': list(range(3, n_samples + 1)),
        'max_ei_history': max_ei_history,
        'simple_regret_history': simple_regret_history,
        'cumulative_best': cumulative_best,
        'loocv_errors': loocv_errors,
        'final_best': cumulative_best[-1],
        'total_samples': n_samples
    }


def plot_convergence_analysis(metrics, output_dir, target_scheme=None):
    """
    绘制收敛性分析图表
    """
    if metrics is None:
        return
    
    scheme_str = f" - {target_scheme}" if target_scheme else ""
    n_list = metrics['n_samples']
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Max EI 衰减曲线
    ax1 = axes[0, 0]
    ax1.semilogy(n_list, metrics['max_ei_history'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=0.01, color='r', linestyle='--', label='收敛阈值 (EI=0.01)')
    ax1.set_xlabel('样本数量', fontsize=11)
    ax1.set_ylabel('Max Expected Improvement (对数刻度)', fontsize=11)
    ax1.set_title(f'Max EI 衰减曲线{scheme_str}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加收敛判断
    final_ei = metrics['max_ei_history'][-1]
    if final_ei < 0.01:
        ax1.text(0.5, 0.95, '状态: 已收敛', transform=ax1.transAxes, 
                fontsize=10, color='green', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax1.text(0.5, 0.95, f'状态: 未收敛 (EI={final_ei:.4f})', transform=ax1.transAxes,
                fontsize=10, color='red', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. Simple Regret 曲线
    ax2 = axes[0, 1]
    ax2.plot(n_list, metrics['simple_regret_history'], 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('样本数量', fontsize=11)
    ax2.set_ylabel('Simple Regret (1 - Yield)', fontsize=11)
    ax2.set_title(f'Simple Regret 曲线{scheme_str}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加 regret 数值标注
    final_regret = metrics['simple_regret_history'][-1]
    ax2.text(0.7, 0.9, f'最终 Regret: {final_regret:.4f}', transform=ax2.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 累积最优产率曲线
    ax3 = axes[1, 0]
    ax3.plot(n_list, metrics['cumulative_best'], 'm-', linewidth=2, marker='^', markersize=5)
    ax3.fill_between(n_list, metrics['cumulative_best'], alpha=0.3, color='magenta')
    ax3.set_xlabel('样本数量', fontsize=11)
    ax3.set_ylabel('累积最优产率', fontsize=11)
    ax3.set_title(f'累积最优产率进展{scheme_str}', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # 标注最终产率
    final_yield = metrics['final_best']
    ax3.axhline(y=final_yield, color='r', linestyle='--', alpha=0.5)
    ax3.text(n_list[-1], final_yield + 0.02, f'{final_yield:.3f}', 
            ha='right', fontsize=10, color='red')
    
    # 4. LOOCV 误差随样本变化
    ax4 = axes[1, 1]
    if metrics['loocv_errors']:
        loocv_n = [e['n'] for e in metrics['loocv_errors']]
        loocv_mae = [e['mae'] for e in metrics['loocv_errors']]
        loocv_std = [e['std'] for e in metrics['loocv_errors']]
        
        ax4.errorbar(loocv_n, loocv_mae, yerr=loocv_std, fmt='o-', color='orange', 
                    linewidth=2, capsize=4, label='LOOCV MAE ± std')
        ax4.set_xlabel('样本数量', fontsize=11)
        ax4.set_ylabel('预测误差 (MAE)', fontsize=11)
        ax4.set_title(f'模型拟合收敛性 (LOOCV){scheme_str}', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 添加趋势判断
        if len(loocv_mae) >= 2:
            if loocv_mae[-1] < loocv_mae[0] * 0.8:
                trend_text = '趋势: 改善中'
                trend_color = 'green'
            elif loocv_mae[-1] > loocv_mae[0] * 1.2:
                trend_text = '趋势: 恶化 (可能过拟合)'
                trend_color = 'red'
            else:
                trend_text = '趋势: 稳定'
                trend_color = 'blue'
            
            ax4.text(0.05, 0.95, trend_text, transform=ax4.transAxes,
                    fontsize=10, color=trend_color, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax4.text(0.5, 0.5, '样本不足，无法计算 LOOCV', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(output_dir, f"Convergence_Analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 收敛性分析图已保存至: {save_path}")
    plt.show()
    
    return save_path


def print_convergence_report(metrics):
    """打印收敛性分析报告"""
    if metrics is None:
        return
    
    print("\n" + "="*60)
    print("           贝叶斯优化收敛性分析报告")
    print("="*60)
    
    # 1. 优化收敛状态
    final_ei = metrics['max_ei_history'][-1]
    print(f"\n【优化过程收敛状态】")
    print(f"  当前 Max EI: {final_ei:.6f}")
    if final_ei < 0.01:
        print(f"  状态: 已收敛 (EI < 0.01)")
        print(f"  建议: 可以停止采样，或进行验证实验")
    elif final_ei < 0.05:
        print(f"  状态: 接近收敛 (0.01 < EI < 0.05)")
        print(f"  建议: 再进行少量实验 (5-10次)")
    else:
        print(f"  状态: 未收敛 (EI > 0.05)")
        print(f"  建议: 继续进行贝叶斯优化采样")
    
    # 2. 优化效率
    print(f"\n【优化效率评估】")
    final_regret = metrics['simple_regret_history'][-1]
    print(f"  最终 Simple Regret: {final_regret:.4f}")
    print(f"  最终最优产率: {metrics['final_best']:.4f}")
    
    if len(metrics['simple_regret_history']) > 5:
        early_regret = np.mean(metrics['simple_regret_history'][:3])
        improvement = early_regret - final_regret
        print(f"  优化提升: {improvement:.4f} (早期Regret - 最终Regret)")
    
    # 3. 样本效率
    print(f"\n【样本效率】")
    print(f"  总样本数: {metrics['total_samples']}")
    
    # 计算产率提升速度
    yields = metrics['cumulative_best']
    if len(yields) > 10:
        first_half = np.mean(yields[:len(yields)//2])
        second_half = np.mean(yields[len(yields)//2:])
        print(f"  前半段平均产率: {first_half:.4f}")
        print(f"  后半段平均产率: {second_half:.4f}")
        print(f"  提升比例: {(second_half - first_half) / first_half * 100:.1f}%")
    
    # 4. 模型拟合质量
    if metrics['loocv_errors']:
        final_loocv = metrics['loocv_errors'][-1]
        print(f"\n【模型拟合质量 (LOOCV)】")
        print(f"  最终 MAE: {final_loocv['mae']:.4f} ± {final_loocv['std']:.4f}")
        if final_loocv['mae'] < 0.1:
            print(f"  评估: 拟合良好 (MAE < 0.1)")
        elif final_loocv['mae'] < 0.2:
            print(f"  评估: 拟合可接受 (0.1 < MAE < 0.2)")
        else:
            print(f"  评估: 拟合较差 (MAE > 0.2)，建议增加样本")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # 定义各方案的目标晶向
    scheme_targets = {
        1: [(1, 0, 3), (1, 0, 2), (3, 0, 1)],
        2: [(1, 1, 4), (1, 1, 5), (1, 0, 5)],
        3: [(1, 2, 4), (1, 2, 5), (2, 1, 4)],
        4: [(1, 0, 3), (1, 1, 4), (1, 2, 4)]
    }
    
    # 检测数据文件
    default_file = "Optimized_Training_Data.csv"
    
    if os.path.exists(default_file):
        data_file = default_file
        print(f"[*] 检测到统一数据文件: {data_file}")
        print("\n请选择要分析的目标晶向方案:")
        print("  方案 1: <103>, <102>, <301>")
        print("  方案 2: <114>, <115>, <105>")
        print("  方案 3: <124>, <125>, <214>")
        print("  方案 4: <103>, <114>, <124>")
        
        while True:
            try:
                scheme_input = input("\n请输入方案编号 (1-4，直接回车默认方案1): ").strip()
                if scheme_input == "":
                    current_scheme = 1
                else:
                    current_scheme = int(scheme_input)
                if current_scheme in [1, 2, 3, 4]:
                    break
                else:
                    print("[!] 请输入 1-4 之间的数字")
            except ValueError:
                print("[!] 请输入有效的数字")
        
        target_scheme = scheme_targets[current_scheme]
        print(f"[*] 已选择方案 {current_scheme}: {target_scheme}")
    else:
        print("[!] 未找到数据文件，请确保 Optimized_Training_Data.csv 存在")
        sys.exit(1)
    
    # 初始化模型
    process_bounds = {
        'Process_Temp': (1000.0, 1500.0),
        'Process_Time': (1.0, 30.0),
        'Process_H2': (0.0, 160.0),
        'Process_Ar': (0.0, 800.0)
    }
    
    print("\n正在初始化模型...")
    optimizer = ContextualBayesianOptimizer(bounds=process_bounds)
    optimizer.train(data_file)
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    # 计算收敛性指标
    print("\n正在计算收敛性指标...")
    metrics = calculate_convergence_metrics(data_file, optimizer)
    
    # 绘制可视化
    if metrics:
        print("\n正在绘制收敛性分析图...")
        plot_convergence_analysis(metrics, output_dir, target_scheme)
        
        # 打印报告
        print_convergence_report(metrics)
        
        print(f"\n所有结果已保存至: {output_dir}/")
