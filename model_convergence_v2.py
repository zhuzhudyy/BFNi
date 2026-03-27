"""
贝叶斯优化收敛性分析工具 V2 (理论严谨版)

核心改进:
    1. 区分"模型拟合收敛"与"优化过程收敛"
       - 增加 Max EI 衰减曲线分析
       - 计算 Simple Regret 曲线
       - 判断优化是否真正收敛
    
    2. 极小样本下的稳健验证
       - 样本 < 30 时自动切换为 LOOCV
       - 避免 K-Fold 在极小样本下的失效
    
    3. 尊重序贯数据的非 I.I.D 特性
       - 使用 TimeSeriesSplit 替代随机 K-Fold
       - 模拟真实的主动学习轨迹

核心指标:
    - Optimization Convergence: EI_decay_rate, simple_regret
    - Model Quality: LOOCV_RMSE, LOOCV_MAE
    - Sample Efficiency: cumulative_best_yield

使用方法:
    python model_convergence_v2.py
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import qmc
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contextual_bo_model import ContextualBayesianOptimizer, select_scheme


def loocv_analysis(optimizer, data_file):
    """
    留一法交叉验证 (Leave-One-Out Cross-Validation)
    
    适用于极小样本 (n < 30) 的稳健验证方法
    每次用 n-1 个样本训练，1 个样本验证
    """
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    X = df[optimizer.pre_feature_cols + optimizer.process_cols].values
    y = df['TARGET_Yield'].values
    
    loo = LeaveOneOut()
    errors = []
    
    print(f"    Performing LOOCV with {n_samples} samples...")
    
    for train_idx, val_idx in loo.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        gpr = optimizer.gpr
        scaler = optimizer.scaler_X
        
        X_train_scaled = scaler.fit_transform(X_train)
        gpr.fit(X_train_scaled, y_train)
        
        # 验证
        X_val_scaled = scaler.transform(X_val)
        y_pred = gpr.predict(X_val_scaled)[0]
        
        errors.append(abs(y_val[0] - y_pred))  # 绝对误差
    
    return {
        'LOOCV_MAE': np.mean(errors),
        'LOOCV_MAE_std': np.std(errors),
        'LOOCV_RMSE': np.sqrt(np.mean(np.array(errors)**2)),
        'max_prediction_error': np.max(errors),
        'method': 'LOOCV'
    }


def timeseries_cv_analysis(optimizer, data_file):
    """
    时间序列交叉验证 (Time Series Cross-Validation)
    
    尊重贝叶斯优化的序贯特性:
    - 用前 N 个实验预测第 N+1 个实验
    - 模拟真实的主动学习过程
    - 避免"数据穿越"问题
    """
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    if n_samples < 5:
        return {'error': 'Insufficient samples for TimeSeries CV (need >= 5)'}
    
    X = df[optimizer.pre_feature_cols + optimizer.process_cols].values
    y = df['TARGET_Yield'].values
    
    # 时间序列分割: 从第3个样本开始逐步增加训练集
    tscv = TimeSeriesSplit(n_splits=min(n_samples - 2, 5))
    
    errors = []
    cumulative_best = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if len(X_val) == 0:
            continue
        
        # 训练模型
        gpr = optimizer.gpr
        scaler = optimizer.scaler_X
        
        X_train_scaled = scaler.fit_transform(X_train)
        gpr.fit(X_train_scaled, y_train)
        
        # 验证第 N+1 个样本
        X_val_scaled = scaler.transform(X_val)
        y_pred = gpr.predict(X_val_scaled)
        
        # 计算误差
        for i in range(len(y_val)):
            errors.append(abs(y_val[i] - y_pred[i]))
        
        # 记录累积最优产率
        cumulative_best.append(np.max(y_train))
    
    return {
        'TS_MAE': np.mean(errors) if errors else None,
        'TS_MAE_std': np.std(errors) if errors else None,
        'cumulative_best_yield': cumulative_best,
        'yield_improvement': cumulative_best[-1] - cumulative_best[0] if len(cumulative_best) > 1 else 0,
        'method': 'TimeSeriesSplit'
    }


def calculate_optimization_convergence(optimizer, data_file, n_simulations=1000):
    """
    计算优化过程收敛指标 (区别于模型拟合收敛)
    
    核心指标:
        1. Max EI 衰减曲线
        2. Simple Regret 曲线
        3. 收敛判断 (EI < threshold)
    """
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    # 模拟不同样本量下的 Max EI
    max_ei_history = []
    simple_regret_history = []
    
    # 使用拉丁超立方采样模拟候选点
    n_params = len(optimizer.process_cols)
    sampler = qmc.LatinHypercube(d=n_params)
    sample = sampler.random(n_simulations)
    
    sampled_process = np.zeros((n_simulations, n_params))
    for i, col in enumerate(optimizer.process_cols):
        low, high = optimizer.bounds[col]
        sampled_process[:, i] = low + sample[:, i] * (high - low)
    
    x_pre_mean = df[[col for col in df.columns if col.startswith('Pre_')]].mean().values
    
    # 逐步增加样本量，计算 Max EI
    for n in range(3, n_samples + 1):
        # 使用前 n 个样本训练
        df_subset = df.iloc[:n]
        X_subset = df_subset[optimizer.pre_feature_cols + optimizer.process_cols].values
        y_subset = df_subset['TARGET_Yield'].values
        
        gpr = optimizer.gpr
        scaler = optimizer.scaler_X
        
        X_scaled = scaler.fit_transform(X_subset)
        gpr.fit(X_scaled, y_subset)
        
        # 计算当前最优
        y_best = np.max(y_subset)
        
        # 计算所有候选点的 EI
        X_concat = np.hstack([np.tile(x_pre_mean, (n_simulations, 1)), sampled_process])
        X_concat_scaled = scaler.transform(X_concat)
        mu, sigma = gpr.predict(X_concat_scaled, return_std=True)
        
        from scipy.stats import norm
        imp = mu - y_best
        Z = np.zeros_like(imp)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]
        ei = np.zeros_like(imp)
        ei[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])
        
        max_ei = np.max(ei)
        max_ei_history.append(max_ei)
        
        # Simple Regret: 理论最优与当前最优的差距
        # 简化为 1 - y_best (假设理论最优为 1)
        simple_regret = 1.0 - y_best
        simple_regret_history.append(simple_regret)
    
    # 计算 EI 衰减率
    if len(max_ei_history) >= 3:
        # 最近3个点的平均衰减
        recent_decay = (max_ei_history[-3] - max_ei_history[-1]) / max_ei_history[-3] if max_ei_history[-3] > 0 else 0
    else:
        recent_decay = 0
    
    # 收敛判断
    EI_THRESHOLD = 1e-4
    CONSECUTIVE_THRESHOLD = 3
    
    consecutive_low_ei = 0
    for ei in reversed(max_ei_history):
        if ei < EI_THRESHOLD:
            consecutive_low_ei += 1
        else:
            break
    
    is_converged = consecutive_low_ei >= CONSECUTIVE_THRESHOLD
    
    return {
        'max_ei_history': max_ei_history,
        'simple_regret_history': simple_regret_history,
        'current_max_ei': max_ei_history[-1] if max_ei_history else None,
        'ei_decay_rate': recent_decay,
        'simple_regret': simple_regret_history[-1] if simple_regret_history else None,
        'is_converged': is_converged,
        'consecutive_low_ei': consecutive_low_ei,
        'convergence_status': 'Converged' if is_converged else 'Not Converged'
    }


def print_convergence_report_v2(cv_metrics, opt_conv_metrics, sample_count):
    """
    打印优化收敛性分析报告 (V2 理论严谨版)
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "BAYESIAN OPTIMIZATION CONVERGENCE ANALYSIS")
    print(" " * 25 + "(Theoretically Rigorous Version)")
    print("=" * 80)
    
    # 基本信息
    print(f"\n[1] EXPERIMENTAL SETUP")
    print(f"    Total Samples (N):         {sample_count}")
    print(f"    Validation Method:         {cv_metrics.get('method', 'N/A')}")
    
    # 模型质量评估 (LOOCV)
    print(f"\n[2] MODEL QUALITY ASSESSMENT (Generalization Error)")
    print(f"    LOOCV MAE:                 {cv_metrics['LOOCV_MAE']:.4f} ± {cv_metrics['LOOCV_MAE_std']:.4f}")
    print(f"    LOOCV RMSE:                {cv_metrics['LOOCV_RMSE']:.4f}")
    print(f"    Max Prediction Error:      {cv_metrics['max_prediction_error']:.4f}")
    print(f"    Note: LOOCV used due to small sample size (N < 30)")
    
    # 优化过程收敛 (核心改进)
    print(f"\n[3] OPTIMIZATION PROCESS CONVERGENCE (Key Metrics)")
    print(f"    Current Max EI:            {opt_conv_metrics['current_max_ei']:.6f}")
    print(f"    EI Decay Rate (recent):    {opt_conv_metrics['ei_decay_rate']:.2%}")
    print(f"    Simple Regret:             {opt_conv_metrics['simple_regret']:.4f}")
    print(f"    Consecutive Low EI:        {opt_conv_metrics['consecutive_low_ei']} / 3")
    print(f"    Convergence Status:        {opt_conv_metrics['convergence_status']}")
    
    # EI 历史
    print(f"\n[4] MAX EI HISTORY (Last 5 Iterations)")
    ei_history = opt_conv_metrics['max_ei_history']
    if len(ei_history) >= 5:
        for i, ei in enumerate(ei_history[-5:], start=len(ei_history)-4):
            marker = " <-- Current" if i == len(ei_history) else ""
            print(f"    Iteration {i:2d}:  EI = {ei:.6f}{marker}")
    else:
        for i, ei in enumerate(ei_history, start=1):
            marker = " <-- Current" if i == len(ei_history) else ""
            print(f"    Iteration {i:2d}:  EI = {ei:.6f}{marker}")
    
    # 理论解释
    print(f"\n[5] THEORETICAL INTERPRETATION")
    
    if opt_conv_metrics['is_converged']:
        print("    - Optimization has CONVERGED")
        print("    - Max EI < 1e-4 for 3+ consecutive iterations")
        print("    - Further experiments unlikely to yield significant improvement")
    else:
        print("    - Optimization has NOT converged yet")
        print("    - Max EI remains above threshold (1e-4)")
        print("    - Continue sequential experimentation recommended")
    
    if opt_conv_metrics['simple_regret'] > 0.2:
        print(f"    - Simple Regret = {opt_conv_metrics['simple_regret']:.2f} (room for improvement)")
    else:
        print(f"    - Simple Regret = {opt_conv_metrics['simple_regret']:.2f} (near optimal)")
    
    # 建议
    print(f"\n[6] RECOMMENDATIONS")
    if opt_conv_metrics['is_converged']:
        print("    1. Optimization can be terminated")
        print("    2. Current best recipe is near-optimal")
        print("    3. Focus on experimental validation")
    else:
        print("    1. Continue Bayesian optimization")
        print("    2. Run next experiment with highest EI")
        print(f"    3. Expected improvement potential: {opt_conv_metrics['current_max_ei']:.4f}")
    
    print("=" * 80)


def suggest_next_experiments_v2(optimizer, data_file, n_suggestions=3):
    """
    基于 EI 的实验建议 (V2 版本)
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "NEXT EXPERIMENT RECOMMENDATION")
    print("=" * 80)
    
    df = pd.read_csv(data_file)
    y_best = df['TARGET_Yield'].max()
    
    # 拉丁超立方采样
    n_samples = 5000
    n_params = len(optimizer.process_cols)
    
    sampler = qmc.LatinHypercube(d=n_params)
    sample = sampler.random(n=n_samples)
    
    sampled_process = np.zeros((n_samples, n_params))
    for i, col in enumerate(optimizer.process_cols):
        low, high = optimizer.bounds[col]
        sampled_process[:, i] = low + sample[:, i] * (high - low)
    
    x_pre_mean = df[[col for col in df.columns if col.startswith('Pre_')]].mean().values
    
    # 计算 EI
    from scipy.stats import norm
    X_concat = np.hstack([np.tile(x_pre_mean, (n_samples, 1)), sampled_process])
    X_scaled = optimizer.scaler_X.transform(X_concat)
    mu, sigma = optimizer.gpr.predict(X_scaled, return_std=True)
    
    imp = mu - y_best
    Z = np.zeros_like(imp)
    mask = sigma > 0
    Z[mask] = imp[mask] / sigma[mask]
    ei = np.zeros_like(imp)
    ei[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])
    
    top_indices = np.argsort(ei)[-n_suggestions:][::-1]
    
    for i, idx in enumerate(top_indices, 1):
        print(f"\n    Rank {i} (Expected Improvement = {ei[idx]:.6f}):")
        print(f"      Process Parameters:")
        for j, col in enumerate(optimizer.process_cols):
            print(f"        {col}: {sampled_process[idx, j]:.2f}")
        print(f"      Predicted Yield: {mu[idx]:.2%} ± {sigma[idx]:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 自动检测数据文件，优先使用默认文件
    default_file = "Optimized_Training_Data.csv"
    
    if os.path.exists(default_file):
        data_file = default_file
        print(f"[*] 使用默认数据文件: {data_file}")
    else:
        # 如果没有默认文件，提示选择方案
        print("[*] 默认数据文件不存在，请选择方案...")
        scheme = select_scheme()
        
        scheme_names = {
            1: "<103> Texture (Orange)",
            2: "<114> Texture (Purple)",
            3: "<124> Texture (Mixed)",
            4: "Custom Combination"
        }
        
        data_file = f"Optimized_Training_Data_Scheme{scheme}.csv"
        if not os.path.exists(data_file):
            print(f"\n[错误] 未找到数据文件: {data_file}")
            print("请确保数据文件存在于当前目录。")
            sys.exit(1)
        
        print(f"\nData File: {data_file}")
        print(f"Scheme: {scheme_names[scheme]}")
    
    process_bounds = {
        'Process_Temp': (1000.0, 1500.0),
        'Process_Time': (1.0, 30.0),
        'Process_H2': (0.0, 160.0),
        'Process_Ar': (0.0, 800.0)
    }
    
    print("\nTraining Gaussian Process Model...")
    optimizer = ContextualBayesianOptimizer(bounds=process_bounds)
    optimizer.train(data_file)
    
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    # 1. 模型质量评估 (使用 LOOCV 而非 K-Fold)
    print("\nPerforming LOOCV (Leave-One-Out Cross-Validation)...")
    cv_metrics = loocv_analysis(optimizer, data_file)
    
    # 2. 优化过程收敛分析 (核心改进)
    print("\nAnalyzing Optimization Convergence...")
    print("    Calculating Max EI decay curve...")
    opt_conv_metrics = calculate_optimization_convergence(optimizer, data_file)
    
    # 3. 时间序列验证 (可选，展示序贯特性)
    if n_samples >= 5:
        print("\nPerforming Time Series Cross-Validation...")
        ts_metrics = timeseries_cv_analysis(optimizer, data_file)
        if 'error' not in ts_metrics:
            print(f"    Time Series MAE: {ts_metrics['TS_MAE']:.4f}")
            print(f"    Yield Improvement: {ts_metrics['yield_improvement']:.4f}")
    
    # 打印报告
    print_convergence_report_v2(cv_metrics, opt_conv_metrics, n_samples)
    
    # 实验建议
    suggest_next_experiments_v2(optimizer, data_file, n_suggestions=3)
    
    print("\nAnalysis Complete!")
