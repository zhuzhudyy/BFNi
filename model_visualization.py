"""
贝叶斯优化代理模型可视化工具 (Model Visualization)

功能:
    1. 真实值 vs 预测值 (Parity Plot)，包含模型预测的 95% 置信区间
    2. 2D 参数空间景观图 (Landscape Contour)，包含均值、标准差和期望提升 (EI)

使用前请确保安装了可视化库:
    pip install matplotlib seaborn
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# 设置中文字体，防止图表中文字符显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contextual_bo_model import ContextualBayesianOptimizer, select_scheme

def create_output_dir():
    """
    创建带时间戳的输出文件夹
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"visualization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] 创建可视化输出文件夹: {output_dir}/")
    return output_dir


def plot_model_parity(optimizer, data_file, output_dir):
    """
    绘制真实值 vs 预测值的对角线图 (Parity Plot)，包含不确定度
    """
    df = pd.read_csv(data_file)
    X_df = df[optimizer.pre_feature_cols + optimizer.process_cols]
    y_true = df['TARGET_Yield'].values
    
    # 预测均值与标准差
    X_scaled = optimizer.scaler_X.transform(X_df)
    y_pred, sigma = optimizer.gpr.predict(X_scaled, return_std=True)
    
    plt.figure(figsize=(8, 6))
    
    # 绘制误差棒 (95% 置信区间 = 1.96 * sigma)
    plt.errorbar(y_true, y_pred, yerr=1.96*sigma, fmt='o', color='blue', 
                 ecolor='lightblue', elinewidth=2, capsize=4, alpha=0.7, label='预测点 (含95%置信区间)')
    
    # 绘制理想对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测 (y=x)')
    
    plt.xlabel('真实目标产率 (True Yield)', fontsize=12)
    plt.ylabel('预测目标产率 (Predicted Yield)', fontsize=12)
    plt.title('代理模型预测能力评估 (Parity Plot)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 从数据文件名提取基础名称
    base_name = os.path.basename(data_file).replace('.csv', '')
    save_path = os.path.join(output_dir, f"{base_name}_ParityPlot.png")
    plt.savefig(save_path, dpi=300)
    print(f"[*] 预测对角线图已保存至: {save_path}")
    plt.show()


def plot_2d_landscape(optimizer, data_file, output_dir, param_x='Process_Temp', param_y='Process_Time', grid_size=100):
    """
    绘制2D参数空间的响应面（固定其他参数为最优值或均值）
    """
    df = pd.read_csv(data_file)
    y_best = df['TARGET_Yield'].max()
    
    # 确定参数的索引
    idx_x = optimizer.process_cols.index(param_x)
    idx_y = optimizer.process_cols.index(param_y)
    
    # 获取参数边界
    bounds_x = optimizer.bounds[param_x]
    bounds_y = optimizer.bounds[param_y]
    
    # 生成网格
    x_vals = np.linspace(bounds_x[0], bounds_x[1], grid_size)
    y_vals = np.linspace(bounds_y[0], bounds_y[1], grid_size)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    
    # 将预处理特征固定为均值
    pre_features_mean = df[optimizer.pre_feature_cols].mean().values
    
    # 将未被选作X或Y轴的其他工艺参数固定为已知的最优组合附近（这里简化为历史均值）
    process_means = df[optimizer.process_cols].mean().values
    
    # 构造输入矩阵
    n_points = grid_size * grid_size
    X_process_array = np.tile(process_means, (n_points, 1))
    X_process_array[:, idx_x] = X_grid.ravel()
    X_process_array[:, idx_y] = Y_grid.ravel()
    
    # 计算 EI
    ei, mu, sigma = optimizer.expected_improvement(X_process_array, pre_features_mean, y_best)
    
    # 转换为网格形状
    Mu_grid = mu.reshape(grid_size, grid_size)
    Sigma_grid = sigma.reshape(grid_size, grid_size)
    EI_grid = ei.reshape(grid_size, grid_size)
    
    # 开始绘图 (1行3列)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 预测均值 (Mean)
    c1 = axes[0].contourf(X_grid, Y_grid, Mu_grid, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title(f'预测目标产率均值 (Mean)', fontsize=14)
    axes[0].set_xlabel(param_x)
    axes[0].set_ylabel(param_y)
    
    # 2. 不确定度 (Sigma)
    c2 = axes[1].contourf(X_grid, Y_grid, Sigma_grid, levels=50, cmap='inferno')
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title(f'模型不确定度 (Standard Deviation)', fontsize=14)
    axes[1].set_xlabel(param_x)
    axes[1].set_ylabel(param_y)
    
    # 3. 期望提升 (Expected Improvement)
    c3 = axes[2].contourf(X_grid, Y_grid, EI_grid, levels=50, cmap='magma')
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title(f'期望提升采集函数 (EI)', fontsize=14)
    axes[2].set_xlabel(param_x)
    axes[2].set_ylabel(param_y)
    
    # 在图上标出已有实验数据的散点
    axes[0].scatter(df[param_x], df[param_y], c='red', edgecolor='white', marker='o', label='历史实验点')
    axes[0].legend()
    axes[1].scatter(df[param_x], df[param_y], c='white', edgecolor='black', marker='o', alpha=0.5)
    axes[2].scatter(df[param_x], df[param_y], c='white', edgecolor='black', marker='o', alpha=0.5)
    
    # 标出全局推荐的最大EI点
    best_idx = np.argmax(ei)
    best_x = X_process_array[best_idx, idx_x]
    best_y = X_process_array[best_idx, idx_y]
    axes[2].scatter(best_x, best_y, c='cyan', edgecolor='black', marker='*', s=300, label='下一轮推荐探索点')
    axes[2].legend()
    
    plt.tight_layout()
    
    # 从数据文件名提取基础名称
    base_name = os.path.basename(data_file).replace('.csv', '')
    save_path = os.path.join(output_dir, f"{base_name}_Landscape.png")
    plt.savefig(save_path, dpi=300)
    print(f"[*] 响应面景观图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # ==========================
    # 1. 初始化设置与模型加载
    # ==========================
    # 自动检测数据文件，优先使用默认文件
    default_file = "Optimized_Training_Data.csv"
    
    if os.path.exists(default_file):
        data_file = default_file
        print(f"[*] 使用默认数据文件: {data_file}")
    else:
        # 如果没有默认文件，提示选择方案
        print("[*] 默认数据文件不存在，请选择方案...")
        scheme = select_scheme()
        data_file = f"Optimized_Training_Data_方案{scheme}.csv"
        
        if not os.path.exists(data_file):
            print(f"\n[错误] 未找到数据文件: {data_file}")
            print("请确保数据文件存在于当前目录。")
            sys.exit(1)
        
    process_bounds = {
        'Process_Temp': (1000.0, 1500.0),  
        'Process_Time': (1.0, 30.0),       
        'Process_H2': (0.0, 160.0),        
        'Process_Ar': (0.0, 800.0)         
    }
    
    print("\n正在训练高斯过程模型...")
    optimizer = ContextualBayesianOptimizer(bounds=process_bounds)
    optimizer.train(data_file)
    
    # ==========================
    # 2. 创建输出文件夹
    # ==========================
    output_dir = create_output_dir()
    
    # ==========================
    # 3. 执行可视化绘图
    # ==========================
    print("\n[1/2] 正在绘制预测对角线图 (Parity Plot)...")
    plot_model_parity(optimizer, data_file, output_dir)
    
    print("\n[2/2] 正在绘制二维工艺参数景观图 (Temp vs Time)...")
    # 这里默认绘制 退火温度(Temp) 和 保温时间(Time) 的关系
    plot_2d_landscape(optimizer, data_file, output_dir, param_x='Process_Temp', param_y='Process_Time')
    
    print(f"\n可视化分析完成！所有图片已保存至文件夹: {output_dir}/")