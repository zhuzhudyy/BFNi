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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import norm
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# 设置中文字体，防止图表中文字符显示为方块
# 使用 SimHei 显示中文，DejaVu Sans 显示负号等特殊字符
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True  # 使用 Unicode 负号，确保正确显示

from bo_optimization.contextual_bo_model import ContextualBayesianOptimizer, select_scheme, SCHEME_TARGETS, DEFAULT_PROCESS_BOUNDS

def create_output_dir(scheme_id=None, n_samples=None):
    r"""
    创建带时间戳的输出文件夹到指定路径
    文件夹命名: D:\毕业设计\织构数据\visualization\model_visualization
    子文件夹命名: Scheme{ID}_N{样本量}_{时间戳}
    """
    # 基础输出路径
    if os.path.exists(r"D:\毕业设计\织构数据"):
        base_output_path = r"D:\毕业设计\织构数据\visualization\model_visualization"
    else:
        base_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "model_visualization")
    
    # 确保基础路径存在
    os.makedirs(base_output_path, exist_ok=True)
    
    # 创建带方案ID、样本量和时间戳的子文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if scheme_id is not None and n_samples is not None:
        folder_name = f"Scheme{scheme_id}_N{n_samples}_{timestamp}"
    else:
        folder_name = f"viz_{timestamp}"
    output_dir = os.path.join(base_output_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] 创建可视化输出文件夹: {output_dir}")
    return output_dir


def plot_model_parity(optimizer, data_file, output_dir):
    """
    绘制真实值 vs 预测值的对角线图 (Parity Plot)，使用LOOCV评估真实预测能力
    
    注意：不使用optimizer.gpr直接预测，因为GPR是精确插值器，
    对训练数据预测会得到完美结果（预测值=真实值，sigma≈0）。
    使用LOOCV才能真实反映模型的泛化能力。
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import LeaveOneOut
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(data_file)
    # 构建完整的特征矩阵（包含预处理特征、目标晶向One-Hot、工艺参数）
    all_feature_cols = optimizer.pre_feature_cols + optimizer.target_cols + optimizer.process_cols
    X = df[all_feature_cols].values
    y_true = df['TARGET_Yield'].values
    
    print(f"    执行LOOCV（留一法交叉验证），样本数: {len(df)}...")
    
    # 使用LOOCV进行真实预测能力评估
    loo = LeaveOneOut()
    y_pred = np.zeros(len(df))
    sigma = np.zeros(len(df))
    
    for train_idx, val_idx in loo.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y_true[train_idx]
        
        # 为当前fold创建新的scaler和模型
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 训练新模型（使用与optimizer相同的核函数参数）
        gpr = GaussianProcessRegressor(
            kernel=optimizer.gpr.kernel_,  # 使用已优化的核函数
            n_restarts_optimizer=0,  # 不需要重新优化
            normalize_y=True,
            random_state=42
        )
        gpr.fit(X_train_scaled, y_train)
        
        # 预测验证样本（X_val_scaled是(1, n_features)，返回的是(1,)数组）
        pred_val, std_val = gpr.predict(X_val_scaled, return_std=True)
        y_pred[val_idx[0]] = pred_val[0] if hasattr(pred_val, '__len__') else pred_val
        sigma[val_idx[0]] = std_val[0] if hasattr(std_val, '__len__') else std_val
    
    print(f"    LOOCV完成 - MAE: {np.mean(np.abs(y_true - y_pred)):.4f}, RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.4f}")
    
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


# DEPRECATED: plot_2d_landscape & plot_process_response_3d 将 Pre_ 特征固定在均值上
# 扫工艺参数，违反了"工艺→微观结构→产率"的物理因果链，3D 曲面不具有物理意义。
# 推荐使用 honest_visualization.py 中的 PDP/ICE/ARD 替代方案。
def plot_2d_landscape(optimizer, data_file, output_dir, param_x='Process_Temp', param_y='Process_Time',
                       grid_size=100, target_scheme=None):
    """
    [DEPRECATED] 绘制2D参数空间的响应面（固定 Pre_ 特征为均值，缺乏物理意义）

    参数:
        target_scheme: 目标晶向方案，如 [(1,0,3), (1,0,2), (3,0,1)]
                      如果为None，则使用所有数据的均值（不推荐）
    """
    df = pd.read_csv(data_file)
    
    # 如果指定了目标晶向方案，筛选对应数据
    if target_scheme is not None and optimizer.target_cols:
        # 构建目标晶向的Multi-Hot编码
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme]
        
        # 筛选包含这些目标晶向的数据
        mask = np.ones(len(df), dtype=bool)
        for key in target_keys:
            if key in df.columns:
                mask = mask & (df[key] == 1.0)
        
        df_filtered = df[mask].copy()
        if len(df_filtered) == 0:
            print(f"[!] 警告: 没有找到目标晶向 {target_scheme} 的数据，使用全部数据")
            df_filtered = df
        else:
            print(f"[*] 响应面针对目标晶向: {target_scheme} (使用 {len(df_filtered)} 个样本)")
        
        # 计算该目标晶向的历史最优产率（winsorize 防异常值）
        y_values = df_filtered['TARGET_Yield'].values
        y_best = min(y_values.max(), np.percentile(y_values, 95))
        
        # 构建目标晶向的Multi-Hot编码
        target_encoding = np.array([1.0 if col in target_keys else 0.0 for col in optimizer.target_cols])
    else:
        df_filtered = df
        y_values = df['TARGET_Yield'].values
        y_best = min(y_values.max(), np.percentile(y_values, 95))
        if optimizer.target_cols:
            target_encoding = df[optimizer.target_cols].mean().values
            print(f"[!] 警告: 未指定目标晶向，使用所有数据的均值")
        else:
            target_encoding = np.array([])
    
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
    
    # 将预处理特征固定为均值（使用筛选后的数据）
    pre_features_mean = df_filtered[optimizer.pre_feature_cols].mean().values
    
    # 将未被选作X或Y轴的其他工艺参数固定为历史均值
    process_means = df_filtered[optimizer.process_cols].mean().values
    
    # 构造输入矩阵
    n_points = grid_size * grid_size
    X_process_array = np.tile(process_means, (n_points, 1))
    X_process_array[:, idx_x] = X_grid.ravel()
    X_process_array[:, idx_y] = Y_grid.ravel()
    
    # 计算 EI（传入目标晶向特征）
    ei, mu, sigma = optimizer.expected_improvement(X_process_array, pre_features_mean, y_best, target_encoding)
    
    # 转换为网格形状
    Mu_grid = mu.reshape(grid_size, grid_size)
    Sigma_grid = sigma.reshape(grid_size, grid_size)
    EI_grid = ei.reshape(grid_size, grid_size)
    
    # 构建标题后缀
    scheme_str = f" - {target_scheme}" if target_scheme else ""
    
    # ====================
    # 绘制三维曲面图 (3D Surface) - 改进版
    # ====================
    fig_3d = plt.figure(figsize=(20, 6))
    
    # 1. 预测均值 3D - 凸型山峰（产率越高山峰越高）
    ax1 = fig_3d.add_subplot(131, projection='3d')
    # 添加光照效果使曲面更有立体感
    surf1 = ax1.plot_surface(X_grid, Y_grid, Mu_grid, cmap='viridis', 
                              edgecolor='none', alpha=0.95, antialiased=True,
                              rstride=2, cstride=2, linewidth=0)
    # 添加底部等高线投影
    ax1.contour(X_grid, Y_grid, Mu_grid, zdir='z', 
                offset=Mu_grid.min(), cmap='viridis', alpha=0.5, levels=10)
    ax1.set_xlabel(param_x, fontsize=10)
    ax1.set_ylabel(param_y, fontsize=10)
    ax1.set_zlabel('预测产率', fontsize=10)
    ax1.set_title(f'预测目标产率均值{scheme_str}', fontsize=12)
    fig_3d.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='产率')
    # 添加历史实验点（抬高一点避免被曲面遮挡）
    if len(df_filtered) > 0:
        z_offset = 0.02  # 抬高偏移量
        ax1.scatter(df_filtered[param_x], df_filtered[param_y], 
                   df_filtered['TARGET_Yield'] + z_offset, c='red', s=80, 
                   edgecolor='white', linewidth=2, marker='o', 
                   label='历史实验点', depthshade=False)
    # 设置视角
    ax1.view_init(elev=30, azim=45)
    
    # 2. 不确定度 3D - 反转显示：置信度/确定性 (凸型图)
    ax2 = fig_3d.add_subplot(132, projection='3d')
    # 将不确定度反转为"置信度"：σ越小 → 值越大（山峰）
    # 使用倒数变换：Confidence = 1 / (1 + σ)，范围在 (0, 1]
    Confidence_grid = 1.0 / (1.0 + Sigma_grid)
    surf2 = ax2.plot_surface(X_grid, Y_grid, Confidence_grid, cmap='inferno', 
                              edgecolor='none', alpha=0.95, antialiased=True,
                              rstride=2, cstride=2, linewidth=0)
    # 在底部添加等高线投影
    ax2.contour(X_grid, Y_grid, Confidence_grid, zdir='z', 
                offset=0, cmap='inferno', alpha=0.5, levels=10)
    ax2.set_xlabel(param_x, fontsize=10)
    ax2.set_ylabel(param_y, fontsize=10)
    ax2.set_zlabel('模型置信度', fontsize=10)
    ax2.set_title(f'模型置信度{scheme_str}', fontsize=12)
    ax2.set_zlim(0, 1)
    fig_3d.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='置信度 (1/(1+σ))')
    # 标记历史实验点位置（置信度最高点）
    if len(df_filtered) > 0:
        for _, row in df_filtered.iterrows():
            x_val, y_val = row[param_x], row[param_y]
            i = np.argmin(np.abs(y_vals - y_val))
            j = np.argmin(np.abs(x_vals - x_val))
            conf_val = Confidence_grid[i, j]
            # 采样点处置信度最高（山峰顶端）
            ax2.scatter([x_val], [y_val], [conf_val], c='cyan', s=120, 
                       edgecolor='white', linewidth=2, marker='^', 
                       label='采样点(高置信度)' if _ == df_filtered.index[0] else "")
    ax2.view_init(elev=35, azim=60)
    
    # 3. EI 3D - 凸型山峰效果（将底部从-0.1提升到0，形成完整山峰）
    ax3 = fig_3d.add_subplot(133, projection='3d')
    # 对EI进行归一化到 [0, 1] 范围，确保是凸型（底部为0，顶部为1）
    EI_enhanced = EI_grid / (EI_grid.max() + 1e-10)
    surf3 = ax3.plot_surface(X_grid, Y_grid, EI_enhanced, cmap='magma', 
                              edgecolor='none', alpha=0.95, antialiased=True,
                              rstride=2, cstride=2, linewidth=0)
    # 添加等高线（从z=0开始，形成完整山峰轮廓）
    ax3.contour(X_grid, Y_grid, EI_enhanced, zdir='z', 
                offset=0, cmap='magma', alpha=0.6, levels=15)
    ax3.set_xlabel(param_x, fontsize=10)
    ax3.set_ylabel(param_y, fontsize=10)
    ax3.set_zlabel('期望提升 (EI)', fontsize=10)
    ax3.set_title(f'期望提升采集函数{scheme_str}', fontsize=12)
    ax3.set_zlim(0, 1)  # 固定范围确保凸型效果
    fig_3d.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label='EI (归一化)')
    # 标出最大EI点（山峰顶端）
    best_idx = np.argmax(ei)
    best_x = X_process_array[best_idx, idx_x]
    best_y = X_process_array[best_idx, idx_y]
    best_ei_norm = EI_enhanced.ravel()[best_idx]
    ax3.scatter([best_x], [best_y], [best_ei_norm], c='lime', s=300, 
               edgecolor='black', linewidth=2, marker='*', 
               label='推荐探索点(山顶)', depthshade=False)
    # 添加历史点（通常在山谷或山腰，因为EI在已采样点较低）
    if len(df_filtered) > 0:
        for _, row in df_filtered.iterrows():
            x_val, y_val = row[param_x], row[param_y]
            i = np.argmin(np.abs(y_vals - y_val))
            j = np.argmin(np.abs(x_vals - x_val))
            ei_val = EI_enhanced[i, j]
            # 历史点用红色标记，通常在EI较低位置
            ax3.scatter([x_val], [y_val], [ei_val], c='red', s=60, 
                       edgecolor='white', linewidth=1, marker='o', alpha=0.8)
    ax3.view_init(elev=40, azim=45)
    
    plt.tight_layout()
    base_name = os.path.basename(data_file).replace('.csv', '')
    save_path_3d = os.path.join(output_dir, f"{base_name}_Landscape_3D.png")
    plt.savefig(save_path_3d, dpi=300)
    print(f"[*] 三维响应面图已保存至: {save_path_3d}")
    plt.show()
    
    # ====================
    # 同时保留二维等高线图 (2D Contour)
    # ====================
    fig_2d, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 预测均值 (Mean)
    c1 = axes[0].contourf(X_grid, Y_grid, Mu_grid, levels=50, cmap='viridis')
    fig_2d.colorbar(c1, ax=axes[0])
    axes[0].set_title(f'预测目标产率均值{scheme_str}', fontsize=14)
    axes[0].set_xlabel(param_x)
    axes[0].set_ylabel(param_y)
    
    # 2. 不确定度 (Sigma)
    c2 = axes[1].contourf(X_grid, Y_grid, Sigma_grid, levels=50, cmap='inferno')
    fig_2d.colorbar(c2, ax=axes[1])
    axes[1].set_title(f'模型不确定度{scheme_str}', fontsize=14)
    axes[1].set_xlabel(param_x)
    axes[1].set_ylabel(param_y)
    
    # 3. 期望提升 (Expected Improvement)
    c3 = axes[2].contourf(X_grid, Y_grid, EI_grid, levels=50, cmap='magma')
    fig_2d.colorbar(c3, ax=axes[2])
    axes[2].set_title(f'期望提升采集函数{scheme_str}', fontsize=14)
    axes[2].set_xlabel(param_x)
    axes[2].set_ylabel(param_y)
    
    # 在图上标出已有实验数据的散点（使用筛选后的数据）
    axes[0].scatter(df_filtered[param_x], df_filtered[param_y], c='red', edgecolor='white', marker='o', label='历史实验点')
    axes[0].legend()
    axes[1].scatter(df_filtered[param_x], df_filtered[param_y], c='white', edgecolor='black', marker='o', alpha=0.5)
    axes[2].scatter(df_filtered[param_x], df_filtered[param_y], c='white', edgecolor='black', marker='o', alpha=0.5)
    
    # 标出全局推荐的最大EI点
    axes[2].scatter(best_x, best_y, c='cyan', edgecolor='black', marker='*', s=300, label='下一轮推荐探索点')
    axes[2].legend()
    
    plt.tight_layout()
    save_path_2d = os.path.join(output_dir, f"{base_name}_Landscape_2D.png")
    plt.savefig(save_path_2d, dpi=300)
    print(f"[*] 二维等高线图已保存至: {save_path_2d}")
    plt.show()

def plot_process_response_3d(optimizer, data_file, output_dir, grid_size=50, target_scheme=None):
    """[DEPRECATED] 绘制固定 Pre_ 均值的 3D 响应曲面 — 缺乏物理意义，推荐 honest_visualization.py"""
    df = pd.read_csv(data_file)
    pre_cols = optimizer.pre_feature_cols
    target_cols = optimizer.target_cols
    process_cols = optimizer.process_cols

    # 确定目标晶向编码与筛选数据
    if target_scheme is not None and target_cols:
        target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme] + ['Target_Scheme']
        # Target_Scheme 在数据中可能存在（multi-hot标记方案归属），但实际 target_encoding 不应包含它
        pure_target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_scheme]
        mask = np.ones(len(df), dtype=bool)
        for key in target_keys:
            if key in df.columns:
                mask = mask & (df[key] == 1.0)
        df_filtered = df[mask].copy()
        # 检查 Target_Scheme 是否在模型 target_cols 中，如果不在则排除
        model_target_keys = [k for k in pure_target_keys if k in target_cols]
        # 额外匹配：数据中可能用 Target_Scheme 但不一定在模型 target_cols
        target_encoding = np.array([1.0 if col in model_target_keys else 0.0 for col in target_cols])
    else:
        df_filtered = df
        target_encoding = np.zeros(len(target_cols)) if target_cols else np.array([])

    pre_mean = df_filtered[pre_cols].mean().values
    process_mean = df_filtered[process_cols].mean().values
    scheme_str = f" — {target_scheme}" if target_scheme else ""

    # 为每个历史实验计算模型预测值（用自身 Pre_ + Process_）
    preds = np.full(len(df_filtered), np.nan)
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        pre_i = row[pre_cols].values.astype(float)
        proc_i = row[process_cols].values.astype(float)
        X_full = np.hstack([pre_i, target_encoding, proc_i]).reshape(1, -1)
        Xs = optimizer.scaler_X.transform(X_full)
        mu, _ = optimizer.gpr.predict(Xs, return_std=True)
        preds[i] = mu[0]

    # 把各实验投影到均值 Pre_ 平面上（用于判断该点是否靠近曲面）
    proj_preds = np.full(len(df_filtered), np.nan)
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        proc_i = row[process_cols].values.astype(float)
        X_proj = np.hstack([pre_mean, target_encoding, proc_i]).reshape(1, -1)
        Xs = optimizer.scaler_X.transform(X_proj)
        mu, _ = optimizer.gpr.predict(Xs, return_std=True)
        proj_preds[i] = mu[0]

    # ---- 工具函数：构建 2D sweep 曲面 ----
    def make_surface(col_a, col_b, bounds_a, bounds_b):
        a_vals = np.linspace(bounds_a[0], bounds_a[1], grid_size)
        b_vals = np.linspace(bounds_b[0], bounds_b[1], grid_size)
        A_grid, B_grid = np.meshgrid(a_vals, b_vals)

        idx_a = process_cols.index(col_a)
        idx_b = process_cols.index(col_b)

        n = grid_size * grid_size
        X_proc = np.tile(process_mean, (n, 1))
        X_proc[:, idx_a] = A_grid.ravel()
        X_proc[:, idx_b] = B_grid.ravel()

        X_concat = np.hstack([
            np.tile(pre_mean, (n, 1)),
            np.tile(target_encoding, (n, 1)),
            X_proc
        ])
        X_scaled = optimizer.scaler_X.transform(X_concat)
        mu, _ = optimizer.gpr.predict(X_scaled, return_std=True)
        return a_vals, b_vals, A_grid, B_grid, mu.reshape(grid_size, grid_size)

    # ===== 图1: 产率 vs 温度 + 时间 =====
    _, _, X1, Y1, Z1 = make_surface(
        'Process_Temp', 'Process_Time',
        optimizer.bounds['Process_Temp'], optimizer.bounds['Process_Time'])

    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='viridis', edgecolor='none',
                              alpha=0.92, antialiased=True, rstride=1, cstride=1)
    ax1.contour(X1, Y1, Z1, zdir='z', offset=Z1.min(), cmap='viridis',
                alpha=0.4, levels=10)

    # 散点：模型对各实验的预测值（自身 Pre_ + 自身 Process_）
    ax1.scatter(df_filtered['Process_Temp'], df_filtered['Process_Time'],
               preds, c='darkred', s=55, edgecolor='white',
               linewidth=0.8, zorder=5, alpha=0.9, label='实验点 (模型预测)')

    ax1.set_xlabel('退火温度 (°C)', fontsize=11)
    ax1.set_ylabel('保温时间 (h)', fontsize=11)
    ax1.set_zlabel('预测产率', fontsize=11)
    ax1.set_title(f'产率 vs 温度 & 时间{scheme_str}', fontsize=14)
    ax1.view_init(elev=28, azim=135)
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='产率')
    ax1.legend(fontsize=9)
    plt.tight_layout()

    base_name = os.path.basename(data_file).replace('.csv', '')
    save1 = os.path.join(output_dir, f"{base_name}_Response3D_Temp-Time.png")
    plt.savefig(save1, dpi=300)
    print(f"[*] 温度-时间 3D 响应曲面已保存至: {save1}")
    plt.show()

    # ===== 图2: 产率 vs H2 + Ar =====
    _, _, X2, Y2, Z2 = make_surface(
        'Process_H2', 'Process_Ar',
        optimizer.bounds['Process_H2'], optimizer.bounds['Process_Ar'])

    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='plasma', edgecolor='none',
                              alpha=0.92, antialiased=True, rstride=1, cstride=1)
    ax2.contour(X2, Y2, Z2, zdir='z', offset=Z2.min(), cmap='plasma',
                alpha=0.4, levels=10)

    ax2.scatter(df_filtered['Process_H2'], df_filtered['Process_Ar'],
               preds, c='darkred', s=55, edgecolor='white',
               linewidth=0.8, zorder=5, alpha=0.9, label='实验点 (模型预测)')

    ax2.set_xlabel('H₂ 流量 (sccm)', fontsize=11)
    ax2.set_ylabel('Ar 流量 (sccm)', fontsize=11)
    ax2.set_zlabel('预测产率', fontsize=11)
    ax2.set_title(f'产率 vs H₂ & Ar{scheme_str}', fontsize=14)
    ax2.view_init(elev=28, azim=135)
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='产率')
    ax2.legend(fontsize=9)
    plt.tight_layout()

    save2 = os.path.join(output_dir, f"{base_name}_Response3D_H2-Ar.png")
    plt.savefig(save2, dpi=300)
    print(f"[*] H2-Ar 3D 响应曲面已保存至: {save2}")
    plt.show()


if __name__ == "__main__":
    # ==========================
    # 1. 初始化设置与模型加载
    # ==========================
    
    # 自动检测数据文件
    default_file = "Optimized_Training_Data.csv"
    
    if os.path.exists(default_file):
        # 存在统一数据文件，需要用户选择要可视化的方案
        data_file = default_file
        print(f"[*] 检测到统一数据文件: {data_file}")
        print("\n请选择要可视化的目标晶向方案:")
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

        target_scheme = SCHEME_TARGETS[current_scheme]
        print(f"[*] 已选择方案 {current_scheme}: {target_scheme}")
    else:
        # 没有统一数据文件，按方案分文件查找
        print("[*] 未检测到统一数据文件，请选择方案...")
        current_scheme = select_scheme()
        data_file = f"Optimized_Training_Data_方案{current_scheme}.csv"

        if not os.path.exists(data_file):
            print(f"\n[错误] 未找到数据文件: {data_file}")
            print("请确保数据文件存在于当前目录。")
            sys.exit(1)

        target_scheme = SCHEME_TARGETS[current_scheme]
        print(f"[*] 使用方案 {current_scheme} 数据文件: {data_file}")

    print("\n正在训练高斯过程模型...")
    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    optimizer.train(data_file)
    
    # 获取样本数量
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    # ==========================
    # 2. 创建输出文件夹
    # ==========================
    output_dir = create_output_dir(scheme_id=current_scheme, n_samples=n_samples)
    
    # ==========================
    # 3. 执行可视化绘图
    # ==========================
    print("\n[1/3] 正在绘制预测对角线图 (Parity Plot)...")
    plot_model_parity(optimizer, data_file, output_dir)

    # 使用 honest_visualization 替代废弃的 3D 曲面
    from bo_optimization.honest_visualization import (
        plot_yield_by_experiment, plot_partial_dependence,
        plot_raw_data_scatter, plot_ard_grouped
    )
    print("\n[2/3] 正在绘制物理诚实的可视化...")
    plot_yield_by_experiment(optimizer, output_dir, target_scheme=target_scheme)
    plot_raw_data_scatter(optimizer, output_dir)

    print("\n[3/3] 正在绘制 PDP 和 ARD 分析...")
    plot_partial_dependence(optimizer, output_dir, target_scheme=target_scheme)
    plot_ard_grouped(optimizer, output_dir)

    print(f"\n可视化分析完成！所有图片已保存至文件夹: {output_dir}/")
    print("  - 包含: Parity Plot (LOOCV验证)、实验产率柱状图、原始数据散点图、PDP、ARD")