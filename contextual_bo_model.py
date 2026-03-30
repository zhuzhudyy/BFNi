import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings

# 忽略高斯过程在极低方差区域可能产生的数值计算警告
warnings.filterwarnings("ignore")

class ContextualBayesianOptimizer:
    def __init__(self, bounds, target_orientations=None):
        """
        初始化上下文贝叶斯优化器（支持多任务学习）
        
        Args:
            bounds: 字典形式的工艺参数物理搜索边界
            target_orientations: 所有可能的目标晶向列表，用于One-Hot编码
        """
        self.bounds = bounds
        self.process_cols = list(bounds.keys())
        self.pre_feature_cols = []
        self.target_cols = []  # 目标晶向One-Hot特征列
        self.target_orientations = target_orientations or []
        
        # 定义核函数：常数核 * Matern核 + 白噪声核
        # Matern(nu=2.5) 适用于具有二阶可导性的物理过程拟合
        # WhiteKernel 用于吸收实验测量中的随机系统误差（噪音）
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
        
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            normalize_y=True,
            random_state=42
        )
        
        self.scaler_X = StandardScaler()
        
    def train(self, csv_path):
        """
        读取历史实验数据并拟合代理模型（支持多任务）
        """
        self.training_df = pd.read_csv(csv_path)
        
        # 自动识别各类特征列
        self.pre_feature_cols = [col for col in self.training_df.columns if col.startswith('Pre_')]
        self.target_cols = [col for col in self.training_df.columns if col.startswith('Target_')]
        
        # 构建特征矩阵 X 与目标向量 y
        # 顺序: EBSD预处理特征 + 目标晶向One-Hot + 工艺参数
        all_feature_cols = self.pre_feature_cols + self.target_cols + self.process_cols
        X_df = self.training_df[all_feature_cols]
        y = self.training_df['TARGET_Yield'].values
        
        # 对输入空间进行标准化处理，加速核函数收敛
        X_scaled = self.scaler_X.fit_transform(X_df)
        
        # 模型拟合
        self.gpr.fit(X_scaled, y)
        print(f"代理模型训练完毕。吸收样本量: {len(self.training_df)}，总特征维度: {len(all_feature_cols)}。")
        print(f"  - EBSD预处理特征: {len(self.pre_feature_cols)} 维")
        print(f"  - 目标晶向One-Hot: {len(self.target_cols)} 维")
        print(f"  - 工艺参数: {len(self.process_cols)} 维")
        print(f"优化后核函数参数: {self.gpr.kernel_}")

    def expected_improvement(self, X_process_array, x_pre_array, y_best, target_onehot=None):
        """
        计算期望提升 (Expected Improvement, EI) 采集函数（支持多任务）
        
        Args:
            X_process_array: 工艺参数数组
            x_pre_array: EBSD预处理特征数组
            y_best: 当前最优产率
            target_onehot: 目标晶向One-Hot编码（可选，默认为全0）
        """
        n_samples = X_process_array.shape[0]
        
        # 如果未提供目标晶向，使用全0（表示无特定目标）
        if target_onehot is None:
            target_onehot = np.zeros(len(self.target_cols))
        
        # 将静态的预处理特征、目标晶向One-Hot与遍历的工艺参数进行拼接
        # 顺序: Pre_ + Target_ + Process_
        X_concat = np.hstack([
            np.tile(x_pre_array, (n_samples, 1)),
            np.tile(target_onehot, (n_samples, 1)),
            X_process_array
        ])
        X_scaled = self.scaler_X.transform(X_concat)
        
        # 预测均值与标准差（不确定度）
        mu, sigma = self.gpr.predict(X_scaled, return_std=True)
        
        # 计算 EI 积分
        with np.errstate(divide='warn'):
            imp = mu - y_best
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei, mu, sigma

    def recommend_next_process(self, new_pre_features, target_orientations=None, n_random_starts=100000):
        """
        给定新样品的初始特征和目标晶向列表，推荐全局最优退火工艺（支持多任务）
        
        Args:
            new_pre_features: EBSD预处理特征（字典或数组）
            target_orientations: 目标晶向列表，如 [(1,0,3), (1,0,2), (3,0,1)]。None表示使用当前训练数据的目标
            n_random_starts: 随机采样点数
        """
        if isinstance(new_pre_features, dict):
            x_pre = np.array([new_pre_features[col] for col in self.pre_feature_cols])
        else:
            x_pre = np.array(new_pre_features)
        
        # 构建目标晶向Multi-Hot编码（与训练时一致）
        if target_orientations is not None and self.target_cols:
            # 激活所有指定的目标晶向
            target_keys = [f"Target_{h}{k}{l}" for h, k, l in target_orientations]
            target_onehot = np.array([1.0 if col in target_keys else 0.0 for col in self.target_cols])
        else:
            # 如果没有指定目标，使用全0
            target_onehot = np.zeros(len(self.target_cols))
            target_keys = []
        
        # 根据目标晶向筛选对应的历史数据，找到该目标组合下的最优产率
        if target_keys and hasattr(self, 'training_df'):
            # 筛选具有相同目标晶向组合的训练样本（所有指定的目标都激活）
            mask = np.ones(len(self.training_df), dtype=bool)
            for key in target_keys:
                if key in self.training_df.columns:
                    mask = mask & (self.training_df[key] == 1.0)
            
            if mask.any():
                y_best = self.training_df.loc[mask, 'TARGET_Yield'].max()
                n_matching = mask.sum()
            else:
                # 如果没有找到完全匹配的目标组合，使用全局最优
                y_best = self.gpr.y_train_.max()
                n_matching = 0
        else:
            # 如果没有指定目标或没有训练数据，使用全局最优
            y_best = self.gpr.y_train_.max()
            n_matching = len(self.gpr.y_train_)
        
        # 在多维工艺参数空间内进行蒙特卡洛随机采样
        sampled_process = np.zeros((n_random_starts, len(self.process_cols)))
        for i, col in enumerate(self.process_cols):
            low, high = self.bounds[col]
            sampled_process[:, i] = np.random.uniform(low, high, n_random_starts)
            
        # 评估所有采样点的 EI 值
        ei_values, mu_values, sigma_values = self.expected_improvement(
            sampled_process, x_pre, y_best, target_onehot
        )
        
        # 锁定最大期望提升点
        best_idx = np.argmax(ei_values)
        best_process = sampled_process[best_idx]
        expected_yield = mu_values[best_idx]
        uncertainty = sigma_values[best_idx]
        
        recommendation = {col: best_process[i] for i, col in enumerate(self.process_cols)}
        
        # 显示目标晶向信息
        if target_orientations:
            target_str = ", ".join([f"<{h}{k}{l}>" for h, k, l in target_orientations])
        else:
            target_str = "当前训练目标"
        print("\n==================================================")
        print(f"目标晶向: {target_str}")
        print(f"历史最优产率 (同目标): {y_best:.2%} (基于 {n_matching} 个样本)")
        print("基于当前预处理微观状态的下一轮最优工艺预测：")
        for k, v in recommendation.items():
            print(f"  > {k}: {v:.2f}")
        print("--------------------------------------------------")
        print(f"  > 预测目标产率 (Mean): {expected_yield:.2%}")
        print(f"  > 模型不确定度 (Std):  {uncertainty:.4f}")
        print("==================================================\n")
        
        return recommendation


def add_new_data_to_training(existing_file, new_data_file):
    """
    将新实验数据追加到现有训练数据集中
    
    参数:
        existing_file: 现有训练数据文件路径
        new_data_file: 新数据文件路径（格式需与现有数据一致）
    """
    import os
    
    if not os.path.exists(new_data_file):
        print(f"错误: 找不到新数据文件 {new_data_file}")
        return False
    
    # 读取现有数据
    df_existing = pd.read_csv(existing_file)
    print(f"现有数据: {len(df_existing)} 条样本")
    
    # 读取新数据
    df_new = pd.read_csv(new_data_file)
    print(f"新数据: {len(df_new)} 条样本")
    
    # 检查列是否一致
    if set(df_existing.columns) != set(df_new.columns):
        print("警告: 列名不完全匹配，尝试对齐...")
        # 只保留共有的列
        common_cols = list(set(df_existing.columns) & set(df_new.columns))
        df_existing = df_existing[common_cols]
        df_new = df_new[common_cols]
    
    # 合并数据
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # 保存合并后的数据
    backup_file = existing_file.replace('.csv', '_备份.csv')
    df_existing.to_csv(backup_file, index=False)
    print(f"已创建备份: {backup_file}")
    
    df_combined.to_csv(existing_file, index=False)
    print(f"数据已更新: {existing_file}")
    print(f"总计: {len(df_combined)} 条样本 (新增 {len(df_new)} 条)")
    
    return True


def select_scheme():
    """
    交互式选择目标晶向方案
    """
    print("=" * 50)
    print("请选择目标晶向方案:")
    print("=" * 50)
    print("  [1] 方案 1: <103> 型织构 (橙色系)")
    print("      目标晶向: (1,0,3), (1,0,2), (3,0,1)")
    print()
    print("  [2] 方案 2: <114> 型织构 (粉紫色系)")
    print("      目标晶向: (1,1,4), (1,1,5), (1,0,5)")
    print()
    print("  [3] 方案 3: <124> 型织构 (混合色)")
    print("      目标晶向: (1,2,4), (1,2,5), (2,1,4)")
    print()
    print("  [4] 方案 4: 自定义组合")
    print("      目标晶向: (1,0,3), (1,1,4), (1,2,4)")
    print("=" * 50)
    
    while True:
        try:
            choice = input("请输入方案编号 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("无效输入，请输入 1-4 之间的数字")
        except KeyboardInterrupt:
            print("\n用户取消，使用默认方案 1")
            return 1
        except Exception as e:
            print(f"输入错误: {e}")


def data_management_menu(data_file):
    """
    数据管理菜单：选择使用现有数据或添加新数据
    """
    print("\n" + "=" * 50)
    print("数据管理选项:")
    print("=" * 50)
    print("  [1] 使用现有训练数据")
    print("  [2] 添加新实验数据到训练集")
    print("=" * 50)
    
    while True:
        choice = input("请选择操作 (1-2): ").strip()
        if choice == '1':
            return data_file
        elif choice == '2':
            new_file = input("请输入新数据文件路径: ").strip()
            # 移除可能的引号
            new_file = new_file.strip('"\'')
            
            if add_new_data_to_training(data_file, new_file):
                return data_file
            else:
                print("添加失败，使用现有数据继续...")
                return data_file
        else:
            print("无效输入，请输入 1 或 2")


if __name__ == "__main__":
    # ==========================
    # 方案选择
    # ==========================
    scheme = select_scheme()
    
    # 根据方案选择对应的数据文件
    scheme_names = {
        1: "<103> 型织构 (橙色系)",
        2: "<114> 型织构 (粉紫色系)",
        3: "<124> 型织构 (混合色)",
        4: "自定义组合"
    }
    
    # 数据文件命名规则: Optimized_Training_Data_方案X.csv
    data_file = f"Optimized_Training_Data_方案{scheme}.csv"
    
    # 如果方案文件不存在，尝试使用默认文件
    import os
    if not os.path.exists(data_file):
        print(f"\n注意: 未找到 {data_file}，尝试使用默认文件 Optimized_Training_Data.csv")
        data_file = "Optimized_Training_Data.csv"
    
    print(f"\n当前使用方案 {scheme}: {scheme_names[scheme]}")
    print(f"数据文件: {data_file}\n")
    
    # ==========================
    # 数据管理：添加新数据或直接使用
    # ==========================
    data_file = data_management_menu(data_file)
    
    # ==========================
    # 物理参数搜索边界设置
    # 请根据您的管式炉及实验安全规范严格修改此范围
    # ==========================
    process_bounds = {
        'Process_Temp': (1000.0, 1500.0),  # 退火温度下限与上限 (℃)
        'Process_Time': (1.0, 30.0),       # 保温时间下限与上限 (h)
        'Process_H2': (0.0, 160.0),        # H2 流量下限与上限 (sccm)
        'Process_Ar': (0.0, 800.0)         # Ar 流量下限与上限 (sccm)
    }
    
    # 实例化并训练模型
    optimizer = ContextualBayesianOptimizer(bounds=process_bounds)
    optimizer.train(data_file)
    
    print("\n模型训练完成！")
    print("\n提示: 使用 predict_new_sample.py 进行新样品的工艺预测")
    print("      运行: python predict_new_sample.py")