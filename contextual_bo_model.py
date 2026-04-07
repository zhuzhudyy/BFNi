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
        
        # 定义核函数：常数核 * Matern核(启用ARD) + 白噪声核
        # Matern(nu=2.5) 适用于具有二阶可导性的物理过程拟合
        # ARD (Automatic Relevance Determination): 为每个特征分配独立的长度尺度
        # 这样模型可以自动学习哪些特征更重要（长度尺度小 = 更敏感）
        # 注意：ARD长度尺度将在train()中根据实际特征维度初始化
        self.ard_length_scale = 1.0  # 临时值，将在train时更新
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=self.ard_length_scale, nu=2.5) + WhiteKernel(noise_level=1e-4)
        
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            normalize_y=True,
            random_state=42
        )
        
        self.scaler_X = StandardScaler()
        
    def _pre_feature_screening(self, df, pre_feature_cols, corr_threshold=0.9):
        """
        特征初筛：计算预处理特征间的皮尔逊相关系数矩阵
        对于相关性极高（r > corr_threshold）的特征组，仅保留一个最具物理意义的代表
        """
        if len(pre_feature_cols) <= 5:
            # 特征数已经很少，不需要筛选
            return pre_feature_cols
        
        pre_df = df[pre_feature_cols]
        corr_matrix = pre_df.corr(method='pearson')
        
        # 找出高相关性特征对
        features_to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > corr_threshold:
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]
                    # 保留名称较短的特征
                    if len(feat_i) <= len(feat_j):
                        features_to_remove.add(feat_j)
                    else:
                        features_to_remove.add(feat_i)
        
        selected_features = [f for f in pre_feature_cols if f not in features_to_remove]
        
        print(f"\n【特征初筛】从 {len(pre_feature_cols)} 个预处理特征中筛选出 {len(selected_features)} 个")
        if features_to_remove:
            print(f"  移除高相关性特征: {list(features_to_remove)[:5]}...")
        
        return selected_features
    
    def train(self, csv_path, enable_feature_screening=True):
        """
        读取历史实验数据并拟合代理模型（支持多任务）
        
        Args:
            csv_path: 训练数据文件路径
            enable_feature_screening: 是否启用特征初筛（小样本时建议启用）
        """
        self.training_df = pd.read_csv(csv_path)
        
        # 自动识别各类特征列
        self.pre_feature_cols = [col for col in self.training_df.columns if col.startswith('Pre_')]
        self.target_cols = [col for col in self.training_df.columns if col.startswith('Target_')]
        
        # 【修复】小样本时进行特征初筛，防止维度灾难
        n_samples = len(self.training_df)
        n_pre_features = len(self.pre_feature_cols)
        
        # 强制启用特征初筛：样本数不足特征数5倍时降维
        if enable_feature_screening and n_samples < n_pre_features * 5:
            print(f"\n[!] 警告: 样本数({n_samples})不足预处理特征数({n_pre_features})的5倍")
            print("    启用特征初筛以防止模型过拟合...")
            # 提高相关性阈值，更激进地降维
            self.pre_feature_cols = self._pre_feature_screening(
                self.training_df, self.pre_feature_cols, corr_threshold=0.8
            )
        
        # 构建特征矩阵 X 与目标向量 y
        # 顺序: EBSD预处理特征 + 目标晶向One-Hot + 工艺参数
        all_feature_cols = self.pre_feature_cols + self.target_cols + self.process_cols
        X_df = self.training_df[all_feature_cols]
        y = self.training_df['TARGET_Yield'].values
        
        n_features = len(all_feature_cols)
        
        # 启用 ARD: 为每个特征分配独立的长度尺度
        # 重新构建带ARD的核函数
        ard_length_scales = [1.0] * n_features  # 每个特征一个长度尺度参数
        
        # 【修复】提高长度尺度下界，防止"长度尺度坍缩"
        # 原设置 (1e-5, 1e5) 会导致小样本时模型退化为全局常数
        # 设置 (1.0, 100) 强制模型保持更强的空间平滑性和泛化能力
        # 下界1.0确保模型必须认为至少1个单位范围内的点是相关的
        ard_bounds = [(1.0, 100.0)] * n_features  # 每个特征独立的搜索范围
        
        kernel_ard = ConstantKernel(1.0, (1e-3, 1e3)) * \
                     Matern(length_scale=ard_length_scales, length_scale_bounds=ard_bounds, nu=2.5) + \
                     WhiteKernel(noise_level=1e-4)
        
        # 更新GPR的核函数
        self.gpr.kernel = kernel_ard
        
        # 对输入空间进行标准化处理，加速核函数收敛
        X_scaled = self.scaler_X.fit_transform(X_df)
        
        # 模型拟合
        self.gpr.fit(X_scaled, y)
        
        # 打印ARD分析结果
        self._print_ard_analysis()
        print(f"代理模型训练完毕。吸收样本量: {len(self.training_df)}，总特征维度: {len(all_feature_cols)}。")
        print(f"  - EBSD预处理特征: {len(self.pre_feature_cols)} 维")
        print(f"  - 目标晶向One-Hot: {len(self.target_cols)} 维")
        print(f"  - 工艺参数: {len(self.process_cols)} 维")
        print(f"优化后核函数参数: {self.gpr.kernel_}")

    def _print_ard_analysis(self):
        """
        打印ARD (Automatic Relevance Determination) 分析结果
        显示每个特征的长度尺度，判断特征重要性
        """
        try:
            # 从优化后的核函数中提取Matern核的长度尺度
            kernel = self.gpr.kernel_
            # 核函数结构: ConstantKernel * Matern + WhiteKernel
            # 所以 kernel.k1 是 ConstantKernel * Matern, kernel.k2 是 WhiteKernel
            matern_kernel = kernel.k1.k2  # 获取Matern核
            length_scales = matern_kernel.length_scale
            
            all_feature_cols = self.pre_feature_cols + self.target_cols + self.process_cols
            
            print("\n" + "="*60)
            print("           ARD 特征重要性分析 (长度尺度越小 = 越重要)")
            print("="*60)
            
            # 按特征类别分组显示
            idx = 0
            
            # EBSD预处理特征
            if self.pre_feature_cols:
                print("\n【EBSD预处理特征】")
                for col in self.pre_feature_cols:
                    if idx < len(length_scales):
                        ls = length_scales[idx]
                        importance = "★★★" if ls < 0.5 else ("★★" if ls < 1.0 else "★")
                        print(f"  {col:25s}: length_scale = {ls:8.4f} {importance}")
                        idx += 1
            
            # 目标晶向One-Hot
            if self.target_cols:
                print("\n【目标晶向One-Hot】")
                for col in self.target_cols:
                    if idx < len(length_scales):
                        ls = length_scales[idx]
                        importance = "★★★" if ls < 0.5 else ("★★" if ls < 1.0 else "★")
                        print(f"  {col:25s}: length_scale = {ls:8.4f} {importance}")
                        idx += 1
            
            # 工艺参数
            if self.process_cols:
                print("\n【工艺参数】")
                for col in self.process_cols:
                    if idx < len(length_scales):
                        ls = length_scales[idx]
                        importance = "★★★" if ls < 0.5 else ("★★" if ls < 1.0 else "★")
                        print(f"  {col:25s}: length_scale = {ls:8.4f} {importance}")
                        idx += 1
            
            # 找出最重要和最不重要的特征
            if len(length_scales) == len(all_feature_cols):
                sorted_indices = np.argsort(length_scales)
                print(f"\n【ARD分析结论】")
                print(f"  最敏感特征 (最重要): {all_feature_cols[sorted_indices[0]]} (ls={length_scales[sorted_indices[0]]:.4f})")
                print(f"  最不敏感特征: {all_feature_cols[sorted_indices[-1]]} (ls={length_scales[sorted_indices[-1]]:.4f})")
            
            print("="*60)
            
        except Exception as e:
            # 如果解析失败，不中断训练流程
            print(f"\n[ARD分析] 无法解析核函数参数: {e}")

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
        
        # ==================== 两步 EI 优化策略 ====================
        # 步骤1: 大规模随机采样，寻找有潜力的区域
        sampled_process = np.zeros((n_random_starts, len(self.process_cols)))
        for i, col in enumerate(self.process_cols):
            low, high = self.bounds[col]
            sampled_process[:, i] = np.random.uniform(low, high, n_random_starts)
            
        # 评估所有采样点的 EI 值
        ei_values, mu_values, sigma_values = self.expected_improvement(
            sampled_process, x_pre, y_best, target_onehot
        )
        
        # 步骤2: 取Top 5作为初始点，进行局部梯度优化
        from scipy.optimize import minimize
        
        # 获取Top 5索引
        top5_indices = np.argsort(ei_values)[-5:][::-1]
        top5_process = sampled_process[top5_indices]
        top5_ei = ei_values[top5_indices]
        
        print(f"\n[*] 随机采样完成，Top 5 EI 值: {top5_ei}")
        print("[*] 对Top 5点进行L-BFGS-B局部优化...")
        
        # 定义边界（用于L-BFGS-B）
        bounds_list = [(self.bounds[col][0], self.bounds[col][1]) for col in self.process_cols]
        
        # 定义负EI函数（用于最小化）
        def neg_ei(x_process):
            x_process = x_process.reshape(1, -1)
            ei, _, _ = self.expected_improvement(x_process, x_pre, y_best, target_onehot)
            return -ei[0]  # 返回负EI用于最小化
        
        # 对每个Top点进行局部优化
        optimized_results = []
        for i, init_point in enumerate(top5_process):
            try:
                result = minimize(
                    neg_ei,
                    init_point,
                    method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                optimized_ei = -result.fun
                optimized_results.append({
                    'ei': optimized_ei,
                    'x': result.x,
                    'init_ei': top5_ei[i],
                    'improvement': optimized_ei - top5_ei[i],
                    'success': result.success
                })
                print(f"    Top {i+1}: EI {top5_ei[i]:.6f} → {optimized_ei:.6f} "
                      f"(提升: {optimized_ei - top5_ei[i]:.6f})")
            except Exception as e:
                print(f"    Top {i+1}: 优化失败 ({e})")
                optimized_results.append({
                    'ei': top5_ei[i],
                    'x': init_point,
                    'init_ei': top5_ei[i],
                    'improvement': 0,
                    'success': False
                })
        
        # 选择优化后EI最大的点
        best_result = max(optimized_results, key=lambda r: r['ei'])
        best_process = best_result['x']
        
        # 获取最终预测值
        _, final_mu, final_sigma = self.expected_improvement(
            best_process.reshape(1, -1), x_pre, y_best, target_onehot
        )
        expected_yield = final_mu[0]
        uncertainty = final_sigma[0]
        
        print(f"\n[*] 最优结果来自: Top {optimized_results.index(best_result)+1} "
              f"(优化{'成功' if best_result['success'] else '失败'})")
        print(f"[*] EI 改进: {best_result['init_ei']:.6f} → {best_result['ei']:.6f} "
              f"(+{(best_result['ei']/best_result['init_ei']-1)*100:.2f}%)")
        
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

    def save_model(self, model_path):
        """
        保存训练好的模型到文件
        
        Args:
            model_path: 模型保存路径
        """
        import pickle
        import os
        
        # 创建模型字典，保存所有必要的状态
        model_state = {
            'gpr': self.gpr,
            'scaler_X': self.scaler_X,
            'bounds': self.bounds,
            'process_cols': self.process_cols,
            'pre_feature_cols': self.pre_feature_cols,
            'target_cols': self.target_cols,
            'target_orientations': self.target_orientations,
            'training_df': self.training_df if hasattr(self, 'training_df') else None
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"\n[*] 模型已保存至: {model_path}")
        print(f"    包含 {len(self.training_df) if hasattr(self, 'training_df') and self.training_df is not None else 0} 个训练样本")
    
    def load_model(self, model_path):
        """
        从文件加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        Returns:
            bool: 加载是否成功
        """
        import pickle
        import os
        
        if not os.path.exists(model_path):
            print(f"[错误] 模型文件不存在: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_state = pickle.load(f)
            
            # 恢复模型状态
            self.gpr = model_state['gpr']
            self.scaler_X = model_state['scaler_X']
            self.bounds = model_state['bounds']
            self.process_cols = model_state['process_cols']
            self.pre_feature_cols = model_state['pre_feature_cols']
            self.target_cols = model_state['target_cols']
            self.target_orientations = model_state['target_orientations']
            self.training_df = model_state.get('training_df', None)
            
            print(f"\n[*] 模型已从 {model_path} 加载")
            if self.training_df is not None:
                print(f"    包含 {len(self.training_df)} 个训练样本")
            return True
            
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            return False


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
        'Process_Temp': (1000.0, 1400.0),  # 退火温度下限与上限 (℃)
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