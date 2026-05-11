import pandas as pd
import numpy as np
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, gamma as gamma_dist
from scipy.optimize import minimize
import warnings

# 忽略高斯过程在极低方差区域可能产生的数值计算警告
warnings.filterwarnings("ignore")

# ============================================================
# ANOVA Matern 核函数（分组 ARD + 交互效应 + 混合 nu）
# ============================================================

class ANOVAMaternKernel(Kernel):
    """
    ANOVA 分解的 Matern 核函数（单类实现，参数平坦化）

    K = σ²_pre × Matern52(Pre_) + σ²_target × Matern52(Target_)
      + σ²_proc × Matern32(Process_) + σ²_inter × Matern52(Pre_) × Matern32(Process_)
      + σ²_noise

    theta 结构（log 尺度）：
    [pre_ls(n_pre), target_ls(n_target), proc_ls(n_proc),
     log(σ²_pre), log(σ²_target), log(σ²_proc), log(σ²_inter), log(σ²_noise)]

    长度尺度共享：交互项复用主效应的长度尺度，无额外参数。
    """

    def __init__(self, n_pre, n_target, n_proc,
                 pre_ls_bounds=(0.3, 5.0), target_ls_bounds=(0.5, 3.0), proc_ls_bounds=(0.3, 3.0)):
        self.n_pre = n_pre
        self.n_target = n_target
        self.n_proc = n_proc
        self.pre_ls_bounds = pre_ls_bounds
        self.target_ls_bounds = target_ls_bounds
        self.proc_ls_bounds = proc_ls_bounds

        # 超参数定义（通过 Hyperparameter 对象让 sklearn 识别）
        self._hyperparameters = []
        if n_pre > 0:
            self._hyperparameters.append(
                Hyperparameter('pre_length_scale', 'numeric', (pre_ls_bounds,) * n_pre, n_pre))
        if n_target > 0:
            self._hyperparameters.append(
                Hyperparameter('target_length_scale', 'numeric', (target_ls_bounds,) * n_target, n_target))
        if n_proc > 0:
            self._hyperparameters.append(
                Hyperparameter('proc_length_scale', 'numeric', (proc_ls_bounds,) * n_proc, n_proc))
        self._hyperparameters.append(
            Hyperparameter('log_var_pre', 'numeric', (-6.0, 6.0)))
        self._hyperparameters.append(
            Hyperparameter('log_var_target', 'numeric', (-6.0, 6.0)))
        self._hyperparameters.append(
            Hyperparameter('log_var_proc', 'numeric', (-6.0, 6.0)))
        self._hyperparameters.append(
            Hyperparameter('log_var_inter', 'numeric', (-6.0, 6.0)))
        self._hyperparameters.append(
            Hyperparameter('log_noise_level', 'numeric', (-10.0, 10.0)))

    @property
    def hyperparameter_length_scale(self):
        """返回第一个 length_scale 超参数（sklearn 兼容）"""
        return self._hyperparameters[0]

    @property
    def theta(self):
        """返回 log 尺度的超参数向量"""
        vals = []
        for hp in self._hyperparameters:
            v = self._get_param(hp.name)
            if 'length_scale' in hp.name:
                vals.append(np.log(np.atleast_1d(v)))
            else:
                vals.append(np.atleast_1d(v))  # 已经是 log 尺度
        return np.concatenate(vals)

    @theta.setter
    def theta(self, value):
        offset = 0
        for hp in self._hyperparameters:
            n = hp.n_elements
            vals = value[offset:offset + n]
            if 'length_scale' in hp.name:
                setattr(self, hp.name, np.exp(vals))
            elif n == 1:
                setattr(self, hp.name, float(vals[0]))
            else:
                setattr(self, hp.name, vals)
            offset += n

    @property
    def bounds(self):
        result = []
        for hp in self._hyperparameters:
            b = np.atleast_2d(hp.bounds)  # shape: (n_elements, 2)
            for row in b:
                lo, hi = row
                if 'length_scale' in hp.name:
                    result.append([np.log(lo), np.log(hi)])
                else:
                    result.append([lo, hi])
        return np.array(result)

    def _get_param(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        # 默认值
        if 'length_scale' in name:
            if 'pre' in name:
                return np.ones(self.n_pre)
            elif 'target' in name:
                return np.ones(self.n_target)
            elif 'proc' in name:
                return np.ones(self.n_proc)
        return 0.0  # log(1.0) = 0

    def get_params(self, deep=True):
        # 返回构造函数参数（sklearn clone 需要）
        return {'n_pre': self.n_pre, 'n_target': self.n_target, 'n_proc': self.n_proc,
                'pre_ls_bounds': self.pre_ls_bounds, 'target_ls_bounds': self.target_ls_bounds,
                'proc_ls_bounds': self.proc_ls_bounds}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _matern_kernel(self, dists, nu):
        """计算 Matern 核值"""
        if nu == 0.5:
            return np.exp(-dists)
        elif nu == 1.5:
            return (1.0 + np.sqrt(3.0) * dists) * np.exp(-np.sqrt(3.0) * dists)
        elif nu == 2.5:
            return (1.0 + np.sqrt(5.0) * dists + 5.0 / 3.0 * dists ** 2) * np.exp(-np.sqrt(5.0) * dists)
        return np.exp(-dists)

    def _scaled_dists(self, X, Y, ls, dim_slice):
        """计算归一化欧氏距离"""
        Xs = X[:, dim_slice] / ls
        Ys = Y[:, dim_slice] / ls if Y is not None else Xs
        sq_sum_X = np.sum(Xs ** 2, axis=1, keepdims=True)
        sq_sum_Y = np.sum(Ys ** 2, axis=1, keepdims=True) if Y is not None else sq_sum_X
        dists = np.sqrt(np.maximum(sq_sum_X + sq_sum_Y.T - 2 * Xs @ Ys.T, 0))
        return dists

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X)
        if Y is None:
            Y = X

        pre_ls = np.atleast_1d(self._get_param('pre_length_scale'))
        target_ls = np.atleast_1d(self._get_param('target_length_scale'))
        proc_ls = np.atleast_1d(self._get_param('proc_length_scale'))
        var_pre = np.exp(self.log_var_pre) if hasattr(self, 'log_var_pre') else 1.0
        var_target = np.exp(self.log_var_target) if hasattr(self, 'log_var_target') else 1.0
        var_proc = np.exp(self.log_var_proc) if hasattr(self, 'log_var_proc') else 1.0
        var_inter = np.exp(self.log_var_inter) if hasattr(self, 'log_var_inter') else 1.0
        noise = np.exp(self.log_noise_level) if hasattr(self, 'log_noise_level') else 1e-4

        pre_dims = list(range(self.n_pre))
        target_dims = list(range(self.n_pre, self.n_pre + self.n_target))
        proc_dims = list(range(self.n_pre + self.n_target, self.n_pre + self.n_target + self.n_proc))

        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))

        # 主效应
        if self.n_pre > 0:
            d_pre = self._scaled_dists(X, Y, pre_ls, pre_dims)
            K += var_pre * self._matern_kernel(d_pre, 2.5)
        if self.n_target > 0:
            d_target = self._scaled_dists(X, Y, target_ls, target_dims)
            K += var_target * self._matern_kernel(d_target, 2.5)
        if self.n_proc > 0:
            d_proc = self._scaled_dists(X, Y, proc_ls, proc_dims)
            K += var_proc * self._matern_kernel(d_proc, 1.5)

        # 交互效应（复用主效应长度尺度）
        if self.n_pre > 0 and self.n_proc > 0:
            K_pre_inter = self._matern_kernel(d_pre, 2.5)
            K_proc_inter = self._matern_kernel(d_proc, 1.5)
            K += var_inter * K_pre_inter * K_proc_inter

        # 噪声
        if Y is X:
            np.fill_diagonal(K, K.diagonal() + noise)

        if eval_gradient:
            return K, np.empty((n, m, 0))  # 梯度暂不支持
        return K

    def diag(self, X):
        var_pre = np.exp(self.log_var_pre) if hasattr(self, 'log_var_pre') else 1.0
        var_target = np.exp(self.log_var_target) if hasattr(self, 'log_var_target') else 1.0
        var_proc = np.exp(self.log_var_proc) if hasattr(self, 'log_var_proc') else 1.0
        var_inter = np.exp(self.log_var_inter) if hasattr(self, 'log_var_inter') else 1.0
        noise = np.exp(self.log_noise_level) if hasattr(self, 'log_noise_level') else 1e-4
        return np.full(X.shape[0], var_pre + var_target + var_proc + var_inter + noise)

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"ANOVAMaternKernel(pre={self.n_pre}, target={self.n_target}, proc={self.n_proc})"


class GPRWithPriors(GaussianProcessRegressor):
    """
    扩展 sklearn GPR，支持：
    1. 对交互项方差施加 Gamma 先验
    2. 多起点优化（num_restarts 次随机重启）
    """

    def __init__(self, kernel, alpha=1e-10, normalize_y=True,
                 n_restarts_optimizer=0, random_state=None,
                 inter_var_prior=None, num_restarts=20):
        super().__init__(
            kernel=kernel, alpha=alpha, normalize_y=normalize_y,
            n_restarts_optimizer=0,
            random_state=random_state
        )
        self.inter_var_prior = inter_var_prior  # (a, b) for Gamma(a,b)
        self.num_restarts = num_restarts

    def fit(self, X, y):
        """多起点优化 + 先验注入"""
        from sklearn.utils import check_random_state
        rng = check_random_state(self.random_state)

        best_lml = -np.inf
        best_theta = None
        best_kernel = None

        for restart in range(self.num_restarts):
            try:
                if restart > 0:
                    # 随机初始化核参数
                    theta = self.kernel.theta.copy()
                    bounds = self.kernel.bounds
                    for i, (lo, hi) in enumerate(bounds):
                        theta[i] = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
                    self.kernel.theta = theta

                # 父类 fit（单次优化）
                super().fit(X, y)

                # 计算带先验的 LML
                lml = self._lml_with_prior()

                if lml > best_lml:
                    best_lml = lml
                    best_theta = self.kernel_.theta.copy()
                    best_kernel = self.kernel_

            except Exception:
                continue

        if best_theta is not None:
            self.kernel_ = best_kernel
            # 重新计算 cholesky
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += self.alpha
            self.L_ = np.linalg.cholesky(K)
            self.log_marginal_likelihood_value_ = best_lml

        return self

    def _lml_with_prior(self):
        """计算带先验的 log marginal likelihood"""
        lml = self.log_marginal_likelihood(self.kernel_.theta, eval_gradient=False)

        if self.inter_var_prior is not None and hasattr(self.kernel_, 'log_var_inter'):
            a, b = self.inter_var_prior
            inter_var = np.exp(self.kernel_.log_var_inter)
            lml += gamma_dist.logpdf(inter_var, a=a, scale=1.0 / b)

        return lml

ALL_TARGET_ORIENTATIONS = [
    (1, 0, 3), (1, 0, 2), (3, 0, 1),
    (1, 1, 4), (1, 1, 5), (1, 0, 5),
    (1, 2, 4), (1, 2, 5), (2, 1, 4),
]

SCHEME_NAMES = {
    1: "<103> 型织构 (橙色系)",
    2: "<114> 型织构 (粉紫色系)",
    3: "<124> 型织构 (混合色)",
    4: "自定义组合",
}

SCHEME_TARGETS = {
    1: [(1, 0, 3), (1, 0, 2), (3, 0, 1)],
    2: [(1, 1, 4), (1, 1, 5), (1, 0, 5)],
    3: [(1, 2, 4), (1, 2, 5), (2, 1, 4)],
    4: [(1, 0, 3), (1, 1, 4), (1, 2, 4)],
}

DEFAULT_PROCESS_BOUNDS = {
    'Process_Temp': (1000.0, 1400.0),
    'Process_Time': (1.0, 30.0),
    'Process_H2': (50.0, 160.0),
    'Process_Ar': (0.0, 800.0),
}

PROCESS_LABELS_CN = {
    'Process_Temp': '退火温度 (°C)',
    'Process_Time': '保温时间 (h)',
    'Process_H2': 'H$_2$ 流量 (sccm)',
    'Process_Ar': 'Ar 流量 (sccm)',
}

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

        # 核函数将在 train() 中根据实际特征维度初始化（ANOVA Matern 架构）
        self.gpr = None
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
        读取历史实验数据并拟合代理模型（ANOVA Matern 核架构）

        Args:
            csv_path: 训练数据文件路径
            enable_feature_screening: 是否启用特征初筛（小样本时建议启用）
        """
        self.training_df = pd.read_csv(csv_path)

        # 自动识别各类特征列
        self.pre_feature_cols = [col for col in self.training_df.columns if col.startswith('Pre_')]
        self.target_cols = [col for col in self.training_df.columns if col.startswith('Target_')]

        # 小样本时进行特征初筛
        n_samples = len(self.training_df)
        n_pre_features = len(self.pre_feature_cols)

        if enable_feature_screening and n_samples < n_pre_features * 5:
            print(f"\n[!] 警告: 样本数({n_samples})不足预处理特征数({n_pre_features})的5倍")
            print("    启用特征初筛以防止模型过拟合...")
            self.pre_feature_cols = self._pre_feature_screening(
                self.training_df, self.pre_feature_cols, corr_threshold=0.8
            )

        # 构建特征矩阵 X 与目标向量 y
        all_feature_cols = self.pre_feature_cols + self.target_cols + self.process_cols
        X_df = self.training_df[all_feature_cols]
        y = self.training_df['TARGET_Yield'].values

        n_pre = len(self.pre_feature_cols)
        n_target = len(self.target_cols)
        n_proc = len(self.process_cols)

        # 构建 ANOVA Matern 核函数
        # K = k_pre + k_target + k_proc + k_inter(pre × proc) + noise
        # - Pre_ 和 Target_ 使用 Matern 5/2（平滑响应）
        # - Process_ 使用 Matern 3/2（允许阈值效应）
        # - 交互项的长度尺度与主效应共享
        kernel = ANOVAMaternKernel(n_pre, n_target, n_proc)

        # 构建带先验的 GPR 模型
        # 交互项方差施加弱 Gamma 先验 Gamma(1.5, 1.0)，防止过拟合
        self.gpr = GPRWithPriors(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            inter_var_prior=(1.5, 1.0),
            num_restarts=20,
            random_state=42
        )

        # 标准化输入
        X_scaled = self.scaler_X.fit_transform(X_df)

        # 模型拟合（多起点优化 + 先验注入）
        print("\n[*] 正在训练 ANOVA Matern GPR 模型（20 次多起点优化）...")
        self.gpr.fit(X_scaled, y)

        # 打印 ARD 分析
        self._print_ard_analysis()
        print(f"\n代理模型训练完毕。吸收样本量: {len(self.training_df)}，总特征维度: {len(all_feature_cols)}。")
        print(f"  - EBSD预处理特征: {n_pre} 维 (Matern 5/2)")
        print(f"  - 目标晶向One-Hot: {n_target} 维 (Matern 5/2)")
        print(f"  - 工艺参数: {n_proc} 维 (Matern 3/2)")
        print(f"  - 交互项: Pre_ × Process_ (共享长度尺度)")

    def _print_ard_analysis(self):
        """打印 ANOVA 核函数的 ARD 分析结果"""
        try:
            kernel = self.gpr.kernel_

            pre_ls = np.atleast_1d(kernel.pre_length_scale)
            target_ls = np.atleast_1d(kernel.target_length_scale)
            proc_ls = np.atleast_1d(kernel.proc_length_scale)
            var_pre = np.exp(kernel.log_var_pre)
            var_target = np.exp(kernel.log_var_target)
            var_proc = np.exp(kernel.log_var_proc)
            var_inter = np.exp(kernel.log_var_inter)

            print("\n" + "="*70)
            print("         ANOVA Matern ARD 分析 (长度尺度越小 = 越重要)")
            print("="*70)

            if self.pre_feature_cols:
                print(f"\n【EBSD预处理特征 (Matern 5/2)】 σ²_pre = {var_pre:.4f}")
                for i, col in enumerate(self.pre_feature_cols):
                    if i < len(pre_ls):
                        ls = pre_ls[i]
                        imp = "★★★" if ls < 0.5 else ("★★" if ls < 1.0 else "★")
                        print(f"  {col:30s}: ls = {ls:8.4f} {imp}")

            if self.target_cols:
                print(f"\n【目标晶向 One-Hot (Matern 5/2)】 σ²_target = {var_target:.4f}")
                for i, col in enumerate(self.target_cols):
                    if i < len(target_ls):
                        ls = target_ls[i]
                        imp = "★★★" if ls < 0.5 else ("★★" if ls < 1.0 else "★")
                        print(f"  {col:30s}: ls = {ls:8.4f} {imp}")

            if self.process_cols:
                print(f"\n【工艺参数 (Matern 3/2, 允许阈值效应)】 σ²_proc = {var_proc:.4f}")
                for i, col in enumerate(self.process_cols):
                    if i < len(proc_ls):
                        ls = proc_ls[i]
                        imp = "★★★" if ls < 0.5 else ("★★" if ls < 1.0 else "★")
                        print(f"  {col:30s}: ls = {ls:8.4f} {imp}")

            total_var = var_pre + var_target + var_proc + var_inter
            inter_ratio = var_inter / (total_var + 1e-10)
            print(f"\n【交互效应 (Pre_ × Process_)】 σ²_inter = {var_inter:.4f}")
            print(f"  交互项占总方差比例: {inter_ratio:.1%}")
            if inter_ratio > 0.1:
                print("  → 交互效应显著，初始状态与工艺参数存在耦合")
            else:
                print("  → 交互效应较弱，主效应主导")

            print("="*70)

        except Exception as e:
            print(f"\n[ARD分析] 无法解析核函数参数: {e}")

    def _compute_y_best(self, x_pre, target_keys, target_orientations, k_neighbors=3):
        """
        计算稳健的 y_best，解决三大问题:
        1. 尺度一致: 始终使用原始尺度 (不依赖 gpr.y_train_ 的归一化值)
        2. 上下文感知: 优先使用与当前 pre_features 相似的样本的产率参考值
        3. 异常值保护: winsorize 到 P95，防止单个极端值 (如 y=1.0) 绑架优化

        Returns:
            y_best: 原始尺度的最优产率参考值
            n_matching: 用于报告的匹配样本数
            source: 来源标签 (用于诊断输出)
        """
        if not hasattr(self, 'training_df') or self.training_df is None:
            return self._get_y_best_from_gpr_with_source()

        df = self.training_df

        # 按目标晶向筛选
        if target_keys:
            mask = np.ones(len(df), dtype=bool)
            for key in target_keys:
                if key in df.columns:
                    mask = mask & (df[key] == 1.0)
            df_target = df[mask].copy()
        else:
            df_target = df.copy()

        if len(df_target) == 0:
            return self._get_y_best_from_gpr_with_source()

        # 按与当前 pre_features 的欧氏距离找 k 近邻 (在 pre 特征空间中)
        pre_cols = self.pre_feature_cols
        if len(pre_cols) > 0 and len(df_target) >= k_neighbors:
            df_pre = df_target[pre_cols].values
            x_pre_arr = np.array(x_pre).reshape(1, -1)
            # 标准化以避免量纲影响
            pre_std = df_pre.std(axis=0, ddof=1)
            pre_std[pre_std < 1e-8] = 1.0
            pre_mean = df_pre.mean(axis=0)
            x_pre_norm = (x_pre_arr - pre_mean) / pre_std
            df_pre_norm = (df_pre - pre_mean) / pre_std
            distances = np.sqrt(((df_pre_norm - x_pre_norm) ** 2).sum(axis=1))
            neighbor_idx = np.argsort(distances)[:k_neighbors]
            neighbor_yields = df_target.iloc[neighbor_idx]['TARGET_Yield'].values
            source = f'最近{k_neighbors}邻(距离{np.min(distances):.3f})'
            n_matching = len(df_target)
        else:
            neighbor_yields = df_target['TARGET_Yield'].values
            source = '同目标全量'
            n_matching = len(df_target)

        # Winsorize: 用 P95 截断，防止异常高值绑架 EI
        y_max_raw = neighbor_yields.max()
        y_p95 = np.percentile(neighbor_yields, 95)
        y_best = min(y_max_raw, y_p95)

        # 如果 y_best 过低，允许适当放宽
        y_p50 = np.percentile(neighbor_yields, 50)
        if y_best < y_p50:
            y_best = y_p50

        # 最终上限：确保 y_best 不超过全局原始值的合理上限
        global_y_max = df['TARGET_Yield'].max()
        global_y_p99 = np.percentile(df['TARGET_Yield'], 99)
        y_best = min(y_best, global_y_p99)
        y_best = max(y_best, global_y_max * 0.5)  # 不低于全局最大的一半

        return float(y_best), n_matching, source

    def _get_y_best_from_gpr_with_source(self):
        """从 GPR 的 y_train_ 反归一化回原始尺度 (紧急回退)"""
        if hasattr(self, 'gpr') and hasattr(self.gpr, '_y_train_std'):
            y_raw = float(self.gpr.y_train_.max() * self.gpr._y_train_std + self.gpr._y_train_mean)
            return y_raw, 0, 'GPR反归一化'
        return 0.5, 0, '默认回退(0.5)'

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
        sigma_safe = np.maximum(sigma, 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - y_best
            Z = imp / sigma_safe
            ei = imp * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
        # 数值保护：浮点运算可能使 EI 微负（imp*Φ(Z) 负项略大于 σ*φ(Z) 正项）
        # 必须钳位到 0，否则 L-BFGS-B 会追逐最负区域而非真正最优
        ei = np.maximum(ei, 0.0)
        ei[sigma <= 1e-12] = 0.0

        return ei, mu, sigma

    def _predict_mu(self, X_process_array, x_pre_array, target_onehot=None):
        """纯预测产率均值（无采集函数），用于寻找最高 μ 点"""
        n_samples = X_process_array.shape[0]
        if target_onehot is None:
            target_onehot = np.zeros(len(self.target_cols))
        X_concat = np.hstack([
            np.tile(x_pre_array, (n_samples, 1)),
            np.tile(target_onehot, (n_samples, 1)),
            X_process_array
        ])
        X_scaled = self.scaler_X.transform(X_concat)
        mu, sigma = self.gpr.predict(X_scaled, return_std=True)
        return mu, sigma

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
        # 【修复】y_best 选择策略（解决尺度不一致 + 上下文感知 + 异常值保护）
        y_best, n_matching, y_best_source = self._compute_y_best(
            x_pre, target_keys, target_orientations
        )
        
        # ==================== 两步 EI 优化策略 ====================
        # 步骤1: 大规模随机采样，寻找有潜力的区域
        h2_idx = self.process_cols.index('Process_H2')
        ar_idx = self.process_cols.index('Process_Ar')
        sampled_process = np.zeros((n_random_starts, len(self.process_cols)))
        for i, col in enumerate(self.process_cols):
            low, high = self.bounds[col]
            sampled_process[:, i] = np.random.uniform(low, high, n_random_starts)
        # 约束: Ar >= 2 * H2，不满足的点重采样至满足
        for j in range(n_random_starts):
            while sampled_process[j, ar_idx] < 2 * sampled_process[j, h2_idx]:
                sampled_process[j, h2_idx] = np.random.uniform(*self.bounds['Process_H2'])
                sampled_process[j, ar_idx] = np.random.uniform(*self.bounds['Process_Ar'])

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
        
        # 定义负EI函数（用于最小化），含 Ar >= 2*H2 约束的平滑二次惩罚
        def neg_ei(x_process):
            x_process = x_process.reshape(1, -1)
            h2, ar = x_process[0, h2_idx], x_process[0, ar_idx]
            ei, _, _ = self.expected_improvement(x_process, x_pre, y_best, target_onehot)
            penalty = 0.0
            if ar < 2 * h2:
                penalty = 1e4 * (2 * h2 - ar) ** 2
            return -ei[0] + penalty
        
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
                # 用真实 EI（不含惩罚）评估
                actual_ei, _, _ = self.expected_improvement(
                    result.x.reshape(1, -1), x_pre, y_best, target_onehot
                )
                optimized_ei = actual_ei[0]
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
        
        # 选择优化后EI最大的可行点（Ar >= 2*H2）
        def is_feasible(x):
            return x[ar_idx] >= 2 * x[h2_idx]

        feasible_ei = [r for r in optimized_results if is_feasible(r['x'])]
        if feasible_ei:
            best_ei_result = max(feasible_ei, key=lambda r: r['ei'])
        else:
            # 所有L-BFGS-B结果均不可行，回退到随机采样中的最佳可行点
            feasible_mask = np.array([
                is_feasible(sampled_process[j]) for j in range(n_random_starts)
            ])
            feasible_ei_vals = ei_values.copy()
            feasible_ei_vals[~feasible_mask] = -np.inf
            fallback_idx = np.argmax(feasible_ei_vals)
            best_ei_result = {'ei': ei_values[fallback_idx], 'x': sampled_process[fallback_idx],
                              'init_ei': ei_values[fallback_idx], 'improvement': 0, 'success': False}
            print("[警告] L-BFGS-B 优化结果均违反 Ar>=2*H2 约束，回退到随机采样可行点")
        best_ei_process = best_ei_result['x']

        # 获取EI最优的预测值
        _, ei_mu, ei_sigma = self.expected_improvement(
            best_ei_process.reshape(1, -1), x_pre, y_best, target_onehot
        )

        # ==================== 最高 μ 点（纯利用，当前最优工艺）====================
        # 复用同一批随机采样的 mu_values，取 Top 5 精炼
        top5_mu_indices = np.argsort(mu_values)[-5:][::-1]
        top5_mu_process = sampled_process[top5_mu_indices]
        top5_mu_vals = mu_values[top5_mu_indices]

        def neg_mu(x_process):
            x_process = x_process.reshape(1, -1)
            mu, _ = self._predict_mu(x_process, x_pre, target_onehot)
            h2, ar = x_process[0, h2_idx], x_process[0, ar_idx]
            penalty = 0.0
            if ar < 2 * h2:
                penalty = 1e4 * (2 * h2 - ar) ** 2
            return -mu[0] + penalty

        mu_optimized = []
        for i, init_point in enumerate(top5_mu_process):
            try:
                result = minimize(neg_mu, init_point, method='L-BFGS-B',
                                  bounds=bounds_list, options={'maxiter': 100, 'ftol': 1e-9})
                # 用真实 mu（不含惩罚）评估
                actual_mu, _ = self._predict_mu(result.x.reshape(1, -1), x_pre, target_onehot)
                mu_optimized.append({'mu': actual_mu[0], 'x': result.x, 'success': result.success})
            except Exception:
                mu_optimized.append({'mu': top5_mu_vals[i], 'x': init_point, 'success': False})

        # 选择可行点中 mu 最大的
        feasible_mu = [r for r in mu_optimized if is_feasible(r['x'])]
        if feasible_mu:
            best_mu_result = max(feasible_mu, key=lambda r: r['mu'])
        else:
            best_mu_result = max(mu_optimized, key=lambda r: r['mu'])
            print("[警告] μ 优化结果均违反 Ar>=2*H2 约束")
        best_mu_process = best_mu_result['x']
        mu_final, mu_sigma_final = self._predict_mu(
            best_mu_process.reshape(1, -1), x_pre, target_onehot
        )

        # ==================== 输出 ====================
        ei_recommendation = {col: best_ei_process[i] for i, col in enumerate(self.process_cols)}
        mu_recommendation = {col: best_mu_process[i] for i, col in enumerate(self.process_cols)}

        if target_orientations:
            target_str = ", ".join([f"<{h}{k}{l}>" for h, k, l in target_orientations])
        else:
            target_str = "当前训练目标"

        print(f"\n[*] EI 最优结果来自: Top {optimized_results.index(best_ei_result)+1} "
              f"(优化{'成功' if best_ei_result['success'] else '失败'})")
        print(f"[*] EI 改进: {best_ei_result['init_ei']:.6f} → {best_ei_result['ei']:.6f} "
              f"(+{(best_ei_result['ei']/best_ei_result['init_ei']-1)*100:.2f}%)")

        print("\n" + "="*60)
        print(f"目标晶向: {target_str}")
        print(f"y_best (优化参考值): {y_best:.2%}")
        print(f"  来源: {y_best_source} | 同目标样本: {n_matching} 个")

        print("\n--- 下一轮实验推荐（最高 EI 点）---")
        print("    平衡探索与利用，信息量最大")
        for k, v in ei_recommendation.items():
            print(f"  > {k}: {v:.2f}")
        print(f"  > 预测产率 (μ): {ei_mu[0]:.2%}")
        print(f"  > 不确定度 (σ): {ei_sigma[0]:.4f}")

        print("\n--- 当前最优工艺（最高 μ 点）---")
        print("    模型最有信心的最优工艺，适合部署")
        for k, v in mu_recommendation.items():
            print(f"  > {k}: {v:.2f}")
        print(f"  > 预测产率 (μ): {mu_final[0]:.2%}")
        print(f"  > 不确定度 (σ): {mu_sigma_final[0]:.4f}")
        print("="*60 + "\n")

        return {
            'next_experiment': ei_recommendation,
            'best_process': mu_recommendation,
            'ei_yield': ei_mu[0],
            'ei_uncertainty': ei_sigma[0],
            'mu_yield': mu_final[0],
            'mu_uncertainty': mu_sigma_final[0],
        }

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

    def extract_ard_importance(self):
        """从训练好的模型中提取 ARD 长度尺度，返回 DataFrame"""
        kernel = self.gpr.kernel_

        pre_ls = np.atleast_1d(kernel.pre_length_scale)
        target_ls = np.atleast_1d(kernel.target_length_scale)
        proc_ls = np.atleast_1d(kernel.proc_length_scale)

        importance_data = []
        for i, col in enumerate(self.pre_feature_cols):
            if i < len(pre_ls):
                ls = pre_ls[i]
                importance_data.append({
                    'feature': col, 'length_scale': ls,
                    'importance': 1.0 / (ls + 1e-10), 'category': 'EBSD预处理',
                })
        for i, col in enumerate(self.target_cols):
            if i < len(target_ls):
                ls = target_ls[i]
                importance_data.append({
                    'feature': col, 'length_scale': ls,
                    'importance': 1.0 / (ls + 1e-10), 'category': '目标晶向',
                })
        for i, col in enumerate(self.process_cols):
            if i < len(proc_ls):
                ls = proc_ls[i]
                importance_data.append({
                    'feature': col, 'length_scale': ls,
                    'importance': 1.0 / (ls + 1e-10), 'category': '工艺参数',
                })

        return pd.DataFrame(importance_data)

    def _maximin_distance(self, X_candidates, X_reference, feature_cols):
        """
        计算候选点到参考集的 maximin distance（归一化空间）

        Args:
            X_candidates: (n_candidates, d) 候选点矩阵
            X_reference: (n_ref, d) 参考点矩阵（已训练样本 + 已选点）
            feature_cols: 特征列名列表，用于确定归一化范围

        Returns:
            d_min: (n_candidates,) 每个候选点到参考集的最小距离
        """
        from sklearn.preprocessing import MinMaxScaler

        # 合并候选点和参考点，统一归一化
        X_all = np.vstack([X_candidates, X_reference])
        scaler = MinMaxScaler()
        X_all_norm = scaler.fit_transform(X_all)

        X_cand_norm = X_all_norm[:len(X_candidates)]
        X_ref_norm = X_all_norm[len(X_candidates):]

        # 计算欧氏距离矩阵 (n_candidates, n_ref)
        # 使用 ||a-b||² = ||a||² + ||b||² - 2a·b
        sq_cand = np.sum(X_cand_norm ** 2, axis=1, keepdims=True)
        sq_ref = np.sum(X_ref_norm ** 2, axis=1, keepdims=True)
        dists = np.sqrt(np.maximum(sq_cand + sq_ref.T - 2 * X_cand_norm @ X_ref_norm.T, 0))

        # 每个候选点到参考集的最小距离
        d_min = dists.min(axis=1)
        return d_min

    def _greedy_select_mixed_score(self, X_candidates, ei_values, X_train_proc, n_select, alpha=0.5):
        """
        贪心选择：混合 EI + maximin distance，每选一个点后动态更新距离基准

        Args:
            X_candidates: (n_cand, d_proc) 候选点（Process_ 空间）
            ei_values: (n_cand,) 每个候选点的 EI 值
            X_train_proc: (n_train, d_proc) 已训练样本的 Process_ 特征
            n_select: 需要选择的点数
            alpha: EI 权重（0~1），(1-alpha) 为 distance 权重

        Returns:
            selected_indices: 选中的候选点索引
            selected_points: 选中的候选点坐标
        """
        remaining = list(range(len(X_candidates)))
        X_virtual = X_train_proc.copy()
        selected_indices = []

        for _ in range(min(n_select, len(remaining))):
            # 计算当前候选点到虚拟参考集的 maximin distance
            X_cand_remaining = X_candidates[remaining]
            d_min = self._maximin_distance(X_cand_remaining, X_virtual,
                                            self.process_cols[:X_candidates.shape[1]])

            # 计算 rank 百分位
            ei_remaining = ei_values[remaining]
            ei_ranks = np.argsort(np.argsort(ei_remaining)) / max(len(ei_remaining) - 1, 1)
            dist_ranks = np.argsort(np.argsort(d_min)) / max(len(d_min) - 1, 1)

            # 混合得分
            scores = alpha * ei_ranks + (1 - alpha) * dist_ranks
            best_local_idx = np.argmax(scores)
            best_global_idx = remaining[best_local_idx]

            # 选中，虚拟加入参考集
            selected_indices.append(best_global_idx)
            X_virtual = np.vstack([X_virtual, X_candidates[best_global_idx:best_global_idx + 1]])
            remaining.pop(best_local_idx)

        return np.array(selected_indices), X_candidates[selected_indices]


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
    
    # 数据文件命名规则: Optimized_Training_Data_方案X.csv
    data_file = f"Optimized_Training_Data_方案{scheme}.csv"

    # 如果方案文件不存在，尝试使用默认文件
    if not os.path.exists(data_file):
        print(f"\n注意: 未找到 {data_file}，尝试使用默认文件 Optimized_Training_Data.csv")
        data_file = "Optimized_Training_Data.csv"

    print(f"\n当前使用方案 {scheme}: {SCHEME_NAMES[scheme]}")
    print(f"数据文件: {data_file}\n")

    # ==========================
    # 数据管理：添加新数据或直接使用
    # ==========================
    data_file = data_management_menu(data_file)

    # 实例化并训练模型
    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    optimizer.train(data_file)
    
    print("\n模型训练完成！")
    print("\n提示: 使用 predict_new_sample.py 进行新样品的工艺预测")
    print("      运行: python predict_new_sample.py")