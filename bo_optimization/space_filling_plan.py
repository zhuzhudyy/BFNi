"""
空间填充实验规划器

用途：新原料到货后，为每份样品分配工艺参数，提升全局模型精度。

使用方法:
    python space_filling_plan.py

流程:
    1. 选择方案（目标晶向）
    2. 输入新样品的 pre.csv 文件或文件夹路径
    3. 系统推荐空间填充工艺参数，并将样品与工艺配对
    4. 输出实验计划表
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import pandas as pd
import numpy as np

from bo_optimization.data_builder import (
    extract_macro_rgb_features,
    find_data_files,
)
from bo_optimization.contextual_bo_model import (
    ContextualBayesianOptimizer, select_scheme, SCHEME_TARGETS, SCHEME_NAMES,
    DEFAULT_PROCESS_BOUNDS, PROCESS_LABELS_CN
)


def _extract_single_sample(folder_path):
    """从单个实验文件夹提取一份样品的特征（多次测量取均值）"""
    pre_files = find_data_files(folder_path, 'pre')
    if not pre_files:
        return None
    all_features = []
    for f in pre_files:
        feat = extract_macro_rgb_features(f, prefix="Pre_")
        if isinstance(feat, dict):
            all_features.append(feat)
    if not all_features:
        return None
    df_multi = pd.DataFrame(all_features)
    # 多次测量取均值 → 一行代表该样品
    return df_multi.mean(axis=0).to_frame().T


def extract_pre_features(pre_path):
    """
    从路径提取样品特征，返回 DataFrame（每行一个样品）

    支持三种输入：
    - 单个 pre.csv 文件 → 1 份样品
    - 单个实验文件夹（含 pre*.csv） → 1 份样品（多次测量取均值）
    - 父文件夹（含多个子文件夹） → 每个子文件夹 = 1 份样品
    """
    if not os.path.isdir(pre_path):
        # 单个文件
        feat = extract_macro_rgb_features(pre_path, prefix="Pre_")
        return pd.DataFrame([feat] if isinstance(feat, dict) else feat)

    # 检查是否直接包含 pre 文件（单实验文件夹）
    pre_files = find_data_files(pre_path, 'pre')
    if pre_files:
        sample_df = _extract_single_sample(pre_path)
        if sample_df is not None:
            print(f"  单样品文件夹（{len(pre_files)} 次测量取均值）")
            return sample_df

    # 父文件夹：遍历子文件夹
    subfolders = [os.path.join(pre_path, d) for d in os.listdir(pre_path)
                  if os.path.isdir(os.path.join(pre_path, d))]
    all_samples = []
    for sub in sorted(subfolders):
        sample_df = _extract_single_sample(sub)
        if sample_df is not None:
            sample_df.index = [os.path.basename(sub)]
            all_samples.append(sample_df)

    if not all_samples:
        raise ValueError(f"在 {pre_path} 中未找到任何包含 pre 文件的样品文件夹")

    features = pd.concat(all_samples, ignore_index=True)
    print(f"  发现 {len(features)} 份样品")
    return features


def assign_samples_to_clusters(X_pre_new, X_pre_train, labels_train, k):
    """将新样品分配到最近的 cluster"""
    from sklearn.cluster import KMeans
    # 用训练数据的 cluster 中心
    centers = np.zeros((k, X_pre_train.shape[1]))
    for c in range(k):
        centers[c] = X_pre_train[labels_train == c].mean(axis=0)
    # 分配
    dists = np.sqrt(((X_pre_new[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    return dists.argmin(axis=1)


def run_space_filling_plan():
    """主流程"""
    # 1. 选择方案
    scheme = select_scheme()

    # 2. 加载模型
    model_path = f"trained_models/model_scheme{scheme}.pkl"
    if not os.path.exists(model_path):
        model_path = "trained_models/model_scheme1.pkl"
        print(f"未找到方案 {scheme} 的模型，使用方案 1")

    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    if not optimizer.load_model(model_path):
        print("模型加载失败，尝试从数据重新训练...")
        data_file = f"Optimized_Training_Data_方案{scheme}.csv"
        if not os.path.exists(data_file):
            data_file = "Optimized_Training_Data.csv"
        optimizer.train(data_file)

    # 3. 显示当前覆盖率
    optimizer.print_coverage_report()

    # 4. 获取新样品数据
    print("=" * 50)
    print("请输入新样品的预处理数据")
    print("=" * 50)
    pre_path = input("pre.csv 文件或文件夹路径: ").strip().strip('"\'')

    if not os.path.exists(pre_path):
        print(f"错误: 路径不存在: {pre_path}")
        return

    # 提取特征
    new_features = extract_pre_features(pre_path)
    n_new = len(new_features)
    print(f"\n提取到 {n_new} 份新样品的特征")

    # 匹配特征列
    pre_cols = optimizer.pre_feature_cols
    missing = [c for c in pre_cols if c not in new_features.columns]
    if missing:
        print(f"[警告] 新样品缺少 {len(missing)} 个特征: {missing[:5]}...")
        # 用训练数据均值填充
        for col in missing:
            new_features[col] = optimizer.training_df[col].mean()

    X_pre_new = new_features[pre_cols].values

    # 5. 询问分配策略
    print(f"\n分配策略:")
    print(f"  [1] 每份样品分配不同工艺（最大化空间覆盖，推荐）")
    print(f"  [2] 所有样品用同一套工艺（验证工艺普适性）")

    while True:
        choice = input("请选择 (1-2): ").strip()
        if choice in ['1', '2']:
            break
        print("请输入 1 或 2")

    # 6. 生成推荐
    if choice == '1':
        # 每份样品分配不同工艺
        recommendations = optimizer.suggest_space_filling(
            n_total_points=n_new, alpha=0.5
        )

        # 将样品分配到推荐点（按 Pre_ 特征聚类匹配）
        if n_new <= len(recommendations):
            # 简单方案：按样品顺序分配
            assignments = []
            for i in range(n_new):
                rec = recommendations[i]
                assignments.append({
                    '样品编号': i + 1,
                    '样品文件': os.path.basename(pre_path) if os.path.isfile(pre_path) else f"样品{i+1}",
                    **{k: f"{v:.1f}" for k, v in rec.items() if k.startswith('Process_')},
                    'Cluster': rec.get('cluster', '-'),
                    'EI': f"{rec.get('ei', 0):.6f}",
                })
        else:
            # 推荐点不够，复用
            assignments = []
            for i in range(n_new):
                rec = recommendations[i % len(recommendations)]
                assignments.append({
                    '样品编号': i + 1,
                    '样品文件': os.path.basename(pre_path) if os.path.isfile(pre_path) else f"样品{i+1}",
                    **{k: f"{v:.1f}" for k, v in rec.items() if k.startswith('Process_')},
                    'Cluster': rec.get('cluster', '-'),
                    'EI': f"{rec.get('ei', 0):.6f}",
                })
    else:
        # 所有样品用同一套工艺
        rec = optimizer.recommend_next_process(X_pre_new[0])
        assignments = []
        for i in range(n_new):
            assignments.append({
                '样品编号': i + 1,
                '样品文件': os.path.basename(pre_path) if os.path.isfile(pre_path) else f"样品{i+1}",
                **{k: f"{v:.1f}" for k, v in rec['next_experiment'].items()},
                'Cluster': '-',
                'EI': '-',
            })

    # 7. 输出实验计划
    df_plan = pd.DataFrame(assignments)
    print("\n" + "=" * 60)
    print("实验计划表")
    print("=" * 60)
    print(df_plan.to_string(index=False))

    # 保存
    output_file = f"space_filling_plan_方案{scheme}.csv"
    df_plan.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n已保存至: {output_file}")

    # 8. 提示下一步
    print(f"\n{'='*60}")
    print("下一步:")
    print(f"  1. 按计划表中的工艺参数进行退火实验")
    print(f"  2. 测量产率后，运行 data_builder.py 合并新数据")
    print(f"  3. 重新运行本脚本查看更新后的覆盖率")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_space_filling_plan()
