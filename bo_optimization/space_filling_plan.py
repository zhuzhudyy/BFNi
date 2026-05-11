"""
空间填充实验规划器（支持增量分配）

用途：新原料到货后，为每份样品分配工艺参数，提升全局模型精度。
支持增量模式：首次运行生成空间填充计划并保存状态文件，
后续每次到新样品时运行，自动分配下一个未使用的采样点。

使用方法:
    python space_filling_plan.py

流程:
    1. 选择方案（目标晶向）
    2. 输入新样品的 pre.csv 文件或文件夹路径
    3. 系统从状态文件中分配下一个空间填充工艺参数
    4. 输出实验计划表，更新状态文件
"""

import os
import sys
import json
from datetime import datetime
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
    DEFAULT_PROCESS_BOUNDS, PROCESS_LABELS_CN, DEFAULT_MIN_PHYSICAL_DISTANCES
)

SPARSITY_DECAY = 0.5  # 每轮间距缩小比例


def _get_state_path(scheme):
    """获取状态文件路径"""
    return f"space_filling_state_方案{scheme}.json"


def _load_state(scheme):
    """加载状态文件，返回 (state_dict, None) 或 (None, error_msg)"""
    path = _get_state_path(scheme)
    if not os.path.exists(path):
        return None, "状态文件不存在"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return state, None
    except Exception as e:
        return None, f"读取状态文件失败: {e}"


def _save_state(scheme, state):
    """保存状态文件"""
    path = _get_state_path(scheme)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _init_state(scheme, recommendations, alpha, round=1, sparsity_multiplier=1.0):
    """首次运行：从 suggest_space_filling 结果初始化状态"""
    points = []
    for rec in recommendations:
        point = {k: v for k, v in rec.items() if k.startswith('Process_')}
        point['cluster'] = rec.get('cluster', 0)
        point['ei'] = rec.get('ei', 0.0)
        point['round'] = round
        points.append(point)

    state = {
        'scheme': scheme,
        'created_at': datetime.now().isoformat(),
        'round': round,
        'sparsity_multiplier': sparsity_multiplier,
        'alpha': alpha,
        'points': points,
        'allocations': [],  # [{index, sample_name, assigned_at}]
    }
    _save_state(scheme, state)
    return state


def _get_next_unallocated(state):
    """返回下一个未分配的点索引，None 表示全部已分配"""
    allocated_indices = {a['index'] for a in state['allocations']}
    for i in range(len(state['points'])):
        if i not in allocated_indices:
            return i
    return None


def _print_state_summary(state):
    """打印当前状态摘要"""
    n_total = len(state['points'])
    n_allocated = len(state['allocations'])
    n_remaining = n_total - n_allocated

    print(f"\n{'='*60}")
    print(f"空间填充状态 (方案 {state['scheme']})")
    print(f"{'='*60}")
    print(f"  总采样点: {n_total}")
    print(f"  已分配:   {n_allocated}")
    print(f"  剩余:     {n_remaining}")

    if n_allocated > 0:
        print(f"\n  已分配记录:")
        for a in state['allocations']:
            idx = a['index']
            pt = state['points'][idx]
            proc_str = ", ".join([f"{k.replace('Process_', '')}={v:.1f}"
                                  for k, v in pt.items() if k.startswith('Process_')])
            print(f"    #{idx+1} → {a['sample_name']}  [{proc_str}]  ({a['assigned_at'][:10]})")

    print(f"{'='*60}")
    return n_remaining


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
    """主流程（支持增量分配）"""
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

    # 4. 加载或初始化状态
    state, err = _load_state(scheme)
    if state is None:
        print(f"\n[状态] {err}，需要初始化空间填充计划")
        n_total = int(input("计划生成多少个空间填充采样点？(默认 10): ").strip() or "10")

        sparsity_multiplier = 1.0
        effective_dist = {k: v * sparsity_multiplier for k, v in DEFAULT_MIN_PHYSICAL_DISTANCES.items()}
        print(f"\n正在生成 {n_total} 个空间填充采样点（间距约束: Temp±{effective_dist['Process_Temp']:.0f}°C, "
              f"Time±{effective_dist['Process_Time']:.0f}h, H₂±{effective_dist['Process_H2']:.0f}sccm, "
              f"Ar±{effective_dist['Process_Ar']:.0f}sccm）...")
        recommendations = optimizer.suggest_space_filling(
            n_total_points=n_total, alpha=0.5, min_physical_distances=effective_dist
        )
        print(f"实际选出: {len(recommendations)} 个采样点")
        state = _init_state(scheme, recommendations, alpha=0.5, round=1, sparsity_multiplier=sparsity_multiplier)
        print(f"状态已保存至: {_get_state_path(scheme)}")
    else:
        # 向后兼容：旧状态文件可能没有 round/sparsity_multiplier
        if 'round' not in state:
            state['round'] = 1
        if 'sparsity_multiplier' not in state:
            state['sparsity_multiplier'] = 1.0

    # 5. 显示状态并检查剩余点
    n_remaining = _print_state_summary(state)

    if n_remaining == 0:
        round_num = state.get('round', 1)
        sparsity_multiplier = state.get('sparsity_multiplier', 1.0)
        print(f"\n第 {round_num} 轮采样点已全部分配完毕！（当前间距倍率: {sparsity_multiplier}）")

        next_multiplier = sparsity_multiplier * SPARSITY_DECAY
        if next_multiplier < 0.1:
            print("间距倍率已低于 0.1，不再继续细化。退出。")
            return

        choice = input(f"是否进入第 {round_num + 1} 轮（间距 ×{next_multiplier}）？(y/n): ").strip().lower()
        if choice == 'y':
            n_new = int(input("新一批采样点数 (默认 10): ").strip() or "10")
            sparsity_multiplier = next_multiplier
            effective_dist = {k: v * sparsity_multiplier for k, v in DEFAULT_MIN_PHYSICAL_DISTANCES.items()}
            print(f"\n正在生成 {n_new} 个新的空间填充采样点（间距约束: Temp±{effective_dist['Process_Temp']:.0f}°C, "
                  f"Time±{effective_dist['Process_Time']:.0f}h, H₂±{effective_dist['Process_H2']:.0f}sccm, "
                  f"Ar±{effective_dist['Process_Ar']:.0f}sccm）...")
            recommendations = optimizer.suggest_space_filling(
                n_total_points=n_new, alpha=0.5, min_physical_distances=effective_dist
            )
            print(f"实际选出: {len(recommendations)} 个采样点")
            for rec in recommendations:
                point = {k: v for k, v in rec.items() if k.startswith('Process_')}
                point['cluster'] = rec.get('cluster', 0)
                point['ei'] = rec.get('ei', 0.0)
                point['round'] = round_num + 1
                state['points'].append(point)
            state['round'] = round_num + 1
            state['sparsity_multiplier'] = sparsity_multiplier
            _save_state(scheme, state)
            n_remaining = _print_state_summary(state)
        else:
            print("退出。")
            return

    # 6. 获取新样品数据
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

    # 匹配特征列（仅用于完整性检查，实际分配不依赖特征值）
    pre_cols = optimizer.pre_feature_cols
    missing = [c for c in pre_cols if c not in new_features.columns]
    if missing:
        print(f"[警告] 新样品缺少 {len(missing)} 个特征: {missing[:5]}...")

    # 7. 为每份样品分配下一个未使用的采样点
    assignments = []
    for i in range(n_new):
        idx = _get_next_unallocated(state)
        if idx is None:
            print(f"\n[警告] 采样点已用完，样品 {i+1} 无法分配")
            break

        point = state['points'][idx]
        sample_name = os.path.basename(pre_path) if os.path.isfile(pre_path) else f"样品{i+1}"

        # 记录分配
        state['allocations'].append({
            'index': idx,
            'sample_name': sample_name,
            'sample_path': pre_path,
            'assigned_at': datetime.now().isoformat(),
        })

        assignments.append({
            '样品编号': i + 1,
            '样品文件': sample_name,
            **{k: f"{v:.1f}" for k, v in point.items() if k.startswith('Process_')},
            '空间填充点': f"#{idx+1}",
            'Cluster': point.get('cluster', '-'),
            'EI': f"{point.get('ei', 0):.6f}",
        })

    # 保存更新后的状态
    _save_state(scheme, state)

    # 8. 输出实验计划
    if not assignments:
        print("\n没有可分配的样品。")
        return

    df_plan = pd.DataFrame(assignments)
    print("\n" + "=" * 60)
    print("本次分配结果")
    print("=" * 60)
    print(df_plan.to_string(index=False))

    # 保存（追加到已有文件）
    output_file = f"space_filling_plan_方案{scheme}.csv"
    if os.path.exists(output_file):
        df_old = pd.read_csv(output_file, encoding='utf-8-sig')
        df_plan = pd.concat([df_old, df_plan], ignore_index=True)
    df_plan.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n已保存至: {output_file}")

    # 9. 提示下一步
    n_remaining_after = len(state['points']) - len(state['allocations'])
    print(f"\n{'='*60}")
    print("下一步:")
    print(f"  1. 按计划表中的工艺参数进行退火实验")
    print(f"  2. 测量产率后，运行 data_builder.py 合并新数据")
    print(f"  3. 下次到新样品时，再运行本脚本分配下一个点")
    print(f"  [剩余采样点: {n_remaining_after}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_space_filling_plan()
