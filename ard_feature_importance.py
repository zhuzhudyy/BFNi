"""
ARD (Automatic Relevance Determination) 特征重要性可视化

功能:
    1. 训练高斯过程模型并启用 ARD
    2. 提取每个特征的长度尺度参数
    3. 可视化特征重要性排序（水平条形图）
    4. 按特征类别分组展示

输出:
    保存路径: D:\毕业设计\织构数据\visualization\ard_importance
    文件名: Scheme{ID}_N{样本量}_{时间戳}.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contextual_bo_model import ContextualBayesianOptimizer


def train_model_with_ard(data_file, scheme_id):
    """训练模型并返回优化器实例"""
    process_bounds = {
        'Process_Temp': (1000.0, 1500.0),
        'Process_Time': (1.0, 30.0),
        'Process_H2': (0.0, 160.0),
        'Process_Ar': (0.0, 800.0)
    }
    
    optimizer = ContextualBayesianOptimizer(bounds=process_bounds)
    optimizer.train(data_file)
    
    return optimizer


def extract_ard_importance(optimizer):
    """从训练好的模型中提取 ARD 长度尺度作为特征重要性"""
    kernel = optimizer.gpr.kernel_
    matern_kernel = kernel.k1.k2  # 获取 Matern 核
    length_scales = matern_kernel.length_scale
    
    all_feature_cols = optimizer.pre_feature_cols + optimizer.target_cols + optimizer.process_cols
    
    # 构建特征重要性数据
    importance_data = []
    for i, col in enumerate(all_feature_cols):
        if i < len(length_scales):
            ls = length_scales[i]
            # 重要性分数：长度尺度的倒数（越小越重要）
            importance_score = 1.0 / (ls + 1e-10)
            
            # 确定特征类别
            if col.startswith('Pre_'):
                category = 'EBSD预处理'
            elif col.startswith('Target_'):
                category = '目标晶向'
            else:
                category = '工艺参数'
            
            importance_data.append({
                'feature': col,
                'length_scale': ls,
                'importance': importance_score,
                'category': category
            })
    
    return pd.DataFrame(importance_data)


def plot_ard_importance(df_importance, scheme_id, n_samples, output_dir):
    """绘制 ARD 特征重要性可视化"""
    
    # 按重要性排序
    df_sorted = df_importance.sort_values('importance', ascending=True)
    
    # 创建颜色映射
    color_map = {
        'EBSD预处理': '#3498db',  # 蓝色
        '目标晶向': '#e74c3c',     # 红色
        '工艺参数': '#2ecc71'      # 绿色
    }
    colors = [color_map[cat] for cat in df_sorted['category']]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # ========== 左图：特征重要性排序（水平条形图）==========
    y_pos = np.arange(len(df_sorted))
    bars = ax1.barh(y_pos, df_sorted['importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['feature'], fontsize=9)
    ax1.set_xlabel('重要性分数 (1/Length Scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'ARD 特征重要性排序\n方案 {scheme_id} | 样本量 N={n_samples}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax1.text(row['importance'] + 0.01, i, f"{row['length_scale']:.3f}", 
                va='center', fontsize=8, color='darkred')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat, alpha=0.8) 
                      for cat in ['EBSD预处理', '目标晶向', '工艺参数']]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # ========== 右图：按类别分组的箱线图 ==========
    categories = ['EBSD预处理', '目标晶向', '工艺参数']
    cat_data = [df_importance[df_importance['category'] == cat]['length_scale'].values 
                for cat in categories]
    
    bp = ax2.boxplot(cat_data, labels=categories, patch_artist=True, 
                     notch=True, showmeans=True)
    
    # 设置箱线图颜色
    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(color_map[cat])
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Length Scale (对数刻度)', fontsize=12, fontweight='bold')
    ax2.set_title('各类特征的长度尺度分布\n(越小表示越重要)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加说明文字
    ax2.text(0.5, 0.95, 'Length Scale < 1: 高重要性\nLength Scale > 100: 低重要性', 
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"Scheme{scheme_id}_N{n_samples}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] ARD 特征重要性图已保存至: {save_path}")
    
    plt.show()
    
    return save_path


def print_top_features(df_importance, top_n=5):
    """打印最重要的特征"""
    print("\n" + "="*60)
    print(f"           最重要的 {top_n} 个特征 (按 ARD 长度尺度)")
    print("="*60)
    
    df_sorted = df_importance.sort_values('length_scale', ascending=True)
    
    print(f"\n{'排名':<4} {'特征名':<25} {'类别':<12} {'长度尺度':<12} {'重要性'}")
    print("-" * 70)
    
    for i, (idx, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        importance_str = "★★★" if row['length_scale'] < 0.5 else ("★★" if row['length_scale'] < 1.0 else "★")
        print(f"{i:<4} {row['feature']:<25} {row['category']:<12} {row['length_scale']:<12.4f} {importance_str}")
    
    print("="*60)


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
    
    if not os.path.exists(default_file):
        print("[!] 未找到数据文件，请确保 Optimized_Training_Data.csv 存在")
        sys.exit(1)
    
    data_file = default_file
    print(f"[*] 使用数据文件: {data_file}")
    
    print("\n请选择要分析的目标晶向方案:")
    print("  方案 1: <103>, <102>, <301>")
    print("  方案 2: <114>, <115>, <105>")
    print("  方案 3: <124>, <125>, <214>")
    print("  方案 4: <103>, <114>, <124>")
    
    while True:
        try:
            scheme_input = input("\n请输入方案编号 (1-4，直接回车默认方案1): ").strip()
            if scheme_input == "":
                scheme_id = 1
            else:
                scheme_id = int(scheme_input)
            if scheme_id in [1, 2, 3, 4]:
                break
            else:
                print("[!] 请输入 1-4 之间的数字")
        except ValueError:
            print("[!] 请输入有效的数字")
    
    print(f"[*] 已选择方案 {scheme_id}: {scheme_targets[scheme_id]}")
    
    # 训练模型
    print("\n正在训练高斯过程模型 (启用 ARD)...")
    optimizer = train_model_with_ard(data_file, scheme_id)
    
    # 获取样本量
    df = pd.read_csv(data_file)
    n_samples = len(df)
    
    # 提取 ARD 重要性
    print("\n正在提取 ARD 特征重要性...")
    df_importance = extract_ard_importance(optimizer)
    
    # 打印最重要的特征
    print_top_features(df_importance, top_n=10)
    
    # 创建输出目录
    output_dir = r"D:\毕业设计\织构数据\visualization\ard_importance"
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制可视化
    print("\n正在绘制 ARD 特征重要性可视化...")
    plot_ard_importance(df_importance, scheme_id, n_samples, output_dir)
    
    print(f"\n分析完成！结果已保存至: {output_dir}/")
