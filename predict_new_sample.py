"""
基于训练好的贝叶斯模型，为新样品预测最优工艺参数

使用方法:
    python predict_new_sample.py
    
流程:
    1. 选择方案 (目标晶向)
    2. 输入 pre.csv 文件路径
    3. 获得最优工艺推荐
"""

import os
import sys
import pandas as pd
import numpy as np

# 导入必要的函数和类
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_builder import extract_macro_rgb_features
from contextual_bo_model import ContextualBayesianOptimizer, select_scheme


def get_pre_file_path():
    """
    交互式获取 pre.csv 文件路径
    """
    print("\n" + "=" * 50)
    print("请输入新样品的预处理数据")
    print("=" * 50)
    
    while True:
        pre_file = input("\n请输入 pre.csv 文件路径: ").strip()
        pre_file = pre_file.strip('"\'')  # 移除可能的引号
        
        if not pre_file:
            print("错误: 文件路径不能为空，请重新输入")
            continue
            
        if not os.path.exists(pre_file):
            print(f"错误: 文件不存在: {pre_file}")
            print("请检查路径是否正确，或重新输入")
            continue
            
        return pre_file


def extract_features_from_file(pre_file):
    """
    从 pre.csv 文件提取特征
    """
    try:
        features = extract_macro_rgb_features(pre_file, target_rgbs=None, prefix="Pre_")
        print(f"\n已从文件读取预处理特征: {pre_file}")
        return features
    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise


def preview_features(features, n=5):
    """
    预览提取的特征
    """
    print("\n输入的预处理特征预览:")
    for k, v in list(features.items())[:n]:
        print(f"  {k}: {v:.4f}")
    if len(features) > n:
        print(f"  ... 共 {len(features)} 个特征")


def predict_optimal_process(data_file, pre_file, scheme_name):
    """
    主预测流程
    
    参数:
        data_file: 训练数据文件路径
        pre_file: 新样品的 pre.csv 文件路径
        scheme_name: 当前使用的方案名称
    """
    # 定义工艺参数搜索边界
    process_bounds = {
        'Process_Temp': (1000.0, 1500.0),  # 退火温度 (℃)
        'Process_Time': (1.0, 30.0),       # 保温时间 (h)
        'Process_H2': (0.0, 160.0),        # H2 流量 (sccm)
        'Process_Ar': (0.0, 800.0)         # Ar 流量 (sccm)
    }
    
    print(f"\n当前使用方案: {scheme_name}")
    print(f"数据文件: {data_file}")
    
    # 加载并训练模型
    print("\n正在加载模型...")
    optimizer = ContextualBayesianOptimizer(bounds=process_bounds)
    optimizer.train(data_file)
    
    # 提取 pre 特征
    print("\n正在提取预处理特征...")
    new_sample_features = extract_features_from_file(pre_file)
    preview_features(new_sample_features)
    
    # 执行预测
    print("\n正在基于预处理特征推荐最优工艺...")
    best_recipe = optimizer.recommend_next_process(new_sample_features)
    
    return best_recipe


if __name__ == "__main__":
    # 方案名称映射
    scheme_names = {
        1: "<103> 型织构 (橙色系)",
        2: "<114> 型织构 (粉紫色系)",
        3: "<124> 型织构 (混合色)",
        4: "自定义组合"
    }
    
    # 自动检测数据文件，优先使用默认文件
    default_file = "Optimized_Training_Data.csv"
    
    if os.path.exists(default_file):
        data_file = default_file
        scheme = 1  # 默认使用方案1的名称
        print(f"[*] 使用默认数据文件: {data_file}")
        print(f"[*] 使用默认方案: {scheme_names[scheme]}")
    else:
        # 如果没有默认文件，提示选择方案
        print("[*] 默认数据文件不存在，请选择方案...")
        scheme = select_scheme()
        data_file = f"Optimized_Training_Data_方案{scheme}.csv"
        
        if not os.path.exists(data_file):
            print(f"\n[错误] 未找到数据文件: {data_file}")
            print("请确保数据文件存在于当前目录。")
            sys.exit(1)
    
    # 获取 pre 文件路径
    pre_file = get_pre_file_path()
    
    # 执行预测
    try:
        best_recipe = predict_optimal_process(data_file, pre_file, scheme_names[scheme])
        print("\n预测完成!")
    except Exception as e:
        print(f"\n预测过程中出错: {e}")
        sys.exit(1)
