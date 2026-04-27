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
from data_builder import (
    extract_macro_rgb_features, 
    find_data_files, 
    merge_multiple_files, 
    extract_features_from_merged_data
)
from contextual_bo_model import ContextualBayesianOptimizer, select_scheme, SCHEME_TARGETS, SCHEME_NAMES, DEFAULT_PROCESS_BOUNDS


def get_pre_file_path():
    """
    交互式获取 pre 数据路径（支持单文件或文件夹）
    """
    print("\n" + "=" * 50)
    print("请输入新样品的预处理数据")
    print("=" * 50)
    print("提示: 可以输入单个 pre.csv 文件路径，或包含多个 pre 文件的文件夹路径")
    
    while True:
        pre_path = input("\n请输入 pre.csv 文件或文件夹路径: ").strip()
        pre_path = pre_path.strip('"\'')  # 移除可能的引号
        
        if not pre_path:
            print("错误: 路径不能为空，请重新输入")
            continue
            
        if not os.path.exists(pre_path):
            print(f"错误: 路径不存在: {pre_path}")
            print("请检查路径是否正确，或重新输入")
            continue
            
        return pre_path


def extract_features_from_path(pre_path):
    """
    从 pre 文件路径或文件夹路径提取特征
    支持: 单文件、包含多个 pre 文件的文件夹
    """
    try:
        if os.path.isdir(pre_path):
            # 文件夹模式：查找并合并所有 pre 文件
            print(f"\n[*] 检测到文件夹路径: {pre_path}")
            pre_files = find_data_files(pre_path, 'pre')
            
            if not pre_files:
                raise ValueError(f"在文件夹 {pre_path} 中未找到任何 pre 文件")
            
            pre_merged = merge_multiple_files(pre_files, file_type='pre')
            features = extract_features_from_merged_data(pre_merged, target_rgbs=None, prefix="Pre_")
            print(f"[*] 已从文件夹合并并提取预处理特征")
        else:
            # 单文件模式
            features = extract_macro_rgb_features(pre_path, target_rgbs=None, prefix="Pre_")
            print(f"\n已从文件读取预处理特征: {pre_path}")
        
        return features
    except Exception as e:
        print(f"读取数据时出错: {e}")
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


def predict_optimal_process(data_file, pre_file, scheme_name, target_orientations=None, 
                            model_path=None, save_model_path=None):
    """
    主预测流程（支持多任务，可指定目标晶向列表）
    
    参数:
        data_file: 训练数据文件路径
        pre_file: 新样品的 pre.csv 文件路径
        scheme_name: 当前使用的方案名称
        target_orientations: 目标晶向列表，如 [(1,0,3), (1,0,2), (3,0,1)]。None表示使用训练数据的目标
        model_path: 预训练模型路径（如果提供则直接加载，否则实时训练）
        save_model_path: 训练后保存模型的路径（可选）
    """
    print(f"\n当前使用方案: {scheme_name}")
    print(f"数据文件: {data_file}")
    if target_orientations:
        targets_str = [f"<{h}{k}{l}>" for h,k,l in target_orientations]
        print(f"目标晶向: {targets_str}")

    # 初始化优化器
    optimizer = ContextualBayesianOptimizer(bounds=DEFAULT_PROCESS_BOUNDS)
    
    # 加载或训练模型
    if model_path and os.path.exists(model_path):
        print(f"\n正在加载预训练模型: {model_path}")
        if not optimizer.load_model(model_path):
            print("[警告] 模型加载失败，将重新训练...")
            print("\n正在训练模型...")
            optimizer.train(data_file)
            if save_model_path:
                optimizer.save_model(save_model_path)
    else:
        if model_path:
            print(f"\n[警告] 模型文件不存在: {model_path}")
        print("\n正在训练模型...")
        optimizer.train(data_file)
        if save_model_path:
            optimizer.save_model(save_model_path)
    
    # 提取 pre 特征
    print("\n正在提取预处理特征...")
    new_sample_features = extract_features_from_path(pre_file)
    preview_features(new_sample_features)
    
    # 执行预测（传入目标晶向列表）
    print("\n正在基于预处理特征推荐最优工艺...")
    best_recipe = optimizer.recommend_next_process(new_sample_features, target_orientations)
    
    return best_recipe


if __name__ == "__main__":
    # 自动检测数据文件，优先使用默认文件
    default_file = "Optimized_Training_Data.csv"
    
    print("\n" + "=" * 50)
    print("请选择目标晶向方案")
    print("=" * 50)
    print("  方案 1: <103> 型织构 (橙色系)")
    print("  方案 2: <114> 型织构 (粉紫色系)")
    print("  方案 3: <124> 型织构 (混合色)")
    print("  方案 4: 自定义组合 (<103>, <114>, <124>)")
    print("-" * 50)
    
    # 检查哪些方案的数据文件存在
    available_schemes = []
    if os.path.exists(default_file):
        available_schemes.append(1)
        available_schemes.append(2)
        available_schemes.append(3)
        available_schemes.append(4)
        print(f"[*] 检测到全局数据文件: {default_file}")
        print(f"    所有方案 (1-4) 均可用")
    else:
        for s in [1, 2, 3, 4]:
            if os.path.exists(f"Optimized_Training_Data_方案{s}.csv"):
                available_schemes.append(s)
        if available_schemes:
            print(f"[*] 可用方案: {available_schemes}")
        else:
            print("[错误] 未找到任何数据文件!")
            sys.exit(1)
    
    # 让用户选择方案
    while True:
        try:
            scheme_input = input(f"\n请输入方案编号 {available_schemes} (默认1): ").strip()
            if not scheme_input:
                scheme = 1
                break
            scheme = int(scheme_input)
            if scheme in available_schemes:
                break
            else:
                print(f"错误: 方案 {scheme} 不可用，请从 {available_schemes} 中选择")
        except ValueError:
            print("错误: 请输入有效的数字")
    
    # 确定数据文件
    if os.path.exists(default_file):
        data_file = default_file
    else:
        data_file = f"Optimized_Training_Data_方案{scheme}.csv"
    
    print(f"[*] 已选择方案 {scheme}: {SCHEME_NAMES[scheme]}")
    print(f"[*] 使用数据文件: {data_file}")
    
    # 模型文件路径
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"model_scheme{scheme}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    # 检查是否存在预训练模型
    use_existing_model = False
    if os.path.exists(model_path):
        print(f"\n[*] 发现预训练模型: {model_path}")
        print("    选项:")
        print("      1. 加载现有模型 (快速)")
        print("      2. 重新训练模型 (耗时但最新)")
        choice = input("    请选择 (1/2, 默认1): ").strip()
        use_existing_model = (choice != '2')
    
    # 获取 pre 文件路径
    pre_file = get_pre_file_path()
    
    # 执行预测（传入该方案的所有目标晶向用于Multi-Hot编码）
    try:
        target_orientations = SCHEME_TARGETS[scheme]
        print(f"[*] 目标晶向: {[f'<{h}{k}{l}>' for h,k,l in target_orientations]}")
        
        # 根据用户选择决定是否加载模型或重新训练
        if use_existing_model:
            best_recipe = predict_optimal_process(
                data_file, pre_file, SCHEME_NAMES[scheme], 
                target_orientations, model_path=model_path
            )
        else:
            best_recipe = predict_optimal_process(
                data_file, pre_file, SCHEME_NAMES[scheme], 
                target_orientations, save_model_path=model_path
            )
        print("\n预测完成!")
        print(f"[*] 模型已保存至: {model_path}")
    except Exception as e:
        print(f"\n预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
