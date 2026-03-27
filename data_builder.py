import os
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

def hkl_to_aztec_rgb(h, k, l):
    """
    黑科技：直接将密勒指数 (h,k,l) 换算为 AZtecCrystal 的标准 IPF Z RGB 颜色
    """
    # 强制排序，映射到标准极图三角形 (0 <= x <= y <= z)
    hkl = np.sort(np.abs([h, k, l]))
    x, y, z = hkl[0], hkl[1], hkl[2]
    if z == 0: return (0, 0, 0)
    
    # 向量归一化
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm
    
    # 立体投影到 XY 平面
    X = x / (1 + z)
    Y = y / (1 + z)
    
    # 立体投影下的三个顶点
    P_001 = np.array([0, 0])
    P_101 = np.array([0, np.sqrt(2) - 1])
    P_111 = np.array([1/(np.sqrt(3)+1), 1/(np.sqrt(3)+1)])
    
    # 重心坐标系混合计算
    v0 = P_001 - P_111
    v1 = P_101 - P_111
    v2 = np.array([X, Y]) - P_111
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    w_R = (d11 * d20 - d01 * d21) / denom  # 001 (红色) 权重
    w_G = (d00 * d21 - d01 * d20) / denom  # 101 (绿色) 权重
    w_B = 1.0 - w_R - w_G                  # 111 (蓝色) 权重
    
    r, g, b = max(0, w_R), max(0, w_G), max(0, w_B)
    
    # 拉伸亮度 (AZtec 默认会将最亮通道拉满到 255)
    max_c = max(r, g, b)
    if max_c > 0:
        r, g, b = r/max_c, g/max_c, b/max_c
        
    return (int(r*255), int(g*255), int(b*255))

def extract_macro_rgb_features(csv_path, target_rgbs=None, tolerance=50, prefix="Pre_"):
    # 强制文本切割法 (解决 AZtec 表头报错)
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        with open(csv_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()
            
    header_idx = 0
    for i, line in enumerate(lines):
        if line.count(',') > 5 and ('Phase' in line or 'X' in line or 'Euler' in line or 'IPF' in line):
            header_idx = i
            break
            
    clean_csv_data = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(clean_csv_data), low_memory=False)
    
    features = {}
    
    # --- 1. 处理 IPF 色彩 ---
    ipf_cols = [c for c in df.columns if 'IPF' in c and 'color' in c.lower()]
    if ipf_cols:
        ipf_col = ipf_cols[0]
        rgb_df = df[ipf_col].astype(str).str.extract(r'(\d+)\s+(\d+)\s+(\d+)').astype(float)
        rgb_df.columns = ['R', 'G', 'B']
        rgb_df = rgb_df.dropna()
        
        if not target_rgbs:
            # 提取预处理宏观特征
            features[f'{prefix}R_Mean'] = rgb_df['R'].mean()
            features[f'{prefix}G_Mean'] = rgb_df['G'].mean()
            features[f'{prefix}B_Mean'] = rgb_df['B'].mean()
            features[f'{prefix}R_Std']  = rgb_df['R'].std()
            features[f'{prefix}G_Std']  = rgb_df['G'].std()
            features[f'{prefix}B_Std']  = rgb_df['B'].std()
        else:
            # 只要像素的颜色落在【任意一个】目标晶向的容差范围内，就算作单晶化成功！
            min_dist = pd.Series(np.inf, index=rgb_df.index)
            for t_rgb in target_rgbs:
                dist = np.sqrt((rgb_df['R'] - t_rgb[0])**2 + 
                               (rgb_df['G'] - t_rgb[1])**2 + 
                               (rgb_df['B'] - t_rgb[2])**2)
                min_dist = np.minimum(min_dist, dist)
                
            features['TARGET_Yield'] = (min_dist <= tolerance).mean()

    # --- 2. 连续变量处理 ---
    hq_cols = [c for c in df.columns if 'Half quadratic' in c or 'HQ' in c or 'GND' in c]
    if hq_cols:
        hq_col = hq_cols[0]
        df[hq_col] = pd.to_numeric(df[hq_col], errors='coerce')
        features[f'{prefix}Defect_Mean'] = df[hq_col].mean()  
        features[f'{prefix}Defect_Std'] = df[hq_col].std()    

    return features

def build_training_dataset(root_dir, target_indices, color_tolerance=50):
    # 将用户输入的密勒指数自动转换为 AZtec 颜色
    target_rgbs = [hkl_to_aztec_rgb(h, k, l) for h, k, l in target_indices]
        
    print(f"正在追踪的单晶目标面及其计算颜色：")
    for idx, (h,k,l) in enumerate(target_indices):
        print(f"   <{h}{k}{l}> -> RGB: {target_rgbs[idx]}")
        
    print(f"\n开始扫描实验数据库并进行色彩降维：{root_dir}")
    all_experiments = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path): continue
            
        pre_path = os.path.join(folder_path, 'pre.csv')
        done_path = os.path.join(folder_path, 'done.csv')
        # 匹配 condition 文件（支持 .xls Excel 或 .csv 格式）
        cond_path = os.path.join(folder_path, 'condition.xls')
        if not os.path.exists(cond_path):
            cond_path = os.path.join(folder_path, 'condition.csv') 

        if os.path.exists(pre_path) and os.path.exists(done_path) and os.path.exists(cond_path):
            try:
                exp_data = extract_macro_rgb_features(pre_path, target_rgbs=None, prefix="Pre_")
                
                # 读取工艺条件（兼容 Excel 和 CSV 格式）
                if cond_path.endswith('.xls') or cond_path.endswith('.xlsx'):
                    cond_df = pd.read_excel(cond_path)
                else:
                    try:
                        cond_df = pd.read_csv(cond_path, encoding='utf-8')
                    except:
                        cond_df = pd.read_csv(cond_path, encoding='gbk')
                
                # 提取工艺参数（兼容不同命名格式）
                if 'temprature(℃)' in cond_df.columns:
                    exp_data['Process_Temp'] = cond_df['temprature(℃)'].iloc[0]
                elif 'Temp' in cond_df.columns:
                    exp_data['Process_Temp'] = cond_df['Temp'].iloc[0]
                    
                if 'time(h)' in cond_df.columns:
                    exp_data['Process_Time'] = cond_df['time(h)'].iloc[0]
                elif 'Time' in cond_df.columns:
                    exp_data['Process_Time'] = cond_df['Time'].iloc[0]
                    
                if 'H2' in cond_df.columns: exp_data['Process_H2'] = cond_df['H2'].iloc[0]
                if 'Ar' in cond_df.columns: exp_data['Process_Ar'] = cond_df['Ar'].iloc[0]
                
                # 计算综合产率
                done_features = extract_macro_rgb_features(done_path, target_rgbs=target_rgbs, tolerance=color_tolerance)
                exp_data['TARGET_Yield'] = done_features.get('TARGET_Yield', 0.0) 
                
                exp_data['Sample_ID'] = folder_name
                all_experiments.append(exp_data)
                print(f"成功提取批次：{folder_name} (综合目标产率：{exp_data['TARGET_Yield']:.1%})")
                            
            except Exception as e:
                print(f"解析文件夹 {folder_name} 时出错：{e}")

    final_df = pd.DataFrame(all_experiments).fillna(0)
    return final_df

if __name__ == "__main__":
    # ==========================
    # 核心参数设置区域
    # ==========================
    ROOT_DATA_DIR = r"D:\毕业设计\织构数据\数据总结" 
    
    # ==========================================
    # 🎯 目标晶向配置区（多方案备选）
    # ==========================================
    # 使用说明：
    # 1. 在下方定义多个目标晶向组合
    # 2. 修改 CURRENT_TARGET 来选择使用哪个组合
    # 3. 每个组合可以包含多个等价的晶向指数
    # ==========================================
    
    # 方案 1: <103> 型织构（橙色系，根据实际数据推荐）
    TARGET_SET_1 = [
        (1, 0, 3),   # <103> - 主目标
        (1, 0, 2),   # <102> - 相近织构
        (3, 0, 1),   # <301> - 对称等价
    ]
    
    # 方案 2: <114> 型织构（粉紫色系）
    TARGET_SET_2 = [
        (1, 1, 4),
        (1, 1, 5),
        (1, 0, 5),
    ]
    
    # 方案 3: <124> 型织构（混合色）
    TARGET_SET_3 = [
        (1, 2, 4),
        (1, 2, 5),
        (2, 1, 4),
    ]
    
    # 方案 4: 自定义组合（你可以自由修改）
    TARGET_SET_4 = [
        (1, 0, 3),
        (1, 1, 4),
        (1, 2, 4),
    ]
    
    # 🎯 当前使用的目标晶向组合（修改这里的数字来切换方案）
    CURRENT_TARGET = 1  # 可选值：1, 2, 3, 4
    
    # 根据选择加载对应的晶向组合
    if CURRENT_TARGET == 1:
        TARGET_INDICES = TARGET_SET_1
        print("当前使用方案 1: <103> 型织构（橙色系）")
    elif CURRENT_TARGET == 2:
        TARGET_INDICES = TARGET_SET_2
        print("当前使用方案 2: <114> 型织构（粉紫色系）")
    elif CURRENT_TARGET == 3:
        TARGET_INDICES = TARGET_SET_3
        print("当前使用方案 3: <124> 型织构（混合色）")
    else:
        TARGET_INDICES = TARGET_SET_4
        print("当前使用方案 4: 自定义组合")
    
    # 色差容忍度 (0=必须纯色，50=稍微带渐变，80+=非常宽松)
    # 根据实际数据，<103> 织构的颜色范围较宽，建议使用 80-100
    COLOR_TOLERANCE = 80 
    
    # ==========================
    print(f"\n目标晶向：{TARGET_INDICES}")
    print(f" 容忍度：{COLOR_TOLERANCE}\n")
    
    training_data = build_training_dataset(ROOT_DATA_DIR, TARGET_INDICES, COLOR_TOLERANCE)
    training_data.to_csv("Optimized_Training_Data.csv", index=False)
    print("\n数据处理完成！文件已保存为 Optimized_Training_Data.csv")