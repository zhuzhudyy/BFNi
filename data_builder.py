import os
import pandas as pd
import numpy as np
import io
import warnings
import re
import glob
warnings.filterwarnings('ignore')

# 从 contextual_bo_model 导入共享常量
from contextual_bo_model import ALL_TARGET_ORIENTATIONS

def find_data_files(folder_path, prefix):
    """
    在文件夹中查找所有带指定前缀的数据文件
    支持格式: prefix.csv, prefix1.csv, prefix2.csv, prefix.xls, prefix.xlsx 等
    
    Args:
        folder_path: 文件夹路径
        prefix: 前缀 (如 'pre', 'done')
    Returns:
        list: 匹配的文件路径列表，按文件名排序
    """
    if not os.path.exists(folder_path):
        return []
    
    matched_files = []
    
    # 支持的文件扩展名
    extensions = ['.csv', '.xls', '.xlsx']
    
    for ext in extensions:
        # 查找 prefix{数字}.ext 格式的文件
        pattern = os.path.join(folder_path, f'{prefix}*{ext}')
        files = glob.glob(pattern)
        
        # 过滤掉不符合命名规则的文件（如 prefix_old.csv）
        for f in files:
            basename = os.path.basename(f)
            # 匹配: pre.csv, pre1.csv, pre2.csv, done.csv, done1.csv 等
            if re.match(rf'^{prefix}\d*\{ext}$', basename, re.IGNORECASE):
                matched_files.append(f)
    
    # 按文件名排序（确保 pre1, pre2, pre10 的顺序正确）
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', os.path.basename(s))]
    
    matched_files.sort(key=natural_sort_key)
    return matched_files


def merge_multiple_files(file_list, file_type='pre'):
    """
    合并多个 pre 或 done 文件的数据
    
    Args:
        file_list: 文件路径列表
        file_type: 'pre' 或 'done'
    Returns:
        DataFrame: 合并后的数据
    """
    if not file_list:
        return None
    
    all_data = []
    print(f"\n  发现 {len(file_list)} 个 {file_type} 文件:")
    
    for i, file_path in enumerate(file_list, 1):
        try:
            # 根据扩展名选择读取方式
            if file_path.endswith('.csv'):
                # 使用文本切割法处理 AZtec 表头
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                header_idx = 0
                for j, line in enumerate(lines):
                    if line.count(',') > 5 and ('Phase' in line or 'X' in line or 'Euler' in line or 'IPF' in line):
                        header_idx = j
                        break
                
                clean_csv_data = "".join(lines[header_idx:])
                df = pd.read_csv(io.StringIO(clean_csv_data), low_memory=False)
            else:
                # Excel 文件
                df = pd.read_excel(file_path)
            
            all_data.append(df)
            print(f"    [{i}] {os.path.basename(file_path)}: {len(df)} 行")
            
        except Exception as e:
            print(f"    [{i}] {os.path.basename(file_path)}: 读取失败 ({e})")
    
    if not all_data:
        return None
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"  合并后总计: {len(merged_df)} 行")
    
    return merged_df


def extract_features_from_merged_data(merged_df, target_rgbs=None, tolerance=50, prefix="Pre_"):
    """
    从合并后的 DataFrame 提取特征（替代 extract_macro_rgb_features）
    
    Args:
        merged_df: 合并后的 DataFrame
        target_rgbs: 目标晶向RGB列表
        tolerance: 颜色容差
        prefix: 特征前缀
    Returns:
        dict: 特征字典
    """
    if merged_df is None or len(merged_df) == 0:
        return {}
    
    features = {}
    df = merged_df
    
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

    # --- 2. 连续变量处理 (GND/Half quadratic) ---
    hq_cols = [c for c in df.columns if 'Half quadratic' in c or 'HQ' in c or 'GND' in c]
    if hq_cols:
        hq_col = hq_cols[0]
        gnd_data = pd.to_numeric(df[hq_col], errors='coerce').dropna()
        
        if len(gnd_data) > 0:
            # 基础统计量
            features[f'{prefix}GND_Mean'] = gnd_data.mean()
            features[f'{prefix}GND_Std'] = gnd_data.std()
            
            # 分位数特征 (捕捉分布形状)
            features[f'{prefix}GND_Q25'] = gnd_data.quantile(0.25)
            features[f'{prefix}GND_Q50'] = gnd_data.quantile(0.50)
            features[f'{prefix}GND_Q75'] = gnd_data.quantile(0.75)
            features[f'{prefix}GND_Q90'] = gnd_data.quantile(0.90)
            features[f'{prefix}GND_Q95'] = gnd_data.quantile(0.95)
            features[f'{prefix}GND_Q99'] = gnd_data.quantile(0.99)
            
            # 分布形状特征
            features[f'{prefix}GND_IQR'] = gnd_data.quantile(0.75) - gnd_data.quantile(0.25)
            features[f'{prefix}GND_Peak'] = gnd_data.mode().iloc[0] if len(gnd_data.mode()) > 0 else gnd_data.median()
            features[f'{prefix}GND_Skewness'] = gnd_data.skew()
            features[f'{prefix}GND_Kurtosis'] = gnd_data.kurtosis()
            
            # 变异系数 (相对离散程度)
            mean_val = gnd_data.mean()
            if mean_val != 0:
                features[f'{prefix}GND_CV'] = gnd_data.std() / abs(mean_val)
            
            # 高低值比例 (物理意义明确)
            q75, q25 = gnd_data.quantile(0.75), gnd_data.quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                high_threshold = q75 + 1.5 * iqr
                low_threshold = q25 - 1.5 * iqr
                features[f'{prefix}GND_HighRatio'] = (gnd_data > high_threshold).mean()
                features[f'{prefix}GND_LowRatio'] = (gnd_data < low_threshold).mean()
    
    return features


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

    # --- 2. 连续变量处理 (GND/Half quadratic) ---
    hq_cols = [c for c in df.columns if 'Half quadratic' in c or 'HQ' in c or 'GND' in c]
    if hq_cols:
        hq_col = hq_cols[0]
        gnd_data = pd.to_numeric(df[hq_col], errors='coerce').dropna()
        
        if len(gnd_data) > 0:
            # 基础统计量
            features[f'{prefix}GND_Mean'] = gnd_data.mean()
            features[f'{prefix}GND_Std'] = gnd_data.std()
            
            # 分位数特征 (捕捉分布形状)
            features[f'{prefix}GND_Q25'] = gnd_data.quantile(0.25)  # 第一四分位数
            features[f'{prefix}GND_Q50'] = gnd_data.quantile(0.50)  # 中位数
            features[f'{prefix}GND_Q75'] = gnd_data.quantile(0.75)  # 第三四分位数
            features[f'{prefix}GND_Q90'] = gnd_data.quantile(0.90)  # 90%分位数 (尾部)
            features[f'{prefix}GND_Q95'] = gnd_data.quantile(0.95)  # 95%分位数 (极端值)
            features[f'{prefix}GND_Q99'] = gnd_data.quantile(0.99)  # 99%分位数 (绝对峰值)
            
            # 四分位距 (IQR) - 衡量数据离散度，对异常值稳健
            features[f'{prefix}GND_IQR'] = gnd_data.quantile(0.75) - gnd_data.quantile(0.25)
            
            # 绝对峰值 (99.9%分位数 - 代表最大GND区域)
            features[f'{prefix}GND_Peak'] = gnd_data.quantile(0.999)
            
            # 偏度 (Skewness) - 衡量分布不对称性
            # 正偏: 峰值在左，长尾在右 (多数区域低GND，少数区域高GND)
            # 负偏: 峰值在右，长尾在左 (多数区域高GND，少数区域低GND)
            if len(gnd_data) > 3 and gnd_data.std() > 0:
                features[f'{prefix}GND_Skewness'] = gnd_data.skew()
            else:
                features[f'{prefix}GND_Skewness'] = 0.0
            
            # 峰度 (Kurtosis) - 衡量尾部厚度 (极端值多少)
            if len(gnd_data) > 4 and gnd_data.std() > 0:
                features[f'{prefix}GND_Kurtosis'] = gnd_data.kurtosis()
            else:
                features[f'{prefix}GND_Kurtosis'] = 0.0
            
            # 变异系数 (CV) - 标准化离散度
            if gnd_data.mean() > 0:
                features[f'{prefix}GND_CV'] = gnd_data.std() / gnd_data.mean()
            else:
                features[f'{prefix}GND_CV'] = 0.0
            
            # 高GND区域比例 (超过均值+2倍标准差的像素比例)
            high_gnd_threshold = gnd_data.mean() + 2 * gnd_data.std()
            features[f'{prefix}GND_HighRatio'] = (gnd_data > high_gnd_threshold).mean()
            
            # 低GND区域比例 (低于25%分位数的像素比例 - 代表再结晶/低缺陷区域)
            features[f'{prefix}GND_LowRatio'] = (gnd_data < gnd_data.quantile(0.25)).mean()

    return features

# 目标晶向组合方案（使用共享的 ALL_TARGET_ORIENTATIONS 替代原始重复定义）
TARGET_SCHEMES = {
    1: {
        'name': '<103> 型织构（橙色系）',
        'indices': [(1, 0, 3), (1, 0, 2), (3, 0, 1)]
    },
    2: {
        'name': '<114> 型织构（粉紫色系）',
        'indices': [(1, 1, 4), (1, 1, 5), (1, 0, 5)]
    },
    3: {
        'name': '<124> 型织构（混合色）',
        'indices': [(1, 2, 4), (1, 2, 5), (2, 1, 4)]
    },
    4: {
        'name': '自定义组合',
        'indices': [(1, 0, 3), (1, 1, 4), (1, 2, 4)]
    }
}


def build_training_dataset_multi_target(root_dir, target_schemes, color_tolerance=20, all_targets=None):
    """
    构建多目标训练数据集
    
    为每个实验样本计算所有目标晶向方案的产率，将1个实验扩展为多行数据
    
    Args:
        root_dir: 数据根目录
        target_schemes: 目标晶向方案字典 {scheme_id: {'name': str, 'indices': [(h,k,l), ...]}}
        color_tolerance: 颜色容差
        all_targets: 所有可能的目标晶向列表（用于One-Hot编码）
    """
    if all_targets is None:
        all_targets = ALL_TARGET_ORIENTATIONS
    
    print(f"将为每个实验计算 {len(target_schemes)} 种目标晶向方案的产率")
    for scheme_id, scheme_info in target_schemes.items():
        rgbs = [hkl_to_aztec_rgb(h, k, l) for h, k, l in scheme_info['indices']]
        print(f"  方案 {scheme_id}: {scheme_info['name']}")
        for (h, k, l), rgb in zip(scheme_info['indices'], rgbs):
            print(f"    <{h}{k}{l}> -> RGB: {rgb}")
    print()
        
    print(f"\n开始扫描实验数据库：{root_dir}")
    all_experiments = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path): continue
        
        # 【新增】查找所有 pre 和 done 文件（支持多文件合并）
        pre_files = find_data_files(folder_path, 'pre')
        done_files = find_data_files(folder_path, 'done')
        
        # 查找 condition 文件（只有一组）
        cond_path = os.path.join(folder_path, 'condition.xls')
        if not os.path.exists(cond_path):
            cond_path = os.path.join(folder_path, 'condition.csv')
        if not os.path.exists(cond_path):
            cond_path = os.path.join(folder_path, 'condition.xlsx')

        if pre_files and done_files and os.path.exists(cond_path):
            try:
                print(f"\n处理批次：{folder_name}")
                
                # 【新增】合并多个 pre 文件并提取特征
                pre_merged = merge_multiple_files(pre_files, file_type='pre')
                pre_features = extract_features_from_merged_data(pre_merged, target_rgbs=None, prefix="Pre_")
                
                # 【新增】合并多个 done 文件（用于后续产率计算）
                done_merged = merge_multiple_files(done_files, file_type='done')
                
                # 读取工艺条件
                if cond_path.endswith('.xls') or cond_path.endswith('.xlsx'):
                    cond_df = pd.read_excel(cond_path)
                else:
                    try:
                        cond_df = pd.read_csv(cond_path, encoding='utf-8')
                    except:
                        cond_df = pd.read_csv(cond_path, encoding='gbk')
                
                process_params = {}
                if 'temprature(℃)' in cond_df.columns:
                    process_params['Process_Temp'] = cond_df['temprature(℃)'].iloc[0]
                elif 'Temp' in cond_df.columns:
                    process_params['Process_Temp'] = cond_df['Temp'].iloc[0]
                    
                if 'time(h)' in cond_df.columns:
                    process_params['Process_Time'] = cond_df['time(h)'].iloc[0]
                elif 'Time' in cond_df.columns:
                    process_params['Process_Time'] = cond_df['Time'].iloc[0]
                    
                if 'H2' in cond_df.columns: process_params['Process_H2'] = cond_df['H2'].iloc[0]
                if 'Ar' in cond_df.columns: process_params['Process_Ar'] = cond_df['Ar'].iloc[0]
                
                # 为每个目标晶向方案计算产率
                for scheme_id, scheme_info in target_schemes.items():
                    exp_data = pre_features.copy()
                    exp_data.update(process_params)
                    
                    # 设置目标晶向One-Hot编码
                    scheme_indices = scheme_info['indices']
                    for hkl in all_targets:
                        key = f"Target_{hkl[0]}{hkl[1]}{hkl[2]}"
                        exp_data[key] = 1.0 if hkl in scheme_indices else 0.0
                    
                    # 【修改】使用合并后的 done 数据计算产率
                    target_rgbs = [hkl_to_aztec_rgb(h, k, l) for h, k, l in scheme_indices]
                    done_features = extract_features_from_merged_data(done_merged, target_rgbs=target_rgbs, tolerance=color_tolerance)
                    exp_data['TARGET_Yield'] = done_features.get('TARGET_Yield', 0.0)
                    
                    exp_data['Sample_ID'] = folder_name
                    exp_data['Target_Scheme'] = scheme_id
                    all_experiments.append(exp_data)
                
                print(f"  ✓ 成功提取：{folder_name} (扩展为 {len(target_schemes)} 行数据)")
                            
            except Exception as e:
                print(f"  ✗ 解析文件夹 {folder_name} 时出错：{e}")
                import traceback
                traceback.print_exc()

    final_df = pd.DataFrame(all_experiments).fillna(0)
    return final_df

if __name__ == "__main__":
    # ==========================
    # 核心参数设置区域
    # ==========================
    ROOT_DATA_DIR = r"D:\毕业设计\织构数据\数据总结" 
    
    # 色差容忍度 (0=必须纯色，50=稍微带渐变，80+=非常宽松)
    COLOR_TOLERANCE = 80 
    
    # 选择要计算的目标晶向方案（可以选1个或多个）
    # 例如：只计算方案1 -> {1: TARGET_SCHEMES[1]}
    # 例如：计算所有方案 -> TARGET_SCHEMES
    SELECTED_SCHEMES = TARGET_SCHEMES  # 默认计算所有4个方案
    
    print("=" * 60)
    print("多目标贝叶斯优化数据构建器")
    print("=" * 60)
    print(f"将为每个实验样本计算 {len(SELECTED_SCHEMES)} 种目标晶向方案的产率")
    print(f"原始样本数 × {len(SELECTED_SCHEMES)} = 最终训练数据行数")
    print(f"色差容忍度: {COLOR_TOLERANCE}")
    print("=" * 60 + "\n")
    
    # 构建多目标训练数据集
    training_data = build_training_dataset_multi_target(
        ROOT_DATA_DIR, 
        SELECTED_SCHEMES, 
        color_tolerance=COLOR_TOLERANCE
    )
    
    # 保存数据
    training_data.to_csv("Optimized_Training_Data.csv", index=False)
    
    print("\n" + "=" * 60)
    print("数据处理完成！")
    print(f"原始实验批次: {training_data['Sample_ID'].nunique()}")
    print(f"目标晶向方案数: {len(SELECTED_SCHEMES)}")
    print(f"最终训练数据行数: {len(training_data)}")
    print(f"特征维度: {len([c for c in training_data.columns if c.startswith('Pre_')])} Pre + " +
          f"{len([c for c in training_data.columns if c.startswith('Target_')])} Target + " +
          f"4 Process")
    print(f"文件已保存: Optimized_Training_Data.csv")
    print("=" * 60)