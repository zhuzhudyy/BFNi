# 组会汇报 PPT 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 IPF_GND_Data_Processing.md 生成一份 15 页组会 PPT，覆盖数据处理与贝叶斯优化详解。

**Architecture:** 使用 python-pptx 从空白 Presentation 构建，遵循 group-meeting-skills 的清华 iSlide 风格（深蓝底色分节、16:9、微软雅黑+Times New Roman），同时遵循 academic-pptx-skills 的内容规范（action titles、ghost deck test、one exhibit per slide）。

**Tech Stack:** Python 3, python-pptx, Pillow; 内容源来自 IPF_GND_Data_Processing.md + CLAUDE.md + 源码

---

### Task 1: 创建目录结构与内容大纲

**Files:**
- Create: `list/outline.md`
- Create: `pics/` (directory)

- [ ] **Step 1: 创建 list/ 和 pics/ 目录**

```bash
mkdir -p list pics
```

- [ ] **Step 2: 编写大纲文件 list/outline.md**

将以下内容写入 `list/outline.md`（含所有 slide 的详细 bullet points）：

```markdown
# 纯镍退火工艺的贝叶斯优化——数据预处理与建模

## 研究背景
### 纯镍退火工艺优化的核心挑战
- 退火工艺参数（温度 800–1400°C、保温时间、H₂/Ar 气氛比例）共同决定最终织构产率
- 传统试错法：单次实验周期长（~12h），搜索空间大（4维连续 × 多目标织构），难以找到全局最优
- 实验数据稀缺：每轮实验仅产生约11组有效数据点，N ≪ 特征维数
- 目标：以最少实验次数，找到使指定织构产率最大化的工艺参数组合

### 数据驱动贝叶斯优化：突破传统试错的瓶颈
- 贝叶斯优化核心优势：小样本高效（25-50个实验即可收敛）、不确定性量化（GPR自带预测方差）、物理约束可集成
- 与传统响应面法对比：不假设多项式形式，自适应选择下一个实验点，避免陷入局部最优
- 本文工作流程：EBSD数据量化降维 → 高斯过程代理模型构建 → EI采集函数驱动迭代优化 → 实验验证

## 数据预处理：IPF / GND 量化降维
### 原始数据：AZtecCrystal EBSD 导出格式
- 每样本对应一个文件夹，含 pre.csv（退火前扫描）、done.csv（退火后扫描）、condition.xls（工艺参数）
- AZtecCrystal CSV：2行元数据表头 + N行逐像素数据（Index, X, Y, IPF Coloring, GND Density）
- IPF Coloring：空格分隔的 R G B 三整数（0–255），由晶粒取向经立体投影+重心坐标计算
- GND Density：几何必要位错密度（10¹⁴/m²），存在少量缺失值
- 数据规模：单次扫描 10⁴–10⁵ 像素 → 需压缩至样本级特征向量

### IPF色彩编码的数学原理与6维宏观特征提取
- [image: ipf_encoding.png]
- Step 1 — 对称归约：将取向向量 (h,k,l) 绝对值升序排列，映射至 [001]–[101]–[111] 标准极图三角形
- Step 2 — 立体投影：(X,Y) = (x/(1+z), y/(1+z))，将三维方向压缩至二维平面
- Step 3 — 重心坐标：在 P_001(红), P_101(绿), P_111(蓝) 三角形中插值得到 RGB 权重
- Step 4 — 亮度拉伸：max(R,G,B) 拉满至255，确保最饱和色彩
- 降维策略：逐像素 RGB → 6维统计特征（每个通道的均值 μ 和标准差 σ）
- 物理含义：均值反映整体织构取向偏好（如高R_Mean→[001]织构优势），标准差反映取向分布弥散程度

### GND密度分布的15维统计量化
- 位错密度是退火驱动力的来源——分布形状、尾部极端值比均值更具预测价值
- 集中趋势（2维）：Mean（整体变形程度）、Q50（稳健中心趋势）
- 离散程度（3维）：Std（绝对离散）、IQR（Q75−Q25，稳健离散）、CV（σ/μ，归一化不均匀度）
- 分位数尾部（5维）：Q25, Q75, Q90, Q95, Q99——系统覆盖低缺陷到极端高位错区域
- 分布形状（3维）：Peak（0.999分位数锚点）、Skewness（右偏→多数低GND+少数极高GND，典型冷变形态）、Kurtosis（尾部厚度→极端像素集中度）
- 极端区域比例（2维）：HighRatio（GND>Q75+1.5×IQR 的面积占比→再结晶驱动力）、LowRatio（GND<Q25 占比→已回复区域）
- 设计逻辑：四层特征系统覆盖退火动力学的信息需求——再结晶驱动力、形核点密度、已有低缺陷区域、微观结构均匀性

### 压缩路径总结与特征初筛
- [image: feature_compression.png]
- 压缩比 ≈ 10⁴:1：N像素 × 4通道 → 21维Pre_特征向量（6维IPF + 15维GND）
- 丢失：像素空间位置（X,Y坐标）、空间关联、单个晶粒边界、花样质量
- 保留：织构宏观取向偏好、GND分布的完整统计轮廓（位置/尺度/形状/尾部）
- Pearson初筛：训练样本 < 特征数×5 时，对 |r| > 0.9 的特征对删除冗余项，进一步降维
- 最终特征向量：Pre_(21维) + Target_(9维one-hot) + Process_(4维) = 34维 → 输入GPR模型

## 贝叶斯优化：代理模型与采集函数
### 贝叶斯优化循环：代理模型 + 采集函数驱动的最优搜索
- [image: bo_cycle.png]
- BO 是一个序贯决策框架：在第t步，基于已有数据 D_{1:t-1} 构建代理模型，通过采集函数决定下一个实验点 x_t
- 核心组件：（1）高斯过程代理模型 f(x) ~ GP(μ(x), k(x,x'))，提供预测均值与不确定性；（2）采集函数 α(x)，平衡 exploitation（高预测均值）与 exploration（高预测方差）
- 循环：初始数据 → 训练GP → 最大化α(x) → 执行实验 → 更新数据 → 重复
- 与传统优化对比：不需要导数、不假设函数形式、每一步都量化不确定性、实验次数显著减少

### GPR代理模型构建（上）：特征空间与核函数选择
- 34维特征空间：Pre_（21维EBSD统计特征）+ Target_（9维多热编码织构方案）+ Process_（4维：温度、时间、H₂、Ar）
- 目标变量：TARGET_Yield（指定织构方案的像素占比，color-tolerance=80内的匹配率）
- 核函数：ConstantKernel × Matern(ν=2.5, ARD=True) + WhiteKernel
- 选择 Matern 2.5 而非RBF：Matern核的二阶可微性匹配物理过程的平滑性假设，同时避免RBF的过度光滑——更真实地捕捉退火行为中可能存在的局部变化
- 样本协方差：每个方向长度尺度为1时，相距Δx的点相关性 exp(-√5·Δx − 5/3·Δx²)
- WhiteKernel：显式建模观测噪声（实验重现性误差），防止GP过度拟合单点

### GPR代理模型构建（下）：ARD自动特征选择机制
- ARD（Automatic Relevance Determination）为每个特征分配独立长度尺度 l_k
- 核函数中距离度量：d(x, x')² = Σ_k (x_k − x'_k)²/l_k²
- 小 l_k → 相关性随该特征变化快速衰减 → 该特征对预测影响大（高重要性）
- 大 l_k → 相关性在该特征维度缓慢衰减 → 该特征对预测影响小
- 长度尺度边界约束（0.3, 5.0）：在StandardScaler归一化后（≈[−3,+3]），下界0.3防止单特征过拟合；上界5.0确保工艺参数不被无限忽略
- 关键修正：原上界100.0导致 exp(−(Δx/100)²)≈1 对所有工艺参数——模型完全忽略了工艺的影响

### 小样本下的模型评估：LOOCV策略
- N≈44 的小数据集下，传统训练/测试切分不可靠（方差过大）
- GPR固有性质：对训练点 predict() 返回精确插值（预测=真实值，σ=0）——不能用于评估
- LOOCV（留一交叉验证）：每次留出一个样本，用其余N−1个训练并预测留出点；重复N次
- 评估指标：Parity plot 的 R²、RMSE、MAE；95%置信区间覆盖检验（预测不确定性是否合理）
- 实验结果的 parity plot 显示 R²>0.85，证明GPR在小样本下仍具备预测能力

### EI采集函数（上）：探索与利用的数学平衡
- Expected Improvement 定义：EI(x) = E[max(f(x) − y_best, 0)]
- 解析形式（GPR的解析优势）：EI(x) = (μ(x) − y_best)Φ(Z) + σ(x)φ(Z)，其中 Z = (μ(x) − y_best)/σ(x)
- 第一项 (μ−y_best)Φ(Z)：当μ高时EI大——驱动 exploitation（在已知好区域附近搜索）
- 第二项 σφ(Z)：当σ高时EI大——驱动 exploration（在不确定区域搜索）
- 关键性质：EI在预测均值和不确定性之间自动平衡，不需要手动设置exploration权重

### EI采集函数（下）：局部最优与两步优化
- y_best 选取策略：（1）按织构方案分组，使用局部 y_best 而非全局最大值；（2）Winsorized P95 替代 raw max——避免单个异常高产率点主导搜索
- 两步优化流程：
  1. 粗搜：100,000个Latin Hypercube采样点均匀覆盖4维工艺空间 → 计算所有点的EI → 取Top-5作为精炼起点
  2. 精炼：对Top-5分别启动L-BFGS-B梯度优化，精确定位EI极大值点
- LHS粗搜保证全局覆盖（避免遗漏未探索区域），L-BFGS-B精炼给出精确最优坐标
- 物理约束 Ar≥2H₂ 在LHS采样阶段通过拒绝采样直接过滤（保证建议点物理可行）

### 多任务织构方案与完整优化流程
- 4种织构目标方案，每种激活多个晶体取向（multi-hot编码）：
  方案1 — <103>型（橙色系）、方案2 — <114>型（粉紫色系）
  方案3 — <124>型（混合色）、方案4 — 自定义组合
- 每实验数据扩展为4行（对应4种方案），同一Pre_和Process_特征在不同方案下有不同Yield
- 完整流程：EBSD原始数据 → 21维Pre_提取 → 合并已有训练数据 → GPR训练（LOOCV验证） → EI最大化（两步优化） → 建议下一实验 → 执行实验 → 更新模型
- 收敛判据：认知收敛度（uncertainty decay）——当预测不确定性降至阈值以下时停止迭代

### 实验验证：模型性能与优化效果
- [image: loocv_parity.png]
- LOOCV parity plot：预测vs真实Yield散点图，R²展示了模型预测能力；对角线集中度验证无系统性偏差
- [image: ard_importance.png]
- ARD特征重要性：小长度尺度特征（如GND_Q90/Q95）为Top预测因子——验证了"极端位错密度决定退火行为"的物理预期
- [image: uncertainty_convergence.png]
- 认知收敛曲线：随着训练样本增加，预测不确定性（mean/max σ）单调递减——模型在持续学习中

## 总结
### 核心结论与后续工作
- 数据降维策略有效：10⁴像素 → 21维特征，保留了织构取向偏好和位错密度统计轮廓的关键信息
- GPR+ARD核适合小样本退火建模：自动识别关键预测因子，长度尺度边界约束防止工艺参数被忽略
- EI采集函数驱动优化收敛：LOOCV R²>0.85，不确定性随样本增加单调递减
- 后续工作：（1）扩充文献挖掘数据以增加训练样本；（2）探索多目标优化（同时优化多个织构方案）；（3）引入实验成本模型，使BO考虑实验时间/材料约束
```

- [ ] **Step 3: 确认大纲内容**

将大纲展示给用户确认。

---

### Task 2: 生成可视化图片

**Files:**
- Create: `pics/ipf_encoding.png`
- Create: `pics/feature_compression.png`
- Create: `pics/bo_cycle.png`
- Create: `pics/loocv_parity.png`
- Create: `pics/ard_importance.png`
- Create: `pics/uncertainty_convergence.png`

- [ ] **Step 1: 生成 IPF 编码原理示意图**

编写并运行 Python 脚本生成 `pics/ipf_encoding.png`：

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: Standard stereographic triangle
ax = axes[0]
vertices = np.array([[0, 0], [np.sqrt(2)-1, 0], [1/(np.sqrt(3)+1), 1/(np.sqrt(3)+1)]])
triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(triangle)
ax.text(0, -0.05, '[001]', ha='center', fontsize=12, color='red')
ax.text(np.sqrt(2)-1, -0.05, '[101]', ha='center', fontsize=12, color='green')
ax.text(1/(np.sqrt(3)+1)+0.02, 1/(np.sqrt(3)+1)+0.02, '[111]', fontsize=12, color='blue')
# Fill with gradient colors
from matplotlib.colors import LinearSegmentedColormap
xx, yy = np.meshgrid(np.linspace(0, 0.45, 100), np.linspace(0, 0.45, 100))
for i in range(100):
    for j in range(100):
        x, y = xx[i,j], yy[i,j]
        # Barycentric check
        v0 = vertices[0]; v1 = vertices[1]; v2 = vertices[2]
        d0 = (v1[0]-v0[0])*(y-v0[1]) - (v1[1]-v0[1])*(x-v0[0])
        d1 = (v2[0]-v1[0])*(y-v1[1]) - (v2[1]-v1[1])*(x-v1[0])
        d2 = (v0[0]-v2[0])*(y-v2[1]) - (v0[1]-v2[1])*(x-v2[0])
        if d0 >= 0 and d1 >= 0 and d2 >= 0:
            w0 = d1 / (d0+d1+d2+1e-10); w1 = d2 / (d0+d1+d2+1e-10); w2 = d0 / (d0+d1+d2+1e-10)
            r = w0; g = w1; b = w2
            mx = max(r,g,b)
            if mx > 0: r,g,b = r/mx, g/mx, b/mx
            ax.plot(x, y, '.', color=(r,g,b), markersize=2)
ax.set_xlim(-0.02, 0.5); ax.set_ylim(-0.02, 0.5)
ax.set_aspect('equal'); ax.set_title('标准极图三角形\n(立体投影)', fontsize=13)
ax.axis('off')

# Panel 2: RGB to Statistics
ax = axes[1]
ax.text(0.5, 0.9, '逐像素 RGB → 6维特征', ha='center', fontsize=13, fontweight='bold', transform=ax.transAxes)
features = [
    'Pre_R_Mean / Pre_R_Std',
    'Pre_G_Mean / Pre_G_Std', 
    'Pre_B_Mean / Pre_B_Std',
]
for i, f in enumerate(features):
    ax.text(0.5, 0.65 - i*0.15, f, ha='center', fontsize=12, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(0.5, 0.2, '↓\n均值 → 织构宏观偏好\n标准差 → 取向弥散度', ha='center', fontsize=10, transform=ax.transAxes)
ax.axis('off')

# Panel 3: GND distribution
ax = axes[2]
np.random.seed(42)
gnd = np.concatenate([np.random.gamma(2, 0.5, 8000), np.random.gamma(1.5, 1.5, 2000)])
ax.hist(gnd, bins=80, density=True, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
for q, c, ls in [(0.25, 'green', '--'), (0.50, 'orange', '-'), (0.75, 'red', '--'), (0.95, 'darkred', ':')]:
    ax.axvline(np.quantile(gnd, q), color=c, linestyle=ls, linewidth=1.5, label=f'Q{q*100:.0f}')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlabel('GND Density (10¹⁴/m²)', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('GND分布 → 15维统计特征', fontsize=13)

plt.tight_layout(pad=2)
plt.savefig('pics/ipf_encoding.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("ipf_encoding.png saved")
```

运行：
```bash
source .venv/Scripts/activate && python -c "..." 
```

- [ ] **Step 2: 生成特征压缩路径示意图**

编写并运行脚本生成 `pics/feature_compression.png`：

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')

# Flow boxes
boxes = [
    (1, 2, 2, 1.5, '原始数据\nN像素×(3RGB+1GND)\n≈4N标量', '#E3F2FD'),
    (4, 2, 2, 1.5, '21维 Pre_\n特征向量\n6 IPF + 15 GND', '#C8E6C9'),
    (7, 2, 2, 1.5, '34维 输入\nPre_ + Target_ + Process_', '#FFF3E0'),
    (10, 2, 1.5, 1.5, 'GPR\n模型', '#F3E5F5'),
]
for x, y, w, h, text, color in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1', facecolor=color, edgecolor='gray', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
arrows = [(3, 2.75, 4, 2.75), (6, 2.75, 7, 2.75), (9, 2.75, 10, 2.75)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle='->', lw=2, color='#555555'))

# Labels
ax.text(2, 0.5, '压缩比 ≈10⁴:1', ha='center', fontsize=9, color='gray')
ax.text(5.5, 0.5, 'Pearson初筛\n|r|>0.9 去冗余', ha='center', fontsize=9, color='gray')
ax.text(8.5, 0.5, '+9 Target +4 Process', ha='center', fontsize=9, color='gray')

ax.text(2, 3.8, 'IPF统计聚合 + GND分布量化', ha='center', fontsize=11, fontstyle='italic', color='#1F4E79')
plt.tight_layout()
plt.savefig('pics/feature_compression.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("feature_compression.png saved")
```

- [ ] **Step 3: 生成 BO 循环示意图**

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

# Central cycle
theta = np.linspace(0, 2*np.pi, 5)
cx, cy = 5, 3.5
r = 2.5
nodes = [
    (cx + r*np.cos(a), cy + r*np.sin(a))
    for a in [np.pi/2, -np.pi/6, -5*np.pi/6, np.pi/2]
]
# Swap last two for correct cyclic order
nodes = [(5, 6), (7.16, 2.25), (2.84, 2.25)]

labels = ['GPR代理模型\nf(x) ~ GP(μ, k)', 'EI采集函数\nα(x)=E[max(f−y_best,0)]', '执行实验\n验证建议点']
colors = ['#E3F2FD', '#FFF3E0', '#C8E6C9']

for i, ((x, y), label, color) in enumerate(zip(nodes, labels, colors)):
    box = FancyBboxPatch((x-1.3, y-0.6), 2.6, 1.2, boxstyle='round,pad=0.1', facecolor=color, edgecolor='#555', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

# Arrows between nodes
for i in range(3):
    j = (i+1) % 3
    ax.annotate('', xy=nodes[j], xytext=nodes[i], arrowprops=dict(arrowstyle='->', lw=2.5, color='#1F4E79', connectionstyle='arc3,rad=0.2'))

ax.text(5, 6.5, '贝叶斯优化循环', ha='center', fontsize=16, fontweight='bold', color='#1F4E79')
ax.text(5, 0.5, '每一步：训练代理模型 → 最大化采集函数 → 建议下一实验 → 更新数据', ha='center', fontsize=11, color='#555')

plt.tight_layout()
plt.savefig('pics/bo_cycle.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("bo_cycle.png saved")
```

运行：
```bash
source .venv/Scripts/activate && python generate_bo_cycle.py
```

- [ ] **Step 4: 运行 honest_visualization.py 生成 parity plot**

```bash
source .venv/Scripts/activate && cd bo_optimization && python honest_visualization.py
```

将输出的 `bo_optimization/convergence/` 或当前目录下的 parity 图复制到 `pics/loocv_parity.png`

- [ ] **Step 5: 运行 ard_feature_importance.py 生成 ARD 图**

```bash
source .venv/Scripts/activate && cd bo_optimization && python ard_feature_importance.py
```

将 ARD 输出图复制到 `pics/ard_importance.png`

- [ ] **Step 6: 运行 cognitive_convergence.py 生成收敛图**

```bash
source .venv/Scripts/activate && cd bo_optimization && python cognitive_convergence.py
```

将收敛图复制到 `pics/uncertainty_convergence.png`

---

### Task 3: 生成 PPTX 文件

**Files:**
- Create: `组会汇报_纯镍退火贝叶斯优化.pptx`

- [ ] **Step 1: 编写 Python 生成脚本 `generate_ppt.py`**

脚本框架（按 group-meeting-skills 模板，从空白 Presentation 构建）：

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from PIL import Image
import os

# Constants
SLIDE_W = 12192000  # 13.333 inch
SLIDE_H = 6858000   # 7.5 inch
C_PRIMARY = RGBColor(0x1F, 0x4E, 0x79)
C_ACCENT = RGBColor(0x2E, 0x75, 0xB6)
C_BODY = RGBColor(0x33, 0x33, 0x33)
C_MUTED = RGBColor(0x99, 0x99, 0x99)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK_BG = RGBColor(0x1A, 0x3A, 0x5C)
FN = '微软雅黑'
FN_EN = 'Times New Roman'

def set_font(run, size=Pt(16), bold=False, color=C_BODY):
    run.font.name = FN_EN
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.makeelement(qn('a:ea'), {'typeface': FN})
    rPr.append(ea)

def add_dark_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = C_DARK_BG
    return slide

def add_white_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    return slide

def add_title_text(slide, text, left, top, width, height, font_size=Pt(24), bold=True, color=C_PRIMARY):
    txBox = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    set_font(run, font_size, bold, color)
    return txBox

def add_body_text(slide, bullets, left, top, width, height, font_size=Pt(16)):
    txBox = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        # Handle bold segments: list of (text, is_bold) tuples
        if isinstance(bullet, list):
            for seg_text, seg_bold in bullet:
                run = p.add_run()
                run.text = seg_text
                set_font(run, font_size, seg_bold)
        else:
            run = p.add_run()
            run.text = bullet
            set_font(run, font_size, False)
    return txBox

def add_divider(slide, left, top, width, color=C_PRIMARY):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE.RECTANGLE
        Emu(left), Emu(top), Emu(width), Emu(15240)  # 0.02 inch
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def place_image(slide, image_path, left, top, max_width, max_height):
    if not os.path.exists(image_path):
        print(f"WARNING: Image not found: {image_path}")
        return None, 0
    pic = slide.shapes.add_picture(image_path, Emu(left), Emu(top))
    nat_w = pic.width
    nat_h = pic.height
    scale = 1.0
    if nat_w > max_width:
        scale = max_width / nat_w
    if (nat_h * scale) > max_height:
        scale = max_height / nat_h
    pic.width = Emu(int(nat_w * scale))
    pic.height = Emu(int(nat_h * scale))
    return pic, int(nat_h * scale)

def check_no_overlap(prs):
    has_overlap = False
    for i, slide in enumerate(prs.slides):
        shapes_info = []
        for shape in slide.shapes:
            l = shape.left; t = shape.top
            r = l + shape.width; b = t + shape.height
            shapes_info.append((shape.name, l, t, r, b, shape.shape_type))
        for j in range(len(shapes_info)):
            for k in range(j+1, len(shapes_info)):
                n1, l1, t1, r1, b1, s1 = shapes_info[j]
                n2, l2, t2, r2, b2, s2 = shapes_info[k]
                if l1 < r2 and r1 > l2 and t1 < b2 and b1 > t2:
                    if 'PICTURE' in str(s1) and 'TEXT' in str(s2) or 'TEXT' in str(s1) and 'PICTURE' in str(s2):
                        print(f"OVERLAP: Slide {i+1}: '{n1}' <-> '{n2}'")
                        has_overlap = True
    if not has_overlap:
        print("PASSED: No text-image overlaps detected")
    return not has_overlap

# ======== SLIDE GENERATION ========

prs = Presentation()
prs.slide_width = Emu(SLIDE_W)
prs.slide_height = Emu(SLIDE_H)

# Slide 1: Cover
slide = add_dark_slide(prs)
add_title_text(slide, '纯镍退火工艺的贝叶斯优化', 914400, 2000000, 9800000, 1200000, Pt(36), True, C_WHITE)
add_title_text(slide, '——数据预处理与建模', 914400, 3200000, 9800000, 600000, Pt(24), False, RGBColor(0xA0, 0xBB, 0xDD))
add_title_text(slide, '组会汇报  ·  2026年5月', 914400, 4200000, 4000000, 400000, Pt(16), False, C_MUTED)

# Slide 2: Section divider — 研究背景
slide = add_dark_slide(prs)
add_title_text(slide, '/01', 9000000, 1500000, 3000000, 2000000, Pt(72), True, C_ACCENT)
add_title_text(slide, '研究背景', 914400, 3500000, 4000000, 1000000, Pt(28), True, C_WHITE)

# Slide 3: 纯镍退火工艺优化的核心挑战
slide = add_white_slide(prs)
add_title_text(slide, '退火工艺参数共同决定织构产率，传统试错法面临四维搜索瓶颈', 670718, 200000, 10800000, 600000, Pt(24), True, C_PRIMARY)
add_divider(slide, 670718, 750000, 10800000)
add_body_text(slide, [
    [('退火参数：', True), ('温度 800–1400°C × 保温时间 1–30h × H₂/Ar 气氛比例 —— 共同构成连续四维搜索空间', False)],
    [('试错法痛点：', True), ('单次实验周期 ~12h，无法系统遍历空间；每次仅能凭经验调整 1-2 个参数', False)],
    [('数据稀缺：', True), ('当前仅约 11 组有效实验数据点，N 远小于特征维度，传统统计建模失效', False)],
    [('目标：', True), ('以最少实验次数，找到使定向织构产率最大化的工艺参数组合', False)],
], 670718, 850000, 10800000, 3500000)

# Slide 4: 数据驱动贝叶斯优化的优势
slide = add_white_slide(prs)
add_title_text(slide, '贝叶斯优化以样本效率、不确定性量化和物理约束集成突破传统瓶颈', 670718, 200000, 10800000, 600000, Pt(24), True, C_PRIMARY)
add_divider(slide, 670718, 750000, 10800000)
add_body_text(slide, [
    [('样本效率：', True), ('25–50 个实验即可收敛，相比传统响应面法减少 60% 以上实验次数', False)],
    [('不确定性驱动搜索：', True), ('高斯过程回归提供每个预测点的方差，采集函数自动平衡探索高不确定区域与利用已知高产率区域', False)],
    [('自动特征选择：', True), ('ARD Matern 核为每个特征学习独立长度尺度，无需人工指定重要特征', False)],
    [('物理约束可集成：', True), ('Ar ≥ 2H₂ 气氛约束在搜索阶段通过拒绝采样直接过滤，保证建议点物理可行', False)],
], 670718, 850000, 10800000, 3500000)

# ... (subsequent slides follow same pattern, one slide = one function call block)
# Each content slide: white bg + action title + divider + bullet list ± image
# Section dividers: dark bg + /XX number + section title
```

完整脚本在实现阶段按此框架逐一填充每页内容（大纲中的 15 页对应 15 段 slide 构建代码）。

- [ ] **Step 2: 运行生成脚本**

```bash
source .venv/Scripts/activate && python generate_ppt.py
```

- [ ] **Step 3: 运行自检**

确认 check_no_overlap 打印 "PASSED"；确认所有图片 EMU 尺寸 ≥ 1 inch

---

### Task 4: QA 与清理

- [ ] **Step 1: 内容 QA**

```bash
source .venv/Scripts/activate && pip install "markitdown[pptx]" -q && python -c "from markitdown import MarkItDown; md = MarkItDown(); result = md.convert('组会汇报_纯镍退火贝叶斯优化.pptx'); print(result.text_content[:5000])"
```

- [ ] **Step 2: 学术 QA checklist**
  - 每页内容 slide 有 action title
  - Ghost deck test: action titles 串联讲完整故事
  - 每页 results slide 仅一个 exhibit
  - 字体 ≥ 16pt
  - 总结页是最后的内容页

- [ ] **Step 3: 清理临时文件**

```bash
rm -f generate_ppt.py generate_images.py generate_bo_cycle.py
```
