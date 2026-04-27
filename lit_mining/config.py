"""
文献挖掘 Agent 配置模块
集中管理检索策略、LLM 参数、特征 schema 映射
"""

# ============================
# 检索配置
# ============================
SEARCH_QUERIES = [
    # 英文检索式 — 覆盖不同材料形态和工艺
    "pure nickel annealing texture EBSD GND recrystallization",
    "nickel cold rolling annealing grain growth orientation IPF",
    "Ni electrodeposited annealing texture evolution EBSD",
    '"nickel" annealing "grain boundary" texture EBSD GND',
    "pure nickel heat treatment microstructure IPF map",
    "nanocrystalline nickel annealing grain growth texture",
    # 中文检索式
    "镍 退火 织构 再结晶 EBSD GND",
    "纯镍 热处理 晶粒取向 微观组织 织构",
    # 相近材料（纯铜、Fe-Ni合金的退火行为可作为先验参考）
    "pure copper annealing texture recrystallization EBSD",
    "Fe-Ni alloy annealing texture GND density",
]

# Semantic Scholar API 配置
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_FIELDS = (
    "title,authors,year,abstract,externalIds,url,"
    "openAccessPdf,journal,publicationTypes,citationCount"
)
MAX_RESULTS_PER_QUERY = 30  # 每个检索式最多取 30 篇
MIN_CITATION_COUNT = 0      # 最低引用数过滤（0=不过滤）

# ============================
# LLM 配置
# ============================
# 可根据实际 API key 和环境切换
LLM_PROVIDER = "openai"  # "anthropic" | "openai" — DeepSeek/OpenAI 兼容

# Anthropic
ANTHROPIC_MODEL_SCREEN = "claude-haiku-4-5-20251001"    # 论文初筛
ANTHROPIC_MODEL_EXTRACT = "claude-sonnet-4-6-20251001"  # 结构化数据提取
ANTHROPIC_MODEL_VISION = "claude-sonnet-4-6-20251001"   # 图表解读（支持图片）

# OpenAI compatible (Paratera 并行科技 — DeepSeek V4 Pro)
OPENAI_MODEL = "DeepSeek-V4-Pro"
OPENAI_BASE_URL = "https://llmapi.paratera.com/v1"

# 提取批次大小（每篇论文单独调用，但可分 chunk）
EXTRACT_MAX_TOKENS = 4096
PDF_MAX_CHARS = 15000  # 超过此长度的论文正文进行滑窗截断

# ============================
# 特征 schema 映射
#   文献字段 → 训练特征列名
# ============================
# 工艺参数直接映射
PROCESS_FIELD_MAP = {
    "temperature_C": "Process_Temp",
    "time_min": "Process_Time_min",       # 中间态，后续转为 hours
    "time_h": "Process_Time",
    "H2_flow_sccm": "Process_H2",
    "Ar_flow_sccm": "Process_Ar",
    "heating_rate_C_per_min": "HeatingRate",  # 新特征
}

# EBSD 特征映射（文献中可能存在的）
EBSD_FIELD_MAP = {
    "gnd_density_mean": "Pre_GND_Mean",       # 单位需统一为 a.u.
    "gnd_density_m2": "Pre_GND_Mean_raw",      # 原始值 (1/m²)，保留后续换算
    "grain_size_um": "Pre_GrainSize",
    "grain_size_nm": "Pre_GrainSize",          # 需转换
    "twin_fraction": "Pre_TwinFraction",
    "recrystallized_fraction": "Pre_RexFraction",
    "dislocation_density": "Pre_GND_Mean",     # 与 GND 相关
    "microhardness_HV": "Pre_Hardness",
}

# 织构 / 产率映射
TEXTURE_FIELD_MAP = {
    "volume_fraction": "TARGET_Yield",          # 通用产率字段
    "{100}<001>": "TARGET_Yield_cube",
    "{110}<001>": "TARGET_Yield_goss",
    "{111}<112>": "TARGET_Yield_brass",
    "{111}<110>": "TARGET_Yield_copper",
    "{112}<111>": "TARGET_Yield_copper_twin",
}

# ============================
# 目标晶向方案定义（与 data_builder.py 保持一致）
# ============================
TARGET_SCHEMES = {
    1: {"name": "<103>型", "indices": [(1,0,3), (1,0,2), (3,0,1)]},
    2: {"name": "<114>型", "indices": [(1,1,4), (1,1,5), (1,0,5)]},
    3: {"name": "<124>型", "indices": [(1,2,4), (1,2,5), (2,1,4)]},
    4: {"name": "自定义",  "indices": [(1,0,3), (1,1,4), (1,2,4)]},
}

# 工艺搜索边界（用于数据合法性校验）
PROCESS_BOUNDS = {
    "Process_Temp": (400.0, 1500.0),    # 文献范围更宽
    "Process_Time": (0.01, 200.0),      # 小时
    "Process_H2": (0.0, 500.0),
    "Process_Ar": (0.0, 2000.0),
}

# ============================
# 输出路径
# ============================
OUTPUT_CSV = "literature_raw_extractions.csv"
FUSED_CSV = "literature_training_data.csv"
CACHE_DIR = "lit_mining/cache"
PAPER_CACHE_JSON = "lit_mining/cache/papers_metadata.json"
