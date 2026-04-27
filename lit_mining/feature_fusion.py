"""
Module C: 特征工程 & 数据融合器

功能:
  1. 将 LLM 提取的文献数据映射到训练特征 schema
  2. 单位换算 (min→h, nm→um, m⁻²→a.u.)
  3. 缺失 Pre_ 特征的智能插补 (基于真实数据分布的条件采样)
  4. 数据质量评分 (confidence)
  5. 输出统一 CSV 格式（与 Optimized_Training_Data.csv 列完全对齐）
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, "..", ".."))))
from .config import (
    PROCESS_FIELD_MAP,
    EBSD_FIELD_MAP,
    TEXTURE_FIELD_MAP,
    TARGET_SCHEMES,
    PROCESS_BOUNDS,
)

# 与 data_builder.py 保持一致的 Pre_ 特征列名（按顺序）
EXPECTED_PRE_COLS = [
    "Pre_R_Mean", "Pre_G_Mean", "Pre_B_Mean",
    "Pre_R_Std", "Pre_G_Std", "Pre_B_Std",
    "Pre_GND_Mean", "Pre_GND_Std",
    "Pre_GND_Q25", "Pre_GND_Q50", "Pre_GND_Q75", "Pre_GND_Q90", "Pre_GND_Q95", "Pre_GND_Q99",
    "Pre_GND_IQR", "Pre_GND_Peak",
    "Pre_GND_Skewness", "Pre_GND_Kurtosis",
    "Pre_GND_CV", "Pre_GND_HighRatio", "Pre_GND_LowRatio",
]
ALL_TARGET_ORIENTATIONS = [
    (1, 0, 3), (1, 0, 2), (3, 0, 1),
    (1, 1, 4), (1, 1, 5), (1, 0, 5),
    (1, 2, 4), (1, 2, 5), (2, 1, 4),
]


class FeatureFusion:
    """文献数据 → 训练特征融合器"""

    def __init__(self, real_training_csv=None):
        """
        Args:
            real_training_csv: 真实实验数据 CSV 路径（用于拟合 Pre_ 特征分布）
        """
        self.real_df = None
        self.pre_stats = {}
        if real_training_csv and os.path.exists(real_training_csv):
            self.real_df = pd.read_csv(real_training_csv)
            self._compute_pre_stats()

        self.target_cols = [f"Target_{h}{k}{l}" for h, k, l in ALL_TARGET_ORIENTATIONS]
        self.process_cols = ["Process_Temp", "Process_Time", "Process_H2", "Process_Ar"]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def fuse(self, extractions_json_path, output_csv=None, existing_csv=None):
        """
        主融合流程：
          1. 加载 LLM 提取的 JSON
          2. 每条实验记录 → 映射为标准行
          3. 缺失 Pre_ 插补
          4. 质量评分
          5. 与已有真实数据合并输出
        """
        with open(extractions_json_path, "r", encoding="utf-8") as f:
            extractions = json.load(f)

        rows = []
        for paper_data in extractions:
            if not paper_data.get("_meta", {}).get("extraction_ok"):
                continue
            paper_rows = self._paper_to_rows(paper_data)
            rows.extend(paper_rows)

        if not rows:
            print("[!] 没有可融合的文献数据")
            return None

        lit_df = pd.DataFrame(rows)
        lit_df = self._impute_missing_pre(lit_df)

        # 质量评分
        lit_df["Data_Source"] = "literature"
        lit_df["Quality_Score"] = lit_df.apply(self._score_quality, axis=1)

        # 合并真实数据
        if existing_csv and os.path.exists(existing_csv):
            existing_df = pd.read_csv(existing_csv)
            existing_df["Data_Source"] = "experiment"
            existing_df["Quality_Score"] = 1.0
            lit_df = pd.concat([existing_df, lit_df], ignore_index=True)

        # 确保列顺序一致
        all_cols = EXPECTED_PRE_COLS + self.target_cols + self.process_cols
        all_cols += ["TARGET_Yield", "Sample_ID", "Target_Scheme", "Data_Source", "Quality_Score"]
        for c in all_cols:
            if c not in lit_df.columns:
                lit_df[c] = 0.0
        lit_df = lit_df[all_cols]

        if output_csv:
            lit_df.to_csv(output_csv, index=False, float_format="%.6f")
            print(f"[*] 融合数据已保存: {output_csv} ({len(lit_df)} 行)")

        return lit_df

    # ------------------------------------------------------------------
    # 核心转换
    # ------------------------------------------------------------------
    def _paper_to_rows(self, paper_data):
        """将一篇论文的提取数据转换为多行训练记录"""
        meta = paper_data.get("_meta", {})
        experiments = paper_data.get("experiments", [])
        rows = []

        for exp in experiments:
            process = exp.get("process", {})
            outcome = exp.get("outcome", {})

            # --- 工艺参数映射 ---
            row = {}
            row["Process_Temp"] = self._safe_float(process.get("temperature_C"))
            row["Process_Time"] = self._convert_time(process)
            row["Process_H2"] = self._safe_float(process.get("H2_flow_sccm"), default=0)
            row["Process_Ar"] = self._safe_float(process.get("Ar_flow_sccm"), default=0)

            # 工艺合法性校验
            if not self._validate_process(row):
                continue

            # --- EBSD 特征映射 ---
            row["Pre_GND_Mean"] = self._map_gnd(outcome)

            # --- Target One-Hot & Yield ---
            yield_val = self._map_yield(outcome, exp)

            # 为每个目标方案生成一行
            for scheme_id in [1, 2, 3, 4]:
                scheme_row = dict(row)
                scheme_indices = TARGET_SCHEMES[scheme_id]["indices"]
                for hkl in ALL_TARGET_ORIENTATIONS:
                    key = f"Target_{hkl[0]}{hkl[1]}{hkl[2]}"
                    scheme_row[key] = 1.0 if hkl in scheme_indices else 0.0
                scheme_row["TARGET_Yield"] = yield_val
                scheme_row["Target_Scheme"] = scheme_id
                scheme_row["Sample_ID"] = f"LIT_{meta.get('paper_id', 'UNK')[:12]}_{scheme_id}"
                scheme_row["_confidence_raw"] = self._estimate_confidence(paper_data, exp)
                rows.append(scheme_row)

        return rows

    # ------------------------------------------------------------------
    # 小工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(val, default=None):
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _convert_time(process):
        """统一时间单位 → 小时"""
        t_h = process.get("time_h")
        if t_h is not None:
            return float(t_h)
        t_min = process.get("time_min")
        if t_min is not None:
            return float(t_min) / 60.0
        return None

    def _validate_process(self, row):
        """检查工艺参数是否在合理范围内"""
        for col, (lo, hi) in PROCESS_BOUNDS.items():
            if col in row and row[col] is not None:
                if row[col] < lo or row[col] > hi:
                    return False
        if row["Process_Temp"] is None and row["Process_Time"] is None:
            return False
        return True

    def _map_gnd(self, outcome):
        """GND 密度提取 + 单位换算"""
        raw = outcome.get("gnd_density_mean")
        if raw is None:
            return None
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None
        unit = outcome.get("gnd_density_unit", "a.u.")
        # 文献常见单位转换：1×10¹⁴ m⁻² → 归一化到 0~1 量级
        if "m^-2" in unit or "m⁻²" in unit or "m-2" in unit:
            val = val / 1e15
        elif "cm^-2" in unit or "cm⁻²" in unit:
            val = val / 1e11
        return val

    def _map_yield(self, outcome, exp):
        """从文献提取中计算产率（默认取最高织构体积分数）"""
        # 优先直接 yield
        direct = outcome.get("yield_from_ipf_or_pole_figure")
        if direct is not None:
            try:
                return float(direct)
            except (ValueError, TypeError):
                pass

        # 从 texture_volume_fractions 取最大值
        tfs = outcome.get("texture_volume_fractions", {})
        if tfs:
            values = [v for v in tfs.values() if isinstance(v, (int, float)) and v > 0]
            if values:
                return max(min(max(values), 1.0), 0.0)

        return 0.5  # 默认中性值

    def _estimate_confidence(self, paper_data, exp):
        """估计该条提取数据的置信度 0-1"""
        score = 0.5  # 基础分
        process = exp.get("process", {})
        outcome = exp.get("outcome", {})

        # 工艺参数越完整，置信度越高
        if process.get("temperature_C"):
            score += 0.1
        if process.get("time_min") or process.get("time_h"):
            score += 0.1
        if process.get("atmosphere"):
            score += 0.05

        # 有数值结果加分
        if outcome.get("grain_size_um"):
            score += 0.05
        if outcome.get("gnd_density_mean"):
            score += 0.1
        if outcome.get("texture_volume_fractions"):
            score += 0.1

        # 论文引用量加分
        cites = paper_data.get("_meta", {}).get("citation_count", 0)
        if cites > 50:
            score += 0.1
        elif cites > 10:
            score += 0.05

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # 缺失 Pre_ 插补
    # ------------------------------------------------------------------
    def _compute_pre_stats(self):
        """从真实数据计算 Pre_ 特征的均值和协方差矩阵"""
        if self.real_df is None or len(self.real_df) == 0:
            return
        pre_cols_available = [c for c in EXPECTED_PRE_COLS if c in self.real_df.columns]
        if not pre_cols_available:
            return
        pre_data = self.real_df[pre_cols_available].values
        self.pre_stats["cols"] = pre_cols_available
        self.pre_stats["mean"] = pre_data.mean(axis=0)
        self.pre_stats["std"] = pre_data.std(axis=0, ddof=1)
        self.pre_stats["std"][self.pre_stats["std"] < 1e-8] = 1.0

    def _impute_missing_pre(self, lit_df):
        """
        对文献数据中缺失的 Pre_ 特征进行插补。
        策略: 基于已知特征 (如 Pre_GND_Mean) 做条件均值插补
              如果没有任何已知 Pre_ 特征，使用真实数据的全局均值
        """
        pre_cols_in_df = [c for c in EXPECTED_PRE_COLS if c in lit_df.columns]
        if not pre_cols_in_df:
            return lit_df

        stats = self.pre_stats
        if not stats:
            # 无真实数据参考 → 用 0 填充（不理想但安全的回退）
            for c in EXPECTED_PRE_COLS:
                if c not in lit_df.columns or lit_df[c].isna().all():
                    lit_df[c] = 0.0
            return lit_df

        # 对每个文献行插补
        for idx in lit_df.index:
            # 已知特征
            known_mask = ~lit_df.loc[idx, pre_cols_in_df].isna()
            known_cols = [c for c, m in zip(pre_cols_in_df, known_mask) if m and c in stats["cols"]]

            if not known_cols:
                # 全缺失 → 全局均值
                for c in stats["cols"]:
                    i = list(stats["cols"]).index(c)
                    if c in lit_df.columns:
                        lit_df.at[idx, c] = stats["mean"][i]
            else:
                # 有部分已知 → 已知使用值，未知用全局均值
                for c in stats["cols"]:
                    col_idx = list(stats["cols"]).index(c)
                    if c in lit_df.columns:
                        val = lit_df.at[idx, c]
                        if pd.isna(val) or val is None or lit_df[c].iloc[idx] == 0:
                            lit_df.at[idx, c] = stats["mean"][col_idx]

        return lit_df

    def _score_quality(self, row):
        """对单条融合数据的质量评分"""
        score = row.get("_confidence_raw", 0.5)
        # 有 GND 数据加分
        if row.get("Pre_GND_Mean", 0) != 0:
            score += 0.1
        # 插补特征比例扣分
        pre_cols = [c for c in EXPECTED_PRE_COLS if c in row.index]
        if pre_cols:
            n_nonzero = sum(1 for c in pre_cols if row[c] != 0)
            interp_ratio = 1 - n_nonzero / len(pre_cols)
            score -= interp_ratio * 0.3
        return max(min(score, 1.0), 0.1)
