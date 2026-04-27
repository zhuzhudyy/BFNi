"""
主流水线：编排文献挖掘全流程

Usage:
    # 推荐流程：检索 → 导出DOI → 手动下载PDF → 本地提取 → 融合
    python -m lit_mining.pipeline --mode search     # 检索 + 导出 download_list.txt
    # → 通过清华图书馆逐篇下载 PDF，放入 lit_mining/local_pdfs/
    python -m lit_mining.pipeline --mode local      # 本地 PDF 提取
    python -m lit_mining.pipeline --mode fuse       # 融合成训练数据

    # 完整自动流程（仅 OA 论文）
    python -m lit_mining.pipeline --mode full
"""

import argparse
from pathlib import Path

# 自动加载项目根目录的 .env 文件
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, "..", ".."))))

from .config import CACHE_DIR, FUSED_CSV, PAPER_CACHE_JSON
from .searcher import LiteratureSearcher
from .extractor import PaperDataExtractor
from .feature_fusion import FeatureFusion


def _ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)


class LiteratureMiningPipeline:
    """文献挖掘主流水线"""

    def __init__(self):
        _ensure_dirs()
        self.searcher = LiteratureSearcher()
        self.extractor = PaperDataExtractor()
        self.fusion = FeatureFusion(real_training_csv="Optimized_Training_Data.csv")

    # ------------------------------------------------------------------
    # Phase 1: 检索 + 导出 DOI 下载清单
    # ------------------------------------------------------------------
    def phase_search(self, max_papers=60):
        print("\n" + "=" * 60)
        print(" Phase 1: 文献检索 ")
        print("=" * 60)
        papers = self.searcher.search_all()
        papers = self.searcher.filter_relevant(papers)
        self.searcher.save_cache()
        papers = papers[:max_papers]

        oa_count = sum(1 for p in papers if p.get("openAccessPdf"))
        paywall_count = len(papers) - oa_count
        print(f"\n[*] 检索完成: {len(papers)} 篇候选")
        print(f"    OA免费: {oa_count} 篇 | 需图书馆下载: {paywall_count} 篇")

        # 导出 DOI 下载清单
        self.searcher.export_doi_list(papers)

        # 在清单末尾添加操作说明
        self._append_manual_instructions()
        return papers

    def _append_manual_instructions(self):
        """追加清华图书馆下载指南"""
        instructions = """
# ============================================
# 手动下载步骤（清华大学）：
# 1. 确保连接清华 VPN 或校内网络
# 2. 打开上面的 doi.org 链接，浏览器会自动跳转到出版社（Elsevier/Springer/Wiley等）
# 3. 页面上应该有 "Download PDF" 或 "View PDF" 按钮 — 因为你用的学校IP直接可以下
# 4. 下载的 PDF 放入项目的 lit_mining/local_pdfs/ 文件夹
# 5. 文件名用论文标题简称即可（不要中文特殊字符）
# 6. 全部下载完成后运行: python -m lit_mining.pipeline --mode local
# ============================================
"""
        # 追加到 download_list.txt
        dl_path = "download_list.txt"
        if os.path.exists(dl_path):
            with open(dl_path, "a", encoding="utf-8") as f:
                f.write(instructions)
            print(f"[*] 下载指南已追加到: {dl_path}")

    # ------------------------------------------------------------------
    # Phase 2a: OA 自动提取（原有逻辑）
    # ------------------------------------------------------------------
    def phase_extract_oa(self, papers, max_papers=20):
        print("\n" + "=" * 60)
        print(" Phase 2a: OA 论文自动提取 ")
        print("=" * 60)

        extractable = []
        for p in papers:
            oa_info = p.get("openAccessPdf")
            if oa_info and oa_info.get("url"):
                extractable.append({
                    "pdf_url": oa_info["url"],
                    "paper_id": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "doi": p.get("externalIds", {}).get("DOI", ""),
                    "citationCount": p.get("citationCount", 0),
                })
        extractable = extractable[:max_papers]
        print(f"[*] {len(extractable)} 篇 OA 论文可自动下载提取")

        for i, item in enumerate(extractable, 1):
            print(f"\n--- [{i}/{len(extractable)}] ---")
            self.extractor.extract_from_pdf_url(
                item["pdf_url"], item["paper_id"], metadata=item
            )
            time.sleep(2.0)

        extractions_path = os.path.join(CACHE_DIR, "extractions.json")
        self.extractor.save_results(extractions_path)
        summary = self.extractor.get_summary()
        print(f"\n[*] 提取概要: {summary}")
        return extractions_path

    # ------------------------------------------------------------------
    # Phase 2b: 本地 PDF 提取（付费论文核心！）
    # ------------------------------------------------------------------
    def phase_extract_local(self, folder_path=None):
        print("\n" + "=" * 60)
        print(" Phase 2b: 本地 PDF 提取（含付费论文）")
        print("=" * 60)

        if folder_path is None:
            folder_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "local_pdfs"
            )
        os.makedirs(folder_path, exist_ok=True)

        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"\n[!] 文件夹为空: {folder_path}")
            print("    请按 download_list.txt 中的清单，通过清华图书馆下载 PDF")
            print("    将下载的 PDF 放入此文件夹后重新运行")
            return None

        print(f"[*] 在 {folder_path} 中发现 {len(pdf_files)} 个 PDF")
        self.extractor.extract_from_local_folder(folder_path)

        extractions_path = os.path.join(CACHE_DIR, "extractions.json")
        self.extractor.save_results(extractions_path)
        summary = self.extractor.get_summary()
        print(f"\n[*] 提取概要: {summary}")
        return extractions_path

    # ------------------------------------------------------------------
    # Phase 3: 融合
    # ------------------------------------------------------------------
    def phase_fuse(self, extractions_path=None):
        print("\n" + "=" * 60)
        print(" Phase 3: 特征工程 & 数据融合 ")
        print("=" * 60)

        if extractions_path is None:
            extractions_path = os.path.join(CACHE_DIR, "extractions.json")
        if not os.path.exists(extractions_path):
            print(f"[!] 未找到提取结果: {extractions_path}")
            return None

        fused_df = self.fusion.fuse(
            extractions_path,
            output_csv=FUSED_CSV,
            existing_csv="Optimized_Training_Data.csv",
        )

        if fused_df is not None:
            exp_count = len(fused_df[fused_df["Data_Source"] == "experiment"])
            lit_count = len(fused_df[fused_df["Data_Source"] == "literature"])
            print(f"    融合完成: {exp_count} 实验 + {lit_count} 文献 = {len(fused_df)} 行")
            print(f"    输出: {FUSED_CSV}")

        return fused_df

    # ------------------------------------------------------------------
    # 组合流程
    # ------------------------------------------------------------------
    def run_full(self):
        """完整自动流程（检索 + OA提取 + 融合）"""
        papers = self.phase_search(max_papers=60)
        if not papers:
            print("[!] 未检索到论文")
            return
        # 先自动提取 OA 的
        extractions_path = self.phase_extract_oa(papers, max_papers=20)
        if extractions_path:
            self.phase_fuse(extractions_path)
        # 提示手动下载付费的
        print("\n" + "=" * 60)
        print(" 还有付费论文需要手动下载 ")
        print("=" * 60)
        print("  查看 download_list.txt → 通过清华图书馆下载 PDF")
        print("  放入 lit_mining/local_pdfs/ → 运行:")
        print("    python -m lit_mining.pipeline --mode local")
        print("    python -m lit_mining.pipeline --mode fuse")


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="文献数据挖掘 Agent")
    parser.add_argument(
        "--mode",
        choices=["full", "search", "extract", "local", "fuse", "stepwise"],
        default="stepwise",
        help="""
运行模式:
  search   - 仅检索 + 导出 DOI 下载清单 (download_list.txt)
  extract  - 自动下载 OA 论文 PDF 并 LLM 提取
  local    - 从 lit_mining/local_pdfs/ 提取本地 PDF（付费论文核心）
  fuse     - 将提取结果融合为训练数据 CSV
  full     - 完整自动流程 (search + extract + fuse + 提示手动下载)
  stepwise - 分步交互模式
""",
    )
    parser.add_argument(
        "--extractions", type=str, default=None,
        help="提取结果 JSON 路径 (配合 --mode fuse 使用)",
    )
    args = parser.parse_args()

    pipeline = LiteratureMiningPipeline()

    if args.mode == "full":
        pipeline.run_full()

    elif args.mode == "search":
        papers = pipeline.phase_search()
        print(f"\n[*] Top 20:")
        for i, p in enumerate(papers[:20], 1):
            doi = p.get("externalIds", {}).get("DOI", "")
            print(f"  {i:2d}. ({p.get('year','?')}) [{p.get('citationCount',0)} cites] "
                  f"{p.get('title','N/A')[:90]}")
        print(f"\n[*] 下载清单: download_list.txt")
        print("[*] 下一步: 通过清华图书馆下载 PDF → 放入 lit_mining/local_pdfs/")
        print("    然后运行: python -m lit_mining.pipeline --mode local")

    elif args.mode == "extract":
        papers = pipeline.phase_search()
        pipeline.phase_extract_oa(papers)

    elif args.mode == "local":
        pipeline.phase_extract_local()
        print("\n[*] 下一步: python -m lit_mining.pipeline --mode fuse")

    elif args.mode == "fuse":
        pipeline.phase_fuse(args.extractions)

    elif args.mode == "stepwise":
        print("\n" + "=" * 60)
        print(" 文献挖掘分步向导 ")
        print("=" * 60)
        print("  [s]  检索 + 导出DOI清单 (download_list.txt)")
        print("  [oa] 自动提取 OA 论文")
        print("  [L]  提取本地 PDF (lit_mining/local_pdfs/)")
        print("  [f]  融合为训练数据 CSV")
        print("=" * 60)

        while True:
            c = input("\n选择 (s/oa/L/f/q): ").strip().lower()
            if c == "q":
                break
            elif c == "s":
                pipeline.phase_search()
            elif c == "oa":
                papers = pipeline.phase_search()
                pipeline.phase_extract_oa(papers)
            elif c == "l":
                pipeline.phase_extract_local()
            elif c == "f":
                pipeline.phase_fuse()
                break

        print("\n[*] 输出文件: literature_training_data.csv")


if __name__ == "__main__":
    main()
