"""
Module B: PDF 解析 + LLM 结构化数据提取器

流程:
  1. 下载 PDF (OA 链接 / Semantic Scholar / Unpaywall)
  2. PyMuPDF 提取正文文本
  3. 调用 LLM 提取结构化 JSON 数据
  4. 支持表格自动检测和图表标题收集
"""

import json
import os
import re
import time
import hashlib
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .config import (
    LLM_PROVIDER,
    ANTHROPIC_MODEL_EXTRACT,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    EXTRACT_MAX_TOKENS,
    PDF_MAX_CHARS,
    CACHE_DIR,
)

# ---------------------------------------------------------------------------
# LLM client 抽象层
# ---------------------------------------------------------------------------
_EXTRACTION_SYSTEM_PROMPT = """\
You are a materials science data extraction specialist. Your task is to read
the text of a scientific paper about nickel (Ni) annealing/heat-treatment and
extract EVERY quantitative experimental data point into a structured JSON format.

IMPORTANT RULES:
- Extract ALL numerical values you find — temperature, time, flow rates, yields,
  grain sizes, GND densities, texture fractions, hardness, etc.
- If a value is stated as a range "800-1000°C", capture both min and max.
- If atmosphere is mentioned (e.g. "Ar+5%H2"), capture the gas composition.
- If the paper reports texture volume fractions (e.g. "Cube component was 35%"),
  capture the exact {hkl}<uvw> notation and fraction.
- If GND density is reported (e.g. "GND density of 2.3×10¹⁴ m⁻²"), extract it
  with its unit.
- For IPF maps / EBSD figures: note in `ebsd_figures` if IPF maps exist, and
  what orientations they show.
- If the paper mentions initial state (cold-rolled, electrodeposited, annealed,
  ECAP, etc.), capture it.
- If a value is NOT present, set it to null. Never invent values.
- Output ONLY valid JSON, no markdown fences, no commentary."""

_EXTRACTION_JSON_SCHEMA = """\
{
  "material": "pure nickel / Ni alloy composition",
  "purity": "99.9% or null",
  "initial_state": "cold-rolled 80% / electrodeposited / ECAP / etc.",
  "initial_grain_size_um": null,
  "experiments": [
    {
      "sample_id": "S1 or as reported",
      "process": {
        "temperature_C": 800,
        "temperature_range_C": [800, 1000],
        "time_min": 60,
        "time_h": 1.0,
        "atmosphere": "Ar + 5% H2",
        "H2_flow_sccm": null,
        "Ar_flow_sccm": null,
        "heating_rate_C_per_min": 10,
        "cooling_method": "furnace / air / water quench"
      },
      "outcome": {
        "grain_size_um": 25.3,
        "gnd_density_mean": 1.2e14,
        "gnd_density_unit": "m^-2",
        "hardness_HV": null,
        "twin_fraction": 0.15,
        "recrystallized_fraction": 0.8,
        "texture_volume_fractions": {
          "cube_{100}<001>": 0.35,
          "goss_{110}<001>": 0.05,
          "brass_{110}<112>": 0.10,
          "copper_{112}<111>": 0.20,
          "S_{123}<634>": 0.08,
          "other": 0.22
        },
        "ipf_orientations_observed": ["{100}", "{111}", "{110}"],
        "yield_from_ipf_or_pole_figure": null
      }
    }
  ],
  "ebsd_available": false,
  "ebsd_figures": [
    {
      "figure_number": "Fig. 3",
      "type": "IPF map / Pole figure / ODF / GND map",
      "caption": "...",
      "conditions_shown": "annealed at 900°C for 1h"
    }
  ],
  "notes": "any additional relevant info"
}"""


def _call_anthropic(system_prompt, user_text, model=None, max_tokens=None):
    """调用 Anthropic Claude API"""
    import anthropic
    client = anthropic.Anthropic()
    model = model or ANTHROPIC_MODEL_EXTRACT
    max_tokens = max_tokens or EXTRACT_MAX_TOKENS

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_text}],
    )
    return resp.content[0].text


def _call_openai(system_prompt, user_text, model=None, max_tokens=None):
    """调用 OpenAI / OpenAI-compatible API"""
    from openai import OpenAI
    client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    model = model or OPENAI_MODEL
    max_tokens = max_tokens or EXTRACT_MAX_TOKENS

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def _call_llm(user_text):
    """统一 LLM 调用入口"""
    system = _EXTRACTION_SYSTEM_PROMPT + "\n\nOUTPUT SCHEMA:\n" + _EXTRACTION_JSON_SCHEMA
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(system, user_text)
    else:
        return _call_openai(system, user_text)


# ---------------------------------------------------------------------------
# PDF 处理
# ---------------------------------------------------------------------------
def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _download_pdf(url, paper_id):
    """下载 PDF 到缓存目录，返回本地路径"""
    _ensure_cache_dir()

    safe_id = hashlib.md5(paper_id.encode()).hexdigest()[:12] if paper_id else "unknown"
    pdf_path = os.path.join(CACHE_DIR, f"{safe_id}.pdf")

    if os.path.exists(pdf_path):
        return pdf_path

    try:
        req = Request(url, headers={"User-Agent": "LitMiningBot/1.0 (academic research)"})
        with urlopen(req, timeout=60) as resp:
            with open(pdf_path, "wb") as f:
                f.write(resp.read())
        return pdf_path
    except Exception as e:
        print(f"    [!] PDF 下载失败: {e}")
        return None


def _extract_text_from_pdf(pdf_path):
    """PyMuPDF 提取正文"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("    [!] PyMuPDF 未安装。 pip install pymupdf")
        return ""

    text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
    except Exception as e:
        print(f"    [!] PDF 解析失败: {e}")
        return ""

    return "\n".join(text_parts)


def _truncate_text(text, max_chars=None):
    """智能截断：按段落边界切割，保留开头+结尾"""
    max_chars = max_chars or PDF_MAX_CHARS
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    paragraphs = text.split("\n\n")
    front, back = [], []
    front_len = 0
    for p in paragraphs:
        if front_len + len(p) > half:
            break
        front.append(p)
        front_len += len(p)

    back_len = 0
    for p in reversed(paragraphs):
        if back_len + len(p) > half:
            break
        back.append(p)
        back_len += len(p)

    return "\n\n".join(front) + "\n\n... [truncated] ...\n\n" + "\n\n".join(reversed(back))


def _clean_json_output(raw_text):
    """从 LLM 输出中提取纯 JSON"""
    # 移除可能的 markdown fence
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉第一行 ```json 和最后一行 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# 数据提取器
# ---------------------------------------------------------------------------
class PaperDataExtractor:
    """论文数据提取器"""

    def __init__(self):
        self.results = []

    def extract_from_local_folder(self, folder_path=None):
        """
        从本地文件夹批量提取 PDF。
        用法：用清华图书馆代理逐篇下载 PDF，放入 lit_mining/local_pdfs/，然后调用此方法。
        """
        if folder_path is None:
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_pdfs")
        if not os.path.isdir(folder_path):
            print(f"[!] 文件夹不存在: {folder_path}")
            print("    请创建该文件夹并将下载的 PDF 放入其中")
            return []

        pdf_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf") and not f.startswith(".")
        ])
        if not pdf_files:
            print(f"[!] {folder_path} 中没有 PDF 文件")
            print("    请通过清华图书馆代理下载论文 PDF 后放入该文件夹")
            return []

        print(f"[*] 发现 {len(pdf_files)} 个本地 PDF 文件")
        results = []
        for i, fname in enumerate(pdf_files, 1):
            pdf_path = os.path.join(folder_path, fname)
            paper_name = fname.replace(".pdf", "").replace("_", " ")[:60]
            print(f"\n--- [{i}/{len(pdf_files)}] {paper_name} ---")

            text = _extract_text_from_pdf(pdf_path)
            if not text or len(text) < 200:
                print(f"    [!] PDF 文本过短，跳过")
                continue

            text = _truncate_text(text)
            print(f"    [*] 提取文本 {len(text)} 字符，调用 LLM...")

            try:
                raw = _call_llm(text)
                clean = _clean_json_output(raw)
                data = json.loads(clean)
                data["_meta"] = {
                    "paper_id": fname[:40],
                    "title": paper_name,
                    "extraction_ok": True,
                    "source": "local_pdf",
                }
                n_exps = len(data.get("experiments", []))
                print(f"    [✓] 提取成功: {n_exps} 条实验记录")
                self.results.append(data)
                results.append(data)
            except json.JSONDecodeError as e:
                print(f"    [!] JSON 解析失败: {e}")
            except Exception as e:
                print(f"    [!] LLM 调用失败: {e}")
            time.sleep(1.5)

        return results

    def extract_from_pdf_url(self, pdf_url, paper_id, metadata=None):
        """完整的单篇提取流程：下载→解析→LLM提取"""
        metadata = metadata or {}

        print(f"  [*] 处理: {metadata.get('title', paper_id)[:60]}...")
        pdf_path = _download_pdf(pdf_url, paper_id)
        if not pdf_path:
            return None

        text = _extract_text_from_pdf(pdf_path)
        if not text or len(text) < 200:
            print(f"    [!] PDF 文本过短，跳过")
            return None

        text = _truncate_text(text)
        print(f"    [*] 提取文本 {len(text)} 字符，调用 LLM...")

        try:
            raw = _call_llm(text)
            clean = _clean_json_output(raw)
            data = json.loads(clean)
            data["_meta"] = {
                "paper_id": paper_id,
                "title": metadata.get("title", ""),
                "year": metadata.get("year", ""),
                "doi": metadata.get("doi", ""),
                "citation_count": metadata.get("citationCount", 0),
                "extraction_ok": True,
            }
            n_exps = len(data.get("experiments", []))
            print(f"    [✓] 提取成功: {n_exps} 条实验记录")
            self.results.append(data)
            return data
        except json.JSONDecodeError as e:
            print(f"    [!] JSON 解析失败: {e}")
            return {"_meta": {"paper_id": paper_id, "extraction_ok": False}, "raw_output": raw[:500]}
        except Exception as e:
            print(f"    [!] LLM 调用失败: {e}")
            return None

    def extract_from_text(self, text, metadata=None):
        """直接从文本提取（当 PDF 已预先解析好时）"""
        metadata = metadata or {}
        text = _truncate_text(text)
        try:
            raw = _call_llm(text)
            clean = _clean_json_output(raw)
            data = json.loads(clean)
            data["_meta"] = dict(metadata, extraction_ok=True)
            self.results.append(data)
            return data
        except Exception as e:
            print(f"    [!] 提取失败: {e}")
            return None

    def extract_batch(self, papers_with_urls, delay=2.0):
        """批量提取多篇论文"""
        for item in papers_with_urls:
            result = self.extract_from_pdf_url(
                item.get("pdf_url", ""),
                item.get("paper_id", str(hash(item.get("title", "")))),
                metadata=item,
            )
            if result:
                yield result
            time.sleep(delay)

    def save_results(self, path):
        """保存提取结果到 JSON"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"[*] {len(self.results)} 条提取结果已保存至: {path}")

    def get_summary(self):
        """返回提取摘要统计"""
        total_exps = sum(len(r.get("experiments", [])) for r in self.results if r.get("_meta", {}).get("extraction_ok"))
        ok = sum(1 for r in self.results if r.get("_meta", {}).get("extraction_ok"))
        fail = len(self.results) - ok
        return {"total_papers": len(self.results), "ok": ok, "fail": fail, "total_experiments": total_exps}
