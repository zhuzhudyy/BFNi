"""
文献抓取与提取执行脚本
- 从 papers_metadata.json 加载论文
- 筛选纯镍相关 OA 论文
- 下载 PDF → LLM 提取结构化数据
- 对无 OA PDF 的论文，从摘要提取
"""
import json
import os
import sys
import time
import hashlib
import ssl
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, "..", ".."))))

from lit_mining.config import CACHE_DIR, PAPER_CACHE_JSON, PDF_MAX_CHARS
from lit_mining.extractor import PaperDataExtractor, _truncate_text

# ─── 纯镍筛选关键词 ───
NI_MUST_KEYWORDS = ["nickel", " ni ", "pure ni", "pure nickel", "ni ", " ni,"]
NI_EXCLUDE_KEYWORDS = [
    "monel", "inconel", "hastelloy", "superalloy", "super alloy",
    "ni-cr", "nicr", "ni-fe", "nife", "ni-w", "niw", "ni-co", "nico",
    "ni-ti", "niti", "ni-mn", "nimn", "ni-al", "nial",
    "stainless steel", "duplex", "maraging",
    "titanium", "magnesium alloy", "aluminum alloy", "al alloy",
    "copper alloy", "cu alloy", "brass", "bronze",
    "shape memory", "sma",
    "composite", "nanocomposite", "coating",
    "corrosion", "oxidation",
    "welding", "friction stir", "fsp", "fssw",
    "laser", "additive manufacturing", "selective laser",
    "hydrogen", "hydride",
    "superelastic", "superplastic",
    "electrical steel", "fe-si", "fesi", "silicon steel",
    "tungsten", "tungsten alloy", "w alloy",
    "zinc oxide", "zno", "ceramic",
]

ANNEAL_KEYWORDS = [
    "anneal", "recrystall", "grain growth", "grain size",
    "heat treatment", "heat-treatment", "thermal",
    "texture", "ebsd", "ipf", "orientation",
    "microstructure", "cold roll", "cold-roll", "hot roll",
    "ecap", "electrodeposit", "severe plastic", "spd",
]


def load_all_papers():
    cache_path = PAPER_CACHE_JSON
    if not os.path.exists(cache_path):
        print(f"[!] 缓存文件不存在: {cache_path}")
        return []
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    seen = set()
    papers = []
    for key, val in cache.items():
        if key.startswith("_"):
            continue
        if isinstance(val, list):
            for p in val:
                pid = p.get("paperId", "")
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append(p)
    return papers


def is_pure_nickel(paper):
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    combined = title + " " + abstract

    has_ni = any(kw in combined for kw in NI_MUST_KEYWORDS)
    if not has_ni:
        # 也检查 externalIds 中的标题
        ext_title = (paper.get("externalIds", {}).get("title") or "").lower()
        has_ni = any(kw in ext_title for kw in NI_MUST_KEYWORDS)

    if not has_ni:
        return False

    for kw in NI_EXCLUDE_KEYWORDS:
        if kw in combined:
            return False

    return True


def is_anneal_related(paper):
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    combined = title + " " + abstract
    return any(kw in combined for kw in ANNEAL_KEYWORDS)


def has_ebsd(paper):
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    return any(kw in text for kw in ["ebsd", "electron backscatter", "orientation imag", "ipf map", "pole figure", "gnd"])


def download_pdf(url, cache_subdir="oa_pdfs"):
    d = os.path.join(CACHE_DIR, cache_subdir)
    os.makedirs(d, exist_ok=True)
    fname = hashlib.md5(url.encode()).hexdigest()[:16] + ".pdf"
    fpath = os.path.join(d, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
        return fpath
    try:
        req = Request(url, headers={"User-Agent": "LitMiningBot/2.0 (academic research; mailto:research@example.com)"})
        with urlopen(req, timeout=90, context=_SSL_CONTEXT) as resp:
            data = resp.read()
        if len(data) < 2000:
            return None
        with open(fpath, "wb") as f:
            f.write(data)
        return fpath
    except Exception as e:
        print(f"      [!] PDF 下载失败: {e}")
        return None


def main():
    print("=" * 60)
    print(" 文献抓取与数据提取")
    print("=" * 60)

    # 1. 加载论文
    print("\n[1/5] 加载论文缓存...")
    all_papers = load_all_papers()
    print(f"      去重后共 {len(all_papers)} 篇")

    # 2. 筛选纯镍 + 退火相关
    print("\n[2/5] 筛选纯镍 + 退火相关论文...")
    ni_papers = [p for p in all_papers if is_pure_nickel(p)]
    print(f"      纯镍相关: {len(ni_papers)} 篇")
    related = [p for p in ni_papers if is_anneal_related(p)]
    print(f"      且退火相关: {len(related)} 篇")

    ebsd_papers = [p for p in related if has_ebsd(p)]
    non_ebsd = [p for p in related if not has_ebsd(p)]
    print(f"      其中含EBSD: {len(ebsd_papers)} 篇, 无EBSD: {len(non_ebsd)} 篇")

    # 排序：EBSD优先 + 高引用
    ebsd_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    non_ebsd.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    candidates = ebsd_papers + non_ebsd

    # 限制数量
    candidates = candidates[:50]
    oa_papers = [p for p in candidates if p.get("openAccessPdf", {}).get("url")]
    no_oa_papers = [p for p in candidates if not p.get("openAccessPdf", {}).get("url")]
    print(f"\n      最终候选: {len(candidates)} 篇")
    print(f"      OA 可下载: {len(oa_papers)} 篇")
    print(f"      需手动下载: {len(no_oa_papers)} 篇")

    # 导出筛选后的下载清单
    _export_filtered_list(ebsd_papers, non_ebsd)

    extractor = PaperDataExtractor()

    # 3. 下载 OA PDF + LLM 提取
    print(f"\n[3/5] 下载并提取 OA 论文 PDF ({len(oa_papers)} 篇)...")
    oa_results = []
    for i, p in enumerate(oa_papers, 1):
        title = (p.get("title") or "N/A")[:70]
        cites = p.get("citationCount", 0)
        ebsd_tag = " [EBSD]" if has_ebsd(p) else ""
        print(f"\n  --- [{i}/{len(oa_papers)}] ({cites} cites){ebsd_tag} {title}")

        oa_url = p.get("openAccessPdf", {}).get("url")
        if not oa_url:
            continue

        pdf_path = download_pdf(oa_url)
        if not pdf_path:
            print(f"      [~] 无 OA PDF，从摘要提取")
            abstract = p.get("abstract") or ""
            if len(abstract) > 200:
                result = extractor.extract_from_text(
                    abstract,
                    metadata={
                        "paper_id": p.get("paperId", ""),
                        "title": p.get("title", ""),
                        "year": p.get("year", ""),
                        "doi": (p.get("externalIds") or {}).get("DOI", ""),
                        "citation_count": cites,
                    },
                )
                if result:
                    oa_results.append(result)
            continue

        print(f"      下载完成 -> {os.path.basename(pdf_path)}")
        result = extractor.extract_from_pdf_url(
            pdf_url=oa_url,
            paper_id=p.get("paperId", ""),
            metadata={
                "pdf_local_path": pdf_path,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "doi": (p.get("externalIds") or {}).get("DOI", ""),
                "citationCount": cites,
            },
        )
        if result:
            oa_results.append(result)
        time.sleep(2.0)

    # 4. 从摘要中提取（无 OA PDF 的论文）
    print(f"\n[4/5] 从摘要提取 ({len(no_oa_papers)} 篇无 OA 论文)...")
    abstract_results = []
    for i, p in enumerate(no_oa_papers[:30], 1):
        title = (p.get("title") or "N/A")[:70]
        abstract = p.get("abstract") or ""
        if len(abstract) < 200:
            continue
        print(f"\n  --- [{i}/{min(len(no_oa_papers), 30)}] {title}")
        result = extractor.extract_from_text(
            abstract,
            metadata={
                "paper_id": p.get("paperId", ""),
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "doi": (p.get("externalIds") or {}).get("DOI", ""),
                "citation_count": p.get("citationCount", 0),
            },
        )
        if result:
            abstract_results.append(result)
        time.sleep(1.5)

    # 5. 保存结果
    print(f"\n[5/5] 保存提取结果...")
    extractions_path = os.path.join(CACHE_DIR, "extractions.json")
    extractor.save_results(extractions_path)

    summary = extractor.get_summary()
    print("\n" + "=" * 60)
    print(" 提取完成")
    print("=" * 60)
    print(f"  总候选论文:     {len(candidates)}")
    print(f"  OA PDF 提取:    {len(oa_papers)} 篇尝试, {summary['ok'] - len(abstract_results)} 篇成功")
    print(f"  摘要提取:       {len(no_oa_papers)} 篇尝试")
    print(f"  成功提取:       {summary['ok']} 篇")
    print(f"  总实验记录:     {summary['total_experiments']} 条")
    print(f"  结果保存至:     {extractions_path}")

    # 简要预览
    if extractor.results:
        print(f"\n  成功提取的前5篇:")
        for r in extractor.results[:5]:
            meta = r.get("_meta", {})
            n_exp = len(r.get("experiments", []))
            title = meta.get("title", "?")[:60]
            print(f"    - [{meta.get('year','?')}] {title} -> {n_exp} 条实验")


def _export_filtered_list(ebsd_papers, non_ebsd):
    """导出筛选后的下载清单"""
    lines = [
        "# 纯镍退火文献筛选结果",
        "# EBSD 优先 + 高引用排序",
        "# 将下载的 PDF 放入 lit_mining/local_pdfs/ 文件夹",
        "",
        "# ====== 含 EBSD 数据（优先下载） ======",
    ]
    for p in ebsd_papers:
        doi = (p.get("externalIds") or {}).get("DOI", "")
        title = (p.get("title") or "N/A")[:100]
        year = p.get("year", "?")
        cites = p.get("citationCount", 0)
        oa = " [OA免费]" if p.get("openAccessPdf", {}).get("url") else ""
        lines.append(f"\n[{year}] cites={cites}{oa} | {title}")
        if doi:
            lines.append(f"  https://doi.org/{doi}")

    lines.append("\n\n# ====== 无明确 EBSD 关键词（次优先） ======")
    for p in non_ebsd[:20]:
        doi = (p.get("externalIds") or {}).get("DOI", "")
        title = (p.get("title") or "N/A")[:100]
        year = p.get("year", "?")
        cites = p.get("citationCount", 0)
        oa = " [OA免费]" if p.get("openAccessPdf", {}).get("url") else ""
        lines.append(f"\n[{year}] cites={cites}{oa} | {title}")
        if doi:
            lines.append(f"  https://doi.org/{doi}")

    path = "download_list_filtered.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"      筛选清单已导出: {path}")


if __name__ == "__main__":
    main()
