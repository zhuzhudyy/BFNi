"""
纯镍退火文献检索脚本
执行 Semantic Scholar 检索，筛选符合 EBSD + 工艺条件的文献
"""

import json
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.parse import quote
from urllib.error import HTTPError, URLError

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lit_mining.config import SEMANTIC_SCHOLAR_BASE, SEMANTIC_SCHOLAR_FIELDS

# 纯镍退火专用检索式 - 严格匹配关键词组 A + B
PURE_NICKEL_QUERIES = [
    # 核心检索式：Pure Nickel + Annealing + EBSD
    '"pure nickel" annealing EBSD texture',
    '"pure nickel" annealing EBSD recrystallization',
    '"pure nickel" annealing EBSD grain growth',
    '"pure nickel" annealing EBSD orientation',
    '"pure nickel" annealing EBSD GND',
    # 变体检索式
    '"polycrystalline nickel" annealing EBSD',
    '"electrodeposited nickel" annealing EBSD texture',
    '"pure Ni" annealing EBSD recrystallization',
    '"nickel" annealing "electron backscatter diffraction" texture',
    '"nickel" annealing EBSD "grain orientation"',
    # 中文检索式
    '纯镍 退火 EBSD 织构',
    '纯镍 再结晶 EBSD 取向',
]

MAX_RESULTS_PER_QUERY = 25
CACHE_DIR = "lit_mining/cache"
OUTPUT_DIR = "lit_mining/output"

def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("lit_mining/local_pdfs", exist_ok=True)

def http_get(url, retries=3, delay=3.0):
    """带重试的 HTTP GET"""
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PureNickelBot/1.0"
    })
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                wait_time = delay * (attempt + 2)
                print(f"      [~] 触发速率限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"      [!] HTTP 错误 {e.code}")
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
        except URLError as e:
            print(f"      [!] 网络错误：{e}")
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    return None

def search_query(query, limit=25):
    """执行单次检索"""
    url = (
        f"{SEMANTIC_SCHOLAR_BASE}/paper/search?"
        f"query={quote(query)}&limit={limit}"
        f"&fields={SEMANTIC_SCHOLAR_FIELDS}"
    )
    print(f"  检索：{query[:50]}...")
    data = http_get(url)
    if data:
        return data.get("data", [])
    return []

def filter_pure_nickel(papers):
    """
    筛选严格符合要求的文献：
    1. 材质必须是纯镍（Pure Nickel）
    2. 必须包含退火工艺
    3. 必须包含 EBSD 表征
    """
    filtered = []
    exclude_terms = [
        "nickel alloy", "ni-cr", "ni-w", "ni-mo", "ni-fe", "ni-cu",
        "superalloy", "inconel", "hastelloy", "monel",
        "composite", "coating", "film", "nanoparticle",
        "molecular dynamics", "simulation", "DFT", "first-principles",
    ]
    required_terms = ["nickel", "anneal"]  # 必须包含
    ebsd_terms = ["ebsd", "electron backscatter", "orientation imaging", "ipf", "texture"]

    for paper in papers:
        title = (paper.get("title", "") or "").lower()
        abstract = (paper.get("abstract", "") or "").lower()
        text = title + " " + abstract

        # 排除镍基合金
        if any(term in text for term in exclude_terms):
            continue

        # 必须包含镍和退火
        if not all(term in text for term in required_terms):
            continue

        # 必须包含 EBSD 相关术语
        if not any(term in text for term in ebsd_terms):
            continue

        filtered.append(paper)

    return filtered

def save_papers(papers, output_path):
    """保存论文列表到 JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"  已保存：{output_path}")

def export_doi_list(papers, output_path):
    """导出 DOI 列表"""
    lines = []
    lines.append("# 纯镍退火 EBSD 文献下载清单")
    lines.append("# 包含工艺参数 (800-1400°C) 和 EBSD 数据的文献")
    lines.append("# 使用清华图书馆代理下载：https://doi.org/DOI 号")
    lines.append("")

    for i, p in enumerate(papers, 1):
        doi = p.get("externalIds", {}).get("DOI", "")
        title = p.get("title", "N/A")[:100]
        year = p.get("year", "?")
        cites = p.get("citationCount", 0)
        oa = " [免费 OA]" if p.get("openAccessPdf") else ""

        lines.append(f"[{i:02d}] [{year}] 引用:{cites:4d}{oa}")
        lines.append(f"     标题：{title}")
        if doi:
            lines.append(f"     DOI: {doi}")
            lines.append(f"     链接：https://doi.org/{doi}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  DOI 列表已导出：{output_path}")

def main():
    print("=" * 70)
    print("纯镍退火 EBSD 文献检索")
    print("=" * 70)

    ensure_dirs()

    all_papers = []
    seen_ids = set()

    for i, query in enumerate(PURE_NICKEL_QUERIES, 1):
        print(f"\n[{i}/{len(PURE_NICKEL_QUERIES)}] 执行检索式...")
        results = search_query(query, MAX_RESULTS_PER_QUERY)

        for paper in results:
            pid = paper.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(paper)

        time.sleep(4.0)  # 请求间隔

    print(f"\n=== 检索完成 ===")
    print(f"原始检索结果：{len(all_papers)} 篇")

    # 筛选符合纯镍 + 退火+EBSD 的文献
    filtered_papers = filter_pure_nickel(all_papers)
    print(f"筛选后（纯镍 + 退火+EBSD）：{len(filtered_papers)} 篇")

    # 按引用数排序
    filtered_papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)

    # 保存结果
    save_papers(all_papers, f"{CACHE_DIR}/papers_all.json")
    save_papers(filtered_papers, f"{CACHE_DIR}/papers_pure_nickel_ebsd.json")
    export_doi_list(filtered_papers, "download_list_pure_nickel.txt")

    # 打印详细列表
    print("\n=== 符合条件的文献列表 ===")
    for i, p in enumerate(filtered_papers[:30], 1):
        title = p.get("title", "N/A")[:70]
        year = p.get("year", "?")
        cites = p.get("citationCount", 0)
        doi = p.get("externalIds", {}).get("DOI", "N/A")
        oa = " [OA]" if p.get("openAccessPdf") else ""
        print(f"  {i:2d}. [{year}] ({cites:4d}){oa} {title}")
        if doi != "N/A":
            print(f"      DOI: {doi[:60]}...")

    print(f"\n=== 数据保存路径 ===")
    print(f"  - 原始检索结果：{os.path.abspath(CACHE_DIR)}/papers_all.json")
    print(f"  - 筛选后结果：{os.path.abspath(CACHE_DIR)}/papers_pure_nickel_ebsd.json")
    print(f"  - DOI 下载列表：{os.path.abspath('download_list_pure_nickel.txt')}")
    print(f"  - 本地 PDF 目录：{os.path.abspath('lit_mining/local_pdfs')}")

    return filtered_papers

if __name__ == "__main__":
    papers = main()
