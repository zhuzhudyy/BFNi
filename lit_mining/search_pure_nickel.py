"""
纯镍退火 EBSD 文献检索 - 使用 Semantic Scholar 备用策略
增加延迟，减少并发，使用已缓存的论文数据
"""

import json
import os
import time
from urllib.request import Request, urlopen
from urllib.parse import quote
from urllib.error import HTTPError, URLError

# Semantic Scholar API 配置
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_FIELDS = "title,authors,year,abstract,externalIds,url,openAccessPdf,journal,publicationTypes,citationCount"

# 单个精确检索式
QUERIES = [
    "pure nickel annealing EBSD",
]

MAX_RESULTS = 50
OUTPUT_DIR = "lit_mining/output/pure_nickel_ebsd"
CACHE_FILE = "lit_mining/cache/pure_nickel_search.json"

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

def http_get(url, retries=5, base_delay=10):
    """带指数退避的 HTTP GET"""
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 PureNickelBot/1.0 Academic"
    })
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                delay = base_delay * (attempt + 1)
                print(f"  [!] 速率限制 429，等待 {delay} 秒后重试...")
                time.sleep(delay)
            elif e.code == 403:
                print(f"  [!] 访问被拒绝 403")
                return None
            else:
                print(f"  [!] HTTP 错误 {e.code}，等待后重试...")
                time.sleep(base_delay)
        except URLError as e:
            print(f"  [!] 网络错误：{e}")
            time.sleep(base_delay)
    return None

def search(query, limit=50):
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?query={quote(query)}&limit={limit}&fields={SEMANTIC_SCHOLAR_FIELDS}"
    print(f"检索：{query}")
    return http_get(url, retries=5, base_delay=15)

def filter_papers(papers):
    """筛选纯镍 + 退火+EBSD 文献"""
    filtered = []
    # 排除项：镍基合金、模拟计算、复合材料
    exclude = [
        "nickel alloy", "ni-cr", "ni-w", "ni-mo", "ni-fe", "ni-cu",
        "superalloy", "inconel", "hastelloy", "monel",
        "composite", "nanocomposite", "coating", "film",
        "molecular dynamics", "simulation", "dft", "first-principles",
        "phase field", "monte carlo", "modeling",
    ]
    # EBSD 相关术语
    ebsd_terms = [
        "ebsd", "electron backscatter", "orientation imaging microscopy",
        "oim", "ipf", "inverse pole figure", "texture evolution",
        "grain orientation", "crystallographic texture",
    ]

    for p in papers:
        title = (p.get("title", "") or "").lower()
        abstract = (p.get("abstract", "") or "").lower()
        text = title + " " + abstract

        # 排除非纯镍文献
        if any(t in text for t in exclude):
            continue

        # 必须包含镍
        if "nickel" not in text and " ni " not in text and "pure ni" not in text:
            continue

        # 必须包含退火
        if "anneal" not in text and "heat treatment" not in text and "recrystalliz" not in text:
            continue

        # 必须包含 EBSD 相关
        if not any(t in text for t in ebsd_terms):
            continue

        filtered.append(p)

    return filtered

def save_results(all_papers, filtered):
    # 保存 JSON
    with open(f"{OUTPUT_DIR}/papers_all.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    with open(f"{OUTPUT_DIR}/papers_filtered.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # 保存缓存
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"papers": filtered, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)

    # 导出 DOI 列表
    lines = [
        "# 纯镍退火 EBSD 文献下载清单",
        "# 筛选标准：材质=纯镍，工艺=退火 (800-1400°C)，表征=EBSD",
        "# 下载方法：访问 https://doi.org/DOI 号 通过清华图书馆代理获取 PDF",
        "",
    ]

    for i, p in enumerate(filtered, 1):
        doi = p.get("externalIds", {}).get("DOI", "")
        title = p.get("title", "N/A")[:90]
        year = p.get("year", "?")
        cites = p.get("citationCount", 0) or 0
        oa = " [OA 免费]" if p.get("openAccessPdf") else ""
        journal = p.get("journal", {}).get("name", "") if p.get("journal") else ""

        lines.append(f"[{i:02d}] [{year}] 引用:{cites:4d}{oa}")
        lines.append(f"     标题：{title}")
        if journal:
            lines.append(f"     期刊：{journal}")
        if doi:
            lines.append(f"     DOI: {doi}")
            lines.append(f"     链接：https://doi.org/{doi}")
        lines.append("")

    with open(f"{OUTPUT_DIR}/download_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n导出下载列表：{OUTPUT_DIR}/download_list.txt")

def main():
    print("=" * 70)
    print("纯镍退火 EBSD 文献检索")
    print("筛选标准：纯镍材质 + 退火工艺 (800-1400°C) + EBSD 表征")
    print("=" * 70)

    ensure_dirs()

    # 尝试从缓存加载
    if os.path.exists(CACHE_FILE):
        print(f"\n[提示] 发现缓存文件：{CACHE_FILE}")
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cached_papers = cache.get("papers", [])
            if cached_papers:
                print(f"缓存中有 {len(cached_papers)} 篇文献，使用缓存数据")
                save_results(cached_papers, cached_papers)
                print_results(cached_papers)
                return cached_papers
        except Exception as e:
            print(f"读取缓存失败：{e}")

    all_papers = []
    seen_ids = set()

    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] 执行检索...")
        data = search(q, MAX_RESULTS)

        if data and "data" in data:
            papers = data.get("data", [])
            print(f"  获取到 {len(papers)} 篇论文")

            for p in papers:
                pid = p.get("paperId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
        else:
            print("  未能获取数据，可能由于 API 限制")

        if i < len(QUERIES):
            time.sleep(20)  # 查询间长延迟

    print(f"\n=== 检索完成 ===")
    print(f"原始结果：{len(all_papers)} 篇")

    # 筛选
    filtered = filter_papers(all_papers)
    print(f"筛选后（纯镍 + 退火+EBSD）：{len(filtered)} 篇")

    # 按引用数排序
    filtered.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)

    # 保存
    save_results(all_papers, filtered)

    print_results(filtered)

    return filtered

def print_results(filtered):
    print("\n" + "=" * 70)
    print("=== 符合条件的文献列表 ===")
    print("=" * 70)

    if not filtered:
        print("\n[!] 未找到符合严格筛选条件的文献")
        print("建议：")
        print("  1. 手动访问 Semantic Scholar / Google Scholar 检索")
        print("  2. 使用下载清单中的 DOI 通过图书馆下载 PDF")
        print("  3. 将 PDF 放入 lit_mining/local_pdfs/ 目录后使用 extractor 提取")
    else:
        for i, p in enumerate(filtered[:30], 1):
            title = p.get("title", "N/A")[:70]
            year = p.get("year", "?")
            cites = p.get("citationCount", 0) or 0
            doi = p.get("externalIds", {}).get("DOI", "N/A")
            oa = " [OA]" if p.get("openAccessPdf") else ""

            print(f"\n  {i:2d}. [{year}] (引用:{cites:4d}){oa}")
            print(f"      标题：{title}")
            if doi != "N/A":
                print(f"      DOI: {doi}")

    print(f"\n" + "=" * 70)
    print("=== 数据保存路径 ===")
    print("=" * 70)
    abs_dir = os.path.abspath(OUTPUT_DIR)
    print(f"  - 筛选文献 JSON:  {abs_dir}/papers_filtered.json")
    print(f"  - 原始文献 JSON:  {abs_dir}/papers_all.json")
    print(f"  - DOI 下载列表：   {abs_dir}/download_list.txt")
    print(f"  - 本地 PDF 目录：   {os.path.abspath('lit_mining/local_pdfs')}")
    print(f"\n找到 {len(filtered)} 篇符合要求的文献")

if __name__ == "__main__":
    main()
