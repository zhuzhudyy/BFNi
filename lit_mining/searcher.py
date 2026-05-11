"""
Module A: 文献检索器
- Semantic Scholar API 检索
- CrossRef DOI 补全
- 去重、相关性排序
- 缓存检索结果避免重复请求
"""

import json
import os
import time
import hashlib
from urllib.request import Request, urlopen
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError

from .config import (
    SEARCH_QUERIES,
    SEMANTIC_SCHOLAR_BASE,
    SEMANTIC_SCHOLAR_FIELDS,
    MAX_RESULTS_PER_QUERY,
    MIN_CITATION_COUNT,
    PAPER_CACHE_JSON,
)


def _ensure_cache_dir():
    os.makedirs(os.path.dirname(PAPER_CACHE_JSON), exist_ok=True)


def _load_cache():
    if os.path.exists(PAPER_CACHE_JSON):
        with open(PAPER_CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    _ensure_cache_dir()
    cache["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PAPER_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _http_get(url, retries=4, delay=5.0):
    """带重试的 HTTP GET，针对 429 错误增加更长的等待时间"""
    req = Request(url, headers={"User-Agent": "LitMiningBot/1.5 (academic research)"})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                print(f"      [~] 触发接口限制(429)，自动等待 {delay * (attempt + 2)} 秒后重试...")
                time.sleep(delay * (attempt + 2))
            else:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
        except URLError as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


class LiteratureSearcher:
    """文献检索器：批量检索 → 去重 → 排序 → 缓存"""

    def __init__(self, queries=None, max_per_query=None):
        self.queries = queries or SEARCH_QUERIES
        self.max_per_query = max_per_query or MAX_RESULTS_PER_QUERY
        self.cache = _load_cache()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def search_all(self, use_cache=True):
        """执行所有检索式，返回去重排序后的论文列表"""
        all_papers = []
        seen_ids = set()

        for query in self.queries:
            results = self._search_one(query, use_cache)
            for paper in results:
                pid = paper.get("paperId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(paper)
            # 延长单次请求间隔，防止被封 IP
            time.sleep(5.0)  

        all_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
        return all_papers

    def filter_relevant(self, papers, min_citations=None):
        """根据引用数过滤 + 标题关键词二次筛选"""
        min_cit = min_citations if min_citations is not None else MIN_CITATION_COUNT
        filtered = [p for p in papers if (p.get("citationCount") or 0) >= min_cit]
        return sorted(filtered, key=lambda p: p.get("citationCount", 0), reverse=True)

    def save_cache(self):
        """Save search results to JSON cache"""
        _save_cache(self.cache)

    def export_doi_list(self, papers, output_path="download_list.txt"):
        """
        导出论文 DOI 列表 + 标题，方便通过清华图书馆代理手动下载 PDF。
        格式：每行一个 DOI，可直接粘贴到 publisher 网站搜索。
        """
        lines = []
        lines.append("# 文献下载清单 — 通过清华图书馆代理逐篇下载 PDF")
        lines.append("# 访问 https://doi.org/DOI号 即可跳转到出版社页面")
        lines.append("# 将下载的 PDF 放入 lit_mining/local_pdfs/ 文件夹\n")
        for i, p in enumerate(papers, 1):
            doi = p.get("externalIds", {}).get("DOI", "")
            title = p.get("title", "N/A")[:100]
            year = p.get("year", "?")
            cites = p.get("citationCount", 0)
            oa = " [免费OA]" if p.get("openAccessPdf") else ""
            lines.append(f"{'OA' if p.get('openAccessPdf') else '需下载'} | [{year}] cites={cites} | {title}")
            if doi:
                lines.append(f"  https://doi.org/{doi}")
            lines.append("")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        oa_count = sum(1 for p in papers if p.get("openAccessPdf"))
        print(f"[*] 下载清单已导出: {output_path}")
        print(f"    共 {len(papers)} 篇 ({oa_count} 篇OA免费, {len(papers)-oa_count} 篇需通过图书馆下载)")
        print(f"    将下载的PDF放入: lit_mining/local_pdfs/ 文件夹")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _search_one(self, query, use_cache):
        cache_key = _query_cache_key(query)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        url = (
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search?"
            f"query={quote(query)}&limit={self.max_per_query}"
            f"&fields={SEMANTIC_SCHOLAR_FIELDS}"
        )
        try:
            data = _http_get(url)
            papers = data.get("data", [])
            self.cache[cache_key] = papers
            return papers
        except Exception as e:
            print(f"  [!] 检索失败 '{query[:40]}...': {e}")
            return []

    def _enrich_with_crossref(self, dois):
        """通过 Crossref API 补全元数据（DOI → abstract/publisher）"""
        enriched = {}
        for doi in dois:
            if not doi:
                continue
            try:
                url = f"https://api.crossref.org/works/{quote(doi)}"
                data = _http_get(url)
                msg = data.get("message", {})
                enriched[doi] = {
                    "abstract": msg.get("abstract", ""),
                    "publisher": msg.get("publisher", ""),
                    "type": msg.get("type", ""),
                }
            except Exception:
                enriched[doi] = {}
            time.sleep(0.5)
        return enriched


def _query_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()[:16]


# ------------------------------------------------------------------
# 便捷函数
# ------------------------------------------------------------------
def search_nickel_papers(max_papers=50):
    """一步式检索镍相关论文"""
    searcher = LiteratureSearcher()
    papers = searcher.search_all()
    papers = searcher.filter_relevant(papers)
    searcher.save_cache()
    return papers[:max_papers]


def _ensure_local_pdf_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_pdfs")
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    print("=" * 60)
    print("文献检索器自检")
    print("=" * 60)
    s = LiteratureSearcher()
    papers = s.search_all()
    papers = s.filter_relevant(papers)
    s.save_cache()
    s.export_doi_list(papers)

    print(f"\n检索到 {len(papers)} 篇去重论文")
    for i, p in enumerate(papers[:15], 1):
        title = p.get("title", "N/A")[:80]
        year = p.get("year", "?")
        cites = p.get("citationCount", 0)
        oa = " [OA]" if p.get("openAccessPdf") else ""
        print(f"  {i:2d}. [{year}] ({cites:4d} cites){oa} {title}")