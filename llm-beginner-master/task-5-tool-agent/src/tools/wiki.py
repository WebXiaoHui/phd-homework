"""wiki 工具：维基百科 MediaWiki API 查询（中英文）。

- 查询含 CJK 字符 → zh.wikipedia.org，否则 en.wikipedia.org；
- 用 generator=search 找最相关条目，再取该条目简介（prop=extracts&exintro&explaintext）；
- 返回：条目标题 + 导言摘要（供下游提取年份/人物/事实）。

依赖网络；离线/被墙时会抛异常——自检对 wiki 的异常按「跳过」处理，不影响其余工具。
"""
import os
import re

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科（自动中英文）并返回相关条目 + 导言摘要。"
            "适合查人物出生年/身份、概念定义、论文/模型提出年份等事实。"
            "中文问题可直接用中文查询词，例如 '图灵机'、'反向传播'；"
            "英文条目也可用英文，如 'Alan Turing'、'Geoffrey Hinton'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查询的条目/关键词，如 '图灵机' 或 'Transformer (machine learning model)'。",
                }
            },
            "required": ["query"],
        },
    },
}

TIMEOUT_SECONDS = 15
_USER_AGENT = "llm-beginner-agent/1.0 (educational task-5; contact: student)"

_CJK_RE = re.compile("[㐀-鿿]")

# 端点可用环境变量覆盖：被墙/需要代理时把 wiki 指向可达的镜像（如带代理网关）。
_ENDPOINT_ENV = {
    "zh": os.environ.get("WIKI_API_ZH", "https://zh.wikipedia.org/w/api.php"),
    "en": os.environ.get("WIKI_API_EN", "https://en.wikipedia.org/w/api.php"),
}


def _api_url(query: str) -> str:
    return (_ENDPOINT_ENV["zh"] if _CJK_RE.search(query)
            else _ENDPOINT_ENV["en"])


def _fetch(url, params):
    resp = requests.get(
        url, params=params, timeout=TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
    )
    resp.raise_for_status()
    return resp.json()


def _search_page(query: str, lang: str):
    """先搜索拿到最相关标题，再取简介。"""
    api = _api_url(query)
    # 第一步：搜标题
    data = _fetch(api, {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 1, "format": "json", "formatversion": 2,
    })
    hits = ((data.get("query") or {}).get("search") or [])
    if not hits:
        raise ValueError(f"在 {lang} 维基百科没有搜到与 {query!r} 相关的条目，请换关键词再试。")
    title = hits[0]["title"]
    # 第二步：取该条目导言
    data = _fetch(api, {
        "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
        "redirects": 1, "titles": title, "exchars": 3000, "format": "json",
        "formatversion": 2,
    })
    pages = (data.get("query") or {}).get("pages") or []
    extract = ""
    if pages:
        extract = (pages[0].get("extract") or "").strip()
    return title, extract


def run(args):
    if not isinstance(args, dict) or "query" not in args:
        raise ValueError("wiki 需要参数 query（str）")
    query = str(args["query"]).strip()
    if not query:
        raise ValueError("query 不能为空")

    try:
        title, extract = _search_page(query, "zh" if _CJK_RE.search(query) else "en")
    except ValueError:
        raise  # 语义性没结果，直接让 agent 换词
    except Exception as e:
        raise RuntimeError(f"访问维基百科失败（请检查网络）：{type(e).__name__}: {e}") from e

    if not extract:
        return f"条目：{title}\n（该页没有可用的文本简介，请换一个更具体的查询词。）"
    return f"条目：{title}\n导言摘要：{extract[:2500]}"


if __name__ == "__main__":
    for q in ["Alan Turing", "Geoffrey Hinton", "图灵机"]:
        print("=" * 60)
        print("Q:", q)
        try:
            print(run({"query": q}))
        except Exception as e:
            print(f"[离线?] {e}")
