"""task-4 · RAG 端到端：检索 -> (rerank) -> 拼 prompt -> Qwen 生成。

对外契约（README / eval/run.py）：
    answer(query: str) -> dict(answer: str, sources: List[dict])

answer 返回 dict，answer 非空字符串、sources 为列表——自检会 `bool(...)`
强转，结构不对整条自检会在写 result.json 前崩（README 专门强调）。
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from src.generator import Generator
from src.paths import (DEFAULT_TOP_K_FINAL, DEFAULT_TOP_K_RECALL,
                       INDEX_DIR, RERANK_MODEL_DIR)
from src.retriever import Retriever

_SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能依据下面【资料】中给出的内容回答用户问题；"
    "若资料不足以回答，请明确回答“资料中未找到相关信息”，不要使用资料之外的"
    "知识编造。回答可以引用资料来源，例如“（p.3）”。")

# 进程级单例，多次调用不重复加载模型
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(INDEX_DIR)
    return _retriever


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _dedupe(docs: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for d in docs:                       # 召回片段重叠，去重后按原序保留
        key = _norm(d.get("text", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _truncate_contexts(docs: List[dict], max_chars: int = 6000) -> List[dict]:
    total, out = 0, []
    for d in docs:
        text = d.get("text", "")
        if total + len(text) > max_chars and out:
            break                        # 留够关键上下文，超长则丢弃尾部
        total += len(text)
        out.append(d)
    return out


def _build_prompt(query: str, contexts: List[dict]) -> List[dict]:
    lines = [f"[{i}]（来源 {c.get('source', '?')}）{c.get('text', '')}"
             for i, c in enumerate(contexts, 1)]
    body = "\n\n".join(lines)
    user = (f"【资料】\n{body}\n\n【问题】\n{query}\n\n"
            f"请基于资料回答。若资料不含答案，请直接说明“资料中未找到相关信息”。")
    return [{"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _maybe_rerank(query: str, docs: List[dict], top_k: int) -> List[dict]:
    if top_k >= len(docs):
        return docs[:top_k]
    if not RERANK_MODEL_DIR.exists():
        print("[rag] 未装 reranker 模型，跳过精排（装了 BAAI/bge-reranker-base 自动启用）")
        return docs[:top_k]
    from src.reranker import Reranker
    try:
        return Reranker().rerank(query, docs, top_k=top_k)
    except Exception as e:               # rerank 失败不影响主线
        print(f"[rag] rerank 失败，退回直接截断：{e}")
        return docs[:top_k]


def retrieve_contexts(query: str, k_recall: int = DEFAULT_TOP_K_RECALL,
                      k_final: int = DEFAULT_TOP_K_FINAL) -> List[dict]:
    """召回 -> 去重 -> （可选 rerank）-> 取前 k_final 作为上下文。"""
    retriever = get_retriever()
    hits = retriever.retrieve(query, k=k_recall)
    hits = _dedupe(hits)
    hits = _maybe_rerank(query, hits, top_k=k_final)
    return hits


def answer(query: str, k_recall: int = DEFAULT_TOP_K_RECALL,
           k_final: int = DEFAULT_TOP_K_FINAL,
           generator: Optional[Generator] = None) -> dict:
    """RAG 端到端：返回 {answer: str, sources: [text/score/source/page, ...]}。"""
    contexts = retrieve_contexts(query, k_recall=k_recall, k_final=k_final)
    contexts = _truncate_contexts(contexts)

    if not contexts:
        return {"answer": "资料库中未检索到相关内容，无法回答。",
                "sources": []}

    # 离线模式：不连生成服务，直接从最相关片段截一段（仅用于管道冒烟测试）
    if os.environ.get("RAG_OFFLINE") == "1":
        return {"answer": "（RAG_OFFLINE=1 离线摘录）" + contexts[0]["text"],
                "sources": contexts}

    if generator is None:
        generator = Generator()
    messages = _build_prompt(query, contexts)
    ans = generator.chat(messages)
    return {"answer": ans, "sources": contexts}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="什么是反向传播？请结合资料回答。")
    ap.add_argument("--k-recall", type=int, default=DEFAULT_TOP_K_RECALL)
    ap.add_argument("--k-final", type=int, default=DEFAULT_TOP_K_FINAL)
    args = ap.parse_args()
    r = answer(args.query, k_recall=args.k_recall, k_final=args.k_final)
    print("\n[回答]", r["answer"])
    print("\n[来源]")
    for s in r["sources"]:
        print(f"  - {s.get('source')}  score={s.get('score', float('nan')):.3f}")
