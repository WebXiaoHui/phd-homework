"""task-4 · 交互/批量问几个问题，看 RAG 答案 + 引用来源。

用法：
    python query_demo.py                          # 内置几个问题
    python query_demo.py --query "过拟合怎么办？"
    RAG_OFFLINE=1 python query_demo.py            # 不起 LLM，只验证检索管道
    python query_demo.py --questions q.jsonl      # 每行 {"question": "..."}
"""
import argparse
import json
from pathlib import Path

from src.rag import answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--questions", default=None, help="jsonl，每行 {question}")
    ap.add_argument("--k-final", type=int, default=5)
    args = ap.parse_args()

    queries = list(args.query)
    if args.questions:
        for line in Path(args.questions).open(encoding="utf-8"):
            line = line.strip()
            if line:
                queries.append(json.loads(line)["question"])
    if not queries:
        queries = ["什么是反向传播？请结合资料回答。",
                   "训练时为什么要划分训练集、验证集和测试集？",
                   "解释一下 dropout 的作用。"]

    for q in queries:
        print("\n" + "=" * 72)
        print("问题：", q)
        try:
            r = answer(q, k_final=args.k_final)
        except Exception as e:
            print(f"[错误] {e}")
            continue
        print("回答：", r["answer"])
        print("来源：")
        for s in r["sources"]:
            sc = s.get("score")
            print(f"  - {s.get('source')}  score={sc if sc is None else round(sc,3)}")
            print(f"    {s.get('text', '')[:80]}…")


if __name__ == "__main__":
    main()
