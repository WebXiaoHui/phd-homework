"""task-4 · 单独跑一遍 gold 召回指标，便于报告贴表格/截图。

与 eval/run.py 的 nndl_gold_recall_at_10 同口径：gold chunk（去掉所有空白）
文本命中任一 gold_anchor 即算该题在对应 rank 命中；输出 Recall@1/3/5/10 与 MRR。

用法：python recall_report.py [--topk 20]
"""
import argparse
import json
import re
from pathlib import Path

from src.retriever import Retriever
from src.paths import GOLD_QA


def norm(t):
    return re.sub(r"\s+", "", str(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    items = [json.loads(line) for line in GOLD_QA.open(encoding="utf-8")
             if line.strip()]
    retriever = Retriever()
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = []
    miss = []
    for it in items:
        anchors = [norm(a) for a in it.get("gold_anchors", [])]
        results = retriever.retrieve(it["question"], k=args.topk)
        rank = None
        matched = None
        for i, r in enumerate(results, 1):
            t = norm(r.get("text", ""))
            for a in anchors:
                if a and a in t:
                    rank, matched = i, a
                    break
            if rank:
                break
        if rank:
            for k in hits:
                if rank <= k:
                    hits[k] += 1
            rr.append(1 / rank)
        else:
            rr.append(0)
            miss.append(it.get("id"))
    n = len(items)
    print(f"\ngold QA: {n} 条  (Retriever k={args.topk})")
    for k in sorted(hits):
        print(f"  Recall@{k:<2}= {hits[k] / n:.3f}  ({hits[k]}/{n})")
    print(f"  MRR       = {sum(rr) / n:.3f}")
    if miss:
        print("  未命中 id：", miss)
    print("（口径：召回文本去空白后命中任一 gold_anchor 即记该题命中）")


if __name__ == "__main__":
    main()
