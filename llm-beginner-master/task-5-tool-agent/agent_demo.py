"""task-5 · 跑单个/多个 ReAct 任务并打印完整 trace + 命中判定。

用法：
    python agent_demo.py --task "计算 (123+456)*789 的结果，并告诉我结果的位数。"
    python agent_demo.py --ids 1 3 5          # 跑 data/tasks.json 里这几题
    python agent_demo.py --all                 # 全部 10 题（约等于 eval 的 M4 但带全文 trace）
    python agent_demo.py --no-verbose          # 不逐 step 打印，只看汇总
"""
import argparse
import json
import re
from pathlib import Path

from src.agent import ReActAgent


def normalize_answer(text):
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def answer_matches(answer, expected_keywords):
    """与 eval/run.py 同口径。"""
    norm_answer = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(keyword) in norm_answer
                       for keyword in expected):
                return False
        elif normalize_answer(expected) not in norm_answer:
            return False
    return True


def load_tasks():
    p = Path(__file__).parent / "data" / "tasks.json"
    if not p.exists():
        print("[提示] data/tasks.json 不存在，先运行：python data/download.py")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="直接给一个任务文本")
    ap.add_argument("--ids", nargs="+", type=int, default=None,
                    help="跑 tasks.json 里的指定题号")
    ap.add_argument("--all", action="store_true", help="跑全部题")
    ap.add_argument("--no-verbose", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    tasks = load_tasks() if (args.ids or args.all) else []
    if args.task:
        queue = [{"id": 0, "task": args.task, "expected_answer_contains": [],
                  "expected_tools": []}]
    elif args.ids:
        by_id = {t["id"]: t for t in tasks}
        queue = [by_id[i] for i in args.ids if i in by_id]
    elif args.all:
        queue = tasks
    else:
        queue = [tasks[0]] if tasks else []
    if not queue:
        print("没有要跑的任务。")
        return

    kw = {"verbose": not args.no_verbose}
    if args.max_steps:
        kw["max_steps"] = args.max_steps
    agent = ReActAgent(**kw)

    n_pass = 0
    for t in queue:
        print("\n" + "#" * 72)
        print(f"题目 {t['id']}：{t['task']}")
        trace = agent.run(t["task"])
        print("\n---- 完整 trace（Thought / Action / Input / Observation）----")
        for s in trace["steps"]:
            if s.get("thought"):
                print(f"Thought: {s['thought']}")
            if s.get("tool"):
                print(f"Action: {s['tool']}")
                print(f"Action Input: {json.dumps(s.get('action_input'), ensure_ascii=False)}")
                print(f"Observation: {s.get('observation')}")
            elif s.get("final_answer") is not None:
                print(f"Final Answer: {s['final_answer']}")
            if s.get("note"):
                print(f"[note] {s['note']}")
        ok = answer_matches(trace["final_answer"], t["expected_answer_contains"])
        n_pass += int(ok)
        print(f"\n>> final_answer: {trace['final_answer']!r}")
        print(f">> 自报 success={trace['success']} | used_tools={trace['used_tools']}"
              f" | 关键词命中={ok}")

    if tasks:
        print(f"\n命中 {n_pass}/{len(queue)} 题（>60% 为过线）")


if __name__ == "__main__":
    main()
