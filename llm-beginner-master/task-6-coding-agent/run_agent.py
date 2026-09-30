"""task-6 · CodingAgent CLI：在某个仓库上跑一个 issue，打印结果/写 trace。

用法：
    python run_agent.py                                  # data/toy-repo + ISSUE.md
    python run_agent.py --repo data/toy-repo --issue-file data/toy-repo/ISSUE.md
    python run_agent.py --issue-text "修复 add，让测试通过"
    python run_agent.py --with-subagents --verbose --json-out trace.json
"""
import argparse
import json
from pathlib import Path

from src.agent import CodingAgent

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "data" / "toy-repo"))
    ap.add_argument("--issue-file", default=None)
    ap.add_argument("--issue-text", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--with-subagents", action="store_true")
    ap.add_argument("--no-skills", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", default=None, help="把 trace 写成 json")
    args = ap.parse_args()

    repo = Path(args.repo)
    if args.issue_text is not None:
        issue = args.issue_text
    elif args.issue_file:
        issue = Path(args.issue_file).read_text(encoding="utf-8")
    else:
        issue = (repo / "ISSUE.md").read_text(encoding="utf-8")

    kwargs = dict(verbose=args.verbose,
                  use_subagents=args.with_subagents,
                  use_skills=not args.no_skills)
    if args.max_steps:
        kwargs["max_steps"] = args.max_steps
    agent = CodingAgent(**kwargs)

    print(f"== 仓库：{repo}")
    print(f"== issue：{issue[:200]}")
    trace = agent.run(str(repo), issue)

    print("\n== 终验 pytest：", trace["final_pytest"][:120])
    print("== tests_passed：", trace["tests_passed"])
    print("== 步数：", trace["num_steps"], "| done：", trace["done"])
    print("\n== patch（前 600 字）==\n", (trace["patch"] or "")[:600])

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(trace, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
        print(f"\n已写 trace → {args.json_out}")
    return 0 if trace["tests_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
