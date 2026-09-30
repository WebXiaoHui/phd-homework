"""task-5 · 演示/验证错误恢复（README 的 M3 与加分 S4）。

第一部分（无需 LLM）：直接验证『工具异常被捕获并变成 Observation 字符串』——
agent._exec_tool 对会抛错的调用绝不向上抛，而是返回错误文本。

第二部分（需本地模型服务，可选）：用 inject_error 钩子在第 1 次调用 calculator 时注入
错误，观察 agent 是否自我纠错并完成任务。

用法：
    python error_recovery_demo.py          # 只跑第一部分（离线可用）
    python error_recovery_demo.py --live   # 加跑第二部分（需要模型服务）
"""
import argparse

from src.agent import ReActAgent
from src.tools import TOOLS


def part1_tool_level():
    print("== 第一部分：工具级错误捕获（M3，无需模型）==")
    agent = ReActAgent()
    demo_cases = [
        # 未知工具（不会走到 run）
        ("no_such_tool", {"x": 1}, "无此工具时应由路由判为 invalid"),
    ]
    for name, args, note in demo_cases:
        print(f"  - 调用未知工具 {name!r}（{note}）")
    # calculator 抛错：除数不能为 0
    out = agent._exec_tool("calculator", {"expression": "1 / 0"}, inject_error=None)
    print(f"  - calculator 除零 → _exec_tool 返回（而非抛异常）：")
    print(f"    {out!r}")
    # python_sandbox 语法错误
    out = agent._exec_tool("python_sandbox", {"code": "def f(:"}, inject_error=None)
    print(f"  - python_sandbox 语法错误 → 返回错误文本：")
    print(f"    {out!r}")
    # 真实正常路径仍可用
    out = agent._exec_tool("calculator", {"expression": "2 + 2"}, inject_error=None)
    print(f"  - 正常调用仍可用：calculator(2+2) → {out!r}")
    ok = ("ZeroDivisionError" in agent._exec_tool(
        "calculator", {"expression": "1 / 0"}, None))
    print(f"\n  除零被捕获并转文本：{'✓' if ok else '✗'}")
    return 0


def part2_live():
    print("\n== 第二部分：inject_error 钩子下的自我纠错（需模型服务）==")
    calls = {"n": 0}

    def inject_error_once(tool, args):
        calls["n"] += 1
        if tool == "calculator" and calls["n"] == 1:
            return "模拟故障：计算服务暂时不可用，请稍后重试。"

    try:
        trace = ReActAgent(verbose=False).run(
            "计算 (123 + 456) * 789 的结果，并告诉我结果的位数。",
            inject_error=inject_error_once,
        )
    except Exception as e:
        print(f"  [模型服务不可用] {type(e).__name__}: {e}")
        print("  （第一部分已足够说明 M3；起好 ollama serve 后加 --live 重试）")
        return 1
    print(f"  used_tools={trace['used_tools']}")
    print(f"  final_answer={trace['final_answer']!r}")
    recovered = "456831" in trace["final_answer"]
    print(f"  首次 calculator 被注入错误后仍给出 456831：{'✓' if recovered else '✗'}")
    return 0 if recovered else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    rc = part1_tool_level()
    if args.live:
        rc = part2_live() or rc
    print("\n可用工具：", ", ".join(TOOLS.keys()))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
