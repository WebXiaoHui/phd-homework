"""src/subagents —— 独立 context 的子 agent（Subagents 层，顶层）。

要点（README 第 5 周）：
- 每个 SubAgent 持有**自己的** message 列表 / 步数上限 / 工具子集，与主 agent 隔离；
- 主 agent 只拿到子 agent 返回的**摘要**，不吞全部 trace；
- 两件内建：explore（代码搜索/定位）、test_repro（复现并归纳测试失败）。

默认只读（read/grep/test/只读 run_command），改动仍由主 agent 落盘。
"""
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许直接以脚本方式调用
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.toolkit import TOOL_SCHEMAS, ToolServer  # noqa: E402


def default_llm_cfg():
    return {
        "base_url": os.environ.get("CODING_AGENT_LLM_BASE_URL")
        or os.environ.get("AGENT_LLM_BASE_URL") or "http://localhost:11434/v1",
        "model": os.environ.get("CODING_AGENT_LLM_MODEL")
        or "qwen2.5-coder:7b-instruct",
        "api_key": os.environ.get("CODING_AGENT_LLM_API_KEY") or "ollama",
    }


def _chat(cfg: dict, messages: list, max_tokens: int = 500) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    resp = client.chat.completions.create(model=cfg["model"], messages=messages,
                                          temperature=0.0, max_tokens=max_tokens)
    return (resp.choices[0].message.content or "").strip()


def _tools_summary(*names):
    by = {s["name"]: s for s in TOOL_SCHEMAS}
    return "; ".join(f"{n}: {by[n]['description']}" for n in names)


_SUBAGENT_PROMPTS = {
    "explore": (
        "你是一个代码检索子 agent。主 agent 给你一个问题，你用工具在仓库里定位答案。\n"
        "每轮输出：\nThought: 说明\nAction: 工具名\n"
        'Action Input: {"参数": "值"}\n'
        "可选工具：" + _tools_summary("list_files", "grep_search", "read_file",
                                       "run_command") + "\n"
        "信息足够后输出：\nFinal: 给主 agent 的简短结论（几行，含文件路径与关键代码）。"
    ),
    "test_repro": (
        "你是一个测试诊断子 agent。主 agent 让你复现并归纳测试失败。\n"
        "每轮输出：\nThought: 说明\nAction: 工具名\n"
        'Action Input: {"参数": "值"}\n'
        "可选工具：" + _tools_summary("run_tests", "read_file", "grep_search",
                                       "run_command") + "\n"
        "请跑一次测试并读相关源码，最后输出：\n"
        "Final: 失败用例清单 + 每条的断言期望/实际 + 最可能的代码缺陷位置。不要改任何文件。"
    ),
}


def _action_tags(text: str):
    """解析子 agent 的 Action/Final，返回 (kind, payload)。"""
    text = (text or "").strip()
    fi = text.find("Final:")
    ai = text.find("Action:")
    if fi != -1 and (ai == -1 or fi < ai):
        return "final", text[fi + len("Final:"):].strip()
    if ai == -1:
        return "final", text  # 没标签就当结论
    m = re.search(r"Action\s*[:：]\s*([A-Za-z_][\w]*)", text)
    if not m:
        return "retry", "缺少工具名。"
    name = m.group(1)
    im = re.search(r"Action\s+Input\s*[:：]\s*(\{.*\})", text)
    if not im:
        return "retry", f"缺少 Action Input（{name}）。"
    raw = im.group(1)
    try:
        args = json.loads(raw)
    except Exception:
        args = None
    if not isinstance(args, dict):
        return "retry", f"Action Input 不是合法 JSON：{raw!r}"
    return "action", (name, args)


class SubAgent:
    """自带 message 列表/工具子集/步数上限的隔离循环。"""

    def __init__(self, kind: str, cfg: dict = None, max_steps: int = 8):
        if kind not in _SUBAGENT_PROMPTS:
            raise ValueError(f"未知子 agent：{kind}（可选 {sorted(_SUBAGENT_PROMPTS)}）")
        self.kind = kind
        self.cfg = cfg or default_llm_cfg()
        self.max_steps = max_steps

    def run(self, server: ToolServer, task: str, verbose: bool = False) -> str:
        messages = [{"role": "system", "content": _SUBAGENT_PROMPTS[self.kind]},
                    {"role": "user", "content": f"子任务：{task}"}]
        for step in range(1, self.max_steps + 1):
            text = _chat(self.cfg, messages, max_tokens=400)
            messages.append({"role": "assistant", "content": text})
            kind, payload = _action_tags(text)
            if kind == "final":
                return str(payload)
            if kind == "action":
                name, args = payload
                obs = server.call(name, args)
            else:
                obs = f"解析失败：{payload}。请重新按格式输出一轮。"
            if verbose:
                print(f"[subagent:{self.kind}:{step}] {text[:200]}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        return "（子 agent 达到步数上限，未给出结论）"


def make_explorer(cfg: dict = None, max_steps: int = 8) -> SubAgent:
    return SubAgent("explore", cfg, max_steps)


def make_test_repro(cfg: dict = None, max_steps: int = 6) -> SubAgent:
    return SubAgent("test_repro", cfg, max_steps)


__all__ = ["SubAgent", "make_explorer", "make_test_repro",
           "default_llm_cfg", "ToolServer"]
