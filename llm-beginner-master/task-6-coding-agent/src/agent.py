"""src/agent.py —— CodingAgent：把 Tools / Skills / Subagents 三层串成一条 agentic loop（M3/M4）。

循环：while not done: model → (Action | Subagent | DONE) → tool observation → loop
- 工具经 ToolServer 进程内调用（与 MCP server 同源实现，见 src/toolkit.py）；
- Skills 按 task 渐进式披露：issue 提到"测试/pytest"会命中 test-runner，把工作流注入 system；
- 可选派发 Subagent（独立 context），默认关闭以保证 toy-repo 路径稳定；
- done 信号：模型显式输出 DONE / Final Answer，或步数耗尽后我们用 pytest 终验一次；
- Trace 每步记录 thought/tool_call/observation，并含 patch（git diff 或 .orig 对比）与 tests_passed。
"""
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 python src/agent.py 直接跑
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.toolkit import TOOLS, ToolServer, list_tools  # noqa: E402
from src.skill_loader import SkillLoader  # noqa: E402

# 默认 LLM 配置（OpenAI 兼容本地端点）
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"
DEFAULT_API_KEY = "ollama"

_RE_THOUGHT = re.compile(r"Thought\s*[:：]\s*(.*)")


def _llm_cfg():
    return {
        "base_url": os.environ.get("CODING_AGENT_LLM_BASE_URL")
        or os.environ.get("AGENT_LLM_BASE_URL") or DEFAULT_BASE_URL,
        "model": os.environ.get("CODING_AGENT_LLM_MODEL") or DEFAULT_MODEL,
        "api_key": os.environ.get("CODING_AGENT_LLM_API_KEY") or DEFAULT_API_KEY,
    }


# ---------------------------------------------------------------- 解析

def _tool_docs() -> str:
    lines = []
    for s in list_tools():
        props = s["input_schema"]["properties"]
        keys = ", ".join(f"{k}={p.get('type', '?')}" for k, p in props.items())
        lines.append(f"- {s['name']}：{s['description']}   参数：{{{keys}}}")
    return "\n".join(lines)


def _detect_done(text: str):
    """返回 (is_done, 收尾文本)。"""
    text = text or ""
    ai = text.find("Action:")
    # 行首的 DONE / Final Answer:
    m = re.search(r"(?im)^(?:DONE|Final\s*Answer\s*[:：]?\s*(.*))$", text)
    if m and (m.group(1) is not None or m.group(0).strip().upper() == "DONE"):
        # Final Answer 同行内容 / DONE
        tail = (m.group(1) or "").strip() if m.group(1) is not None else ""
        return True, tail or m.group(0).strip()
    # 兜底：Final Answer 出现在任何 Action 之前
    fi = text.find("Final Answer")
    if fi != -1 and (ai == -1 or fi < ai):
        return True, text[fi + len("Final Answer"):].lstrip(" ：:").strip()
    # DONE 出现在中间但没有任何 Action（说明已经收尾）
    if ai == -1 and re.search(r"(?i)\bDONE\b", text):
        return True, text
    return False, ""


def _parse_action(text: str):
    """返回 ("action", name, args) 或 ("invalid", reason)。"""
    text = text or ""
    m = re.search(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if not m:
        return "invalid", "没有解析到 Action: <工具名>，请按格式输出。"
    name = m.group(1)
    if name not in TOOLS:
        return "invalid", (f"未知工具 {name!r}。可用：{', '.join(TOOLS)}。"
                           f" 如需结束请输出 DONE。")
    im = re.search(r"Action\s+Input\s*[:：]\s*(\{.*\})", text)
    if not im:
        return "invalid", f"Action: {name} 缺少 Action Input（单行 JSON）。"
    raw = im.group(1)
    for cand in (raw, raw.replace("'", '"')):
        try:
            obj = json.loads(cand)
            break
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return "invalid", f"Action Input 不是合法 JSON：{raw!r}。"
    return "action", (name, obj)


def _compute_patch(repo: Path) -> str:
    """优先 git diff；无 git/无改动退回 calculator.py.orig 对比。"""
    try:
        proc = subprocess.run(["git", "-C", str(repo), "diff", "--"],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and (proc.stdout or "").strip():
            return proc.stdout.strip()
    except Exception:
        pass
    orig, cur = repo / "calculator.py.orig", repo / "calculator.py"
    if orig.exists() and cur.exists():
        a = orig.read_text(encoding="utf-8").splitlines(keepends=True)
        b = cur.read_text(encoding="utf-8").splitlines(keepends=True)
        if a != b:
            return "".join(difflib.unified_diff(
                a, b, fromfile="a/calculator.py", tofile="b/calculator.py")).strip()
    return ""


# ---------------------------------------------------------------- CodingAgent

class CodingAgent:
    def __init__(self, cfg: dict = None, *, use_skills: bool = True,
                 use_subagents: bool = False, max_steps: int = 14,
                 skills_dir: str = None, verbose: bool = False,
                 compact_at_chars: int = 28_000):
        self.cfg = cfg or _llm_cfg()
        self.use_skills = use_skills
        self.use_subagents = use_subagents
        self.max_steps = max_steps
        self.verbose = verbose
        self.compact_at_chars = compact_at_chars
        root = Path(__file__).resolve().parents[1]
        self.skills_dir = skills_dir or str(root / "src" / "skills")
        self._client = None  # 懒加载，构造不联网

    # -- LLM ------------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "缺少 openai 客户端：pip install 'openai>=1.30'。"
                    "同时确认本地模型服务已起（ollama serve / vLLM）。") from e
            self._client = OpenAI(base_url=self.cfg["base_url"],
                                  api_key=self.cfg["api_key"])
        return self._client

    def _chat(self, messages, max_tokens=600) -> str:
        resp = self._ensure_client().chat.completions.create(
            model=self.cfg["model"], messages=messages, temperature=0.0,
            max_tokens=max_tokens, stream=False)
        return (resp.choices[0].message.content or "").strip()

    def _compact(self, messages) -> list:
        """简易 context 压缩：只留 system + 任务 + 最近一轮对话，其余压成摘要。"""
        if messages[0]["role"] == "system":
            system, rest = messages[0], messages[1:]
        else:
            system, rest = None, messages
        # 保留最后一对 assistant+user
        kept = rest[-2:] if len(rest) >= 2 else rest
        dropped = rest[:-2] if len(rest) >= 2 else []
        if not dropped:
            return messages
        digest = ("（较早的对话已压缩。此前动作摘要："
                  + "; ".join(
                      f"调用了 {m['content'][:80].splitlines()[0][:80]!r}"
                      for m in dropped if m["role"] == "assistant")
                  + "）")
        out = ([system] if system else []) + [
            {"role": "user", "content": digest}] + kept
        return out

    # -- 主循环 ----------------------------------------------------------

    def run(self, repo_path: str, issue: str) -> dict:
        """在 repo_path 上解决 issue，返回 Trace（dict：steps/patch/tests_passed/...）。"""
        repo = Path(repo_path)
        server = ToolServer(repo)
        self._ensure_client()

        system_parts = [
            "你是一个能读代码、改代码、跑测试的编程 agent（mini Claude Code 的本地版）。",
            "循环按下面格式作答，每轮只做一个动作：",
            "Thought: 一句话说明你这一步在做什么",
            "Action: 工具名",
            'Action Input: {"参数": "值"}',
            "你会收到 Observation: 工具输出/错误。信息足够、且 `run_tests` 全绿后，",
            "输出单独一行 DONE（或在末尾写 Final Answer: …）结束。",
            "",
            "可用工具：",
            _tool_docs(),
            "",
            "规则：",
            "1. 先侦查再动手：用 read_file / grep_search / list_files 定位，别凭空改写。",
            "2. 改动尽量最小：优先 edit_file 做精确替换；改完必须 read_file 复核一次。",
            "3. 禁止改 test_* 测试文件与 *.orig 快照（工具会拒绝）。",
            "4. 不要编造内容；不确定文件内容就先 read。",
            "5. 工具报错是正常反馈：读 Observation 修正参数重试，最多约 2 次再换思路。",
            "6. 只有 run_tests 返回 ✅ pytest 通过（退出码 0）才算完成，再输出 DONE。",
        ]
        if self.use_subagents:
            system_parts += [
                "",
                "也可以把子任务交给独立子 agent（自己的 context）：",
                "Subagent: explore|test_repro",
                "Subagent Task: 给它的问题描述",
            ]
        sys_msg = "\n".join(system_parts)

        # Skills 渐进式披露：命中就注入完整正文
        if self.use_skills:
            loader = SkillLoader(self.skills_dir)
            name, body = loader.load_matching(issue)
            if name:
                sys_msg += f"\n\n[已加载 Skill：{name}]\n{body}"

        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"任务/问题：{issue}"}]
        steps = []
        done_text = ""

        for n in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n──── step {n} ────")
            text = self._chat(messages)
            messages.append({"role": "assistant", "content": text})
            if self.verbose:
                print("[assistant]", text[:400])

            # 1) 收尾信号
            is_done, done_text = _detect_done(text)
            if is_done:
                steps.append({"step": n, "thought": _thought(text), "tool": None,
                              "tool_input": None, "observation": None,
                              "done": True, "final_message": done_text or text})
                break

            # 2) 可选：派发子 agent
            if self.use_subagents:
                sm = re.search(
                    r"Subagent\s*[:：]\s*(explore|test_repro)\s*\n\s*"
                    r"Subagent\s+Task\s*[:：]\s*(.*)", text, re.IGNORECASE)
                if sm:
                    from src.subagents import SubAgent
                    kind, task = sm.group(1).lower(), sm.group(2).strip()
                    try:
                        summary = SubAgent(kind, self.cfg,
                                           max_steps=8).run(server, task)
                    except Exception as e:
                        summary = f"[子 agent 失败] {type(e).__name__}: {e}"
                    steps.append({"step": n, "thought": _thought(text),
                                  "tool": f"subagent:{kind}", "tool_input": {"task": task},
                                  "observation": summary, "raw": text})
                    messages.append({"role": "user",
                                     "content": f"Observation（子 agent 摘要）: {summary}"})
                    if self.verbose:
                        print("[subagent summary]", summary[:300])
                    continue

            # 3) Action → 工具
            kind, payload = _parse_action(text)
            if kind == "action":
                name, args = payload
                obs = server.call(name, args)
                steps.append({"step": n, "thought": _thought(text), "tool": name,
                              "tool_input": args, "observation": obs, "raw": text})
            else:
                obs = f"[Action 解析失败] {payload} 请重新输出一轮（Thought+Action+Input）。"
                steps.append({"step": n, "thought": _thought(text), "tool": None,
                              "tool_input": None, "observation": obs, "raw": text})

            messages.append({"role": "user", "content": f"Observation: {obs}"})
            if self.verbose:
                print("[observation]", obs[:400])

            if sum(len(m["content"]) for m in messages) > self.compact_at_chars:
                messages = self._compact(messages)

        # 终验（不信任 agent 自报）：跑一次 pytest
        tests_obs = server.call("run_tests", {})
        tests_passed = tests_obs.startswith("✅")
        patch = _compute_patch(repo)

        return {
            "repo": str(repo),
            "steps": steps,
            "patch": patch,
            "tests_passed": tests_passed,
            "final_pytest": tests_obs,
            "num_steps": len(steps),
            "done": bool(done_text),
            "success": tests_passed,  # 自报信号；eval 只看 tests_passed 终验
        }


# Trace 是 dict（含 steps/patch/tests_passed），README 自检用 trace.get 取值
Trace = dict


def _thought(text: str) -> str:
    m = _RE_THOUGHT.search(text or "")
    return (m.group(1).strip() if m else "")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    repo = root / "data" / "toy-repo"
    issue = (repo / "ISSUE.md").read_text(encoding="utf-8")
    tr = CodingAgent(verbose=True).run(str(repo), issue)
    print("\n======== 结果 ========")
    print("tests_passed:", tr["tests_passed"])
    print("steps:", tr["num_steps"], "| done:", tr["done"])
    print("patch 前 400 字：\n", (tr["patch"] or "")[:400])
