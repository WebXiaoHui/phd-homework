"""手写 ReAct Agent（约 200 行核心循环）。

ReActAgent.run(task) 做多轮：
    Thought -> Action -> Action Input -> (工具执行) -> Observation -> … -> Final Answer

设计要点（对应 README 的 M2/M3）：
- 工具路由：把模型输出的 Action / Action Input 解析出来，在 src.tools.TOOLS 里查名执行；
- 错误恢复：工具抛异常或 Action 解析失败，一律把错误消息塞回 Observation 让 agent 自我纠错，
  单次失败不 crash 整个循环（M3）；
- 终止条件：模型输出 Final Answer，或步数达上限（默认 12）；
- trace：记录每一轮的 Thought / Action / Action Input / Observation 全文，便于调试与提交报告；
- success 仅代表「agent 自己给出了 Final Answer」——自检不信任它，按答案关键词判定。

S4（可选）钩子：run(task, inject_error=callable)，callable(tool_name, args)->str|None，
返回字符串时用它顶替工具真实输出，用于验证错误恢复。
"""
import json
import os
import re
import sys

# 兼容不同启动方式（python src/agent.py / python -m src.agent / eval/run.py 导入）
if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)

from src.tools import TOOLS, all_schemas  # noqa: E402

# 后端配置（OpenAI 兼容）
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_API_KEY = "ollama"

# 正则：Final Answer / Thought
_FINAL_RE = re.compile(r"^Final\s+Answer\s*[:：]?\s*(.*)",
                       re.DOTALL | re.IGNORECASE | re.MULTILINE)
_THOUGHT_RE = re.compile(r"Thought\s*[:：]\s*(.*)")

_TOOL_NAMES = sorted(TOOLS.keys())


def extract_json_obj(raw: str):
    """从模型给的『Action Input 行』里尽力解析出一个 dict。

    返回 (dict|None, 原始字符串)。解析失败返回 (None, cand)。
    """
    s = (raw or "").strip()
    a, b = s.find("{"), s.rfind("}")
    cand = s[a:b + 1] if (a != -1 and b > a) else s
    for variant in (cand, cand.replace("'", '"')):  # 兼容单引号版本（值里无引号才安全）
        try:
            obj = json.loads(variant)
        except Exception:
            continue
        return (obj, cand) if isinstance(obj, dict) else (None, cand)
    return None, cand


def parse_step(text: str):
    """解析模型一轮输出。

    返回：
      ("final",   final_text)          —— 模型输出 Final Answer，循环终止
      ("action",  tool_name, arg_dict) —— 模型要调用工具
      ("invalid", reason)              —— 解析失败（缺标签/坏 JSON），塞回 Observation 重试
    """
    text = (text or "").strip()
    if not text:
        return "invalid", "模型没有输出内容，请重新作答。"

    ai = text.find("Action:")
    m_final = _FINAL_RE.search(text)
    fi_pos = m_final.start() if m_final else -1

    # Final Answer 出现在任何 Action 之前（或根本没有 Action）→ 终止
    if fi_pos != -1 and (ai == -1 or fi_pos < ai):
        return "final", m_final.group(1).strip()
    if ai == -1:
        return "invalid", "输出里没有 Action: 或 Final Answer:，请严格按格式重新输出。"

    # 解析 Action 名
    rest = text[ai + len("Action:"):]
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*", rest)
    if not m:
        return "invalid", "Action: 之后缺少工具名，请补全。"
    name = m.group(1)

    # 解析 Action Input（该 Action 之后的单行/就近 JSON）
    mi = re.search(r"Action\s+Input\s*[:：]\s*(\{.*\})", text[ai:])
    if not mi:
        return "invalid", f"检测到 Action: {name}，但缺少 Action Input（应为 {json.dumps({'参数': '值'}, ensure_ascii=False)} 这样的单行 JSON）。请补全。"
    arg_dict, cand = extract_json_obj(mi.group(1))
    if arg_dict is None:
        return "invalid", f"Action Input 不是合法 JSON：{cand!r}。请用双引号、单行输出参数 JSON。"
    if name not in TOOLS:
        return "invalid", f"未知工具：{name}。可用工具：{', '.join(_TOOL_NAMES)}。请重新 Action。"
    return "action", name, arg_dict


def _build_system_prompt() -> str:
    lines = [
        "你是一个会调用工具来完成任务的 AI 助手。任务可能要求：计算数学式、跑一段 Python、"
        "检索本地文件、或查维基百科获取事实。",
        "",
        "请循环按下面的格式作答，每一轮只输出一个 Action：",
        "Thought: 一句话说明你这一步想做什么、为什么调这个工具",
        "Action: 工具名",
        'Action Input: {"参数名": "值"}',
        "",
        "随后你会收到环境回执（Observation: 工具输出，或错误信息）。",
        "当你已有足够信息回答用户时，**停止调用工具**，用这个格式收尾：",
        "Final Answer: 给用户的最终中文回答",
        "",
        "可用工具：",
    ]
    for schema in all_schemas():
        fn = schema["function"]
        lines.append(f"- {fn['name']}：{fn['description']}")
        lines.append("    参数 schema："
                     f"{json.dumps(fn['parameters']['properties'], ensure_ascii=False)}"
                     f"；必填：{fn['parameters'].get('required', [])}")
    lines += [
        "",
        "规则：",
        "1. Thought / Action / Action Input 必须严格分三行；Action Input 必须是单行合法 JSON"
        "（键和值都用英文双引号）。",
        "2. 每个数值、年份、人名都要用工具查得/算出，不要凭记忆编造。",
        "3. 若 Observation 是错误信息：把错误原因写进 Thought，修正参数后重试（同一问题最多"
        "约 2 次）；仍不行就换一个工具或换一种问法。",
        "4. 最终 Final Answer 用中文，并**把关键数字或事实原样写出来**（如计算结果 456831、"
        "年份 1947、文件名 todo_note.md）。",
        "",
        "示例（注意其格式）：",
        "Thought: 我需要计算 2 + 3 * 4。",
        "Action: calculator",
        'Action Input: {"expression": "2 + 3 * 4"}',
        "(此后环境会回 Observation: 14)",
        "Thought: 结果是 14，信息足够，直接回答。",
        "Final Answer: 2 + 3 * 4 = 14。",
        "",
        "任务开始。",
    ]
    return "\n".join(lines)


class ReActAgent:
    """手写 ReAct 循环。"""

    def __init__(self, base_url: str = None, model: str = None,
                 api_key: str = None, max_steps: int = 12,
                 max_tokens: int = 400, verbose: bool = False):
        self.base_url = base_url or os.environ.get(
            "AGENT_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("AGENT_LLM_MODEL") or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get(
            "AGENT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        self.max_steps = int(os.environ.get("AGENT_MAX_STEPS", max_steps))
        self.max_tokens = int(os.environ.get("AGENT_MAX_TOKENS", max_tokens))
        self.verbose = verbose
        self._client = None  # 懒加载：构造不联网，自检构造 ReActAgent() 不会崩

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "缺少 openai 客户端：pip install 'openai>=1.30'。"
                    "或检查本地模型服务（ollama serve / vLLM）是否已启动。") from e
            self._client = OpenAI(base_url=self.base_url,
                                  api_key=self.api_key)
        return self._client

    def _chat(self, messages) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

    # -- trace / 工具执行 -------------------------------------------------

    def _thought_of(self, text: str) -> str:
        m = _THOUGHT_RE.search(text or "")
        return (m.group(1).strip() if m else "")

    def _exec_tool(self, name, arg_dict, inject_error):
        if inject_error is not None:
            try:
                injected = inject_error(name, arg_dict)
            except Exception as e:  # 钩子自身异常也当作注入的错误
                injected = f"inject_error 钩子异常：{e}"
            if injected:
                return f"[注入的错误] {injected}"
        try:
            out = TOOLS[name].run(arg_dict)
            return str(out)
        except Exception as e:
            # M3：工具异常捕获后塞回 Observation，绝不 crash 循环
            return f"工具 {name} 执行出错：{type(e).__name__}: {e}"

    # -- 主循环 -----------------------------------------------------------

    def run(self, task: str, inject_error=None, verbose: bool = None) -> dict:
        """执行一个任务，返回 AgentTrace（dict）：{steps, final_answer, success, ...}。

        steps：列表，每个工具轮 = {thought, tool, action_input, raw, observation}；
        终止轮带 final_answer 字段。success：仅表示 agent 自己输出了 Final Answer。
        """
        verbose = self.verbose if verbose is None else verbose
        self._ensure_client()

        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": f"任务：{task}"},
        ]
        steps: list = []
        final_answer = ""
        terminated = False

        for n in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n──── step {n} ────")
            text = self._chat(messages)
            messages.append({"role": "assistant", "content": text})
            if verbose:
                print("[assistant]", text[:500])

            kind = parse_step(text)
            if kind[0] == "final":
                final_answer = kind[1]
                steps.append({
                    "step": n, "thought": self._thought_of(text), "tool": None,
                    "action_input": None, "raw": text, "observation": None,
                    "final_answer": final_answer,
                })
                terminated = True
                break

            if kind[0] == "action":
                _, name, arg_dict = kind
                obs = self._exec_tool(name, arg_dict, inject_error)
                steps.append({
                    "step": n, "thought": self._thought_of(text), "tool": name,
                    "action_input": arg_dict, "raw": text, "observation": obs,
                })
            else:  # invalid：解析失败 → 把原因当 Observation 要求重试
                reason = kind[1]
                obs = f"Action 解析失败：{reason} 请重新输出完整一轮（Thought + Action + Action Input）。"
                steps.append({
                    "step": n, "thought": self._thought_of(text), "tool": None,
                    "action_input": None, "raw": text, "observation": obs,
                })

            messages.append({"role": "user",
                             "content": f"Observation: {obs}"})
            if verbose:
                print("[observation]", obs[:500])

        if not terminated:
            # 步数上限：把最后一次内容当 final_answer 兜底，success 仍按是否有 Final Answer
            final_answer = text or ""
            steps.append({
                "step": n, "thought": self._thought_of(text), "tool": None,
                "action_input": None, "raw": text, "observation": None,
                "note": f"达到步数上限 {self.max_steps}，未显式输出 Final Answer。",
            })

        used_tools = [s["tool"] for s in steps if s.get("tool")]
        return {
            "task": task,
            "success": bool(terminated and final_answer),
            "steps": steps,
            "final_answer": final_answer,
            "num_steps": len(steps),
            "used_tools": used_tools,
        }


# 便于复用的别名（README：AgentTrace 是 dict，包含 steps/final_answer/success）
AgentTrace = dict


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "计算 (123 + 456) * 789 的结果，并告诉我结果的位数。"
    tr = ReActAgent(verbose=True).run(q)
    print("\n======== FINAL ========")
    print(tr["final_answer"])
    print("\nsuccess(自报):", tr["success"], "| used_tools:", tr["used_tools"])
