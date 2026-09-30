"""mcp_server.py —— 把 toolkit 的 9 个原子工具暴露成 MCP stdio server（M1）。

- 模块顶层导出 `list_tools() -> List[dict]`（自检枚举工具，>=5）；
- 内部导出 `run_tool(name, args, repo=None)`（agent/外部进程共用）；
- `python src/mcp_server.py` 独立启动：走 FastMCP stdio 握手（需 `pip install mcp`）。
  进程内默认目标仓库：环境变量 CODING_AGENT_REPO，缺省 data/toy-repo。

MCP 客户端只 import list_tools/run_tool 也能用（纯进程内），不依赖子进程握手，
从而在无显示环境与离线情况下同样稳定。
"""
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 python src/mcp_server.py 直接跑
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.toolkit import TOOLS, ToolServer  # noqa: E402


def default_repo() -> Path:
    env = os.environ.get("CODING_AGENT_REPO")
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parents[1] / "data" / "toy-repo")


def list_tools() -> list:
    """返回全部工具的 OpenAI 风格 schema（每个 dict 含 name/description/input_schema）。"""
    return [dict(TOOLS[name]["schema"]) for name in TOOLS]


def run_tool(name: str, args: dict, repo=None) -> str:
    """进程内调用一个工具。repo 缺省用默认仓库。"""
    server = ToolServer(repo or default_repo())
    return server.call(name, args)


def _type_hint(props_desc: dict):
    t = props_desc.get("type")
    if t == "string":
        return str
    if t == "boolean":
        return bool
    if t == "array":
        return list
    if t == "number":
        return float
    if t == "integer":
        return int
    return str


def _make_fastmcp_server():
    """用 mcp SDK 的 FastMCP 把工具注册成 MCP tools；版本不兼容时报清晰提示。"""
    try:
        from mcp.server.fastmcp import FastMCP
        from pydantic import Field, create_model
    except ImportError as e:
        raise RuntimeError("需要 mcp Python SDK：pip install 'mcp>=0.9' "
                           "（pydantic 随之安装）。") from e

    repo = default_repo()
    mcp = FastMCP("mini-coding-agent")

    for name, meta in TOOLS.items():
        schema = meta["schema"]
        desc = schema["description"]
        props = schema["input_schema"]["properties"]
        required = schema["input_schema"].get("required", [])
        fields = {}
        for pn, pd in props.items():
            hint = _type_hint(pd)
            if pn in required:
                fields[pn] = (hint, Field(description=pd.get("description", "")))
            else:
                fields[pn] = (hint, Field(default=None,
                                          description=pd.get("description", "")))
        Model = create_model(f"{name}_args", **fields)

        @mcp.tool(name=name, description=desc)
        def _call(args: Model) -> str:
            return ToolServer(repo).call(name, args.model_dump())

    return mcp


def main():
    try:
        mcp = _make_fastmcp_server()
    except RuntimeError as e:
        print(f"[mcp_server] {e}", file=sys.stderr)
        print("[mcp_server] 独立 stdio server 需要安装 mcp SDK；"
              "不过 eval 只 import list_tools()，不依赖它。", file=sys.stderr)
        sys.exit(2)
    # stdio 握手（本机 Claude Code / 其他 MCP client 会以子进程方式拉起）
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
