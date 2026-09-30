"""src/tools：4 类工具的注册表。

每个工具模块统一导出：
- TOOL_SCHEMA: dict   —— OpenAI function calling 格式的 schema（含 name/description/parameters）
- run(args: dict) -> str

自检按固定参数键调用 run：
    calculator     -> {"expression": str}
    python_sandbox -> {"code": str}
    file_search    -> {"pattern": str, "dir": str}
    wiki           -> {"query": str}
"""
from . import calculator, file_search, python_sandbox, wiki

# 路由顺序（决定 system prompt 里工具列表的顺序）
TOOL_LIST = [calculator, python_sandbox, file_search, wiki]

# name -> module 的路由表，供 src.agent.ReActAgent 使用
TOOLS = {m.TOOL_SCHEMA["function"]["name"]: m for m in TOOL_LIST}


def all_schemas() -> list:
    """返回全部工具的 OpenAI function calling schema 列表。"""
    return [m.TOOL_SCHEMA for m in TOOL_LIST]


__all__ = [
    "TOOL_LIST", "TOOLS", "all_schemas",
    "calculator", "python_sandbox", "file_search", "wiki",
]
