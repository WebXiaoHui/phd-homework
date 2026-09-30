"""task-6-coding-agent：Mini Coding Agent（复刻极简 Claude Code 的本地版）。

能力分三层，对应 README 的「能力三层栈」：
- Tools（底层）：src/toolkit.py 的原子工具 + src/mcp_server.py 的 MCP 包装；
- Skills（中层）：src/skill_loader.py + src/skills/ 下 SKILL.md，按需渐进式披露；
- Subagents（顶层）：src/subagents/，独立 context 的子 agent；
- src/agent.py 的 CodingAgent 用一条 agentic loop 把它们串起来。

在 data/toy-repo 上应做到：读懂 ISSUE → 修 calculator.add → python -m pytest 全绿 → Trace。
"""
