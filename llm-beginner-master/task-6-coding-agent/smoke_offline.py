"""task-6 · 离线冒烟：M1/M2/工具层不依赖模型即可验证（等价 eval 前两项 + 工具可用性）。

用法：
    python smoke_offline.py

不需要本地模型服务；若 data/toy-repo 已生成会顺带试一下 read_file/run_tests。
"""
from pathlib import Path

ROOT = Path(__file__).parent

rc = 0


def check(label, ok, extra=""):
    global rc
    print(f"  [{'✓' if ok else '✗'}] {label} {extra}")
    if not ok:
        rc = 1


def main():
    print("== M1 · mcp_server.list_tools ==")
    from src.mcp_server import list_tools as mcp_list_tools
    tools = mcp_list_tools()
    names = [t.get("name") for t in tools]
    check("list_tools() 返回 >=5 个工具", isinstance(tools, list) and len(tools) >= 5,
          f"（{len(tools)} 个：{names}）")
    check("每个工具都含 name/description/input_schema",
          all(t.get("name") and t.get("description") and t.get("input_schema")
              for t in tools))

    print("\n== M2 · SkillLoader ==")
    from src.skill_loader import SkillLoader
    loader = SkillLoader(str(ROOT / "src" / "skills"))
    skills = loader.list_skills()
    check("list_skills() >= 2 个且都有 name+description",
          len(skills) >= 2 and all(s.get("name") and s.get("description") for s in skills),
          f"（{len(skills)} 个：{[s['name'] for s in skills]}）")
    hit = loader.match("跑一下 pytest 看看哪里挂了")
    check("match('跑 pytest…') 命中 test-runner", hit == "test-runner", f"（{hit!r}）")

    print("\n== 工具层 · ToolServer on data/toy-repo ==")
    toy = ROOT / "data" / "toy-repo"
    if (toy / "calculator.py").exists():
        from src.toolkit import ToolServer
        srv = ToolServer(str(toy))
        out = srv.call("read_file", {"path": "calculator.py"})
        check("read_file 能读 calculator.py", "def add" in out)
        out = srv.call("run_tests", {})
        check("run_tests 返回文本（现为失败状态也正常）", "pytest" in out and ("❌" in out or "✅" in out),
              f"（{out[:40]}…）")
        out = srv.call("write_file",
                       {"path": "../outside.txt", "content": "x"})
        check("write_file 拒绝越界路径", out.startswith("[工具错误]"))
        out = srv.call("edit_file",
                       {"path": "calculator.py", "old_string": "return a - b",
                        "new_string": "return a + b"})
        check("edit_file 能替换 add 的符号", "完成 1 处替换" in out)
        # 还原，避免污染后续 eval（eval 也会自己从 .orig 重置）
        srv.call("write_file", {"path": "calculator.py",
                                "content": (toy / "calculator.py.orig").read_text(encoding="utf-8")})
    else:
        print("  （data/toy-repo 不存在，先跑 python data/download.py；跳过工具试跑）")

    print("\n== 结论 ==", "离线冒烟通过（M1+M2+工具层可用）" if rc == 0 else "有失败项")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
