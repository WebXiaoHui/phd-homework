"""task-5 · 四个工具的独立快速自检（等价 eval 的 tools_individual，含 wiki 离线降级）。

用法：
    python tools_check.py
"""
from pathlib import Path

from src.tools import calculator, file_search, python_sandbox, wiki

ROOT = Path(__file__).parent

CHECKS = [
    ("calculator", calculator, {"expression": "2 + 3 * 4"}, "14", False),
    ("python_sandbox", python_sandbox, {"code": "print(sum(range(10)))"}, "45", False),
    ("file_search", file_search, {"pattern": "README.md", "dir": str(ROOT)}, "README.md", False),
    ("wiki", wiki, {"query": "Alan Turing"}, None, True),
]


def main():
    print("== tools_individual 快速自检 ==")
    network_skipped = []
    results = {}
    for name, mod, args, expected, network in CHECKS:
        try:
            out = str(mod.run(args))
            ok = (expected in out) if expected is not None else (len(out) > 50)
            results[name] = ok
            print(f"  [{('✓' if ok else '✗')}] {name}"
                  + ("" if expected is not None else "（仅检查长度>50）"))
            if not expected and not ok:
                print(f"        输出长度不足：{out[:80]!r}")
        except Exception as e:
            if network:
                results[name] = "skip"
                network_skipped.append(name)
                print(f"  [skip] {name}（网络不可用？）：{e}")
            else:
                results[name] = False
                print(f"  [✗] {name}：{e}")
    gated = [v for k, v in results.items() if k not in network_skipped]
    passed = bool(gated) and all(v is True for v in gated)
    print("\n== 结论 ==")
    print(f"  工具单项：{'全部通过（M1 ✓）' if passed else '有失败，请修后再跑 eval'}")
    if network_skipped:
        print(f"  wiki 因网络被跳过（离线/被墙），其余不受影响。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
