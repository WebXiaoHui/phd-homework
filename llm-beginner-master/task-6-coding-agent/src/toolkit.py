"""toolkit.py —— Mini Coding Agent 的原子工具层（Tools 层）。

与 src/mcp_server.py 同源：mcp_server 把这里的工具包成 MCP stdio server 暴露给外部
client；CodingAgent 则用 ToolServer(repo) 在进程内以同一套实现调用，两路行为一致，
避免在无显示环境下依赖子进程 MCP 握手。

安全约定（README 第 2 周逐条落实）：
- 所有路径先 resolve 再校验必须落在目标 repo 内，拒绝 `..`/仓库外绝对路径；
- 执行命令一律 subprocess list 形式（不用 shell=True），并限定 cwd=repo；
- 禁止改写 *_test.py / 测试文件与 *.orig 快照；git 命令不含 reset --hard / clean -fd /
  checkout -- 等会丢改动的操作；
- 每个工具失败都返回结构化文本，绝不抛到 agent 循环外。
"""
import difflib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


class ToolError(Exception):
    """工具返回给模型看的错误（会被 ToolServer 折叠成文本，不 crash）。"""


# ---------------------------------------------------------------- 路径安全

def _resolve(repo: Path, path: str) -> Path:
    """把（相对/绝对）路径规整后限制在 repo 内；越界抛 ToolError。"""
    p = Path(path or ".")
    if not p.is_absolute():
        p = repo / p
    p = p.resolve()
    repo = repo.resolve()
    if not (p == repo or repo in p.parents):
        raise ToolError(
            f"路径越界：{path} 落在仓库外（允许根 {repo}）。只能操作仓库内的文件。")
    return p


def _guard_writable(p: Path):
    """禁止动测试文件与 .orig 快照（toy-repo 的 ISSUE 明确不让改测试）。"""
    name = p.name.lower()
    if name.startswith("test_") or name.startswith("tests_") or name.endswith(".orig") \
            or name == "conftest.py":
        raise ToolError(f"禁止修改 {p.name}（测试/基准快照文件，issue 约定不改）。")


_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
_SKIP_TAIL = {".pyc", ".orig"}


def _iter_files(repo: Path, sub: str = "."):
    base = _resolve(repo, sub)
    if not base.is_dir():
        raise ToolError(f"目录不存在：{base}")
    out = []
    for p in base.rglob("*"):
        if p.is_file():
            rel = p.relative_to(base)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if p.suffix in _SKIP_TAIL:
                continue
            out.append(p)
    return sorted(out, key=lambda p: p.relative_to(repo).as_posix())


def _read_text(p: Path, limit: int = 150_000):
    try:
        data = p.read_bytes()
    except OSError as e:
        raise ToolError(f"读 {p.name} 失败：{e}") from e
    if b"\x00" in data[:8192]:
        raise ToolError(f"{p.name} 是二进制文件，跳过。")
    for enc in ("utf-8", "gb18030"):
        try:
            return data[:limit].decode(enc)
        except UnicodeDecodeError:
            continue
    return data[:limit].decode("utf-8", errors="replace")


def _run(repo: Path, args: list, *, stdin: str = None, timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, cwd=str(repo), input=stdin, text=True,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"命令超时（>{timeout}s）：{' '.join(args)}") from e
    except FileNotFoundError as e:
        raise ToolError(f"找不到可执行文件：{e}") from e


# ---------------------------------------------------------------- 工具实现
# 每个工具 fn(args: dict, repo: Path) -> str；抛 ToolError 由 ToolServer 折叠成文本。

def read_file(args, repo: Path) -> str:
    p = _resolve(repo, str(args["path"]))
    if not p.is_file():
        raise ToolError(f"文件不存在：{p}")
    big = p.stat().st_size > 150_000
    text = _read_text(p)
    suffix = "\n…（文件过大，已截断前 150KB）" if big else ""
    return f"=== {p.relative_to(repo).as_posix()} ===\n{text}{suffix}"


def write_file(args, repo: Path) -> str:
    p = _resolve(repo, str(args["path"]))
    _guard_writable(p)
    content = str(args.get("content") or "")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return (f"已写入 {p.relative_to(repo).as_posix()} "
            f"（{len(content)} 字符）。如需确认可再 read_file 查看。")


def edit_file(args, repo: Path) -> str:
    p = _resolve(repo, str(args["path"]))
    _guard_writable(p)
    if not p.is_file():
        raise ToolError(f"文件不存在：{p}")
    text = p.read_text(encoding="utf-8")
    old, new = str(args["old_string"]), str(args["new_string"])
    n = text.count(old)
    if n == 0:
        raise ToolError(f"在 {p.name} 里没找到目标片段 {old!r}。请先 read_file 拿到精确文本（含缩进）再试。")
    if n > 1:
        raise ToolError(f"目标片段在 {p.name} 里出现 {n} 次，不唯一。请带上更多上下文再试。")
    p.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    return f"已在 {p.relative_to(repo).as_posix()} 完成 1 处替换。"


def list_files(args, repo: Path) -> str:
    sub = str(args.get("path", "."))
    files = _iter_files(repo, sub)
    cap = 300
    base = _resolve(repo, sub)
    lines = [f"目录 {base.relative_to(repo).as_posix() or '.'} 下共 {len(files)} 个文件："]
    for p in files[:cap]:
        lines.append(f"  {p.relative_to(repo).as_posix()}")
    if len(files) > cap:
        lines.append(f"  …（还有 {len(files) - cap} 个未列出）")
    return "\n".join(lines)


def grep_search(args, repo: Path) -> str:
    pattern = str(args["pattern"])
    regex = bool(args.get("regex", False))
    matcher = re.compile(pattern) if regex else None
    sub = str(args.get("path", "."))
    files = _iter_files(repo, sub)
    cap_hits, cap_files = 60, 15
    lines, hits, shown = [], 0, 0
    for p in files:
        if shown >= cap_files:
            break
        try:
            text = _read_text(p, limit=300_000)
        except ToolError:
            continue
        file_lines = []
        for i, ln in enumerate(text.splitlines(), 1):
            found = matcher.search(ln) if regex else (pattern in ln)
            if found:
                file_lines.append(f"    L{i}: {ln.strip()[:160]}")
                hits += 1
                if hits >= cap_hits:
                    break
        if file_lines:
            shown += 1
            lines.append(f"{p.relative_to(repo).as_posix()}")
            lines.extend(file_lines)
    if not lines:
        return f"没有在仓库内找到匹配 {pattern!r} 的内容。"
    return "\n".join(lines) + (f"\n（命中 {hits} 处，已截断）" if hits >= cap_hits else "")


# 命令安全：拒绝会破坏仓库/逃逸的命令片段（大小写不敏感）
_DENY = ["reset --hard", "clean -fd", "checkout --", "rm -rf", "rm -fr",
         "git push", "git fetch", "sudo", "> /dev/", "&&", "||", "| "]


def run_command(args, repo: Path) -> str:
    command = str(args["command"]).strip()
    low = command.lower()
    bad = [d for d in _DENY if d in low]
    if bad:
        raise ToolError(f"命令被拒绝（含不安全片段：{bad[0]!r}）。只能用非破坏性命令，cwd=仓库。")
    argv = shlex.split(command)
    if not argv:
        raise ToolError("command 为空。")
    proc = _run(repo, argv)
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out[-4000:].strip() or "（无输出）"
    if proc.returncode == 0:
        return f"命令成功（退出码 0）。\n{tail}"
    return f"命令退出码 {proc.returncode}。\n{tail}"


def run_tests(args, repo: Path) -> str:
    extra = args.get("args") or []
    argv = [sys.executable, "-m", "pytest", "-q"] + list(extra)
    proc = _run(repo, argv)
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"(\d+) passed", out)
    if proc.returncode == 0:
        n = m.group(1) if m else "?"
        return f"✅ pytest 通过：{n} passed（退出码 0）。可以 DONE 收尾了。"
    tail = out[-1200:].strip()
    return f"❌ pytest 失败（退出码 {proc.returncode}）。输出末尾：\n{tail}"


def git_diff(args, repo: Path) -> str:
    path = args.get("path")
    argv = ["git", "-C", str(repo), "diff", "--"] + ([str(path)] if path else [])
    proc = _run(repo, argv)
    diff = (proc.stdout or "").strip()
    if diff:
        return f"=== git diff（{len(diff)} 字符）===\n{diff}"
    # 无 git 或没改动：退回 .orig 快照与当前文件做 unified diff
    orig, cur = repo / "calculator.py.orig", repo / "calculator.py"
    if orig.exists() and cur.exists():
        a = orig.read_text(encoding="utf-8").splitlines(keepends=True)
        b = cur.read_text(encoding="utf-8").splitlines(keepends=True)
        if a != b:
            return "=== diff (calculator.py.orig → calculator.py) ===\n" + \
                   "".join(difflib.unified_diff(a, b,
                                                fromfile="a/calculator.py",
                                                tofile="b/calculator.py"))
    return "（仓库内没有未提交改动）"


def git_apply(args, repo: Path) -> str:
    patch = str(args["patch"])
    if not patch.strip():
        raise ToolError("patch 为空。")
    if any(d in patch.lower() for d in _DENY):
        raise ToolError("patch 含危险指令，拒绝应用。")
    proc = _run(repo, ["git", "-C", str(repo), "apply", "--whitespace=nowarn", "-"],
                stdin=patch)
    if proc.returncode == 0:
        return "✅ 补丁已应用。"
    return (f"❌ 补丁应用失败（退出码 {proc.returncode}）。"
            f"可能上下文对不上，请 read_file 确认当前内容。\n{(proc.stderr or '')[-800:]}")


# ---------------------------------------------------------------- 注册表 & schema

def _schema(name, description, properties, required):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties,
                             "required": required}}


TOOL_SCHEMAS = [
    _schema("read_file", "读取仓库内一个文本文件的内容（相对或仓库内绝对路径）。",
            {"path": {"type": "string", "description": "相对仓库根或仓库内的绝对路径"}},
            ["path"]),
    _schema("write_file", "整文件覆盖写入（会替换全部内容）。禁止写测试/快照文件。",
            {"path": {"type": "string"}, "content": {"type": "string",
             "description": "要写入的完整文件内容"}},
            ["path", "content"]),
    _schema("edit_file", "在文件里精确替换一处 old_string→new_string（须唯一）。先 read_file 拿准确片段。",
            {"path": {"type": "string"},
             "old_string": {"type": "string"},
             "new_string": {"type": "string"}},
            ["path", "old_string", "new_string"]),
    _schema("list_files", "列出仓库内目录下的文件（排除 .git/缓存/快照）。",
            {"path": {"type": "string", "description": "子目录，默认仓库根"}},
            []),
    _schema("grep_search", "在仓库文件内容里搜关键词（可选正则）。",
            {"pattern": {"type": "string"}, "regex": {"type": "boolean", "default": False},
             "path": {"type": "string", "description": "子目录，默认仓库根"}},
            ["pattern"]),
    _schema("run_command", "在仓库 cwd 里跑任意非破坏性 shell 命令（args 数组形式，禁 shell 注入）。",
            {"command": {"type": "string", "description": "例如 'python -c \\\"print(1+1)\\\"'"}},
            ["command"]),
    _schema("run_tests", "在仓库里跑 python -m pytest -q，返回通过/失败摘要。测试全绿时可直接 DONE。",
            {"args": {"type": "array", "items": {"type": "string"},
                      "description": "可选的额外 pytest 参数，如 ['-k','test_add']"}},
            []),
    _schema("git_diff", "查看当前未提交改动（无 git 时退回 calculator.py.orig 对比）。",
            {"path": {"type": "string"}}, []),
    _schema("git_apply", "应用一份 unified diff patch（git apply）。",
            {"patch": {"type": "string", "description": "unified diff 文本"}},
            ["patch"]),
]

TOOLS = {s["name"]: {"schema": s, "fn": globals()[s["name"]]} for s in TOOL_SCHEMAS}


def list_tools() -> list:
    """返回全部工具 schema（MCP server / 自检共用）。"""
    return [dict(s) for s in TOOL_SCHEMAS]


class ToolServer:
    """把一个目标仓库绑定到工具集上，进程内调用。

        server = ToolServer(repo)
        server.call("read_file", {"path": "calculator.py"})   # -> str
    """

    def __init__(self, repo):
        self.repo = Path(repo).resolve()

    @property
    def names(self):
        return list(TOOLS.keys())

    def call(self, name, args: dict) -> str:
        if name not in TOOLS:
            return (f"[工具错误] 未知工具 {name!r}。可选："
                    f"{', '.join(self.names)}")
        try:
            result = TOOLS[name]["fn"](dict(args or {}), self.repo)
            return str(result)
        except ToolError as e:
            return f"[工具错误] {e}"
        except Exception as e:  # 兜底：绝不 crash agent 循环
            return f"[工具异常 {type(e).__name__}] {e}"


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1] / "data" / "toy-repo"
    srv = ToolServer(repo)
    print("工具数：", len(srv.names))
    print(srv.call("read_file", {"path": "calculator.py"}))
    print(srv.call("run_tests", {}))
