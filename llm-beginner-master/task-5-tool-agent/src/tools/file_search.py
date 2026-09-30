"""file_search 工具：在本地目录里按文件名 / 内容检索，并返回内容片段。

特性：
- 路径越界保护：dir 先 resolve，必须落在允许根（本任务仓库根）内，`../../` 会被拒；
- 文件名命中：支持 glob（*.md）或大小写不敏感子串（README.md）；
- 内容命中：在文本文件内容里检索 pattern，命中返回路径 + 命中处片段；
- 文件名命中文件较少时自动附「内容预览（开头）」，便于"读文件内容"类问题；
- 输出包含『共找到 N 个匹配文件』，可直接用于计数问题。

参数固定键：{"pattern": str, "dir": str}
"""
import fnmatch
import re
from pathlib import Path

# 允许的检索根：本任务仓库根（src/tools/file_search.py 上溯 3 层）
ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ROOTS = [ROOT]

MAX_FILES_SCANNED = 5000
MAX_MATCHES_SHOWN = 100
PREVIEW_CHARS = 240
SNIPPET_CHARS = 160
READ_LIMIT = 500_000  # 每个文件最多读的字节数

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在本地目录里检索文件名或文件内容。"
            "pattern 可以是文件名（如 README.md）、文件名 glob（如 *.md）或文件内容关键词"
            "（如 TODO）。dir 是相对本任务目录的路径，例如 data/agent-fixtures。"
            "会返回匹配文件的相对路径；文件名命中且命中很少时会附文件开头内容预览，"
            "内容命中会附命中片段。也可直接用它读某个小文件的内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "要匹配的文件名/glob 或内容关键词，如 README.md / *.md / TODO。",
                },
                "dir": {
                    "type": "string",
                    "description": "要检索的目录（相对本任务目录），如 data/agent-fixtures。",
                },
            },
            "required": ["pattern", "dir"],
        },
    },
}


def _resolve_allowed(dirarg):
    base = Path(str(dirarg or "."))
    if not base.is_absolute():
        base = ROOT / base
    base = base.resolve()
    for root in ALLOWED_ROOTS:
        r = root.resolve()
        if base == r or r in base.parents:
            return base
    raise ValueError(f"目录越界：{base} 不在允许范围 {ALLOWED_ROOTS[0]} 内；"
                     f"file_search 只能读任务仓库内的文件")


def _read_text(path: Path):
    """尽力按文本解码一个文件；无法解码（二进制）返回 ''。"""
    try:
        data = path.read_bytes()[:READ_LIMIT]
    except OSError:
        return ""
    for enc in ("utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def _collect_files(base: Path):
    files = []
    for p in base.rglob("*"):
        if p.is_file():
            files.append(p)
            if len(files) >= MAX_FILES_SCANNED:
                break
    return sorted(files, key=lambda p: p.relative_to(base).as_posix())


def _preview(text, limit=PREVIEW_CHARS):
    """把文本压成单行片段便于回显。"""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit] + ("…" if len(flat) > limit else "")


def run(args):
    if not isinstance(args, dict) or "pattern" not in args:
        raise ValueError("file_search 需要参数 pattern（str）与 dir（str）")
    pattern = str(args["pattern"])
    base = _resolve_allowed(args.get("dir", "."))
    if not base.is_dir():
        raise ValueError(f"目录不存在：{base}")

    is_glob = any(ch in pattern for ch in "*?[")
    files = _collect_files(base)

    name_hits, content_hits = [], []
    for p in files:
        name = p.name
        if (fnmatch.fnmatch(name, pattern) if is_glob
                else pattern.lower() in name.lower()):
            name_hits.append(p)
            continue  # 已算文件名命中；内容里再命中会在下面按内容补，但列表避免重复计数
    # 内容命中（与文件名命中相互独立，路径可能重复显示——注明类别即可）
    if not (is_glob and name_hits):
        for p in files:
            if p in name_hits:
                continue
            text = _read_text(p)
            if text and pattern in text:
                content_hits.append(p)

    lines = [f"检索目录：{base}", f"pattern：{pattern!r}", ""]
    if name_hits:
        shown = name_hits[:MAX_MATCHES_SHOWN]
        lines.append(f"【文件名命中】{len(name_hits)} 个：")
        for i, p in enumerate(shown):
            lines.append(f"  - {p.relative_to(base).as_posix()}")
            if i < 3:  # 命中少时给开头预览，方便"读内容"类问题
                text = _read_text(p)
                if text:
                    lines.append(f"    内容预览：{_preview(text)}")
        if len(name_hits) > len(shown):
            lines.append(f"  …（还有 {len(name_hits) - len(shown)} 个未列出）")
    if content_hits:
        shown = content_hits[:MAX_MATCHES_SHOWN]
        lines.append(f"【内容命中】{len(content_hits)} 个：")
        for p in shown:
            text = _read_text(p)
            idx = text.find(pattern)
            lines.append(f"  - {p.relative_to(base).as_posix()}")
            if idx != -1:
                snippet = text[max(0, idx - 40): idx + len(pattern) + SNIPPET_CHARS]
                lines.append(f"    命中片段：{_preview(snippet, SNIPPET_CHARS + 40)}")
        if len(content_hits) > len(shown):
            lines.append(f"  …（还有 {len(content_hits) - len(shown)} 个未列出）")
    if not name_hits and not content_hits:
        lines.append("没有匹配的文件（可换关键词或换目录再试）。")

    total = len(name_hits) + len(content_hits)
    lines.append("")
    lines.append(f"共找到 {total} 个匹配文件。")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run({"pattern": "*.md", "dir": "data/agent-fixtures"}))
    print()
    print(run({"pattern": "TODO", "dir": "data/agent-fixtures"}))
