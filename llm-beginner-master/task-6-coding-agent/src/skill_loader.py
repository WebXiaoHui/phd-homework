"""skill_loader.py —— Skills 层加载器（约 60 行）。

约定（对齐 Anthropic Skills）：
    src/skills/<name>/SKILL.md   —— YAML front-matter（name / description）+ Markdown 正文

设计：
- list_skills()：只扫 front-matter，不读正文 → 轻量索引；
- match(task_text)：按 description 与 name 的关键词打分，命中才 load（渐进式披露）；
- load(name)：命中后把完整正文返回，供塞进 agent context。

front-matter 解析：优先 PyYAML，缺失时退回简单 `key: value` 行解析（够覆盖 name/description）。
"""
import re
from pathlib import Path

# name / description 之外的 front-matter 字段不关心
_FM_RE = re.compile(r"\s*^(\w[\w-]*)\s*:\s*(.*?)\s*$", re.M)


def _split_front_matter(text: str):
    """返回 (meta: dict, body: str)。"""
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    fm = text[3:end] if end != -1 else text[3:]
    body = text[end + 4:] if end != -1 else ""
    try:
        import yaml
        meta = yaml.safe_load(fm) or {}
        meta = meta if isinstance(meta, dict) else {}
    except Exception:
        meta = {}
    if not meta:  # PyYAML 缺失/解析失败：朴素行解析
        for m in _FM_RE.finditer(fm):
            meta[m.group(1)] = m.group(2).strip().strip("\"'")
    body = re.sub(r"\n*---\s*$", "", body).strip()
    return meta, body


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.root = Path(skills_dir)

    def _index(self):
        """扫描 src/skills/*/SKILL.md，返回 name -> {meta, body, path}。"""
        out = {}
        if not self.root.is_dir():
            return out
        for md in sorted(self.root.glob("*/SKILL.md")):
            try:
                meta, body = _split_front_matter(md.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = meta.get("name") or md.parent.name
            out[name] = {"meta": meta, "body": body, "path": md,
                         "description": meta.get("description", "")}
        return out

    def list_skills(self) -> list:
        """每项含 name / description（自检要求缺一即挂）。"""
        return [{"name": name, "description": info["description"],
                 "path": str(info["path"])}
                for name, info in self._index().items()]

    def load(self, name: str) -> str:
        """返回 SKILL.md 完整正文（去 front-matter）。找不到抛 ValueError。"""
        idx = self._index()
        if name not in idx:
            raise ValueError(f"没有名为 {name!r} 的 Skill。可选：{sorted(idx)}")
        info = idx[name]
        body = info["body"]
        extra = (f"\n（Skill 目录：{info['path'].parent}，若有 scripts/references 可再引用）"
                 if (info["path"].parent / "scripts").exists()
                 or (info["path"].parent / "references").exists() else "")
        return f"# Skill: {name}\n\n{body}{extra}".strip()

    # ------------------------------------------------------------ 渐进式披露

    def match(self, task_text: str) -> str:
        """按 description/name 里的词在 task 中的命中数打分，返回最相关的 Skill 名（可 None）。"""
        task = (task_text or "").lower()
        best, best_score = None, 0
        # 改用 list_skills()，确保拿到完整的 description
        for info in self.list_skills():
            hay = (f"{info.get('name', '')} {info.get('description', '')}").lower()
            tokens = [w for w in re.split(r"[\s，。,.！？!?:：/()、；;]+", hay) if len(w) >= 2]
            score = sum(1 for w in set(tokens) if w in task)
            if score > best_score:
                best, best_score = info.get("name"), score
        return best if best_score > 0 else None

    def load_matching(self, task_text: str):
        """命中则返回 (name, body)，否则 (None, None)。"""
        name = self.match(task_text)
        if name is None:
            return None, None
        try:
            return name, self.load(name)
        except ValueError:
            return None, None


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "skills"
    loader = SkillLoader(str(root))
    print("skills:", [s["name"] for s in loader.list_skills()])
    print("meta ok:", all(s.get("name") and s.get("description")
                          for s in loader.list_skills()))
    for task in ["跑一下测试并诊断失败原因", "给这次改动写一段 PR 描述",
                 "review 一下代码质量"]:
        hit = loader.match(task)
        print(f"task={task!r} -> skill={hit!r}")
