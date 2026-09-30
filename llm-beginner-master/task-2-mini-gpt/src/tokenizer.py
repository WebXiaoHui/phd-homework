"""任务二 · 手写简化版字节级 BPE tokenizer（不用 tiktoken / sentencepiece）。

思路（GPT-2 同款 byte-level BPE 的简化实现）：
- 基础词表 = UTF-8 的 256 个字节（id 0..255）。任何文本 -> bytes -> int 序列，
  天然覆盖中文等任意 Unicode，且**未登录字不会丢**。
- 迭代 merge：统计当前相邻 (a, b) 对频次，合并频次最高的对为新 id（从 256 起）。
- encode = 按训练得到的 merge 顺序把相邻对逐轮合并；
  decode = 每个 id 展开回它代表的字节串，再整体 utf-8 解码。
- 因为 merge 只是把若干相邻字节拼成一个 token，任何合法切分还原出来的字节串
  都与原文相同 => **encode→decode 一定能还原**（roundtrip 有保证）。

对外接口（与 eval/run.py 对齐）：
    encode(text) -> List[int]
    decode(ids)  -> str
    vocab_size   -> int
    from_pretrained(path)
"""
import json
import re
from collections import Counter
from pathlib import Path

_BASE = 256  # 字节基础词表大小


def _bpe_learn(piece_toks, n_merges):
    """贪心学习 n_merges 次 merge。piece_toks 是若干 token 列表（各代表一个"词"）。"""
    merges = []
    for _ in range(n_merges):
        counts = Counter()
        for toks in piece_toks:
            if len(toks) > 1:
                counts.update(zip(toks, toks[1:]))
        if not counts:
            break
        # 取出现频次最高的对；并列时取先遇到的（max 稳定，便于复现）
        (a, b), _ = max(counts.items(), key=lambda kv: (kv[1],))
        merges.append((a, b))

        nid = _BASE + len(merges)
        for idx, toks in enumerate(piece_toks):
            if len(toks) < 2:
                continue
            out = []
            i = 0
            n = len(toks)
            while i < n:
                if i + 1 < n and toks[i] == a and toks[i + 1] == b:
                    out.append(nid)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            piece_toks[idx] = out
    return merges


class BPETokenizer:
    def __init__(self, merges=None):
        """merges: 按创建顺序的 [(a, b), ...]，第 k 个合并得到 id = 256 + k。"""
        self.merges_ordered = []      # 创建顺序，encode 按此顺序应用
        self.merges = {}              # (a, b) -> id
        if merges:
            for pair in merges:
                a, b = int(pair[0]), int(pair[1])
                nid = _BASE + len(self.merges_ordered)
                self.merges_ordered.append((a, b))
                self.merges[(a, b)] = nid
        self.rev_merges = {v: k for k, v in self.merges.items()}

    # ---------------- 接口 ----------------
    @property
    def vocab_size(self) -> int:
        return _BASE + len(self.merges_ordered)

    def encode(self, text: str) -> list[int]:
        """文本 -> token id 列表（字节级 BPE）。"""
        toks = list(text.encode("utf-8"))
        for a, b in self.merges_ordered:
            nid = self.merges[(a, b)]
            out = []
            i, n = 0, len(toks)
            while i < n:
                if i + 1 < n and toks[i] == a and toks[i + 1] == b:
                    out.append(nid)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            toks = out
        return toks

    def decode(self, ids: list[int]) -> str:
        """token id 列表 -> 文本（id 先还原为字节串再 utf-8 解码）。"""
        buf = bytearray()
        for idx in ids:
            stack = [idx]
            while stack:                       # 迭代展开合并树，避免递归深度
                cur = stack.pop()
                if cur < _BASE:
                    buf.append(cur)
                else:
                    a, b = self.rev_merges[cur]
                    stack.append(b)            # 先处理左子再右子 -> 顺序正确
                    stack.append(a)
        return bytes(buf).decode("utf-8", errors="replace")

    # ---------------- 保存 / 加载 ----------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [[a, b] for a, b in self.merges_ordered],
            "vocab_size": self.vocab_size,
            "base": _BASE,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["merges"])

    # ---------------- 训练 ----------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 1024,
              max_chars: int | None = None) -> "BPETokenizer":
        """从语料学习 BPE。

        max_chars：限制参与训练的字符数（大语料只学一个子集即可，decode 仍然全量可用）。
        vocab_size 目标是 256 + 每次 merge，实际可能因无对可并而略小于目标。
        """
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        # 按空白切"词"，merge 不跨空白（与 GPT-2 观察一致：跨空白合并意义不大）
        pieces = re.findall(r"\S+", text)
        piece_toks = [list(p.encode("utf-8")) for p in pieces]
        merges = _bpe_learn(piece_toks, vocab_size - _BASE)
        return cls(merges)
