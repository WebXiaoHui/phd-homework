"""任务三 · Qwen chat template + loss masking（只对 assistant 内容算 loss）。

对外接口（README / eval/run.py 契约）：
- format_messages(messages: List[dict]) -> str
    把 [{role, content}, ...] 拼成 Qwen 官方格式：
        <|im_start|>user\n...<|im_end|>
        <|im_start|>assistant\n...<|im_end|>
- build_labels(input_ids, messages, tokenizer=None) -> labels
    返回与 input_ids 等长的 LongTensor：assistant 内容所在位置填该 token
    id（训练目标），user / system / 模板控制符（含 <|im_start|>assistant）
    一律填 -100。

设计说明（为什么能对上）：mock / MOSS 里每段 assistant 内容在完整 ids
里是「连续的一段」，而 Qwen 是 GPT2 风格 BPE——空白与 <|im_start|> 等
特殊 token 处不会跨段 merge，因此对内容单独 encode 得到的 token 序列
是整段 ids 的一个连续子序列，顺序做子序列匹配即可精确定位。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"


# --------------------------------------------------------------------------
# Qwen chat template
# --------------------------------------------------------------------------

def _as_text(content) -> str:
    """把 content 归一化成文本（可能是 str / dict / list of blocks）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text", b.get("content", ""))))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content)


def format_messages(messages: List[dict]) -> str:
    """Qwen 官方 chat template（Jinja 展开后的等价文本）。

    <|im_start|>{role}\n{content}<|im_end|>\n  逐条拼接。
    """
    if not messages:
        return ""
    text = ""
    out = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = _as_text(m.get("content"))
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def build_labels(input_ids, messages: List[dict],
                 tokenizer=None) -> torch.Tensor:
    """给「训练哪个位置」做 mask：assistant 内容位置 -> token id，其余 -100。

    - input_ids: token ids（torch.Tensor / list[int]），即 format_messages
      的输出被 tokenize 后的完整序列；
    - messages:  与 tokenize 时完全一致的对话结构（含多轮多个 assistant）；
    - tokenizer: 默认懒加载 models/Qwen2.5-0.5B 的 AutoTokenizer。
    """
    tok = tokenizer if tokenizer is not None else _default_tokenizer()
    ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
    n = len(ids)
    labels = [-100] * n

    # `<|im_end|>` 的 token id：assistant 内容之后的收尾符也一并监督
    # （模型要会“自己结束回合”，否则生成时不知停）。
    im_end_id = None
    try:
        _enc = tok.encode("<|im_end|>", add_special_tokens=False)
        if len(_enc) == 1:
            im_end_id = _enc[0]
    except Exception:
        im_end_id = None

    cursor = 0          # 顺序扫描起点：上一个 assistant 片段(含收尾符)之后
    misses = 0
    for m in messages:
        if str(m.get("role", "")).strip().lower() != "assistant":
            continue
        content = _as_text(m.get("content"))
        if not content:
            continue
        needle = tok.encode(content, add_special_tokens=False)
        if not needle:
            continue
        start = _find_subseq(ids, needle, cursor)
        if start is None:          # 兜底：从头再找（极少见，避免 cursor 卡死）
            start = _find_subseq(ids, needle, 0)
        if start is None:
            misses += 1
            continue
        for j, t in enumerate(needle):
            labels[start + j] = t
        cursor = start + len(needle)
        # 紧邻的收尾 <|im_end|> 也算作要预测的目标，并让 cursor 跳过它
        if im_end_id is not None and cursor < n and ids[cursor] == im_end_id:
            labels[cursor] = im_end_id
            cursor += 1

    if misses:
        print(f"[chat] 警告：{misses} 个 assistant 片段没在 ids 中对齐"
              "（可能被截断或与模板拼接有关），这些位置不参与 loss")
    return torch.tensor(labels, dtype=torch.long)


def _find_subseq(hay: List[int], needle: List[int], start: int):
    """hay[start:] 里找第一个 needle 的起点；找不到返回 None。"""
    ln = len(needle)
    if ln == 0:
        return None
    end = len(hay) - ln
    i = max(0, start)
    while i <= end:
        if hay[i] == needle[0] and hay[i:i + ln] == needle:
            return i
        i += 1
    return None


def _default_tokenizer():
    from transformers import AutoTokenizer

    path = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() \
        else "Qwen/Qwen2.5-0.5B"
    return AutoTokenizer.from_pretrained(path)


# --------------------------------------------------------------------------
# 便捷：给推理拼 prompt（加上开头的 assistant 提示符再让模型续写）
# --------------------------------------------------------------------------

def build_generation_text(prompt: str) -> str:
    """user 一问 + 开 assistant 头，模型从这里开始续写回答。"""
    return format_messages([{"role": "user", "content": prompt}]) \
        + "<|im_start|>assistant\n"


def clean_generated(text: str) -> str:
    """把生成文本截到第一个 <|im_end|>（或 <|endoftext|>），去掉控制符。"""
    for sep in ("<|im_end|>", "<|endoftext|>"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip()


def pad_token_id_of(tokenizer) -> int:
    """Qwen 若没设 pad，就用 eos 顶替，并回填 tokenizer.pad_token。"""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer.pad_token_id
    return 0
