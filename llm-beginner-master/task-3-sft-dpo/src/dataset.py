"""任务三 · SFT / DPO 数据解析与打包。

MOSS-003-sft（SFT）与 DPO 偏好数据在 HuggingFace 上没有统一 schema，
网上流通版本字段各异（conversation / turns / messages / chat …；
role 也可能是 from / speaker）。因此这里写成**容错解析**：

- 先按若干常见顶层容器键找「多轮列表」；
- 每个 turn 支持两种形态：
    (a) {role, content} 类；
    (b) {human, assistant} 类（MOSS-003 官方），一个 turn 展开成两条消息；
- 全都找不到时退回「扁平键」形态（instruction/output、prompt/completion…）。

配合 `train_sft.py --inspect` 打印首个样本做体检。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from src.chat import build_labels, format_messages

CONTAINER_KEYS = ("conversation", "messages", "turns", "dialog", "dialogue",
                  "chat", "data")
ROLE_KEYS = ("role", "from", "speaker", "who", "by", "name")
CONTENT_KEYS = ("content", "value", "text", "utterance", "message")

# MOSS-003 类：一个 turn 同时携带人类与助手两条内容
HUMAN_KEYS = ("human", "user", "prompt", "query", "question", "问", "用户")
ASSIST_KEYS = ("assistant", "ai", "bot", "gpt", "response", "answer",
               "completion", "回复", "助手", "回答")

_USER_ALIAS = {"user", "human", "customer", "question", "prompt", "usr",
               "h", "使用者", "用户", "用户问"}
_ASSIST_ALIAS = {"assistant", "ai", "model", "bot", "gpt", "response",
                 "answer", "completion", "回复", "助手", "机器人"}
_SYSTEM_ALIAS = {"system", "sys"}


# --------------------------------------------------------------------------
# jsonl 读取
# --------------------------------------------------------------------------

def load_jsonl(path: Union[str, Path]) -> Iterable[dict]:
    """逐行读 jsonl，跳过空行与解析失败的行。"""
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 3:
                    print(f"[data] 第 {line_no} 行不是合法 JSON，已跳过")
    if bad:
        print(f"[data] 共跳过 {bad} 个坏行")


# --------------------------------------------------------------------------
# SFT 对话解析
# --------------------------------------------------------------------------

def _dig_container(node, depth=0):
    """在多轮容器键里找「列表型对话」；嵌套也找。"""
    if depth > 4 or not isinstance(node, dict):
        return None
    for k in CONTAINER_KEYS:
        v = node.get(k)
        if isinstance(v, list) and len(v) > 0:
            return v
    for k in ("content", "data", "inner", "body", "value"):
        v = node.get(k)
        if isinstance(v, dict):
            r = _dig_container(v, depth + 1)
            if r is not None:
                return r
    return None


def _role_of(turn: dict) -> Optional[str]:
    for k in ROLE_KEYS:
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _content_of(turn: dict):
    for k in CONTENT_KEYS:
        v = turn.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return v
    return None


def _norm_role(role: str) -> Optional[str]:
    r = (role or "").strip().lower()
    if r in _USER_ALIAS:
        return "user"
    if r in _ASSIST_ALIAS:
        return "assistant"
    if r in _SYSTEM_ALIAS:
        return "system"
    if any(u in r for u in ("user", "human", "用户", "问")):
        return "user"
    if any(a in r for a in ("assistant", "ai", "model", "bot", "回答", "助手")):
        return "assistant"
    if "system" in r or "系统" in r:
        return "system"
    return None


def _to_str(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return _to_str(x.get("content") or x.get("text") or "")
    if isinstance(x, list):
        return "".join(_to_str(i) for i in x)
    return str(x) if x is not None else ""


def _turn_to_msg(turn) -> Optional[dict]:
    """把 {role, content} 形态的 turn 转成标准消息。"""
    if not isinstance(turn, dict):
        return None
    role = _norm_role(_role_of(turn) or "")
    content = _content_of(turn)
    if role is None or content is None:
        return None
    return {"role": role, "content": content}


def _moss_turn_pair(turn) -> Optional[List[dict]]:
    """MOSS-003 形态：一个 turn 里同时含 human / assistant 两条。

    返回 [user_msg, assistant_msg] 或 None。
    """
    if not isinstance(turn, dict):
        return None

    human_val = next((turn[k] for k in HUMAN_KEYS
                      if isinstance(turn.get(k), (str, list, dict))
                      and _to_str(turn[k]).strip()), None)
    assist_val = next((turn[k] for k in ASSIST_KEYS
                       if isinstance(turn.get(k), (str, list, dict))
                       and _to_str(turn[k]).strip()), None)

    if human_val is None and assist_val is None:
        return None

    out: List[dict] = []
    if human_val is not None:
        out.append({"role": "user", "content": _to_str(human_val)})
    if assist_val is not None:
        out.append({"role": "assistant", "content": _to_str(assist_val)})
    return out if out else None


def _turns_to_msgs(turns: Sequence) -> List[dict]:
    """统一的 turns → messages 展开。"""
    msgs: List[dict] = []
    for t in turns:
        # 优先按 MOSS-003 的双字段 turn 处理
        pair = _moss_turn_pair(t)
        if pair is not None:
            msgs.extend(pair)
            continue
        # 否则走通用 role/content 解析
        m = _turn_to_msg(t)
        if m is not None:
            msgs.append(m)
    return msgs


def _flat_fallback(item: dict) -> Optional[List[dict]]:
    """扁平形态：Alpaca 系 instruction/output + 可选 history。"""
    msgs: List[dict] = []
    system = item.get("system")
    if isinstance(system, str) and system.strip():
        msgs.append({"role": "system", "content": system.strip()})

    history = item.get("history")
    if isinstance(history, list):
        for h in history:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                msgs.append({"role": "user",
                             "content": _to_str(h[0]) if h[0] else ""})
                msgs.append({"role": "assistant",
                             "content": _to_str(h[1]) if h[1] else ""})

    ptn = ("instruction", "prompt", "input", "query", "question", "问题")
    ans = ("output", "response", "answer", "completion", "reply", "回答")
    p = next((item[k] for k in ptn if isinstance(item.get(k), str)
              and item[k].strip()), None)
    extra = item.get("input")
    if isinstance(p, str) and isinstance(extra, str) and extra.strip():
        p = p + "\n" + extra.strip()
    a = next((item[k] for k in ans if isinstance(item.get(k), str)
              and item[k].strip()), None)
    if p is not None and a is not None:
        msgs.append({"role": "user", "content": p})
        msgs.append({"role": "assistant", "content": a})
    return msgs if msgs else None


def to_messages(item: dict) -> Optional[List[dict]]:
    """把一个 SFT jsonl 行解析成 [{role, content}, ...]，解析不了返回 None。"""
    if not isinstance(item, dict):
        return None
    turns = _dig_container(item)
    if turns is None:
        return _flat_fallback(item)
    msgs = _turns_to_msgs(turns)
    return msgs if msgs else None


# --------------------------------------------------------------------------
# SFT token 化
# --------------------------------------------------------------------------

def tokenize_messages(msgs: Sequence[dict], tokenizer, max_len: int):
    """-> {input_ids, labels}，或 None（超过 max_len，丢弃）。"""
    if not msgs:
        return None
    text = format_messages(msgs)
    if not text or not text.strip():
        return None
    ids = tokenizer.encode(text)
    if len(ids) < 2 or len(ids) > max_len:
        return None
    labels = build_labels(ids, msgs, tokenizer=tokenizer)
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": labels}


def make_sft_samples(data_path: Union[str, Path], tokenizer, max_len: int,
                     max_samples: Optional[int] = None,
                     inspect: bool = False):
    """读取并 token 化 SFT 样本。

    返回 (samples, stats)。stats 含 parsed / schema_skip / long_skip。
    inspect=True 时只诊断、不做 token 化。
    """
    path = Path(data_path)
    samples: List[dict] = []
    parsed = schema_skip = long_skip = 0
    diag = 0

    for i, item in enumerate(load_jsonl(path)):
        msgs = to_messages(item)

        # ---- schema 过滤 ----
        if not msgs:
            schema_skip += 1
            if inspect and schema_skip <= 2:
                diag += 1
                keys = list(item.keys()) if isinstance(item, dict) else type(item)
                print(f"[inspect] 第 {i+1} 行未解析出对话，原始字段：{keys}")
            # 注意：无论是否 inspect，schema 未通过都不再做长度统计
            continue

        # ---- schema 通过 ----
        parsed += 1

        if inspect:
            if parsed <= 2:
                diag += 1
                print(f"\n[inspect] 样本#{parsed} 解析出的消息（前 2 轮）：")
                for m in msgs[:2]:
                    c = m["content"] if isinstance(m["content"], str) else str(m["content"])
                    print(f"  - {m['role']}: {c[:60]}{'…' if len(c) > 60 else ''}")
            if diag >= 3:
                break
            continue  # inspect 模式不 token 化、不累计 long_skip

        # ---- 正常模式：token 化 ----
        sample = tokenize_messages(msgs, tokenizer, max_len)
        if sample is None:
            long_skip += 1
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break

    stats = {"parsed": parsed, "schema_skip": schema_skip,
             "long_skip": long_skip}
    return samples, stats


# --------------------------------------------------------------------------
# DPO 偏好数据
# --------------------------------------------------------------------------

def _messages_from_turn_list(lst: list) -> Optional[List[dict]]:
    msgs = _turns_to_msgs(lst)
    return msgs if msgs else None


def to_dpo_messages(item: dict) -> Optional[Tuple[List[dict], List[dict]]]:
    """把一个偏好行解析成 (chosen_messages, rejected_messages)。"""
    ch, rj = item.get("chosen"), item.get("rejected")

    # A：整个对话是角色列表
    if isinstance(ch, list) and isinstance(rj, list):
        ch_msgs = _messages_from_turn_list(ch)
        rj_msgs = _messages_from_turn_list(rj)
        if ch_msgs and rj_msgs:
            return ch_msgs, rj_msgs

    # B：prompt + 两个纯文本回答
    prompt = item.get("prompt")
    if isinstance(prompt, str) and isinstance(ch, str) and isinstance(rj, str):
        ch_msgs = [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": ch}]
        rj_msgs = [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": rj}]
        return ch_msgs, rj_msgs

    # C：hiyouga/DPO-En-Zh-20k 格式：question / response_chosen / response_rejected
    question = item.get("question")
    chosen = item.get("response_chosen")
    rejected = item.get("response_rejected")
    system = item.get("system")
    if isinstance(question, str) and isinstance(chosen, str) and isinstance(rejected, str):
        ch_msgs: List[dict] = []
        rj_msgs: List[dict] = []
        if isinstance(system, str) and system.strip():
            ch_msgs.append({"role": "system", "content": system.strip()})
            rj_msgs.append({"role": "system", "content": system.strip()})
        ch_msgs.append({"role": "user", "content": question})
        ch_msgs.append({"role": "assistant", "content": chosen})
        rj_msgs.append({"role": "user", "content": question})
        rj_msgs.append({"role": "assistant", "content": rejected})
        return ch_msgs, rj_msgs

    # D：conversations + chosen/rejected（本数据集实际格式）
    #    conversations: [{"from": "human", "value": "..."}, ...]
    #    chosen:        {"from": "gpt", "value": "..."}
    #    rejected:      {"from": "gpt", "value": "..."}
    convs = item.get("conversations")
    if isinstance(convs, list) and isinstance(ch, dict) and isinstance(rj, dict):
        shared = _messages_from_turn_list(convs)
        ch_msg = _turn_to_msg(ch)
        rj_msg = _turn_to_msg(rj)
        if shared and ch_msg and rj_msg:
            # chosen = 共享 prompt + chosen 回复
            ch_msgs = list(shared) + [ch_msg]
            rj_msgs = list(shared) + [rj_msg]
            # 确保最后一条是 assistant
            if ch_msgs[-1]["role"] == "assistant" and rj_msgs[-1]["role"] == "assistant":
                return ch_msgs, rj_msgs

    return None


def make_dpo_samples(data_path: Union[str, Path], tokenizer, max_len: int,
                     max_samples: Optional[int] = None,
                     inspect: bool = False):
    """读取并 token 化 DPO 偏好对。"""
    path = Path(data_path)
    samples: List[dict] = []
    parsed = schema_skip = long_skip = 0
    diag = 0

    for i, item in enumerate(load_jsonl(path)):
        pair = to_dpo_messages(item)
        if pair is None:
            schema_skip += 1
            if inspect and schema_skip <= 2:
                diag += 1
                keys = list(item.keys()) if isinstance(item, dict) else type(item)
                print(f"[inspect] 第 {i+1} 行未解析出偏好对，原始字段：{keys}")
            continue

        parsed += 1
        if inspect:
            if parsed <= 2:
                diag += 1
                ch_msgs, rj_msgs = pair
                print(f"\n[inspect] 偏好对#{parsed}  prompt="
                      f"{str(ch_msgs[0]['content'])[:50]}…  "
                      f"chosen_len={len(str(ch_msgs[-1]['content']))}  "
                      f"rejected_len={len(str(rj_msgs[-1]['content']))}")
            if diag >= 3:
                break
            continue

        sample = _pair_to_sample(pair, tokenizer, max_len)
        if sample is None:
            long_skip += 1
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break

    stats = {"parsed": parsed, "schema_skip": schema_skip,
             "long_skip": long_skip}
    return samples, stats


def _pair_to_sample(pair, tokenizer, max_len):
    ch_msgs, rj_msgs = pair
    ch = tokenize_messages(ch_msgs, tokenizer, max_len)
    rj = tokenize_messages(rj_msgs, tokenizer, max_len)
    if ch is None or rj is None:
        return None
    return {"chosen_ids": ch["input_ids"], "chosen_labels": ch["labels"],
            "rejected_ids": rj["input_ids"], "rejected_labels": rj["labels"]}


# --------------------------------------------------------------------------
# collate
# --------------------------------------------------------------------------

def pad_tensors(seqs: Sequence[torch.Tensor], pad_id: int):
    """把若干 1-D LongTensor 右补齐到 batch 最长。"""
    B = len(seqs)
    L = max(s.numel() for s in seqs)
    padded = torch.full((B, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        l = s.numel()
        padded[i, :l] = s
        mask[i, :l] = 1
    return padded, mask