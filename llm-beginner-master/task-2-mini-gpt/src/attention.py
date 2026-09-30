"""任务二 · causal 多头自注意力 + KV cache。

- 手写 scaled dot-product，causal mask 用上三角 -inf（True=屏蔽），softmax 前填。
- KV cache：每层维护历史 (K, V)，沿**序列维**（dim=2）拼接。增量解码时
  每个新词元只 forward 它自己，Q 只有 1 行，却能看到缓存的全部历史 K/V。
- 位置：本批 Q/K 的绝对位置 = cache 历史长度 + 本批序号（RoPE 外推正是
  靠这个 offset 才和全量前向对得上）。
"""
import math

import torch
import torch.nn as nn

from src.rope import apply_rotary_pos_emb, rotary_freqs


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        assert (n_embd // n_head) % 2 == 0, "RoPE 的 half-split 需要偶数 head_dim"
        self.n_head = n_head
        self.d_head = n_embd // n_head

        self.Wq = nn.Linear(n_embd, n_embd, bias=False)
        self.Wk = nn.Linear(n_embd, n_embd, bias=False)
        self.Wv = nn.Linear(n_embd, n_embd, bias=False)
        self.Wo = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.last_attn_weights = None        # 可视化用（可选）

    def forward(self, x, cache=None):
        """x: (B, T, D)。cache: 可选 (k, v)，形状 (B, n_head, T_prev, d_head)。

        返回 (out, new_cache)，new_cache 总是拼接后的 (k, v)。
        """
        B, T, _ = x.shape
        h = self.n_head
        d = self.d_head

        q = self.Wq(x).view(B, T, h, d).transpose(1, 2)   # (B,h,T,d)
        k = self.Wk(x).view(B, T, h, d).transpose(1, 2)
        v = self.Wv(x).view(B, T, h, d).transpose(1, 2)

        # 绝对位置 offset = 历史 key 数（cache 在序列维拼接）
        offset = 0 if cache is None else cache[0].shape[2]
        positions = torch.arange(offset, offset + T, device=x.device)
        cos, sin = rotary_freqs(positions, d)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # 拼接历史 K/V（沿序列长度维）
        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_cache = (k, v)

        L = k.shape[2]                                  # 总历史长度
        # causal：本批第 r 行（绝对位置 offset+r）只允许 key 位置 <= offset+r
        rows = torch.arange(T, device=x.device).unsqueeze(1)   # (T,1)
        cols = torch.arange(L, device=x.device).unsqueeze(0)   # (1,L)
        mask = (cols > offset + rows).unsqueeze(0).unsqueeze(0)  # (1,1,T,L), True=屏蔽

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        self.last_attn_weights = probs.detach()
        probs = self.attn_dropout(probs)

        out = probs @ v                                 # (B,h,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, h * d)
        return self.resid_dropout(self.Wo(out)), new_cache
