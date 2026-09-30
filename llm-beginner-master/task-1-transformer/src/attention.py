"""任务一 · 手写 attention（禁止调用 nn.MultiheadAttention）。

对外接口（与 eval/run.py 对齐）：
    scaled_dot_product_attention(Q, K, V, mask=None) -> Tensor
        - Q / K / V 形状均为 (B, H, T, D)，返回同形状的加权求和结果。
        - mask 为 bool 张量：True = 该位置被屏蔽（padding 或 causal 都用它），
          形状需可广播到 (B, H, T, T)。例如：
              padding: (B, 1, 1, T)    # 屏蔽第 j 个 key 是 pad 的位置
              causal : (T, T)          # 上三角 True
        - 被屏蔽位置在 softmax 之前填 -inf（而不是乘 0），否则概率会泄漏。

实现要点（对应 tutor 检查项）：
    - 缩放因子是 1/sqrt(d_k)，d_k = Q 最后一维 = 每个 head 的维度。
    - softmax 作用于最后一维（key 序列方向）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention_probs(Q, K, V, mask=None):
    """算注意力分布（softmax 前屏蔽），返回 (B, H, T, T)。"""
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力。

    Q/K/V: (B, H, T, D)；mask: bool，True=屏蔽，可广播到 (B, H, T, T)。
    返回: (B, H, T, D)
    """
    probs = _attention_probs(Q, K, V, mask=mask)
    return probs @ V


class MultiHeadAttention(nn.Module):
    """多头自注意力：单线性投影分头 + 并行点积注意力 + 拼头输出投影。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q / K / V 各用独立投影（不偷懒共享一个）
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 可视化用：记录最近一次前向的注意力分布 (B, H, T, T)，不参与梯度
        self.last_attn_weights = None

    def forward(self, x, attn_mask=None):
        """x: (B, T, d_model)；attn_mask: bool，True=屏蔽，可广播到 (B, H, T, T)。"""
        B, T, _ = x.shape

        def project(linear):
            # (B, T, d_model) -> (B, H, T, d_k)
            return (linear(x)
                    .view(B, T, self.n_heads, self.d_k)
                    .transpose(1, 2))

        Q, K, V = project(self.Wq), project(self.Wk), project(self.Wv)

        probs = _attention_probs(Q, K, V, mask=attn_mask)   # (B, H, T, T)
        self.last_attn_weights = probs.detach()
        probs = self.dropout(probs)

        out = probs @ V                                     # (B, H, T, d_k)
        # 拼回 (B, T, d_model)：先还原维度再 contiguous，否则 view 报错
        out = (out.transpose(1, 2)
               .contiguous()
               .view(B, T, self.d_model))
        return self.Wo(out)
