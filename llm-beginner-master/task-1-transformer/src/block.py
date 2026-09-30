"""任务一 · 手写 Transformer encoder block。

采用 Pre-LN 结构（GPT / BERT 之后的主流做法），写作时保持显式一致：
    x = x + SubLayer(LayerNorm(x))        # Pre-LN
等价写法为 Post-LN（x = LayerNorm(x + SubLayer(x))），本实现固定用 Pre-LN。
"""
import torch.nn as nn

from src.attention import MultiHeadAttention


class PositionwiseFFN(nn.Module):
    """两层前馈网络，中间维度通常取 4 * d_model。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个完整 encoder block = 多头注意力 + FFN，各自套 LayerNorm 与 residual。"""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout)

    def forward(self, x, attn_mask=None):
        """attn_mask: bool，True=屏蔽，透传给多头注意力。"""
        # Pre-LN：先 LN 再子层，residual 在外面
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x
