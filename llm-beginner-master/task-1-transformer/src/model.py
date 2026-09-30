"""任务一 · Transformer 文本分类模型。

对外接口（与 eval/run.py 对齐）：
    class TransformerClassifier
    load_for_eval(ckpt_path: str) -> (model, tokenize_fn)

约定细节：
    tokenize_fn(text: str) -> LongTensor，形状 (T,)（T = max_len，已截断补齐）
    model(ids)               接受形状 (B, T) 的 id 张量 -> logits (B, num_classes)
"""
import torch
import torch.nn as nn

from src.block import TransformerBlock
from src.tokenizer import CharTokenizer


class TransformerClassifier(nn.Module):
    """词嵌入 + 可学习位置嵌入 -> N 层 encoder block -> masked mean pooling -> 分类头。"""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int | None = None, num_classes: int = 2,
                 max_len: int = 200, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)      # 可学习位置编码
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)                  # 最后一层输出再过一次 LN
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        """ids: (B, T)；返回 logits (B, num_classes)。"""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)

        x = self.token_emb(ids) + self.pos_emb(pos).unsqueeze(0)  # (B, T, D)
        x = self.dropout(x)

        # padding mask：屏蔽 key 为 <pad> 的列，形状 (B, 1, 1, T)，可广播到 (B,H,T,T)
        attn_mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(2)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm(x)

        # masked mean pooling：均值只看非 pad 词元（等价于忽略 PAD 输出）
        valid = (ids != self.pad_id).unsqueeze(-1)          # (B, T, 1)
        summed = (x * valid).sum(dim=1)                     # (B, D)
        counts = (ids != self.pad_id).sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = summed / counts

        return self.head(pooled)                            # (B, num_classes)


def load_for_eval(ckpt_path: str):
    """从 ckpt/best.pt 恢复模型与字符 tokenizer。

    ckpt 内容为一个 dict：
        {"model": state_dict, "vocab": {token:id}, "config": {超参...}}
    返回 (model, tokenize_fn)，模型置于 CPU 且处于 eval 模式。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tokenizer = CharTokenizer(ckpt["vocab"], max_len=cfg["max_len"])

    model = TransformerClassifier(
        vocab_size=len(tokenizer),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        num_classes=cfg["num_classes"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
        pad_id=cfg["pad_id"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    def tokenize_fn(text: str) -> torch.LongTensor:
        return tokenizer.encode(text)                       # 形状 (max_len,)

    return model, tokenize_fn
