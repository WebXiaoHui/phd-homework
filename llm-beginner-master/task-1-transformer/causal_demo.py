"""任务一 · causal mask 与 toy 语言模型（对应 DoD M4，为任务二预热）。

两种用法：
1) 快速自检（必跑，无训练开销）：验证「未来位置 V 改动不影响过去输出」，
   与 eval/run.py 的 causal_mask 测试是同一性质检查。
        python causal_demo.py

2) 可选的 toy 语言模型训练 + 自回归生成（演示同一个 attention 加上 causal
   mask 就能做 next-token prediction）：
        python causal_demo.py --train-toy [--text 任意.txt]

不调 nn.MultiheadAttention；decoder-only 的 causal 化只靠 attention 的 mask。
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.attention import scaled_dot_product_attention
from src.block import TransformerBlock
from src.tokenizer import CharTokenizer

DEFAULT_TEXT = Path(__file__).resolve().parent.parent / "poetryFromTang.txt"


def make_causal_mask(T: int, device=None) -> torch.BoolTensor:
    """上三角 True = 屏蔽（未来词元不可见），形状 (T, T)。"""
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(device)


def check_no_leak() -> float:
    """复刻自检逻辑：改最后一个位置的 V，过去位置的输出应完全不变。"""
    torch.manual_seed(0)
    B, H, T, D = 1, 1, 5, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = make_causal_mask(T)

    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    V2 = V.clone()
    V2[:, :, -1] = 999.0                       # 大幅篡改未来 token 的 value
    out2 = scaled_dot_product_attention(Q, K, V2, mask=mask)

    leaked = (out[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
    status = "通过" if leaked < 1e-6 else "失败"
    print(f"[DoD-M4 检查] 未来 V 篡改后过去输出最大偏差 = {leaked:.3e} -> {status}")
    return leaked


# --------------------------------------------------------------------------- #
# 可选：极简 causal 语言模型（仅演示，非任务二实现）
# --------------------------------------------------------------------------- #

class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff=d_model * 4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        """ids: (B, ctx)，返回 next-token logits (B, ctx, vocab)。"""
        B, T = ids.shape
        x = self.emb(ids)
        causal = make_causal_mask(T, ids.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=causal)
        return self.head(self.norm(x))


class SlidingWindowDataset(Dataset):
    def __init__(self, ids, ctx):
        self.x, self.y = [], []
        for i in range(len(ids) - ctx):
            self.x.append(ids[i:i + ctx])
            self.y.append(ids[i + 1:i + ctx + 1])    # 预测下一个 token
        self.x = torch.tensor(self.x, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


@torch.no_grad()
def generate(model, tokenizer, seed, steps=80, ctx=64, temperature=0.9, top_k=20):
    model.eval()
    ids = [tokenizer.vocab.get(c, tokenizer.unk_id) for c in seed]
    out = list(ids)
    for _ in range(steps):
        window = ids[-ctx:]
        x = torch.tensor(window, dtype=torch.long).unsqueeze(0)
        logits = model(x)[0, -1] / max(temperature, 1e-3)
        logits[logits < torch.topk(logits, top_k)[0][-1]] = float("-inf")
        nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
        ids.append(nxt)
        out.append(nxt)
    return "".join(tokenizer.itos.get(i, "�") for i in out)


def train_toy(text_path: Path, epochs: int = 3, ctx: int = 64, batch: int = 128,
              lr: float = 3e-3):
    text = Path(text_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.build([text], max_len=ctx + 16)
    raw = [tokenizer.vocab.get(c, tokenizer.unk_id) for c in text]
    print(f"[toy] 文本 {len(raw)} 字符, vocab={len(tokenizer)}")

    ds = SlidingWindowDataset(raw, ctx)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model = ToyCausalLM(vocab_size=len(tokenizer))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x).reshape(-1, len(tokenizer)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"[toy] epoch {ep} loss = {tot / max(1, len(loader)):.4f}")

    seed = text[:12]
    print("[toy] 生成示例（seed =", seed, "）：")
    print("   ", generate(model, tokenizer, seed)[len(seed):])
    print("[toy] 注意：这是极小 toy，只为观察 causal LM 的行为；真正的 "
          "mini-GPT 在任务二实现。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-toy", action="store_true", help="训练 toy 语言模型并生成")
    ap.add_argument("--text", default=str(DEFAULT_TEXT), help="训练文本文件")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    leaked = check_no_leak()
    if args.train_toy:
        train_toy(Path(args.text), epochs=args.epochs)
    if leaked < 1e-6:
        print("\n[通过] 自检会同样通过（eval/run.py -> causal_mask）")


if __name__ == "__main__":
    main()
