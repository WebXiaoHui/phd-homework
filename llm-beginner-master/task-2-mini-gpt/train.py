"""任务二 · 预训练脚本：next-token prediction（mini-GPT，集成 RoPE + KV cache）。

用法（先 cd 到本任务目录，下载好数据）：
    python train.py                       # 默认读 data/train.txt，唐诗 quick-start
    python train.py --epochs 30 --lr 5e-4 # 调参
    python train.py --n-embd 128 --n-layer 2   # 显存/CPU 紧张的缩小配置

产物：
    ckpt/tokenizer.json  训练好的 BPE 词表
    ckpt/best.pt         训练好的模型 state_dict + config（自检读它）
    figures/ppl_curve.png  （可选 --save-curve）dev 困惑度曲线

训练结束后跑自检：python eval/run.py
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

DATA = Path(__file__).parent / "data"
CKPT = Path(__file__).parent / "ckpt"


class BlockDataset(Dataset):
    """把一长串 token 切成若干非重叠窗口，每窗长 block_size+1（多 1 个做目标）。"""

    def __init__(self, ids, block_size):
        self.windows = []
        i = 0
        while i + block_size + 1 <= len(ids):
            self.windows.append(ids[i:i + block_size + 1])
            i += block_size

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        return torch.tensor(w, dtype=torch.long)


class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(1, total_steps)
        self.eta_min = eta_min
        self.base = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            f = self._step / self.warmup
        else:
            p = (self._step - self.warmup) / (self.total - self.warmup)
            f = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        lr = self.eta_min + (self.base - self.eta_min) * f
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


@torch.no_grad()
def estimate_ppl(model, tokenizer, text, block_size, device, max_tokens=4096):
    """复刻 eval/run.py 的困惑度算法：按 block 非重叠切窗、累加 NLL。

    注意窗口最多 block_size+1，让模型始终在训练长度内、不进 RoPE 外推区。
    """
    ids = tokenizer.encode(text)[:max_tokens]
    model.eval()
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i:i + block_size + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)
        nll += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               x[:, 1:].reshape(-1), reduction="sum").item()
        n_tok += x.size(1) - 1
    return math.exp(nll / max(1, n_tok)), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--ckpt-dir", default=str(CKPT))
    ap.add_argument("--save-curve", default=None, help="如 figures/ppl_curve.png")

    # tokenizer
    ap.add_argument("--vocab-size", type=int, default=1024,
                    help="BPE 目标词表大小 = 256 字节 + merges")
    ap.add_argument("--tok-sample-chars", type=int, default=None,
                    help="只从前 N 个字符学 BPE（大语料提速用）")
    # 模型
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    # 训练
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  ("mps" if getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available() else "cpu"))
    print(f"[info] device = {device}")

    data_dir = Path(args.data_dir)
    train_text = (data_dir / "train.txt").read_text(encoding="utf-8")
    dev_text = (data_dir / "dev.txt").read_text(encoding="utf-8")
    info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8")) \
        if (data_dir / "dataset_info.json").exists() else {}
    print(f"[info] dataset={info.get('dataset','?')} "
          f"train_chars={len(train_text)} dev_chars={len(dev_text)}")

    # 1) BPE tokenizer
    tokenizer = BPETokenizer.train(train_text, vocab_size=args.vocab_size,
                                   max_chars=args.tok_sample_chars)
    # 先自检一下 roundtrip，训练前就把 tokenizer 问题暴露出来
    for s in ["床前明月光", "Hello, world!", "深度学习需要数学基础"]:
        rt = tokenizer.decode(tokenizer.encode(s))
        assert s == rt, f"tokenizer roundtrip 失败: {s!r} -> {rt!r}"
    tok_dir = Path(args.ckpt_dir)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok_dir / "tokenizer.json")
    print(f"[info] tokenizer vocab_size = {tokenizer.vocab_size} "
          f"-> {tok_dir / 'tokenizer.json'}")

    # 2) 数据
    train_ids = tokenizer.encode(train_text)
    ds = BlockDataset(train_ids, args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"[info] train_tokens={len(train_ids)} blocks={len(ds)}")

    # 3) 模型
    cfg = dict(vocab_size=tokenizer.vocab_size, n_embd=args.n_embd,
               n_head=args.n_head, n_layer=args.n_layer,
               block_size=args.block_size, dropout=args.dropout)
    model = MiniGPT(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = args.epochs * len(loader)
    sched = WarmupCosineLR(opt, warmup_steps=int(args.warmup_frac * total_steps),
                           total_steps=total_steps)

    # 4) 训练
    best_ppl = float("inf")
    history = {"train_loss": [], "ppl": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen, t0 = 0.0, 0, time.time()
        for batch in loader:
            x = batch.to(device)
            logits = model(x[:, :-1])                 # 去掉最后一 token 预测
            loss = F.cross_entropy(logits.reshape(-1, cfg["vocab_size"]),
                                   x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            total_loss += loss.item() * x.size(1)
            n_seen += x.size(1)

        train_loss = total_loss / max(1, n_seen)
        ppl, n_tok = estimate_ppl(model, tokenizer, dev_text,
                                  args.block_size, device)
        history["train_loss"].append(train_loss)
        history["ppl"].append(ppl)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"dev_ppl={ppl:.2f} ({time.time()-t0:.0f}s)")

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({"model": model.state_dict(), "config": cfg},
                       tok_dir / "best.pt")

    print(f"[done] best dev ppl = {best_ppl:.2f} -> {tok_dir / 'best.pt'}")
    if args.save_curve:
        _save_curve(args.save_curve, history)
    print("接下来运行：python eval/run.py 查看自检结果")


def _save_curve(path, history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib 未安装，跳过画图")
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["train_loss"], marker="o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["ppl"], marker="s", color="tab:red", label="dev ppl")
    ax2.set_ylabel("dev ppl", color="tab:red")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"[done] 曲线已存 {path}")


if __name__ == "__main__":
    main()
