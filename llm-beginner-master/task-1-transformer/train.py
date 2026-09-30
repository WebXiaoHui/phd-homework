"""任务一 · 训练脚本：在 ChnSentiCorp 上训练 Transformer 情感分类器。

用法（先 cd 到本任务目录并下载好数据）：
    python train.py                       # 默认超参（d_model=128, heads=4, layers=4）
    python train.py --epochs 8 --lr 1e-3  # 自行调参

产物：
    ckpt/best.pt     训练好的 checkpoint（含 state_dict + vocab + config）
    figures/train_curve.png   （可选 --save-curve）训练 loss / dev acc 曲线

训练结束后跑自检：
    python eval/run.py
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.model import TransformerClassifier
from src.tokenizer import CharTokenizer

# ChnSentiCorp 列名
TEXT_COL, LABEL_COL = "text", "label"


def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer on ChnSentiCorp")
    p.add_argument("--data-dir", default="data", help="data/*.parquet 所在目录")
    p.add_argument("--ckpt", default="ckpt/best.pt", help="保存 best 模型路径")
    p.add_argument("--save-curve", default="eval/figures/train_curve.png",
                   help="存 loss/acc 曲线图路径，如 figures/train_curve.png")

    # 模型
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # 训练
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda",
                   help="auto / cuda / cpu")
    p.add_argument("--min-freq", type=int, default=1,
                   help="词表过滤：出现次数低于该值的字符归 <unk>")
    return p.parse_args()


class SentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.x = [tokenizer.encode(t) for t in texts]   # 已定长 (max_len,)
        self.y = [int(l) for l in labels]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], torch.tensor(self.y[i], dtype=torch.long)


def load_data(data_dir):
    train = pd.read_parquet(Path(data_dir) / "train.parquet")
    dev = pd.read_parquet(Path(data_dir) / "validation.parquet")
    return (train[TEXT_COL].tolist(), train[LABEL_COL].tolist(),
            dev[TEXT_COL].tolist(), dev[LABEL_COL].tolist())


class WarmupCosineLR:
    """线性 warmup + cosine 退火，手动实现（无需 scheduler 封装也能讲清楚）。"""

    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        self.opt = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            factor = self._step / self.warmup_steps
        else:
            prog = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        lr = self.eta_min + (self.base_lr - self.eta_min) * factor
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cpu")
    print(f"[info] device = {device}")

    # 1. 数据 + 词表
    tr_texts, tr_labels, dv_texts, dv_labels = load_data(args.data_dir)
    tokenizer = CharTokenizer.build(tr_texts, max_len=args.max_len,
                                    min_freq=args.min_freq)
    vocab_size = len(tokenizer)
    print(f"[info] train={len(tr_texts)} dev={len(dv_texts)} vocab_size={vocab_size}")

    train_ds = SentDataset(tr_texts, tr_labels, tokenizer)
    dev_ds = SentDataset(dv_texts, dv_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False)

    # 2. 模型
    cfg = dict(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff or 4 * args.d_model, num_classes=args.num_classes,
        max_len=args.max_len, dropout=args.dropout, pad_id=tokenizer.pad_id,
    )
    model = TransformerClassifier(vocab_size=vocab_size, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params = {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    sched = WarmupCosineLR(optimizer,
                           warmup_steps=int(args.warmup_frac * total_steps),
                           total_steps=total_steps)

    # 3. 训练
    out_ckpt = Path(args.ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "dev_acc": [], "lr": []}
    best_acc, best_state = -1.0, None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            lr_now = sched.step()

            total_loss += loss.item() * y.numel()
            n_seen += y.numel()
            global_step += 1

        train_loss = total_loss / max(1, n_seen)
        dev_acc = evaluate(model, dev_loader, device)
        history["train_loss"].append(train_loss)
        history["dev_acc"].append(dev_acc)
        history["lr"].append(lr_now)
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} dev_acc={dev_acc:.4f} "
              f"lr={lr_now:.2e} ({time.time()-t0:.0f}s)")

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 4. 保存 best
    torch.save({"model": best_state, "vocab": tokenizer.vocab, "config": cfg,
                "best_dev_acc": best_acc}, out_ckpt)
    print(f"[done] best dev acc = {best_acc:.4f} -> {out_ckpt}")

    if args.save_curve:
        _save_curve(args.save_curve, history)

    print("\n接下来运行：python eval/run.py 查看自检结果")


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
    ax1.set_ylabel("loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["dev_acc"], marker="s", color="tab:red", label="dev acc")
    ax2.set_ylabel("dev acc", color="tab:red")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"[done] 训练曲线已存 {path}")


if __name__ == "__main__":
    main()
