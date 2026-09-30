"""任务三 · SFT：在 Qwen2.5-0.5B（注入手写 LoRA）上用 MOSS 多轮对话微调。

只对 assistant 内容（含收尾 <|im_end|>）算 next-token loss；user / system /
模板控制符一律 mask 掉（-100）。原始权重全程冻结，只训 lora_A/lora_B。

用法（先 cd 本任务目录、跑完 data/download.py 的下载提示）：
    python train_sft.py --data data/moss-sft/moss-003-sft-no-tools.jsonl
    python train_sft.py --data ... --inspect          # 先体检一行数据（解析出几轮）
    python train_sft.py --data ... --max-samples 200  --epochs 1   # 小批跑通 pipeline

产物：ckpt/sft/adapter_config.json + adapter.bin（即“SFT 后的 LoRA 权重目录”）
跑完自检：python eval/run.py
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.chat import pad_token_id_of
from src.dataset import make_sft_samples, pad_tensors
from src.lora import build_pretrained, count_trainable, inject_lora, save_adapter

ROOT = Path(__file__).parent


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

class SftDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_collate(pad_id: int):
    def collate(batch):
        ids, mask = pad_tensors([b["input_ids"] for b in batch],
                                pad_id=pad_id)
        labels, _ = pad_tensors([b["labels"] for b in batch], pad_id=-100)
        return ids, labels, mask
    return collate


def shifted_ce_loss(logits, input_ids, labels):
    """手动移位做因果交叉熵：训练目标 = assistant 内容 token。

    位置约定：labels[p] == token id 表示“位置 p 的 token 要能被预测”
    （也就是用 logits[p-1] 去预测 input_ids[p]），其余 -100。
    """
    B, T, V = logits.shape
    shift_logits = logits[:, :-1].reshape(-1, V)
    shift_targets = input_ids[:, 1:].reshape(-1)
    keep = (labels[:, 1:] != -100).reshape(-1)          # 要监督的位置
    ce = F.cross_entropy(shift_logits, shift_targets, reduction="none")
    n_tokens = keep.sum().item()
    if n_tokens <= 0:
        return ce.mean(), 0.0
    return (ce * keep).sum() / n_tokens, n_tokens


class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(warmup_steps + 1, total_steps)
        self.eta_min = eta_min
        self.base = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        s = min(self._step, self.total)
        if s <= self.warmup:
            f = s / self.warmup
        else:
            p = (s - self.warmup) / (self.total - self.warmup)
            f = 0.5 * (1 + math.cos(math.pi * p))
        lr = self.eta_min + (self.base - self.eta_min) * f
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def resolve_device(arg: str):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-0.5B"),
                    help="基座模型目录")
    ap.add_argument("--data", default=str(ROOT / "data" / "moss-sft" /"moss-003-sft-data_1w.jsonl"),
                    help="MOSS SFT jsonl（moss-003-sft-no-tools.jsonl）")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ckpt" / "sft"))
    ap.add_argument("--target-modules", default="q_proj,v_proj")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=2000,
                    help="最多取多少条做训练（先把 pipeline 跑通，再放大）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="梯度累积步数，等效 batch = batch-size * grad-accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="开 gradient checkpointing 省显存（略慢）")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--save-every", type=int, default=100,
                    help="每多少步存一次 adapter（crash 保险），结束时必存")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--inspect", action="store_true",
                    help="只解析并打印前几条数据用于体检，不训练")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" and device == "cuda" \
        else torch.float32
    if args.dtype == "bf16" and device != "cuda":
        print("[warn] bf16 需要 CUDA，已回退 fp32")

    print("=" * 64)
    print(f"[sft] device={device} dtype={dtype}  data={args.data}")
    print(f"[sft] LoRA r={args.r} alpha={args.alpha} "
          f"targets={args.target_modules}")

    # 1) 基座 + LoRA 注入
    model, tok = build_pretrained(args.base, dtype=dtype, device=device)
    inject_lora(model, [m.strip() for m in args.target_modules.split(",")
                        if m.strip()],
                r=args.r, alpha=args.alpha, dropout=args.lora_dropout,
                seed=args.seed)
    trainable, total = count_trainable(model)
    print(f"[sft] 可训练参数 {trainable:,} / {total:,} = "
          f"{trainable / max(total, 1):.4%}")
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 2) 数据
    samples, stats = make_sft_samples(args.data, tok, args.max_len,
                                      max_samples=None if args.inspect
                                      else args.max_samples,
                                      inspect=args.inspect)
    print(f"[sft] 解析 {stats['parsed']} 行，schema 跳过 {stats['schema_skip']}"
          f"，超长跳过 {stats['long_skip']}")
    if args.inspect:
        print("\n[inspect] 上面若显示『未解析出对话』，说明你的字段不在"
              " src/dataset.py 覆盖范围内，见操作流程 4.3")
        return
    if not samples:
        raise SystemExit("[错误] 没有解析出任何可训练样本：请先 --inspect 检查"
                         "数据 schema，或确认文件路径/是否为解压后的 jsonl")
    pad_id = pad_token_id_of(tok)
    ds = SftDataset(samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(pad_id), drop_last=False)
    print(f"[sft] 训练样本 {len(samples)} 条，共 {len(loader)} micro-batch"
          f"，epochs={args.epochs}  pad_id={pad_id}")

    # 3) 优化器（只含 requires_grad 的 LoRA 参数）
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    macro_per_epoch = max(1, len(loader) // max(1, args.grad_accum))
    total_steps = args.epochs * macro_per_epoch
    sched = WarmupCosineLR(opt, warmup_steps=args.warmup_steps,
                           total_steps=total_steps)

    out_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) 训练
    model.train()
    opt.zero_grad()
    micro = macro = 0
    accum_loss = accum_tokens = 0.0
    global_loss = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for input_ids, labels, attn in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = attn.to(device)
            logits = model(input_ids=input_ids,
                           attention_mask=attn).logits
            loss, ntok = shifted_ce_loss(logits, input_ids, labels)
            (loss / max(1, args.grad_accum)).backward()
            micro += 1
            accum_loss += loss.item() * max(1, ntok)
            accum_tokens += max(ntok, 1)

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                macro += 1
                global_loss = accum_loss / accum_tokens
                accum_loss = accum_tokens = 0.0

                if macro == 1 or macro % args.log_every == 0:
                    el = time.time() - t0
                    sps = macro * args.grad_accum * args.batch_size / max(el, 1e-6)
                    print(f"[sft] epoch {epoch} step {macro}/{total_steps} | "
                          f"loss {global_loss:.4f} | lr "
                          f"{opt.param_groups[0]['lr']:.2e} | {sps:.1f} seq/s | "
                          f"{el:.0f}s")
                if macro % args.save_every == 0:
                    _save(model, args, out_dir, {"step": macro,
                                                 "loss": global_loss,
                                                 "samples": len(samples)})

    if accum_tokens:
        global_loss = accum_loss / accum_tokens
    _save(model, args, out_dir, {"step": macro, "loss": global_loss,
                                 "samples": len(samples),
                                 "epochs": args.epochs})
    print(f"[sft] 完成：loss={global_loss:.4f}  adapter -> {out_dir}")
    print("接下来：python eval/run.py 看自检；python train_dpo.py 做偏好对齐；"
          "python src/compare.py 对比 base/SFT/DPO")


def _save(model, args, out_dir, meta):
    meta = dict(meta)
    meta.update(base_model=args.base, epochs=args.epochs,
                max_len=args.max_len)
    save_adapter(model, out_dir,
                 target_modules=[m.strip() for m in
                                 args.target_modules.split(",") if m.strip()],
                 r=args.r, alpha=args.alpha, meta=meta)


if __name__ == "__main__":
    main()
