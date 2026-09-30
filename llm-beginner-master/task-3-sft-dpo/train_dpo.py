"""任务三 · DPO：在 SFT 之上做偏好对齐（直接偏好优化）。

两套参数：
- policy：基座 + SFT adapter（load_adapter 载入 ckpt/sft），可训练 LoRA；
- ref   ：另一个基座 + 同一份 SFT adapter，**冻结且只 forward、不参与反向**。

每个 batch 对 chosen / rejected 各 forward 一次：policy×2 + ref×2 共 4 次。
loss = -log σ(β·( logπ/π_ref(chosen) - logπ/π_ref(rejected) ))

用法（先跑完 train_sft.py）：
    python train_dpo.py --data data/dpo/dpo_en_zh.jsonl
    python train_dpo.py --data ... --inspect
    python train_dpo.py --data ... --max-samples 200  # 小批跑通 pipeline

产物：ckpt/dpo/adapter_config.json + adapter.bin
对比三个模型：python src/compare.py
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.chat import pad_token_id_of
from src.dataset import make_dpo_samples, pad_tensors
from src.lora import (build_pretrained, count_trainable, load_adapter,
                      save_adapter)

ROOT = Path(__file__).parent


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

class DpoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_collate(pad_id: int):
    def collate(batch):
        cids, cmask = pad_tensors([b["chosen_ids"] for b in batch], pad_id)
        clabels, _ = pad_tensors([b["chosen_labels"] for b in batch], -100)
        rids, rmask = pad_tensors([b["rejected_ids"] for b in batch], pad_id)
        rlabels, _ = pad_tensors([b["rejected_labels"] for b in batch], -100)
        return (cids, cmask, clabels, rids, rmask, rlabels)
    return collate


def response_logprob(logits, input_ids, labels):
    """一条序列里 assistant 内容 token 的 log 概率之和。

    用模型自己的 shift：位置 p 的 token 由 logits[p-1] 预测，因此
    (labels[p] != -100) 的那些 target 落在 shift 后的第 p-1 列。
    """
    shift_logits = logits[:, :-1]                       # (B, T-1, V)
    shift_targets = input_ids[:, 1:]                    # (B, T-1)
    keep = (labels[:, 1:] != -100).float()              # (B, T-1)
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_lp = logp.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * keep).sum(-1)                      # (B,)


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
    ap.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    ap.add_argument("--init-adapter", default=str(ROOT / "ckpt" / "sft"),
                    help="SFT LoRA adapter 目录（DPO 的起点，缺它会报错）")
    ap.add_argument("--data", required=True, help="DPO 偏好 jsonl")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ckpt" / "dpo"))

    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="DPO 用较小 lr")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO 温度系数 β（默认 0.1）")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=2)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init_adapter = Path(args.init_adapter)
    if not (init_adapter / "adapter_config.json").exists():
        raise SystemExit(f"[错误] 没找到 SFT adapter：{init_adapter}。"
                         "DPO 要在 SFT 之上做，请先跑 train_sft.py 生成 ckpt/sft/")

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
    print(f"[dpo] device={device} dtype={dtype}  data={args.data}")
    print(f"[dpo] init(SFT) adapter = {init_adapter}  beta={args.beta}")

    # 1) policy 与 ref 都 = 基座 + SFT adapter
    policy, tok = build_pretrained(args.base, dtype=dtype, device=device)
    load_adapter(policy, init_adapter)
    policy.to(device)

    ref, _ = build_pretrained(args.base, dtype=dtype, device=device)
    load_adapter(ref, init_adapter)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    trainable, total = count_trainable(policy)
    print(f"[dpo] policy 可训练参数 {trainable:,} / {total:,} = "
          f"{trainable / max(total, 1):.4%}  (ref 已冻结)")

    if args.grad_checkpoint:
        policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    ref.config.use_cache = False

    # 2) 数据
    samples, stats = make_dpo_samples(args.data, tok, args.max_len,
                                      max_samples=None if args.inspect
                                      else args.max_samples,
                                      inspect=args.inspect)
    print(f"[dpo] 偏好对 {stats['parsed']}，schema 跳过 {stats['schema_skip']}"
          f"，超长跳过 {stats['long_skip']}")
    if args.inspect:
        print("\n[inspect] 打印正常即可正式训练；若『未解析出偏好对』，"
              "见操作流程 4.4")
        return
    if not samples:
        raise SystemExit("[错误] 没有解析出偏好对：先 --inspect，或换 "
                         "hiyouga/DPO-En-Zh-20k 格式的数据")

    pad_id = pad_token_id_of(tok)
    loader = DataLoader(DpoDataset(samples), batch_size=args.batch_size,
                        shuffle=True, collate_fn=make_collate(pad_id))

    lora_params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    macro_per_epoch = max(1, len(loader) // max(1, args.grad_accum))
    total_steps = args.epochs * macro_per_epoch
    sched = WarmupCosineLR(opt, warmup_steps=args.warmup_steps,
                           total_steps=total_steps)

    out_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) 训练
    policy.train()
    opt.zero_grad()
    micro = macro = 0
    accum = accum_n = 0.0
    loss_trace = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for (cids, cmask, clabels, rids, rmask, rlabels) in loader:
            cids, cmask, clabels = cids.to(device), cmask.to(device), \
                clabels.to(device)
            rids, rmask, rlabels = rids.to(device), rmask.to(device), \
                rlabels.to(device)

            # --- policy：chosen + rejected（算梯度的两路） ---
            logp_ch = response_logprob(policy(input_ids=cids,
                                              attention_mask=cmask).logits,
                                       cids, clabels)
            logp_rj = response_logprob(policy(input_ids=rids,
                                              attention_mask=rmask).logits,
                                       rids, rlabels)

            # --- ref：冻结 + no_grad，只供 logp 参考 ---
            with torch.no_grad():
                logp_ch_ref = response_logprob(
                    ref(input_ids=cids, attention_mask=cmask).logits,
                    cids, clabels)
                logp_rj_ref = response_logprob(
                    ref(input_ids=rids, attention_mask=rmask).logits,
                    rids, rlabels)

            # DPO: -log σ(β·[(logπ-π_ref)_chosen - (logπ-π_ref)_rejected])
            margin = args.beta * ((logp_ch - logp_ch_ref)
                                  - (logp_rj - logp_rj_ref))
            loss = -F.logsigmoid(margin).mean()
            (loss / max(1, args.grad_accum)).backward()
            micro += 1
            accum += loss.item() * len(margin)
            accum_n += len(margin)
            acc = (margin > 0).float().mean().item()

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                macro += 1
                mloss = accum / accum_n
                loss_trace.append(mloss)
                accum = accum_n = 0.0
                if macro == 1 or macro % args.log_every == 0:
                    el = time.time() - t0
                    print(f"[dpo] epoch {epoch} step {macro}/{total_steps} | "
                          f"loss {mloss:.4f} | margin acc {acc:.2f} | "
                          f"lr {opt.param_groups[0]['lr']:.2e} | {el:.0f}s")
                if macro % args.save_every == 0:
                    save_adapter(policy, out_dir,
                                 meta={"step": macro, "loss": mloss,
                                       "beta": args.beta})

    if accum_n:
        mloss = accum / accum_n
        loss_trace.append(mloss)
    else:
        mloss = loss_trace[-1] if loss_trace else float("nan")
    save_adapter(policy, out_dir, meta={"step": macro, "loss": mloss,
                                        "beta": args.beta,
                                        "samples": len(samples)})
    print(f"[dpo] 完成：loss={mloss:.4f}  adapter -> {out_dir}")
    print("提示：reword margin 应随训练上升（chosen/rejected 被拉开）。"
          "用 src/compare.py 对比 base / SFT / DPO。")


if __name__ == "__main__":
    main()
