"""任务一 · 注意力可视化（DoD M5，交付 >= 3 张热图）。

先训练出 ckpt/best.pt 再运行本脚本：
    python visualize_attention.py

默认在 validation 集挑 3 类样本画注意力热图（到 figures/ 下）：
    1. 一条正面样本（label=1）
    2. 一条负面样本（label=0）
    3. 一条长句样本
每个都取第 --layer 层的注意力权重，--head 可指定某头或 'mean' 平均多头。

用法：
    python visualize_attention.py                       # 默认
    python visualize_attention.py --layer 0 --head mean
    python visualize_attention.py --head 0              # 单独看第 0 个头
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from src.model import load_for_eval

try:
    import matplotlib
    matplotlib.use("Agg")          # 无显示器环境下也能存图
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError as e:
    raise SystemExit("[错误] 需要 matplotlib：pip install matplotlib") from e


def setup_cjk_font():
    """尽量挑一个能显示中文的字体，找不到则退回默认并提示。"""
    candidates = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
                  "SimHei", "Microsoft YaHei", "PingFang SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            return name
    plt.rcParams["axes.unicode_minus"] = False
    print("[warn] 未找到中文字体，坐标轴词元可能显示为方块（可安装 Noto Sans CJK）")
    return None


def pick_samples(dev):
    """返回 [(名字, text, true_label), ...]：正面 / 负面 / 最长句各一。"""
    texts, labels = dev["text"].tolist(), dev["label"].tolist()

    def find(pred):
        for i, (t, y) in enumerate(zip(texts, labels)):
            if pred(int(y)):
                return t, int(y)
        return texts[0], int(labels[0])

    pos_t, pos_y = find(lambda y: y == 1)
    neg_t, neg_y = find(lambda y: y == 0)
    longest_i = max(range(len(texts)), key=lambda i: len(texts[i]))
    return [("positive", pos_t, pos_y), ("negative", neg_t, neg_y),
            ("longest", texts[longest_i], int(labels[longest_i]))]


def extract_weights(model, tokenizer, text, layer_idx):
    """前向一次，返回第 layer_idx 层的注意力权重 (H, T, T) 与 tokens。"""
    ids = tokenizer.encode(text)
    tokens = tokenizer.tokens_of(text)          # 去掉 <pad> 的字符列表
    n = len(tokens)
    with torch.no_grad():
        model(ids.unsqueeze(0))                 # 触发各 block 记录 last_attn_weights
    w = model.blocks[layer_idx].attn.last_attn_weights  # (1, H, T, T)
    return w[0, :, :n, :n].cpu().numpy(), tokens, n


def save_heatmap(w2d, tokens, title, path):
    n = w2d.shape[0]
    plt.figure(figsize=(max(7, n * 0.28), max(6, n * 0.24)))
    im = plt.imshow(w2d, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="attention prob")
    plt.xticks(range(n), tokens, rotation=90, fontsize=8)
    plt.yticks(range(n), tokens, fontsize=8)
    plt.xlabel("key")
    plt.ylabel("query")
    plt.title(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--data", default="data/validation.parquet")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--layer", type=int, default=0, help="取哪一层的注意力")
    ap.add_argument("--head", default="mean",
                    help="平均所有头用 'mean'，或给整数选某一头，如 0")
    args = ap.parse_args()

    setup_cjk_font()
    model, tokenize_fn = load_for_eval(args.ckpt)
    # 坐标轴要显示原始字符，需从 ckpt 里恢复 vocab 重建 CharTokenizer
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    from src.tokenizer import CharTokenizer
    tokenizer = CharTokenizer(ckpt["vocab"], max_len=ckpt["config"]["max_len"])

    n_layers = len(model.blocks)
    if not (0 <= args.layer < n_layers):
        raise SystemExit(f"[错误] --layer 超出范围 [0, {n_layers})")

    dev = pd.read_parquet(args.data)
    samples = pick_samples(dev)

    head_cfg = args.head
    for name, text, true_y in samples:
        w3d, tokens, n = extract_weights(model, tokenizer, text, args.layer)
        if head_cfg == "mean":
            w2d = w3d.mean(axis=0)
            head_tag = "mean-heads"
        else:
            head_idx = int(head_cfg)
            if not (0 <= head_idx < w3d.shape[0]):
                raise SystemExit(f"[错误] --head 超出范围 [0, {w3d.shape[0]})")
            w2d = w3d[head_idx]
            head_tag = f"head{head_idx}"

        with torch.no_grad():
            logits = model(tokenize_fn(text).unsqueeze(0))
        pred = int(logits.argmax(-1).item())

        path = Path(args.outdir) / f"attn_{name}_layer{args.layer}_{head_tag}.png"
        save_heatmap(w2d, tokens,
                     f"{name}  layer={args.layer}  head={head_tag}  "
                     f"true={true_y} pred={pred}  ({len(tokens)} chars)",
                     path)

        # 辅助观察：每行 query 注意力最重的 key 词
        focus = sorted(range(n),
                       key=lambda j: w2d.mean(axis=0)[j], reverse=True)[:8]
        print(f"    {name}: 平均注意力最重的 top key 词 = "
              f"{' '.join(tokens[j] for j in focus)}")

    print(f"\n完成，热图已保存到 {args.outdir}/（>=3 张）。\n"
          f"建议比较 positive / negative 两张，看模型是否把注意力放在"
          f"「不错 / 失望」这类情感词上。")


if __name__ == "__main__":
    main()
