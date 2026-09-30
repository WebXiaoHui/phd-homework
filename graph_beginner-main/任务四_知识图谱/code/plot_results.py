# -*- coding: utf-8 -*-
"""
plot_results.py —— 把实验结果画成图（任务四：知识图谱）

生成四张图到 ../results/ 目录：

    图1  fig1_model_comparison.png  三个模型在两个数据集上的 MRR / Hits@1 / Hits@10
    图2  fig2_dim_effect.png        embedding 维度的影响（折线图）
    图3  fig3_lr_effect.png         学习率的影响（折线图）
    图4  fig4_time.png              训练与评估耗时对比

同时打印一张汇总表（MRR / Hits@1 / Hits@3 / Hits@10 / 平均排名 / 耗时）。

【关于"层数"的说明】
    报告要求分析"网络层数"的影响。但任务四的三个模型都没有"层数"这个超参数：
        TransE 就是 h + r ≈ t 一个公式，没有层
        RotatE 就是复数乘法 h ∘ r ≈ t，也没有层
        ConvE 是"一层卷积 + 一层全连接"，改层数不是它的常规用法
    所以本任务用 **embedding 维度（dim）** 作为"模型容量"的等价消融，
    对应图 2。README 里也做了说明。

【怎么运行？】
    python plot_results.py
"""

import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from datasets import DATASET_NAMES     # noqa: E402
from models import DEFAULT_DIM, MODEL_NAMES   # noqa: E402
from utils import RESULT_DIR, load_results    # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

MODEL_COLORS = {"TransE": "#4C72B0", "RotatE": "#DD8452", "ConvE": "#55A868"}


def dedup(records):
    """同一组配置跑过多次时，只保留最后一次的结果。"""
    best = OrderedDict()
    for r in records:
        key = (r["dataset"], r["model"], r["dim"], r["lr"], r["loss"])
        best[key] = r
    return list(best.values())


def find(records, **kw):
    for r in records:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


# ---------------------------------------------------------------------------
# 图1：模型对比
# ---------------------------------------------------------------------------
def plot_model_comparison(records):
    """每个数据集一个子图，横轴是模型，三种颜色代表 MRR / Hits@1 / Hits@10。"""
    datasets = [ds for ds in DATASET_NAMES
                if any(r["dataset"] == ds for r in records)]
    if not datasets:
        print("  [跳过] 没有主实验数据")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, datasets):
        models = [m for m in MODEL_NAMES
                  if find(records, dataset=ds, model=m, dim=DEFAULT_DIM[m])
                  or any(r["dataset"] == ds and r["model"] == m for r in records)]
        x = np.arange(len(models))
        w = 0.26
        specs = [("test_mrr", "MRR", "#4C72B0"),
                 ("test_hits@1", "Hits@1", "#DD8452"),
                 ("test_hits@10", "Hits@10", "#55A868")]

        for j, (key, label, color) in enumerate(specs):
            vals = []
            for m in models:
                r = find(records, dataset=ds, model=m, dim=DEFAULT_DIM[m]) \
                    or find(records, dataset=ds, model=m)
                vals.append(r[key] if r else 0.0)
            bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=color)
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel("测试指标（越高越好）")
        r0 = find(records, dataset=ds, model=models[0]) if models else None
        extra = ""
        if r0:
            extra = (f"\n实体数 {r0['num_entities']}，关系数 {r0['num_relations']}，"
                     f"测试集评估 {r0['n_params']:,} 个参数的模型")
        ax.set_title(f"{ds}{extra}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("图1  三种知识图谱补全模型的链接预测性能（过滤式评估）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULT_DIR, "fig1_model_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图2：embedding 维度的影响
# ---------------------------------------------------------------------------
def plot_dim_effect(records):
    dims = sorted({r["dim"] for r in records})
    if len(dims) < 2:
        print("  [跳过] 没有维度消融数据，请先运行：python run_all.py --stage dim")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for model in MODEL_NAMES:
        xs, ys = [], []
        for d in dims:
            r = find(records, model=model, dim=d, lr=0.001, loss="bce") \
                or find(records, model=model, dim=d)
            if r:
                xs.append(d)
                ys.append(r["test_mrr"])
        if xs:
            ax.plot(xs, ys, "o-", label=model, color=MODEL_COLORS[model],
                    linewidth=2, markersize=7)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy, f" {yy:.3f}", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("embedding 维度 dim（对数刻度）")
    ax.set_ylabel("测试 MRR（越高越好）")
    ax.set_title("embedding 维度 vs MRR")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax = axes[1]
    for model in MODEL_NAMES:
        xs, ys = [], []
        for d in dims:
            r = find(records, model=model, dim=d, lr=0.001, loss="bce") \
                or find(records, model=model, dim=d)
            if r:
                xs.append(d)
                ys.append(r["test_hits@10"])
        if xs:
            ax.plot(xs, ys, "s-", label=model, color=MODEL_COLORS[model],
                    linewidth=2, markersize=7)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy, f" {yy:.3f}", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("embedding 维度 dim（对数刻度）")
    ax.set_ylabel("测试 Hits@10（越高越好）")
    ax.set_title("embedding 维度 vs Hits@10")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.suptitle("图2  embedding 维度对知识图谱补全性能的影响"
                 "（在 WN18RR 上做消融）", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig2_dim_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图3：学习率的影响
# ---------------------------------------------------------------------------
def plot_lr_effect(records):
    lrs = sorted({r["lr"] for r in records})
    if len(lrs) < 2:
        print("  [跳过] 没有学习率消融数据，请先运行：python run_all.py --stage lr")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ylabel in [(axes[0], "test_mrr", "测试 MRR（越高越好）"),
                            (axes[1], "test_hits@10", "测试 Hits@10（越高越好）")]:
        for model in MODEL_NAMES:
            xs, ys = [], []
            for lr in lrs:
                r = find(records, model=model, lr=lr, dim=DEFAULT_DIM[model])
                if r:
                    xs.append(lr)
                    ys.append(r[key])
            if xs:
                ax.plot(xs, ys, "o-", label=model, color=MODEL_COLORS[model],
                        linewidth=2, markersize=7)
                for xx, yy in zip(xs, ys):
                    ax.text(xx, yy, f" {yy:.3f}", fontsize=8, va="bottom")
        ax.set_xscale("log")
        ax.set_xticks(lrs)
        ax.set_xticklabels([f"{l:g}" for l in lrs])
        ax.set_xlabel("学习率（对数刻度）")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("图3  学习率对知识图谱补全性能的影响（在 WN18RR 上做消融）",
                 fontsize=13)
    out = os.path.join(RESULT_DIR, "fig3_lr_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图4：耗时对比
# ---------------------------------------------------------------------------
def plot_time(records):
    rows = [(ds, m) for ds in DATASET_NAMES for m in MODEL_NAMES
            if find(records, dataset=ds, model=m, dim=DEFAULT_DIM[m])
            or find(records, dataset=ds, model=m)]
    if not rows:
        print("  [跳过] 缺少主实验数据")
        return

    labels = [f"{ds}\n{m}" for ds, m in rows]
    per_epoch, total = [], []
    for ds, m in rows:
        r = find(records, dataset=ds, model=m, dim=DEFAULT_DIM[m]) \
            or find(records, dataset=ds, model=m)
        per_epoch.append(max(r["train_time_per_epoch"], 1e-3))
        total.append(max(r["train_time_total"], 1e-3))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = [MODEL_COLORS[m] for _, m in rows]
    bars = ax.bar(x, per_epoch, 0.6, color=colors)
    for b, v in zip(bars, per_epoch):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("每轮训练耗时 (秒)")
    ax.set_title("平均每轮训练耗时")

    ax = axes[1]
    bars = ax.bar(x, total, 0.6, color=colors)
    for b, v in zip(bars, total):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("完整训练总耗时 (秒)")
    ax.set_title("完整训练总耗时")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("图4  三种模型的训练耗时对比", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(RESULT_DIR, "fig4_time.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------
def print_table(records):
    print("\n" + "=" * 126)
    print("知识图谱补全实验结果汇总（过滤式评估，两个方向取平均）")
    print("=" * 126)
    print(f"{'数据集':<12}{'模型':<9}{'dim':>5}{'lr':>8}{'损失':>7}"
          f"{'MRR':>9}{'Hits@1':>9}{'Hits@3':>9}{'Hits@10':>9}{'平均排名':>10}"
          f"{'参数量':>12}{'每轮耗时':>10}{'总耗时':>10}")
    print("-" * 126)
    for ds in DATASET_NAMES:
        for model in MODEL_NAMES:
            r = find(records, dataset=ds, model=model, dim=DEFAULT_DIM[model],
                     lr=0.001) or find(records, dataset=ds, model=model)
            if not r:
                continue
            print(f"{r['dataset']:<12}{r['model']:<9}{r['dim']:>5}{r['lr']:>8.4f}"
                  f"{r['loss']:>7}{r['test_mrr']:>9.4f}{r['test_hits@1']:>9.4f}"
                  f"{r['test_hits@3']:>9.4f}{r['test_hits@10']:>9.4f}"
                  f"{r['test_mean_rank']:>10.1f}{r['n_params']:>12,}"
                  f"{r['train_time_per_epoch']:>9.2f}s"
                  f"{r['train_time_total']:>9.1f}s")
    print("=" * 126)

    # ---- 维度消融汇总 ----
    dims = sorted({r["dim"] for r in records})
    if len(dims) >= 2:
        print("\n" + "=" * 88)
        print("维度消融：测试 MRR（WN18RR）")
        print("=" * 88)
        header = f"{'模型':<10}" + "".join(f"{'dim=' + str(d):>14}" for d in dims)
        print(header)
        print("-" * 88)
        for model in MODEL_NAMES:
            cells = ""
            found = False
            for d in dims:
                r = find(records, model=model, dim=d, lr=0.001)
                if r:
                    cells += f"{r['test_mrr']:>14.4f}"
                    found = True
                else:
                    cells += f"{'--':>14}"
            if found:
                print(f"{model:<10}{cells}")
        print("=" * 88)

    # ---- 学习率消融汇总 ----
    lrs = sorted({r["lr"] for r in records})
    if len(lrs) >= 2:
        print("\n" + "=" * 88)
        print("学习率消融：测试 MRR（WN18RR）")
        print("=" * 88)
        print(f"{'模型':<10}" + "".join(f"{'lr=' + f'{l:g}':>14}" for l in lrs))
        print("-" * 88)
        for model in MODEL_NAMES:
            cells = ""
            found = False
            for lr in lrs:
                r = find(records, model=model, lr=lr, dim=DEFAULT_DIM[model])
                if r:
                    cells += f"{r['test_mrr']:>14.4f}"
                    found = True
                else:
                    cells += f"{'--':>14}"
            if found:
                print(f"{model:<10}{cells}")
        print("=" * 88)


def main():
    records = load_results("results.jsonl")
    if not records:
        print("没有找到实验结果。请先运行：python run_all.py --stage main")
        return 1

    records = dedup(records)
    print(f"共读取到 {len(records)} 条实验记录（已去重）\n")
    print("正在生成图表……")
    plot_model_comparison(records)
    plot_dim_effect(records)
    plot_lr_effect(records)
    plot_time(records)
    print_table(records)
    print(f"\n所有图片已保存到：{RESULT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
