# -*- coding: utf-8 -*-
"""
plot_results.py —— 把实验结果画成图（任务三：图分类）

生成五张图到 ../results/ 目录：

    图1  fig1_model_comparison.png  四个模型在五个数据集上的表现（柱状图）
    图2  fig2_pooling_effect.png    池化方法的影响（每个数据集一个子图）
    图3  fig3_full_vs_batch.png     全图训练 vs 分批次训练：精度 & 耗时
    图4  fig4_layers_effect.png     网络层数的影响（折线图）
    图5  fig5_lr_effect.png         学习率的影响（折线图）

同时会在终端打印几张汇总表。

【关于"越大越好"和"越小越好"】
    TU 系列数据集是分类任务，指标是**准确率**，越大越好；
    ZINC 是回归任务，指标是 **MAE**，越小越好。
    画图的时候如果混在一起会看不懂，所以图 1 分成两栏：
    左边画分类准确率，右边画 ZINC 的 MAE。

【怎么运行？】
    python plot_results.py
    # 前提：先跑过 run_all.py
"""

import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")           # 不弹窗口，直接存成图片文件（服务器上也能跑）
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from datasets import TU_DATASETS, ZINC_NAME     # noqa: E402
from models import MODEL_NAMES                  # noqa: E402
from utils import RESULT_DIR, load_results      # noqa: E402

# 中文字体设置（Windows 上用微软雅黑，Linux 上退回到黑体/DejaVu）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

MODEL_COLORS = {"GCN": "#4C72B0", "GAT": "#DD8452",
                "GraphSAGE": "#55A868", "GIN": "#C44E52"}
POOL_COLORS = {"avg": "#4C72B0", "max": "#DD8452",
               "min": "#55A868", "sum": "#C44E52"}

ALL_DATASETS = TU_DATASETS + [ZINC_NAME]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def dedup(records):
    """同一组配置跑过多次时，只保留最后一次的结果（方便重跑某个实验后更新）。"""
    best = OrderedDict()
    for r in records:
        key = (r["dataset"], r["model"], r["pooling"], r["mode"],
               r["lr"], r["layers"])
        best[key] = r
    return list(best.values())


def find(records, **kw):
    """按字段精确查找一条记录，找到就返回，找不到返回 None。"""
    for r in records:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def metric_of(r):
    """
    取一条记录的主指标数值。分类返回准确率(0~1)，回归返回 MAE。

    【为什么要先判断 None？】
        find() 在"这条配置还没跑"的时候返回 None。如果直接取 r["best_test_metric"]
        就会抛 TypeError: 'NoneType' object is not subscriptable。

        而"还没跑完就出图"是很正常的情况（比如你想先看看已经跑完的那部分结果），
        所以这里返回 None，让调用方自己决定怎么处理（跳过、或者显示成 '--'）。
    """
    if r is None:
        return None
    return r["best_test_metric"]


def metric_text(r):
    """把指标格式化成好读的字符串。"""
    if r["task_type"] == "classification":
        return f"{r['best_test_metric'] * 100:.2f}%"
    return f"{r['best_test_metric']:.4f}"


def is_regression(ds):
    return ds == ZINC_NAME


# ---------------------------------------------------------------------------
# 图1：模型对比
# ---------------------------------------------------------------------------
def plot_model_comparison(records):
    """
    左图：4 个 TU 数据集上的分类准确率（4 模型 × 4 数据集）
    右图：ZINC 上的 MAE（4 模型）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    # ---- 左：分类准确率 ----
    ax = axes[0]
    x = np.arange(len(TU_DATASETS))
    w = 0.2
    for i, model in enumerate(MODEL_NAMES):
        vals = []
        for ds in TU_DATASETS:
            r = find(records, dataset=ds, model=model, pooling="avg",
                     mode="sample", lr=0.01)
            if r is None:
                r = find(records, dataset=ds, model=model, mode="sample")
            vals.append(metric_of(r) * 100 if r else 0.0)
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=model,
                      color=MODEL_COLORS[model])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TU_DATASETS, fontsize=9)
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("分类数据集：准确率（越高越好）")
    ax.set_ylim(0, 105)
    # 二分类的随机猜测基线是 50%，六分类的 ENZYMES 是 100/6≈16.7%
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.02, 51.5, "二分类随机基线 50%", transform=ax.transData,
            fontsize=7.5, color="gray")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # ---- 右：ZINC 回归 MAE ----
    ax = axes[1]
    vals, names, colors = [], [], []
    for model in MODEL_NAMES:
        r = find(records, dataset=ZINC_NAME, model=model, pooling="avg",
                 mode="sample", lr=0.01)
        if r is None:
            r = find(records, dataset=ZINC_NAME, model=model, mode="sample")
        vals.append(metric_of(r) if r else 0.0)
        names.append(model)
        colors.append(MODEL_COLORS[model])

    bars = ax.bar(np.arange(len(names)), vals, 0.6, color=colors)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("测试 MAE（越低越好）")
    ax.set_title(f"回归数据集 {ZINC_NAME}：MAE")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("图1  四种 GNN 模型在不同图数据集上的表现"
                 "（平均池化 + 分批次训练）", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig1_model_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图2：池化方法的影响（任务三的重点）
# ---------------------------------------------------------------------------
def plot_pooling_effect(records):
    """
    每个数据集一个子图：横轴是模型，三种颜色代表三种池化方法。
    这样一眼就能看出"换了池化方法，性能差多少"。
    """
    poolings = ["avg", "max", "min"]
    datasets = [ds for ds in ALL_DATASETS
                if any(find(records, dataset=ds, pooling=p) for p in poolings)]
    if not datasets:
        print("  [跳过] 没有池化对比数据，请先运行：python run_all.py --stage pool")
        return

    n = len(datasets)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, ds in zip(axes, datasets):
        models = [m for m in MODEL_NAMES
                  if any(find(records, dataset=ds, model=m, pooling=p)
                         for p in poolings)]
        x = np.arange(len(models))
        w = 0.26
        for j, pooling in enumerate(poolings):
            vals = []
            for model in models:
                r = find(records, dataset=ds, model=model, pooling=pooling,
                         mode="sample")
                vals.append(metric_of(r) if r else 0.0)
            bars = ax.bar(x + (j - 1) * w, vals, w,
                          label=f"{pooling.upper()}Pool",
                          color=POOL_COLORS[pooling])
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=15)
        if is_regression(ds):
            ax.set_ylabel("测试 MAE（越低越好）")
            ax.set_title(f"{ds}（回归）")
        else:
            ax.set_ylabel("测试准确率（0~1）")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{ds}（分类）")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    # 用不到的格子留白
    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle("图2  三种池化方法（Avg / Max / Min）对图分类性能的影响",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULT_DIR, "fig2_pooling_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图3：全图训练 vs 分批次训练
# ---------------------------------------------------------------------------
def plot_full_vs_batch(records):
    """三栏柱状图：精度、每轮耗时、总耗时。"""
    datasets = ["MUTAG", "PROTEINS", "IMDB-BINARY"]
    labels, m_full, m_samp, t_full, t_samp, tot_full, tot_samp = \
        [], [], [], [], [], [], []

    for ds in datasets:
        for model in MODEL_NAMES:
            rf = find(records, dataset=ds, model=model, mode="full", pooling="avg")
            rs = find(records, dataset=ds, model=model, mode="sample", pooling="avg")
            if rf and rs:
                labels.append(f"{ds}\n{model}")
                m_full.append(metric_of(rf))
                m_samp.append(metric_of(rs))
                t_full.append(max(rf["train_time_per_epoch"], 1e-4))
                t_samp.append(max(rs["train_time_per_epoch"], 1e-4))
                tot_full.append(max(rf["train_time_total"], 1e-4))
                tot_samp.append(max(rs["train_time_total"], 1e-4))

    if not labels:
        print("  [跳过] 缺少全图/分批对比数据，请先运行：python run_all.py --stage batch")
        return

    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.bar(x - w / 2, [v * 100 for v in m_full], w, label="全图训练", color="#4C72B0")
    ax.bar(x + w / 2, [v * 100 for v in m_samp], w, label="分批次训练", color="#DD8452")
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("精度对比")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 105)

    ax = axes[1]
    ax.bar(x - w / 2, t_full, w, label="全图训练", color="#4C72B0")
    ax.bar(x + w / 2, t_samp, w, label="分批次训练", color="#DD8452")
    ax.set_ylabel("每一轮训练耗时 (秒，对数刻度)")
    ax.set_yscale("log")
    ax.set_title("每轮训练耗时\n（全图训练一轮只更新 1 次参数，所以快得多）")

    ax = axes[2]
    ax.bar(x - w / 2, tot_full, w, label="全图训练", color="#4C72B0")
    ax.bar(x + w / 2, tot_samp, w, label="分批次训练", color="#DD8452")
    ax.set_ylabel("完整训练总耗时 (秒，对数刻度)")
    ax.set_yscale("log")
    ax.set_title("总训练耗时\n（注意全图训练的轮数是分批次的 4 倍）")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("图3  全图训练 vs 分批次训练（图分类）", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig3_full_vs_batch.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图4：网络层数的影响
# ---------------------------------------------------------------------------
def plot_layers_effect(records):
    layers_list = [1, 2, 3, 4]
    datasets = ["MUTAG", "PROTEINS"]
    models = ["GCN", "GIN"]

    avail = [(ds, m) for ds in datasets for m in models
             if any(find(records, dataset=ds, model=m, layers=L) for L in layers_list)]
    if not avail:
        print("  [跳过] 没有层数消融数据，请先运行：python run_all.py --stage layers")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"GCN": "o-", "GIN": "s--"}
    colors = {"MUTAG": "#4C72B0", "PROTEINS": "#DD8452",
              "IMDB-BINARY": "#55A868", "ENZYMES": "#C44E52"}
    for ds, model in avail:
        xs, ys = [], []
        for L in layers_list:
            r = find(records, dataset=ds, model=model, layers=L, pooling="avg")
            if r:
                xs.append(L)
                ys.append(metric_of(r))
        if xs:
            ax.plot(xs, ys, styles.get(model, "o-"), label=f"{ds} + {model}",
                    color=colors.get(ds, None), linewidth=2, markersize=7)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy, f" {yy:.2f}", fontsize=8, va="bottom")

    ax.set_xticks(layers_list)
    ax.set_xlabel("GNN 卷积层数")
    ax.set_ylabel("测试准确率（0~1，越高越好）")
    ax.set_title("图4  网络层数对图分类性能的影响")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = os.path.join(RESULT_DIR, "fig4_layers_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 图5：学习率的影响
# ---------------------------------------------------------------------------
def plot_lr_effect(records):
    lrs = [0.001, 0.005, 0.01, 0.05]
    datasets = ["MUTAG", "PROTEINS"]
    models = ["GCN", "GIN"]

    avail = [(ds, m) for ds in datasets for m in models
             if any(find(records, dataset=ds, model=m, lr=lr) for lr in lrs)]
    if not avail:
        print("  [跳过] 没有学习率消融数据，请先运行：python run_all.py --stage lr")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"GCN": "o-", "GIN": "s--"}
    colors = {"MUTAG": "#4C72B0", "PROTEINS": "#DD8452"}
    for ds, model in avail:
        xs, ys = [], []
        for lr in lrs:
            r = find(records, dataset=ds, model=model, lr=lr, pooling="avg")
            if r:
                xs.append(lr)
                ys.append(metric_of(r))
        if xs:
            ax.plot(xs, ys, styles.get(model, "o-"), label=f"{ds} + {model}",
                    color=colors.get(ds, None), linewidth=2, markersize=7)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy, f" {yy:.2f}", fontsize=8, va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("学习率 (log scale)")
    ax.set_ylabel("测试准确率（0~1，越高越好）")
    ax.set_title("图5  学习率对图分类性能的影响")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = os.path.join(RESULT_DIR, "fig5_lr_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


# ---------------------------------------------------------------------------
# 终端汇总表
# ---------------------------------------------------------------------------
def print_table(records):
    print("\n" + "=" * 118)
    print("表1  主实验汇总（平均池化 + 分批次训练）")
    print("=" * 118)
    print(f"{'数据集':<14}{'模型':<11}{'任务':<8}{'验证指标':>10}{'测试指标':>10}"
          f"{'参数量':>10}{'批次/轮':>9}{'每轮耗时':>10}{'训练总耗时':>11}")
    print("-" * 118)
    for ds in ALL_DATASETS:
        for model in MODEL_NAMES:
            r = find(records, dataset=ds, model=model, pooling="avg", mode="sample")
            if not r:
                continue
            print(f"{r['dataset']:<14}{r['model']:<11}{r['task_type'][:4]:<8}"
                  f"{r['best_val_metric']:>10.4f}{r['best_test_metric']:>10.4f}"
                  f"{r['n_params']:>10,}{r['n_batches_per_epoch']:>9}"
                  f"{r['train_time_per_epoch']:>9.4f}s"
                  f"{r['train_time_total']:>10.2f}s")
    print("=" * 118)

    # ---- 池化对比汇总：按数据集求"四个模型的平均值" ----
    rows = []
    for ds in ALL_DATASETS:
        vals = {}
        for pooling in ["avg", "max", "min"]:
            ms = [metric_of(find(records, dataset=ds, model=m, pooling=pooling,
                                 mode="sample"))
                  for m in MODEL_NAMES]
            ms = [v for v in ms if v is not None]
            if ms:
                vals[pooling] = float(np.mean(ms))
        if vals:
            rows.append((ds, vals))

    if rows:
        print("\n" + "=" * 96)
        print("表2  池化方法对比（表格里是四个模型在该数据集上的**平均**测试指标）")
        print("=" * 96)
        print(f"{'数据集':<14}{'AvgPooling':>14}{'MaxPooling':>14}{'MinPooling':>14}"
              f"{'最好的池化':>14}")
        print("-" * 96)
        for ds, vals in rows:
            r = find(records, dataset=ds)
            reg = r and r["task_type"] == "regression"
            # 回归是 MAE 越小越好，分类是准确率越大越好
            best = min(vals, key=vals.get) if reg else max(vals, key=vals.get)
            cells = "".join(f"{vals.get(p, float('nan')):>14.4f}"
                            for p in ["avg", "max", "min"])
            print(f"{ds:<14}{cells}{best:>14}")
        print("-" * 96)
        if any(find(records, dataset=ds) and
               find(records, dataset=ds)["task_type"] == "regression"
               for ds, _ in rows):
            print("提示：ZINC 是回归任务，MAE 越低越好；其他是分类任务，准确率越高越好。")
        print("=" * 96)

    # ---- 全图 vs 分批 ----
    pairs = []
    for ds in ["MUTAG", "PROTEINS", "IMDB-BINARY"]:
        for model in MODEL_NAMES:
            rf = find(records, dataset=ds, model=model, mode="full", pooling="avg")
            rs = find(records, dataset=ds, model=model, mode="sample", pooling="avg")
            if rf and rs:
                pairs.append((ds, model, rf, rs))

    if pairs:
        print("\n" + "=" * 116)
        print("表3  全图训练 vs 分批次训练")
        print("=" * 116)
        print(f"{'数据集':<12}{'模型':<11}{'方式':<8}{'轮数':>6}{'测试准确率':>12}"
              f"{'每轮耗时':>11}{'训练总耗时':>12}{'总耗时倍数':>12}")
        print("-" * 116)
        for ds, model, rf, rs in pairs:
            ratio = rf["train_time_total"] / max(rs["train_time_total"], 1e-6)
            for r in (rf, rs):
                label = "全图" if r["mode"] == "full" else "分批次"
                print(f"{ds:<12}{model:<11}{label:<8}{r['epochs_run']:>6}"
                      f"{r['best_test_metric']*100:>11.2f}%"
                      f"{r['train_time_per_epoch']:>10.4f}s"
                      f"{r['train_time_total']:>11.2f}s"
                      + (f"{ratio:>11.2f}x" if r["mode"] == "full" else ""))
        print("=" * 116)


def main():
    records = load_results("results.jsonl")
    if not records:
        print("没有找到实验结果。请先运行：python run_all.py --stage main")
        return 1

    records = dedup(records)
    print(f"共读取到 {len(records)} 条实验记录（已去重）\n")
    print("正在生成图表……")
    plot_model_comparison(records)
    plot_pooling_effect(records)
    plot_full_vs_batch(records)
    plot_layers_effect(records)
    plot_lr_effect(records)
    print_table(records)
    print(f"\n所有图片已保存到：{RESULT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
