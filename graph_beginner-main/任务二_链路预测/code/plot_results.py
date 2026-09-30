# -*- coding: utf-8 -*-
"""
plot_results.py —— 把实验结果画成图（任务二：链路预测）

生成四张图到 ../results/ 目录：

    图1  model_comparison.png   四个模型在三个数据集上的 AUC（柱状图）
    图2  full_vs_sample.png     全图训练 vs 采样训练：AUC & 耗时（三栏柱状图）
    图3  lr_effect.png          学习率对 AUC 的影响（折线图）
    图4  layers_effect.png      层数对 AUC 的影响（折线图）

【怎么运行？】
    python plot_results.py
    # 前提：先跑过 run_all.py
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

from utils import RESULT_DIR, load_results   # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

MODEL_COLORS = {"GCN": "#4C72B0", "GAT": "#DD8452",
                "GraphSAGE": "#55A868", "GIN": "#C44E52"}
DATASETS = ["Cora", "Citeseer", "Flickr"]
MODELS = ["GCN", "GAT", "GraphSAGE", "GIN"]


def dedup(records):
    """同一组配置跑过多次时，只保留最后一次的结果。"""
    best = OrderedDict()
    for r in records:
        key = (r["dataset"], r["model"], r["mode"], r["decoder"],
               r["lr"], r["layers"])
        best[key] = r
    return list(best.values())


def find(records, **kw):
    for r in records:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


# ---------------------------------------------------------------------------
def plot_model_comparison(records):
    """图1：模型对比柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mode, title in zip(
            axes, ["full", "sample"],
            ["全图训练 (full-batch)", "子图采样训练 (mini-batch)"]):
        x = np.arange(len(DATASETS))
        width = 0.2
        for i, model in enumerate(MODELS):
            vals = []
            for ds in DATASETS:
                r = find(records, dataset=ds, model=model, mode=mode,
                         decoder="dot", lr=0.01, layers=2)
                if r is None:
                    r = find(records, dataset=ds, model=model, mode=mode)
                vals.append(r["best_test_auc"] * 100 if r else 0)

            bars = ax.bar(x + (i - 1.5) * width, vals, width,
                          label=model, color=MODEL_COLORS[model])
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel("测试 AUC (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(0.02, 51, "随机猜测基线 AUC=50%", transform=ax.transData,
                fontsize=7.5, color="gray")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("图1  四种 GNN 模型在三个数据集上的链路预测 AUC", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig1_model_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


def plot_full_vs_sample(records):
    """图2：全图 vs 采样，比 AUC、每轮耗时、总耗时。"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    labels, acc_f, acc_s, t_f, t_s, tot_f, tot_s = [], [], [], [], [], [], []
    for ds in DATASETS:
        for model in MODELS:
            rf = find(records, dataset=ds, model=model, mode="full")
            rs = find(records, dataset=ds, model=model, mode="sample")
            if rf and rs:
                labels.append(f"{ds}\n{model}")
                acc_f.append(rf["best_test_auc"] * 100)
                acc_s.append(rs["best_test_auc"] * 100)
                t_f.append(max(rf["train_time_per_epoch"], 1e-4))
                t_s.append(max(rs["train_time_per_epoch"], 1e-4))
                tot_f.append(max(rf["train_time_total"], 1e-4))
                tot_s.append(max(rs["train_time_total"], 1e-4))

    if not labels:
        print("  [跳过] 缺少主实验数据")
        return

    x = np.arange(len(labels))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, acc_f, w, label="全图训练", color="#4C72B0")
    ax.bar(x + w / 2, acc_s, w, label="采样训练", color="#DD8452")
    ax.set_ylabel("测试 AUC (%)")
    ax.set_title("准确率对比")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)

    ax = axes[1]
    ax.bar(x - w / 2, t_f, w, label="全图训练", color="#4C72B0")
    ax.bar(x + w / 2, t_s, w, label="采样训练", color="#DD8452")
    ax.set_ylabel("每轮训练耗时 (秒，对数刻度)")
    ax.set_yscale("log")
    ax.set_title("每轮训练耗时\n(采样一轮要做很多个 batch)")

    ax = axes[2]
    ax.bar(x - w / 2, tot_f, w, label="全图训练", color="#4C72B0")
    ax.bar(x + w / 2, tot_s, w, label="采样训练", color="#DD8452")
    ax.set_ylabel("完整训练总耗时 (秒，对数刻度)")
    ax.set_yscale("log")
    ax.set_title("总训练耗时")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("图2  全图训练 vs 子图采样训练（链路预测）", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig2_full_vs_sample.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


def plot_lr_effect(records):
    """图3：学习率影响。"""
    lrs = [0.001, 0.005, 0.01, 0.05]
    dss = [ds for ds in ["Cora", "Flickr"]
           if any(r["dataset"] == ds and r["model"] == "GCN"
                  and r["mode"] == "full" and r["layers"] == 2 for r in records)]
    if not dss:
        print("  [跳过] 没有学习率消融数据，请先运行：python run_all.py --stage lr")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Cora": "#4C72B0", "Flickr": "#DD8452"}
    for ds in dss:
        xs, ys = [], []
        for lr in lrs:
            r = find(records, dataset=ds, model="GCN", mode="full",
                     decoder="dot", lr=lr, layers=2)
            if r:
                xs.append(lr)
                ys.append(r["best_test_auc"] * 100)
        if xs:
            ax.plot(xs, ys, "o-", label=f"{ds} (GCN, 2层)",
                    color=colors.get(ds), linewidth=2, markersize=7)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy + 0.8, f"{yy:.1f}", ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("学习率 (log scale)")
    ax.set_ylabel("测试 AUC (%)")
    ax.set_title("图3  学习率对链路预测性能的影响")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = os.path.join(RESULT_DIR, "fig3_lr_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


def plot_layers_effect(records):
    """图4：层数影响。"""
    layer_list = [2, 3, 4]
    dss = [ds for ds in ["Cora", "Flickr"]
           if any(r["dataset"] == ds and r["model"] == "GCN" and r["mode"] == "full"
                  and r["lr"] == 0.01 and r["layers"] in layer_list for r in records)]
    if not dss:
        print("  [跳过] 没有层数消融数据，请先运行：python run_all.py --stage layers")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Cora": "#4C72B0", "Flickr": "#DD8452"}
    for ds in dss:
        xs, ys = [], []
        for L in layer_list:
            r = find(records, dataset=ds, model="GCN", mode="full",
                     decoder="dot", lr=0.01, layers=L)
            if r:
                xs.append(L)
                ys.append(r["best_test_auc"] * 100)
        if xs:
            ax.plot(xs, ys, "s-", label=f"{ds} (GCN)",
                    color=colors.get(ds), linewidth=2, markersize=8)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy + 0.8, f"{yy:.1f}", ha="center", fontsize=8)

    ax.set_xticks(layer_list)
    ax.set_xlabel("GNN 层数")
    ax.set_ylabel("测试 AUC (%)")
    ax.set_title("图4  网络层数对链路预测性能的影响")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = os.path.join(RESULT_DIR, "fig4_layers_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")


def print_table(records):
    print("\n" + "=" * 104)
    print("实验汇总表（链路预测）")
    print("=" * 104)
    print(f"{'数据集':<10}{'模型':<11}{'方式':<8}{'解码器':<7}{'lr':>7}{'层数':>5}"
          f"{'验证AUC':>9}{'测试AUC':>9}{'测试H@50':>10}{'每轮耗时':>10}{'总耗时':>9}")
    print("-" * 104)
    for ds in DATASETS:
        for model in MODELS:
            for mode in ["full", "sample"]:
                r = find(records, dataset=ds, model=model, mode=mode)
                if not r:
                    continue
                print(f"{r['dataset']:<10}{r['model']:<11}{r['mode']:<8}"
                      f"{r['decoder']:<7}{r['lr']:>7}{r['layers']:>5}"
                      f"{r['best_val_auc']*100:>8.2f}%{r['best_test_auc']*100:>8.2f}%"
                      f"{r['best_test_hits50']:>10.4f}"
                      f"{r['train_time_per_epoch']:>9.3f}s"
                      f"{r['train_time_total']:>8.1f}s")
    print("=" * 104)


def main():
    records = load_results("results.jsonl")
    if not records:
        print("没有找到实验结果。请先运行：python run_all.py --stage main")
        return 1
    records = dedup(records)
    print(f"共读取到 {len(records)} 条实验记录（已去重）\n")
    print("正在生成图表……")
    plot_model_comparison(records)
    plot_full_vs_sample(records)
    plot_lr_effect(records)
    plot_layers_effect(records)
    print_table(records)
    print(f"\n所有图片已保存到：{RESULT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
