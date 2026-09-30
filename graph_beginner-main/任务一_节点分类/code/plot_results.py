# -*- coding: utf-8 -*-
"""
plot_results.py —— 把实验结果画成图（任务一：节点分类）

【这个脚本做什么？】
    读取 results/results.jsonl（run_all.py 跑出来的结果），
    自动生成四张图，直接放进作业报告里：

        图1  model_comparison.png  不同模型在各数据集上的测试准确率（柱状图）
        图2  full_vs_sample.png    全图训练 vs 采样训练：准确率 & 每轮耗时（双柱图）
        图3  lr_effect.png         学习率对准确率的影响（折线图）
        图4  layers_effect.png     网络层数对准确率的影响（折线图）

【怎么运行？】
    python plot_results.py
    # 图片会保存到 ../results/ 目录

【注意】
    - 图3、图4 需要有 lr / layers 消融实验的数据（先跑 run_all.py --stage lr / layers）
      没有数据时会自动跳过并给出提示。
    - 如果某张图上的柱子高了低了，先看看是不是只跑了一部分实验，或者重复跑过
      （重复的记录只保留最后一次）。
"""

import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")           # 不弹窗，直接存文件（服务器/无界面环境必须这么设）
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from utils import RESULT_DIR, load_results   # noqa: E402

# ---------------------------------------------------------------------------
# 全局画图风格
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False      # 让负号正常显示
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

# 四个模型固定用四种颜色，保证所有图里颜色一致，看着专业
MODEL_COLORS = {
    "GCN": "#4C72B0",
    "GAT": "#DD8452",
    "GraphSAGE": "#55A868",
    "GIN": "#C44E52",
}
DATASETS = ["Cora", "Citeseer", "Flickr"]
MODELS = ["GCN", "GAT", "GraphSAGE", "GIN"]


def dedup(records):
    """
    去重：同一组 (dataset, model, mode, lr, layers) 如果跑了多次，只保留最后一次。
    这样重复运行 run_all.py 也不会把图搞乱。
    """
    best = OrderedDict()
    for r in records:
        key = (r["dataset"], r["model"], r["mode"], r["lr"], r["layers"])
        best[key] = r          # 后出现的覆盖前面的
    return list(best.values())


def find(records, **kwargs):
    """按条件找一个结果记录，找不到返回 None。"""
    for r in records:
        if all(r.get(k) == v for k, v in kwargs.items()):
            return r
    return None


# ---------------------------------------------------------------------------
# 图1：不同模型在各数据集上的准确率
# ---------------------------------------------------------------------------
def plot_model_comparison(records):
    """柱状图：x 轴是数据集，每个数据集下面 4 根柱子对应 4 个模型。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mode, title in zip(
            axes, ["full", "sample"],
            ["全图训练 (full-batch)", "子图采样训练 (mini-batch)"]):
        x = np.arange(len(DATASETS))
        width = 0.2

        for i, model in enumerate(MODELS):
            vals, errs = [], []
            for ds in DATASETS:
                r = find(records, dataset=ds, model=model, mode=mode)
                # 主实验默认 lr=0.01, layers=2，这里就按这个筛选
                if r is None or r["lr"] != 0.01 or r["layers"] != 2:
                    r = find(records, dataset=ds, model=model, mode=mode)
                vals.append(r["best_test_acc"] * 100 if r else 0)
                errs.append(0)

            bars = ax.bar(x + (i - 1.5) * width, vals, width,
                          label=model, color=MODEL_COLORS[model])
            # 把数值写在柱子顶上，方便读
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel("测试准确率 (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("图1  四种 GNN 模型在三个数据集上的节点分类准确率", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig1_model_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")
    return out


# ---------------------------------------------------------------------------
# 图2：全图训练 vs 采样训练
# ---------------------------------------------------------------------------
def plot_full_vs_sample(records):
    """双柱图：左图比准确率，右图比每轮训练耗时。"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- 左：准确率 ---
    ax = axes[0]
    labels, acc_full, acc_samp = [], [], []
    for ds in DATASETS:
        for model in MODELS:
            rf = find(records, dataset=ds, model=model, mode="full")
            rs = find(records, dataset=ds, model=model, mode="sample")
            if rf and rs:
                labels.append(f"{ds}\n{model}")
                acc_full.append(rf["best_test_acc"] * 100)
                acc_samp.append(rs["best_test_acc"] * 100)

    if labels:
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, acc_full, w, label="全图训练", color="#4C72B0")
        ax.bar(x + w / 2, acc_samp, w, label="采样训练", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("测试准确率 (%)")
        ax.set_title("准确率对比")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    # --- 中：每轮训练耗时（对数坐标，因为不同数据集差好几个数量级）---
    ax = axes[1]
    t_full, t_samp = [], []
    for ds in DATASETS:
        for model in MODELS:
            rf = find(records, dataset=ds, model=model, mode="full")
            rs = find(records, dataset=ds, model=model, mode="sample")
            if rf and rs:
                t_full.append(max(rf["train_time_per_epoch"], 1e-4))
                t_samp.append(max(rs["train_time_per_epoch"], 1e-4))

    if t_full:
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, t_full, w, label="全图训练", color="#4C72B0")
        ax.bar(x + w / 2, t_samp, w, label="采样训练", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("每轮训练耗时 (秒，对数刻度)")
        ax.set_yscale("log")
        ax.set_title("每轮训练耗时对比\n(采样训练一轮里要做很多个 batch)")
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=9)

    # --- 右：训练到收敛的总时间 ---
    ax = axes[2]
    tot_full, tot_samp = [], []
    for ds in DATASETS:
        for model in MODELS:
            rf = find(records, dataset=ds, model=model, mode="full")
            rs = find(records, dataset=ds, model=model, mode="sample")
            if rf and rs:
                tot_full.append(max(rf["train_time_total"], 1e-4))
                tot_samp.append(max(rs["train_time_total"], 1e-4))

    if tot_full:
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, tot_full, w, label="全图训练", color="#4C72B0")
        ax.bar(x + w / 2, tot_samp, w, label="采样训练", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("完整训练总耗时 (秒，对数刻度)")
        ax.set_yscale("log")
        ax.set_title("总训练耗时对比")
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("图2  全图训练 vs 子图采样训练：性能与运行时间", fontsize=13)
    out = os.path.join(RESULT_DIR, "fig2_full_vs_sample.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")
    return out


# ---------------------------------------------------------------------------
# 图3：学习率的影响
# ---------------------------------------------------------------------------
def plot_lr_effect(records):
    """折线图：横轴学习率（对数刻度），纵轴准确率。"""
    lrs = [0.001, 0.005, 0.01, 0.05]
    datasets_with_data = [
        ds for ds in ["Cora", "Flickr"]
        if any(r["lr"] != 0.01 or True for r in records if r["dataset"] == ds)
        and any(r["dataset"] == ds and r["mode"] == "full" and r["layers"] == 2
                for r in records)
    ]
    if not datasets_with_data:
        print("  [跳过] 没有学习率消融的数据，请先运行：python run_all.py --stage lr")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Cora": "#4C72B0", "Flickr": "#DD8452"}

    for ds in datasets_with_data:
        xs, ys = [], []
        for lr in lrs:
            r = find(records, dataset=ds, model="GCN", mode="full", lr=lr, layers=2)
            if r:
                xs.append(lr)
                ys.append(r["best_test_acc"] * 100)
        if xs:
            ax.plot(xs, ys, "o-", label=f"{ds} (GCN, 2层)",
                    color=colors.get(ds, None), linewidth=2, markersize=7)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy + 1.2, f"{yy:.1f}", ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("学习率 (log scale)")
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("图3  学习率对节点分类性能的影响")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    out = os.path.join(RESULT_DIR, "fig3_lr_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")
    return out


# ---------------------------------------------------------------------------
# 图4：网络层数的影响
# ---------------------------------------------------------------------------
def plot_layers_effect(records):
    """折线图：横轴层数，纵轴准确率。"""
    layer_list = [2, 3, 4]
    datasets_with_data = [
        ds for ds in ["Cora", "Flickr"]
        if any(r["dataset"] == ds and r["mode"] == "full"
               and r["lr"] == 0.01 and r["layers"] in layer_list
               for r in records)
    ]
    if not datasets_with_data:
        print("  [跳过] 没有层数消融的数据，请先运行：python run_all.py --stage layers")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Cora": "#4C72B0", "Flickr": "#DD8452"}

    for ds in datasets_with_data:
        xs, ys = [], []
        for L in layer_list:
            r = find(records, dataset=ds, model="GCN", mode="full", lr=0.01, layers=L)
            if r:
                xs.append(L)
                ys.append(r["best_test_acc"] * 100)
        if xs:
            ax.plot(xs, ys, "s-", label=f"{ds} (GCN)", color=colors.get(ds, None),
                    linewidth=2, markersize=8)
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy + 1.2, f"{yy:.1f}", ha="center", fontsize=8)

    ax.set_xticks(layer_list)
    ax.set_xlabel("GNN 层数")
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("图4  网络层数对节点分类性能的影响（观察过平滑现象）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    out = os.path.join(RESULT_DIR, "fig4_layers_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存 {out}")
    return out


# ---------------------------------------------------------------------------
# 文字汇总表
# ---------------------------------------------------------------------------
def print_table(records):
    """在终端打印一张文字版汇总表，方便直接复制进报告。"""
    print("\n" + "=" * 96)
    print("实验汇总表（测试准确率 / 每轮训练耗时 / 参数量）")
    print("=" * 96)
    header = (f"{'数据集':<10}{'模型':<11}{'方式':<8}{'lr':>7}{'层数':>5}"
              f"{'验证acc':>10}{'测试acc':>10}{'每轮耗时':>11}{'总耗时':>10}{'参数量':>11}")
    print(header)
    print("-" * 96)

    for ds in DATASETS:
        for model in MODELS:
            for mode in ["full", "sample"]:
                r = find(records, dataset=ds, model=model, mode=mode)
                if r is None:
                    continue
                print(f"{r['dataset']:<10}{r['model']:<11}{r['mode']:<8}"
                      f"{r['lr']:>7}{r['layers']:>5}"
                      f"{r['best_val_acc']*100:>9.2f}%{r['best_test_acc']*100:>9.2f}%"
                      f"{r['train_time_per_epoch']:>10.3f}s"
                      f"{r['train_time_total']:>9.1f}s"
                      f"{r['num_params']:>11,}")
    print("=" * 96)


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
