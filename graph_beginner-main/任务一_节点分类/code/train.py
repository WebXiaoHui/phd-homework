# -*- coding: utf-8 -*-
"""
train.py —— 节点分类训练主脚本（任务一）

本脚本同时支持两种训练方式，用 --mode 切换：

    --mode full    全图训练（full-batch）
                   每个 epoch 把整张图一次性喂给模型。
                   优点：实现简单；缺点：图太大时显存放不下。

    --mode sample  子图采样训练（mini-batch）
                   用 PyG 自带的 NeighborLoader，每个 batch 只取
                   "一小批种子节点 + 它们的多跳邻居"，组成子图训练。
                   优点：显存可控，能扩展到超大图；缺点：实现更复杂。

【怎么运行？（在本文件所在目录下，或用绝对路径）】

    # 全图训练：Cora + GCN
    python train.py --dataset Cora --model GCN --mode full

    # 采样训练：Flickr + GAT
    python train.py --dataset Flickr --model GAT --mode sample

    # 改学习率和层数
    python train.py --dataset Cora --model GCN --mode full --lr 0.005 --layers 3

    # 一键跑完所有对比实验（推荐）
    python run_all.py

【输出】
    - 终端会打印每个 epoch 的 loss / 准确率
    - logs/ 目录下会保存完整日志
    - results/ 目录下会追加一行 JSON 结果（供 plot_results.py 画图）
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset, dataset_summary
from models import build_model, MODEL_NAMES
from utils import (LOG_DIR, Timer, add_common_args, count_parameters,
                   format_seconds, resolve_device, save_result, set_seed,
                   setup_logger)


# ---------------------------------------------------------------------------
# 模型训练一个 epoch（全图）
# ---------------------------------------------------------------------------
def run_epoch_full(model, data, optimizer):
    """
    全图训练：整张图一次前向 + 反向。

    返回 (loss, 训练耗时秒数)
    """
    model.train()
    optimizer.zero_grad()

    t0 = time.time()
    out = model(data.x, data.edge_index)                    # 所有节点都算一遍
    loss = F.cross_entropy(out[data.train_mask],            # 只在训练节点上算 loss
                           data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    dt = time.time() - t0

    return float(loss), dt


# ---------------------------------------------------------------------------
# 模型训练一个 epoch（子图采样）
# ---------------------------------------------------------------------------
def run_epoch_sampler(model, loader, optimizer, device):
    """
    采样训练：遍历 DataLoader，每个 batch 是一张子图。

    关键点说明（新手最容易困惑的地方）：
        NeighborLoader 返回的子图里，节点顺序是：
            [ 种子节点 (batch_size 个) , 采样到的邻居节点 ... ]
        所以我们只取前 batch_size 个节点的输出算 loss，这就是
        `out[:batch.batch_size]` 和 `batch.y[:batch.batch_size]` 的含义。

    返回 (平均loss, 训练耗时秒数, batch数量)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)
        # 只对"种子节点"计算损失
        out = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]

        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        n_batches += 1
    dt = time.time() - t0

    return total_loss / max(n_batches, 1), dt, n_batches


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, device, measure_time=False):
    """
    评估模型。注意：**两种训练方式都用全图推理来评估**。

    为什么？
        采样训练只是"为了省显存"的一种训练技巧，评估时图能放进显存就直接全图算，
        这样得到的指标才是真正意义上的节点分类准确率。

    返回 (train_acc, val_acc, test_acc, 推理耗时)
    """
    model.eval()
    t0 = time.time()
    out = model(data.x, data.edge_index)
    dt = time.time() - t0

    pred = out.argmax(dim=1)   # 每个节点取分数最高的类别

    def acc(mask):
        return float((pred[mask] == data.y[mask]).float().mean())

    return acc(data.train_mask), acc(data.val_mask), acc(data.test_mask), dt


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="节点分类训练脚本（任务一）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="Cora", help="Cora / Citeseer / Flickr")
    parser.add_argument("--model", default="GCN", help=" / ".join(MODEL_NAMES))
    parser.add_argument("--mode", default="full", choices=["full", "sample"],
                        help="full=全图训练, sample=子图采样训练")
    # 采样相关参数
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="[仅 sample 模式] 每个 batch 的种子节点数")
    parser.add_argument("--num_neighbors", type=int, default=10,
                        help="[仅 sample 模式] 每层采样多少个邻居")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每多少轮评估一次（大图可以调大，节省时间）")
    add_common_args(parser)
    args = parser.parse_args()

    # ---------------- 0. 准备工作 ----------------
    set_seed(args.seed)
    device = resolve_device(args.device)

    # 日志文件名：自动带上关键参数，方便区分不同实验
    log_name = args.log or f"{args.dataset}_{args.model}_{args.mode}_lr{args.lr}_L{args.layers}.log"
    logger = setup_logger(
        name=f"task1_{args.dataset}_{args.model}_{args.mode}_{args.lr}_{args.layers}_{args.tag}",
        log_file=os.path.join(LOG_DIR, log_name),
        quiet=args.quiet,
    )

    logger.info("=" * 78)
    logger.info(f"任务一 节点分类 | 数据集={args.dataset} 模型={args.model} "
                f"训练方式={args.mode} 学习率={args.lr} 层数={args.layers}")
    logger.info("=" * 78)
    logger.info(f"超参数：hidden={args.hidden}, dropout={args.dropout}, "
                f"weight_decay={args.weight_decay}, epochs={args.epochs}, seed={args.seed}")

    # ---------------- 1. 加载数据 ----------------
    try:
        data, in_dim, num_classes = load_dataset(args.dataset)
    except Exception as e:
        logger.info(f"[错误] 数据加载失败：{e}")
        return 1

    data = data.to(device)
    logger.info(f"数据集信息：{dataset_summary(data)}")

    # ---------------- 2. 建模型 ----------------
    model = build_model(args.model, in_dim, args.hidden, num_classes,
                        num_layers=args.layers, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    logger.info(f"模型参数量：{n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ---------------- 3. 准备训练数据 ----------------
    loader = None
    steps_per_epoch = 1     # 每个 epoch 有多少次参数更新
    if args.mode == "sample":
        # 延迟 import，全图模式就不需要这个模块
        from torch_geometric.loader import NeighborLoader

        # 每一层采样多少个邻居，层数要和模型层数一致
        num_neighbors = [args.num_neighbors] * args.layers
        # 种子节点 = 训练节点。用 nonzero 取出下标（比传 bool mask 更稳妥）
        train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)

        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            input_nodes=train_nodes,
            shuffle=True,
            num_workers=0,      # Windows 下用多进程 DataLoader 容易出问题，这里设为 0
        )
        steps_per_epoch = len(loader)
        logger.info(f"采样设置：batch_size={args.batch_size}, "
                    f"num_neighbors={num_neighbors}, "
                    f"每轮 batch 数={steps_per_epoch}")
    else:
        logger.info("全图训练：每个 epoch 做 1 次全图前向 + 反向")

    # ---------------- 4. 训练循环 ----------------
    logger.info("-" * 78)
    logger.info(f"{'Epoch':>6} | {'Loss':>8} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'时间/轮':>9}")
    logger.info("-" * 78)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    history = []            # 记录每轮指标，最后一起保存
    train_time_total = 0.0  # 只累计"训练"的时间，不含评估
    train_time_list = []    # 每轮训练耗时
    infer_time_full = 0.0   # 全图推理一次要多久（衡量推理开销）

    t_start = Timer.now()

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        if args.mode == "sample":
            loss, dt, n_batches = run_epoch_sampler(model, loader, optimizer, device)
        else:
            loss, dt = run_epoch_full(model, data, optimizer)
            n_batches = 1

        train_time_total += dt
        # 第 1 轮通常包含 CUDA 初始化等一次性开销，统计平均耗时时把它排除更准确
        if epoch > 1:
            train_time_list.append(dt)

        # ---- 评估 ----
        # 每 eval_every 轮评估一次；最后一轮一定要评估，保证 best 值是最新的
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            train_acc, val_acc, test_acc, infer_dt = evaluate(
                model, data, device, measure_time=True)
            infer_time_full = infer_dt

            # 以"验证集准确率"为标准挑选最佳模型（这是标准做法：
            # 绝对不能看测试集来挑模型，否则测试结果就不客观了）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch

            history.append({"epoch": epoch, "loss": loss,
                            "train_acc": train_acc, "val_acc": val_acc,
                            "test_acc": test_acc, "train_time": dt})
            logger.info(f"{epoch:>6d} | {loss:>8.4f} | {train_acc:>8.4f} | "
                        f"{val_acc:>8.4f} | {test_acc:>8.4f} | {dt:>8.3f}s")

    wall_time = Timer.now() - t_start

    # ---------------- 5. 汇总 ----------------
    mean_epoch_time = (sum(train_time_list) / len(train_time_list)
                       if train_time_list else train_time_total / args.epochs)

    logger.info("-" * 78)
    logger.info("【结果汇总】")
    logger.info(f"  最佳验证准确率 : {best_val_acc:.4f}  (第 {best_epoch} 轮)")
    logger.info(f"  对应的测试准确率: {best_test_acc:.4f}")
    logger.info(f"  训练总耗时     : {format_seconds(train_time_total)} "
                f"(不含评估) / 整体墙钟时间 {format_seconds(wall_time)}")
    logger.info(f"  平均每轮训练耗时: {format_seconds(mean_epoch_time)} "
                f"(每轮 {steps_per_epoch} 次参数更新)")
    logger.info(f"  全图推理耗时   : {format_seconds(infer_time_full)}")
    logger.info(f"  模型参数量     : {n_params:,}")

    # ---------------- 6. 保存结果 ----------------
    record = {
        "task": "node_classification",
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "lr": args.lr,
        "layers": args.layers,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size if args.mode == "sample" else None,
        "num_neighbors": args.num_neighbors if args.mode == "sample" else None,
        "best_val_acc": round(best_val_acc, 4),
        "best_test_acc": round(best_test_acc, 4),
        "best_epoch": best_epoch,
        "train_time_total": round(train_time_total, 3),
        "train_time_per_epoch": round(mean_epoch_time, 4),
        "wall_time": round(wall_time, 3),
        "infer_time_full": round(infer_time_full, 4),
        "steps_per_epoch": steps_per_epoch,
        "num_params": n_params,
        "tag": args.tag,
    }
    path = save_result(record, args.result_file)
    logger.info(f"结果已追加保存到：{path}")

    # 把每轮曲线也存一份，plot_results.py 会用到
    hist_path = os.path.join(os.path.dirname(path), "history",
                             f"{args.dataset}_{args.model}_{args.mode}_lr{args.lr}_L{args.layers}.json")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    import json
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
