# -*- coding: utf-8 -*-
"""
train.py —— 图分类训练脚本（任务三：图分类）

【这个脚本能做什么？】

    训练一个 GNN 模型做图分类（或图回归），并报告测试集结果。
    支持 4 个模型 × 4 种池化 × 各种超参数，全部通过命令行参数控制。

【两种训练方式（对应报告要求 3.2）】

    --mode sample （默认，推荐）
        分批次训练：每批装 batch_size 张图，一个 epoch 有多个 batch。
        这是图分类的标准做法，也是显存能承受的做法。

    --mode full
        全图训练：把所有训练图**一次全部**塞进一个 batch。
        小数据集（MUTAG 只有 188 张图）可以这么干，好处是：
          - 一个 epoch 只有 1 次参数更新，梯度是"精确"的；
          - 没有 batch 之间的抖动，BatchNorm 统计量更稳定。
        坏处是：
          - 显存占用大，图一多就爆显存（ZINC 就别想全图训练了）；
          - 一个 epoch 只更新一次参数，收敛需要的 epoch 数暴增。
        本脚本会把两种方式的**精度**和**耗时**都记下来，方便对比。

【分类 vs 回归，代码里怎么区分？】

    分类（MUTAG/PROTEINS/ENZYMES/IMDB-BINARY）：
        损失 CrossEntropyLoss，指标 = 准确率（越大越好）
    回归（ZINC）：
        损失 L1Loss(=MAE)，指标 = MAE（**越小越好**）

    注意"越小越好"这件事：选最优模型时，回归任务要取最小值，
    所以下面用 `is_better()` 这个函数统一处理，而不是到处写 >/<。

【怎么运行？看几个例子】

    # 默认：MUTAG 上的 GCN，平均池化
    python train.py --dataset MUTAG --model GCN

    # GIN + 求和池化 + 4 层
    python train.py --dataset MUTAG --model GIN --pooling sum --layers 4

    # ZINC 回归任务
    python train.py --dataset ZINC --model GIN --pooling sum --epochs 50

    # 全图训练（一次把所有图塞进去）
    python train.py --dataset MUTAG --model GCN --mode full

    # 只看命令会怎么跑，不真的训练
    python train.py --dataset MUTAG --model GCN --dry_run
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from datasets import ALL_DATASETS, TU_DATASETS, ZINC_NAME, load_dataset   # noqa: E402
from models import MODEL_NAMES, POOL_LABELS, POOL_NAMES, build_model      # noqa: E402
from utils import (LOG_DIR, RESULT_DIR, add_common_args, count_parameters,   # noqa: E402
                   format_seconds, resolve_device, save_result, set_seed,
                   setup_logger, Timer)


# ---------------------------------------------------------------------------
# 指标：统一"越大越好"还是"越小越好"这个差异
# ---------------------------------------------------------------------------
# 分类用准确率（越大越好），回归用 MAE（越小越好）
HIGHER_IS_BETTER = {"classification": True, "regression": False}

# 指标名字，用来写日志和保存结果
METRIC_NAME = {"classification": "准确率", "regression": "MAE"}


def is_better(new, old, task_type):
    """判断新结果是不是比旧结果更好（回归任务要反过来比）。"""
    if old is None:
        return True
    if HIGHER_IS_BETTER[task_type]:
        return new > old
    return new < old


# ---------------------------------------------------------------------------
# 训练一个 epoch
# ---------------------------------------------------------------------------
def run_one_epoch(model, loader, optimizer, criterion, device, task_type):
    """
    在训练集上跑一个 epoch。

    返回 (平均损失, 耗时秒数, batch 个数)

    注意计时：这里**不包含**数据搬运到 GPU 之外的东西，
    但包含了每个 batch 的 forward + backward + optimizer.step()，
    这正是我们想比较的"训练开销"。
    """
    model.train()
    total_loss = 0.0
    n_graphs = 0
    n_batches = 0

    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)

        if task_type == "classification":
            # out: (num_graphs, num_classes), batch.y: (num_graphs,)
            loss = criterion(out, batch.y)
        else:
            # 回归：out 是 (num_graphs, 1)，标签也要变成 (num_graphs, 1)
            loss = criterion(out, batch.y.view(-1, 1).float())

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs
        n_graphs += batch.num_graphs
        n_batches += 1

    dt = time.time() - t0
    return total_loss / max(n_graphs, 1), dt, n_batches


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, task_type):
    """
    在验证集或测试集上评估。返回 (平均损失, 指标值, 耗时秒数)。

    为什么要用 @torch.no_grad()？
        评估不需要算梯度，关掉梯度追踪能省很多显存和时间。
        这是评估的常见做法（也是任务一、二里都用的）。
    """
    model.eval()
    total_loss = 0.0
    n_graphs = 0
    n_correct = 0
    abs_err_sum = 0.0          # 回归用：绝对误差之和，最后除以样本数就是 MAE

    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)

        if task_type == "classification":
            loss = criterion(out, batch.y)
            pred = out.argmax(dim=-1)
            n_correct += int((pred == batch.y).sum().item())
        else:
            target = batch.y.view(-1, 1).float()
            loss = criterion(out, target)
            abs_err_sum += float((out - target).abs().sum().item())

        total_loss += float(loss.item()) * batch.num_graphs
        n_graphs += batch.num_graphs

    dt = time.time() - t0
    avg_loss = total_loss / max(n_graphs, 1)
    if task_type == "classification":
        metric = n_correct / max(n_graphs, 1)
    else:
        metric = abs_err_sum / max(n_graphs, 1)
    return avg_loss, metric, dt


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="图分类/图回归训练脚本（任务三）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)

    g = p.add_argument_group("任务三专用参数")
    g.add_argument("--dataset", type=str, default="MUTAG", choices=ALL_DATASETS,
                   help="用哪个数据集")
    g.add_argument("--model", type=str, default="GCN", choices=MODEL_NAMES,
                   help="用哪个 GNN 模型")
    g.add_argument("--pooling", type=str, default="avg", choices=POOL_NAMES,
                   help="池化方式：avg=平均 max=最大 min=最小 sum=求和")
    g.add_argument("--mode", type=str, default="sample", choices=["full", "sample"],
                   help="full=全图训练(所有图一个批次)  sample=分批次训练")
    g.add_argument("--max_graphs", type=int, default=5000,
                   help="ZINC 最多用多少张训练图（ZINC 太大，取子集；其他数据集忽略此项）")
    g.add_argument("--patience", type=int, default=50,
                   help="验证集多少轮没提升就提前停止（0 表示不提前停止）")
    g.add_argument("--no_early_stop", action="store_true", help="关闭提前停止")
    g.add_argument("--pool_bn_eps", type=float, default=1e-5,
                   help="池化层 BatchNorm 的 eps（默认 1e-5 = PyTorch 标准值）。"
                        "用 MinPooling 时验证损失会指数级爆炸（因为图向量方差接近 0，"
                        "BatchNorm 等于除以一个接近 0 的数），"
                        "把这里调大（比如 0.1）就能稳住。详见 models.py 里的注释")
    g.add_argument("--pool_norm", type=str, default="ln",
                   choices=["bn", "ln", "none"],
                   help="池化后用哪种归一化。ln=LayerNorm（默认，推荐：按每张图"
                        "独立归一化，不存在训练/推理统计量对不上的问题）；"
                        "bn=BatchNorm（经典做法，但在池化后会因为统计量漂移"
                        "而崩掉，MaxPooling 用 bn 时验证准确率会卡在多数类不动）；"
                        "none=不归一化。详见 models.py 里 GraphClassifier 的注释")
    g.add_argument("--dry_run", action="store_true", help="只打印配置，不训练")

    args = p.parse_args()

    # ---------------- 基础设置 ----------------
    set_seed(args.seed)
    device = resolve_device(args.device)
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    # 实验名里带上池化后的归一化方式（pn=pool_norm）。
    # 为什么必须带？因为 bn 和 ln 会让同样的池化方式跑出完全不同的结果，
    # 不带的话两次实验的日志/历史文件会重名、互相覆盖。
    exp_name = (f"{args.dataset}_{args.model}_pool-{args.pooling}_{args.mode}"
                f"_pn-{args.pool_norm}_L{args.layers}_lr{args.lr}"
                + (f"_{args.tag}" if args.tag else ""))

    log_file = os.path.join(LOG_DIR, args.log or f"train_{exp_name}.log")
    logger = setup_logger(f"train3_{exp_name}", log_file, quiet=args.quiet)

    def log(msg=""):
        logger.info(msg)

    log("=" * 78)
    log(f"任务三 图分类 —— {exp_name}")
    log(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 78)
    log(f"数据集={args.dataset}  模型={args.model}  "
        f"池化={POOL_LABELS[args.pooling]}（{args.pooling}）")
    log(f"层数={args.layers}  隐藏维度={args.hidden}  dropout={args.dropout}")
    log(f"学习率={args.lr}  权重衰减={args.weight_decay}  轮数={args.epochs}")
    log(f"训练方式={'全图训练' if args.mode == 'full' else '分批次训练'}"
        f"  批次大小设置={args.batch_size}  随机种子={args.seed}")

    if args.dry_run:
        log("\n[--dry_run] 配置检查完毕，没有真的开始训练。")
        return 0

    # ---------------- 读数据 ----------------
    log("\n" + "-" * 78)
    log("第 1 步：加载数据")
    log("-" * 78)
    t_load = Timer()
    with t_load:
        train_list, val_list, test_list, in_dim, num_classes, task_type = \
            load_dataset(args.dataset, max_graphs=args.max_graphs,
                         seed=args.seed, verbose=not args.quiet)
    log(f"  数据加载用时 {format_seconds(t_load.elapsed)}")
    log(f"  节点特征维度={in_dim}  输出维度={num_classes}  任务类型={task_type}")

    from torch_geometric.loader import DataLoader

    # ---------------- 决定批次大小 ----------------
    # full 模式：一个 batch 装下所有训练图
    # sample 模式：用命令行给的 batch_size
    if args.mode == "full":
        train_bs = len(train_list)
        # 全图训练的"每个 epoch 只有 1 次更新"，所以轮数要放大一些，
        # 否则和分批训练比就不公平了（分批一个 epoch 更新几十次）。
        eff_epochs = args.epochs * 4
    else:
        train_bs = args.batch_size
        eff_epochs = args.epochs

    # 【小心一个坑】如果最后一个 batch 恰好只剩 1 张图，
    # 池化后的 BatchNorm 会因为"只有 1 个样本、方差为 0"而算出 NaN。
    # 解决办法：训练集长度正好是 batch_size 的倍数 +1 时，丢掉最后那个 batch。
    # （模型里也做了保护，这里是双保险）
    drop_last = (len(train_list) > train_bs) and (len(train_list) % train_bs == 1)

    train_loader = DataLoader(train_list, batch_size=train_bs, shuffle=True,
                              drop_last=drop_last)
    # 验证/测试不需要打乱，也不需要 drop_last
    val_loader = DataLoader(val_list, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

    log(f"  训练集 {len(train_list)} 张图 / 验证集 {len(val_list)} / 测试集 {len(test_list)}")
    log(f"  训练批次大小={train_bs}"
        f"{'（=全部训练图，全图训练）' if args.mode == 'full' else ''}"
        f"  每个 epoch 有 {len(train_loader)} 个 batch")
    if args.mode == "full":
        log(f"  【注意】全图训练一个 epoch 只更新 1 次参数，"
            f"所以把轮数放大到 {eff_epochs} 轮，和分批训练公平对比")

    # ---------------- 建模型 ----------------
    log("\n" + "-" * 78)
    log("第 2 步：搭建模型")
    log("-" * 78)
    set_seed(args.seed)          # 再固定一次种子，保证不同池化方式下模型初始化一致
    model = build_model(args.model, in_dim, args.hidden, num_classes,
                        num_layers=args.layers, dropout=args.dropout,
                        pooling=args.pooling,
                        pool_bn_eps=args.pool_bn_eps,
                        pool_norm=args.pool_norm).to(device)
    n_param = count_parameters(model)
    log(f"  {args.model} + {POOL_LABELS[args.pooling]}")
    log(f"  可训练参数量 = {n_param:,}")
    log(f"  模型结构：\n{model}")

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ---------------- 训练 ----------------
    log("\n" + "-" * 78)
    log("第 3 步：开始训练")
    log("-" * 78)
    log(f"{'轮次':>6}{'训练损失':>12}{'训练指标':>12}{'验证损失':>12}"
        f"{'验证' + METRIC_NAME[task_type]:>12}{'耗时':>10}")

    best_val = None
    best_test = None
    best_epoch = -1
    best_state = None

    train_time_total = 0.0            # 纯训练时间（不含评估）
    train_time_list = []              # 逐轮训练时间（用来算每轮平均）
    epoch_rows = []
    n_no_improve = 0
    t_all = time.time()

    for epoch in range(1, eff_epochs + 1):
        tr_loss, tr_dt, n_batches = run_one_epoch(
            model, train_loader, optimizer, criterion, device, task_type)
        train_time_total += tr_dt

        # 第 1 轮往往因为 CUDA 初始化、缓存分配而特别慢，
        # 算平均耗时的时候先排除掉，不然结果会失真。
        # （这一点和任务一、任务二的处理保持一致）
        if epoch > 1:
            train_time_list.append(tr_dt)

        val_loss, val_metric, _ = evaluate(
            model, val_loader, criterion, device, task_type)

        # 训练指标：为了不额外花一次前向的时间，这里直接报告训练损失，
        # 所以上面表头写的是"训练损失 / 训练指标"两列，训练指标只在
        # 需要的时候（比如最后一行）算一次就行，这里用 loss 代替显示。
        log(f"{epoch:>6}{tr_loss:>12.4f}{'-':>12}{val_loss:>12.4f}"
            f"{val_metric:>12.4f}{tr_dt:>9.2f}s")

        epoch_rows.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_metric": round(val_metric, 5),
            "train_seconds": round(tr_dt, 4),
            "n_batches": n_batches,
        })

        # ---- 保存验证集上最好的模型 ----
        if is_better(val_metric, best_val, task_type):
            best_val = val_metric
            best_epoch = epoch
            # 把权重复制一份存起来（不能直接存引用，因为训练会继续改它）
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            n_no_improve = 0
        else:
            n_no_improve += 1

        # ---- 提前停止 ----
        if (not args.no_early_stop) and args.patience > 0 \
                and n_no_improve >= args.patience:
            log(f"\n  验证集连续 {args.patience} 轮没有提升，提前停止训练。")
            break

    total_wall = time.time() - t_all

    # ---------------- 用最好的模型测测试集 ----------------
    log("\n" + "-" * 78)
    log("第 4 步：用验证集上最好的模型测测试集")
    log("-" * 78)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metric, test_dt = evaluate(
        model, test_loader, criterion, device, task_type)
    train_loss_final, train_metric_final, _ = evaluate(
        model, train_loader, criterion, device, task_type)

    metric_unit = "准确率" if task_type == "classification" else "MAE"
    fmt = "{:.4f}" if task_type == "classification" else "{:.4f}"
    log(f"  最好的一轮（第 {best_epoch} 轮）：验证集{metric_unit} = {best_val:.4f}")
    log(f"  训练集{metric_unit} = {fmt.format(train_metric_final)}"
        f"   （和验证集差很多就说明过拟合了）")
    log(f"  测试集损失 = {test_loss:.4f}")
    log(f"  测试集{metric_unit} = {fmt.format(test_metric)}   <<< 这是最终成绩")
    log(f"  测试耗时 {format_seconds(test_dt)}")

    # ---------------- 统计时间 ----------------
    per_epoch = (sum(train_time_list) / len(train_time_list)
                 if train_time_list else train_time_total / max(1, len(epoch_rows)))
    log("\n" + "-" * 78)
    log("时间统计")
    log("-" * 78)
    log(f"  纯训练总耗时      = {format_seconds(train_time_total)}"
        f"（不含评估，共 {len(epoch_rows)} 轮）")
    log(f"  平均每轮训练耗时  = {format_seconds(per_epoch)}")
    log(f"  每个 epoch 的 batch 数 = {len(train_loader)}")
    log(f"  整个脚本总耗时    = {format_seconds(total_wall)}")

    # ---------------- 保存结果 ----------------
    record = {
        "task": "graph_classification",
        "dataset": args.dataset,
        "model": args.model,
        "pooling": args.pooling,
        # 池化后用的归一化方式。一定要记下来！
        # 因为 BatchNorm 和 LayerNorm 会让**同样的池化方式**跑出完全不同的结果
        # （BatchNorm 时 MaxPooling 的验证准确率会卡在多数类不动），
        # 不记录的话，混在一起分析就会得出错误结论。
        "pool_norm": args.pool_norm,
        "mode": args.mode,
        "task_type": task_type,
        "layers": args.layers,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": train_bs,
        "n_batches_per_epoch": len(train_loader),
        "epochs_run": len(epoch_rows),
        "best_epoch": best_epoch,
        "n_params": n_param,
        "best_val_metric": best_val,
        "best_test_metric": test_metric,
        "train_metric_final": train_metric_final,
        "test_loss": test_loss,
        "train_time_total": train_time_total,
        "train_time_per_epoch": per_epoch,
        "wall_time_total": total_wall,
        "n_train": len(train_list),
        "n_val": len(val_list),
        "n_test": len(test_list),
        "seed": args.seed,
        "tag": args.tag,
    }
    path = save_result(record, args.result_file)

    # 逐轮历史单独存一个文件，方便以后画训练曲线
    hist_dir = os.path.join(RESULT_DIR, "history")
    os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, f"{exp_name}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"config": record, "epochs": epoch_rows},
                  f, ensure_ascii=False, indent=2)

    log(f"\n结果已追加到：{path}")
    log(f"逐轮历史已保存：{hist_path}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
