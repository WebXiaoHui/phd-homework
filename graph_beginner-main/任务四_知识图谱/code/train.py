# -*- coding: utf-8 -*-
"""
train.py —— 知识图谱补全训练脚本（任务四：知识图谱）

【这个脚本在做什么？】

    训练一个 KGE 模型，学会判断三元组 (h, r, t) 是真是假，
    然后用"链接预测"的方式评估：把正确答案藏起来，看模型能不能排到前面。

【训练流程】

    1) 从训练集里取一批正样本三元组（比如 1024 个）
    2) 给每个正样本抽 K 个负样本（替换头实体或尾实体）
    3) 算分数：正样本分数应该高，负样本分数应该低
    4) 用损失函数把"正样本分高、负样本分低"这个目标变成梯度，更新参数
    5) 重复很多轮，定期在验证集上看 MRR 有没有提升，最后用最好的模型测测试集

【两种损失函数，怎么选？】

    --loss bce    （默认，推荐）
        把问题看成"二分类"：每个三元组猜它是真的还是假的。
        正样本标签 1、负样本标签 0，用 BCEWithLogitsLoss。
        好处：三个模型可以用完全一样的损失函数，对比最公平。

    --loss margin
        排序损失：要求"正样本分数 - 负样本分数 >= margin"。
        TransE / RotatE 的原论文用的就是这个。
        坏处：如果正负样本分数的绝对值整体漂移，损失仍然为 0，
              不太稳定（这也是为什么现在很多框架默认用 bce）。

【评估指标：MRR / Hits@K 是什么？】

    对每个测试三元组 (h, r, t)：
        1) 用模型给**所有**实体当尾实体的分数排个名
        2) 看正确答案 t 排在第几名，记为 rank

    然后：
        MRR      = rank 的倒数的平均值。
                  排第 1 名得 1 分，第 2 名得 0.5 分，第 10 名得 0.1 分。
                   越大越好，范围 (0, 1]
        Hits@1   = 有多少比例的三元组，正确答案排在第 1 名
        Hits@3   = 有多少比例排在前 3 名
        Hits@10  = 有多少比例排在前 10 名
                  越大越好，随机猜的话 Hits@10 ≈ 10 / 实体数，非常小

    还要算两个方向：
        "猜尾实体"：(h, r, ?)  -> 排名
        "猜头实体"：(?, r, t)  -> 排名
    最后把两个方向的结果平均，这是学术界报告结果的标准做法。

【怎么运行？看几个例子】

    # WN18RR 上的 TransE（默认配置）
    python train.py --dataset WN18RR --model TransE

    # FB15k-237 上的 RotatE，跑快一点
    python train.py --dataset FB15k-237 --model RotatE --epochs 50

    # ConvE
    python train.py --dataset WN18RR --model ConvE --dim 200

    # 只看配置，不训练
    python train.py --dataset WN18RR --model TransE --dry_run
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from datasets import DATASET_NAMES, load_dataset, sample_negatives   # noqa: E402
from models import DEFAULT_DIM, DEFAULT_MARGIN, MODEL_NAMES, build_model   # noqa: E402
from utils import (LOG_DIR, RESULT_DIR, add_common_args, count_parameters,  # noqa: E402
                   format_seconds, resolve_device, save_result, set_seed,
                   setup_logger, Timer)


# ---------------------------------------------------------------------------
# 过滤索引：为了"过滤式评估"提前准备好"要删掉哪些候选"
# ---------------------------------------------------------------------------
def build_filter_index(triples, filter_dict, key_fn):
    """
    给每个三元组预先算好"评估时要屏蔽掉的候选实体 id 列表"。

    为什么要预先算？
        每个 epoch 评估时都要用一次。如果每次临时去查字典、建列表，
        测试集有两万条三元组时会明显变慢。预先算一次，后面直接查列表就行。

    参数：
        triples     : (3, N) 三元组
        filter_dict : hr2tails 或 rt2heads 字典
        key_fn      : 怎么从三元组取出字典的 key
                      （猜尾实体时用 (h, r)，猜头实体时用 (r, t)）
    """
    out = []
    h_all, r_all, t_all = triples[0].tolist(), triples[1].tolist(), triples[2].tolist()
    for h, r, t in zip(h_all, r_all, t_all):
        key = key_fn(h, r, t)
        vals = filter_dict.get(key)
        out.append(list(vals) if vals else None)
    return out


# ---------------------------------------------------------------------------
# 评估：过滤式 MRR / Hits@K
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataset, triples, tail_filters, head_filters,
             device, chunk=256, max_triples=None, log=None):
    """
    在给定三元组集合上算 MRR / Hits@1 / Hits@3 / Hits@10。

    参数：
        triples      : (3, N) 要评估的三元组
        tail_filters : 猜尾实体时，每条三元组要屏蔽的实体 id 列表
        head_filters : 猜头实体时，每条三元组要屏蔽的实体 id 列表
        chunk        : 一次处理多少条（控制显存：chunk × 实体数 就是分数矩阵的大小）
        max_triples  : 最多评估多少条（验证集太大时可以用它加速）

    返回一个 dict：{mrr, hits@1, hits@3, hits@10, 两个方向分别的值}
    """
    model.eval()
    E = dataset.num_entities
    n = triples.size(1)
    if max_triples and n > max_triples:
        triples = triples[:, :max_triples]
        tail_filters = tail_filters[:max_triples]
        head_filters = head_filters[:max_triples]
        n = max_triples

    NEG_INF = float("-inf")
    ranks_tail = torch.empty(n, dtype=torch.float)
    ranks_head = torch.empty(n, dtype=torch.float)

    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        h = triples[0, start:end]
        r = triples[1, start:end]
        t = triples[2, start:end]
        c = end - start

        # ---------- 方向 1：猜尾实体 (h, r, ?) ----------
        # scores 的形状是 (c, E)，每一行是"这个实体当尾实体的分数"
        scores = model.score_all_tails(h, r)

        # 【关键】先把正确答案的分数保存下来，再去屏蔽其他候选。
        # 屏蔽的时候会把正确答案自己也设成 -inf，所以顺序不能反。
        true_score = scores.gather(1, t.view(-1, 1)).squeeze(1)

        for i in range(c):
            filt = tail_filters[start + i]
            if filt:
                scores[i, filt] = NEG_INF

        # 排名 = 比正确答案分数高的候选有几个 + 1
        # （并列时算"乐观排名"，也就是并列的都给较好的名次，
        #   这是知识图谱领域报告结果的标准做法）
        ranks_tail[start:end] = (scores > true_score.view(-1, 1)).sum(dim=1) + 1.0

        # ---------- 方向 2：猜头实体 (?, r, t) ----------
        scores = model.score_all_heads(r, t)
        true_score = scores.gather(1, h.view(-1, 1)).squeeze(1)

        for i in range(c):
            filt = head_filters[start + i]
            if filt:
                scores[i, filt] = NEG_INF

        ranks_head[start:end] = (scores > true_score.view(-1, 1)).sum(dim=1) + 1.0

    dt = time.time() - t0

    def metrics(ranks):
        return {
            "mrr": float((1.0 / ranks).mean()),
            "hits@1": float((ranks <= 1).float().mean()),
            "hits@3": float((ranks <= 3).float().mean()),
            "hits@10": float((ranks <= 10).float().mean()),
            "mean_rank": float(ranks.mean()),
        }

    mt = metrics(ranks_tail)
    mh = metrics(ranks_head)
    # 两个方向取平均，这是标准报告方式
    both = {k: (mt[k] + mh[k]) / 2.0
            for k in ["mrr", "hits@1", "hits@3", "hits@10", "mean_rank"]}

    result = {
        "mrr": both["mrr"], "hits@1": both["hits@1"],
        "hits@3": both["hits@3"], "hits@10": both["hits@10"],
        "mean_rank": both["mean_rank"],
        "mrr_tail": mt["mrr"], "hits@10_tail": mt["hits@10"],
        "mrr_head": mh["mrr"], "hits@10_head": mh["hits@10"],
        "eval_seconds": dt,
        "n_evaluated": n,
    }
    if log:
        log(f"    评估了 {n} 条三元组，用时 {format_seconds(dt)}")
    return result


# ---------------------------------------------------------------------------
# 训练一个 epoch
# ---------------------------------------------------------------------------
def run_one_epoch_1n(model, train_triples, optimizer, criterion,
                     num_entities, batch_size, device, label_smoothing=0.1):
    """
    【1-N 打分训练】—— 这是 ConvE 原论文的训练方式，本代码默认用它。

    什么叫 1-N？
        普通做法是"1 个正样本 + N 个负样本"（N 一般取几十上百）。
        1-N 打分的做法是：**一次算出这个 (h, r) 在所有实体上的分数**，
        然后拿它和"只有一个 1、其余全是 0"的标签做二分类。

        举例：batch 里有一条 (姚明, 出生于, 上海)
            模型输出 40943 个分数（每个实体一个）
            标签是 [0, 0, ..., 1（上海那个位置）, ..., 0, 0]

    为什么这样更好？
        1) 快得多：普通做法要编码 B×(1+N) 次，1-N 只要编码 B 次。
           N=64 时就快了 65 倍（对 ConvE 这种卷积模型尤其关键）。
        2) 负样本"全覆盖"：不是随机抽几十个，而是把全部实体都当负样本，
           没有采样偏差。

    代价：要把 (B, 实体数) 这么大的分数矩阵和标签都放进显存。
        所以 1-N 训练的 batch_size 要比普通做法小一些。

    【标签平滑】为什么不能直接用 0/1 当标签？
        因为负样本有 4 万多个、正样本只有 1 个，极度不平衡。
        如果标签严格用 0，模型只要"全部输出负"就能把损失降得很低，
        学不到东西。所以给负样本标签留一点点正数（比如 0.1/E），
        给正样本标签留一点点负数（1-0.1），这叫标签平滑（label smoothing），
        是 ConvE 论文里的标准做法，能显著稳定训练。
    """
    model.train()
    n_triples = train_triples.size(1)
    perm = torch.randperm(n_triples, device=device)

    total_loss = 0.0
    n_batches = 0

    t0 = time.time()
    for start in range(0, n_triples, batch_size):
        idx = perm[start:start + batch_size]
        pos = train_triples[:, idx]                  # (3, B)
        h, r, t = pos[0], pos[1], pos[2]
        B = h.size(0)

        # 一次算出所有实体当尾实体的分数
        scores = model.score_all_tails(h, r)         # (B, E)

        # 构造平滑后的标签
        smooth = label_smoothing / num_entities
        targets = torch.full_like(scores, smooth)
        # 正确的位置给 1 - label_smoothing + smooth
        targets.scatter_(1, t.view(-1, 1),
                         1.0 - label_smoothing + smooth)

        optimizer.zero_grad()
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    dt = time.time() - t0
    return total_loss / max(n_batches, 1), dt, n_batches


def run_one_epoch_sampled(model, train_triples, optimizer, criterion, loss_type,
                          num_neg, num_entities, batch_size, device,
                          label_smoothing=0.0):
    """
    【负采样训练】每个正样本随机抽 N 个负样本。

    这是 TransE / RotatE 原论文的做法。本代码里它作为 1-N 的备选方案，
    用 --loss bce 或 --loss margin 打开。

    返回 (平均损失, 耗时, batch 个数)。
    """
    model.train()
    n_triples = train_triples.size(1)
    perm = torch.randperm(n_triples, device=device)

    total_loss = 0.0
    n_batches = 0

    t0 = time.time()
    for start in range(0, n_triples, batch_size):
        idx = perm[start:start + batch_size]
        pos = train_triples[:, idx]                 # (3, B)
        B = pos.size(1)

        # ---- 正样本分数 ----
        pos_score = model.score(pos[0], pos[1], pos[2])       # (B,)

        # ---- 抽负样本并打分 ----
        neg = sample_negatives(pos, num_neg, num_entities, device)   # (3, B*K)
        neg_score = model.score(neg[0], neg[1], neg[2])              # (B*K,)

        optimizer.zero_grad()

        if loss_type == "bce":
            # 二分类视角：正样本当 1，负样本当 0
            scores = torch.cat([pos_score, neg_score])
            if label_smoothing > 0:
                labels = torch.cat([
                    torch.full((B,), 1.0 - label_smoothing, device=device),
                    torch.full((B * num_neg,), label_smoothing, device=device),
                ])
            else:
                labels = torch.cat([
                    torch.ones(B, device=device),
                    torch.zeros(B * num_neg, device=device),
                ])
            loss = criterion(scores, labels)
        else:
            # 排序视角：正样本分数要比每个负样本都高出 margin
            pos_rep = pos_score.repeat_interleave(num_neg)
            targets = torch.ones_like(neg_score)
            loss = criterion(pos_rep, neg_score, targets)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    dt = time.time() - t0
    return total_loss / max(n_batches, 1), dt, n_batches


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="知识图谱补全训练脚本（任务四）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)

    g = p.add_argument_group("任务四专用参数")
    g.add_argument("--dataset", type=str, default="WN18RR", choices=DATASET_NAMES,
                   help="用哪个知识图谱")
    g.add_argument("--model", type=str, default="TransE", choices=MODEL_NAMES,
                   help="用哪个 KGE 模型")
    g.add_argument("--dim", type=int, default=0,
                   help="embedding 维度（0 = 用该模型的推荐值）")
    g.add_argument("--margin", type=float, default=0.0,
                   help="margin 排序损失里的 margin（0 = 用推荐值）")
    g.add_argument("--num_neg", type=int, default=64,
                   help="每条正样本抽多少个负样本（只在 loss=bce/margin 时生效）")
    g.add_argument("--loss", type=str, default="bce", choices=["1n", "bce", "margin"],
                   help="损失函数：bce=负采样二分类（默认，三个模型通用） "
                        "margin=负采样排序损失 1n=1-N打分（只建议给 ConvE 用）")
    g.add_argument("--label_smoothing", type=float, default=0.1,
                   help="标签平滑系数（0 表示不平滑）。只在 loss=1n 时生效")
    g.add_argument("--neg_label_smoothing", type=float, default=0.0,
                   help="负采样训练(bce/margin)时的标签平滑系数。默认 0，"
                        "因为负采样时每个正样本只有几十个负样本，"
                        "把标签压到 0.9/0.1 会给损失加一个下界，"
                        "模型学到这点区分度就再也不下降了（实测 loss 卡在 0.35 不动）")
    g.add_argument("--eval_every", type=int, default=5,
                   help="每多少轮在验证集上评估一次")
    g.add_argument("--eval_chunk", type=int, default=256,
                   help="评估时一次算多少条（越大越快但越占显存）")
    g.add_argument("--valid_limit", type=int, default=3000,
                   help="验证集最多评估多少条（0=全部，加速用）")
    g.add_argument("--patience", type=int, default=0,
                   help="验证集多少轮没提升就停止训练（0=不提前停止）")
    g.add_argument("--dry_run", action="store_true", help="只打印配置，不训练")

    args = p.parse_args()

    # ---------------- 基础设置 ----------------
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.dim == 0:
        args.dim = DEFAULT_DIM[args.model]
    if args.margin == 0.0:
        args.margin = DEFAULT_MARGIN[args.model]
    # ConvE 的 dim 有整除约束，提前检查，免得训练到一半才报错
    if args.model == "ConvE" and (2 * args.dim) % 10 != 0:
        raise SystemExit(
            f"[错误] ConvE 要求 2*dim 能被 10 整除（要排成 10 行的二维矩阵），"
            f"当前 dim={args.dim}。请改用 dim=100 / 200 这类值。")

    exp_name = (f"{args.dataset}_{args.model}_dim{args.dim}_{args.loss}"
                f"_lr{args.lr}"
                + (f"_{args.tag}" if args.tag else ""))
    log_file = os.path.join(LOG_DIR, args.log or f"train_{exp_name}.log")
    logger = setup_logger(f"train4_{exp_name}", log_file, quiet=args.quiet)

    def log(msg=""):
        logger.info(msg)

    log("=" * 78)
    log(f"任务四 知识图谱补全 —— {exp_name}")
    log(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 78)
    log(f"数据集={args.dataset}  模型={args.model}  embedding维度={args.dim}")
    if args.loss == "1n":
        log(f"损失函数=1-N 打分（一次给所有实体打分，二分类 + 标签平滑 "
            f"{args.label_smoothing}）")
    else:
        log(f"损失函数={args.loss}  负样本数/正样本={args.num_neg}  "
            f"margin={args.margin}  标签平滑={args.neg_label_smoothing}")
    log(f"学习率={args.lr}  权重衰减={args.weight_decay}  轮数={args.epochs}  "
        f"批次大小={args.batch_size}")
    log(f"随机种子={args.seed}")

    if args.dry_run:
        log("\n[--dry_run] 配置检查完毕，没有真的开始训练。")
        return 0

    # ---------------- 读数据 ----------------
    log("\n" + "-" * 78)
    log("第 1 步：加载知识图谱数据")
    log("-" * 78)
    t_load = Timer()
    with t_load:
        ds = load_dataset(args.dataset, verbose=not args.quiet)
    log(f"  数据加载用时 {format_seconds(t_load.elapsed)}")
    ds.to(device)
    log(f"  训练集三元组 = {ds.train.size(1)}")
    log(f"  验证集三元组 = {ds.valid.size(1)}")
    log(f"  测试集三元组 = {ds.test.size(1)}")

    # 预先算好评估要用的过滤表
    log("\n  正在准备过滤式评估要用的索引……")
    t_f = Timer()
    with t_f:
        valid_tail_filt = build_filter_index(
            ds.valid, ds.hr2tails, lambda h, r, t: (h, r))
        valid_head_filt = build_filter_index(
            ds.valid, ds.rt2heads, lambda h, r, t: (r, t))
        test_tail_filt = build_filter_index(
            ds.test, ds.hr2tails, lambda h, r, t: (h, r))
        test_head_filt = build_filter_index(
            ds.test, ds.rt2heads, lambda h, r, t: (r, t))
    log(f"  准备完成，用时 {format_seconds(t_f.elapsed)}")
    log("  【为什么要过滤？】像 (姚明,出生于,上海) 和 (姚明,出生于,松江) "
        "可能都是真的，")
    log("     如果排名时把'其他也对的答案'算进去，名次会被白白拉低。"
        "所以评估时要把它们屏蔽掉。")

    # ---------------- 建模型 ----------------
    log("\n" + "-" * 78)
    log("第 2 步：搭建模型")
    log("-" * 78)
    model = build_model(args.model, ds.num_entities, ds.num_relations,
                        dim=args.dim, margin=args.margin).to(device)
    n_param = count_parameters(model)
    log(f"  {args.model}：实体数={ds.num_entities}  关系数={ds.num_relations}")
    log(f"  可训练参数量 = {n_param:,}")
    log(f"  模型结构：")
    for line in str(model).split("\n"):
        log(f"    {line}")

    if args.loss in ("1n", "bce"):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MarginRankingLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ---------------- 训练 ----------------
    log("\n" + "-" * 78)
    log("第 3 步：开始训练")
    log("-" * 78)
    log(f"{'轮次':>6}{'训练损失':>12}{'验证MRR':>11}{'验证H@1':>10}"
        f"{'验证H@10':>11}{'每轮耗时':>11}")

    best_val_mrr = None
    best_state = None
    best_epoch = -1
    best_test = None
    train_time_total = 0.0
    train_time_list = []
    epoch_rows = []
    n_no_improve = 0
    t_all = time.time()

    for epoch in range(1, args.epochs + 1):
        if args.loss == "1n":
            tr_loss, tr_dt, n_batches = run_one_epoch_1n(
                model, ds.train, optimizer, criterion, ds.num_entities,
                args.batch_size, device, label_smoothing=args.label_smoothing)
        else:
            tr_loss, tr_dt, n_batches = run_one_epoch_sampled(
                model, ds.train, optimizer, criterion, args.loss,
                args.num_neg, ds.num_entities, args.batch_size, device,
                label_smoothing=args.neg_label_smoothing)
        train_time_total += tr_dt
        if epoch > 1:      # 第 1 轮含 CUDA 初始化，计时会失真
            train_time_list.append(tr_dt)

        # ---- 定期在验证集上评估 ----
        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) \
            or (epoch == args.epochs)
        if do_eval:
            val = evaluate(model, ds, ds.valid, valid_tail_filt, valid_head_filt,
                           device, chunk=args.eval_chunk,
                           max_triples=args.valid_limit or None)
            vmrr = val["mrr"]
            log(f"{epoch:>6}{tr_loss:>12.4f}{vmrr:>11.4f}{val['hits@1']:>10.4f}"
                f"{val['hits@10']:>11.4f}{tr_dt:>10.2f}s")

            epoch_rows.append({
                "epoch": epoch,
                "train_loss": round(tr_loss, 5),
                "val_mrr": round(vmrr, 5),
                "val_hits@1": round(val["hits@1"], 5),
                "val_hits@10": round(val["hits@10"], 5),
                "train_seconds": round(tr_dt, 4),
            })

            if best_val_mrr is None or vmrr > best_val_mrr:
                best_val_mrr = vmrr
                best_epoch = epoch
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                n_no_improve = 0
            else:
                n_no_improve += 1
                if args.patience > 0 and n_no_improve >= args.patience:
                    log(f"\n  验证集连续 {args.patience} 次评估没有提升，"
                        f"提前停止训练。")
                    break
        else:
            log(f"{epoch:>6}{tr_loss:>12.4f}{'--':>11}{'--':>10}{'--':>11}"
                f"{tr_dt:>10.2f}s")

    total_wall = time.time() - t_all

    # ---------------- 用最好的模型测测试集 ----------------
    log("\n" + "-" * 78)
    log("第 4 步：用验证集上最好的模型测测试集（过滤式评估）")
    log("-" * 78)
    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"  最好的一次在第 {best_epoch} 轮，验证集 MRR = {best_val_mrr:.4f}")

    test = evaluate(model, ds, ds.test, test_tail_filt, test_head_filt,
                    device, chunk=args.eval_chunk, max_triples=0 or None)
    best_test = test

    log("")
    log("  ┌─────────────────────────────────────────────────────────┐")
    log("  │                测试集结果（过滤式评估）                 │")
    log("  ├─────────────────────────────────────────────────────────┤")
    log(f"  │  MRR      = {test['mrr']:.4f}                                    │")
    log(f"  │  Hits@1   = {test['hits@1']:.4f}                                    │")
    log(f"  │  Hits@3   = {test['hits@3']:.4f}                                    │")
    log(f"  │  Hits@10  = {test['hits@10']:.4f}    <<< 最常用来对比的指标        │")
    log(f"  │  平均排名 = {test['mean_rank']:.1f}（越小越好；实体总数 {ds.num_entities}）  │")
    log("  ├─────────────────────────────────────────────────────────┤")
    log(f"  │  猜尾实体 (h,r,?)： MRR={test['mrr_tail']:.4f}  "
        f"Hits@10={test['hits@10_tail']:.4f}          │")
    log(f"  │  猜头实体 (?,r,t)： MRR={test['mrr_head']:.4f}  "
        f"Hits@10={test['hits@10_head']:.4f}          │")
    log("  └─────────────────────────────────────────────────────────┘")

    # ---------------- 时间统计 ----------------
    per_epoch = (sum(train_time_list) / len(train_time_list)
                 if train_time_list else train_time_total / max(1, len(epoch_rows)))
    log("")
    log("时间统计：")
    log(f"  纯训练总耗时     = {format_seconds(train_time_total)}")
    log(f"  平均每轮训练耗时 = {format_seconds(per_epoch)}")
    log(f"  每个 epoch 的 batch 数 = "
        f"{math.ceil(ds.train.size(1) / args.batch_size)}")
    log(f"  整个脚本总耗时   = {format_seconds(total_wall)}")

    # ---------------- 保存结果 ----------------
    record = {
        "task": "knowledge_graph",
        "dataset": args.dataset,
        "model": args.model,
        "dim": args.dim,
        "margin": args.margin,
        "loss": args.loss,
        "num_neg": args.num_neg,
        "label_smoothing": (args.label_smoothing if args.loss == "1n"
                            else args.neg_label_smoothing),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_run": args.epochs,
        "best_epoch": best_epoch,
        "n_params": n_param,
        "num_entities": ds.num_entities,
        "num_relations": ds.num_relations,
        "best_val_mrr": best_val_mrr,
        "test_mrr": test["mrr"],
        "test_hits@1": test["hits@1"],
        "test_hits@3": test["hits@3"],
        "test_hits@10": test["hits@10"],
        "test_mean_rank": test["mean_rank"],
        "test_mrr_tail": test["mrr_tail"],
        "test_mrr_head": test["mrr_head"],
        "train_time_total": train_time_total,
        "train_time_per_epoch": per_epoch,
        "wall_time_total": total_wall,
        "seed": args.seed,
        "tag": args.tag,
    }
    path = save_result(record, args.result_file)

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
