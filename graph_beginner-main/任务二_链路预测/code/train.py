# -*- coding: utf-8 -*-
"""
train.py —— 链路预测训练主脚本（任务二）

和任务一一样，支持两种训练方式：

    --mode full    全图训练：一次把整张图编码成 embedding，然后给所有训练边打分
    --mode sample  子图采样训练：用 NeighborLoader 采一批种子节点构成的子图，
                   只在这个子图内部给边打分

【全图 vs 采样：链路预测里有点不一样的地方，新手请注意】

    节点分类里，"样本"就是节点，很好切分。
    链路预测里，"样本"是边，麻烦一点：

    - 全图模式：直接把所有训练边拿来打分，最直接。
    - 采样模式：我们要先决定"给哪些边打分"。本代码的做法是：
          1) 用训练边的两个端点作为"种子节点"去采样
          2) NeighborLoader 返回的子图里，前 batch_size 个节点就是这批种子
          3) 子图中两端**都在种子集合里**的边，就是这一批要训练的正样本边
      这样正的样本边一定能拿到自己两个端点的 embedding，逻辑上是自洽的。

【评估指标】
    AUC      ：随机取一条真边、一条假边，真边分数更高的概率（0.5=瞎猜，1=完美）
    Hits@50  ：每条真边和 100 条假边一起排名，排进前 50 名的比例
               （随机猜大约 50/101 ≈ 0.495，所以明显高于 0.5 才算学到了东西）

【怎么运行？】
    python train.py --dataset Cora --model GCN --mode full
    python train.py --dataset Flickr --model GAT --mode sample --batch_size 4096
    python run_all.py            # 一键跑完所有对比实验
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import (auc_score, build_link_split, hits_at_k,
                      sample_negative_edges)
from models import DECODER_NAMES, MODEL_NAMES, build_decoder, build_encoder
from utils import (LOG_DIR, RESULT_DIR, Timer, add_common_args,
                   count_parameters, format_seconds, resolve_device,
                   save_result, set_seed, setup_logger)


# ---------------------------------------------------------------------------
# 打分工具（分块进行，避免一次算几百万条边把显存撑爆）
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_edges_chunked(encoder, decoder, x, edge_index, edge_label_index,
                        chunk_size=300_000):
    """
    给一大批边打分，分块计算。

    为什么要分块？
        Flickr 的测试集有 4.5 万条真边 + 450 万条假边。
        如果一次性算，光是把 z[src] 这个 (450万, 64) 的张量取出来就要 1GB 显存，
        显卡直接爆掉。分块算就没事了。

    返回：(E,) 的打分结果
    """
    z = encoder(x, edge_index)
    n = edge_label_index.size(1)
    out = []
    for start in range(0, n, chunk_size):
        sub = edge_label_index[:, start:start + chunk_size]
        out.append(decoder(z, sub))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# 训练一个 epoch
# ---------------------------------------------------------------------------
def run_epoch_full(encoder, decoder, sp, optimizer, device, neg_ratio=1):
    """
    全图模式训练一个 epoch。

    步骤：
        1) 用训练边构成的图，把整张图编码成 embedding z
        2) 真边打分 -> 希望分数高（标签 1）
        3) 重新随机抽一批假边打分 -> 希望分数低（标签 0）
        4) 把正负样本的损失加起来，反向传播
    """
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    t0 = time.time()
    z = encoder(sp.x, sp.train_mp)
    pos_score = decoder(z, sp.train_pos)

    # 每一轮重新抽负样本：每轮见到的"假边"都不一样，模型不容易记住固定的答案
    n_neg = sp.train_pos.size(1) * neg_ratio
    neg_edge = sample_negative_edges(n_neg, sp.num_nodes, sp.real_keys, device)
    neg_score = decoder(z, neg_edge)

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    loss.backward()
    optimizer.step()
    dt = time.time() - t0
    return float(loss), dt, 1


def sample_in_batch_negatives(count, batch_size, n_id, real_keys,
                              num_nodes, device):
    """
    在**一个采样出来的子图内部**抽负样本。

    难点：NeighborLoader 返回的是"局部编号"，我们需要：
        局部编号 --(查 n_id)--> 全局编号 --(编码成整数)--> 判断是不是真实边

    参数：
        batch_size : 这批种子节点的数量（局部编号 0 ~ batch_size-1 就是种子）
        n_id       : 子图节点在**原图**中的编号，长度 = 子图节点数
    """
    out_a, out_b = [], []
    collected = 0
    for _ in range(50):
        if collected >= count:
            break
        need = count - collected
        cand = max(int(need * 1.5) + 64, 256)
        a = torch.randint(0, batch_size, (cand,), device=device)
        b = torch.randint(0, batch_size, (cand,), device=device)

        # 条件1：不是同一个节点（去自环）
        ok = a != b
        # 条件2：映射回全局编号后，这个节点对在原图里真的没边
        ga = n_id[a].to(torch.int64)
        gb = n_id[b].to(torch.int64)
        ok &= ~torch.isin(ga * num_nodes + gb, real_keys)

        a, b = a[ok], b[ok]
        take = min(need, a.numel())
        out_a.append(a[:take])
        out_b.append(b[:take])
        collected += take
        if a.numel() == 0:
            break

    if collected == 0:
        return None
    return torch.stack([torch.cat(out_a), torch.cat(out_b)], dim=0)


def run_epoch_sampler(encoder, decoder, loader, sp, optimizer, device):
    """
    采样模式训练一个 epoch。

    每个 batch：
        1) 拿到子图（前 batch_size 个节点 = 种子节点）
        2) 编码子图里的所有节点
        3) 子图中"两端都在种子集合里"的边，就是这个 batch 的正样本边
        4) 在种子集合内部随机抽负样本边
        5) 算二分类损失并更新参数
    """
    encoder.train()
    decoder.train()

    total_loss, n_batches = 0.0, 0
    t0 = time.time()

    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size      # 这批种子节点个数

        row, col = batch.edge_index
        # 只保留"两端都是种子节点"的边（局部编号 < bs）
        # 这里不用 row<col 去重：无向图里 (u,v) 和 (v,u) 都算正样本，
        # 这是链路预测里常见的做法，不影响的。
        m = (row < bs) & (col < bs) & (row != col)
        if int(m.sum()) < 2:
            continue               # 这个 batch 内部边太少，跳过（避免梯度没意义）

        pos_edge = torch.stack([row[m], col[m]], dim=0)

        z = encoder(batch.x, batch.edge_index)
        pos_score = decoder(z, pos_edge)

        neg_edge = sample_in_batch_negatives(
            pos_edge.size(1), bs, batch.n_id, sp.real_keys, sp.num_nodes, device)
        if neg_edge is None:
            continue
        neg_score = decoder(z, neg_edge)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
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
def evaluate(encoder, decoder, sp, device, which="val", chunk_size=300_000):
    """
    评估：用**全图**做一次前向，然后给验证集 / 测试集的边打分。

    为什么评估要用全图而不是子图？
        理由和任务一一样：评估时图放得下就全图算，得到的指标才代表
        "这个模型对整张图的理解有多好"，不受采样随机性的干扰。

    返回：(AUC, Hits@50, 推理耗时)
    """
    encoder.eval()
    decoder.eval()

    pos_edge = sp.val_pos if which == "val" else sp.test_pos
    neg_edge = sp.val_neg if which == "val" else sp.test_neg

    t0 = time.time()
    z = encoder(sp.x, sp.train_mp)

    # --- 分块给正样本打分 ---
    pos_scores = []
    for s in range(0, pos_edge.size(1), chunk_size):
        pos_scores.append(decoder(z, pos_edge[:, s:s + chunk_size]))
    pos_score = torch.cat(pos_scores)

    # --- 分块给负样本打分 ---
    # 注意：负样本是按"每条正样本配 neg_per_pos 个"的顺序排好的，
    #       所以这里必须按同样的顺序分块，Hits@K 的分组才对得上。
    neg_scores = []
    for s in range(0, neg_edge.size(1), chunk_size):
        neg_scores.append(decoder(z, neg_edge[:, s:s + chunk_size]))
    neg_score = torch.cat(neg_scores)
    dt = time.time() - t0

    auc = auc_score(pos_score, neg_score)
    h50 = hits_at_k(pos_score, neg_score, k=50, neg_per_pos=sp.neg_per_pos)
    return auc, h50, dt


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="链路预测训练脚本（任务二）")
    parser.add_argument("--dataset", default="Cora", help="Cora / Citeseer / Flickr")
    parser.add_argument("--model", default="GCN", help=" / ".join(MODEL_NAMES))
    parser.add_argument("--mode", default="full", choices=["full", "sample"])
    parser.add_argument("--decoder", default="dot", choices=DECODER_NAMES,
                        help="dot=内积解码器, mlp=MLP解码器")
    parser.add_argument("--emb_dim", type=int, default=64, help="节点 embedding 维度")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="[仅 sample 模式] 每批种子节点数")
    parser.add_argument("--num_neighbors", type=int, default=10,
                        help="[仅 sample 模式] 每层采样邻居数")
    parser.add_argument("--neg_ratio", type=int, default=1,
                        help="[仅 full 模式] 训练时每条正样本边配几个负样本")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每多少轮评估一次（大图建议调大）")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证边比例")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="测试边比例")
    add_common_args(parser)
    args = parser.parse_args()

    # ---------------- 0. 准备 ----------------
    set_seed(args.seed)
    device = resolve_device(args.device)

    log_name = args.log or (f"{args.dataset}_{args.model}_{args.mode}_"
                            f"{args.decoder}_lr{args.lr}_L{args.layers}.log")
    logger = setup_logger(
        name=f"task2_{args.dataset}_{args.model}_{args.mode}_{args.decoder}_{args.lr}_{args.layers}",
        log_file=os.path.join(LOG_DIR, log_name), quiet=args.quiet)

    logger.info("=" * 78)
    logger.info(f"任务二 链路预测 | 数据集={args.dataset} 模型={args.model} "
                f"训练方式={args.mode} 解码器={args.decoder}")
    logger.info(f"超参数：lr={args.lr}, layers={args.layers}, hidden={args.hidden}, "
                f"emb_dim={args.emb_dim}, dropout={args.dropout}, epochs={args.epochs}")
    logger.info("=" * 78)

    # ---------------- 1. 数据划分 ----------------
    try:
        sp = build_link_split(args.dataset, val_ratio=args.val_ratio,
                              test_ratio=args.test_ratio, seed=args.seed,
                              device=device)
    except Exception as e:
        logger.info(f"[错误] 数据准备失败：{e}")
        return 1
    logger.info("数据划分：" + sp.describe())

    # ---------------- 2. 模型 ----------------
    encoder = build_encoder(args.model, sp.x.size(1), args.hidden, args.emb_dim,
                            num_layers=args.layers, dropout=args.dropout).to(device)
    decoder = build_decoder(args.decoder, args.emb_dim).to(device)

    n_params = count_parameters(encoder) + count_parameters(decoder)
    logger.info(f"模型参数量：{n_params:,}（编码器 {count_parameters(encoder):,} "
                f"+ 解码器 {count_parameters(decoder):,}）")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # ---------------- 3. 采样器 ----------------
    loader = None
    steps_per_epoch = 1
    if args.mode == "sample":
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        # 只含训练边的图，用它来采样子图
        #
        # 【为什么要多存一个 n_id ？】—— 这是本环境 PyG 2.0.4 的一个兼容处理
        #   我们采样子图后，需要知道"子图里的第 i 个节点，在原图里是第几号节点"，
        #   这样才能判断抽出来的负样本在原图里到底是不是真的没有边。
        #   新版 PyG（>=2.1）的 NeighborLoader 会自动返回一个 batch.n_id，
        #   但 2.0.4 还没有这个功能。
        #   好在 PyG 的 filter_node_store_ 会把所有"长度等于节点数"的属性
        #   按采样结果一起切分，所以我们把 n_id = 0,1,2,...,N-1 当作一个
        #   节点属性存进去，采样后它就自动变成"子图节点对应的原图编号"了。
        num_nodes = sp.x.size(0)
        train_graph = Data(x=sp.x.cpu(),
                           edge_index=sp.train_mp.cpu(),
                           n_id=torch.arange(num_nodes, dtype=torch.long))

        # 种子节点 = 出现在训练边里的所有节点
        seeds = torch.unique(sp.train_pos.reshape(-1))

        loader = NeighborLoader(
            train_graph,
            num_neighbors=[args.num_neighbors] * args.layers,
            batch_size=args.batch_size,
            input_nodes=seeds,
            shuffle=True,
            num_workers=0,
        )
        steps_per_epoch = len(loader)
        logger.info(f"采样设置：batch_size={args.batch_size}, "
                    f"num_neighbors={[args.num_neighbors] * args.layers}, "
                    f"种子节点={seeds.numel()}, 每轮 batch 数={steps_per_epoch}")
    else:
        logger.info(f"全图训练：每个 epoch 给全部 {sp.train_pos.size(1)} 条训练边打分")

    # ---------------- 4. 训练 ----------------
    logger.info("-" * 78)
    logger.info(f"{'Epoch':>6} | {'Loss':>8} | {'Val AUC':>9} | {'Val H@50':>9} | "
                f"{'Test AUC':>9} | {'时间/轮':>9}")
    logger.info("-" * 78)

    best_val_auc = -1.0
    best_test_auc = 0.0
    best_test_h50 = 0.0
    best_epoch = 0
    history = []
    train_time_total = 0.0
    train_time_list = []
    eval_time = 0.0

    t_start = Timer.now()
    for epoch in range(1, args.epochs + 1):
        if args.mode == "sample":
            loss, dt, nb = run_epoch_sampler(encoder, decoder, loader, sp, optimizer, device)
        else:
            loss, dt, nb = run_epoch_full(encoder, decoder, sp, optimizer, device,
                                          neg_ratio=args.neg_ratio)

        train_time_total += dt
        if epoch > 1:                       # 第 1 轮有初始化开销，统计时排除
            train_time_list.append(dt)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_auc, val_h50, ev_t = evaluate(encoder, decoder, sp, device, "val")
            test_auc, test_h50, _ = evaluate(encoder, decoder, sp, device, "test")
            eval_time += ev_t

            # 用验证集 AUC 挑最佳模型（不能看测试集挑，否则结果不客观）
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_h50 = test_h50
                best_epoch = epoch

            history.append({"epoch": epoch, "loss": loss, "val_auc": val_auc,
                            "val_hits50": val_h50, "test_auc": test_auc,
                            "test_hits50": test_h50, "train_time": dt})
            logger.info(f"{epoch:>6d} | {loss:>8.4f} | {val_auc:>9.4f} | "
                        f"{val_h50:>9.4f} | {test_auc:>9.4f} | {dt:>8.3f}s")

    wall_time = Timer.now() - t_start
    mean_epoch_time = (sum(train_time_list) / len(train_time_list)
                       if train_time_list else train_time_total / args.epochs)

    # ---------------- 5. 汇总 ----------------
    logger.info("-" * 78)
    logger.info("【结果汇总】")
    logger.info(f"  最佳验证 AUC      : {best_val_auc:.4f}  (第 {best_epoch} 轮)")
    logger.info(f"  对应的测试 AUC    : {best_test_auc:.4f}")
    logger.info(f"  对应的测试 Hits@50: {best_test_h50:.4f}  "
                f"(随机猜约 {50 / (sp.neg_per_pos + 1):.3f})")
    logger.info(f"  训练总耗时        : {format_seconds(train_time_total)} / "
                f"整体墙钟 {format_seconds(wall_time)}")
    logger.info(f"  平均每轮训练耗时  : {format_seconds(mean_epoch_time)} "
                f"(每轮 {steps_per_epoch} 次参数更新)")
    logger.info(f"  评估累计耗时      : {format_seconds(eval_time)}")

    # ---------------- 6. 保存 ----------------
    record = {
        "task": "link_prediction",
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "decoder": args.decoder,
        "lr": args.lr,
        "layers": args.layers,
        "hidden": args.hidden,
        "emb_dim": args.emb_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size if args.mode == "sample" else None,
        "num_neighbors": args.num_neighbors if args.mode == "sample" else None,
        "best_val_auc": round(best_val_auc, 4),
        "best_test_auc": round(best_test_auc, 4),
        "best_test_hits50": round(best_test_h50, 4),
        "best_epoch": best_epoch,
        "train_time_total": round(train_time_total, 3),
        "train_time_per_epoch": round(mean_epoch_time, 4),
        "eval_time_total": round(eval_time, 3),
        "wall_time": round(wall_time, 3),
        "steps_per_epoch": steps_per_epoch,
        "num_params": n_params,
        "tag": args.tag,
    }
    path = save_result(record, args.result_file)
    logger.info(f"结果已追加保存到：{path}")

    hist_path = os.path.join(RESULT_DIR, "history",
                             f"{args.dataset}_{args.model}_{args.mode}_"
                             f"{args.decoder}_lr{args.lr}_L{args.layers}.json")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
