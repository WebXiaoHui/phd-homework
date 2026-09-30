# -*- coding: utf-8 -*-
"""
diagnose_pooling.py —— 诊断"最小池化为什么这么差"（任务三）

【为什么要写这个脚本？】

    跑实验时发现：MinPooling 的验证损失会飙到 88 这种离谱的数值，
    测试准确率掉到 0.40（二分类瞎猜都有 0.5）。这看起来像代码写错了，
    但也可能是一个真实的数学现象。

    所以写这个脚本，把"池化之后的向量长什么样"直接打印出来看，
    用数据说话，而不是靠猜。

【核心怀疑：最小池化 + ReLU = 向量塌缩成同一个常数】

    模型里每一层后面都有 ReLU，所以节点特征里有大量 0（负的全被截成 0）。
    对这样的特征做"逐维取最小"：

        假设某个维度上，10 个节点的值是 [0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
        取最小 → 0
        另一张图这个维度是 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        取最小 → 0
        再换一张图 → 还是 0

    结果是：**不同图经过最小池化后，得到的图向量几乎一模一样**。
    图向量没有区分度，分类器就学不到东西。

    更糟的是，后面还接了一个 BatchNorm1d：
    BatchNorm 要"减均值、除以标准差"。如果一批图向量的方差接近 0，
    就是"除以一个接近 0 的数"，会把微小的数值波动放大几百倍，
    于是验证损失爆掉。

【这个脚本做什么？】

    对每种池化方式，随机初始化一个模型，跑一个 batch，然后统计：
        - 池化后图向量的**平均标准差**（越小说明越塌缩）
        - 池化后图向量第一维的取值范围
        - 经过池化层 BatchNorm+ReLU 之后的数值范围
    再算一下"池化前的节点特征"作对照，看塌缩到底是池化造成的还是本来就有。

【怎么运行？】
    python diagnose_pooling.py
    python diagnose_pooling.py --dataset MUTAG --model GCN
"""

import argparse
import os
import sys

import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from datasets import load_dataset                                    # noqa: E402
from models import POOL_NAMES, build_model, pool_nodes               # noqa: E402
from torch_geometric.loader import DataLoader                        # noqa: E402
from utils import resolve_device, set_seed                           # noqa: E402


def main():
    p = argparse.ArgumentParser(description="诊断不同池化方式对图向量的影响")
    p.add_argument("--dataset", default="PROTEINS", help="用哪个数据集")
    p.add_argument("--model", default="GAT", help="用哪个模型")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_graphs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device("auto")

    train_list, _, _, in_dim, out_dim, task_type = load_dataset(
        args.dataset, max_graphs=args.max_graphs, verbose=False)
    loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader)).to(device)

    print("=" * 78)
    print(f"池化方式诊断：数据集={args.dataset}  模型={args.model}  "
          f"层数={args.layers}  隐藏维度={args.hidden}")
    print(f"一个 batch 里有 {batch.num_graphs} 张图，共 {batch.num_nodes} 个节点")
    print("=" * 78)

    for pooling in POOL_NAMES:
        # 固定随机种子，保证不同池化方式用的是同一套初始权重，公平对比
        set_seed(args.seed)
        model = build_model(args.model, in_dim, args.hidden, out_dim,
                            num_layers=args.layers, pooling=pooling).to(device)
        model.eval()          # 用 eval 模式：BatchNorm 走滑动统计量，不受 batch 影响

        with torch.no_grad():
            h = model.encode(batch.x, batch.edge_index)     # 池化前的节点特征
            g = pool_nodes(h, batch.batch, pooling)         # 池化后的图向量
            g_bn = model.pool_norm(g)                       # 过池化层 BatchNorm
            g_out = torch.relu(g_bn)                        # 再过 ReLU

        # 关键指标 1：图向量的"平均标准差"。
        # 这个数越小，说明不同图的向量越像、越没有区分度（塌缩）。
        g_std = g.std(dim=0).mean().item()
        h_std = h.std(dim=0).mean().item()

        # 关键指标 2：池化层 BatchNorm 的输入方差。
        # 越接近 0，后面的"除以标准差"就越危险。
        g_var = g.var(dim=0).mean().item()

        # 关键指标 3：经过 BatchNorm 之后数值被放大了多少
        amp = (g_bn.abs().mean().item() / (g.abs().mean().item() + 1e-12))

        print(f"\n【{pooling.upper():<4}池化】")
        print(f"  池化前的节点特征      : 平均标准差 = {h_std:9.4f}   "
              f"（各维度标准差取平均）")
        print(f"  池化后的图向量        : 平均标准差 = {g_std:9.4f}   "
              f"← 越小说明不同图越像")
        print(f"  池化后图向量的维度方差: {g_var:.6f}   "
              f"← 接近 0 就是塌缩了")
        print(f"  第一维取值范围        : [{g[:, 0].min().item():.4f}, "
              f"{g[:, 0].max().item():.4f}]")
        print(f"  过 BatchNorm 后绝对值 : 均值 = {g_bn.abs().mean().item():9.4f}  "
              f"（放大倍数 {amp:8.2f}×）")
        print(f"  过 ReLU 后非零比例    : "
              f"{(g_out > 0).float().mean().item() * 100:5.1f}%")

    # ---- 再做一次"不同图之间两两相似度"的检查 ----
    print("\n" + "=" * 78)
    print("图向量之间的相似度（余弦相似度，越接近 1 说明越分不出来）")
    print("=" * 78)
    for pooling in POOL_NAMES:
        set_seed(args.seed)
        model = build_model(args.model, in_dim, args.hidden, out_dim,
                            num_layers=args.layers, pooling=pooling).to(device)
        model.eval()
        with torch.no_grad():
            g = model.pool_nodes_out(batch) if hasattr(model, "pool_nodes_out") \
                else pool_nodes(model.encode(batch.x, batch.edge_index),
                                batch.batch, pooling)
        gn = torch.nn.functional.normalize(g, dim=1)
        sim = gn @ gn.t()
        n = sim.size(0)
        off = (sim.sum() - sim.diag().sum()) / (n * n - n)
        print(f"  {pooling.upper():<4}: 不同图之间的平均余弦相似度 = {off:.4f}")
    print("\n说明：平均相似度越接近 1，说明所有图被池化成了几乎一样的向量，")
    print("      分类器拿到的输入没有区分度，自然学不到东西。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
