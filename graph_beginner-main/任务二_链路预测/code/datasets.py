# -*- coding: utf-8 -*-
"""
datasets.py —— 链路预测的数据划分（任务二：链路预测）

【链路预测到底在干什么？】
    给一张图，把一部分边藏起来，让模型去猜"这两个节点之间本来有没有边"。
    本质是一个二分类问题：有边=正样本，没边=负样本。

【本作业要求了解的知识点：训练集/验证集/测试集怎么划分】

    链路预测的划分比节点分类麻烦，因为"边"和"节点"不一样。这里要特别小心
    **信息泄漏（data leakage）**：如果测试用的边在训练时也被用来传递消息，
    那模型等于提前看到了答案，指标会虚高。

    所以本代码采用"无泄漏"的划分方式：

        第 1 步：把图变成无向图（加反向边），去掉自环
        第 2 步：只保留 u < v 的边，得到"唯一无向边集合"（每条边只算一次）
                 —— 这一步很关键！如果不做，同一条边 (u,v) 和 (v,u)
                    可能一个进训练集、一个进测试集，等于抄答案
        第 3 步：随机打乱，按 85% / 5% / 10% 切成 训练边 / 验证边 / 测试边
        第 4 步：**训练时的消息传递只用"训练边"构成的图**
                 验证/测试时也用同一张图（只用训练边），只是换一批边来打分

    负样本（不存在的边）怎么来？
        随机抽一对节点，如果这对节点在图里真的没边，就当成负样本。
        负样本不能是图里已有的边，否则就成了"假负样本"，会把模型教坏。

【几个名词对照】
    正样本 (positive) ：真实存在的边
    负样本 (negative) ：随机抽出来的、不存在的边
    edge_label_index  ：要打分的那些边，形状 (2, E)
    edge_label        ：每条边对应的标签，1=真边，0=假边
"""

import os
import sys

import torch

# Windows 控制台默认 GBK 编码，改成 UTF-8 避免打印中文/符号时报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# 复用任务一写好的数据加载逻辑（Cora / Citeseer / Flickr 的读取方式是一样的）
from torch_geometric.datasets import Flickr, Planetoid

from utils import DATA_DIR


def _load_base_dataset(name):
    """
    读取原始数据集（只要 x / edge_index / y，不做链路预测的划分）。

    说明：为了让每个任务的文件夹都能"单独提交、单独运行"，
    这里没有跨文件夹 import 任务一的代码，而是自己写了一份（内容很短）。
    """
    if name in ("Cora", "Citeseer"):
        ds = Planetoid(root=DATA_DIR, name=name)
    elif name == "Flickr":
        # 注意：PyG 的 Flickr 类的 raw 目录就是 root/raw，不会再拼数据集名字
        ds = Flickr(root=os.path.join(DATA_DIR, "Flickr"))
    else:
        raise ValueError(f"不支持的数据集：{name}。可选：Cora / Citeseer / Flickr")

    data = ds[0]
    data.x = data.x.float()      # Flickr 原始特征是 float64，转成 float32 省一半显存
    data.y = data.y.long()
    return data, ds.num_features, ds.num_classes


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def make_undirected(edge_index):
    """
    把有向边变成无向边：给每条边补一条反向边，然后去重。

    例子：输入 [[0,1],[1,0]]（0->1 和 1->0）
          输出还是 [[0,1],[1,0]]（去重后只留一份）
    """
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # torch.unique(..., dim=1) 会按"列"去重，正好就是我们要的边的去重
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def remove_self_loops(edge_index):
    """去掉自己连自己的边（自环）。自环在链路预测里没有意义。"""
    keep = edge_index[0] != edge_index[1]
    return edge_index[:, keep]


def edge_keys(edge_index, num_nodes):
    """
    把一条边 (u, v) 编码成一个整数 u * num_nodes + v。

    为什么要这么做？
        判断"某个随机抽出来的节点对是不是真的存在边"时，
        用整数集合查起来比一个一个比要快得多（可以用 torch.isin 向量化）。
    """
    return edge_index[0].to(torch.int64) * num_nodes + edge_index[1].to(torch.int64)


def sample_negative_edges(num_samples, num_nodes, real_keys, device, seed=None):
    """
    随机抽 num_samples 条"不存在的边"作为负样本。

    做法：随机抽节点对 -> 扔掉自环 -> 扔掉真实存在的边 -> 不够就再抽一轮。
    这种"抽了再筛"的方式叫拒绝采样（rejection sampling）。

    参数：
        real_keys : 图里**所有**真实边的整数编码（正反两个方向都要放进去），
                    用来排除"抽到的其实是真边"的情况
        seed      : 传了就用固定随机种子（结果可复现），不传就每轮随机

    实现细节：
        1) 随机数发生器必须和"抽出来的张量"在同一个设备上，
           不然 PyTorch 会报 "Expected a 'cuda' device type for generator"。
           所以下面按 device 来建 generator。
        2) 如果一次性要抽几百万条（比如 Flickr 的测试集），
           直接开一个巨大的张量会爆内存，所以分批处理（每批 50 万条）。
    """
    # 在正确的设备上创建随机数发生器
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    CHUNK = 500_000          # 单批最多抽多少条
    out_src, out_dst = [], []
    collected = 0
    # 最多循环 200 轮，防止图太稠密时死循环（那时"不存在的边"本来就很少）
    for _ in range(200):
        if collected >= num_samples:
            break
        need = min(num_samples - collected, CHUNK)
        # 多抽一些，因为会有一部分被筛掉
        cand = max(int(need * 1.3) + 64, 256)
        u = torch.randint(0, num_nodes, (cand,), device=device, generator=generator)
        v = torch.randint(0, num_nodes, (cand,), device=device, generator=generator)

        # 条件1：不是自环
        ok = u != v
        # 条件2：不是真实存在的边
        keys = u.to(torch.int64) * num_nodes + v.to(torch.int64)
        ok &= ~torch.isin(keys, real_keys)

        u, v = u[ok], v[ok]
        take = min(need, u.numel())
        out_src.append(u[:take])
        out_dst.append(v[:take])
        collected += take

    if collected < num_samples:
        raise RuntimeError(
            f"负采样失败：只抽到 {collected}/{num_samples} 条不存在的边。"
            f"说明图太稠密，随机抽到的节点对几乎都是真实边。")

    return torch.stack([torch.cat(out_src), torch.cat(out_dst)], dim=0)


# ---------------------------------------------------------------------------
# 链路预测的数据划分
# ---------------------------------------------------------------------------
class LinkSplit:
    """
    保存一次链路预测划分的所有内容。用起来像这样：

        split = build_link_split('Cora')
        z = encoder(split.x, split.train_edge_index)       # 编码
        score = decoder(z, split.train_pos_edge_index)     # 打分
        auc = evaluate(z, decoder, split.test_pos, split.test_neg)
    """

    def __init__(self, name, x, y, train_pos, val_pos, test_pos,
                 train_mp, val_neg, test_neg, real_keys, num_nodes, neg_per_pos):
        self.name = name
        self.x = x                        # (N, F) 节点特征
        self.y = y                        # (N,)   节点标签（链路预测用不到，留着备用）
        self.num_nodes = num_nodes
        self.train_pos = train_pos        # (2, E_tr) 训练用正样本边
        self.val_pos = val_pos            # (2, E_va) 验证用正样本边
        self.test_pos = test_pos          # (2, E_te) 测试用正样本边
        self.train_mp = train_mp          # (2, 2*E_tr) 消息传递用的边（双向）
        self.val_neg = val_neg            # (2, E_va * neg_per_pos) 固定的验证负样本
        self.test_neg = test_neg          # (2, E_te * neg_per_pos) 固定的测试负样本
        self.real_keys = real_keys        # 所有真实边的编码，用于负采样时排除
        self.neg_per_pos = neg_per_pos    # 每条正样本边配多少个负样本

    def describe(self):
        return (f"{self.name}: 节点={self.num_nodes}, "
                f"训练边={self.train_pos.size(1)}, "
                f"验证边={self.val_pos.size(1)}, "
                f"测试边={self.test_pos.size(1)}, "
                f"消息传递边={self.train_mp.size(1)}, "
                f"每个测试正样本配 {self.neg_per_pos} 个负样本")


def build_link_split(name, val_ratio=0.05, test_ratio=0.10, seed=42,
                     device="cpu", eval_neg_per_pos=100):
    """
    读数据 + 做链路预测划分。

    参数：
        name             : 'Cora' / 'Citeseer' / 'Flickr'
        val_ratio        : 验证边占的比例
        test_ratio       : 测试边占的比例（剩下的给训练）
        seed             : 随机种子，保证每次划分结果一样
        device           : 数据放在 CPU 还是 GPU
        eval_neg_per_pos : 评估时每条正样本边配多少个负样本。
                           为什么要配多个？因为 Hits@K 这种"排序指标"需要
                           让正样本和一批负样本比大小，只配 1 个没法排序。

    返回：
        LinkSplit 对象
    """
    data, num_features, num_classes = _load_base_dataset(name)

    x = data.x.float()
    num_nodes = x.size(0)

    # ---- 第 1 步：无向化 + 去自环 ----
    edge_index = make_undirected(data.edge_index.to(torch.int64))
    edge_index = remove_self_loops(edge_index)

    # ---- 第 2 步：只保留 u < v，得到"唯一无向边" ----
    row, col = edge_index
    only_one_dir = row < col
    undirected = torch.stack([row[only_one_dir], col[only_one_dir]], dim=0)
    num_edges = undirected.size(1)

    # ---- 第 3 步：随机切分 ----
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=g)

    n_test = int(num_edges * test_ratio)
    n_val = int(num_edges * val_ratio)

    test_pos = undirected[:, perm[:n_test]]
    val_pos = undirected[:, perm[n_test:n_test + n_val]]
    train_pos = undirected[:, perm[n_test + n_val:]]

    # ---- 第 4 步：消息传递用的图 = 只用训练边（加上反向边）----
    train_mp = torch.cat([train_pos, train_pos.flip(0)], dim=1)

    # ---- 第 5 步：准备负采样用到的"全部真实边"集合 ----
    # 注意这里用的是**原始完整图**的边，而不是只用训练边，
    # 这样抽出来的负样本在整张图里都确实是"不存在的边"，更干净。
    # 记得放到目标设备上：后面做 torch.isin 时两边必须在同一个设备。
    real_keys = edge_keys(edge_index, num_nodes).to(device)

    # ---- 第 6 步：验证集和测试集的负样本固定下来（保证每次评估标准一致）----
    # 每条正样本边配 eval_neg_per_pos 个负样本，这样后面才能算 Hits@K 排序指标
    val_neg = sample_negative_edges(val_pos.size(1) * eval_neg_per_pos,
                                    num_nodes, real_keys, device, seed=seed + 1)
    test_neg = sample_negative_edges(test_pos.size(1) * eval_neg_per_pos,
                                     num_nodes, real_keys, device, seed=seed + 2)

    # ---- 搬到指定设备上 ----
    def to_dev(t):
        return t.to(device) if torch.is_tensor(t) else t

    return LinkSplit(
        name=name,
        x=to_dev(x),
        y=to_dev(data.y.long()),
        train_pos=to_dev(train_pos),
        val_pos=to_dev(val_pos),
        test_pos=to_dev(test_pos),
        train_mp=to_dev(train_mp),
        val_neg=to_dev(val_neg),
        test_neg=to_dev(test_neg),
        real_keys=to_dev(real_keys),
        num_nodes=num_nodes,
        neg_per_pos=eval_neg_per_pos,
    )


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------
def auc_score(pos_score, neg_score):
    """
    AUC（ROC 曲线下面积）—— 链路预测最常用的指标。

    含义：随机取一条正样本和一条负样本，正样本分数更高的概率。
        AUC = 0.5  ->  模型完全没有区分能力（瞎猜）
        AUC = 1.0  ->  完美区分

    这里用"秩和公式"手算，不依赖 sklearn，逻辑更透明：
        AUC = (所有正负样本对的比较结果之和) / (正样本数 × 负样本数)
    等价的做法是把两个分数拼起来排序，看正样本的排名之和。
    """
    n_pos, n_neg = pos_score.numel(), neg_score.numel()
    if n_pos == 0 or n_neg == 0:
        return 0.0

    all_scores = torch.cat([pos_score, neg_score])
    # 升序排名（从 1 开始）
    order = torch.argsort(torch.argsort(all_scores)) + 1
    rank_sum_pos = order[:n_pos].sum().item()
    # 秩和公式（Mann-Whitney U 检验）
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def hits_at_k(pos_score, neg_score, k=50, neg_per_pos=100):
    """
    Hits@K —— "排序指标"：每条正样本边和 neg_per_pos 条负样本边放一起排名，
    如果正样本排进前 K 名，就算命中。

    返回命中比例（0~1，越大越好）。随机猜的话大约等于 K / (neg_per_pos+1)。

    用法举例：Hits@50 且 neg_per_pos=100 -> 随机猜时大约 50/101 ≈ 0.495

    【推导过程，别看晕了 —— 这里最容易写反】
        设 better = "这条正样本的分数比多少个负样本高"
        那么负样本中比它高的有 (neg_per_pos - better) 个
        它的排名 rank = (neg_per_pos - better) + 1
        命中条件 rank <= k
        代入化简：better >= neg_per_pos - k + 1
    """
    n_pos = pos_score.numel()
    if n_pos == 0:
        return 0.0

    # (n_pos, neg_per_pos)：每条正样本 vs 它对应的那一组负样本
    neg = neg_score.view(n_pos, neg_per_pos)
    # 每条正样本分数比多少个负样本高
    better = (pos_score.view(-1, 1) > neg).sum(dim=1)
    # 排名进入前 k 名 = 比 (neg_per_pos - k) 个以上的负样本都高
    hit = (better >= (neg_per_pos - k + 1)).float().mean()
    return float(hit)


if __name__ == "__main__":
    print("=" * 78)
    print("检查链路预测数据划分")
    print("=" * 78)
    for name in ["Cora", "Citeseer", "Flickr"]:
        print(f"\n>>> {name}")
        try:
            sp = build_link_split(name)
            print("    " + sp.describe())

            # 做一个简单的健全性检查：训练/验证/测试的边不能重合
            def key_set(e, n):
                return set((e[0] * n + e[1]).tolist())

            k_tr = key_set(sp.train_pos, sp.num_nodes)
            k_va = key_set(sp.val_pos, sp.num_nodes)
            k_te = key_set(sp.test_pos, sp.num_nodes)
            print(f"    训练∩验证 = {len(k_tr & k_va)}  训练∩测试 = {len(k_tr & k_te)}  "
                  f"验证∩测试 = {len(k_va & k_te)}  （都应该是 0，否则说明有信息泄漏）")

            # 负样本不能撞上真实边
            neg_keys = set((sp.test_neg[0] * sp.num_nodes + sp.test_neg[1]).tolist())
            all_real = set(sp.real_keys.tolist())
            print(f"    测试负样本数 = {len(neg_keys)}，"
                  f"其中撞上真实边的有 {len(neg_keys & all_real)} 条（应该是 0）")
            print("    [OK]")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [FAIL] {type(e).__name__}: {e}")
