# -*- coding: utf-8 -*-
"""
models.py —— 链路预测用的 GNN 编码器 + 解码器（任务二：链路预测）

【链路预测和节点分类有什么不同？】

    节点分类：输入图 -> 每个节点输出一个类别            （一层输出头就完事）
    链路预测：输入图 -> 判断"某两个节点之间该不该有边"   （需要两步）

    所以链路预测的模型由两部分组成：

        1) 编码器 Encoder（就是 GCN/GAT/GraphSAGE/GIN）
             把每个节点变成一个向量 z_i（我们叫它"节点表示"或 embedding）

        2) 解码器 Decoder（本文件里的 DotDecoder / MLPDecoder）
             输入两个节点的向量 (z_u, z_v)，输出一个分数
             分数越高 = 越可能有边

    这种设计叫 "编码器-解码器"（Encoder-Decoder）框架，是链路预测的标准套路。
    好处是：编码器可以用任意 GNN，解码器也可以随便换，两者互相独立。

【本文件包含】
    GCN / GAT / GraphSAGE / GIN   —— 四个编码器（和任务一基本一样，
                                     只是输出的是 embedding 而不是类别分数）
    DotDecoder                    —— 最简单也最常用的解码器：两个向量做内积
    MLPDecoder                    —— 用一个小 MLP 算分数，表达能力更强
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

# Windows 控制台默认是 GBK 编码，改成 UTF-8 避免中文/符号打印时报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 编码器 1：GCN
# ---------------------------------------------------------------------------
class GCNEncoder(nn.Module):
    """
    多层 GCN 编码器，输出每个节点的 embedding。

    和任务一的区别只有一处：最后一层的输出维度是 out_dim（embedding 维度），
    而不是类别数。因为链路预测不需要分类，只需要一个"表示向量"。
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ---------------------------------------------------------------------------
# 编码器 2：GAT
# ---------------------------------------------------------------------------
class GATEncoder(nn.Module):
    """多层 GAT 编码器（多头注意力）。"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 dropout=0.5, heads=8):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads,
                                  concat=True, dropout=dropout))
        self.norms.append(nn.BatchNorm1d(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                                      concat=True, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim * heads))
        # 最后一层单头，输出 out_dim 维 embedding
        self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1,
                                  concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ---------------------------------------------------------------------------
# 编码器 3：GraphSAGE
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(nn.Module):
    """多层 GraphSAGE 编码器（mean 聚合）。"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 dropout=0.5, aggr="mean"):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim, aggr=aggr))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ---------------------------------------------------------------------------
# 编码器 4：GIN
# ---------------------------------------------------------------------------
class GINEncoder(nn.Module):
    """多层 GIN 编码器（sum 聚合 + MLP）。"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GINConv(self._mlp(in_dim, hidden_dim), train_eps=True))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(self._mlp(hidden_dim, hidden_dim), train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(GINConv(self._mlp(hidden_dim, out_dim), train_eps=True))

    @staticmethod
    def _mlp(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                             nn.Linear(out_dim, out_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ---------------------------------------------------------------------------
# 解码器 1：内积解码器（Dot Product）
# ---------------------------------------------------------------------------
class DotDecoder(nn.Module):
    """
    最简单的解码器：分数 = z_u · z_v（两个向量的内积）。

    直觉：如果两个节点的表示向量"方向一致、都很长"，内积就大，
          模型就认为它们之间有边。

    对应的就是经典的 "点积模型"，类似矩阵分解。
    参数量为 0（没有可学习参数）。
    """

    def forward(self, z, edge_label_index):
        """
        参数：
            z               : (N, D) 所有节点的 embedding
            edge_label_index: (2, E) 要打分的 E 条边（正样本或负样本）
        返回：
            (E,) 每条边的分数
        """
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


# ---------------------------------------------------------------------------
# 解码器 2：MLP 解码器
# ---------------------------------------------------------------------------
class MLPDecoder(nn.Module):
    """
    用一个小 MLP 来打分：score = MLP([z_u ; z_v])   （; 表示拼接）

    比内积更灵活：不仅能表达"越相似越可能有边"，
    还能学到别的规律（比如某类节点之间总是有条边）。

    代价是多了点参数，训练也更慢一点。
    """

    def __init__(self, emb_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        # 把两个节点的 embedding 按最后一维拼起来 -> (E, 2D)
        h = torch.cat([z[src], z[dst]], dim=-1)
        # 输出 (E,)，用 squeeze(-1) 把最后一维去掉
        return self.mlp(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 统一的工厂函数
# ---------------------------------------------------------------------------
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "GIN"]
DECODER_NAMES = ["dot", "mlp"]


def build_encoder(name, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
    """根据名字创建编码器。"""
    name = name.lower()
    if name == "gcn":
        return GCNEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif name == "gat":
        heads = 8 if hidden_dim >= 8 else 1
        per_head = max(hidden_dim // heads, 1)
        return GATEncoder(in_dim, per_head, out_dim, num_layers, dropout, heads=heads)
    elif name in ("graphsage", "sage"):
        return GraphSAGEEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif name == "gin":
        return GINEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    else:
        raise ValueError(f"未知模型：{name}。可选：{MODEL_NAMES}")


def build_decoder(name, emb_dim):
    """根据名字创建解码器。"""
    name = name.lower()
    if name == "dot":
        return DotDecoder()
    elif name == "mlp":
        return MLPDecoder(emb_dim)
    else:
        raise ValueError(f"未知解码器：{name}。可选：{DECODER_NAMES}")


if __name__ == "__main__":
    # 自测：随机小图，检查四个编码器 + 两个解码器能否正常前向传播
    print("=" * 78)
    print("模型自测（编码器 + 解码器 前后向传播）")
    print("=" * 78)
    torch.manual_seed(0)
    N, F_in, EMB = 20, 16, 8
    x = torch.randn(N, F_in)
    edge_index = torch.randint(0, N, (2, 60))
    edge_label_index = torch.randint(0, N, (2, 10))

    for name in MODEL_NAMES:
        for decoder_name in DECODER_NAMES:
            enc = build_encoder(name, F_in, EMB, EMB, num_layers=2)
            dec = build_decoder(decoder_name, EMB)
            z = enc(x, edge_index)
            s = dec(z, edge_label_index)
            n_param = (sum(p.numel() for p in enc.parameters())
                       + sum(p.numel() for p in dec.parameters()))
            ok = tuple(z.shape) == (N, EMB) and tuple(s.shape) == (10,)
            print(f"  {name:10s} + {decoder_name:4s}  z={tuple(z.shape)}  "
                  f"score={tuple(s.shape)}  参数量={n_param:6d}  "
                  f"{'[OK]' if ok else '[FAIL]'}")
