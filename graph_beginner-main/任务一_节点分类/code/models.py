# -*- coding: utf-8 -*-
"""
models.py —— 四种主流 GNN 模型（任务一：节点分类）

本文件实现四个模型：GCN、GAT、GraphSAGE、GIN。
每个模型都遵循同一套接口，方便公平对比：

    输入：x (N, in_dim), edge_index (2, E)
    输出：out (N, num_classes)      # 每个节点的类别 logits

【四个模型的核心思想（一句话版）】
    GCN        ：邻居特征做"加权平均"（权重由节点度决定），再乘一个可学习矩阵。
                 公式 H' = Â X W，其中 Â 是归一化后的邻接矩阵。
    GAT        ：邻居的权重不是固定的，而是"学出来"的注意力分数。
                 可以理解为：谁和我更像，我就多听谁的。
    GraphSAGE  ：把"自己"和"邻居聚合结果"拼起来（或相加），
                 可以灵活选择 mean / max / sum 聚合方式。
    GIN        ：用"求和"聚合 + 一个多层感知机(MLP)，理论表达能力最强
                 （和 WL 图同构测试一样强）。

【为什么它们长得这么像？】
    这是 GNN 的通用范式 —— "消息传递(message passing)"：
        每一层做两件事：
          1) 聚合(aggregate)：把邻居的信息收集起来
          2) 更新(update)   ：和自身信息结合，得到新的表示
    不同模型的区别只在于"怎么聚合"和"怎么更新"。
    所以下面的代码结构完全一样，只是把卷积层换掉了。

【关于 BatchNorm】
    四个模型都统一加了 BatchNorm1d + ReLU + Dropout。
    统一结构是为了让对比实验"只变了模型，其他都一样"，这样结论才可靠。
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
# 1. GCN —— Graph Convolutional Network (Kipf & Welling, 2017)
# ---------------------------------------------------------------------------
class GCN(nn.Module):
    """
    多层 GCN。每一层就是一个 GCNConv，后面接 BatchNorm / ReLU / Dropout。

    参数：
        in_dim     : 输入特征维度
        hidden_dim : 隐藏层维度
        out_dim    : 输出类别数
        num_layers : 总共几层（>=2）
        dropout    : dropout 概率
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2, "至少要有 2 层（一层隐藏层 + 一层输出层）"
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 第一层：in_dim -> hidden_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 中间的隐藏层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层：hidden_dim -> out_dim（不加 BN/ReLU，直接输出 logits）
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        # 逐个隐藏层：卷积 -> BN -> ReLU -> Dropout
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 输出层
        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# 2. GAT —— Graph Attention Network (Veličković et al., 2018)
# ---------------------------------------------------------------------------
class GAT(nn.Module):
    """
    多层 GAT。GATConv 支持"多头注意力"：heads 个头各自算一遍，再把结果拼起来。

    维度上的坑（新手最容易搞错的地方）：
        当 heads>1 且 concat=True 时，输出维度 = 每个头的维度 × 头数。
        所以设 hidden_dim=8、heads=8 时，这一层实际输出 8*8=64 维。
        最后一层我们让 heads=1 且 concat=False，输出维度就等于 out_dim。
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 dropout=0.5, heads=8):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.heads = heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 第一层：输出 hidden_dim * heads 维
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads,
                                  concat=True, dropout=dropout))
        self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # 中间层：输入 hidden_dim*heads，输出还是 hidden_dim*heads
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                                      concat=True, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # 最后一层：单头、不拼接，输出 out_dim
        self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1,
                                  concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# 3. GraphSAGE —— SAmple and aggreGatE (Hamilton et al., 2017)
# ---------------------------------------------------------------------------
class GraphSAGE(nn.Module):
    """
    多层 GraphSAGE。用 PyG 的 SAGEConv 实现。

    SAGEConv 的核心操作（默认 aggr='mean'）：
        h_new = W1 * h_self + W2 * mean(h_neighbors)
    也就是"自己"和"邻居平均值"各乘一个矩阵后相加。

    aggr 可以换成 'mean' / 'max' / 'sum' / 'add'，本实现默认用 mean。
    """

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
        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# 4. GIN —— Graph Isomorphism Network (Xu et al., 2019)
# ---------------------------------------------------------------------------
class GIN(nn.Module):
    """
    多层 GIN。PyG 的 GINConv 需要传入一个"神经网络"作为参数（通常是个 MLP）。

    GINConv 做的事情：
        h_new = MLP( (1 + eps) * h_self + sum(h_neighbors) )
    注意这里是 **sum** 而不是 mean。作者从理论上证明了：
    只有 sum 聚合才能区分不同的多重集合，表达能力才和 WL 测试一样强。

    train_eps=True 表示 eps 这个系数也是可学习的。
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # GINConv 的输出维度 = 传入 MLP 的输出维度
        self.convs.append(GINConv(self._mlp(in_dim, hidden_dim), train_eps=True))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GINConv(self._mlp(hidden_dim, hidden_dim), train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(GINConv(self._mlp(hidden_dim, out_dim), train_eps=True))

    @staticmethod
    def _mlp(in_dim, out_dim):
        """构造一个两层 MLP：Linear -> ReLU -> Linear。"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# 5. 统一的工厂函数
# ---------------------------------------------------------------------------
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "GIN"]


def build_model(name, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
    """
    根据名字创建模型。这样上层脚本只要写 build_model('GCN', ...) 就行了，
    增加新模型时也只需要改这一个地方。

    用法：
        model = build_model('GAT', in_dim=1433, hidden_dim=8, out_dim=7, num_layers=2)
    """
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif name == "gat":
        # GAT 每个头用 hidden_dim=8、8 个头，这样总维度是 64，和别的模型量级接近。
        # 如果 hidden_dim 已经很小(<=8)，就只用一个头，避免维度太小。
        heads = 8 if hidden_dim >= 8 else 1
        per_head = max(hidden_dim // heads, 1)
        return GAT(in_dim, per_head, out_dim, num_layers, dropout, heads=heads)
    elif name in ("graphsage", "sage"):
        return GraphSAGE(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif name == "gin":
        return GIN(in_dim, hidden_dim, out_dim, num_layers, dropout)
    else:
        raise ValueError(f"未知模型：{name}。可选：{MODEL_NAMES}")


if __name__ == "__main__":
    # 自测：随便造一个小图，看看四个模型能不能正常前向传播
    print("=" * 78)
    print("模型自测（用一个随机小图跑一遍前向传播）")
    print("=" * 78)
    torch.manual_seed(0)
    N, F_in, C = 20, 16, 3
    x = torch.randn(N, F_in)
    edge_index = torch.randint(0, N, (2, 60))

    for name in MODEL_NAMES:
        for layers in [2, 3]:
            model = build_model(name, F_in, 8, C, num_layers=layers)
            out = model(x, edge_index)
            n_param = sum(p.numel() for p in model.parameters())
            ok = tuple(out.shape) == (N, C)
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {name:10s} layers={layers}  输出={tuple(out.shape)}  "
                  f"参数量={n_param:6d}  {status}")
