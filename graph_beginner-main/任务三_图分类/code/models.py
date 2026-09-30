# -*- coding: utf-8 -*-
"""
models.py —— 四种主流 GNN + 四种池化（任务三：图分类）

【图分类的模型长什么样？】

    任务一的节点分类模型是这样的：

        x (N, F) --GNN层--> z (N, hidden) --输出层--> 每个节点的类别

    图分类不一样，我们要给**整张图**一个标签，但 GNN 输出的是"每个节点"的
    表示。所以中间要多一步：把 N 个节点的表示压缩成 1 个图表示。
    这一步就叫 **池化 (Pooling)**，也常叫"读出 (readout)"。

        x (N, F) --GNN层--> z (N, hidden) --池化--> g (hidden) --分类头--> 图的类别

    所以整个模型分成三段：
        1) 卷积编码器 (encoder)：堆几层 GNN，让每个节点吸收邻居信息
        2) 池化层 (pooling)     ：把 N 个节点向量合并成 1 个向量
        3) 分类头 (classifier)  ：几层全连接，从图向量得到类别

【四种池化方法，区别在哪？】

    假设一张图有 3 个节点，某个特征维度上的值是 [1, 5, 3]：

        AvgPooling（平均）: (1+5+3)/3 = 3              —— 看"平均水平"
        MaxPooling（最大）: max(1,5,3) = 5             —— 看"最突出的那个"
        MinPooling（最小）: min(1,5,3) = 1             —— 看"最不突出的那个"
        SumPooling（求和）: 1+5+3 = 9                  —— 看"总量"

    直觉理解：
        - 平均池化：对图的**大小**不敏感（10 个节点和 100 个节点的
          "平均值"可能差不多），所以更适合"图的规模差异很大"的数据集。
        - 最大/最小池化：只留下极值，对噪声敏感但对"局部特殊结构"敏感。
          最小池化在实践里用得最少，但作业要求对比，就一起实现了。
        - 求和池化：保留了图的规模信息。GIN 的原论文理论证明用的就是 sum，
          因为只有 sum 才能区分"有两个相同子结构"和"有一个"的区别。

【一个重要的小技巧：为什么要给"图向量"过一遍 BatchNorm + 激活？】

    直接池化出来的图向量数值范围可能很不稳定（尤其是 sum 池化，
    节点越多值越大）。所以池化后先做一次归一化，让后面的分类头好训练。
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

# Windows 控制台默认是 GBK 编码，改成 UTF-8 避免中文/符号打印时报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 池化的名字和对应的实现
# ---------------------------------------------------------------------------
POOL_NAMES = ["avg", "max", "min", "sum"]

POOL_LABELS = {
    "avg": "AvgPooling（平均池化）",
    "max": "MaxPooling（最大池化）",
    "min": "MinPooling（最小池化）",
    "sum": "SumPooling（求和池化）",
}


def pool_nodes(x, batch, how):
    """
    把每个节点的向量聚合成每张图的向量。

    参数：
        x     : (N, hidden) 所有图的节点表示拼在一起（PyG 的 batch 会把多张图拼成一张大图）
        batch : (N,) 每个节点属于这一批里的第几张图
                例如 batch = [0,0,0,1,1] 表示前 3 个节点属于第 0 张图，后 2 个属于第 1 张图
        how   : 'avg' / 'max' / 'min' / 'sum'

    返回：
        (num_graphs, hidden) 每张图一个向量
    """
    if how == "avg":
        return global_mean_pool(x, batch)
    elif how == "max":
        return global_max_pool(x, batch)
    elif how == "sum":
        return global_add_pool(x, batch)
    elif how == "min":
        # PyG 没有直接提供 global_min_pool，所以这里用 scatter_reduce 自己实现。
        # 思路：把同一个图的所有节点分成一组，在每一组里取最小值。
        num_graphs = int(batch.max().item()) + 1
        out = torch.full((num_graphs, x.size(1)), float("inf"),
                         dtype=x.dtype, device=x.device)
        # scatter_reduce 的 "amin" 就是"分组取最小值"的意思
        idx = batch.view(-1, 1).expand(-1, x.size(1))
        out = out.scatter_reduce(0, idx, x, reduce="amin", include_self=True)
        return out
    else:
        raise ValueError(f"不支持的池化方式：{how}。可选：{POOL_NAMES}")


# ---------------------------------------------------------------------------
# 卷积层构造：四种模型各自提供"搭一层卷积"的函数
#
# 为什么这么写？
#   因为四个模型的编码器结构完全一样（conv -> BN -> ReLU -> Dropout），
#   唯一的区别就是中间那个 conv 是哪种。把差异抽成一个函数，
#   整个 GraphClassifier 就能写成一份代码，减少重复、也更容易对比。
# ---------------------------------------------------------------------------
def _make_conv(name, in_dim, out_dim, dropout, heads=8):
    """按名字造一层卷积。"""
    name = name.lower()
    if name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif name == "gat":
        # GAT 是"多头注意力"。这里每层实际输出 out_dim * heads 维，
        # 所以 GraphClassifier 里会把 out_dim 预先除以 heads 来补偿，
        # 保证 GAT 最终输出的维度和其他模型一样（公平对比的关键）。
        return GATConv(in_dim, out_dim, heads=heads, concat=True, dropout=dropout)
    elif name in ("graphsage", "sage"):
        return SAGEConv(in_dim, out_dim, aggr="mean")
    elif name == "gin":
        # GINConv 需要一个神经网络当参数，一般用两层 MLP
        return GINConv(
            nn.Sequential(nn.Linear(in_dim, out_dim),
                          nn.ReLU(),
                          nn.Linear(out_dim, out_dim)),
            train_eps=True)
    else:
        raise ValueError(f"未知模型：{name}")


def _conv_out_dim(name, per_head, heads=8):
    """
    某一层卷积实际输出的宽度。

    非 GAT 的模型：输出就是传进去的维度。
    GAT：每个头输出 per_head 维，拼起来是 per_head * heads 维。
    """
    if name.lower() == "gat" and heads > 1:
        return per_head * heads
    return per_head


def _safe_bn(bn, g):
    """
    安全地做 BatchNorm。

    【为什么需要这个？】
        BatchNorm1d 在训练模式下会统计这一批数据的均值和方差。
        如果这一批**只有 1 个样本**，方差就是 0，除以 0 会得到 NaN，
        训练直接崩掉。（池化后的向量正好是"每张图一个样本"，
        所以最后一个不满的 batch 如果只剩 1 张图就会踩到这个坑。）

        解决：遇到这种情况就临时切到 eval 模式，改用训练过程中
        累积下来的滑动平均统计量。这样既能算，也不会污染统计量。
    """
    # 只有 BatchNorm 才有"单样本方差为 0"的问题；
    # LayerNorm（按样本自己归一化）和 Identity 都不受影响，直接调用即可。
    if isinstance(bn, nn.BatchNorm1d) and bn.training and g.size(0) == 1:
        bn.eval()
        out = bn(g)
        bn.train()
        return out
    return bn(g)


# ---------------------------------------------------------------------------
# 主模型：编码器 + 池化 + 分类头
# ---------------------------------------------------------------------------
class GraphClassifier(nn.Module):
    """
    通用的图分类模型：GCN / GAT / GraphSAGE / GIN 四种骨干共用这一个外壳。

    参数：
        backbone   : 'GCN' / 'GAT' / 'GraphSAGE' / 'GIN'
        in_dim     : 节点特征维度
        hidden_dim : 隐藏层维度
        out_dim    : 输出维度
                     - 分类任务：类别数（配合 CrossEntropyLoss）
                     - 回归任务：1（配合 L1Loss）
        num_layers : 卷积层数（>=1）
                     注意：图分类里 3~5 层很常见（比节点分类深），
                     因为池化会把节点信息汇总，不会像节点分类那样
                     出现"深层过平滑导致所有节点一样"的问题那么严重。
        dropout    : Dropout 概率
        pooling    : 池化方式 'avg' / 'max' / 'min' / 'sum'
    """

    def __init__(self, backbone, in_dim, hidden_dim, out_dim,
                 num_layers=3, dropout=0.5, pooling="avg", heads=8,
                 pool_bn_eps=1e-5, pool_norm="bn"):
        super().__init__()
        assert num_layers >= 1, "至少要有 1 层卷积"
        self.backbone = backbone
        self.dropout = dropout
        self.pooling = pooling
        self.heads = heads if backbone.lower() == "gat" else 1

        # 为了让 GAT 的最终宽度和其他模型一致，先算出"每个头多少维"。
        # 例如 hidden_dim=128、heads=8 -> per_head=16，拼起来还是 128 维。
        per_head = max(hidden_dim // self.heads, 1)
        # 这个 w 就是编码器每一层的实际输出宽度（四个模型都一样，保证公平对比）
        w = _conv_out_dim(backbone, per_head, self.heads)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # ---- 第 1 层：in_dim -> w ----
        self.convs.append(_make_conv(backbone, in_dim, per_head, dropout, self.heads))
        self.norms.append(nn.BatchNorm1d(w))

        # ---- 中间层：w -> w ----
        for _ in range(num_layers - 1):
            self.convs.append(_make_conv(backbone, w, per_head, dropout, self.heads))
            self.norms.append(nn.BatchNorm1d(w))

        # ---- 池化后的一层归一化 + 激活，让图向量数值稳定 ----
        #
        # 【为什么 eps 是个可以调的参数？】
        #   BatchNorm 做的是 (x - 均值) / sqrt(方差 + eps)。
        #   eps 是个很小的数（默认 1e-5），作用只是防止"除以 0"。
        #
        #   但如果池化后的图向量方差**非常小**（不同图的向量长得几乎一样），
        #   那么 sqrt(方差 + 1e-5) 也很小，等于"除以一个接近 0 的数"，
        #   会把微小的波动放大几百倍 —— 训练时用批次统计量还好，
        #   推理时用累积的滑动统计量，就会导致验证损失指数级爆炸
        #   （实测 MinPooling 会涨到 148 这种离谱的值）。
        #
        #   把 eps 调大（比如 0.1）相当于给分母加一个下限，
        #   数值就稳住了。默认仍是 PyTorch 的 1e-5，不改变标准行为。
        #
        # 【2024 补充：BatchNorm 在池化后还有第二个坑（更隐蔽）】
        #   BatchNorm 训练时用"这一批的均值方差"，推理时用"累积的滑动均值方差"。
        #   但池化后的图向量分布会**随着编码器训练而漂移**（尤其是 MaxPooling：
        #   图向量的数值会越训越大）。等推理时用的还是早期累积下来的旧统计量，
        #   归一化结果就整体偏移了 —— 表现是"训练损失一直在降，
        #   验证损失却一路涨、验证准确率卡在多数类不动"，看起来像过拟合，
        #   其实是归一化统计量对不上。
        #
        #   根治办法是换用 **LayerNorm**：它对每张图**独立**归一化，
        #   不依赖任何跨样本的累积统计量，所以训练和推理的行为完全一致，
        #   根本不存在"统计量漂移"这回事。现代 GNN 做图级 readout 时
        #   基本都用 LayerNorm 就是这个原因。
        #
        #   pool_norm='bn' → BatchNorm（经典做法，任务里默认保留）
        #   pool_norm='ln' → LayerNorm（推荐，训练/推理一致）
        #   pool_norm='none' → 不做归一化（对照用）
        if pool_norm == "bn":
            self.pool_norm = nn.BatchNorm1d(w, eps=pool_bn_eps)
        elif pool_norm == "ln":
            self.pool_norm = nn.LayerNorm(w)
        elif pool_norm == "none":
            self.pool_norm = nn.Identity()
        else:
            raise ValueError(f"不支持的 pool_norm：{pool_norm}，可选 bn / ln / none")

        # ---- 分类头：两层全连接 ----
        # 图向量 (w) -> w//2 -> out_dim
        self.classifier = nn.Sequential(
            nn.Linear(w, max(w // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(w // 2, 1), out_dim),
        )

    def encode(self, x, edge_index):
        """前 N 层卷积：得到每个节点的表示（形状 (N, hidden_dim)）。"""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch):
        """
        参数：
            x          : (N, in_dim)  这一批所有图的节点特征（已由 PyG 拼好）
            edge_index : (2, E)       这一批所有图的边
            batch      : (N,)         每个节点属于第几张图

        返回：
            (num_graphs, out_dim)
        """
        # 第 1 步：卷积编码
        z = self.encode(x, edge_index)
        # 第 2 步：池化，把节点表示压成图表示
        g = pool_nodes(z, batch, self.pooling)
        # 池化后归一化 + 激活（sum 池化的数值范围大，这一步尤其重要）
        g = F.relu(_safe_bn(self.pool_norm, g))
        g = F.dropout(g, p=self.dropout, training=self.training)
        # 第 3 步：分类头
        return self.classifier(g)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "GIN"]


def build_model(name, in_dim, hidden_dim, out_dim, num_layers=3,
                dropout=0.5, pooling="avg", pool_bn_eps=1e-5, pool_norm="bn"):
    """
    根据名字创建模型。上层脚本只要写 build_model('GCN', ...) 就行。

    用法：
        model = build_model('GIN', in_dim=7, hidden_dim=128, out_dim=2,
                            num_layers=3, pooling='sum')

    pool_bn_eps : 池化后那层 BatchNorm 的 eps。默认 1e-5（PyTorch 标准值）。
                  用 MinPooling 时如果验证损失爆炸，可以调大这个值，见
                  GraphClassifier.__init__ 里的说明。
    """
    return GraphClassifier(name, in_dim, hidden_dim, out_dim,
                           num_layers=num_layers, dropout=dropout,
                           pooling=pooling, pool_bn_eps=pool_bn_eps,
                           pool_norm=pool_norm)


if __name__ == "__main__":
    # 自测：造两张小图拼成一批，看看 4 个模型 × 4 种池化能不能正常前向传播
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    print("=" * 78)
    print("模型自测（4 个模型 × 4 种池化 × 2 种层数）")
    print("=" * 78)
    torch.manual_seed(0)

    F_in, C = 7, 2
    graphs = []
    for i in range(6):
        n = 8 + i * 2
        graphs.append(Data(x=torch.randn(n, F_in),
                           edge_index=torch.randint(0, n, (2, n * 2)),
                           y=torch.tensor([i % C])))
    loader = DataLoader(graphs, batch_size=3)

    n_ok, n_fail = 0, 0
    for name in MODEL_NAMES:
        for pooling in POOL_NAMES:
            for layers in [1, 3]:
                model = build_model(name, F_in, 32, C,
                                    num_layers=layers, pooling=pooling)
                batch = next(iter(loader))
                out = model(batch.x, batch.edge_index, batch.batch)
                ok = tuple(out.shape) == (3, C)
                n_ok += ok
                n_fail += (not ok)
                status = "[OK]" if ok else "[FAIL]"
                n_param = sum(p.numel() for p in model.parameters())
                print(f"  {name:10s} pooling={pooling:4s} layers={layers}  "
                      f"输出={tuple(out.shape)}  参数量={n_param:7d}  {status}")

    # 单独测一下回归输出（out_dim=1）
    print("\n回归任务自测（out_dim=1）：")
    model = build_model("GCN", F_in, 32, 1, num_layers=3, pooling="sum")
    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.batch)
    print(f"  输出形状={tuple(out.shape)}（应该是 (3, 1)）"
          f"  {'[OK]' if tuple(out.shape) == (3, 1) else '[FAIL]'}")

    print(f"\n合计：{n_ok} 个通过，{n_fail} 个失败")
