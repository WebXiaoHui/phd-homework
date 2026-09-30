# 任务二、图上的链路预测（GCN / GAT / GraphSAGE / GIN）

> 框架：PyTorch Geometric 2.0.4 + PyTorch 2.3.1
> 数据集：Cora、Citeseer、Flickr

---

## 一、这个任务在做什么？

**链路预测**：给一张图，把一部分边藏起来，让模型猜"这两个节点之间本来有没有边"。

```
输入：一张残缺的图（一部分边被藏起来了）  →  输出：任意两个节点之间有边的概率
```

本质是一个**二分类**问题：

- **正样本**：真实存在的边
- **负样本**：随机抽出来的、图里不存在的"假边"

**预测"未来"**：社交网络推荐"你可能认识的人"、电商推荐"你可能想买的搭配"，
用的都是这个技术。

---

## 二、目录结构

```
任务二_链路预测/
├── data/                       # 数据集（已下载好，无需再下）
│   ├── Cora/{raw,processed}/
│   ├── Citeseer/{raw,processed}/
│   └── Flickr/{raw,processed}/
├── code/
│   ├── prepare_data.py         # 下载/检查数据集
│   ├── datasets.py             # ★ 无泄漏的边划分 + 负采样 + 评估指标
│   ├── models.py               # 4 个编码器 + 2 个解码器
│   ├── train.py                # ★ 训练+测试脚本（单次实验）
│   ├── run_all.py              # 一键跑完所有对比实验
│   ├── plot_results.py         # 画图 + 打印汇总表
│   └── utils.py                # 日志、随机种子、结果保存等公共工具
├── logs/                       # 每次实验的日志
├── results/                    # results.jsonl（结果）+ *.png（图）
└── README.md
```

---

## 三、训练和测试的脚本（作业要求必写）

| 作用 | 命令 |
|---|---|
| **训练 + 测试（单次实验）** | `python code/train.py --dataset Cora --model GCN --mode full` |
| **采样训练（子图）** | `python code/train.py --dataset Flickr --model GAT --mode sample --batch_size 4096` |
| **批量跑所有实验** | `python code/run_all.py --stage all` |
| **生成图表和汇总表** | `python code/plot_results.py` |
| 检查数据集是否就绪 | `python code/prepare_data.py` |

**注意：本任务没有单独的 `test.py`。** 训练脚本 `train.py` 一条龙完成
"训练 → 每轮在验证集上评估 AUC → 挑最好的轮次 → 用那个轮次在测试集上评估 →
把指标写进 `results/results.jsonl`"，所以测试就是训练脚本的最后一步。

### 单次实验的常用参数

```bash
python code/train.py \
    --dataset Cora \        # Cora | Citeseer | Flickr
    --model   GCN \         # GCN | GAT | GraphSAGE | GIN
    --mode    full \        # full(全图) | sample(子图采样)
    --decoder dot \         # dot(内积) | mlp(多层感知机)
    --emb_dim 64 \          # 节点 embedding 维度
    --batch_size 2048 \     # 采样模式下每批多少个种子节点
    --num_neighbors 10 \    # 每层每个节点采几个邻居
    --neg_ratio 1 \         # 每条正边配几条负边
    --val_ratio 0.05 \      # 验证边比例
    --test_ratio 0.10 \     # 测试边比例
    --layers 2 \            # 消息传递层数
    --hidden 128 \          # 隐藏层维度
    --lr 0.01 \             # 学习率
    --epochs 100 \          # 训练轮数
```

---

## 四、数据集和「边的划分」（作业要求了解的知识点）

数据集和任务一完全一样：

| 数据集 | 节点数 | 边数 | 特征维度 | 特点 |
|---|---|---|---|---|
| **Cora** | 2,708 | 10,556 | 1,433 | 论文引用网络 |
| **Citeseer** | 3,327 | 9,104 | 3,703 | 特征更稀疏、边更少 |
| **Flickr** | 89,250 | 899,756 | 500 | 规模是 Cora 的 33 倍 |

### 1. 链路预测的划分比节点分类麻烦在哪？

节点分类里"样本"是节点，切分很直接（Planetoid 自带官方划分）。
链路预测里"样本"是**边**，切分要小心一个陷阱：**信息泄漏**。

> **信息泄漏（data leakage）**：如果测试用的边在训练时也参与了消息传递，
> 模型等于提前看到了答案，指标会虚高到不可信。
> 这是链路预测里最容易犯、后果最严重的错误。

### 2. 本代码的「无泄漏」划分流程

```
第 1 步：图变成无向图（补上反向边），去掉自环
第 2 步：只保留 u < v 的边，得到"唯一无向边集合"（每条边只算一次）
        ← 这一步很关键！
第 3 步：随机打乱，按 85% / 5% / 10% 切成 训练边 / 验证边 / 测试边
第 4 步：消息传递**只用训练边**构成的图
        验证/测试时也用同一张图（还只用训练边），只是换一批边来打分
```

**第 2 步为什么关键？** 如果不做去重，同一条边可能以 `(u,v)` 和 `(v,u)`
两种形式分别进入训练集和测试集 —— 相当于把答案抄给了模型，
AUC 会飙到 0.99，但模型其实什么也没学到。

**第 4 步为什么关键？** 编码器（GCN 等）在算节点 embedding 时，
是靠"沿着边传递消息"来聚合邻居信息的。如果测试边也参与传递，
那节点 embedding 里就"混进"了测试边的信息，评估就不公平了。
所以测试边的信息**只用于最后打分，不参与消息传递**。

### 3. 负样本怎么造？

```
随机抽一对节点 (i, j)
如果是同一节点 → 丢掉
如果图里真的有边 i-j → 丢掉（这是"假负样本"，会把模型教坏）
否则 → 当成负样本
```

- 训练时每条正边配 `--neg_ratio` 条负边（默认 1 条），比例 1:1。
- 评估时每条真边配 **100 条**负边，更严格（见下面 Hits@50 的说明）。

---

## 五、模型结构：编码器-解码器（Encoder-Decoder）

链路预测的标准范式是两段式：

```
                    ┌── 编码器（Encoder）──┐   ┌── 解码器（Decoder）──┐
节点特征 + 边 ──►   │  GCN/GAT/SAGE/GIN    │──►│  dot 或 mlp          │──►  这条边的分数
                    │  算出每个节点的向量 z │   │  用两个向量算分数      │
                    └──────────────────────┘   └──────────────────────┘
```

**为什么要分两段？** 因为编码器只负责"把节点变成向量"，
解码器只负责"用两个向量判断有没有边"。想换编码器或换解码器，
另一边完全不用改。

### 编码器（4 个，和任务一完全一样）

| 模型 | 核心思想 | 代码 |
|---|---|---|
| **GCN** | 邻居特征加权平均（权重按度数固定） | `GCNConv` |
| **GAT** | 用注意力**学出**每个邻居的权重（8 头） | `GATConv(heads=8)` |
| **GraphSAGE** | 自己 + 邻居均值，拼起来再变换 | `SAGEConv(aggr="mean")` |
| **GIN** | 邻居求和 + MLP，理论表达能力最强 | `GINConv` |

> 原理详解见任务一的 README。**注意**：编码器最后一层**不做 softmax**，
> 输出的是 `emb_dim` 维的向量（不是类别概率），因为这里要的是"表示",
> 不是"分类结果"。

### 解码器（2 个）

| 解码器 | 公式 | 直觉 |
|---|---|---|
| **DotDecoder**（默认） | `score = z_i · z_j`（向量内积） | 两个向量越"同向"，分数越高。简单、快、参数为零 |
| **MLPDecoder** | `score = MLP([z_i \|\| z_j])` | 把两个向量拼起来过一个小 MLP，能学更复杂的关系 |

**怎么选？** 内积解码器是链路预测的经典做法，
在**同配图**（同类节点倾向相连）上效果很好且不增加参数；
MLP 表达能力强，但参数多、在稀疏图上容易过拟合。

---

## 六、全图训练 vs 子图采样训练

对应作业要求第 4 条。和任务一一样用 `--mode` 切换，但**链路预测有个特殊之处**：

### 1. 采样训练怎么定义"一批边"？

节点分类里"样本"就是节点，采样很直接。链路预测里"样本"是**边**，
要先想清楚"这一批训练哪些边"。

本代码的做法：

```
1. 用训练边的两个端点作为"种子节点"去采样
2. NeighborLoader 返回的子图里，排在前面的是这批种子节点
3. 子图中"两端都在种子集合里"的边，就是这一批要训练的正样本边
```

这样做的**好处**是逻辑自洽：正样本边的两个端点一定在子图里，
一定能拿到自己的 embedding，不会出现"要打分但节点不在"的情况。

### 2. 两种模式对比

| | `--mode full`（全图） | `--mode sample`（子图采样） |
|---|---|---|
| 一批内容 | 整张图编码一次，给所有训练边打分 | 一小批种子节点构成的子图 |
| 显存占用 | 随图规模线性增长 | 和总图规模无关 |
| 一批的边数 | 全部训练边 | 子图内部、两端都在种子里的边 |
| 每轮耗时 | 慢 | 快 |
| 能不能扩展到超大图 | 不能 | 能 |

### 3. 评估为什么要分块（`score_edges_chunked`）？

Flickr 的测试集有 4.5 万条真边 + **450 万条**负边。
如果一次性算完，光是取出 `z[src]` 这个 `(450万, 64)` 的张量就要 1GB 显存，
显卡直接爆。

所以代码里把边**分块打分**（每块 30 万条），算完一块拼接一块，
显存恒定、不随测试集规模增长。

---

## 七、评估指标

### AUC（Area Under the ROC Curve）

```
含义：随机取一条真边、一条假边，模型给真边打分更高的概率
范围：0.5 = 瞎猜，1.0 = 完美
```

计算方式用的是 **Mann-Whitney U 统计量**（而不是画 ROC 曲线再算面积），
简单说就是"所有（真边, 假边）配对里，真边分数更大的比例"，
不用调阈值、不用 sklearn。

### Hits@50

```
含义：每条真边和 100 条假边混在一起排名，真边排进前 50 名的比例
范围：随机猜大约是 50 / 101 ≈ 0.495
```

**注意**：Hits@50 在随机情况下**约等于 0.5**，不是 0。
所以看到 0.6 就以为学得不错是错的 —— 要**明显高于 0.5** 才算学到了东西。
（这和任务四的 Hits@10 不一样，那边是从"所有实体"里排名，
随机猜的值几乎是 0。）

### 为什么两个指标都要看？

- AUC 看的是**整体排序质量**，对每条边的排名位置都敏感。
- Hits@50 看的是**头部质量**，只关心"有没有排进前 50"。
- 实际推荐系统里，用户只看前几个结果，所以 Hits@K 更贴近业务；
  但 AUC 更稳定、更适合做模型之间的横向比较。

---

## 八、对比实验的设计

`run_all.py` 里有三组实验，一共 **38 次**：

| stage | 内容 | 次数 |
|---|---|---|
| `main` | 4 模型 × 3 数据集 × 2 种模式 | 24 |
| `lr` | Cora、Flickr 上 GCN 全图，lr ∈ {0.001, 0.005, 0.01, 0.05} | 8 |
| `layers` | Cora、Flickr 上 GCN 全图，layers ∈ {2, 3, 4} | 6 |

```bash
python code/run_all.py --stage all --epochs 100   # 全部跑
python code/run_all.py --stage main --dry_run     # 先看看会跑哪些命令
```

**评估频率**：Cora/Citeseer 每轮都评估（图小，评估快）；
Flickr 每 5 轮评估一次（要算几百万条负样本的分数，很费时间），
但第 1 轮和最后一轮一定会评估。

---

## 九、结果文件说明

| 路径 | 内容 |
|---|---|
| `results/results.jsonl` | 每个实验一行 JSON，含所有超参数和测试指标 |
| `results/fig1_model_comparison.png` | 四个模型 × 三个数据集的 AUC / Hits@50 对比 |
| `results/fig2_full_vs_sample.png` | 全图 vs 采样（AUC / 每轮耗时 / 总耗时） |
| `results/fig3_layers_effect.png` | 网络层数的影响 |
| `results/fig4_lr_effect.png` | 学习率的影响 |
| `logs/*.log` | 完整训练日志 |

`results.jsonl` 的关键字段：

```
dataset, model, mode, decoder, emb_dim, layers, hidden, dropout, lr,
weight_decay, batch_size, num_neighbors, neg_ratio, epochs, best_epoch,
num_params, steps_per_epoch, best_val_auc, best_test_auc,
best_test_hits50, train_time_total, train_time_per_epoch, eval_time_total,
wall_time, seed, tag
```

---

## 十、参考

- **GCN**: Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
- **GAT**: Veličković et al. *Graph Attention Networks.* ICLR 2018.
- **GraphSAGE**: Hamilton et al. *Inductive Representation Learning on Large Graphs.* NIPS 2017.
- **GIN**: Xu et al. *How Powerful are Graph Neural Networks?* ICLR 2019.
- **链路预测综述**: Zhang & Chen. *Link Prediction Based on Graph Neural Networks.* NIPS 2018.
- **子图采样**: <https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html>
