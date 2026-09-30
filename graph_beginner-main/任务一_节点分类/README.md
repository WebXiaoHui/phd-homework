# 任务一、节点分类（GCN / GAT / GraphSAGE / GIN）

> 框架：PyTorch Geometric 2.0.4 + PyTorch 2.3.1
> 数据集：Cora、Citeseer、Flickr

---

## 一、这个任务在做什么？

**节点分类**：给图上的一部分节点（比如 10%）的标签，让模型预测剩下节点的标签。

```
输入：一张图（节点 + 边）+ 少量已知标签   →   输出：每个节点的类别
```

类比：社交网络里你知道一小部分人的兴趣标签，猜其他人喜欢什么。

**为什么要用"子图采样训练"？**

全图训练一次要把**整张图**的邻接矩阵和所有节点特征放进显存。
Cora 只有 2708 个节点，随便跑；但 Flickr 有 8.9 万个节点、500 维特征，
真实的工业级图可能有几十亿节点 —— 全图训练根本放不下。

解决办法是**子图采样**：每个 batch 只取"一小批种子节点 + 它们的多跳邻居"，
组成一个小图来训练。这样显存占用就和图的总规模**解耦**了。

**这也是本任务第 4 条要求的内容**：
"利用框架自带的 Sampler 采样子图进行训练，并与全图训练进行性能和运行时间的对比"。

---

## 二、目录结构

```
任务一_节点分类/
├── data/                       # 数据集（已下载好，无需再下）
│   ├── Cora/{raw,processed}/
│   ├── Citeseer/{raw,processed}/
│   └── Flickr/{raw,processed}/
├── code/
│   ├── prepare_data.py         # 下载/检查数据集
│   ├── datasets.py             # 读 Planetoid / Flickr，打印统计信息
│   ├── models.py               # 4 个模型
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
"训练 → 每轮在验证集上评估 → 挑最好的轮次 → 用那个轮次在测试集上评估 →
把指标写进 `results/results.jsonl`"，所以测试就是训练脚本的最后一步。
想要单独跑测试，可以直接用这个脚本最后打印 + 落盘的 `best_test_acc` 字段。

### 单次实验的常用参数

```bash
python code/train.py \
    --dataset Cora \        # Cora | Citeseer | Flickr
    --model   GCN \         # GCN | GAT | GraphSAGE | GIN
    --mode    full \        # full(全图) | sample(子图采样)
    --batch_size 1024 \     # 采样模式下每批多少个种子节点
    --num_neighbors 10 \    # 每层每个节点采几个邻居（采样模式）
    --layers  2 \           # 消息传递层数
    --hidden  128 \         # 隐藏层维度
    --lr      0.01 \        # 学习率
    --dropout 0.5 \         # Dropout 概率
    --epochs  200 \         # 训练轮数
    --eval_every 1 \        # 每几轮评估一次
    --seed    42            # 随机种子
```

加 `--dry_run` 只打印配置不训练。加 `--quiet` 少打日志（`run_all.py` 会用）。

---

## 四、数据集

| 数据集 | 节点数 | 边数 | 特征维度 | 类别数 | 划分方式 | 特点 |
|---|---|---|---|---|---|---|
| **Cora** | 2,708 | 10,556 | 1,433 | 7 | 官方标准划分（140 训练 / 500 验证 / 1000 测试） | 论文引用网络，节点是论文、边是引用，特征是一篇论文的词袋 |
| **Citeseer** | 3,327 | 9,104 | 3,703 | 6 | 官方标准划分 | 和 Cora 类似，但**特征更稀疏、边更少**，所以准确率明显更低 |
| **Flickr** | 89,250 | 899,756 | 500 | 7 | **要自己随机划分** | 图片分享网站的用户关系图，规模是 Cora 的 **33 倍**，是真正需要采样训练的数据集 |

### 关于 Cora / Citeseer 的划分

Planetoid 数据集（Cora / Citeseer）PyG 自带官方划分：
每个类别取 20 个节点做训练、500 个做验证、1000 个做测试。
**所有论文都用这个划分**，所以结果可以直接和论文比。

### 关于 Flickr 的划分

Flickr 原本是个**多标签分类**数据集，PyG 的 `Flickr` 类把它处理成了
7 类的单标签任务，但**没有给标准划分**。所以代码自己按
**每类分层**的方式随机切成 训练 : 验证 : 测试。
`seed=42` 固定，保证可复现。

### 三个数据集的难度对比

```
Cora      (2708 点，边密，特征好)      →  GCN 约 81%     ← 简单
Citeseer  (3327 点，边稀，特征稀疏)    →  GCN 约 71%     ← 中等
Flickr    (89250 点，89 万条边)        →  GCN 约 50%     ← 难
```

Flickr 只有 50% 左右，不是代码写错了 —— 这个数据集本身标签噪声就大，
7 分类随机猜是 14%，50% 已经是正常水平。

---

## 五、四个模型是怎么实现的？

所有模型的骨架都是 **几层消息传递 + 一层分类头**，只有"消息传递层"不同。

### 1. GCN —— 最基础的图卷积

```
h_i' = W · Σ_j  (1 / sqrt(d_i · d_j)) · h_j
```

**直觉**：把邻居的特征做**加权平均**（度数大的邻居权重小，避免大节点主导），
再过一层可学习的线性变换 `W`。

- 优点：简单、快、参数少，小数据集上就很能打。
- 缺点：权重 `1/sqrt(d_i d_j)` 是**按图结构固定算好的**，不是学出来的，
  没法区分"哪个邻居更重要"。

代码：`GCNConv(hidden, hidden)`

### 2. GAT —— 带注意力机制的图卷积

```
e_ij = LeakyReLU( a^T [W h_i || W h_j] )      # 算 i 对邻居 j 的注意力分数
α_ij = softmax_j(e_ij)                        # 在邻居上归一化
h_i' = Σ_j α_ij · W h_j                       # 按注意力加权求和
```

**直觉**：邻居的权重不写死，而是**让模型自己学**"该听谁的"。
好比投票时有人说话更有分量。

- **多头注意力**：本实现用 8 个头（`heads=8`），每个头独立学一套注意力，
  最后拼起来。相当于 8 个人从不同角度看同一件事。
- 优点：表达能力强，通常在这几个数据集上表现最好。
- 缺点：参数多、慢；大数据集上显存吃紧（Flickr 上采样训练时才勉强跑得动）。

代码：`GATConv(..., heads=8, concat=True)`

### 3. GraphSAGE —— 聚合邻居

```
h_i' = W · [ h_i || MEAN( { h_j : j ∈ 邻居(i) } ) ]
```

**直觉**：把"我自己"和"邻居的平均"**拼起来**再变换，既保留自身信息，
又吸收邻居信息。

- 设计目的是**归纳式学习**：它学的是"怎么聚合邻居"这个**函数**，
  而不是"每个节点的具体向量"，所以能直接用到训练时没见过的新节点上
  （工业界的图每天都在加新用户，这一点很关键）。
- 本实现用均值聚合（`aggr="mean"`）。原论文还试过 LSTM、Max Pooling，
  实验发现**均值最简单也最好用**。
- 优点：对新节点泛化好，实现简单。
- 缺点：均值把所有邻居一视同仁。

代码：`SAGEConv(hidden, hidden, aggr="mean")`

### 4. GIN —— 理论上表达能力最强

```
h_i' = MLP( (1 + ε) · h_i + Σ_j h_j )
```

**直觉**：邻居特征**求和**（注意：求和，不是平均！），自己乘 `1+ε`（ε 可学习），
再过一个小 MLP。

- 理论上：**求和聚合 + MLP + 足够多层**，GNN 的区分能力可以达到
  "WL 图同构测试"的上界。而均值/最大值聚合做不到。
- 为什么必须求和？平均会抹掉"邻居个数"这个信息，求和能保留。
- 缺点：求和使数值随度数增长，层数一多容易梯度爆炸/过平滑。
- 注意：**GIN 是为图分类设计的**，用在节点分类上通常不如 GCN/GAT 稳。

代码：`GINConv(2层 MLP, train_eps=True)`

---

## 六、全图训练 vs 子图采样训练（本任务的重点）

对应作业要求第 4 条。

| | `--mode full`（全图） | `--mode sample`（子图采样） |
|---|---|---|
| 一个 epoch 几次参数更新 | **1 次** | 种子节点数 / batch_size 次 |
| 一个 batch 的内容 | 整张图 | 一小批种子节点 + 它们的多跳邻居 |
| 显存占用 | 随图规模**线性增长** | **和总图规模无关**（只和 batch 大小有关） |
| 每轮耗时 | 慢（整张图前向+反向） | 快（只算子图） |
| 梯度质量 | 全量梯度，方向最准 | 有采样噪声，但一次 epoch 走很多步 |
| 能不能扩展到超大图 | **不能** | 能 |
| 邻居信息 | 完整 | **有采样偏差**（每层只采 10 个邻居，采不全） |

### 1. 采样是怎么做的？—— `NeighborLoader`

用的是 PyG 自带的 `NeighborLoader`（这就是作业说的"框架自带的 Sampler"）：

```python
loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],   # 第1层采10个邻居，第2层再采10个
    batch_size=1024,          # 每批1024个种子节点
    input_nodes=data.train_mask,
    shuffle=True,
)
```

工作过程：

```
1. 从训练节点里随机抽 10 个当种子节点
2. 每个种子节点，从它的邻居里随机采 10 个
3. 每个被采到的邻居，再采它的 10 个邻居（层数=2，所以采两跳）
4. 把这些节点和它们之间的边拼成一张小图，送进模型
5. 只在"种子节点"上算损失（采来的邻居只是为了提供信息，不参与 loss）
```

- `num_neighbors=[10, 10]` 是**每层每个节点采几个邻居**。
  层数越多，子图会指数级膨胀（10 → 100 → 1000），所以要限制。
- 只对种子节点算 loss 很重要：不然同一个节点会被算很多次，
  损失函数会被高频节点主导。

### 2. 两者怎么公平对比？

**问题**：全图模式下 1 个 epoch 只更新 1 次参数；
采样模式下如果 batch_size 比训练节点数还大（比如 batch=1024，
而 Cora 只有 140 个训练节点），1 个 epoch 也只有 1 次更新 ——
那采样就纯粹只带来了"邻居信息不全"的坏处，一点好处都没有。

所以代码里：
- Cora / Citeseer 用 `batch_size=128`（Cora 140 个训练节点 → 每轮 2 次更新；
  Citeseer 120 个 → 每轮 1 次）；
- Flickr 用 `batch_size=4096`（8.9 万训练节点 → 每轮 11 次更新；
  batch 再小的话采样开销会让 epoch 长得没法接受）。

并且结果里记录了 `steps_per_epoch`（每个 epoch 几次参数更新），
分析时**必须把它和耗时一起看**，不能只比"每轮耗时" ——
否则会得出错误的结论（见下面第 3 节的实测数据）。

### 3. 实测结果：和"教科书结论"不一样的地方

跑完之后实测的数据是这样的（`--epochs 150`，RTX 3070 Ti）：

| 数据集 | 模型 | 全图 每轮耗时 | 全图 更新次数/轮 | 全图 测试acc | 采样 每轮耗时 | 采样 更新次数/轮 | 采样 测试acc |
|---|---|---|---|---|---|---|---|
| Cora | GCN | 0.003s | 1 | 77.9% | 0.008s | 2 | **80.3%** |
| Cora | GAT | 0.009s | 1 | 77.0% | 0.016s | 2 | **79.8%** |
| Cora | GraphSAGE | 0.003s | 1 | 78.4% | 0.013s | 2 | **81.8%** |
| Cora | GIN | 0.005s | 1 | 75.5% | 0.014s | 2 | 76.4% |
| Citeseer | GCN | 0.006s | 1 | **67.4%** | 0.009s | 1 | 64.9% |
| Flickr | GCN | **0.015s** | 1 | **52.3%** | **2.03s** | 11 | 50.4% |
| Flickr | GAT | **0.029s** | 1 | 52.4% | **5.01s** | 11 | 51.7% |
| Flickr | GraphSAGE | **0.089s** | 1 | 51.8% | **6.86s** | 11 | 52.4% |

**结论一：在本次实验的规模下，采样训练反而比全图训练"每轮慢很多"**
（Flickr 上慢了 **100~200 倍**），这和"采样训练更快"的直觉相反。

**为什么会这样？** 因为这两者的瓶颈完全不同：

- **全图训练**的瓶颈是 **GPU 计算**。但这里模型很小（2 层、128 维），
  Flickr 的稀疏矩阵乘法在 GPU 上只要 0.015 秒 —— GPU 根本没吃饱。
- **采样训练**的瓶颈是 **CPU 构图**。PyG 的 `NeighborLoader` 是在 **CPU 上**
  做采样、拼子图的：每批 4096 个种子节点，两跳采样后子图有约 4.5 万个节点，
  光是"把子图组装出来"就要花 2~7 秒，GPU 只能在旁边等着。

所以更准确的说法是：

> **子图采样省的是"显存"，不是"时间"。**
> 它带来的是额外的 CPU 采样开销。
> 只有当整张图**大到放不进显存**（或大到全图计算的耗时超过采样开销）时，
> 采样训练才有时间上的优势。

**结论二：采样训练在小数据集上不一定更差，有时反而更好。**
Cora 上采样训练的准确率**比全图高 2~3 个点**（GCN 77.9% → 80.3%）。
两个原因：

1. **采样相当于一种正则化**：每轮只看局部的子图，模型不容易记住全局结构，
   小数据集上反而缓解了过拟合；
2. **更新次数更多**：Cora 训练节点只有 140 个，但 batch_size=128，
   每个 epoch 有 2 次参数更新（全图只有 1 次）。

反过来，Citeseer 上采样就更差（GCN 67.4% → 64.9%），因为那边
batch_size=128 > 训练节点数，每个 epoch 只有 **1 次**更新，采样只带来了
邻居信息的缺失，没有得到"多更新几次"的好处。

**结论三：分析"性能 vs 时间"时必须两个维度一起看。**
只看"每轮耗时"会得出"全图更快"的结论，但全图每个 epoch 只更新 **1 次**
参数，而 Flickr 采样每个 epoch 更新 **11 次**。要比较公平，应该看
"**达到某个准确率需要多少墙钟时间**"。`results.jsonl` 里的
`steps_per_epoch` 字段就是为这个分析准备的。

**那什么时候非用采样不可？** 当图大到全图放不进显存时。
Flickr 的 8.9 万节点 × 500 维特征在这里只要几百 MB，全图完全跑得动；
但如果节点数是 **1000 万**，全图模式会直接 OOM ——
那时采样训练就是唯一的选择，慢一点也比跑不起来强。
**这才是"采样训练"真正的价值所在。**

---

## 七、对比实验的设计

`run_all.py` 里有三组实验：

### `--stage main` 主实验（24 次）

```
4 个模型 × 3 个数据集 × 2 种训练方式 = 24
```

同时回答"不同神经网络的影响"和作业要求第 4 条的"全图 vs 采样对比"。

### `--stage lr` 学习率（8 次）

```
Cora、Flickr 上，GCN 全图训练，lr ∈ {0.001, 0.005, 0.01, 0.05} = 8
```

对应报告要求 3.1 的"学习率的影响"。

### `--stage layers` 网络层数（6 次）

```
Cora、Flickr 上，GCN 全图训练，layers ∈ {2, 3, 4} = 6
```

对应报告要求 3.1 的"网络层数的影响"。

```bash
python code/run_all.py --stage all --epochs 150   # 全部跑
python code/run_all.py --stage main --dry_run     # 先看看会跑哪些命令
```

**断点续跑**：结果追加写在 `results/results.jsonl`，中途中断后重新运行会
**自动跳过已经跑完的实验**（按"数据集+模型+模式+学习率+层数+种子"判断）。
想全部重跑加 `--no_skip_done`。

---

## 八、结果文件说明

| 路径 | 内容 |
|---|---|
| `results/results.jsonl` | 每个实验一行 JSON，含所有超参数和测试指标 |
| `results/fig1_model_comparison.png` | 四个模型 × 三个数据集的准确率对比 |
| `results/fig2_full_vs_sample.png` | **全图 vs 采样（准确率 / 每轮耗时）** |
| `results/fig3_layers_effect.png` | 网络层数的影响 |
| `results/fig4_lr_effect.png` | 学习率的影响 |
| `logs/<数据集>_<模型>_<模式>_lr<..>_L<..>.log` | 完整训练日志 |

`results.jsonl` 的关键字段：

```
dataset, model, mode, layers, hidden, dropout, lr, weight_decay, batch_size,
num_neighbors, epochs, best_epoch, num_params, steps_per_epoch,
best_val_acc, best_test_acc, train_time_total, train_time_per_epoch,
infer_time_full, wall_time, seed, tag
```

其中 `steps_per_epoch` 特别重要 —— 分析"全图 vs 采样"的耗时和性能时，
必须结合"每个 epoch 更新了几次参数"一起看，否则结论会错。

---

## 九、参考

- **GCN**: Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
- **GAT**: Veličković et al. *Graph Attention Networks.* ICLR 2018.
- **GraphSAGE**: Hamilton et al. *Inductive Representation Learning on Large Graphs.* NIPS 2017.
- **GIN**: Xu et al. *How Powerful are Graph Neural Networks?* ICLR 2019.
- **子图采样**: <https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html>
- **Flickr 数据集**: Zeng et al. *GraphSAINT: Graph Sampling Based Inductive Learning Method.* ICLR 2020.
