# 任务三、图分类（GCN / GAT / GraphSAGE / GIN）

> 框架：PyTorch Geometric 2.0.4 + PyTorch 2.3.1
> 数据集：TUDataset（MUTAG / PROTEINS / ENZYMES / IMDB-BINARY）、ZINC

---

## 一、这个任务在做什么？

**图分类**：给一整张图，判断它属于哪一类。

```
输入：一整张图（比如一个分子）     →  输出：一个类别（比如"能致突变 / 不能致突变"）
```

**和任务一（节点分类）的区别：**

| | 任务一 节点分类 | 任务三 图分类 |
|---|---|---|
| 输入 | **一张大图**，给其中几个节点分类 | **一堆小图**，给每张图整体分类 |
| 输出粒度 | 节点级 | 图级 |
| 样本单位 | 一个节点 | 一张图 |
| 关键步骤 | 消息传递 | 消息传递 + **池化（readout）** |

**"池化"就是本任务的重点**：节点级表示 → 图级表示，怎么把 N 个节点的向量
压成 1 个向量？不同的压法（Avg / Max / Min）效果差很多。

```
            消息传递                    池化              分类
节点特征 x → [GCN/GAT/SAGE/GIN] → 每个节点的向量 → [Avg/Max/Min] → 一张图的向量 → [MLP] → 类别
```

---

## 二、目录结构

```
任务三_图分类/
├── data/                       # 数据集（已下载好，无需再下）
│   ├── MUTAG/raw/MUTAG/...
│   ├── PROTEINS/  ENZYMES/  IMDB-BINARY/
│   └── ZINC/raw/zinc_*.jsonl        # 自己解析的回归数据集
├── code/
│   ├── prepare_data.py         # 下载/检查数据集
│   ├── datasets.py             # 读 TUDataset / ZINC，度独热特征，分层划分
│   ├── models.py               # 4 个模型 + 4 种池化方式
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
| **训练 + 测试（单次实验）** | `python code/train.py --dataset MUTAG --model GCN --pooling avg` |
| **批量跑所有实验** | `python code/run_all.py --stage all` |
| **生成图表和汇总表** | `python code/plot_results.py` |
| 检查数据集是否就绪 | `python code/prepare_data.py` |

**注意：本任务没有单独的 `test.py`。** 训练脚本 `train.py` 一条龙完成
"训练 → 每轮在验证集上评估 → 挑最好的轮次 → 用那个轮次在测试集上评估 →
把指标写进 `results/results.jsonl`"，所以测试就是训练脚本的最后一步。

### 单次实验的常用参数

```bash
python code/train.py \
    --dataset MUTAG \       # MUTAG | PROTEINS | ENZYMES | IMDB-BINARY | ZINC
    --model   GCN \         # GCN | GAT | GraphSAGE | GIN
    --pooling avg \         # avg | max | min | sum   ← 本任务的重点
    --pool_norm ln \        # ln(默认,推荐) | bn(经典但有坑) | none
                            #   池化之后用什么归一化，见第六节的实测发现
    --pool_bn_eps 1e-5 \    # 仅 pool_norm=bn 时有效，池化后 BN 的 eps
    --mode    sample \      # sample(分批次) | full(全图一个批次)
    --layers  3 \           # 消息传递层数
    --hidden  128 \         # 隐藏层维度
    --lr      0.01 \        # 学习率
    --dropout 0.5 \         # Dropout 概率
    --batch_size 32 \       # 批次大小（一堆图拼成一个大图）
    --epochs  150 \         # 训练轮数
    --max_graphs 5000 \     # ZINC 最多用多少张训练图（ZINC 很大，默认限制）
    --patience 50 \         # 验证集多少轮没提升就提前停止
    --no_early_stop         # 关掉提前停止
```

加 `--dry_run` 只打印配置不训练。加 `--quiet` 少打日志（`run_all.py` 会用）。

---

## 四、数据集

### 1. TUDataset —— 图分类的标准数据集

作业要求用 TUDataset。它里面有 100 多个数据集，本任务挑了 4 个有代表性的：

| 数据集 | 图数 | 类别数 | 图平均节点数 | 节点特征 | 特点 |
|---|---|---|---|---|---|
| **MUTAG** | 188 | 2 | 17.9 | 有（7 维） | 最小的分子图数据集，用来快速验证流程 |
| **PROTEINS** | 1113 | 2 | 39.1 | 有（3 维） | 蛋白质图，判断是不是酶 |
| **ENZYMES** | 600 | 6 | 32.6 | 有（18 维） | **6 分类**，明显更难 |
| **IMDB-BINARY** | 1000 | 2 | 19.8 | **没有** | 电影合作图，要自己造节点特征 |

**IMDB-BINARY 没有节点特征**，所以要用**度独热（degree one-hot）**：
节点的度数（有几个邻居）离散化成 [0, 1, 2, ..., k]，做 one-hot 当特征。
代码在 `datasets.py` 的 `degree_one_hot()`，`k` 取 `min(最大度数, 100)`。

### 2. ZINC —— 分子性质回归

ZINC 是**分子图数据集**，任务是预测分子的一个数值性质（constrained solubility），
是个**回归**任务，不是分类。

- 每一行是一个 JSON，形如 `{"nodes": [{"label": "C"}, ...], "edges": [...], "smiles": "..."}`
- 原子类型（C / N / O / ...）做 one-hot 当节点特征
- 图很多，代码默认只用 **前 5000 张**（`--max_graphs`），否则全图训练模式会爆显存
- 评价指标用 **MAE**（平均绝对误差），**越小越好**

> **注意**：分类任务看"准确率，越大越好"，回归任务看"MAE，越小越好"。
> 代码里用 `HIGHER_IS_BETTER = {"classification": True, "regression": False}`
> 统一处理，挑最优轮次的时候会自动区分方向。

### 3. 训练/验证/测试怎么划分的？

TUDataset 官方**没有给划分**，所以要自己切。这里用**分层划分**（stratified split）：

```
1. 先按类别把图分组
2. 每一类内部打乱，按 8:1:1 切出  训练 : 验证 : 测试
3. 把各类的切分结果拼起来
4. 最后再整体打乱一次
```

| 集合 | 比例 | 作用 |
|---|---|---|
| 训练集 | 80% | 更新参数 |
| 验证集 | 10% | 每一轮评估，挑最好的轮次（**不参与训练**） |
| 测试集 | 10% | 最后只用一次，报告结果 |

**为什么要分层？** ENZYMES 有 6 类，如果纯随机切，某一类可能全被切到测试集里，
训练时模型根本没见过这个类，测试结果就没意义了。分层保证**每个集合里
各类的比例都和原始数据一致**。

**为什么最后还要再打乱一次？** 因为分层切完之后数据是按类别排好序的，
如果用 `full` 模式（所有图一个批次）训练，同一个 batch 里全是同一类的图，
BatchNorm 的均值方差会被这个类别主导，训练会不稳定。

`seed=42` 固定随机种子，保证每次划分结果一样，实验可复现。

---

## 五、四个模型是怎么实现的？

所有模型的结构都一样：**几层消息传递 → 池化 → 分类头（MLP）**，
只有"消息传递那一层"换掉了。

### 1. GCN —— 最基础的图卷积

```
x_i' = W · ( 1/(1+d_i) · Σ_j  x_j / sqrt(1+d_i) / sqrt(1+d_j) )
```

**直觉**：把邻居的特征**加权平均**一下（度大的邻居权重小），再过一层线性变换。

- 优点：简单、快、参数少。
- 缺点：权重是**固定**的（按度数算好的），不是学出来的，表达力有限。

代码：`GCNConv(hidden, hidden)`

### 2. GAT —— 带注意力机制的图卷积

```
e_ij = LeakyReLU( a^T [W h_i || W h_j] )      # 算 i 对 j 的注意力
α_ij = softmax_j(e_ij)                        # 归一化
h_i' = Σ_j α_ij · W h_j                       # 加权求和
```

**直觉**：邻居的权重不是固定的，而是**让模型自己学**"哪个邻居更重要"。

- **多头注意力**：本实现用 8 个头（`heads=8`），每个头独立算一套注意力，
  最后把结果拼起来。好比 8 个人从不同角度看同一件事，比一个人看更全面。
- 优点：能学到"重要邻居"的概念，在很多数据集上效果最好。
- 缺点：参数多、慢，小数据集上容易过拟合。

代码：`GATConv(..., heads=8, concat=True)`
**注意**：8 个头拼起来后，输出维度是 `隐藏维度/8 × 8 = 隐藏维度`，
所以每个头分到的维度是 `hidden // heads`。这里算错会导致 BatchNorm 报
`running_mean should contain 32 elements not 256` 的错误。

### 3. GraphSAGE —— 采样+聚合

```
h_i' = W · [ h_i || MEAN( { h_j : j ∈ 邻居(i) } ) ]
```

**直觉**：把"自己的特征"和"邻居特征的平均"**拼起来**再变换，
既保留自己，又吸收邻居信息。

- 原论文的设计目的是**归纳式学习**：模型学的是"怎么聚合邻居"这个函数，
  而不是"每个节点的具体向量"，所以能直接用到训练时没见过的新节点/新图上。
- 本实现聚合方式用均值（`aggr="mean"`）。原论文还试过 LSTM、Max Pooling 等，
  实验发现均值最简单也最好用。
- 优点：对新节点泛化好，实现简单。
- 缺点：均值聚合会把所有邻居一视同仁，没法区分重要性。

代码：`SAGEConv(hidden, hidden, aggr="mean")`

### 4. GIN —— 表达能力最强的那个

```
h_i' = MLP( (1 + ε) · h_i + Σ_j h_j )
```

**直觉**：先把邻居特征**求和**（注意：是求和不是平均！），
把自己乘一个系数 `1+ε`（ε 可学习），再过一个小 MLP。

- 为什么 GIN 特别？图神经网络的理论里有个结论：**用求和聚合 + MLP +
  足够多的层，GNN 的区分能力可以达到"WL 图同构测试"的上界**，
  也就是说，理论上它能区分任意两张不同的图。
  而均值/最大值聚合做不到这一点（会丢信息）。
- 为什么必须是求和而不是平均？因为平均会**抹掉图的规模信息** ——
  "1 个邻居特征为 1"和"10 个邻居特征都是 1"，平均后都是 1，分不出来；
  求和后是 1 和 10，能分出来。图分类恰恰很在意图的规模。
- 缺点：求和会让数值随度数增大而变大，深层时会不稳定。

代码：`GINConv(2层 MLP, train_eps=True)`，`train_eps=True` 表示 ε 也是可学习的。

---

## 六、池化方法对比（本任务的重点）

节点级向量 → 图级向量。设一张图有 N 个节点，每个节点是 d 维向量：

| 池化 | 公式 | 直觉 | 保留了什么信息 | 丢了什么 |
|---|---|---|---|---|
| **AvgPooling** | `(1/N) · Σ x_i` | 求平均 | 整体分布趋势 | **规模信息**（图多大看不出来） |
| **MaxPooling** | `max_i x_i`（逐维取最大） | 每维取最突出的值 | **最显著的特征** | 平均值、规模 |
| **MinPooling** | `min_i x_i`（逐维取最小） | 每维取最小的值 | **最"不突出"的特征** | 最显著特征 |
| SumPooling | `Σ x_i` | 求总和 | 规模 + 分布 | —— |

### ⚠️ 实测发现的坑一：池化后接 BatchNorm，会让 MaxPooling 完全学不动

**这是本任务里最有价值的一个发现，报告里一定要写。**

一开始按"教科书做法"实现：池化 → BatchNorm1d → ReLU → 分类头。
结果跑 PROTEINS 数据集时，**MaxPooling 的表现离谱地差**：

```
PROTEINS + GCN + MaxPooling（池化后接 BatchNorm）
  轮次    训练损失    验证损失    验证准确率
   20     0.5947     0.6048      0.6396     ← 看起来正常
   40     0.5736     0.6007      0.6847     ← 还在涨
   60     0.5452     0.5569      0.7477     ← 好好的
```

但换成 MaxPooling 的组合里，另一些配置却是这样：

```
  轮次    训练损失    验证损失    验证准确率
   20     0.6800     0.7000      0.4054     ← 卡住
   40     0.6600     0.8500      0.4054     ← 训练在降
   60     0.6400     1.0000      0.4054     ← 验证损失反着涨
```

**验证准确率 0.4054 一直不动** —— 而 PROTEINS 的多数类占比正好就是 0.4054。
也就是说模型**退化成了"永远预测多数类"**，什么都没学到。

第一反应是"代码写错了"。但用 `diagnose_pooling.py` 把池化后的图向量
直接打印出来看（随机初始化、还没训练）：

```
【MIN 池化】
  池化后图向量的维度方差: 0.000000   ← 塌缩了
【MAX 池化】
  池化后图向量的维度方差: 0.4xx      ← 正常
```

MinPooling 确实塌缩，但 **MaxPooling 的向量是正常的**，
在随机初始化下四种池化的图向量相似度都在 0.98 左右（都一样，正常）。
所以问题**不在池化本身**，而在池化**后面**那一层。

#### 真正的原因：BatchNorm 的「训练/推理统计量错位」

BatchNorm 有两套统计量：

| 阶段 | 用的是哪套均值/方差 |
|---|---|
| 训练 (`model.train()`) | **当前这个 batch** 自己的均值方差 |
| 推理 (`model.eval()`) | 训练过程中**累积的滑动平均** |

关键在于：**池化后的图向量分布会随着编码器一起漂移**。
编码器每一轮都在变，池化出来的图向量数值范围也就跟着变 ——
MaxPooling 尤其明显，因为它的输出是"逐维最大值"，
编码器把激活值训得越大，池化结果就越大。

于是出现了下面的错位：

```
第 60 轮时：
  池化图向量的真实均值 ≈ 3.0
  滑动统计量里存的还是早期累积的均值 ≈ 0.5     ← 对不上

  训练时：用 3.0 归一化 → 结果正常 → 损失在降 ✅
  推理时：用 0.5 归一化 → 结果整体偏移 → logits 全错 → 只会输出多数类 ❌
```

表现就是"**训练损失一路降，验证损失一路涨、验证准确率卡在多数类**"。
看起来像过拟合，其实是归一化统计量对不上，**训练和推理看到了两个不同的模型**。

#### 解决办法：把池化后的 BatchNorm 换成 LayerNorm

`--pool_norm`（默认已经是 `ln`）：

| 取值 | 含义 | 效果 |
|---|---|---|
| `bn` | BatchNorm1d（经典做法） | 上面说的坑，MaxPooling 会退化 |
| **`ln`** | **LayerNorm（默认，推荐）** | 对**每张图独立**归一化，不依赖跨样本的累积统计量 |
| `none` | 不做归一化 | 对照组 |

LayerNorm 按**单张图自己的**均值和方差归一化，
训练和推理的行为**完全一致**，根本不存在"统计量漂移"这回事。
现代 GNN 做图级 readout（池化后）基本都用 LayerNorm，就是这个原因。

实测对比（PROTEINS + GCN + MaxPooling，同样条件）：

| `pool_norm` | 验证准确率 | 现象 |
|---|---|---|
| `bn` | 0.4054 | 卡在多数类，验证损失 0.70 → 1.00 反向上涨 |
| `ln` | **0.72 ~ 0.75** | 正常学习，验证损失下降 |
| `none` | 0.77 | 也能学（说明问题确实出在 BN 上） |

**另一个相关参数 `--pool_bn_eps`**：BatchNorm 做 `(x - μ) / sqrt(σ² + eps)`。
如果池化后向量的方差非常小，`sqrt(σ²+eps)` 也很小，
等于"除以一个接近 0 的数"，会把微小波动放大几百倍。
把 `eps` 调大（如 0.1）相当于给分母加一个下限。
默认仍是 PyTorch 标准值 `1e-5`；如果你想用 `pool_norm=bn` 复现经典做法，
可以配合 `--pool_bn_eps 0.1` 缓解。

### ⚠️ 实测发现的坑二：MinPooling 是真的退化，不是代码 bug

换成 LayerNorm 之后，MinPooling **依然学不动**（验证准确率仍是多数类，
验证损失涨到 4.7）。这次可以确定不是归一化的问题了。

原因是数学上的：每一层卷积后面都有 **ReLU**，所以节点特征里有**大量精确的 0**。
对这样的特征逐维取最小：

```
图 A 的某个维度： [0, 0, 0, 0, 0, 0, 0, 3]  → 最小 = 0
图 B 的同一个维度：[0, 0, 0, 0, 0, 0, 0, 0]  → 最小 = 0
图 C 的同一个维度：[0, 0, 0, 1, 0, 0, 0, 0]  → 最小 = 0
```

只要**任何一个节点**在该维度上是 0（ReLU 之后极其常见），
整张图这一维的最小值就**恒等于 0**。于是不同图的图向量几乎一模一样，
分类器拿到的输入没有区分度，自然学不到东西。

**所以结论是：MinPooling 在 ReLU 激活的 GNN 上本质退化。**
这不是实现错误，而是一个**真实的、值得写进报告的负面结果**——
它恰好解释了"为什么实际做图分类几乎没人用 MinPooling"。

### 为什么 MaxPooling / AvgPooling 是主流？

图分类里，判断一个分子有没有某种性质，看的是"有没有出现某个关键子结构"，
也就是**最突出的那个特征（Max）**。所以经验上 **Max ≈ Avg > Min**，
在 MUTAG 这种"看有没有关键基团"的数据集上 Max 往往还略好一点。

代码在 `models.py` 的 `pool_nodes()`。其中 MinPooling 用
`scatter_reduce(..., reduce="amin")` 实现（PyG 2.0.4 没有内置的 global_min_pool）：

```python
out = torch.full((num_graphs, d), float("inf"))      # 先填 +inf
out = out.scatter_reduce(0, batch_idx_expanded, x, reduce="amin", include_self=True)
```

**为什么初值要填 `+inf`？** 因为要求最小值，初值必须是"比任何数都大"，
这样第一个写入的值一定会被选中。

### 怎么自己验证这个发现？

```bash
cd 任务三_图分类/code
python diagnose_pooling.py --dataset PROTEINS --model GAT
```

这个脚本会把每种池化方式"池化后图向量的标准差、维度方差、
过归一化层后被放大的倍数、过 ReLU 后非零比例、不同图之间的平均余弦相似度"
全部打印出来，**用数据说话而不是靠猜**。

---

## 七、全图训练 vs 分批次训练

对应报告要求 3.2。

| | `--mode full`（全图） | `--mode sample`（分批次） |
|---|---|---|
| 一个 epoch 几次参数更新 | **1 次** | 训练图数 / batch_size 次 |
| 一个批次的内容 | 所有训练图拼成一张大图 | 32（或 64）张图拼成一张大图 |
| 显存占用 | 高（整个训练集） | 低 |
| 梯度质量 | 全量梯度，最准 | 随机梯度，有噪声 |
| 收敛速度 | 每个 epoch 走一步 | 每个 epoch 走很多步 |
| 能不能处理大数据集 | **不能**（ZINC 全图必爆显存） | 能 |
| 适合 | 小数据集 | 大数据集 |

### 代码里的关键细节

**为了让两者可比**，代码做了这个处理：

```python
if args.mode == "full":
    train_bs = len(train_list)      # 一个批次装下所有训练图
    eff_epochs = args.epochs * 4    # 轮数 ×4，让它多走几步
else:
    train_bs = args.batch_size
    eff_epochs = args.epochs
```

**为什么 full 模式要 `×4`？** 全图模式下每个 epoch 只更新 1 次参数，
跑 150 轮就是 150 步。而 sample 模式 batch=32、训练图 800 张时，
每个 epoch 有 25 步，150 轮就是 3750 步。**步数差 25 倍**，
这样比"性能"其实是在比"谁训练的步数多"，不公平。
乘以 4 是个折中：既承认全图训练"一步更值钱"，又不至于让它跑太久。

**为什么 ZINC 不做全图对比？** ZINC 用了 5000 张训练图，全图一个批次
在 8GB 显存上直接 OOM。所以 `BATCH_DATASETS = ["MUTAG", "PROTEINS", "IMDB-BINARY"]`，
只在这三个数据集上比。

**BatchNorm 的坑之一（除零）**：如果某个 batch 里只剩 1 张图，BatchNorm 在训练模式下
方差为 0（只有一个样本，没有"批次内变化"），会算出 NaN。
（BatchNorm 在池化层后面还有一个更隐蔽的坑 —— 训练/推理统计量错位，
见第六节，那个坑会直接让 MaxPooling 退化。）
代码做了双保险：
1. `_safe_bn()`：如果发现 `batch 里有 1 张图 且 在 training 模式`，
   临时切到 eval 模式用滑动平均的统计量；
2. `drop_last`：如果"训练图数 % batch_size == 1"，就丢掉最后一个不满的批次。

---

## 八、对比实验的设计

`run_all.py` 里有五组实验，一共 **130 条**：

| stage | 内容 | 条目数 | 回答的问题 |
|---|---|---|---|
| `main` | 4 模型 × 5 数据集，平均池化 | 20 | 不同神经网络对性能的影响 |
| `pool` | 3 种池化 × 4 模型 × 4 个 TU 数据集<br>+ 3 种池化 × 2 模型 × ZINC | 48 + 6 | **不同池化方法的影响（本任务重点）** |
| `batch` | 2 种模式 × 4 模型 × 3 数据集 | 24 | 全图训练 vs 分批次（性能 + 时间） |
| `layers` | 4 种层数 × 2 模型 × 2 数据集 | 16 | 网络层数的影响 |
| `lr` | 4 种学习率 × 2 模型 × 2 数据集 | 16 | 学习率的影响 |

```bash
python code/run_all.py --stage all      # 全部跑（比较久）
python code/run_all.py --stage pool     # 只跑池化对比
python code/run_all.py --stage main --dry_run    # 先看看会跑哪些命令
```

**关于"130 条"和实际训练次数的区别**：
这 130 条里，有些条目其实是**同一个配置** ——
比如"主实验 MUTAG+GCN 平均池化"和"池化对比里 MUTAG+GCN+avg"，
配置完全一样，没必要跑两遍。按
"数据集+模型+池化+模式+学习率+层数+种子"去重后，
**实际只需要训练 92 次**（38 条是重复引用的同一份结果）。
`run_all.py` 的 `--skip_done` 会自动做到这一点，不会浪费算力。

**断点续跑**：结果追加写在 `results/results.jsonl`，中途中断后重新运行会
**自动跳过已经跑完的实验**。想全部重跑加 `--no_skip_done`。

**设计上的取舍**：
- **ZINC 的池化对比只测 GCN 和 GIN**：ZINC 慢，4 个模型 × 3 种池化要跑很久，
  挑两个最有代表性的够了。
- **层数/学习率消融只在 MUTAG 和 PROTEINS 上做**：一个分子图、一个蛋白质图，
  代表性足够，全做 5 个数据集时间上不划算。

---

## 九、结果文件说明

| 路径 | 内容 |
|---|---|
| `results/results.jsonl` | 每个实验一行 JSON，含所有超参数和测试指标 |
| `results/history/<实验名>.json` | 该次实验每一轮的训练/验证指标 |
| `results/fig1_model_comparison.png` | 四个模型在 5 个数据集上的对比 |
| `results/fig2_pooling_effect.png` | **池化方法的影响（重点图）** |
| `results/fig3_full_vs_batch.png` | 全图 vs 分批次（准确率 / 每轮耗时 / 总耗时） |
| `results/fig4_layers_effect.png` | 网络层数的影响 |
| `results/fig5_lr_effect.png` | 学习率的影响 |
| `logs/train_<实验名>.log` | 完整训练日志 |

`results.jsonl` 的关键字段：

```
dataset, model, pooling, pool_norm, mode, task_type, layers, hidden, dropout, lr,
weight_decay, batch_size, n_batches_per_epoch, epochs_run, best_epoch,
n_params, best_val_metric, best_test_metric, train_metric_final, test_loss,
train_time_total, train_time_per_epoch, wall_time_total,
n_train, n_val, n_test, seed, tag
```

**为什么一定要记录 `pool_norm`？** 因为它是"池化后归一化层"的取值，
直接决定结果可不可用（见第六节：`bn` 会让 MaxPooling 退化到多数类）。
实验名里也带了这个字段（如 `PROTEINS_GCN_pool-max_sample_pn-ln_L3_lr0.01`），
这样看日志和结果时能一眼分清"这批结果是用哪种归一化跑的"。

> ⚠️ **历史遗留**：`results/_bn_backup/` 里存的是早期用 `pool_norm=bn`
> 跑出来的结果（MaxPooling 那批数值不可用），搬到了带 `_bn_backup` 后缀的
> 目录里留作对照证据，**不要拿它写报告**。

**怎么区分分类和回归？** 看 `task_type` 字段：`classification` 用准确率，
`regression` 用 MAE。汇总表里也会写清楚。

---

## 十、实验结果怎么在报告里分析

`plot_results.py` 会打印三张表 + 画五张图，报告里可以直接用：

1. **主实验汇总表**：每个数据集上四个模型的准确率/MAE → 横向比模型
2. **池化方法对比表**：每种池化在所有模型上的平均 → 回答作业要求的"池化影响"
3. **全图 vs 分批次表**：准确率、每轮耗时、总耗时 → 回答报告要求 3.2

建议在报告里重点分析这几点：

- **池化**：
  1. 为什么 MinPooling 普遍最差？（提示：ReLU 后大量 0，最小值恒为 0）
  2. **（最有价值的一条）** 池化后面那层归一化用 BatchNorm 还是 LayerNorm，
     会让 MaxPooling 的结果天差地别（0.4054 ↔ 0.75）。
     原因是 BatchNorm 训练用批次统计量、推理用累积滑动统计量，
     而池化图向量的分布会随编码器漂移 → 训练和推理看到的是两个模型。
     换 LayerNorm 就解决了。详见第六节。
- **模型**：MUTAG 这种小数据集上，是不是 GAT 反而不如 GCN？
  （提示：参数多容易过拟合，188 张图撑不起 8 头注意力）
- **层数**：为什么层数从 2 加到 4，性能先升后降？
  （提示：**过平滑**——消息传递次数太多，所有节点的表示趋于相同）
- **全图 vs 分批**：注意观察"每轮耗时"，全图模式每轮慢得多，
  但一个 epoch 只更新 1 次；分批模式每轮快，更新很多次。
  要比"总耗时"和"最终精度"，不能只比一个。

---

## 十一、参考

- **GCN**: Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
- **GAT**: Veličković et al. *Graph Attention Networks.* ICLR 2018.
- **GraphSAGE**: Hamilton et al. *Inductive Representation Learning on Large Graphs.* NIPS 2017.
- **GIN**: Xu et al. *How Powerful are Graph Neural Networks?* ICLR 2019.
- **TUDataset**: <https://chrsmrrs.github.io/datasets/>
- **ZINC**: <https://github.com/graphdeeplearning/benchmarking-gnns>
- **PyG 文档**: <https://pytorch-geometric.readthedocs.io/>
