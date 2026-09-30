# 图神经网络入门 —— 四个任务的完整实现

> 📁 **仓库导航**：本仓库是新生入学任务的三个部分 ——
> ① 图神经网络（`graph_beginner-main/`，4 个任务）② LLM 入门（`llm-beginner-master/`，6 个任务）
> ③ Transformers 实战（`transformers_tasks-main/`，若干子任务）。
> **各任务的实测结果、指标表与分析都写在 [`实验报告/`](实验报告/README.md) 里**（含原始数据出处与复现命令），
> 本文件是其中"图神经网络"部分的项目说明。

> 框架：**PyTorch Geometric 2.0.4** + PyTorch 2.3.1 + CUDA 12.1
> 数据集：Cora / Citeseer / Flickr（节点分类、链路预测）、
> TUDataset + ZINC（图分类）、WN18RR / FB15k-237（知识图谱）

---

## 一、四个任务一览

| 任务 | 内容 | 数据集 | 模型 | 重点 |
|---|---|---|---|---|
| [任务一](任务一_节点分类/README.md) | **节点分类** | Cora, Citeseer, Flickr | GCN, GAT, GraphSAGE, GIN | 全图训练 vs 子图采样 |
| [任务二](任务二_链路预测/README.md) | **链路预测** | Cora, Citeseer, Flickr | GCN, GAT, GraphSAGE, GIN | 无泄漏的边划分 |
| [任务三](任务三_图分类/README.md) | **图分类** | MUTAG, PROTEINS, ENZYMES, IMDB-BINARY, ZINC | GCN, GAT, GraphSAGE, GIN | **池化方法对比** |
| [任务四](任务四_知识图谱/README.md) | **知识图谱补全** | WN18RR, FB15k-237 | TransE, RotatE, ConvE | 过滤式评估 |

### 四个任务的核心区别

```
任务一 节点分类      一张大图  →  猜每个节点是什么类
任务二 链路预测      一张大图  →  猜两个节点之间有没有边
任务三 图分类        一堆小图  →  猜每张图整体是什么类     ← 多了"池化"这一步
任务四 知识图谱      一堆三元组 → 猜 (头实体, 关系, ?) 的尾实体  ← 关系也要学
```

**一句话串起来**：任务一、二是"**节点级**"任务（大图、节点分类），
任务三是"**图级**"任务（小图、整图分类），任务四是"**关系级**"任务
（一个头实体通过不同关系连到不同尾实体）。

---

## 二、目录结构

```
入学任务/
├── 任务一_节点分类/
│   ├── data/               # 数据集（已下载好）
│   ├── code/               # 全部代码
│   ├── logs/               # 每次实验的完整日志
│   ├── results/            # results.jsonl + 图表
│   └── README.md           # 写明训练和测试脚本
├── 任务二_链路预测/
│   └── （同上结构）
├── 任务三_图分类/
│   └── （同上结构）
├── 任务四_知识图谱/
│   └── （同上结构）
├── requirements.txt        # 运行环境
├── 运行说明.md              # ★ 剩下的实验怎么跑（操作手册，照着敲命令就行）
└── README.md               # 本文件
```

> 📌 **想直接开始跑实验？** 看 [**运行说明.md**](运行说明.md)。
> 那份文档写了每个任务还剩多少实验、具体敲什么命令、断了怎么续跑、结果怎么判断好坏。
> 本文件主要讲"每个任务在做什么、为什么这么做"。

每个任务的 `code/` 目录里是同一套结构的 6 个脚本：

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 下载/检查数据集（**只用来准备数据，不训练**） |
| `datasets.py` | 读数据、划分训练/验证/测试集、负采样 |
| `models.py` | 模型定义 |
| `train.py` | ★ **训练 + 测试**（单次实验，一条龙跑完） |
| `run_all.py` | 一键跑完所有对比实验 |
| `plot_results.py` | 画图 + 打印汇总表 |
| `utils.py` | 日志、随机种子、结果落盘等公共工具 |

---

## 三、怎么运行？（新手按这个顺序来）

### 第 0 步：确认环境

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available())"
```

应该打印 `2.3.1+cu121 True`。如果报错，看 `requirements.txt` 里的安装说明。

### 第 1 步：检查数据集

每个任务的 `data/` 目录里数据集**已经下载好了**，可以直接用。
想确认一下，随便跑一个：

```bash
cd 任务一_节点分类/code
python prepare_data.py
```

### 第 2 步：先单独跑一次，确认流程通顺

```bash
# 任务一：Cora 上跑 GCN，全图训练
python train.py --dataset Cora --model GCN --mode full --epochs 20
```

跑完看一眼：
- 终端会打印每个 epoch 的 loss 和准确率
- `logs/` 下多了一个日志文件
- `results/results.jsonl` 多了一行

三个都正常，说明环境没问题。

### 第 3 步：一键跑完全部对比实验

```bash
python run_all.py --stage all      # 各任务支持的具体 stage 见各自的 README
```

**这会比较久**（任务三 130 次实验、任务四 27 次实验）。
建议用 `--dry_run` 先看一眼要跑什么，再决定要不要用
`--epochs 50` 之类的参数缩短。

> ⚠️ **只想"看一眼"的时候，一定要加 `--dry_run`。**
> 不加的话 `run_all.py` 会**直接开始训练**，哪怕你把输出管进了 `grep`——
> 管道只是藏起了输出，进程照样在跑。（这个坑真踩过：本来只想数一数还剩几个实验，
> 结果后台真的跑起来了。）

### 第 4 步：出图

```bash
python plot_results.py
```

`results/` 目录下会多出几张 png，终端会打印汇总表。

---

## 四、每个任务怎么训练 / 怎么测试（作业要求）

**四个任务都没有单独的 `test.py`**，因为训练脚本 `train.py` 已经一条龙包含了
测试流程：

```
train.py 做的全部事情：
  1. 读数据、划分训练/验证/测试集
  2. 搭模型、定义损失函数和优化器
  3. for epoch in 1..N:
         在训练集上训练一轮
         每隔几轮 → 在验证集上评估 → 记录指标，变好就存下模型参数
  4. 训练结束后，用"验证集上最好那一轮"的参数，在测试集上评估一次
  5. 把这一轮的所有超参数 + 测试指标追加写进 results/results.jsonl
```

**为什么用"验证集上最好的那一轮"而不是最后一轮？**
因为训练到后面会**过拟合** —— 训练集损失还在降，但验证集指标已经开始变差。
如果直接拿最后一轮去测测试集，结果会偏悲观，也不公平。
用验证集挑轮次是标准做法（**注意：绝对不能用测试集挑轮次**，那就作弊了）。

| 任务 | 训练 + 测试的命令 |
|---|---|
| 任务一 | `python 任务一_节点分类/code/train.py --dataset Cora --model GCN --mode full` |
| 任务二 | `python 任务二_链路预测/code/train.py --dataset Cora --model GCN --mode full` |
| 任务三 | `python 任务三_图分类/code/train.py --dataset MUTAG --model GCN --pooling avg` |
| 任务四 | `python 任务四_知识图谱/code/train.py --dataset WN18RR --model TransE` |

用 `--help` 可以看每个脚本的全部参数：

```bash
python train.py --help
```

---

## 五、报告要求对应的实验

作业的报告要求是：

> 3.1 分析不同参数（**学习率、网络层数**）和不同的神经网络对性能的影响
> 3.2 测试**全图训练和分批次训练**对模型性能和运行时间的影响（**任务四不需要**）

代码里的对应关系：

| 报告要求 | 任务一 | 任务二 | 任务三 | 任务四 |
|---|---|---|---|---|
| 不同神经网络 | `--stage main` | `--stage main` | `--stage main` | `--stage main` |
| 学习率 | `--stage lr` | `--stage lr` | `--stage lr` | `--stage lr` |
| 网络层数 | `--stage layers` | `--stage layers` | `--stage layers` | **用 `--stage dim` 代替**（见下） |
| 全图 vs 分批 | `--stage main`（含） | `--stage main`（含） | `--stage batch` | **不适用** |
| 池化方法（任务三专属） | —— | —— | `--stage pool` | —— |

> **⚠️ 任务四为什么没有"网络层数"？**
>
> TransE 就是 `h + r ≈ t` 一个公式，RotatE 就是复数乘法，ConvE 是
> "一层卷积 + 一层全连接" —— **三个模型都没有"层数"这个超参数**。
> 所以任务四用 **embedding 维度 `dim`** 作为"网络容量"的等价消融，
> 对应 `--stage dim`。这是知识图谱领域的标准做法（KGE 论文做的参数
> 敏感性实验都是 dim）。详见[任务四的 README](任务四_知识图谱/README.md)。

> **⚠️ 任务四为什么不需要"全图 vs 分批"？**
>
> 作业原文写的就是"任务四不需要"。而且从原理上讲，知识图谱的"全图训练"
> 和"分批训练"区别和 GNN 完全不同 —— 知识图谱根本没有"整张图的邻接矩阵"
> 这个概念，实体是靠三元组索引的，哪来的"全图"？

---

## 六、环境与踩坑记录

### 环境

```
Python        3.9.18
PyTorch       2.3.1 + CUDA 12.1
PyG           2.0.4
显卡          NVIDIA GeForce RTX 3070 Ti (8 GB)
系统          Windows 11 + WSL2 (Ubuntu 22.04)
```

详细安装说明见 `requirements.txt`。

### 踩过的坑（都已在代码里处理）

1. **PyG 版本必须锁 2.0.4**
   更新的 PyG 改了 import 路径和算子签名。比如 `global_min_pool` 在这个版本里
   没有，代码用 `scatter_reduce(..., reduce="amin")` 自己实现。

2. **BatchNorm 在 batch 只有 1 张图时会出 NaN**
   BatchNorm 在训练模式下要算"批次内的均值和方差"，批次里只有 1 个样本时
   方差为 0，会算出 NaN。代码做了双保险：一个 `_safe_bn()` 守卫（发现
   batch=1 就临时切 eval 模式），一个 `drop_last`（丢掉最后那个不满的批次）。

3. **GAT 多头注意力的维度**
   8 个头拼起来后输出维度是 `每头维度 × 8`，如果算成 `隐藏维度 × 8`，
   BatchNorm 会报 `running_mean should contain 32 elements not 256`。
   代码里专门写了 `_conv_out_dim()` 处理。

4. **负采样训练时不要开标签平滑**
   每个正样本只配 64 个负样本时，标签从 1/0 改成 0.9/0.1 会让 BCE 损失
   加一个约 0.325 的硬下界，模型学到这点区分度后就再也不下降了
   （实测损失死死卡在 0.3526）。任务四里拆成两个参数处理。

5. **1-N 打分不能用于距离型模型**
   用 1-N 打分训练 TransE，损失能降到 0.0168 看起来完美，但 MRR ≈ 0.0008。
   原因是 TransE 只要把 `h + r` 推到无穷远就能让所有分数一起变低 ——
   这是距离型模型特有的**退化逃逸**。所以任务四默认用负采样 BCE。

6. **链路预测的信息泄漏**
   划分边时如果不去重（只保留 `u < v`），同一条边会以 `(u,v)` 和 `(v,u)`
   两种形式分别进训练集和测试集，AUC 会虚高到 0.99。而且测试边绝对不能
   参与消息传递。详见任务二的 README。

7. **评估时分块计算**
   Flickr 的测试集有 450 万条负样本，一次性算会要 1GB 显存直接爆掉。
   代码按 30 万条一块分块算。

8. **国内网络下载数据集**
   PyG 默认从 github 拉数据，国内经常超时。`prepare_data.py` 里配了
   jsdelivr / ghproxy / 清华镜像等备用源。

9. **Windows 控制台编码**
   Windows 的 GBK 编码打印中文和特殊符号会报错，代码开头统一加了
   `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`。

10. **⚠️ 图分类：池化后面接 BatchNorm 会让 MaxPooling 完全失效**（最隐蔽的一个）
    现象是验证准确率死死卡在多数类占比（PROTEINS 上是 0.4054），
    验证损失反而一路涨，看起来像过拟合。
    真正的原因是 BatchNorm **训练时用当前批次统计量、推理时用累积滑动统计量**，
    而池化图向量的分布会随编码器一起漂移 —— 于是训练和推理看到的是两个不同的模型。
    换成 **LayerNorm**（对每张图独立归一化，不依赖任何跨样本累积量）后，
    准确率从 0.4054 回到 0.72~0.77。任务三默认已改为 `--pool_norm ln`。
    详见[任务三的 README](任务三_图分类/README.md)。

11. **⚠️ 知识图谱：权重衰减必须设成 0**（同样隐蔽，因为它不报错）
    通用默认值 `weight_decay=5e-4` 对 KGE 模型是致命的：
    实测 WN18RR + TransE 同样跑 12 轮，`5e-4` 时验证 MRR 只有 **0.0007**，
    设成 `0` 则是 **0.0936** —— 相差 130 倍，而且损失不变 NaN、不报错，
    只是安静地产出一份错误结果。
    原因是 Adam 会把"权重衰减"这一项也归一化到 `lr` 量级，
    于是它变成一个和真实梯度**一样大的、恒定把参数往 0 拉的力**，
    而 KGE 的参数几乎全是 embedding。任务四已写死 `DEFAULT_WD = 0.0`。
    详见[任务四的 README](任务四_知识图谱/README.md)。

12. **图分类：MinPooling 是真的退化，不是代码 bug**
    ReLU 之后节点特征里有大量精确的 0，逐维取最小 → 只要有一个节点是 0，
    这一维的最小值就恒为 0。结果不同图池化出来的向量几乎一模一样，
    分类器没有区分度。这是个**真实的负面结果**，
    正好解释了为什么实际做图分类几乎没人用 MinPooling。
    诊断脚本：`python diagnose_pooling.py --dataset PROTEINS --model GAT`。

### 三个操作教训

**教训一：`pkill -f` 的模式一定要写窄。**

跑实验时用了 `pkill -f "run_all.py --stage"` 想停一个任务，
结果这个模式**把另外两个正在跑的任务一起杀了**，损失了两个小时的进度。

后来给所有 `run_all.py` 加了 **`--skip_done`（默认开启）**：
所有结果追加写在 `results/results.jsonl`，重新运行时按配置字段自动跳过
已完成的实验，中断后可以直接续跑，不会重跑。

**教训二：想看实验列表必须加 `--dry_run`。**

`python run_all.py --stage all | grep 本次要跑` 并不能"只看不跑"——
管道只是藏起了输出，`run_all.py` 照样在后台真的开始训练了。
只看列表请用 `--dry_run`。

**教训三：杀掉壳进程之后，子进程可能变孤儿继续跑。**

用 `TaskStop`/杀进程之后，`train.py` 的子进程可能活下来占着显存和内存。
杀完一定要回头确认：`nvidia-smi` 里没有计算进程、
`tasklist.exe /FI "IMAGENAME eq python.exe"` 里没有残留的 python。
杀之前先用 `Get-CimInstance Win32_Process` 看清楚那个 PID 到底在跑什么，
别误杀别人的程序。

---

## 七、结果怎么看

每个任务的 `results/` 目录里：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 每个实验一行 JSON，含全部超参数 + 测试指标。**这是原始数据** |
| `history/<name>.json` | 该次实验每一轮的训练/验证曲线 |
| `fig*.png` | `plot_results.py` 生成的图表 |
| `logs/*.log` | 完整训练日志（终端看到的一切都在里面） |

**`results.jsonl` 怎么用？** 它就是一个"实验台账"，
一行 = 一次实验。想自己分析的话：

```python
import json
rows = [json.loads(l) for l in open("results/results.jsonl", encoding="utf-8")]
# 按条件筛，比如找 Flickr 上所有采样训练的实验
for r in rows:
    if r["dataset"] == "Flickr" and r["mode"] == "sample":
        print(r["model"], r["best_test_acc"])
```

`plot_results.py` 干的就是这件事，只是顺手画成了图。

---

## 八、参考

| 模型 | 论文 |
|---|---|
| GCN | Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017 |
| GAT | Veličković et al. *Graph Attention Networks.* ICLR 2018 |
| GraphSAGE | Hamilton et al. *Inductive Representation Learning on Large Graphs.* NIPS 2017 |
| GIN | Xu et al. *How Powerful are Graph Neural Networks?* ICLR 2019 |
| TransE | Bordes et al. *Translating Embeddings for Modeling Multi-relational Data.* NIPS 2013 |
| RotatE | Sun et al. *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.* ICLR 2019 |
| ConvE | Dettmers et al. *Convolutional 2D Knowledge Graph Embeddings.* AAAI 2018 |

- PyG 官方文档：<https://pytorch-geometric.readthedocs.io/>
- KGE 框架（任务四参考）：<https://github.com/Maxioo/kge_framework>
- DGL 子图采样教程（作业参考资料）：<https://docs.dgl.ai/tutorials/large/L0_neighbor_sampling_overview.html>
