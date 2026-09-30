# 文本匹配（无监督 SimCSE / ESimCSE）实验报告

> 代码仓库：`transformers_tasks-main/text_matching/unsupervised/simcse`（`train.py` / `model.py` / `utils.py` / `train.sh` / `readme.md`）
> 硬件：单卡 NVIDIA GeForce RTX 3070 Ti（8 GB）+ Windows 11
> **数据来源说明**：本目录**没有 `日志.md`**。本报告的全部数值只来自三类物证——
> ① 终端截图：`logs/LCQMC/屏幕截图 2026-09-22 162356.png`（**最可靠的原始打印**）；
> ② 训练曲线 PNG：`logs/LCQMC/ERNIE-ESimCSE.png`；
> ③ checkpoint 目录：`checkpoints/LCQMC/{model_400, model_800, model_1200, model_1600, model_2000, model_2400, model_2800, model_best}`。
> 曲线反解值与配置表的汇总见 `实验报告/_原始数据/transformers_结果提取.md`，本报告未补造任何数字。

---

## 1. 任务目标与实验设置

**任务目标**：在**没有任何人工标注**的前提下，用对比学习（SimCSE / ESimCSE）训练一个中文句向量模型，使其在语义相似度判断（LCQMC 句对二分类）与排序（Spearman 相关）两个口径上都可用。

这篇与上一篇有监督文本匹配报告的**根本区别**：上一篇用 1,416 条**带标签的句对**做有监督二分类；本篇用 477,532 条**无标签的孤立句子**做自监督对比学习，标签完全不需要。训练完成后，它天然是「双塔 + 句向量」形态，可直接用于语义检索，无需任何分类头。

**数据集**：`data/LCQMC/`

| 划分 | 文件 | 规模 | 内容 |
|---|---|---|---|
| 训练集（无标签） | `train.txt` | **477,532 行** | 每行**一句独立文本**（无标签、无配对） |
| 验证/测试集（有标签） | `dev.tsv` | **8,802 行** | `句子A \t 句子B \t label` |

本报告实测的语料统计：
- `dev.tsv` 标签分布：**正例 4,402 条、负例 4,400 条**，正例占比 **50.0%（精确均衡）**；
- `train.txt` 字符长度：均值 10.9、中位数 10、最长 131；
- `dev.tsv` 单句字符长度：均值 12.5、中位数 12、最长 36；**超过 64 字符的样本占比为 0**。

> 这两条统计（正负 50/50、句子几乎都在 64 字符以内）在 §4.2 解释「recall 0.995 而 precision 0.55」时会起决定作用——它排除了「类别不均衡导致」这一最常见的解释。

**模型**：`AutoModel.from_pretrained("nghuyong/ernie-3.0-base-zh")` 作为编码器，外面包一层 `SimCSE`（`model.py`），内部把 768 维 `pooler_output` 经 `Linear(768→256)` 压到 256 维并 L2 归一化。

**超参**（取自 `train.sh`）：

| 参数 | 取值 |
|---|---|
| `--model` | `nghuyong/ernie-3.0-base-zh` |
| `--train_path` / `--dev_path` | `data/LCQMC/train.txt` / `data/LCQMC/dev.tsv` |
| `--batch_size` | **64** |
| `--max_seq_len` | **64** |
| `--learning_rate` | **1e-5** |
| `--dropout` | **0.3** |
| `--weight_decay` | 0.0（默认值） |
| `--warmup_ratio` | 0.0（默认值，无 warmup） |
| `--num_train_epochs` | **8** |
| `--valid_steps` | **400** |
| `--logging_steps` | 50 |
| `--duplicate_ratio` | 0.32（默认值，ESimCSE 的随机词重复） |
| `--device` | `cuda:2`（脚本原值） |

**评测节奏与规模换算**（关键，用来判断训练完成度）：

```
每 epoch 步数 = ceil(477532 / 64) = 7,462 step
完整 8 epoch 应有 7,462 × 8 ≈ 59,696 step
每 400 step 评测一次 → 完整训练应有 ≈ 149 次评测
```

**实际跑到的位置**：`checkpoints/LCQMC/` 只有 `model_400 … model_2800`（间隔 400，共 7 个）与 `model_best`，其中**只有 6 个（step 400 ~ 2400）真正触发了评测并写入了曲线**（曲线末端止于 2400，`model_2800` 是训练末段保存的权重但未再评测）。因此训练停在 **step 2800 ≈ 第 1 个 epoch 的 37.5%**（`2800 / 7462 = 0.375`）。**8 个 epoch 只完成了不到四成中的一个 epoch**——这是本篇最需要说明的局限（见 §5.1）。

**指标定义**（`train.py: evaluate_model(model, metric, data_loader, cosine_similarity_threshold=0.5)`）：

| 指标 | 定义 |
|---|---|
| `cos_sim` | query 句向量与 doc 句向量的余弦相似度（`F.cosine_similarity`） |
| `predictions` | `1 if cos_sim > 0.5 else 0`（**硬编码阈值 0.5**） |
| accuracy / precision / recall / f1 | `evaluate.combine([...])` 在 `dev.tsv` 8,802 条上按正类计算 |
| `spearman_corr` | `scipy.stats.spearmanr(labels, sims).correlation`，即**用连续 cos 值对 0/1 标签做排序相关**，衡量排序质量而非阈值质量 |

> 注意这两个口径的分离：P/R/F1 只看「阈值 0.5 切出来的二分类」，spearman 只看「连续分数的排序」。**两者一起看才能区分「模型学得不好」与「阈值选得不好」。**

---

## 2. 方法与实现要点

### 2.1 无监督 SimCSE 的核心思想：用 dropout 造正例

对比学习需要一个「锚点—正例—负例」的三元结构。SimCSE 的巧妙之处在于**不需要任何外部标注也不需要数据增强规则**：把**同一句话**分别过两次编码器，两次的 dropout mask 不同，就得到一对「语义相同但表示略有差异」的向量，天然构成正例；而**同一个 batch 内的其他句子**天然构成负例。

```python
# model.py: SimCSE.forward
q = get_pooled_embedding(query_input_ids, ...)   # 同一句话，第 1 次前向（dropout mask A）
d = get_pooled_embedding(doc_input_ids,   ...)   # 同一句话，第 2 次前向（dropout mask B）
cos_sim = q @ d.T                                # (batch, batch)
cos_sim -= margin_diag                           # 对角（正例）位置减 margin
cos_sim *= scale                                 # scale=20
loss = CrossEntropyLoss(cos_sim, labels=arange(batch))   # 对角为正例的 in-batch 分类
```

`utils.convert_example(mode='train')` 里 `query = doc = example.strip()`——**训练时 query 与 doc 就是同一行文本**，正例的「不同」完全由 dropout 提供。

### 2.2 关键超参的物理含义

| 组件 | 取值 | 作用 |
|---|---|---|
| `--dropout 0.3` | `nn.Dropout(0.3)` | **比默认 0.1 更大**——dropout 越强，正例对的「噪声」越大，对比任务越难，防止模型走「两次前向几乎相同 → 直接抄」的捷径 |
| `output_embedding_dim` | 256（`Linear(768→256)`） | 维度压缩，降低句向量存储与检索成本 |
| `F.normalize(p=2)` | — | 句向量 L2 归一化，使内积等价于余弦，且防止模型靠放大模长来降低 loss |
| `margin` | 0.0 | 所有正例的 cos 统一减 0；`>0` 时会把正例阈值整体压低（SimCSE 论文的 `margin` 变体） |
| `scale` | 20 | 把 cos ∈ [-1,1] 放大 20 倍后送进 CrossEntropy，**避免 softmax 饱和**、稳定梯度 |
| in-batch 负例 | batch 内其余 63 条 | batch=64 → 每个正例对应 63 个负例；batch 越大负例越丰富，因此用了 64 |

### 2.3 ESimCSE 的额外增强：`word_repetition`（`--duplicate_ratio 0.32`）

`model.py` 的 docstring 明确写的是 **ESimCSE**，`train.py` 每次前向前会对 query/doc 做 `word_repetition`：

- 目的：仅靠 dropout 构造的正例「长度不变、词序不变」，正例过于简单，容易造成 **anisotropy 崩塌**（所有句向量挤在同一个锥形小区域里，两两 cos 都很高）。
- 做法：对长度 ≥ 5 的句子，按 `dup_rate=0.32` 的比例**随机重复若干词**，把正例的句长拉长，从而**打破「正例对长度相同」这一捷径**，迫使模型学习更鲁棒的语义表示。

### 2.4 训练流程（`train.py`）

- `AdamW`，`bias` / `LayerNorm.weight` 走 no-decay 分组；
- 线性调度、`warmup_ratio=0.0`（即**没有 warmup**），`max_train_steps = 8 × 7462 = 59,696`；
- **注意学习率调度的时间尺度被拉得极长**：因为 `max_train_steps` 按 8 个完整 epoch 计算，而在 step 2800（实际停止点）时进度只有 `2800/59696 = 4.7%`，**学习率几乎还停留在初始的 1e-5 附近，几乎没有任何衰减**。这与文本分类报告里「学习率只退火到 40%」的问题同源，但这里更极端。
- 每 400 step：保存 `model_{step}` → 在全部 8,802 条 dev 上评测 → 写 5 条曲线（accuracy / precision / recall / f1 / spearman_corr）→ 用 `f1` 更新 `best_f1` 并另存 `model_best`。

---

## 3. 实验结果

### 3.1 终端截图的原始打印（**最可靠的一手数值**）

`logs/LCQMC/屏幕截图 2026-09-22 162356.png` 的逐字记录：

```
Evaluation precision: 0.54459, recall: 0.99614, F1: 0.70419, spearman_corr: 0.56181
global step 2050, epoch: 1, loss: 0.16381, speed: 0.19 step/s
global step 2100, epoch: 1, loss: 0.16099, speed: 0.18 step/s
global step 2150, epoch: 1, loss: 0.15843, speed: 0.18 step/s
global step 2200, epoch: 1, loss: 0.15586, speed: 0.19 step/s
global step 2250, epoch: 1, loss: 0.15335, speed: 0.18 step/s
global step 2300, epoch: 1, loss: 0.15096, speed: 0.15 step/s
global step 2350, epoch: 1, loss: 0.14875, speed: 0.15 step/s
global step 2400, epoch: 1, loss: 0.14650, speed: 0.15 step/s
Evaluation precision: 0.54625, recall: 0.99546, F1: 0.70541, spearman_corr: 0.56527
global step 2450, epoch: 1, loss: 0.14450, speed: 0.18 step/s
global step 2500, epoch: 1, loss: 0.14232, speed: 0.18 step/s
global step 2550, epoch: 1, loss: 0.14028, speed: 0.13 step/s
global step 2600, epoch: 1, loss: 0.13818, speed: 0.14 step/s
global step 2650, epoch: 1, loss: 0.13622, speed: 0.07 step/s
global step 2700, epoch: 1, loss: 0.13432, speed: 0.05 step/s
global step 2750, epoch: 1, loss: 0.13252, speed: 0.12 step/s
global step 2800, epoch: 1, loss: 0.13076, speed: 0.15 step/s
```

> 数值来源：**终端截图原始打印**（未经任何反解）。

据此得到两个精确的评测点：

| 评测点 | precision | recall | F1 | spearman_corr |
|---|---|---|---|---|
| **step 2050** | **0.54459** | **0.99614** | **0.70419** | **0.56181** |
| **step 2400** | **0.54625** | **0.99546** | **0.70541** | **0.56527** |

两个评测点之间的 F1 提升仅 0.00122、spearman 提升 0.00346——**已经进入平台期**。

训练侧：loss 从 step 2050 的 0.16381 单调降到 step 2800 的 0.13076；训练速度约 **0.15 step/s**（截图内区间 0.05 ~ 0.19 step/s，后段抖动明显）。

### 3.2 曲线图反解值（与截图互相印证）

`ERNIE-ESimCSE.png` 的 6 宫格里，5 条 eval 曲线的 x 轴覆盖 step 400 ~ 2400（**6 个评测点**，与 §1 中 `checkpoints/LCQMC/` 的 `model_400 / model_800 / model_1200 / model_1600 / model_2000 / model_2400` 完全对应）：

| 指标 | 曲线最大值 | 曲线最小值 | 曲线末端（step 2400） |
|---|---|---|---|
| eval/accuracy | max **0.636** | — | 0.584 |
| eval/precision | max **0.592** | — | 0.546 |
| eval/recall | max **0.997** | min **0.877** | **0.995** |
| eval/f1 | max **0.707** | — | **0.7054** |
| eval/spearman_corr | max **0.566** | — | **0.565** |

> 数值来源：曲线图 `ERNIE-ESimCSE.png` 像素反解。**由训练曲线图的刻度标定反解，精度约 ±0.002。**
> **recall 曲线的完整反解区间是 0.877（min）~ 0.997（max），末端为 0.995，全程没有被坐标轴裁剪**——反解出的末端 **0.995** 与终端截图打印的 **0.99546** 吻合到小数点后 3 位，因此这条曲线也进入了 §3.3 的交叉验证表。
> 每行说明：**曲线只记录到 step 2400**（第 6 个也是最后一个评测点），step 2800 只保存了 checkpoint、没有触发评测（按 `valid_steps=400` 推算下一个评测点应是 2800，但图中曲线末端止于 2400，与 `checkpoints/LCQMC/` 里同时存在 `model_2800` 而无对应评测记录一致）。`model_best` 对应的应是 step 2400 那一轮，与截图里「F1 0.70419（step 2050）→ 0.70541（step 2400），best 更新」的顺序吻合。

### 3.3 交叉验证：曲线反解 ↔ 终端截图

| 指标 | 截图给出的值 | 曲线反解值 | 是否一致 |
|---|---|---|---|
| SimCSE F1 | 0.70541（step 2400） | 0.7054（末端） | ✅ |
| SimCSE spearman | 0.56527（step 2400） | 0.5650（末端） | ✅ |
| **SimCSE recall** | **0.99546**（step 2400） | **0.995**（末端） | ✅ |

三者**独立吻合到小数点后 3 位**（截图是终端原始文本，反解是像素刻度标定）。这一致性说明：
1. 截图的数值可信；
2. 「曲线反解」这套读图方法的误差确实在 ±0.002 量级，因此本报告其他只能反解得到的数字（如 accuracy 0.636、precision 0.592）也应按同一精度采信。
   > 这条结论并非一开始就成立：初版读图脚本用「每行背景像素 > 60% 轴宽」判定坐标轴范围，**把贴着坐标轴顶部的长平段误判成"非坐标轴"，导致坐标轴被截短**，SimCSE 的 recall 因此被反解成 1.0146（超过物理上限 1）这类不可能的值。修正后（阈值降至 35% + 用公共行带把被截短的子图吸附回去）全部 33 条曲线距轴边界都是 16~19 px（= matplotlib 默认 5% 边距），**没有任何曲线被裁剪**，recall 反解值也随之与截图吻合。修正过程见 `_原始数据/transformers_结果提取.md` 的 F 节。

### 3.4 汇总表

| 指标 | 数值 | 来源 | 口径 |
|---|---|---|---|
| F1 | **0.70541** | 终端截图（step 2400） | 阈值 0.5 二分类 |
| precision | **0.54625** | 终端截图（step 2400） | 阈值 0.5 二分类 |
| recall | **0.99546** | 终端截图（step 2400） | 阈值 0.5 二分类 |
| spearman_corr | **0.56527** | 终端截图（step 2400） | 连续 cos 排序 |
| F1 / spearman | 0.70419 / 0.56181 | 终端截图（step 2050） | 同上 |
| accuracy（末端） | 0.584 | 曲线反解（±0.002） | 阈值 0.5 二分类 |
| accuracy（峰值） | 0.636 | 曲线反解（±0.002） | 阈值 0.5 二分类 |
| precision（末端） | 0.546 | 曲线反解（±0.002） | 阈值 0.5 二分类 |
| recall（反解 min / max / 末端） | **0.877 / 0.997 / 0.995** | 曲线反解（±0.002） | 阈值 0.5 二分类 |

---

## 4. 结果分析

### 4.1 首先必须确认：这不是一个「类别不均衡」的故事

「recall 0.995 而 precision 0.55」最常见的解释是「正类极少、模型全判正」。**本实验的数据直接排除了这一解释**：

- `dev.tsv` 的正负样本是 **4,402 : 4,400（正例 50.0%）**，完全均衡；
- 若模型「全判为正」，则 precision = 0.50、recall = 1.00、F1 = 0.667、accuracy = 0.50；
- 而实测是 precision 0.54625 / recall 0.99546 / F1 0.70541 / accuracy 0.584。

**对比可以看出：模型相对「全判正」的基线只多赚了 0.046 的 precision、0.039 的 F1、0.084 的 accuracy。** 也就是说，**在 0.5 这个阈值下，模型把 8,802 条里绝大多数样本都判成了「相似」**，只有少数被判为不相似，而其中判对的略多于一半多一点。这不是数据偏斜，而是 §4.2 的阈值/表示问题。

### 4.2 为什么「recall 0.995 而 precision 0.55」：阈值 0.5 落在余弦分布的高密度区

从源码可以精确定位原因（`evaluate_model` 的 `cosine_similarity_threshold=0.5`）：

1. **评测口径本身偏宽松**：判定「相似」只需要 `cos > 0.5`。对 L2 归一化的句向量而言，cos 0.5 意味着夹角 60°，这在语义空间里是**相当宽松**的门槛。
2. **对比学习会把句向量推入一个窄锥（anisotropy / 表示崩塌）**：SimCSE 的目标是「正例尽量近、batch 内负例尽量远」，而正例是**同一句话**，几乎必然被压到 cos≈1；在 256 维空间里，经过 2800 step 的训练后，**任意两条中文短句的 cos 普遍被抬到 0.5 以上**。于是判定阈值 0.5 对绝大多数样本都返回「相似」——recall 逼近 1，precision 掉到 0.55。
3. **本实验的文本长度加剧了这一点**：dev 单句中位长仅 12 个字符，`max_seq_len=64` 意味着**大部分 token 是 padding**。而 `model.py` 的 `get_pooled_embedding` 调用编码器时**只传了 `input_ids` 与 `token_type_ids`、没有传 `attention_mask`**：
   ```python
   pooled_embedding = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)["pooler_output"]
   ```
   BERT/ERNIE 在 `attention_mask=None` 时会依据 `token_type_ids` 推导注意力掩码，**短句的 padding 位置有可能被纳入注意力**，使不同句子的 `pooler_output` 因为「共享一大片 padding」而被系统性地拉近。这与观测到的「整体 cos 偏高」方向一致。⚠️ 这一条是**基于代码与观测方向的合理推断**，本报告不把它当作已验证的结论；验证方法很简单（在 `get_pooled_embedding` 中显式传入 `attention_mask` 并重跑评测），已列入 §6.2。
4. **同时，spearman_corr = 0.565 说明排序能力是真实存在的**：连续 cos 值与 0/1 标签的 Spearman 相关为 0.565，属于中等偏上的正相关。也就是说，**模型的分数是「有序」的，只是绝对水平整体偏高、与固定阈值 0.5 不匹配**。
5. **一个直观的数值推演**：在一个正负各半、且 cos 分数整体被抬高的分布上，把阈值从 0.5 移到更高的分位点（例如让预测为正的比例从 ~99% 降到 ~50%），precision 会显著上升、recall 下降，F1 有很大概率改善。**当前 0.70541 的 F1 是「阈值失配」下的值，不是模型能力的上限。**

**把 §3.1 的两个评测点放在一起看**：step 2050 → 2400 时，precision 0.54459 → 0.54625（+0.0017）、recall 0.99614 → 0.99546（−0.0007）、spearman 0.56181 → 0.56527（+0.0035）。**recall 微降、precision 与 spearman 微升**，方向与「模型逐步把负例的 cos 压下去一点点」一致，但幅度极小，说明在 0.5 阈值下模型已经**饱和**——继续训练带来的 F1 增益会非常有限，除非同时调阈值。

⚠️ 关于 accuracy 的一个读数细节：曲线反解的 accuracy 末端为 **0.584**，而 recall 末端高达 0.995、precision 只有 0.546。用「几乎所有样本判正、precision 0.546」反推，accuracy 应当接近 0.546（因为 recall≈1 时 accuracy ≈ 预测为正的比例 × precision + 判负中判对的比例，在正例占 50% 时 accuracy ≈ 0.5×precision + 0.5）。**0.584 与 0.546 之间的约 0.04 差距落在曲线反解 ±0.002 之外的量级**，最可能的解释是：① 五条曲线的「末端」并非严格对齐同一个 x（反解取的是每张子图最右侧的橙色像素列，各子图最右列不一定落在同一评测点上）；② accuracy 曲线本身有明显起伏（峰值 0.636 出现在中间某个评测点，其后回落到 0.584），说明该指标在 0.5 阈值下抖动较大。**因此本报告以终端截图的 0.54625 / 0.99546 / 0.70541 为权威值**，曲线反解的 accuracy 0.584 只作趋势参考。

### 4.3 为什么无监督目标能在零标注数据上拿到 0.705 的 F1

这是本篇最核心的「为什么有效」问题，可以拆成三层：

**（1）正例虽然是「同一句话」，但约束是「跨样本」的。** 每 step 的 batch 有 64 条句子，模型要在 64×64 的相似度矩阵上做 64 类分类，每个正例要同时胜过 **63 个负例**。所以虽然正例不需要标注，**负例信号来自 batch 内的天然分布**，这是一个相当强的判别约束（64 分类的随机基线只有 1/64 ≈ 1.6%）。

**（2）dropout mask 提供的是「语义不变性」的正则。** `--dropout 0.3`（比常规的 0.1 高 3 倍）意味着两次前向的隐藏单元有 30% 被随机置零。模型要让两次表示的 cos 尽量大，就必须把语义编码到**冗余、鲁棒**的方向上，而不能依赖某个具体神经元的激活。这正是 SimCSE 论文「dropout 就是最小成本的数据增强」的直觉。

**（3）ESimCSE 的词重复进一步打破捷径。** `--duplicate_ratio 0.32` 随机重复词，使正例对的**长度与词频分布发生改变**，迫使模型对齐「打乱/重复后的语义」而不是「表层 token 序列」。这对中文短句尤其重要，因为「同一句话过两遍」这个正例在字面上是完全相同的，模型极易退化成复制。

**（4）预训练权重已经提供了大部分语义能力。** `ernie-3.0-base-zh` 本身在海量中文语料上预训练过，句向量空间已有相当好的语义结构；SimCSE 的作用是**在这个空间上做「各向同性化 + 语义聚焦」的后处理**，而不是从零学语义。这解释了为什么只用 1e-5 这样的小学习率、且只训了 0.375 个 epoch，就能把 dev 上的 F1 推到 0.705。

**（5）与有监督方案的对比定位**：同语料的 PointWise（有监督，1416 条带标签句对）F1 0.90，Sentence-BERT（有监督双塔）0.816，而本篇 SimCSE **零标注** F1 0.705。三者的语料规模与任务都存在差异（见 §5.6），但这个梯度大体上符合「无监督对比学习 < 有监督双塔 < 有监督单塔」的经验规律，也说明**在完全没有标注数据时，SimCSE 可以在零标注成本下拿到一个可用的语义检索底座**。

### 4.4 loss 的一个值得注意的形态：先崩、再跳、后缓

曲线图左上角的 `train/train_loss` 呈三段式：从 0.36 快速掉到约 0.06（step 400 附近），**随后回升到约 0.34（step 500 附近）**，再缓慢单调降到 0.15（step 2400）。同期的评测也在 step 400 出现高点（F1 从 step 400 的高位回落到 step 800 的低点：曲线上 F1 在 x=400 处约 0.707 → x=800 处约 0.702，是全段最低）。

这与 `train.sh` 的关键设置吻合：**`warmup_ratio=0.0` 且 `lr=1e-5` 恒定**（调度器按 59,696 step 计算总长度，到 step 2800 只走了 4.7%，学习率几乎不衰减）。训练最开始 loss 迅速下降到 0.06，是因为模型很快找到了「让同句子的两次 dropout 表示完全对齐」的捷径（正例太容易）；随后 ESimCSE 的 `word_repetition`（0.32 的重复比例）与 in-batch 负例的难度把 loss 重新推回 0.34，模型被迫学习更泛化的表示，然后从 step 500 起稳定缓降。

> ⚠️ 上面对「loss 回升」的解释是本报告基于 `duplicate_ratio=0.32` 与训练日志形态的**分析性推断**，不是从产物中直接观测到的因果。可验证的做法是把 `--duplicate_ratio` 设为 0 重跑，观察 loss 是否仍然出现 0.06 → 0.34 的跳变。

---

## 5. 踩坑与局限

1. **训练远未完成（最严重）**：完整 8 epoch 需 59,696 step，实际只跑到 **step 2800（4.7% 的总进度、37.5% 的第 1 个 epoch）**，评测只发生 6 次（曲线可见 step 400 ~ 2400）。按截图 0.15 step/s 的速度折算，1 个 epoch 约需 **13.8 小时**（7462 / 0.15 / 3600），8 epoch 理论需要 **110 小时以上**——这是一个在单张 RTX 3070 Ti 上**实际上不可能跑完**的配置。因此所有性能数字都是「训练极早期」的读数。
2. **学习率几乎没有衰减**：`max_train_steps` 按 8 epoch 计算（59,696），而实际只走了 4.7%，`lr_scheduler` 的线性衰减形同虚设，模型基本一直以 1e-5 训练。这既是「跑得慢」的原因，也让「末段指标已进入平台」的结论带有「是学习率没退火、不是真的到顶」的歧义。
3. **评测阈值硬编码 0.5**：`evaluate_model(..., cosine_similarity_threshold=0.5)` 是函数默认参数，训练脚本没有把它暴露成命令行参数，也没有做阈值扫描。这直接导致 recall 0.995 / precision 0.55 的失衡工作点，**当前 0.70541 的 F1 严重低估了模型的排序能力**（spearman 0.565 才是更公平的能力读数）。
4. **`get_pooled_embedding` 未传 `attention_mask`**：只传了 `input_ids` 与 `token_type_ids`，短句的 padding 可能被计入池化（见 §4.2 第 3 条）。对「中位长度 12 字符、max_seq_len=64」的数据集来说，这会让每个句向量都带着一大片 padding 的贡献，是句向量相似度整体偏高的一个**可疑（未验证）**来源。
5. **无 warmup**：`warmup_ratio=0.0`，而 batch 只有 64、学习率 1e-5 且无衰减，训练早期存在表示抖动的风险（与 §4.4 的 loss 跳变可能相关）。
6. **与有监督结果不可直接比较**：SimCSE 用 LCQMC 的 477,532 条**无标签**句子训练、在 8,802 条 LCQMC 句对上评测；而 PointWise / Sentence-BERT / DSSM 用 `comment_classify` 的 1,416 条**带标签**句对训练、在 352 条 dev 上评测。**数据集、任务域、训练信号、评测集全都不同**，因此本报告 §4.3 里「0.705 vs 0.816 vs 0.90」只能作为「无监督 < 有监督」的定性参照，不能作为严格对照。
7. **单卡 8 GB 与 `cuda:2` 配置残留**：`train.sh` 写 `--device "cuda:2"`，本机只有 1 张卡，实际必然被改；batch 64 / max_len 64 在 8 GB 上已经是相当吃紧的配置，无法再靠加大 batch 来丰富 in-batch 负例。
8. **训练速度抖动剧烈**：截图显示 step/s 在 0.05 ~ 0.19 之间跳动（2650 step 时只有 0.07、2700 step 时只有 0.05），同一进程内吞吐相差近 4 倍，说明存在显存换页或其它进程争用，这会进一步拖长实际训练时间。

---

## 6. 结论与改进建议

### 6.1 结论

- 用 **ernie-3.0-base-zh + ESimCSE（dropout 0.3、`duplicate_ratio` 0.32）**在 **477,532 条无标注中文句子**上做自监督对比学习，batch 64 / max_seq_len 64 / lr 1e-5 / 8 epoch / `valid_steps` 400。
- **实测（终端截图原始打印，step 2400，dev 8,802 条）**：precision **0.54625**、recall **0.99546**、F1 **0.70541**、spearman_corr **0.56527**；step 2050 的对应值为 0.54459 / 0.99614 / 0.70419 / 0.56181。曲线反解给出 F1 0.7054、spearman 0.565、**recall 0.995（min 0.877 / max 0.997）**，**三条指标均与截图吻合到小数点后 3 位**，交叉验证通过。
- **「recall 0.995 而 precision 0.55」不是类别不均衡造成的**（dev 正负恰为 4,402:4,400，正例 50.0%），而是**余弦相似度整体被抬高 + 评测阈值硬编码 0.5 偏宽松**共同造成的：阈值落在分数分布的高密度区，几乎所有样本都被判「相似」。**spearman_corr 0.565 表明排序能力真实存在**，只是与固定阈值失配。
- **无监督目标能在零标注下拿到 0.705 F1** 的原因：in-batch 负例（每个正例对抗 63 个负例）提供判别信号、dropout 0.3 提供语义不变性正则、ESimCSE 的词重复打破「同句复制」捷径、以及 ERNIE-3.0 预训练权重本身已具备的语义结构。
- **重要限定**：训练只完成到 **step 2800 ≈ 第 1 个 epoch 的 37.5%（总计划的 4.7%）**，学习率几乎没有衰减；F1 在两个相邻评测点间只提升 0.00122，说明**在 0.5 阈值下已进入平台**。这些数字应视为「训练早期的可用结果」，而非该配置的上限。

### 6.2 改进建议

1. **做阈值扫描（收益最高、成本最低）**：在 dev 上枚举阈值 t ∈ [0, 1]，同时报告 precision/recall/F1-t 曲线与 P-R 曲线，给出**最佳阈值下的 F1**（而不是固定的 0.5）。这把「阈值失配」与「表示质量」两个问题解耦，预计是提升 F1 的第一优先项。同时把 `--cosine_similarity_threshold` 暴露成命令行参数，并在 `model_best` 的选择标准上从「固定阈值的 F1」改为 **spearman 或 AUC**（对阈值不敏感的排序指标）。
2. **修 `get_pooled_embedding` 的 mask**：显式把 `attention_mask` 传进编码器（`self.encoder(..., attention_mask=attention_mask)`），并让 `convert_example` 在 train/evaluate 两种模式下都输出 `attention_mask`。用「改动前后 dev 上的 cos 分布直方图」量化 padding 的影响。
3. **把训练规模降到可完成的量级**：单卡 RTX 3070 Ti 上 8 epoch 需 110 小时以上，不可行。建议二选一——(a) 把 `--num_train_epochs` 改为 1，并把 `lr_scheduler` 的 `max_train_steps` 与之一致（让学习率真正退火到 0）；(b) 保持 epoch 数但**下采样训练集**（例如取 10 万条），使总步数落在几千步内。
4. **补上 warmup**：把 `--warmup_ratio` 设成 0.06（与仓库其它任务一致），抑制训练早期的表示抖动。
5. **消融 ESimCSE 的两个增强**：分别跑 `--duplicate_ratio 0` 与 `--dropout 0.1`，与基线（0.32 / 0.3）对比 F1 与 spearman，验证 §4.4 关于「loss 回落—跳升」的解释，并确认两个增强各自的贡献。
6. **与有监督方案做公平对照**：把 SimCSE 训好的句向量直接冻结，在其上只用 `comment_classify` 的少量标签训一个 `Linear(256→2)`（linear probing），与 Sentence-BERT 的 0.816 对比，才能定量回答「无监督预训练 + 少量监督」相对「端到端有监督」的差距。
7. **落盘 cos 分布**：评测时把全部 8,802 条的 `(label, cos_sim)` 写文件，既能事后扫阈值、也能画分数直方图，是排查 §4.2 类问题最直接的证据。

---

## 7. 复现命令

以下命令逐字取自 `text_matching/unsupervised/simcse/train.sh`（工作目录为 `transformers_tasks-main/text_matching/unsupervised/simcse`）：

```sh
python train.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/LCQMC/train.txt" \
    --dev_path "data/LCQMC/dev.tsv" \
    --save_dir "checkpoints/LCQMC" \
    --img_log_dir "logs/LCQMC" \
    --img_log_name "ERNIE-ESimCSE" \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --batch_size 64 \
    --max_seq_len 64 \
    --valid_steps 400 \
    --logging_steps 50 \
    --num_train_epochs 8 \
    --device "cuda:2"
```

> 注：`--device "cuda:2"` 为脚本原值；本机只有一张 GPU，复现时请改为 `cuda:0`。

**依赖安装与推理（按 readme）：**

```sh
pip install -r ../requirements.txt
python inference.py
```

**按 §6.2 建议的补跑版本**（把训练规模压到单卡可完成、补 warmup、并先把 mask 修好）：

```sh
# ① 单 epoch 完整训练 + 学习率真正退火（约 14 小时，可在后台跑）
python train.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/LCQMC/train.txt" \
    --dev_path "data/LCQMC/dev.tsv" \
    --save_dir "checkpoints/LCQMC" \
    --img_log_dir "logs/LCQMC" \
    --img_log_name "ERNIE-ESimCSE" \
    --learning_rate 1e-5 \
    --dropout 0.3 \
    --batch_size 64 \
    --max_seq_len 64 \
    --valid_steps 400 \
    --logging_steps 50 \
    --num_train_epochs 1 \
    --warmup_ratio 0.06 \
    --device "cuda:0"

# ② 用 model_best 在 dev 上扫阈值，报告最佳阈值下的 P/R/F1 与 spearman
python inference.py
```

**预期产物**：`logs/LCQMC/ERNIE-ESimCSE.png`（5 条 eval 曲线 + loss 曲线）、`checkpoints/LCQMC/{model_400 … }` 与 `model_best`（本次为 `model_400` ~ `model_2800` 共 7 个 + `model_best`）。
