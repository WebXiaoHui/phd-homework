# Prompt 学习（PET 与 p-tuning）实验报告

> 代码仓库：`transformers_tasks-main/prompt_tasks/{PET, p-tuning}`（`pet.py` / `p_tuning.py` / `Template.py` / `verbalizer.py` / `utils.py` / `RDropLoss.py`）
> 硬件：单卡 NVIDIA GeForce RTX 3070 Ti（8 GB）+ Windows 11
> **数据来源说明**：本目录**没有 `日志.md`**。本报告的全部数值只来自三类物证——
> ① 终端截图：`p-tuning/logs/comment_classify/屏幕截图 2026-09-22 181302.png`（**唯一保留了逐类指标的原始打印**）；
> ② 训练曲线 PNG：`PET/logs/comment_classify/BERT-PET.png`、`p-tuning/logs/comment_classify/BERT.png`；
> ③ checkpoint 目录：`PET/checkpoints/comment_classify/`（17 个）、`p-tuning/checkpoints/comment_classify/{model_10, model_20, model_best}`。
> 曲线反解值与配置表的汇总见 `实验报告/_原始数据/transformers_结果提取.md`，本报告未补造任何数字。
> ⚠️ PET 的训练过程**没有保留终端截图**（`PET/logs/comment_classify/` 下只有 `BERT-PET.png`），因此 PET 的指标只有曲线反解这一个来源。

---

## 1. 任务目标与实验设置

**任务目标**：在同一批商品评论语料上，用**极少量的标注数据（61 条训练样本）**比较两种提示学习（prompt learning）范式：

- **PET（Pattern-Exploiting Training）**：人工设计**离散硬模板**（把分类任务改写成完形填空），再用**verbalizer**把「标签词」映射回真实类别；
- **p-tuning**：不人工设计模板词，而是插入一段**可学习的连续提示向量（p-embedding）**，让模型自己学出「模板」。

两者都不新增分类头，而是**复用预训练模型的 MLM（掩码语言模型）目标**——这正是提示学习与上一篇「加 `Linear` 分类头做微调」的本质区别。

**数据集**：`data/comment_classify/`，格式为 `label\t文本`。**PET 与 p-tuning 用的是完全相同的 `train.txt`（`diff` 结果一致）**，只是 dev 划分不同。

| 项 | PET | p-tuning |
|---|---|---|
| `train.txt` | **61 条**（`_原始数据/transformers_结果提取.md` 记载） | **61 条**（同上） |
| `dev.txt` | **589 条** | **415 条** |
| 训练集类别数 | **8 类** | **8 类** |
| dev 类别数 | **10 类** | **10 类** |

> 说明：本报告直接按 `wc -l` 统计得到 62 / 590 / 416 行，而 `transformers_结果提取.md` 的配置表与 §E.3 记载的是 61 / 589 / 415——差值恰为 1 行，属文件行数口径（末行换行）差异。本报告在表格中采用结果提取文件的权威数字（61 / 589 / 415），并如实标注这 1 行的差异。

**训练集类别分布（PET 与 p-tuning 完全相同，8 类）**：

| 类别 | 书籍 | 平板 | 水果 | 洗浴 | 电脑 | 蒙牛 | 衣服 | 酒店 | 手机 | 电器 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 3 | 12 | 7 | 6 | 2 | 1 | 16 | 16 | **0** | **0** |

**dev 类别分布（10 类）**：

| 类别 | 书籍 | 平板 | 手机 | 水果 | 洗浴 | 电器 | 电脑 | 蒙牛 | 衣服 | 酒店 |
|---|---|---|---|---|---|---|---|---|---|---|
| PET dev | 44 | 92 | 17 | 108 | 87 | 3 | 32 | 19 | 100 | 88 |
| p-tuning dev | 29 | 67 | 13 | 78 | 67 | 2 | 22 | 13 | 70 | 56 |

> ⚠️ **这是本任务最关键、也最容易被忽略的一个事实**：训练集只有 8 类，而 dev 有 10 类。**`手机` 与 `电器` 在训练集中一条样本都没有**——模型在训练时从未见过这两个类别的 verbalizer 映射被激活，评测时却要求它输出这两个标签。这直接解释了后面 `电器` 的 F1 = 0（见 §4.2）。

**模型与超参**（均取自各自的 `train.sh`）：

| 参数 | PET | p-tuning |
|---|---|---|
| 骨干 | `bert-base-chinese` | `bert-base-chinese` |
| 加载方式 | 未显式指定，见 §2.1 | `AutoModelForMaskedLM.from_pretrained(...)` |
| checkpoint 中 `architectures` | `BertForMaskedLM` | `BertForMaskedLM` |
| 参数量（safetensors 头部统计） | **102,290,312（102.29 M）** | **102,290,312（102.29 M）** |
| `--batch_size` | 8 | 8 |
| `--max_seq_len` | **256** | **128** |
| `--learning_rate` | 5e-5（默认值） | 5e-5（默认值） |
| `--num_train_epochs` | **200** | **20** |
| `--valid_steps` | **40** | **20** |
| `--logging_steps` | 5 | 5 |
| `--max_label_len` | 2 | 2 |
| `--p_embedding_num` | —（不使用） | **15** |
| `--rdrop_coef` | **5e-2** | 0.0（默认值，即未启用） |
| `--device` | `cuda:1`（脚本原值） | `cuda:0` |

**评测节奏与规模换算**（关键，用来判断训练完成度）：

```
每 epoch 步数 = ceil(61 / 8) = 8 step
PET：200 epoch × 8 = 1,600 step，每 40 step 评测 → 完整训练应有 40 次评测
p-tuning：20 epoch × 8 = 160 step，每 20 step 评测 → 完整训练应有 8 次评测
```

**实际跑到的位置**：

- **PET**：`checkpoints/comment_classify/` 下有 `model_10, model_20, …, model_160` 共 16 个 step 快照 + `model_best`，**最大 step = 160**。即实际只跑了 **160 / 1,600 = 10%** 的配置训练量，只发生 **4 次评测**（step 40 / 80 / 120 / 160）。曲线图 `BERT-PET.png` 的 x 轴也正好止于 **160**，与此吻合。
- **p-tuning**：只有 `model_10, model_20, model_best`，最大 step = 20，**只发生 1 次成功的评测**（step 20），但按 160 step 的总量看，模型在 step 20 之后**仍在继续训练**（截图显示到 step 20 之后还有 loss 打印，见 §3.3）。

**指标定义**（`class_metrics.py` 的 `ClassEvaluator`，两个任务共用）：
- `accuracy` / `precision` / `recall` / `f1`：sklearn 同名函数，**`average='weighted'`**（按各类样本数加权），保留 2 位小数；
- 逐类 `class_metrics`：由混淆矩阵逐类算 P/R/F1（`precision = M[i,i]/sum(M[:,i])`、`recall = M[i,i]/sum(M[i,:])`）；
- **评测路径不同**：PET 走 `verbalizer.batch_find_main_label(predictions)`（先得到子标签，再映射回主标签），p-tuning 在评测时把 `mask_positions` 处的 argmax token **直接**当作标签（`convert_logits_to_ids`），两步的差异见 §2.4。

---

## 2. 方法与实现要点

### 2.1 PET：人工硬模板 + verbalizer

**（1）模板构造（`Template.py: HardTemplate`）。** `prompt.txt` 只有一行：

```
这是一条{MASK}评论：{textA}。
```

`HardTemplate.parse_prompt()` 把它拆成 `['这','是','一','条','MASK','评','论','：','textA','。']`，其中 `{...}` 包裹的是自定义字段（`custom_tokens = {'MASK','textA'}`）。`__call__` 再把 `textA` 填入评论文本、把 `MASK` 展开成 `max_label_len=2` 个 `[MASK]`，得到形如：

```
[CLS] 这 是 一 条 [MASK] [MASK] 评 论 ： 某个评论内容 。 [SEP]
```

注意 `utils.py` 里的 `content = content[:max_seq_len-10]`——**先把评论截短，再拼模板**，防止 `[MASK]` 被截断（模板在句首，若先截断评论再拼接反而更安全，这里是先截评论）。

**（2）Verbalizer（`verbalizer.py`）。** `verbalizer.txt` 是 1 对 1 映射：

```
电脑→电脑  水果→水果  平板→平板  衣服→衣服  酒店→酒店
洗浴→洗浴  书籍→书籍  蒙牛→蒙牛  手机→手机
```

`Verbalizer` 把每个标签词转成 token id（`max_label_len=2`，两字词占两个 `[MASK]` 位；单字词用 `[PAD]` 补齐）。训练时用 `batch_find_sub_labels` 把 `mask_labels` 扩成子标签集合，评测时用 `batch_find_main_label` 把预测出的 token 序列映射回主标签。

**（3）损失（`utils.py: mlm_loss`）。** 只对 `mask_positions` 位置的 logits 计算交叉熵：

```python
single_mask_logits = single_logits[single_mask_positions]        # (label_len, vocab)
single_mask_logits = single_mask_logits.repeat(n_sub_labels, 1, 1).reshape(-1, vocab)
cur_loss = CrossEntropyLoss()(single_mask_logits, sub_mask_labels.flatten())
cur_loss = cur_loss / len(single_sub_mask_labels)
loss = sum_over_batch(cur_loss) / batch_size / masked_lm_scale
```

即**把分类任务彻底改写成了 MLM 完形填空**，除 MLM 头外不新增任何参数，损失只在 `[MASK]` 位置上回传。

**（4）R-Drop（`RDropLoss.py`）——本次配置里实际未生效。** `train.sh` 传了 `--rdrop_coef 5e-2`，`pet.py` 也确实 `from RDropLoss import RDropLoss` 并构造了 `rdrop_loss = RDropLoss()`，但**训练循环里的 `mlm_loss(...)` 调用并没有把 `args.rdrop_coef` 传进去**，也没有 `compute_kl_loss` 的调用（`pet.py` 中 `rdrop_loss` 仅在 161 行被构造、此后未再被使用）。因此：

> **PET 这次运行实际是「硬模板 + verbalizer + 纯 MLM 交叉熵」，R-Drop 正则并未参与训练。** 这是一个配置与实现不一致的坑（对照 `p_tuning.py` 第 207~213 行，那里有完整的 `if args.rdrop_coef > 0: ce_loss + kl_loss * args.rdrop_coef` 分支）。复现时不能把本报告的结果当作「PET + R-Drop」的成绩。

### 2.2 p-tuning：连续提示 + 全参数微调

**（1）提示插入（`utils.py: convert_example`）。** p-tuning 不用任何模板文字，而是把 15 个 `[unused]` token 拼在**句首**：

```python
p_tokens = ["[unused{}]".format(i+1) for i in range(p_embedding_num)]   # 15 个
tmp = [CLS] + [MASK]*max_label_len + 正文 + [SEP]                        # MASK 紧跟 [CLS]
input_ids = p_tokens_ids + tmp                                           # p_token + [CLS][MASK][MASK]正文[SEP]
mask_positions = [len(p_tokens_ids) + 1 + i for i in range(2)]           # 第 17、18 位
```

得到的序列形如 `[unused1..unused15] [CLS] [MASK] [MASK] 评论... [SEP]`——**`[MASK]` 被放在 `[CLS]` 之后、正文之前**，与 PET 把 `[MASK]` 放在模板中间（"这是一条[MASK][MASK]评论："）形成鲜明对照。

**（2）"软"在哪里？** `p_tuning.py` 直接 `AutoModelForMaskedLM.from_pretrained(...)`，**没有自定义 `nn.Module`、没有新建 `nn.Parameter`**。所谓连续提示，就是 `[unused1..15]` 这 15 个 token 的**词嵌入行**（`word_embeddings` 里对应 15 行）。因为 `AdamW` 的 `optimizer_grouped_parameters` 包含了**模型全部参数**，所以这 15 行嵌入会随训练一起被更新——**从"固定的、语义无关的 one-hot 行"变成"任务相关的连续向量"**。这就是 p-tuning 的"软提示"在代码层面的落地方式。

> 这一点值得单独强调：**该实现是「全参数微调 + 可学习提示嵌入」，而不是 P-Tuning v2 那种「冻结主干、只训 prompt encoder」的参数高效微调。** 从 checkpoint 参数量可以佐证——p-tuning 的 `model.safetensors` 是 102,290,312 参数，与 PET 的 102,290,312 **完全相同**，两者都是完整的 `BertForMaskedLM`（对照文本分类任务的 BERT + `Linear(768→8)` 是 109,488,392 参数）。所以本次 p-tuning **没有节省任何可训练参数**。

**（3）损失。** `rdrop_coef=0.0`（`train.sh` 未传），因此走 `else` 分支，只算一次 `mlm_loss`，与 PET 的损失完全同构。

### 2.3 两者的范式差异（本报告的核心对比）

| 维度 | PET | p-tuning |
|---|---|---|
| 提示形态 | **离散、人工**：`这是一条[MASK][MASK]评论：{textA}。` | **连续、可学习**：15 个 `[unused]` 的嵌入 |
| `[MASK]` 位置 | 模板中间（词与词之间，语义通顺） | `[CLS]` 之后（无对应自然语言位置） |
| 人工先验 | 强：模板措辞 + verbalizer 标签词都是人写的 | 弱：只规定"插 15 个 token、`[MASK]` 紧跟 `[CLS]`" |
| 需要 verbalizer | **需要**（标签 → 标签词的显式映射） | 需要（同目录 `verbalizer.txt`，用于把 `[MASK]` 的 argmax 映射回类别） |
| 可训练参数 | 全部（102.29 M） | 全部（102.29 M），提示即其中 15 行嵌入 |
| `max_seq_len` | 256（模板 + 正文更长） | 128（15 个提示 token 已占 15 位） |
| 训练量 | 200 epoch（配置 1,600 step，实跑 160 step） | 20 epoch（配置 160 step） |
| 正则 | 配置了 `rdrop 5e-2` 但**未生效** | 未配置 |

### 2.4 一个容易被忽视的实现差异：评测映射路径

- **PET** 的 `evaluate_model` 先把金标签的 token id 还原成文字（`''.join(convert_ids_to_tokens(t))`），预测则走 `verbalizer.batch_find_main_label(predictions)` 做「子标签 → 主标签」映射，**再用 `ClassEvaluator` 按字符串标签计算指标**。
- **p-tuning** 的评测直接 `convert_logits_to_ids(logits, mask_positions)` 取 argmax token，**没有经过 verbalizer 的映射**（`evaluate_model` 里 `verbalizer` 参数被接收但未用于预测路径）。

这个差异意味着：**只有当模型对两个 `[MASK]` 位置都预测出与标签词完全一致的 token 时，p-tuning 才算对**（例如「酒店」必须两位都猜中）。这会让 p-tuning 的指标相对更"硬"。不过在本实验里，两个任务的 `verbalizer` 都是 1 对 1 映射，所以这条差异的实际影响有限。

---

## 3. 实验结果

### 3.1 p-tuning：唯一保留逐类指标的原始打印

`p-tuning/logs/comment_classify/屏幕截图 2026-09-22 181302.png` 的逐字记录（两段，第一段是 step 20 之前的中间评测，第二段是最终评测）：

```
第一段（历史 best 更新到 0.62 的那次）：
best F1 performence has been updated: 0.00000 --> 0.62000
Each Class Metrics are: {'书籍': {'precision': 0.95, 'recall': 0.66, 'f1': 0.78},
 '平板': {'precision': 0.3, 'recall': 0.88, 'f1': 0.44}, '手机': {'precision': 0.4, 'recall': 0.31, 'f1': 0.35},
 '水果': {'precision': 0.96, 'recall': 0.55, 'f1': 0.7}, '洗浴': {'precision': 0.84, 'recall': 0.54, 'f1': 0.65},
 '电器': {'precision': 0, 'recall': 0, 'f1': 0}, '电脑': {'precision': 0, 'recall': 0, 'f1': 0},
 '蒙牛': {'precision': 1.0, 'recall': 0.15, 'f1': 0.27}, '衣服': {'precision': 0.97, 'recall': 0.54, 'f1': 0.7},
 '酒店': {'precision': 0.91, 'recall': 0.91, 'f1': 0.91}}
global step 15, epoch: 1, loss: 1.11805, speed: 12.54 step/s
global step 20, epoch: 2, loss: 0.86735, speed: 13.06 step/s
C:\ProgramData\Anaconda3\envs\transformer\Lib\site-packages\sklearn\metrics\_classification.py:1879: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.

第二段（最终评测）：
Evaluation precision: 0.75000, recall: 0.65000, F1: 0.64000
best F1 performence has been updated: 0.62000 --> 0.64000
Each Class Metrics are: {'书籍': {'precision': 0.96, 'recall': 0.83, 'f1': 0.89},
 '平板': {'precision': 0.55, 'recall': 0.4, 'f1': 0.47}, '手机': {'precision': 0.98, 'recall': 0.67, 'f1': 0.8},
 '水果': {'precision': 0.65, 'recall': 0.78, 'f1': 0.78}, '洗浴': {'precision': 0.86, 'recall': 0.74, 'f1': 0.74},
 '电器': {'precision': 0, 'recall': 0, 'f1': 0}, '电脑': {'precision': 0.5, 'recall': 0.05, 'f1': 0.08},
 '蒙牛': {'precision': 1.0, 'recall': 1.0, 'f1': 0.56}, '衣服': {'precision': 0.0, 'recall': 0.38, 'f1': 0.04},
 '酒店': {'precision': 1.0, 'recall': 0.89, 'f1': 0.94}}
```

> 数值来源：**终端截图原始打印**。

**最终评测的总体指标（权威口径）**：

| 指标 | 数值 | 来源 |
|---|---|---|
| p-tuning precision | **0.75000** | 终端截图原始打印 |
| p-tuning recall | **0.65000** | 终端截图原始打印 |
| p-tuning F1（best） | **0.64000** | 终端截图原始打印 |
| 曲线反解 max precision / recall / accuracy / F1 | 0.760 / 0.650 / 0.650 / **0.640** | 曲线图 `p-tuning/BERT.png` 反解（**由训练曲线图的刻度标定反解，精度约 ±0.002**） |

> 曲线反解的 max F1 = 0.640 与截图的 0.64000 **互相印证**（`transformers_结果提取.md` 的 D 表已记录这一致性）。

**逐类指标（最终评测）**——本报告将截图中可读出的数值原样列出：

| 类别 | precision | recall | 打印的 f1 | **按 P/R 重算的 f1 = 2PR/(P+R)** |
|---|---|---|---|---|
| 书籍 | 0.96 | 0.83 | 0.89 | 0.890 |
| 平板 | 0.55 | 0.40 | 0.47 | 0.463 |
| 手机 | 0.98 | 0.67 | 0.80 | 0.796 |
| 水果 | 0.65 | 0.78 | 0.78 | 0.709 |
| 洗浴 | 0.86 | 0.74 | 0.74 | 0.796 |
| **电器** | **0** | **0** | **0** | 0（训练集中无此类的任何样本） |
| 电脑 | 0.50 | 0.05 | 0.08 | 0.091 |
| 蒙牛 | 1.0 | 1.0 | 0.56 | **1.000** |
| 衣服 | 0.0 | 0.38 | 0.04 | 0（P+R>0 但打印 0） |
| 酒店 | 1.0 | 0.89 | 0.94 | 0.942 |

**必须如实指出两处不一致**（不做修饰）：

1. **多数类别的 P/R/F1 满足 `f1 = 2PR/(P+R)`**（书籍、平板、手机、电脑、酒店五类吻合到 ±0.01），**但 `蒙牛`、`衣服`、`水果`、`洗浴` 四类明显不满足**。例如 `蒙牛` 打印 P=1.0、R=1.0、F1=0.56（按公式应为 1.00）；`衣服` 打印 P=0.0、R=0.38、F1=0.04（按公式应为 0）。
2. **按本报告能读到的 dev 划分（10 类）复算加权平均，无法复现终端打印的 0.75000 / 0.65000 / 0.64000**：以 p-tuning dev 的类别支持数（书籍 29、平板 67、手机 13、水果 78、洗浴 67、电器 2、电脑 22、蒙牛 13、衣服 70、酒店 56）对第二段 P/R 加权，得到 **weighted P ≈ 0.637、R ≈ 0.619**；取各类支持数分别对 F1 加权，得到 **weighted F1 ∈ [0.446, 0.578]（随 F1 取"打印值"或"重算值"而变）**。三者都显著低于打印的 0.75 / 0.65 / 0.64。

由于截图明确显示 `Evaluation precision: 0.75000, recall: 0.65000, F1: 0.64000` 是程序自己 `print` 出来的、且 F1 = 0.64 已由曲线图独立印证，**本报告以打印值 0.75 / 0.65 / 0.64 为权威结果**，并建议把"逐类指标对不上加权值"列为需要复现核实的疑点（可能的来源包括：两次评测的样本划分不同、dev 文件与本次运行的文件版本不同、或截图并非同一次运行的两段）。**在没有终端完整日志的情况下，本报告不为这一差异给出因果结论。**

### 3.2 PET：曲线反解（无终端截图）

`PET/logs/comment_classify/BERT-PET.png` 的 4 个 eval 子图 x 轴覆盖 step 20 ~ 160（4 个评测点：40 / 80 / 120 / 160）：

| 指标 | 曲线最大值 | 曲线末端（step 160） |
|---|---|---|
| eval/accuracy | max **0.769** | 0.769 |
| eval/precision | max **0.799** | 0.799 |
| eval/recall | max **0.769** | 0.769 |
| eval/f1 | max **0.749** | **0.747** |

> 数值来源：曲线图 `BERT-PET.png` 像素反解。**由训练曲线图的刻度标定反解，精度约 ±0.002。**
> 四个评测点的粗略形状：accuracy 0.68 → 0.78 → 0.77 → 0.77（step 40 处出现峰值 0.78 后轻微回落）；F1 0.65 → **0.766（峰值）** → 0.748 → 0.749。
> `transformers_结果提取.md` 的 B 表记载 PET 的 F1 为「max 0.749 / 末端 0.747」，本报告采用这一记载；同时提示曲线形状显示**真正的峰值出现在 step 40 附近（F1 ≈ 0.766）**，`model_best` 应当对应那一轮。

同期 `train/train_loss` 从约 2.4 快速降到 step 40 的约 0.5，此后到 step 160 缓降到约 0.18（**epoch 只从 5 走到 20**）。

### 3.3 p-tuning 的训练进程与损失

截图中的训练打印：

```
global step 15, epoch: 1, loss: 1.11805, speed: 12.54 step/s
global step 20, epoch: 2, loss: 0.86735, speed: 13.06 step/s
```

> 数值来源：**终端截图原始打印**。

loss 从 step 15 的 1.11805 降到 step 20 的 0.86735（仍是全程累计平均），速度约 **12.5 ~ 13.1 step/s**。注意这里的 `epoch` 计数与 `train.sh` 的 `num_train_epochs=20` 对齐（8 step = 1 epoch），说明 p-tuning 的训练循环**确实跑到了配置的 20 epoch / 160 step**，只是评测只在 step 10 与 step 20 触发了两次（`model_10`、`model_20`）。

### 3.4 汇总：PET vs p-tuning vs 全量有监督 PointWise

| 方案 | 训练样本 | 训练量 | accuracy | precision | recall | F1 | 指标来源 |
|---|---|---|---|---|---|---|---|
| **PET**（硬模板 + verbalizer） | **61** | 200 epoch（实跑 160 step = 10%） | 0.769 | 0.799 | 0.769 | **0.747**（峰值 0.749） | 曲线反解（±0.002） |
| **p-tuning**（软提示，15 个） | **61** | 20 epoch（实跑 160 step = 100%） | 0.650 | 0.750 | 0.650 | **0.640** | 终端截图（0.64000） |
| PointWise（单塔全量微调，见报告 02） | 1,416 | 10 epoch（实跑 200 step = 11%） | 0.94 | 0.86 | 0.92 | **0.90** | 曲线反解（±0.002） |

> ⚠️ 三者的**评测集不同**（PET 589 条 / p-tuning 415 条 / PointWise 352 条）、**训练集与类别数也不同**（前两者训练集只有 8 类、dev 有 10 类；PointWise 用全量 1,416 条句对、2 分类）。因此上表只能读出**趋势**，不能做严格的两两比较。

---

## 4. 结果分析

### 4.1 为什么 PET（0.747）明显高于 p-tuning（0.640）

两者用的是**完全相同的 61 条训练数据、相同的骨干（bert-base-maskedLM，102.29 M）、相同的 MLM 损失**。差异只在「提示长什么样」，因此 0.107 的 F1 差距可以归因到提示设计本身：

1. **硬模板带来了真实语言先验。** `这是一条[MASK][MASK]评论：{textA}。` 是一句**通顺的中文**，而 `[MASK]` 处于"一条 ___ 评论"这个在预训练语料里高频出现的语境中。BERT 的 MLM 头在预训练阶段见过大量类似句式，因此**只需很少的梯度就能把「评论 → 类别词」这个映射激活**。p-tuning 的提示是 `[unused1..unused15] [CLS] [MASK][MASK] 正文 [SEP]`——15 个 `[unused]` token 在预训练时**从未出现过**（BERT 中文词表里 `[unused]` 的嵌入是随机初始化的），`[MASK]` 又紧跟 `[CLS]`、后面直接接正文，没有任何自然语言语境的支撑。
2. **p-tuning 的软提示需要从随机初始化开始学。** 15 × 768 = 11,520 个可学习浮点参数在 61 条样本、160 step 的预算下，**样本复杂度严重不足**——它必须同时学"提示长什么样"和"如何映射到标签"，而 PET 只需要学后者。
3. **PET 用的 `max_seq_len=256` vs p-tuning 的 128**，PET 能容纳更长的评论；虽然本语料评论不长，但模板本身占了约 10 个 token，这削弱了 p-tuning 的有效正文长度。
4. **共同点是两者都远低于 PointWise 的 0.90**（见 §4.4）。

一个有意思的旁证：**p-tuning 的 loss 在 step 15 时还是 1.11805、step 20 时 0.86735**，而 PET 在 step 40 时 loss 已降到约 0.5。**p-tuning 的优化难度（收敛慢）与它的低指标是同一件事的两面**。

### 4.2 逐类指标揭示的核心问题：小样本下的类别不均衡 + 类别覆盖缺失

这是本报告最重要的一段分析，数据全部来自 §3.1 的截图。把 p-tuning 最终评测的逐类指标与 dev 支持数放在一起：

| 类别 | train 条数 | dev 条数 | precision | recall | 打印 f1 | 解读 |
|---|---|---|---|---|---|---|
| 酒店 | 16 | 56 | 1.00 | 0.89 | **0.94** | **表现最好**：训练样本最多（16）之一，且"酒店"在评论中有较强的指示词 |
| 书籍 | 3 | 29 | 0.96 | 0.83 | 0.89 | 只有 3 条训练却表现很好，说明"书"字的词汇线索很强 |
| 手机 | **0** | 13 | 0.98 | 0.67 | 0.80 | **训练为零样本却拿到 0.80**——只能来自预训练的先验（"手机"两字与评论的词汇共现），不是学到的 |
| 水果 | 7 | 78 | 0.65 | 0.78 | 0.78 | 中规中矩 |
| 洗浴 | 6 | 67 | 0.86 | 0.74 | 0.74 | 中规中矩 |
| 平板 | 12 | 67 | 0.55 | 0.40 | 0.47 | 与"电脑"混淆（见下） |
| 电脑 | 2 | 22 | 0.50 | 0.05 | 0.08 | **几乎全漏**：2 条样本不足以学到，且被"平板"抢走 |
| 蒙牛 | 1 | 13 | 1.0 | 1.0 | 0.56 | 样本极少但"蒙牛"是强指示词 |
| **衣服** | 16 | 70 | 0.0 | 0.38 | 0.04 | **precision 为 0**：预测出的"衣服"全错 |
| **电器** | **0** | 2 | **0** | **0** | **0** | **dev 有 2 条、train 一条都没有，必然全错** |

三条结论：

**（1）`电器` 的 F1 = 0 是「训练集类别覆盖缺失」的必然结果，不是模型能力问题。** 训练集只有 8 类，`电器` 与 `手机` 都不在其中；`电器` 在 p-tuning dev 里只有 2 条支持，模型没有任何机会见到"电器"这个标签词被激活。这属于**数据集划分缺陷**，应在复现时修正（从更大的 dev 或额外语料中为缺失类别补充训练样本）。

**（2）`衣服` 的 precision = 0（70 条 dev 支持）说明这不是缺失类别的特例，而是"样本多但被系统性误判"。** `衣服` 训练有 16 条（与 `酒店` 并列最多），却 precision = 0，最可能的解释是**它与其它类别的评论共享大量词汇**（服装评论里大量出现"面料/尺码/洗过/包装"等词，与"洗浴"高度重叠），在 61 条样本、8 类的条件下模型无法建立稳定的决策边界。注意 `衣服` 打印的 R=0.38 与 F1=0.04 **在公式上不自洽**（§3.1 第 1 条），因此更稳妥的表述是：**「衣服」这一类在最终评测中 precision 必须视为 0 或接近 0（sklearn 也会打印 `UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples`）**。

**（3）`酒店 0.91 / 0.94`（两段打印）与 `电器 0`、`衣服 0` 的巨大落差，就是"小样本类别不均衡"的教科书式表现。** 在 61 条、8 类的训练集里，各类样本数从 1 到 16 相差 16 倍；模型必然把容量集中在样本多、区分度高的类别上（酒店、书籍、蒙牛），而牺牲长尾（电脑 2 条、平板 12 条被混淆、无样本的电器直接归零）。**所以评价 PET/p-tuning 时只看加权平均的 0.747 / 0.640 会掩盖真实的问题：模型实际上只在少数几个类上可用。**

### 4.3 PET 与 p-tuning 恰好都在训练极早期就达到最好成绩

两条曲线都指向同一个现象：**提示学习在小样本上是"秒收敛"的**。

- **PET**：F1 在 **step 40（约第 5 个 epoch）** 就到峰值（曲线读得约 0.766），之后 step 80 回落到 0.748、step 160 为 0.749；`train_loss` 在 step 40 已从 2.4 降到约 0.5。也就是说，**后面 120 个 step（占实跑量的 75%）几乎没有带来 F1 提升，反而略有下降**。
- **p-tuning**：第一次评测（step 10 之前）F1 = 0.62，第二次（step 20）F1 = 0.64，**10 个 step 只涨 0.02**，且 loss 从 1.12 只降到 0.87，仍处高位。

**这说明：在 61 条样本上，两种提示学习方法的瓶颈都不是"训练不够久"，而是"标注数据太少 + 类别覆盖不全"。** 这也解释了为什么 PET 用 200 epoch（1,600 step 计划量）而实际只跑了 160 step 就停——继续跑下去大概率只是过拟合 61 条样本。

### 4.4 为什么两者都低于全量数据训练的 PointWise（0.90）

这是 prompt learning 最经典的经验结论，本实验的数据与之吻合，四层原因：

**（1）监督信号量差 23 倍。** PET/p-tuning 用 61 条，PointWise 用 1,416 条**带标签的句对**。即便 prompt learning 的样本效率更高，23 倍的差距也不是范式能弥补的——prompt learning 的卖点本来就是"**在标注极少时**用更少的样本达到"可接受的"效果，而不是"超过全量微调"。

**（2）任务定义不同，难度也不同。** PointWise 是**句对二分类**（匹配/不匹配），输入里已经给出了候选类别描述，模型只需判断"这段评论是否在说这个类别"；PET/p-tuning 是**单句 10 分类**，必须在没有任何候选提示的情况下从 10 个类里选一个。后者的判别空间更大、更容易出错（尤其对 `电器/衣服` 这类词汇重叠的类别）。

**（3）类别覆盖与类别数不同。** PointWise 的 label 是二值的 0/1，天然没有"某一类零样本"的问题；PET/p-tuning 面对 10 类、其中 2 类训练时完全没见过，**这部分错误是数据缺陷而非方法缺陷**，但它已经被计入 0.640 / 0.747 这两个聚合数字里。

**（4）训练状态的公平性**：三者都没有跑完（PET 实跑 10%、PointWise 实跑 11%、p-tuning 名义跑完但只评测了 2 次），因此这三个数字都是"早期读数"。**但趋势方向是一致的**：61 条样本的提示学习拿不到全量微调的精度，这符合一般结论。

---

## 5. 踩坑与局限

1. **PET 的 R-Drop 配置未生效**（最隐蔽的一个坑）：`train.sh` 传了 `--rdrop_coef 5e-2`，但 `pet.py` 的训练循环里 `mlm_loss(...)` **没有接收该参数、也没有调用 `rdrop_loss.compute_kl_loss`**（`rdrop_loss` 对象构造后未被使用）。因此本次 PET 的结果**不能标称为 "PET + R-Drop"**。对照 `p_tuning.py` 有完整的 R-Drop 分支，说明这是 PET 那一份代码的遗漏。
2. **训练集的类别覆盖缺失**：`train.txt` 只有 8 类，而 dev 有 10 类——**`手机` 与 `电器` 训练样本数为 0**。`电器` 在 dev 中 F1 = 0 是必然结果；`手机` 的 0.80（p-tuning）只能归因于预训练先验。这是**数据划分层面的缺陷**，会直接污染所有聚合指标的解读。
3. **PET 没有终端截图**：`PET/logs/comment_classify/` 下只有 `BERT-PET.png`，所有 PET 数值只有曲线反解一个来源，无法做「截图 ↔ 反解」交叉验证（对比 SimCSE 与 p-tuning 都完成了这步验证）。同时 §3.2 显示 PET 的真实峰值（step 40 附近，F1≈0.766）与 B 表记载的 max 0.749 存在读数差异，也与缺少原始打印有关。
4. **PET 实跑量只有配置的 10%**：配置 200 epoch = 1,600 step，实跑 160 step、仅 4 次评测，`checkpoints/` 只到 `model_160`。而 `max_train_steps` 仍按 1,600 计算，学习率几乎没退火。
5. **p-tuning 实际没有节省参数**：它是「全参数微调 + 15 行可学习嵌入」，`model.safetensors` 与 PET 完全等大（102,290,312 参数）。**把它当作 "parameter-efficient tuning" 来汇报是不准确的**；真正的 P-Tuning v2 需要冻结主干，本仓库未实现该分支。
6. **p-tuning 的逐类指标与聚合指标对不上**（§3.1）：多数类满足 `f1 = 2PR/(P+R)`，但 `蒙牛`/`衣服`/`水果`/`洗浴` 不满足；且用本报告的 dev 划分复算加权值（P≈0.637 / R≈0.619）无法复现终端打印的 0.75 / 0.65 / 0.64。**在没有完整终端日志的前提下无法定位原因**，只能标记为待核实的疑点。
7. **评测只发生在极少数 step**：PET 4 次、p-tuning 2 次。曲线点的数量不足以判断过拟合拐点，也没法做早停。
8. **`max_seq_len` 不一致**：PET 256、p-tuning 128，构成未控制变量。
9. **单卡 8 GB + 设备配置残留**：`PET/train.sh` 写 `--device "cuda:1"`，本机只有一张卡，实际必然被改。
10. **两段 p-tuning 打印之间没有 `Evaluation` 行**：截图第一段出现了 `Each Class Metrics are: ...`，但**其对应的 `Evaluation precision/recall/F1` 行没有被截图包含**（只看到 `best F1 ... 0.00000 --> 0.62000`）。因此第一段那次评测的聚合值缺失，无法复算。
11. **`utils.py` 里的 `UndefinedMetricWarning`**：sklearn 明确警告「有类别没有任何预测样本，precision 被设为 0」。这既解释了 `电器` 的 P=0，也提示**加权平均会被这类 0 值稀释**，报告这类指标时应同时给出逐类结果。

---

## 6. 结论与改进建议

### 6.1 结论

- **PET（人工硬模板 `这是一条[MASK][MASK]评论：{textA}。` + 1:1 verbalizer）**：用 **61 条**训练样本、`bert-base-chinese`、batch 8 / max_len 256 / lr 5e-5 / 200 epoch（实跑 160 step，约为配置量的 10%）/ `valid_steps=40`，得到 accuracy 0.769、precision 0.799、recall 0.769、**F1 0.747**（曲线峰值 0.749，出现在 step 40 附近）。**注意：配置里的 `--rdrop_coef 5e-2` 在代码中未生效，本次实为纯 MLM 交叉熵。**
- **p-tuning（15 个 `[unused]` 连续提示，`[MASK]` 紧跟 `[CLS]`）**：同样的 **61 条**、batch 8 / max_len 128 / 20 epoch / `valid_steps=20`，**最终评测 precision 0.75000 / recall 0.65000 / F1 0.64000**（终端截图原始打印，与曲线反解的 max F1 = 0.640 互相印证）。
- **逐类指标（p-tuning 最终评测，终端截图原始打印）**显示小样本类别不均衡的典型形态：**`酒店` 的 F1 最高（precision 1.00 / recall 0.89，打印 f1 0.94），`电器` 的 precision/recall/F1 全为 0**——因为训练集只有 8 类，`电器`（dev 2 条）与 `手机`（dev 13 条）**训练样本数为 0**；`衣服` 虽有 16 条训练样本但 precision = 0（打印 f1 0.04）；`电脑` recall 仅 0.05。
- **PET 优于 p-tuning（0.747 vs 0.640）的合理解释**：PET 的硬模板是通顺中文且 `[MASK]` 落在预训练高频语境中，而 p-tuning 的 `[unused]` 嵌入是随机初始化、需要从零学起；在 61 条样本的预算下，后者同时要学"提示形态"和"标签映射"，样本复杂度不够。
- **两者都明显低于全量数据训练的 PointWise（F1 0.90）**，原因叠加了四点：监督信号量相差 23 倍（61 vs 1,416）、任务难度不同（10 类单句分类 vs 句对二分类）、类别覆盖缺失（10 类中 2 类零训练样本）、以及三者都没跑完（PET 10%、PointWise 11%）。**这符合"提示学习在小样本上划算、数据量充足时不如全量微调"的一般结论。**

### 6.2 改进建议

1. **先修数据划分（第一优先）**：为 `手机`、`电器` 补充训练样本（至少每类 5~10 条），否则这两个类别的 0 分永远无法归因于方法。同时把 train/dev 的类别集合对齐后再报告聚合指标。
2. **让 PET 的 R-Drop 真正生效**：在 `pet.py` 里照 `p_tuning.py` 补上 `if args.rdrop_coef > 0:` 分支（两次前向 + `ce_loss` 平均 + `kl_loss * rdrop_coef`），然后做 `rdrop_coef ∈ {0, 5e-2}` 的消融，才能知道 R-Drop 在 61 条样本上是否真有收益。
3. **把逐类指标与混淆矩阵落盘**：本任务的证据缺口（§3.1 的两处不一致、PET 无截图）根源都是"只 print 到终端"。建议在 `evaluate_model` 后追加写文件：`logs/comment_classify/class_metrics_step{N}.txt`，内容含逐类 P/R/F1、支持数、混淆矩阵。这既能自证，也能立刻定位 `衣服/平板/电脑` 的混淆对。
4. **补上评测次数**：PET 的 `valid_steps` 从 40 收到 10（1,600 step 下 160 次评测）、p-tuning 保持 20（160 step 下 8 次），确保能画出真正的收敛/过拟合曲线。
5. **把训练量调到可完成且与调度器一致**：PET 若每 epoch 只有 8 step，200 epoch 也仅 1,600 step，实跑 160 step 说明中途被终止——要么补跑完，要么把 `num_train_epochs` 设为 20 让 `max_train_steps` 与实跑量一致，避免"学习率只退火 10%"。
6. **控制 `max_seq_len` 变量**：把 p-tuning 也设为 256（或把 PET 降到 128）重跑一组，排除序列长度的影响。
7. **补一个「无提示」的对照基线**：用 61 条样本直接训练 `BertForSequenceClassification`（加 `Linear` 分类头），对比 F1。这才能定量回答"prompt learning 相对普通小样本微调到底有没有优势"，是当前最缺的一组对照。
8. **若要真正做参数高效微调**：把 `p_tuning.py` 改成冻结 BERT 全部参数、只训练 p-embedding（并对 p-embedding 单独设一个更大的学习率，如 1e-2 ~ 1e-3），才是 P-Tuning v2 语义下的"软提示"。
9. **修 `--device` 配置**：把 `PET/train.sh` 的 `cuda:1`、`SimCSE/train.sh` 的 `cuda:2` 统一为本机实际存在的 `cuda:0`。

---

## 7. 复现命令

以下命令逐字取自两个 `train.sh`（工作目录分别为 `transformers_tasks-main/prompt_tasks/PET` 与 `.../p-tuning`）。

**PET：**

```sh
python pet.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "BERT" \
    --verbalizer "data/comment_classify/verbalizer.txt" \
    --prompt_file "data/comment_classify/prompt.txt" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 40  \
    --logging_steps 5 \
    --num_train_epochs 200 \
    --max_label_len 2 \
    --rdrop_coef 5e-2 \
    --device "cuda:1"
```

**p-tuning：**

```sh
python p_tuning.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --verbalizer "data/comment_classify/verbalizer.txt" \
    --save_dir "checkpoints/comment_classify/" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "BERT" \
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 20  \
    --logging_steps 5 \
    --num_train_epochs 20 \
    --max_label_len 2 \
    --p_embedding_num 15 \
    --device "cuda:0"
```

> 注：PET 的 `--device "cuda:1"` 为脚本原值；本机只有一张 GPU，复现时请改为 `cuda:0`。
> 注：PET 的 `--rdrop_coef 5e-2` 在当前 `pet.py` 实现中不会被使用（见 §2.1 与 §5.1），如需真正启用请先补齐代码分支。

**依赖安装与推理（按 readme）：**

```sh
pip install -r ../../requirements.txt
python inference.py
```

**按 §6.2 建议的补跑版本**（PET：补齐 R-Drop + 打密评测 + 落盘逐类指标）：

```sh
# ① 先给 train.txt 补上「手机」「电器」两类样本，使 train/dev 类别集合一致
# ② 修好 pet.py 的 rdrop 分支后，重跑（评测间隔 10，评测 160 次）
python pet.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "BERT-PET" \
    --verbalizer "data/comment_classify/verbalizer.txt" \
    --prompt_file "data/comment_classify/prompt.txt" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 10 \
    --logging_steps 5 \
    --num_train_epochs 200 \
    --max_label_len 2 \
    --rdrop_coef 5e-2 \
    --device "cuda:0"
```

**预期产物**：`PET/logs/comment_classify/BERT-PET.png`（4 条 eval 曲线 + loss 曲线）、`PET/checkpoints/comment_classify/{model_10 … model_160, model_best}`；`p-tuning/logs/comment_classify/BERT.png` 与 `屏幕截图 2026-09-22 181302.png`、`p-tuning/checkpoints/comment_classify/{model_10, model_20, model_best}`。
