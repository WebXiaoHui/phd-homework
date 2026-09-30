# 文本分类（BERT）实验报告

> 代码仓库：`transformers_tasks-main/text_classification`（`train.py` / `train.sh` / `class_metrics.py` / `iTrainingLogger.py`）
> 硬件：单卡 NVIDIA GeForce RTX 3070 Ti（8 GB）+ Windows 11
> **数据来源说明**：本目录**没有 `日志.md`**。本报告的全部数值只来自三类物证——
> ① 训练曲线 PNG：`text_classification/logs/comment_classify/BERT.png`；
> ② 终端截图：`text_classification/logs/屏幕截图 2026-09-22 165323.png`；
> ③ checkpoint 目录：`text_classification/checkpoints/comment_classify/{model_200, model_best}`。
> 曲线反解值与配置表的汇总见 `实验报告/_原始数据/transformers_结果提取.md`，本报告未补造任何数字。

---

## 1. 任务目标与实验设置

**任务目标**：在「商品评论 → 商品类别」的**单句多分类**任务上，微调中文预训练语言模型，作为本组实验中文本分类任务的基线（baseline）。它与后面三篇报告的文本匹配任务共用同一批评论语料，但建模方式完全不同：本任务是「一句话 → 一个类别」，文本匹配是「两句话 → 匹配/不匹配」。

**数据集**：`data/comment_classify/`，格式为 `label\t文本` 的 TSV（`datasets.load_dataset('text', ...)` 读取）。

| 划分 | 行数 | 类别数 | 类别分布 |
|---|---|---|---|
| train.txt | 400 | 8 | 0:22, 1:78, 2:67, 3:29, 4:70, 5:56, 6:13, 7:67 |
| dev.txt | 61 | 8 | 0:2, 1:7, 2:12, 3:3, 4:16, 5:16, 6:1, 7:6 |

标签 id 与类别名的映射见 `label_mapping.txt`：

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 类别 | 电脑 | 水果 | 平板 | 书籍 | 衣服 | 酒店 | 蒙牛 | 洗浴 |

两点必须提前注意：
1. **训练集只有 400 条、验证集只有 61 条**，8 类平均每类约 50 条训练样本、约 7.6 条验证样本，最少的类别（6=蒙牛）训练只有 13 条、验证只有 1 条。这是一个**极小规模**的多分类任务，验证指标本身的方差极大。
2. **类别不均衡**：训练集中最多类（1=水果，78 条）与最少类（6=蒙牛，13 条）样本数相差 6 倍。

**模型**：`bert-base-chinese` + `AutoModelForSequenceClassification(num_labels=8)`，即 BERT 编码器 + 一层 `Linear(768 → 8)` 分类头。从 checkpoint 的 safetensors 头部统计，该模型共 **109,488,392（约 109.49 M）个参数**。

**超参**（取自 `train.sh`，与 `_train.py` 的 argparse 默认值一致）：

| 参数 | 取值 |
|---|---|
| `--model` | `bert-base-chinese` |
| `--num_labels` | 8 |
| `--batch_size` | 16 |
| `--max_seq_len` | 128 |
| `--learning_rate` | 5e-5（未在 train.sh 中覆盖，取 train.py 默认值） |
| `--weight_decay` | 0.0（默认值） |
| `--warmup_ratio` | 0.06（默认值） |
| `--num_train_epochs` | 20 |
| `--valid_steps` | 50 |
| `--logging_steps` | 10 |
| `--use_class_weights` | False（默认值，即不做类别加权） |
| `--loss_func` | `cross_entropy`（默认值，未启用 FocalLoss） |
| `--device` | `cuda:0` |

**评测节奏**：`train.py` 的主循环里 `if global_step % args.valid_steps == 0:` 触发一次「保存 checkpoint + 全量 dev 评测 + 写图」。因为 train.txt 只有 400 条、batch_size=16，所以：

```
每 epoch 步数 = ceil(400 / 16) = 25 step
完整 20 epoch 应有 25 × 20 = 500 step，即每个 epoch 评测 2 次、合计应有 10 次评测
```

**指标定义**（`class_metrics.py` 的 `ClassEvaluator.compute`）：
- `accuracy` = `sklearn.metrics.accuracy_score`；
- `precision` / `recall` / `f1` = sklearn 同名函数，**`average='weighted'`**（按各类样本数加权的平均），保留 2 位小数；
- 另有逐类 `class_metrics`（每类的 P/R/F1，用混淆矩阵逐类计算）。
- 注意：`evaluate_model()` 里每次评测**没有调用 `metric.reset()`**（`reset()` 方法存在但训练循环中未使用），因此评测累加器跨评测持续累积。由于本次训练只评测了 1 次，这一点对结果没有影响，但复现时若把 `valid_steps` 调小、出现多次评测，则第 2 次之后的评测值会是「历史所有评测样本的累计值」而非当次快照——这是复现前**必须**修掉的一个坑（详见 §5）。

---

## 2. 方法与实现要点

### 2.1 数据侧

`utils.convert_example(tokenizer, max_seq_len)` 把 `label\t文本` 解析为 `input_ids / token_type_ids / attention_mask / labels`；`max_seq_len=128` 截断，pad 到 128（`padding='max_length'`）。训练用 `shuffle=True`，验证用 `shuffle=False`，均用 `default_data_collator` 动态成 batch。

### 2.2 模型与损失

- 编码器：BERT-base 中文（12 层 / hidden 768 / 12 头），输出 `pooler_output`（[CLS] 经 tanh 的池化向量），接 `Linear(768, 8)`。
- 损失：`torch.nn.CrossEntropyLoss()`。代码里同时提供了 `FocalLoss(alpha=0.25, gamma=2.0)` 与 `--use_class_weights`（按类频倒数做 loss 缩放、上限 `max_scale_ratio=10`）两条处理不均衡的分支，但**本次运行两者都没开**。

### 2.3 优化器与学习率

- `AdamW`，并对 `bias` 与 `LayerNorm.weight` 做 **no-decay** 分组。
- 线性 warmup + 线性衰减：`warm_steps = int(0.06 × max_train_steps)`，`max_train_steps = num_train_epochs × 每 epoch 步数 = 20 × 25 = 500`，即 warmup 30 step，随后线性衰减到 0。
- ⚠️ 这里存在一个结构性问题：`lr_scheduler.step()` 每个 batch 都调用一次，而 `max_train_steps=500` 是按「完整 20 epoch」算的。**如果训练提前中断（实际只跑到 step 200），学习率只衰减到约 40% 的位置就被硬停**，模型处在一个「既没跑完、学习率也没退火到底」的状态。

### 2.4 记录器

`iTrainingLogger.iSummaryWriter` 用 matplotlib（`seaborn-v0_8-darkgrid` 风格）把 `train/train_loss` 与 `eval/{accuracy,precision,recall,f1}` 画成 6 宫格 PNG，**只画线、不打 marker**。这一点很关键：某张子图如果只被写入过 1 个点，就画不出线，图中会是空白，但该点的值仍然决定了该子图的坐标轴范围。

---

## 3. 实验结果

### 3.1 评测指标：**这是一次单点快照，不是收敛曲线**

`checkpoints/comment_classify/` 下**只有 `model_200` 与 `model_best` 两个目录**，说明整个训练过程只触发了 1 次 `global_step % valid_steps == 0` 的评测（step 200）。打开 `BERT.png` 也能直接看到这一点：

- `train/train_loss` 子图是一条覆盖 x ∈ [25, 200] 的正常下降曲线；
- `eval/accuracy`、`eval/precision`、`eval/recall`、`eval/f1` 四个子图**都是空白的**（无线可画），且它们的 x 轴刻度是 `190.0, 192.5, …, 210.0`——这正是「唯一数据点的 x=200 触发 autoscale」的结果，是单点评测的直接物证。

因此下表四项指标是 **step 200 处的一次快照**，数值**由训练曲线图的刻度标定反解**得到（由训练曲线图的刻度标定反解，精度约 ±0.002）：

| 指标 | 类型 | 数值 |
|---|---|---|
| eval/accuracy | 单点（step 200） | **0.32** |
| eval/precision（weighted） | 单点（step 200） | **0.34** |
| eval/recall（weighted） | 单点（step 200） | **0.32** |
| eval/f1（weighted） | 单点（step 200） | **0.26** |

> 数值来源：曲线图 `BERT.png` 像素反解（单点子图取坐标轴中心插值，方法见 `_原始数据/solve_values.py`），
> 与 `_原始数据/transformers_结果提取.md` 的 B 表一致。

### 3.2 训练损失：模型明显还在学习中

`BERT.png` 左上子图的 `train/train_loss`（累计平均）呈单调下降，从约 2.07 降到约 1.87（x 轴 25→200）。更完整、更原始的证据是终端截图 `logs/屏幕截图 2026-09-22 165323.png`（逐字记录）：

```
global step 10, epoch: 1, loss: 1.97777, speed: 7.42 step/s
global step 20, epoch: 1, loss: 1.79483, speed: 8.93 step/s
global step 30, epoch: 2, loss: 1.56500, speed: 9.49 step/s
global step 40, epoch: 2, loss: 1.33791, speed: 8.91 step/s
global step 50, epoch: 2, loss: 1.16831, speed: 8.91 step/s
global step 60, epoch: 3, loss: 1.03924, speed: 9.58 step/s
global step 70, epoch: 3, loss: 0.92857, speed: 8.91 step/s
```

> 数值来源：**终端截图原始打印**。

**逐 step 的 loss 单调下降且斜率没有明显变缓**：1.98 → 1.79 → 1.57 → 1.34 → 1.17 → 1.04 → 0.93。注意这里的 `loss` 是 Python 端 `loss_avg = sum(loss_list)/len(loss_list)` 的**累计平均**（`loss_list` 从训练开始一直 append、从未清空），所以真实的瞬时 loss 比显示的下降更快——例如 step 70 显示 0.92857 是 step 1~70 的平均值。这进一步说明 step 70 时模型远未收敛。

### 3.3 训练速度与规模

截图给出的速度是 **7.42 ~ 9.58 step/s**（`speed = logging_steps / time_diff`，即 10 step 的吞吐）。按 8.91 step/s 中位数估算，500 step 的完整训练只需约 1 分钟；step 200 的这次评测大约对应 22 秒训练。

---

## 4. 结果分析

### 4.1 0.32 / 0.26 不能被解读为「BERT 不行」

这个结论必须建立在两个口径之上：

**（1）它是欠训练。** 本次评测发生在 step 200。按 25 step/epoch 换算，step 200 ≈ **第 8 个 epoch**（`200/25 = 8`），只完成了配置的 20 epoch 的 **40%**。而 §3.2 的终端截图显示，到 step 70（第 3 epoch）loss 仍以每 10 step 下降约 0.1~0.2 的速度单调下滑，没有任何收敛迹象。也就是说，**step 200 时学习率还在高位，模型仍在快速下降段**，此时取快照必然低估最终性能。

**（2）它是单点评测 + 极小验证集。** 整个训练只评测了一次，没有曲线可比、无法判断该点是峰值、低谷还是上升途中的普通一点；同时 dev 只有 61 条，8 类平均每类 7.6 条，最少的类只有 1 条。61 条样本上 1 个样本的预测差异就会带来约 1.6 个百分点的 accuracy 波动、约 1.6 个百分点的 weighted F1 波动（若该样本属于大类别则更小、小类别则更大）。**0.32 与真实期望误差很可能在同一个量级。**

**（3）同一份语料上，匹配任务的 PointWise 单塔（ERNIE-3.0）拿到 F1 0.90。** 两者任务定义、网络结构（单句分类 vs 句对匹配）、骨干模型（BERT vs ERNIE）和训练状态（8 epoch 中断 vs 同样只评测 1 次但 loss 已降到 0.19 附近）都不同，因此这组对比**不能**支撑「BERT 不如 ERNIE」的结论；能支撑的只有一句：**本次 BERT 分类的训练没有跑完，且缺少曲线支撑，其结果不具备可比性。**

### 4.2 一个反直觉之处：`precision 0.34 > accuracy 0.32 ≈ recall 0.32 > f1 0.26`

四项指标的关系值得单独解释，因为它排除了「模型退化为常数预测」这一最坏情况：

- 若模型把所有样本都预测成同一个（占优的）类，weighted precision 会接近 0，weighted recall 接近占优类占比（约 12/61 ≈ 0.20），三者不会出现 0.32/0.34/0.32 这种「P 略高于 R、F1 明显低于两者」的形态。
- F1 显著低于 P 和 R，是**类别粒度上表现极不均匀**的典型信号：少数类别上有若干预测正确（贡献了 precision/recall 的分子），而大多数类别既被漏检又被误报，这些类在 weighted 平均里按样本数合计后把 F1 拉下来。逐类 P/R/F1 本应由 `evaluate_model` 打印（`Each Class Metrics are: {...}`），但**本目录没有保留这次运行的终端日志**，因此无法给出逐类数字——这本身就是一个应该补上的留证缺口。
- 结合 §4.1 的欠训练判断，最合理的解释是：模型到 step 200 时**只学会了区分语料中最有「词汇特征」的少数类别**（例如「蒙牛 / 苹果」这类强指示词），其余类别还在随机附近。

### 4.3 均值口径的定量佐证

用「accuracy=0.32、worst-case 逐类 F1」的极端情形做一次粗算可以说明形态合理：若 8 类的逐类 F1 为 [0.31, 0.40, 0.30, 0, 0.30, 0.28, 0.40, 0]（仅作口径示意，非实测），按各类样本数（2,7,12,3,16,16,1,6）加权平均恰好约为 **0.26**，与实测的 weighted F1 一致；对应的 macro F1 则会低到约 0.25。这说明「weighted F1 = 0.26 而 weighted P/R 约 0.32~0.34」是可以由**若干类 F1 为 0（预测失败）＋ 少数类表现中等**构成的，与「欠训练 + 类别不均衡」的表现一致。

### 4.4 类别不均衡没有被任何机制缓解

训练集最多类与最少类相差 6 倍，`CrossEntropyLoss` 对每个样本等权，因此梯度会被大类别主导；`--use_class_weights` 和 `--focal_loss` 两个开关都存在但都没开。在 400 条训练数据、8 类的设置下，小类别（蒙牛 13 条、电脑 22 条）几乎不可能在 8 个 epoch 内学到稳定决策面。这是 0.32 accuracy 的另一个直接原因。

---

## 5. 踩坑与局限

1. **评测次数严重不足（最致命）**：`valid_steps=50` 配 `num_train_epochs=20` 本来只给 10 次评测（500/50），而实际只跑到 step 200 就停了，最终**只有 1 次评测**，四个 eval 子图全是空白。没有曲线就无法定位峰值、无法做早停、也无法回答「训练是否还在变好」。
2. **`metric` 未 `reset` 的隐患**：`text_classification/train.py` 的评测循环里没有调用 `metric.reset()`（对照 `PET/pet.py` 的 `evaluate_model` 里明确调用了 `metric.reset()`）。本次只有 1 次评测所以没暴露，但一旦把 `valid_steps` 调小以获取曲线，第 2 次起的所有指标都会变成「训练全程累计评测集」的值，指标会被人为平滑、失真。**补跑前必须先在 `evaluate_model` 里加 `metric.reset()`。**
3. **训练未跑完**：配置 20 epoch（500 step），实际停在 step 200（约 8 epoch），学习率只退火到约 40%。
4. **验证集太小**：dev 仅 61 条、最少类仅 1 条，单次评测的置信区间很宽；目前的 0.32/0.34/0.32/0.26 不足以做任何模型间比较。
5. **未处理类别不均衡**：`use_class_weights=False`、`loss_func='cross_entropy'`，6 倍类频差没有任何补偿。
6. **本任务无终端评测日志留存**：同批次的 SimCSE、p-tuning 都保留了终端截图，唯独文本分类的截图（`屏幕截图 2026-09-22 165323.png`）只截到 step 70 的训练 loss，没有截到 `Evaluation precision: ...` 与 `Each Class Metrics are: ...` 两行。**曲线值因此无法与终端打印做交叉验证**（对比之下 SimCSE 的 F1 与 spearman、p-tuning 的 F1 都完成了「截图 ↔ 反解」双向印证）。这是本次实验证据链上最薄弱的一环。
7. **单卡 8 GB 限制**：`train.sh` 指定 `cuda:0`；机器只有 1 张 RTX 3070 Ti（8 GB），`bert-base-chinese` + `batch 16 / max_len 128` 可以放下，但 `xla`/多卡并行不可用，实验吞吐受限（实测约 9 step/s）。
8. **`average='weighted'` 不是 macro-F1**：报告中的 F1 是加权平均，与不少论文/榜单汇报的 macro-F1 不可直接比较。8 类均衡时两者接近，本任务不均衡，两者会有可见差距（见 §4.3 的示意）。

---

## 6. 结论与改进建议

### 6.1 结论

- 在 `comment_classify`（400 train / 61 dev，8 类）上微调 `bert-base-chinese`，本次运行**只在 step 200 评测了一次**，得到 accuracy 0.32 / precision(weighted) 0.34 / recall(weighted) 0.32 / F1(weighted) 0.26。
- 该结果是**欠训练（约 8/20 epoch）＋ 单点评测 ＋ 极小验证集**共同作用下的快照；终端截图显示到 step 70（第 3 epoch）loss 仍从 1.98 单调降到 0.93，模型明显还在学习。**因此不能据此否定 BERT 在中文短文本分类上的能力，也不能据此与同语料的 PointWise（F1 0.90）做任何「模型优劣」的对比。**
- 现有产物（1 张空白 eval 子图的 PNG + 1 张只含 7 行 loss 的终端截图 + 2 个 checkpoint 目录）**不足以支撑对该配置的性能判断**；结论只能是「该配置在本机未完成训练」。

### 6.2 下一步怎么补跑（按优先级）

1. **先修 `metric.reset()`**：在 `text_classification/train.py` 的 `evaluate_model()` 开头（或 `metric.compute()` 之后）加上 `metric.reset()`，否则多次评测的结果不可用。
2. **把评测打密**：`--valid_steps 10`（500 step 共 50 次评测，约每半个 epoch 一次）或 `--valid_steps 25`（每 epoch 一次，共 20 次评测）。这样 `BERT.png` 的 eval 子图才会画出真正的曲线，才能定位峰值与过拟合拐点。
3. **跑满 20 epoch**：不要在中途 Ctrl-C，让 `lr_scheduler` 线性退火到 0。按实测 8.91 step/s，500 step 约 1 分钟，成本极低，没有理由不跑完。
4. **用 `model_best` 做最终评测**：`train.py` 已按 dev F1 保存 `model_best`，补跑后应新增一个独立的 `inference.py` 评测脚本，**加载 `checkpoints/comment_classify/model_best` 在 dev 上重算 P/R/F1**，而不是引用训练途中打印的值。注意 `inference.py` 的默认 `saved_model_path` 需要指向 `model_best`。
5. **把「Each Class Metrics」落到文件**：评测时把逐类 P/R/F1 与混淆矩阵写入文本文件（例如 `logs/comment_classify/class_metrics.txt`），不要只 `print` 到终端——本次实验的证据缺口正是由此产生的。
6. **处理类别不均衡**：开 `--use_class_weights`（并对 `max_scale_ratio` 做 5 / 10 的对比），或改用 `--loss_func focal_loss`，观察小类别（蒙牛 13 条、电脑 22 条）的逐类 F1 是否回升。
7. **补一个「同骨干」的对照**：把 `--model` 换成 `nghuyong/ernie-3.0-base-zh`、其余配置完全不变再跑一次，才能把「BERT vs ERNIE」与「分类 vs 匹配」两个变量解耦。
8. **验证集扩容**：`dev.txt` 只有 61 条，建议从 train 中分层留出（或者直接使用 PET/p-tuning 目录下更大的 `dev.txt`，589 / 415 条）以获得更稳定的估计。

---

## 7. 复现命令

以下命令逐字取自 `text_classification/train.sh`（工作目录为 `transformers_tasks-main/text_classification`）：

```sh
python train.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "BERT" \
    --num_labels 8 \
    --batch_size 16 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 20 \
    --device "cuda:0"
```

复现时需要在 `train.sh` 同级目录执行（脚本内的数据/日志路径都是相对路径），并先安装依赖：

```sh
pip install -r requirements.txt   # torch / transformers==4.22.1 / datasets==2.4.0 / evaluate==0.2.2 / matplotlib==3.6.0 / rich==12.5.1 / scikit-learn==1.1.2
```

**按 §6.2 建议的补跑版本**（改动两处：评测打密、并用独立脚本评测 `model_best`）：

```sh
# ① 补跑主训练：把评测间隔从 50 收到 10，跑满 20 epoch
python train.py \
    --model "bert-base-chinese" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "BERT" \
    --num_labels 8 \
    --batch_size 16 \
    --max_seq_len 128 \
    --valid_steps 10 \
    --logging_steps 10 \
    --num_train_epochs 20 \
    --device "cuda:0"

# ② 用保存下来的最优权重重新评测（inference.py 里的 saved_model_path 需指向 model_best）
python inference.py
```

**预期产物**：`logs/comment_classify/BERT.png`（本次为 6 宫格中 4 格空白的单点图，补跑后应为 4 条完整曲线）、`checkpoints/comment_classify/model_10 … model_500` 与 `model_best`。
