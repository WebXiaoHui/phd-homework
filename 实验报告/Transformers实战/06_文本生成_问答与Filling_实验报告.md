# Transformers 实战 06：文本生成——生成式问答（T5）与 Filling 数据增强实验报告

> 实验对象：`transformers_tasks-main/answer_generation/`（① 答案生成）与
> `transformers_tasks-main/data_augment/filling_model/`（② Filling / 数据增强）
> 数据来源：本报告全部数值取自 `实验报告/_原始数据/transformers_结果提取.md`
> （A 表配置、B 表曲线反解、C 表终端截图逐字记录、E 表口径说明）；未编造任何数值。
> ⚠️ **本目录没有 `日志.md`**：该部分的实验结果只以「训练曲线 PNG + 终端截图 + checkpoint 目录」的形式存在。

---

# 第一部分：生成式问答（answer_generation，T5）

## 1. 任务目标与实验设置

### 1.1 任务目标

通过输入「一段原文（context）+ 一个问题（question）」，输出该问题的答案；属于**生成式问答**（与 UIE 的抽取式问答相对），采用 T5 的 text-to-text 框架实现。

### 1.2 数据集

| 项目 | 内容 |
|---|---|
| 数据 | 百度开源问答数据集 **DuReaderQG**，`data/DuReaderQG/{train,dev}.json` |
| 规模 | **train 14,520 / dev 983**（数据文件口径，见下方注） |
| 单样本结构 | `{"context": 参考文章, "answer": 答案, "question": 问题, "id": int}`，每行一个 JSON |
| 输入模板 | `问题：{question}{[SEP]}原文：{context}` |
| 输出模板 | `答案：{answer}{[EOS]}` |

> **口径注（±1 条差异）**：本次逐行核对显示 `dev.json` 实际含 **984** 条记录（末条 `"id": 983`），`answer_generation/readme.md` 的示例日志也打印 `num_rows: 984`；数据文件给出的是 983。本报告统一采用数据文件口径 **983**，所有换算按近似处理。

### 1.3 模型与超参数

配置来自 `answer_generation/train.sh` 与源码 `train.py`：

| 参数 | 取值 |
|---|---|
| 预训练模型 | `uer/t5-base-chinese-cluecorpussmall` |
| 结构 | T5ForConditionalGeneration：`d_model=768`、encoder 12 层 / decoder 12 层、12 头、`vocab_size=21,228`、`n_positions=512`（`model_200/config.json`） |
| batch size | 32 |
| max_source_seq_len / max_target_seq_len | 256 / 32 |
| learning rate | 5e-5（`train.sh`） |
| num_train_epochs | 50 |
| logging_steps / valid_steps | 10 / 500 |
| 优化器 / warmup | AdamW + linear schedule，`warmup_ratio=0.06` |
| 保存目录 / 日志图 | `checkpoints/DuReaderQG` / `logs/DuReaderQG/T5-Base-Chinese.png` |

### 1.4 指标定义

评测实现见 `answer_generation/bleu_metrics.py`（移植自 PaddleNLP `paddlenlp/metrics/bleu.py`），对每条预测-参考对累加 1/2/3/4-gram 的匹配数，最后计算各阶精度并乘简洁惩罚（brevity penalty）：

$$\text{BLEU}=bp\cdot\exp\!\Big(\sum_{n=1}^{N} w_n\log p_n\Big),\quad w_n=1/N$$

`evaluate_model` 一次返回 **BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4** 四个值，`model_best` 以 BLEU-4 最大为准。

> **口径提示**：`evaluate_model` 直接把 `model.generate()` 的输出 token id 序列与 `batch['labels']` 的 token id 序列逐位比较，**没有先 decode、也没有跳过特殊 token/padding**（`labels` 的 padding 为 `-100`）。因此这里的 BLEU 与「decode 成字符串后用 sacreBLEU 计算」的标准口径**不可直接横比**，数值会被特殊符号与 padding 稀释。这一点在阅读 0.097/0.063/0.035/0.023 时必须记住。

---

## 2. 方法与实现要点

### 2.1 T5 的中文适配

- `uer/t5-base-chinese-cluecorpussmall` 是标准的 **seq2seq（encoder–decoder）** 架构（`T5ForConditionalGeneration`，encoder 12 层 + decoder 12 层），其预训练目标是 **span corruption（去噪）**：随机挖掉文本片段，让 encoder 读带哨兵符的残缺文本、decoder 还原被挖片段。这与本任务「读段落、回答一个问题」在形式上并不相同——下游任务要求模型既理解问题、又能在原文中定位答案片段——但 T5 的 text-to-text 统一框架允许把 QA 表达成「输入模板 → 输出答案」的序列生成任务，因此可以直接迁移。
- 该 checkpoint 使用 BERT 系中文 tokenizer，源码因此手动改写特殊符号：`tokenizer.eos_token = tokenizer.sep_token`、`tokenizer.bos_token = tokenizer.cls_token`（`train.py` 87–88 行），`convert_example` 也据此处理 `[CLS]`/`[SEP]`（见 §2.2）。

### 2.2 数据构造（`convert_example`）

- `decoder_input_ids = output_ids[:-2]`：因为 BERT tokenizer 会额外加 `[CLS]`/`[SEP]`，截掉尾部两位实现 right-shift（源码注释明确说明）；
- `labels = output_ids[1:-1]`，并用 `-100` 填充到 `max_target_seq_len`，使 padding 不参与 loss；
- encoder 输入截断/补齐到 `max_source_seq_len=256`。

### 2.3 训练与评测

- 每 `valid_steps=500` 步：保存 `model_{step}` → 在 dev 上 `model.generate()` → 计算 4 个 BLEU，写入曲线 → 若 BLEU-4 刷新则另存 `model_best`。
- `generate()` **没有传入 `max_length`/`num_beams`**，实际使用模型自带生成配置（`decoder_start_token_id=101`、`pad_token_id=0`），即默认（贪婪）解码。

---

## 3. 实验结果

### 3.1 指标

数值均来自训练曲线反解（**首次出现，由训练曲线图的刻度标定反解，精度约 ±0.002**）：

| 指标 | max | 末端 | 来源 |
|---|---|---|---|
| `eval/bleu-size-1` | **0.097** | 0.096 | 曲线图反解（±0.002） |
| `eval/bleu-size-2` | **0.063** | 0.063 | 曲线图反解（±0.002） |
| `eval/bleu-size-3` | **0.035** | 0.035 | 曲线图反解（±0.002） |
| `eval/bleu-size-4` | **0.023** | 0.023 | 曲线图反解（±0.002） |

### 3.2 产物清单（本次 `ls -la` 核对）

| 路径 | 内容 | 是否存在 |
|---|---|---|
| `answer_generation/logs/DuReaderQG/T5-Base-Chinese.png` | 训练曲线图 | ✅ 存在 |
| `answer_generation/checkpoints/DuReaderQG/` | `model_200, model_400, model_600, model_800, model_best` | ✅ 存在（5 个） |
| `model_1000` 及之后 | 计划 50 epoch，但无更后 checkpoint | ❌ 不存在 |

**由 checkpoint 反推训练量**：dev/train 按数据文件口径，`14,520 / 32 ≈ 454 step/epoch`，落盘到 `model_800` 即 **约 1.8 个 epoch**——距计划中的 50 个 epoch（约 22,700 step）仅完成约 3.5%。

### 3.3 数据文件中的终端记录

数据文件 C 表收录的 4 张终端截图分别来自文本分类、SimCSE、p-tuning、filling，**不包含答案生成**，因此本子任务没有 stdout 级别的原值可互证，其数值依据是 D 表验证过的读图方法论（该方法在 SimCSE、p-tuning 两处与截图完全吻合）。

> 注：`answer_generation/readme.md` 仅用一段训练日志片段示意「启动成功」，那是**上游仓库的示例输出**，并非本次运行的 stdout，本报告不予引用为本次结果。

---

## 4. 结果分析

### 4.1 指标形态：1-gram 到 4-gram 单调递减属正常

0.097 → 0.063 → 0.035 → 0.023 逐阶下降，是 BLEU 的固有形态：n 越大、n-gram 越稀疏，越容易被「一个字/一个词顺序不同」直接清零。因此**不能用 BLEU-4 的绝对值去判断模型「完全不会」**，更合理的读法是：模型已经学到了一定的词面相关性（BLEU-1 近 0.1），但 4-gram 级别的精确重合很低。

### 4.2 BLEU-4 = 0.023 意味着什么

- 字面含义：生成的答案与金标准答案在**四元组**层面的重合率约 2.3%；结合 BLEU-1 约 9.7%，可以判断模型**能生成通顺、主题相关的中文答案片段，但内容的精确匹配度很低**——即「会说人话，但经常答不到点上」。
- 为什么中文生成任务的 BLEU 天然偏低：
  1. 中文答案通常很短（几个字到十几个字），n-gram 样本极少，**一个字的差异就让高阶 n-gram 全部失配**；
  2. 同一语义有多种合法表述（同义词、语序、量词），BLEU 只认字面重合，对同义改写极其苛刻；
  3. 本实现的 BLEU 还在 token id 序列上计算（含特殊符号/padding，见 §1.4 口径提示），进一步压低了读数。
- 除指标因素外的训练因素：
  1. **严重欠训练**：落盘仅到 `model_800`（约 1.8 epoch），而计划是 50 epoch；
  2. **模型规模**：T5-base 级别（`d_model=768`，12+12 层），不是 large/xxl；
  3. **评测集小**：dev 仅约 983 条，单条答案长度受 `max_target_seq_len=32`（含 `答案：` 前缀与特殊符）限制，长答案被截断；
  4. `generate()` 未启用 beam search，贪婪解码对生成质量不利。

### 4.3 末端 ≈ max 说明什么

四个 BLEU 的末端值都几乎等于最大值，只能说明**在已完成的约 800 step 内指标没有明显回退**，并不代表模型已经收敛——它只是一段很短训练里的局部平台。真正判断收敛需要跑满计划轮数。

---

# 第二部分：Filling 模型 / 数据增强（data_augment/filling_model，T5）

## 1. 任务目标与实验设置

### 1.1 任务目标

Filling 模型是 **Mask Then Fill 数据增强策略**的核心组件：对于「关键信息段 + 非关键信息段」构成的句子，随机 `[MASK]` 掉一部分**非关键片段**，再用生成模型把它填回来，从而在保持关键信息（实体/关系）不变的前提下生成新的表述，用于扩充 UIE 信息抽取的训练正例、提升 recall（`data_augment/filling_model/readme.md`、`UIE/readme.md` §5.2）。

示例（来自 readme）：

```text
原句：  大年三十 我从 北京 的大兴机场 飞回 了 成都。
挖空：  大年三十 我从 北京 [MASK] 飞回 了 成都。
填回：  大年三十 我从 北京 首都机场作为起点，飞回 了 成都。
```

### 1.2 数据集

- 由 `parse_data.py` 从 `data/dataset_text.txt`（DuIE 文本）自动构造：
  - `MIN_MASK_LEN_RATIO = 0.1`、`MAX_MASK_LEN_RATIO = 0.5`（按 jieba 词粒度随机决定被挖片段长度）、`RANDOM_MASK_PER_SAMPLE = 2`（每句挖 2 次）；
  - 样本格式：`"{带[MASK]的文本}"中[MASK]位置的文本是：\t{被挖掉的原文}`；
  - 按 9:1 切分 train/dev。
- 规模（本次 `wc -l` 核对，与数据文件 A 表一致）：**train.tsv 350,134 行 / dev.tsv 38,904 行**。

### 1.3 模型与超参数

配置来自 `data_augment/filling_model/train.sh` 与源码：

| 参数 | 取值 |
|---|---|
| 预训练模型 | `uer/t5-base-chinese-cluecorpussmall` |
| batch size | 128 |
| max_source_seq_len / max_target_seq_len | 128 / 32 |
| learning rate（配置值） | 1e-4 |
| learning rate（实际生效值） | **5e-5**（源码 `optimizer` 硬编码，见 §5 第 1 条） |
| num_train_epochs | 20 |
| logging_steps / valid_steps | 50 / 500 |
| 保存目录 / 日志目录 | `checkpoints/t5` / `logs` |

### 1.4 指标定义

与答案生成完全一致：BLEU-1/2/3/4（`bleu_metrics.py`，同一份实现，按被挖片段与生成片段比较）。

---

## 2. 方法与实现要点

- 训练框架与 `answer_generation/train.py` 基本同构：同一份 `utils.convert_example`、同一份 `bleu_metrics.py`、同一套 `iTrainingLogger` 曲线绘制逻辑，差别在数据模板与超参。
- 输出模板同样为 `答案：{answer}{[EOS]}`，输入为 `"{masked_text}"中[MASK]位置的文本是：`。
- 训练完成后，`UIE/web_da.py` 把 filling 模型路径填入 `filling_model_path`，即可在标注/增强平台上批量生成增强样本（readme §5.2）。
- `inference.py` 提供单条 masked text 的填词演示。

---

## 3. 实验结果

### 3.1 未取到可引用的评测指标

**本子任务没有可引用的指标数值。** 依据如下（均来自数据文件 C 表第 4 条与本次 `ls` 核对）：

| 证据 | 实际情况 |
|---|---|
| 曲线 PNG | `logs/` 下**没有** BLEU 曲线图；`assets/T5-Base-Chinese.png` 是**上游仓库自带示例图**，不是本次产物 |
| 终端截图 | `logs/屏幕截图 2026-09-22 164846.png` 是**训练过程的终端记录，不含任何评测指标行**（没有 `Evaluation bleu4` 之类的输出） |
| checkpoint | 仅 `checkpoints/t5/model_200` 一个目录 |

因此本子任务只能报告「训练曾跑到 200 step」，**不得写成任何 BLEU 数值**。

### 3.2 为什么 200 step 必然没有评测结果

`train.sh` 配置 `valid_steps=500`，即**第一次评测发生在 step 500**；而落盘只到 `model_200`。也就是说，运行在触发第一次 BLEU 评测之前就已经停止，日志里**不可能出现评测行**——这与「指标为 0」是两回事。

按 `350,134 / 128 ≈ 2,735 step/epoch` 估算，200 step 仅相当于 **约 0.07 个 epoch（不到 8%）**，训练几乎还没开始。

---

## 4. 结果分析

### 4.1 答案生成：BLEU 偏低的归因链

| 层次 | 具体因素 | 证据 |
|---|---|---|
| 训练量 | 仅约 1.8 epoch / 800 step，计划 50 epoch | `checkpoints/DuReaderQG` 落到 `model_800` |
| 模型 | T5-base 规模，非 large/xxl | `config.json`：d_model 768、12+12 层 |
| 数据 | dev 仅约 983 条；答案短、`max_target_seq_len=32` 截断 | A 表 + `convert_example` |
| 指标 | 中文生成 BLEU 天然偏低；本实现还在 token id 上算、含特殊符 | `bleu_metrics.py` + `evaluate_model` |
| 解码 | 贪婪解码，无 beam search | `generate()` 未传生成参数 |

### 4.2 Filling：为什么没有结果

- 训练**未跑到第一次评测点**（`valid_steps=500` vs 落盘 `model_200`）；
- `logs/` 下既无曲线 PNG，也没有含评测行的截图；
- 因此该子任务的结论是「**已实现、已启动、未产出可引用指标**」，而非「效果为 0」。

### 4.3 两个 T5 任务的对比意义

同样是 `uer/t5-base-chinese-cluecorpussmall`，答案生成任务在约 800 step 后 BLEU-4 达 0.023，而 filling 连第一次评测都没到。这说明**两次运行的实际训练量差异很大**（800 vs 200 step），报告在横向比较时必须先对齐训练量，否则容易把「没跑够」误判成「任务更难」。

---

## 5. 踩坑与局限

1. **命令行 `--learning_rate` 在 T5 脚本中不生效（重要）**：`answer_generation/train.py`（第 116 行）与 `data_augment/filling_model/train.py`（第 144 行）都写死了 `torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)`，并未使用 `args.learning_rate`。前者 `train.sh` 恰好也传 5e-5，看不出问题；**后者 `train.sh` 传的是 1e-4，实际训练却是 5e-5**。因此 A 表记录的「filling lr 1e-4」只是配置意图，真实生效值为 5e-5，复现时需特别注意。
2. **BLEU 口径非标准**：在 token id 上匹配、含特殊符与 padding、未 decode，数值不可与 sacreBLEU 直接比较；同时 `evaluate_model` 用 `batch['labels']` 作参考，`-100` padding 也会进入 n-gram 统计。
3. **`generate()` 未设生成长度/束搜索**：依赖默认生成配置，答案可能过早截断，直接压低 BLEU。
4. **`valid_steps=500` 与实际落盘 `model_200` 不一致**（filling），说明运行在评测前中断；答案生成同样只跑到 `model_800` 就停止，未按 50 epoch 完成。
5. **评测成本高**：每 500 step 要在整个 dev 上 `generate()` 一遍（答案生成约 983 条、filling 约 38,904 条），filling 的一次评测尤其昂贵，这可能是其未能推进到 500 step 的原因之一。
6. **没有独立测试集**：`model_best` 按 dev 的 BLEU-4 挑选，报告与选模共用 dev，存在选择偏差。
7. **数据文件与磁盘行数相差 1 条**（DuReaderQG dev 983 vs 984，见 §1.2 口径注）。
8. **本目录没有 `日志.md`**，stdout 未落盘，两个 T5 任务都只能靠 PNG/截图/checkpoint 反推训练进度。

---

## 6. 结论与改进建议

### 6.1 结论

- **答案生成（DuReaderQG）已跑通但严重欠训练**：BLEU-1/2/3/4 的 max 分别为 0.097 / 0.063 / 0.035 / 0.023（曲线反解，±0.002），末端与最大值几乎相同；只完成约 1.8 个 epoch（计划 50）。BLEU-4 = 0.023 表示模型能生成通顺答案，但内容精确匹配度很低，属于「欠训练 + 中文 BLEU 特性 + 口径偏严」共同作用的结果。
- **Filling 模型未取到可引用指标**：只有一张不含评测行的终端截图，`logs/` 下无曲线图，checkpoint 只有 `model_200`；它在 `valid_steps=500` 的第一次评测之前就已停止。

### 6.2 具体可执行的改进建议

**答案生成**

| 序号 | 建议 | 预期作用 |
|---|---|---|
| 1 | 把训练跑满（至少 5~10 epoch，或 22,700 step 的完整计划） | 直接验证 BLEU 是否仍随训练量上升 |
| 2 | 修复 `optimizer` 硬编码 lr 的问题，改用 `args.learning_rate` | 让 `train.sh` 的超参真正生效，便于做消融 |
| 3 | 评测改为 `decode(skip_special_tokens=True)` 后再算 BLEU，并补充 ROUGE-L / EM / F1 | 得到可与文献横比的标准指标 |
| 4 | 生成加 `num_beams=4`、显式 `max_length`，并适当增大 `max_target_seq_len` | 减少截断、提升生成质量 |
| 5 | 扩大 dev 规模，并划出独立 test split | 降低方差与选模偏差 |
| 6 | 增加抽取式问答（UIE）作为对照基线 | 区分「生成式 QA 本身难」与「T5 欠训练」 |

**Filling / 数据增强**

| 序号 | 建议 | 预期作用 |
|---|---|---|
| 1 | 至少训练到触发第一次评测（≥500 step），最好完整跑 1 个 epoch（约 2,735 step）再看 BLEU-4 | 获得第一个可引用指标 |
| 2 | 先把 `optimizer` 的 lr 改为读取 `args.learning_rate`，再确认 1e-4 是否合适 | 保证超参与报告一致 |
| 3 | 视显存把 `batch_size` 从 128 下调，或把 dev 采样一部分用于训练中评测 | 降低评测成本，避免再次卡在首次评测前 |
| 4 | 训练完成后接 `UIE/web_da.py`（填入 `filling_model_path`）做端到端增强验证 | 验证增强数据对 UIE recall 的实际增益 |

---

## 7. 复现命令

### 7.1 答案生成

```sh
cd answer_generation

# 训练（即 train.sh 内容）
python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/DuReaderQG" \
    --train_path "data/DuReaderQG/train.json" \
    --dev_path "data/DuReaderQG/dev.json" \
    --img_log_dir "logs/DuReaderQG" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 32 \
    --max_source_seq_len 256 \
    --max_target_seq_len 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --valid_steps 500 \
    --device cuda:0

# 推理（加载 checkpoints/DuReaderQG/model_best）
python inference.py
```

### 7.2 Filling / 数据增强

```sh
cd data_augment/filling_model

# 第 1 步：由 dataset_text.txt 生成 train.tsv / dev.tsv（如已存在可跳过）
python parse_data.py

# 第 2 步：训练（即 train.sh 内容）；
#         注意：脚本内 optimizer 硬编码 lr=5e-5，--learning_rate 1e-4 实际不生效
python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/t5" \
    --train_path "data/train.tsv" \
    --dev_path "data/dev.tsv" \
    --img_log_dir "logs" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 128 \
    --max_source_seq_len 128 \
    --max_target_seq_len 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --logging_steps 50 \
    --valid_steps 500 \
    --device cuda:0

# 第 3 步：单条填词推理
python inference.py

# 第 4 步（可选）：把训练好的模型接入 UIE 数据增强平台
cd ../../UIE && streamlit run web_da.py --server.port 8904
# 并在 web_da.py 中把 filling_model_path 指向 data_augment/filling_model/checkpoints/t5/model_best
```

> 若要获得本报告缺失的 filling 指标，最小补跑目标是：**修改 lr 读取方式后，训练至少 500 step（触发第一次 `Evaluation bleu4`），并保留 stdout 到 `logs/*.log`**。
