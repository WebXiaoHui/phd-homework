# Transformers 实战 05：RLHF 第一阶段——基于偏好排序的奖励模型（Reward Model）实验报告

> 实验对象：`transformers_tasks-main/RLHF/`
> 数据来源：本报告全部数值取自 `实验报告/_原始数据/transformers_结果提取.md`
> （A 表配置、B 表曲线反解、C 表终端截图逐字记录、D 表交叉验证、E 表口径说明），
> 未编造任何数值。凡超出该文件的内容，均在正文中标注其来源（`readme.md`、源码或本次 `ls` 核对）。
> ⚠️ **本目录没有 `日志.md`**：该部分的实验结果只以「训练曲线 PNG + 终端截图 + checkpoint 目录」的形式存在。

---

## 1. 任务目标与实验设置

### 1.1 任务背景与目标

RLHF（Reinforcement Learning from Human Feedback）在工程上通常拆成两个阶段：

1. **奖励模型（Reward Model, RM）阶段**：用一个带打分头的编码器学习人类偏好，使「更受偏好的回答」得分高于「较差回答」；
2. **PPO 阶段**：以 RM（或代理奖励）作为 reward，用强化学习继续优化生成模型。

`RLHF/readme.md` 一共给出 4 个示例：（1）基于情感识别模型的正向评论生成（No Human Reward）、（2）基于人工打分的评论生成（With Human Reward）、（3）**基于人工排序序列训练 Reward Model**、（4）RankList 人工标注平台。

**本作业只运行了其中第 3 个示例（奖励模型训练），PPO 阶段没有运行**（证据见 §3.2、§4.5）。因此本报告描述的是「RLHF 链路只走完第一阶段」。

### 1.2 数据集

| 项目 | 内容 |
|---|---|
| 数据目录 | `RLHF/data/reward_datasets/sentiment_analysis/` |
| 文件 | `train.tsv`、`dev.tsv` |
| 样本规模 | **train 12,327 / dev 3,081**（数据文件口径，见下方注） |
| 单样本结构 | 一行即一个 **rank list**：4 条评论按「越靠前越偏正向情绪」用 `\t` 分隔（本次核对：前 2,000 行均为 4 个字段） |
| 任务性质 | 偏好排序（pairwise / listwise ranking），不是普通二分类 |

> **口径注（±1 条差异）**：本次 `ls`/逐字节核对显示磁盘上 `train.tsv` / `dev.tsv` 分别含 **12,328 / 3,082** 条非空记录，与数据文件给出的 12,327 / 3,081 相差 1 条（属文件尾行 / `load_dataset` 统计口径差异）。本报告统一采用数据文件口径 **12,327 / 3,081**；后文所有「step/epoch」换算均按该口径的近似值处理。

数据示例（`readme.md`）：

```text
1.买过很多箱这个苹果了，一如既往的好，汁多味甜～	2.名不副实。	3.拿过来居然屏幕有划痕，顿时就不开心了	4.什么手机啊！一台充电很慢，信号不好！退了！又买一台竟然是次品。
```

### 1.3 模型与超参数

配置来自 `RLHF/train_reward_model.sh`，与数据文件 A 表一致：

| 参数 | 取值 |
|---|---|
| 预训练模型 | `nghuyong/ernie-3.0-base-zh` |
| Reward 头 | `nn.Linear(768, 1)` 接在 encoder 的 `pooler_output` 上（`RLHF/model.py`） |
| batch size | 32（每个 batch = 32 个 rank list） |
| max_seq_len | 128 |
| learning rate | 1e-5 |
| num_train_epochs | 10 |
| valid_steps | 50 |
| logging_steps | 10 |
| 优化器 / 调度器 | AdamW + linear schedule，`warmup_ratio = 0.0` |
| device | cuda:0 |
| 保存目录 | `checkpoints/reward_model/sentiment_analysis` |
| 日志图 | `logs/reward_model/sentiment_analysis/ERNIE Reward Model.png` |

### 1.4 指标定义

- **`eval/accuracy`（本次唯一评测指标）**：对 dev 中每一条 rank list 单独前向后得到 4 个得分，只有当这 4 个得分**按降序排列与标注顺序完全一致**时才计为正确，返回正确条数 / 总条数（源码 `evaluate_model`，`sorted(rank_rewards, reverse=True) == rank_rewards`）。注意这是**整条全序完全正确率**，不是 pairwise accuracy，也不是「最高分命中率」。
  - 对 4 元素列表，**随机基线 = 1/4! = 1/24 ≈ 0.0417**。
- **训练损失**：`compute_rank_list_loss`（`RLHF/model.py`），即 Bradley–Terry 风格的 pairwise ranking loss——对一条 rank list 内所有 $(i<j)$ 组合求 $\log\sigma(r_i-r_j)$ 的均值再取负：
  $$\mathcal{L}=-\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\log\sigma(r_i-r_j)$$
  4 条评论对应 $C_4^2=6$ 个偏好对。该损失在随机初始化、得分无差异时期望值约为 $-\log 2\approx-0.693$，越负表示偏好间距越大、学得越充分。

---

## 2. 方法与实现要点

### 2.1 Reward Model 结构

`RLHF/model.py` 定义了最简打分模型：共享 ERNIE-3.0-base 编码器，取 `pooler_output`（768 维），经一个线性层映射到标量 reward。没有额外 MLP，参数极少（新增 769 个参数）。

### 2.2 损失函数

`compute_rank_list_loss` 采用**双重循环遍历所有前项-后项组合**，把 `F.logsigmoid(rank_rewards[i] - rank_rewards[j])` 累加、除以对数、再取负。它不依赖 torchvision 式的 margin 超参，天然把「chosen 得分 > rejected 得分」作为优化目标。

### 2.3 训练循环

- 每个 batch 内先对 32 个 rank list 逐一取出 4 条文本，**逐条（batch=1）前向**收集 reward，再计算 listwise loss（源码 141–159 行）。因此 `--batch_size 32` 指的是「32 个 rank list」，单卡实际前向次数是 $32\times4=128$ 次/step。
- 优化器只对 `bias`/`LayerNorm.weight` 关闭 weight decay（`weight_decay` 默认 0.0）。
- 数据加载：`load_dataset('text', ...)` 后经 `convert_example` 分词；`train_dataloader` 的 `shuffle=False`（注意这是源码事实，见 §5）。

### 2.4 评测与保存

- 每 `valid_steps` 步：先 `torch.save(model, model_{step}/model.pt)` 并保存 tokenizer，再在 dev 上算 `eval/accuracy` 写入曲线，若刷新历史最优则另存 `model_best`。
- 所有结果通过 `iTrainingLogger.iSummaryWriter` 画成 PNG（本项目没有把 stdout 落盘为日志）。
- 环境变量 `HF_ENDPOINT=https://hf-mirror.com`（源码第 24 行）。

---

## 3. 实验结果

### 3.1 指标

| 指标 | 数值 | 来源 | 备注 |
|---|---|---|---|
| `eval/accuracy` | **max 0.666 / 末端 0.666**（由训练曲线图的刻度标定反解，精度约 ±0.002） | B 表曲线反解 | 偏好对二分类（全序正确）准确率；max 与末端几乎相同 |
| 训练规模 | **≥ 2,800 step** | `checkpoints/` 目录，落盘到 `model_2800` | 按 12,327/32 ≈ 385 step/epoch 估算，约 7.3 个 epoch |
| 随机基线 | 1/24 ≈ 0.0417 | 数学推导（4 元全序） | 用于判断 0.666 是否「有效」 |

### 3.2 产物清单（本次 `ls -la` 核对）

| 路径 | 内容 | 是否存在 |
|---|---|---|
| `RLHF/logs/reward_model/sentiment_analysis/ERNIE Reward Model.png` | 训练曲线图（loss + eval/accuracy） | ✅ 存在 |
| `RLHF/checkpoints/reward_model/sentiment_analysis/` | `model_200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800` + `model_best`，共 **15 个** | ✅ 存在 |
| `RLHF/checkpoints/ppo_sentiment_gpt/`（PPO 产物） | PPO 训练输出 | ❌ **不存在** |
| `RLHF/logs/PPO-Sentiment-Zh.png`（PPO 曲线） | PPO reward 曲线 | ❌ **不存在** |

> 说明：`RLHF/assets/PPO-Sentiment-Zh.png` 存在，但它是**上游仓库自带的示例图**（`readme.md` 中引用的插图），不是本次运行的产物，不能作为本次实验证据。

### 3.3 关于 checkpoint 数量的口径问题

`train_reward_model.sh` 配置 `valid_steps=50`，若严格按此执行应每 50 step 落盘一次；但磁盘上实际只保留了 **200 的整数倍**（14 个）加 `model_best`（共 15 个）。数据文件 A 表同样记录为「15 个，model_200~2800 + best」。二者不一致（详见 §5 第 1 条），本报告一律以落盘产物为准。

---

## 4. 结果分析

### 4.1 0.666 是否有意义：先看参照系

若按 4 元全序的随机基线 1/24 ≈ 0.0417 衡量，**0.666 约为随机水平的 16 倍**，说明奖励模型确实学到了「正向评论得分高于负向评论」的排序信号，RLHF 第一阶段在方向上是成立的。

### 4.2 为什么精度停在 0.666，而不是更高

需要强调：**0.666 这个数字本身被指标口径「压低」了**。原因是多方面的：

1. **指标过于严格**。`evaluate_model` 要求一条 rank list 的 4 个得分**完整降序**才算对；只要 6 个偏好对里有 1 对次序颠倒，整条就判错。若改为 pairwise accuracy 或「top-1 命中」，同一模型的数值会显著更高。因此 0.666 不能直接与二分类的 0.9 之类数字横比。
2. **偏好数据本身带噪**。情感强弱是连续、主观的量，而 rank list 被强制离散成 4 档；评论来自不同商品/场景，可能包含中性、反讽、混合情感，甚至存在「两条评论难分高下」的样本。这类样本构成不可消除的 Bayes 误差，会把准确率钉在某个上限附近。
3. **训练不充分**。学习率仅 `1e-5`，`warmup_ratio=0`，配合 linear schedule，训练后期 lr→0；规划 10 个 epoch（约 3,852 step），而落盘 checkpoint 到 2,800 step（约 7.3 epoch）。数据文件同时显示曲线 **max = 末端 = 0.666**，即指标在观测区间内已经进入平台，继续训下去短期收益有限。
4. **输入被截断**。`max_seq_len=128`，长评论被截断，可能正好丢掉关键情感词，导致本可区分的样本被打成平手。
5. **模型容量与数据量不匹配**。`ernie-3.0-base-zh` 是 base 规模，训练集仅 1.2 万余条偏好对，对细粒度情感排序而言偏小。
6. **训练顺序固定**。源码中 `train_dataloader` 使用 `shuffle=False`，样本顺序恒定，可能造成批次间分布偏置、影响收敛质量（见 §5 第 4 条）。

### 4.3 loss 视角的佐证

该 ranking loss 在「得分无差异」时期望约 $-0.693$（因 $\mathbb{E}[-\log\sigma(0)]=\log 2$）。本次运行没有把 loss 数值落盘（PNG 曲线以外的 stdout 已丢失），因此**无法引用本次 loss 曲线读数**；但从「max=末端=0.666、曲线后期走平」可以推断模型已进入一个受数据噪声限制的平台，而非仍在快速下降的阶段。

### 4.4 与终端截图的关系

`RLHF/logs/` 下**没有本次运行的终端截图**（只有曲线 PNG）。数据文件 C 表记录的 4 张截图分别来自文本分类、SimCSE、p-tuning、filling 四个子任务，**不包含 RLHF**。因此 0.666 无法像 SimCSE 那样与 stdout 互证，其可信度依据是 D 表的方法论验证：该读图方法在 SimCSE（F1 0.70541）、p-tuning（F1 0.64000）两处与终端截图完全吻合，故其余曲线值按同精度（约 ±0.002）采信。

> 注意：`RLHF/readme.md` 中给出的评测输出、loss 打印与推理打分示例，均来自**上游仓库的示例运行**，不是本次实验的结果，本报告不予引用为本次指标。

### 4.5 PPO 阶段为什么没有运行、为什么可以判定「未运行」

**判定依据（均为本次 `ls` 事实）**：

- `RLHF/checkpoints/` 下**只有 `reward_model/`**，没有任何 `ppo_sentiment_gpt/` 目录；
- `RLHF/logs/` 下**只有 `reward_model/`**，没有 PPO 曲线 `PPO-Sentiment-Zh.png`；
- PPO 脚本 `RLHF/ppo_sentiment_example.py` 的 `save_dir` 配置为 `checkpoints/ppo_sentiment_gpt`，该路径不存在。

**未运行的合理原因**：

1. **算力成本高**。`ppo_sentiment_example.py` 的配置为 `steps=20000`、`batch_size=128`、`gen_len=16`、`ppo_epochs=4`，需要同时加载 GPT-2（`uer/gpt2-chinese-cluecorpussmall`）策略模型与参考模型、以及情感分类模型（`uer/roberta-base-finetuned-jd-binary-chinese`），显存与时长都远超本机单卡预算；
2. **第二阶段与第一阶段在代码上并未真正串接**。该 PPO 示例的 reward 来自**情感分类 pipeline** 的判分（`sentiment_pipe`），而不是本次训练出的 reward model。也就是说，即使跑通 PPO，它消费的也不是本报告 §3 的 RM 权重，二者属于仓库中两条独立的示例链路；
3. **输入分布不一致**。RM 训练数据是「4 条评论的偏好排序」，PPO 示例的输入是「4 个固定 prompt 的续写」，RM 与 PPO 之间缺少一层打分用的偏好数据桥接。

因此「RLHF 只完成第一阶段」既是**算力现实**，也是**代码链路现状**共同决定的结果。

---

## 5. 踩坑与局限

1. **`valid_steps=50` 与实际落盘 checkpoint 不一致**：落盘的全是 200 的整数倍（14 个 + best）。要么实际运行时改过参数、要么中间产物被清理，仓库中没有记录可查（无 `日志.md`、无 stdout），只能以落盘产物为准。
2. **打印文案与语义不符**：源码 181–184 行在刷新最优时打印 `best F1 performence has been updated`，但实际写入曲线的指标是 `eval/accuracy`，容易让读者误以为存在 F1 指标。
3. **评测极慢**：`evaluate_model` 对 dev 每条 rank list 的每条文本做一次 batch=1 前向，dev 3,081 条即约 12,324 次前向，训练中途频繁触发会很耗时；这也可能是 checkpoint 稀疏/训练未能跑满的一个诱因。
4. **`train_dataloader` 使用 `shuffle=False`**（源码 109 行），偏好数据顺序固定，不利于泛化。
5. **诊断指标过少**：只记录 `eval/accuracy`，缺少 reward 分布、pairwise accuracy、loss margin（正负样本得分差）等，难以定位「是数据噪声还是欠拟合」。
6. **没有独立测试集**：`model_best` 直接在 dev 上挑选，最终报告的 0.666 与模型选择共用同一份 dev，存在选择偏差。
7. **全流程无文本日志**：本目录没有 `日志.md`，stdout 也未落盘，训练中途的 loss/acc 全部不可追溯，只能依赖自动生成的 PNG。
8. **数据文件与磁盘行数相差 1 条**（见 §1.2 口径注），做精确 step/epoch 换算时需留意。

---

## 6. 结论与改进建议

### 6.1 结论

- 奖励模型阶段**已训练并落盘**：ERNIE-3.0-base-zh + 线性打分头，偏好对排序训练，落盘 15 个 checkpoint（`model_200~model_2800` + `model_best`）；
- 评测指标 `eval/accuracy` 的 **max 与末端均为 0.666**（曲线反解，±0.002），约为 4 元全序随机基线（0.0417）的 16 倍，说明排序信号确实被学到；
- 未能更高的主因是**严格的全序指标口径 + 偏好数据噪声 + 训练不充分（lr 1e-5、约 7.3 epoch）+ 输入截断**，而非模型结构错误；
- **PPO 阶段未运行**，`checkpoints/` 与 `logs/` 下均无任何 PPO 产物，因此 RLHF 链路只走完第一阶段。

### 6.2 具体可执行的改进建议

| 序号 | 建议 | 预期作用 |
|---|---|---|
| 1 | 在评测中**同时输出 pairwise accuracy、top-1 accuracy、平均得分 margin** | 把「全序严格指标」与「真实排序质量」区分开，避免 0.666 被误读为「模型很差」 |
| 2 | 把 `learning_rate` 提到 2e-5 ~ 5e-5，或把 `num_train_epochs` 加到 20 | 验证曲线平台是「数据上限」还是「欠训练」 |
| 3 | 数据清洗：剔除 4 档情感不可分/中性/矛盾的 rank list，做标注一致性抽样复核 | 直接降低标签噪声，抬高精度上限 |
| 4 | 把 `max_seq_len` 提到 256/512，或对长评做保留情感词的截断 | 减少关键信息丢失 |
| 5 | 划分独立 `test` split，`model_best` 只在 test 上评一次 | 消除 dev 上的选择偏差 |
| 6 | 修正源码中 `best F1` 文案，并把 `eval/accuracy` 的保存节奏与 `valid_steps` 对齐 | 报告口径自洽、checkpoint 可解释 |
| 7 | 用 `inference_reward_model.py` 对正/负样例做定性打分检查 | 人工验证打分单调性 |
| 8 | 若要继续 RLHF 第二阶段，需先决定 reward 来源（改用本次 RM 而非情感 pipeline），并准备多卡/大显存环境 | 打通 RM→PPO 链路 |

---

## 7. 复现命令

### 7.1 本次实际运行的奖励模型训练

```sh
cd RLHF
# 与 train_reward_model.sh 完全一致
python train_reward_model.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/reward_datasets/sentiment_analysis/train.tsv" \
    --dev_path "data/reward_datasets/sentiment_analysis/dev.tsv" \
    --save_dir "checkpoints/reward_model/sentiment_analysis" \
    --img_log_dir "logs/reward_model/sentiment_analysis" \
    --img_log_name "ERNIE Reward Model" \
    --batch_size 32 \
    --max_seq_len 128 \
    --learning_rate 1e-5 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

### 7.2 奖励模型推理（定性检查）

```sh
cd RLHF
python inference_reward_model.py
```

### 7.3 PPO 阶段（**本次未运行**，仅列出补跑入口）

```sh
cd RLHF
pip install -r ../requirements.txt          # 依赖含 trl
python ppo_sentiment_example.py
```

补跑前需注意：脚本内 `config` 为 `steps=20000 / batch_size=128 / ppo_epochs=4 / gen_len=16 / save_freq=5`，产物默认落在 `checkpoints/ppo_sentiment_gpt`，曲线落在 `logs/PPO-Sentiment-Zh.png`；且其 reward 来自情感分类 pipeline，若要真正复用本报告的 RM，需要改造 reward 计算部分。
