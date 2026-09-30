# Transformers 实战（transformers_tasks-main）—— 结果原始数据

> 本文件由 `实验报告/_原始数据/extract_all.py`（曲线图像素反解）+ 人工读取刻度标签 + 各任务
> `train.sh` 配置 + `logs/` 下的终端截图汇总而成，是撰写该部分实验报告的**唯一数据来源**。
> ⚠️ 该目录**没有 `日志.md`**：这一部分的实验结果只以「训练曲线 PNG + 终端截图 + checkpoint 目录」的形式存在。

---

## A. 各子任务配置（取自各自的 `train.sh`）

| 子任务 | 脚本 | 模型 | 数据集 | 关键超参 | 评测节奏 | 产物（logs/ + checkpoints/） |
|---|---|---|---|---|---|---|
| 文本分类 | `text_classification/train.py` | `bert-base-chinese` | comment_classify（8 类；train 400 / dev 61） | batch 16、max_len 128、lr 5e-5（默认）、epochs 20、num_labels 8 | 每 50 step | `logs/comment_classify/BERT.png`；`checkpoints/comment_classify/{model_200, model_best}` |
| 文本匹配 PointWise | `text_matching/supervised/train_pointwise.py` | `nghuyong/ernie-3.0-base-zh` | comment_classify | batch 8、max_len 128、epochs 10 | 每 50 step | `logs/comment_classify/ERNIE-PointWise.png`；`checkpoints/comment_classify/{model_200, model_best}` |
| 文本匹配 DSSM（双塔） | `train_dssm.py` | ernie-3.0-base-zh | comment_classify | batch 8、max_len 128、epochs 10 | 每 50 step | `ERNIE-DSSM.png`；`checkpoints/comment_classify/dssm/{model_0, model_200, model_best}` |
| 文本匹配 Sentence-BERT（双塔） | `train_sentence_transformer.py` | ernie-3.0-base-zh | comment_classify | batch 8、max_len 256、epochs 10 | 每 50 step | `Sentence-Ernie.png`；`checkpoints/.../sentence_transformer/{model_0, model_200, model_best}` |
| 无监督 SimCSE | `text_matching/unsupervised/simcse/train.py` | ernie-3.0-base-zh | LCQMC（`train.txt` 477,532 **行句子**，无监督 SimCSE 只把它当句子语料用；eval 用 `dev.tsv` 的 8,801 对句对 + 0/1 标签） | lr 1e-5、dropout 0.3、batch 64、max_len 64、epochs 8 | 每 400 step | `logs/LCQMC/ERNIE-ESimCSE.png`；`checkpoints/LCQMC/`（8 个，model_400~2800 + best） |
| PET（prompt） | `prompt_tasks/PET/pet.py` | bert-base-chinese | comment_classify **小样本**：train 61 / dev 589 | batch 8、max_len 256、epochs 200、max_label_len 2、rdrop 5e-2 | 每 40 step | `logs/comment_classify/BERT-PET.png`；`checkpoints/comment_classify/`（17 个） |
| p-tuning（prompt） | `prompt_tasks/p-tuning/p_tuning.py` | bert-base-chinese | comment_classify **小样本**：train 61 / dev 415 | batch 8、max_len 128、epochs 20、p_embedding_num 15 | 每 20 step | `logs/comment_classify/BERT.png`；`checkpoints/comment_classify/{model_10, model_20, model_best}` |
| RLHF 奖励模型 | `RLHF/train_reward_model.py` | ernie-3.0-base-zh | sentiment_analysis 偏好对（train 12,327 / dev 3,081） | batch 32、max_len 128、lr 1e-5、epochs 10 | 每 50 step | `logs/reward_model/sentiment_analysis/ERNIE Reward Model.png`；`checkpoints/.../`（15 个，model_200~2800 + best） |
| 答案生成 | `answer_generation/train.py` | `uer/t5-base-chinese-cluecorpussmall` | DuReaderQG（train 14,520 / dev 983） | batch 32、source 256 / target 32、lr 5e-5、epochs 50 | 每 500 step | `logs/DuReaderQG/T5-Base-Chinese.png`；`checkpoints/DuReaderQG/{model_200,400,600,800,best}` |
| Filling / 数据增强 | `data_augment/filling_model/train.py` | uer/t5-base-chinese-cluecorpussmall | train.tsv 350,134 行 | batch 128、source 128 / target 32、lr 1e-4、epochs 20 | 每 500 step | `logs/屏幕截图 2026-09-22 164846.png`（无 PNG 曲线）；`checkpoints/t5/model_200` |
| ChatGLM-6B 微调 | `LLM/chatglm_finetune/train.py` | ChatGLM-6B + LoRA(r=8) | 自备 jsonl | batch 1、epochs 2、lr 3e-5 | 每 1000 step | **无任何产物**（`log/fintune_log/` 为空、无 checkpoints） |
| LLM zero-shot 应用 | `LLM/zero-shot/llm_*.py` | chatglm-6b | 三个任务的 prompt | —— | —— | **无任何产物**（脚本被改过，但没有输出文件） |
| UIE / LLMsTrainer / llms_mbti | —— | —— | —— | —— | —— | 脚本未被改动，也**没有运行产物** |

> ✅ 结论：实际跑过并有结果留存的子任务共 **10 个**（上表前 10 行）；
> ChatGLM 微调、LLM zero-shot、UIE、LLMsTrainer、llms_mbti 只保留了代码，没有结果。

---

## B. 曲线图反解出的指标

方法：`iTrainingLogger` 用 `ax.plot(color='darkorange')` 画线、不打 marker；
**只记录过一个点的子图画不出线**，但坐标轴范围仍是该点的 autoscale 结果，于是
「该点 = 坐标轴正中 = 刻度序列的中位数」；曲线子图则用白色网格线与刻度标签做线性标定后读出。

| 子任务 | 指标 | 类型 | 数值 | 备注 |
|---|---|---|---|---|
| 文本分类 BERT | eval/accuracy | 单点 | **0.32** | 整个训练只在 step 200 评测过 1 次 |
| 文本分类 BERT | eval/precision | 单点 | **0.34** | 同上 |
| 文本分类 BERT | eval/recall | 单点 | **0.32** | 同上 |
| 文本分类 BERT | eval/f1 | 单点 | **0.26** | 8 分类，只看 dev 61 条，方差大 |
| 文本匹配 PointWise | eval/accuracy | 单点 | **0.94** | 只在 step 200 评测 1 次 |
| 文本匹配 PointWise | eval/precision | 单点 | **0.86** | |
| 文本匹配 PointWise | eval/recall | 单点 | **0.92** | |
| 文本匹配 PointWise | eval/f1 | 单点 | **0.90** | 单塔效果明显好于双塔 |
| 文本匹配 DSSM | eval/accuracy | 曲线 | max 0.769 / 末端 0.767 | |
| 文本匹配 DSSM | eval/precision | 曲线 | max 0.617 / 末端 0.616 | |
| 文本匹配 DSSM | eval/recall | 曲线 | max 0.954 / 末端 0.604 | 末端回落，曲线整体在下降 |
| 文本匹配 DSSM | eval/f1 | 曲线 | max 0.610 / 末端 0.610 | |
| 文本匹配 Sentence-BERT | eval/accuracy | 曲线 | max 0.890 / 末端 0.889 | |
| 文本匹配 Sentence-BERT | eval/precision | 曲线 | max 0.815 / 末端 0.813 | |
| 文本匹配 Sentence-BERT | eval/recall | 曲线 | max 0.822 / 末端 0.821 | |
| 文本匹配 Sentence-BERT | eval/f1 | 曲线 | max 0.818 / 末端 0.816 | |
| 无监督 SimCSE | eval/accuracy | 曲线 | max 0.636 / 末端 0.584 | |
| 无监督 SimCSE | eval/precision | 曲线 | max 0.592 / 末端 0.546 | |
| 无监督 SimCSE | eval/recall | 曲线 | ≈0.90~1.00 | 曲线顶端贴着 1.00 刻度（含误差，按 ~0.99 看） |
| 无监督 SimCSE | eval/f1 | 曲线 | max 0.707 / **末端 0.7054** | 与终端截图 0.70541 **互相印证** ✅ |
| 无监督 SimCSE | eval/spearman_corr | 曲线 | max 0.566 / **末端 0.565** | 与终端截图 0.56527 **互相印证** ✅ |
| PET | eval/accuracy | 曲线 | max 0.769 / 末端 0.769 | 小样本（61 条）训练 |
| PET | eval/precision | 曲线 | max 0.799 / 末端 0.799 | |
| PET | eval/recall | 曲线 | max 0.769 / 末端 0.769 | |
| PET | eval/f1 | 曲线 | max 0.749 / 末端 0.747 | |
| p-tuning | eval/accuracy | 曲线 | max 0.650 / 末端 0.650 | 小样本（61 条）训练 |
| p-tuning | eval/precision | 曲线 | max 0.760 / 末端 0.750 | |
| p-tuning | eval/recall | 曲线 | max 0.650 / 末端 0.650 | |
| p-tuning | eval/f1 | 曲线 | max 0.640 / **末端 0.640** | 与终端截图 "F1: 0.64000" **互相印证** ✅ |
| RLHF 奖励模型 | eval/accuracy | 曲线 | max 0.666 / 末端 0.666 | 偏好对二分类准确率 |
| 答案生成 T5 | eval/bleu-size-1 | 曲线 | max 0.097 / 末端 0.096 | |
| 答案生成 T5 | eval/bleu-size-2 | 曲线 | max 0.063 / 末端 0.063 | |
| 答案生成 T5 | eval/bleu-size-3 | 曲线 | max 0.035 / 末端 0.035 | |
| 答案生成 T5 | eval/bleu-size-4 | 曲线 | max 0.023 / 末端 0.023 | BLEU-4 0.023 属于"能生成、但远不完美" |

---

## C. 终端截图逐字记录（终端里打印的真实数值）

1. `text_classification/logs/屏幕截图 2026-09-22 165323.png`（文本分类，**第二次运行**）
   ```
   global step 10, epoch: 1, loss: 1.97777, speed: 7.42 step/s
   global step 20, epoch: 1, loss: 1.79483, speed: 8.93 step/s
   global step 30, epoch: 2, loss: 1.56500, speed: 9.49 step/s
   global step 40, epoch: 2, loss: 1.33791, speed: 8.91 step/s
   global step 50, epoch: 2, loss: 1.16831, speed: 8.91 step/s
   global step 60, epoch: 3, loss: 1.03924, speed: 9.58 step/s
   global step 70, epoch: 3, loss: 0.92857, speed: 8.91 step/s
   ```
   → 截图时训练只到 step 70，loss 从 1.98 降到 0.93；本次运行没有触发评测（`valid_steps=50` 时应有评测，
   但 `logs/comment_classify/BERT.png` 的修改时间仍是 9-20，说明这次运行没走到写图那一步）。

2. `text_matching/unsupervised/simcse/logs/LCQMC/屏幕截图 2026-09-22 162356.png`（SimCSE）
   ```
   Evaluation precision: 0.54459, recall: 0.99614, F1: 0.70419, spearman_corr: 0.56181
   global step 2050, epoch: 1, loss: 0.16381, speed: 0.19 step/s
   global step 2100, epoch: 1, loss: 0.16099, speed: 0.18 step/s
   ... (略) ...
   global step 2400, epoch: 1, loss: 0.14650, speed: 0.15 step/s
   Evaluation precision: 0.54625, recall: 0.99546, F1: 0.70541, spearman_corr: 0.56527
   global step 2800, epoch: 1, loss: 0.13076, speed: 0.15 step/s
   ```
   → **最可靠的原始数值**：step 2400 时 precision 0.54625 / recall 0.99546 / F1 0.70541 / spearman 0.56527；
   训练速度约 0.15~0.19 step/s（LCQMC 47.7 万条句子，batch 64）。注意 recall 高达 0.995 而 precision 只有 0.55，
   说明该评测口径下正类判定极度宽松。

3. `prompt_tasks/p-tuning/logs/comment_classify/屏幕截图 2026-09-22 181302.png`（p-tuning）
   ```
   best F1 performence has been updated: 0.00000 --> 0.62000
   ...
   Evaluation precision: 0.75000, recall: 0.65000, F1: 0.64000
   best F1 performence has been updated: 0.62000 --> 0.64000
   Each Class Metrics are: {'书籍': {...'f1': 0.78}, '平板': {...}, '水果': {...}, '洗浴': {...},
                            '电器': {'precision': 0, 'recall': 0, 'f1': 0}, '电脑': {...f1: 0.5}, '蒙牛': {...f1: 0.8}, '酒店': {...f1: 0.91}}
   ```
   → p-tuning 最好一轮：precision 0.75 / recall 0.65 / **F1 0.64**；8 类里 `电器` 的 F1 = 0
   （该类的 precision/recall 都是 0），典型的小样本类别不均衡问题。

4. `data_augment/filling_model/logs/屏幕截图 2026-09-22 164846.png`
   → 该截图未包含可读的评测指标行（曲线目录里也没有 PNG），**该子任务没有可引用的指标数值**，
   `checkpoints/t5/model_200` 只说明训练跑到过 200 step。

---

## D. 与终端截图的交叉验证（方法可信度）

| 子任务 | 截图给出的值 | 曲线反解值 | 是否一致 |
|---|---|---|---|
| SimCSE F1 | 0.70541（step 2400） | 0.7054（末端） | ✅ |
| SimCSE spearman | 0.56527（step 2400） | 0.5650（末端） | ✅ |
| p-tuning F1 | 0.64000（best） | 0.6400~0.6401（max） | ✅ |

三项独立吻合，说明"刻度标定 + 像素反解"这套读图方法是可靠的；
其余曲线值也应视为同精度（约 ±0.002）的读数。

---

## E. 需要注意的口径问题

1. **单点 vs 曲线**：`valid_steps` 与 `num_train_epochs` 组合决定了评测次数。
   文本分类与 PointWise 只评测了 **1 次**（`checkpoints` 里只有 `model_200`），
   所以曲线图上那份"评测指标"是**一次快照**，不是收敛曲线。
2. **文本分类的 8 分类 F1 只有 0.26、accuracy 0.32**：远低于同数据集的 PointWise（F1 0.90）。
   两者网络结构不同（单句分类 vs 句对匹配）、训练轮数不同（20 vs 10 epoch），
   且 BERT 那次训练只跑了 200 step（约 5 个 epoch），属于**欠训练 + 单点评测**，
   不应据此判断"BERT 不如 ERNIE"。报告里已按此口径说明。
3. **PET / p-tuning 用的是小样本集**：train.txt 只有 61 条（dev 589 / 415 条），
   PET 训练 200 epoch、p-tuning 20 epoch，二者 F1 分别为 0.747 / 0.640，
   都明显低于全量数据训练的 PointWise（0.90）——符合"提示学习在小样本上划算、
   数据量充足时不如全量微调"的一般结论。
4. **无产物的子任务**（ChatGLM 微调、LLM zero-shot、UIE、LLMsTrainer、llms_mbti）：
   只能报告"代码已实现、未运行/未留结果"，**不得编造指标**。
