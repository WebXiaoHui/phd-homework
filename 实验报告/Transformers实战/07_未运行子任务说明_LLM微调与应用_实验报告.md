# Transformers 实战 07：未运行子任务说明——LLM 微调与应用

> 覆盖范围：`transformers_tasks-main/` 中 4 类**只保留了代码、没有运行产物**的子任务。
> 数据来源：各子任务 `readme.md` 与脚本头部注释、`train.sh`、配置 YAML，以及本次对 `logs/`、`checkpoints/` 的 `ls` 核对。
> ⚠️ **本报告不含任何模型指标**，因为这些子任务不存在可引用的运行结果；本目录也**没有 `日志.md`**，
> 从而无法从日志还原「为什么没跑」的现场记录，只能由空目录与代码状态反推。
> 本报告的目标是「实验现状说明 + 复现所需条件」，不是结果报告。

---

## 0. 总览

| # | 子任务 | 目录 | 代码状态（本次核对 mtime） | 运行产物 | 未运行主因（推断） |
|---|---|---|---|---|---|
| ① | ChatGLM-6B 微调（LoRA / P-Tuning） | `LLM/chatglm_finetune/` | `train.py` 等被改过（9-22 18:20 前后） | `log/fintune_log/` **为空**；**无** `checkpoints/` 目录 | 6B 模型 + 本机 8GB 显存 |
| ② | LLM zero-shot 三类 prompt 应用 | `LLM/zero-shot/` | 三个脚本被改过（9-22 18:18） | **无任何输出文件** | 需要可用的 chatglm-6b（约 12GB 权重 / ~13GB 显存） |
| ③ | UIE 通用信息抽取 | `UIE/` | **未改动**（全部 9-15） | **无** `logs/`、**无** `checkpoints/` | 从未运行 |
| ④ | LLMsTrainer + llms_mbti | `LLM/LLMsTrainer/`、`LLM/llms_mbti/` | LLMsTrainer 的 `train_reward_model.py`、`iTrainingLogger.py` 被改过（9-22 18:26）；llms_mbti 未改动 | **无** `checkpoints/`、**无** `log/`；llms_mbti 无测试输出 | 7B 模型 + 多卡/DeepSpeed；mbti 需下载 7B 模型 |

> **贯穿全篇的一条重要提醒**：上游仓库 `assets/` 与部分数据文件里带着一批「示例结果」
> （如 `RLHF/assets/PPO-Sentiment-Zh.png`、`LLM/zero-shot/assets/llm_*_res.png`、
> `UIE/assets/UIE Base No Aug.png`、`LLM/chatglm_finetune/assets/ChatGLM Fine-Tune.png`、
> `data_augment/filling_model/assets/T5-Base-Chinese.png`、`LLM/llms_mbti/assets/mbti_visualization.png`、
> `LLM/llms_mbti/llms_mbti.json`）。**这些是仓库自带的演示资产，不是本次实验的产物**，
> 报告中不得把它们当作本次结果引用。

---

## 1. ChatGLM-6B 微调（`LLM/chatglm_finetune`）

### 1.1 实现了什么

参考 `mymusise/ChatGLM-Tuning` 与 ChatGLM 官方的 ptuning 实现，对 **ChatGLM-6B** 做参数高效微调，使其对齐指定的输出格式（信息抽取 + 文本分类混合任务）。支持两种微调方式：

| 项目 | LoRA 配置 | P-Tuning 配置 |
|---|---|---|
| 开关 | `--use_lora True` | `--use_ptuning True` |
| 关键超参 | `lora_rank 8` | `pre_seq_len 128` |
| learning rate | 3e-5 | 2e-4 |
| save_freq | 1000 | 200 |
| save_dir | `checkpoints/finetune` | `checkpoints/ptuning` |
| img_log_name | `ChatGLM Fine-Tune` | `ChatGLM P-Tuning` |
| 共用 | `train_path data/mixed_train_dataset.jsonl`、`dev_path data/mixed_dev_dataset.jsonl`、`batch_size 1`、`num_train_epochs 2`、`logging_steps 100`、`max_source_seq_len 400`、`max_target_seq_len 300`、`img_log_dir log/fintune_log`、`device cuda:0` | 同左 |

- 数据为「Instruction + Input → target」两段式 JSONL：`context` 是用户输入（含 Instruction 与 Input），`target` 是期望输出；覆盖「找三元组输出 JSON」与「判断评论属于什么类别」两类样本（`readme.md` §2）。
- 数据规模（本次 `wc -l` 核对）：`mixed_train_dataset.jsonl` **902 行**、`mixed_dev_dataset.jsonl` **122 行**（另有 4 行的 `dataset.jsonl` 示例文件）。
- 训练侧使用 `peft-chatglm`（需单独 `python setup.py install`）、`torch.cuda.amp.autocast`；评测用 dev loss，按 **min eval loss** 保存 `model_best`；支持 `--quantization_bit` 量化开关（默认 `None`）。

### 1.2 为什么没有结果

| 证据 | 实际情况 |
|---|---|
| `log/fintune_log/` | 目录**存在但为空**——说明进程至少走到 `iSummaryWriter` 初始化，但**从未写入任何一个评测点**（`logging_steps=100`，即没到第一个日志周期） |
| `checkpoints/finetune`、`checkpoints/ptuning` | **均不存在**，即从未保存过任何模型 |
| 终端截图 / 曲线 PNG | `log/` 与 `assets/` 下无本次运行的截图或曲线（`assets/ChatGLM Fine-Tune.png` 是仓库演示图） |

**根本原因（硬件门槛）**：

- ChatGLM-6B 是 60 亿参数模型，fp16 权重约 12GB；`readme.md` 在 zero-shot 部分明确指出「加载模型大概需要 **13G 左右的显存**」；
- `train.sh` 使用的是**未量化**加载（`--quantization_bit` 默认 `None`），即使 `batch_size=1`、`max_source_seq_len=400 / max_target_seq_len=300`，加上激活与梯度，**本机 8GB 显存无法容纳**；
- 此外还需额外约 12GB 磁盘下载模型权重，并准备独立的 `llm_env` 虚拟环境（`readme.md` §1 要求 `conda create -n llm_env python=3.8` + `peft-chatglm`）。

因此该子任务是「**代码已落地、因显存不足而未启动/未完成**」。

### 1.3 要跑需要什么条件

- **硬件**：≥16GB 显存的单卡，或使用 8bit/4bit 量化把显存压到 8GB 量级；
- **软件**：独立 conda 环境，安装 `requirements.txt` 与 `peft-chatglm`；
- **数据**：`data/mixed_train_dataset.jsonl`（已存在，902 条）与 `data/mixed_dev_dataset.jsonl`（122 条）；
- **模型**：`THUDM/chatglm-6b` 权重（约 12GB）。

### 1.4 结论

**没有可引用的指标。** `readme.md` 中的评测 loss 打印、best 模型刷新提示与单卡/双卡耗时等，均来自**上游仓库的示例运行**，不是本次产物。

---

## 2. LLM zero-shot 应用（`LLM/zero-shot`）

### 2.1 实现了什么

基于 ChatGLM-6B，用**同一个大模型 + 不同 prompt**（in-context learning）解决三类 NLP 任务，对应三个脚本：

| 脚本 | 任务 | prompt 设计要点 | 测试样例 |
|---|---|---|---|
| `llm_classification.py` | 文本分类 | `class_examples` 提供 6 类（人物 / 书籍 / 电视剧 / 电影 / 城市 / 国家）各一段百科文本；`init_prompts()` 构造 `pre_history`（User 句子 → Bot 类别）；推理时提问 `“{sentence}”是 {class_list} 里的什么类别？` | 5 段百科描述（加拿大、《琅琊榜》、《满江红》、布宜诺斯艾利斯、张译） |
| `llm_text_matching.py` | 文本匹配 | `examples = {'是': [...], '不是': [...]}` 提供相似/不相似示例；prompt 为 `句子一: ...\n句子二: ...\n上面两句话是相似的语义吗？` | 3 个句对（改头像、司马懿连招 ×2） |
| `llm_information_extraction.py` | 信息抽取 | **先分类再抽取**：用 `class_examples` 判类别 → 按 `schema`（人物/书籍/电视剧及各自属性）拼 prompt → 要求输出 JSON → `clean_response` 清洗 | 2 段百科句子（张译、《琅琊榜》） |

三者共同点：`THUDM/chatglm-6b` + `trust_remote_code=True` + `.half().to('cuda:0')`，通过 `model.chat(tokenizer, prompt, history=...)` 逐条推理并打印结果。

### 2.2 为什么没有结果

| 证据 | 实际情况 |
|---|---|
| 目录内容 | `LLM/zero-shot/` 下只有 `llm_classification.py`、`llm_text_matching.py`、`llm_information_extraction.py`、`playground.py`、`readme.md`、`requirements.txt` 与 `assets/`；**没有任何输出文件**（无 `.json`/`.txt`/`.log`/截图产物） |
| `assets/` 中的三张 `llm_*_res.png` | 是**仓库自带的推理效果示例图**，不是本次运行输出 |

**根本原因**：

- 加载 ChatGLM-6B 约需 13GB 显存（`readme.md` 明确写出），脚本**硬编码 `.half()` 的未量化加载**，本机 8GB 显存会 OOM；
- 脚本本身也**不落盘**——结果只 `print` 到 stdout，即使跑过，没有重定向就留不下证据；
- 需要可用的 chatglm-6b 服务或本地权重（约 12GB 磁盘）。

### 2.3 要跑需要什么条件

- 一个可用的 chatglm-6b：本地量化加载（如 `model.quantize(8)`）、或调用已有的 ChatGLM 服务/API；
- 独立 `llm_env` 环境（`readme.md` §1）；
- 把 stdout 重定向保存，例如 `python llm_classification.py 2>&1 | tee logs/zero_shot_cls.log`。

### 2.4 结论

**没有可引用的指标。** `readme.md` §2.3 / §3.3 / §4.3 给出的「期望输出」是**任务设计目标**，`assets/` 中的三张效果图是**仓库示例截图**，都不是本次实测结果。

---

## 3. UIE 通用信息抽取（`UIE`）

### 3.1 实现了什么

用 `transformers` 复刻 PaddleNLP 版本的 UIE（Universal Information Extraction），已实现的能力（`readme.md`）：

- UIE 预训练模型自动下载；
- UIE Fine-Tuning 脚本（`train.py` / `train.sh`）；
- 信息抽取、事件抽取的**数据增强**（SwapSPO、Mask Then Fill，见 `Augmenter.py`）；
- **自分析负例生成**（Auto Neg，用于提升 precision）。

训练配置（`UIE/train.sh`）：

| 参数 | 取值 |
|---|---|
| pretrained_model | `uie-base-zh` |
| save_dir / img_log_dir / img_log_name | `checkpoints/DuIE` / `logs/` / `UIE Base` |
| train_path / dev_path | `data/DuIE/train.txt` / `data/DuIE/dev.txt` |
| batch_size / max_seq_len | 32 / 256 |
| learning_rate / num_train_epochs | 5e-5 / 20 |
| logging_steps / valid_steps | 10 / 100 |
| device | cuda:0 |

数据（本次核对）：`data/DuIE/train.txt` 443,733 字节、`dev.txt` 58,298 字节、`doccano_ext.jsonl`；readme 说明数据为「DuIE 数据集中随机抽取的 100 条」。**`test.txt` 为 0 字节空文件**。

### 3.2 为什么没有结果

| 证据 | 实际情况 |
|---|---|
| 脚本状态 | `UIE/` 下所有 `.py` / `.sh` 的 mtime 均为 **9-15**（与仓库初始一致），即**未被改动** |
| 产物目录 | `UIE/logs/`、`UIE/checkpoints/` **均不存在** |
| `assets/` 中的曲线图 | `UIE Base No Aug.png`、`mask_then_fill.png`、`swap_spo.png` 等是**仓库自带示例图** |

**根本原因**：该子任务**从未启动**。可能的阻碍包括：需要额外下载 `uie-base-zh` 权重并准备其运行环境；`test.txt` 为空导致缺少测试集；以及 UIE 的增强策略（Mask Then Fill）依赖第 06 篇中的 filling 模型（当前只有一个 `model_200`，尚无可用指标）。

### 3.3 要跑需要什么条件

- `pip install -r requirements.txt`，并准备 `uie-base-zh` 权重；
- 补齐 `data/DuIE/test.txt`（当前为空）；
- 若要复现 Mask Then Fill 增强，需要先训好 filling 模型（见第 06 篇）。

### 3.4 结论

**没有可引用的指标。** `UIE/readme.md` §5.4 的 DA 策略对比表与其中的评测日志，都是**上游仓库针对 100 条 DuIE 数据报告的结果**，不是本次实验产物。

---

## 4. LLMsTrainer（从零训练大模型 / RM 训练框架）与 llms_mbti

### 4.1 LLMsTrainer 实现了什么

一个「从零/继续训练大模型」的框架（最早参考 Open-Llama），`readme.md` 列出的能力：

- [x] 继续预训练（Continue Pretraining）
- [x] 指令微调（Instruction Tuning）
- [x] 奖励模型训练（Reward Model）
- [ ] 强化学习（Reinforcement Learning）——**未实现**

工程落地包括：`accelerate` + DeepSpeed 配置（`configs/accelerate_configs/ds_stage1/2/3/3_offload.yaml`）、三套训练配置、数据压缩脚本、采样比例可视化工具、词表扩充/embedding 初始化工具、单机与多机启动脚本。

三套 YAML 的关键超参：

| 配置 | 数据（路径） | 关键超参 | 产物路径 |
|---|---|---|---|
| `configs/pretrain_configs/llama.yaml` | `data/pretrain_data/MNBVC_{news,qa,wiki}/*.jsonl.zst`；`sample_policy_file: configs/sample_policy/pretrain/MNBVC.json` | `seq_length 2048`、`train_batch_size 1`、`gradient_accumulation_steps 30`、`lr 5.0e-5`、`num_training_steps 10000`、`num_warmup_steps 100`、`save_total_limit 3`、`gradient_checkpointing_enable true` | `work_dir: checkpoints/pretrain/open_llama_7b_v2`；`img_log_dir: log/pretrain/open_llama_7b_v2` |
| `configs/sft_configs/llama.yaml` | `data/sft_data/sharegpt/*.jsonl.zst` | `seq_length 2048`、`train_batch_size 1`、`grad accum 30`、`lr 1.0e-5`、`num_training_steps 50000`、`resize_model_vocab_size true` | `work_dir: checkpoints/sft/ShareGPT` |
| `configs/reward_model_configs/llama.yaml` | `data/reward_model_data/sentiment_comments.jsonl` | `seq_length 2048`、`batch_size 1`、`lr 1.0e-6`、`min_lr 1.0e-7`、`num_training_epochs 1`、`save_total_limit 1` | `work_dir: checkpoints/reward_model/llama7b/sentiment_comments` |

- 基座模型统一为 `openlm-research/open_llama_7b_v2`（**7B**）。
- 启动脚本：`train_llms.sh`（`accelerate launch ... train_llms.py --train_config --model_config`）、`train_reward_model.sh`、`train_multi_node_llms.sh`（多机，`num_processes=8`）。
- 数据准备：先 `cd data && python compress_data.py`，把 `data/shuffled_data/{pretrain,sft}` 下的原始 JSONL 压缩为 `.jsonl.zst`（`SHARD_SIZE=10`）。
- 本次核对：`train_reward_model.py` 与 `iTrainingLogger.py` 的 mtime 为 **9-22 18:26**，说明这两个脚本被改动过；其余文件保持 9-15。

**现有数据规模**（本次 `ls -l` 核对）：`data/shuffled_data/pretrain/` 下 `MNBVC_news.jsonl`、`MNBVC_qa.jsonl`、`MNBVC_wiki.jsonl`；`data/shuffled_data/sft/sharegpt.jsonl`（约 29.5MB）；`data/reward_model_data/sentiment_comments.jsonl`（约 388KB，`readme.md` 说明为 1000 条偏序对）。

### 4.2 为什么没有结果

| 证据 | 实际情况 |
|---|---|
| `LLM/LLMsTrainer/checkpoints/` | **不存在** |
| `LLM/LLMsTrainer/log/` | **不存在** |
| 配置指向的压缩数据 | `data/pretrain_data/`、`data/sft_data/`（YAML 中 `data` 字段的目标路径）**均不存在**——只有未压缩的 `data/shuffled_data/` |

→ 也就是说，**连第一步数据压缩都没有执行**，训练自然无从谈起。

**根本原因**：`open_llama_7b_v2` 是 7B 模型，配置 `seq_length 2048`、`train_batch_size 1`，需要 `accelerate` + DeepSpeed（最好多卡或开启 stage-3 offload）；本机单卡 8GB 显存不可行。

### 4.3 llms_mbti 实现了什么、为什么没跑

- `get_llms_mbti.py`：读取 `mbti_questions.json`（MBTI 题库）与 `few_shot_examples`，让 HF 上的大模型逐题作答，统计 E/I、S/N、T/F、J/P 四维得分，写入 `llms_mbti.json`；
- `readme.md` 给出的测试对象示例为 `baichuan-inc/Baichuan-7B`、`bigscience/bloom-7b1`（均为 7B 级），也说明 ChatGPT/GPT-4 需改用 OpenAI API；
- `web.py` 用 streamlit 把 `llms_mbti.json` 可视化。

**为什么没跑**：每个被测模型都是 7B 级，需下载约 13GB 权重并占用大显存，本机 8GB 显存不可行；`readme.md` 也提示该方法仅适用于 HuggingFace 上的模型。

> **重要澄清**：仓库中已存在的 `llms_mbti.json`（放在 `LLM/llms_mbti/` 下）包含 ChatGPT、GPT-4、Baichuan、Bloom、OpenLlama 等多个模型的 MBTI 结果，这是**上游仓库自带的示例/预置结果**，`readme.md` 中展示的 MBTI 结论也只是示例输出，**均非本次实验结果**。

### 4.4 要跑需要什么条件

**LLMsTrainer**

1. 多卡 GPU（或大显存单卡）+ 安装 `requirements.txt`（含 accelerate / deepspeed / transformers）；
2. 先跑通数据压缩：`cd LLM/LLMsTrainer/data && python compress_data.py`（默认只跑 `batch_compress_preatrain_data()`，需按 readme 注释切换/取消注释 `batch_compress_sft_data()`）；
3. 再把 YAML 中 `data` 路径指向真实数据，然后：

```sh
sh train_llms.sh configs/accelerate_configs/ds_stage1.yaml \
    configs/pretrain_configs/llama.yaml \
    openlm-research/open_llama_7b_v2
```

**llms_mbti**

```sh
cd LLM/llms_mbti
pip install -r requirements.txt
python get_llms_mbti.py     # 会下载 7B 级模型并写 llms_mbti.json
streamlit run web.py --server.port 8001
```

### 4.5 结论

**没有可引用的指标。** LLMsTrainer 与 llms_mbti 都停留在「代码/框架就绪、数据未压缩、模型未下载」的状态。

---

## 5. 踩坑与局限（跨子任务）

1. **仓库演示资产与本次产物极易混淆**：`assets/` 下的示例图与 `llms_mbti.json` 的预置结果都不是本次实验证据；报告与答辩中必须区分，否则会把上游结果误当自己的成绩。
2. **本机 8GB 显存是 6B/7B 类任务的硬门槛**：ChatGLM-6B、zero-shot、LLMsTrainer、llms_mbti 全部卡在这一条；且 `train.sh` / 脚本默认都未开量化。
3. **环境隔离要求**：`chatglm_finetune` 与 `zero-shot` 的 readme 都明确要求新建 `llm_env`（`peft-chatglm` 需源码安装），与仓库其他任务环境不兼容。
4. **训练脚本不落盘 stdout**：`log/fintune_log/` 为空但目录存在，说明只走完了 logger 初始化；这类「空目录」是判断「跑到哪一步」的唯一线索。
5. **数据准备存在前置依赖**：LLMsTrainer 的配置指向压缩后的 `.jsonl.zst`，但 `compress_data.py` 从未运行；UIE 的 `test.txt` 为 0 字节；filling 数据增强依赖尚未训好的 filling 模型。
6. **没有 `日志.md`**：Transformers 部分整体缺少文字日志，未运行子任务连「为什么没跑」的现场记录都没有，只能靠空目录、mtime 与配置路径反推。
7. **代码改动 ≠ 运行**：`zero-shot`（9-22 18:18）、`chatglm_finetune`（9-22 18:20 前后）、`LLMsTrainer`（9-22 18:26）的脚本都被改动过，但没有任何运行产物——这正是「只改了代码、没有跑」的直接证据。

---

## 6. 结论与改进建议

### 6.1 结论

4 类子任务**全部没有可引用的实验结果**，现状是「代码已实现（部分还被改动/调试过），但因算力或环境门槛未运行/未产出」：

- ① ChatGLM-6B 微调：LoRA（r=8）/ P-Tuning 两套配置齐全，`log/fintune_log/` 为空、无 checkpoint；
- ② LLM zero-shot：分类 / 匹配 / 抽取三脚本齐全，无任何输出文件；
- ③ UIE：脚本未改动，`logs/`、`checkpoints/` 均不存在；
- ④ LLMsTrainer 与 llms_mbti：脚本被改过，但无 `checkpoints/`、无 `log/`，数据压缩步骤也未执行。

### 6.2 建议的补跑优先级

| 优先级 | 子任务 | 理由 | 最小前置条件 |
|---|---|---|---|
| 高 | ② LLM zero-shot | 无需训练，量化后单卡即可跑；能最快产出**定性**案例，补齐报告「应用」章节 | chatglm-6b 权重 + `model.quantize(8)` |
| 高 | ③ UIE | 单卡即可微调；可顺带验证信息抽取基线 | `uie-base-zh` 权重；补 `test.txt` |
| 中 | ① ChatGLM-6B 微调（先 LoRA） | 能验证参数高效微调链路；但需 16GB+ 显存或量化 | 独立 `llm_env` + `peft-chatglm` + 16GB 显存 |
| 低 | ④ LLMsTrainer / llms_mbti | 需多卡 / DeepSpeed；成本最高、框架性大于结论性 | 多卡 + 先跑 `compress_data.py` |

**通用补跑要求**：任何一次补跑都应把 stdout 落盘（`| tee logs/xxx.log`）并保留 `checkpoints/`，才能形成可引用的证据链。

---

## 7. 复现命令（**以下命令本次均未执行**，仅作为补跑入口）

### 7.1 ChatGLM-6B 微调

```sh
cd LLM/chatglm_finetune

# 环境（readme §1）
conda create -n llm_env python=3.8
conda activate llm_env
pip install -r requirements.txt
cd peft-chatglm && python setup.py install && cd ..

# LoRA（train.sh 当前生效的配置）
python train.py \
    --train_path data/mixed_train_dataset.jsonl \
    --dev_path data/mixed_dev_dataset.jsonl \
    --use_lora True \
    --lora_rank 8 \
    --batch_size 1 \
    --num_train_epochs 2 \
    --save_freq 1000 \
    --learning_rate 3e-5 \
    --logging_steps 100 \
    --max_source_seq_len 400 \
    --max_target_seq_len 300 \
    --save_dir checkpoints/finetune \
    --img_log_dir "log/fintune_log" \
    --img_log_name "ChatGLM Fine-Tune" \
    --device cuda:0

# P-Tuning（train.sh 中被注释）
python train.py \
    --train_path data/mixed_train_dataset.jsonl \
    --dev_path data/mixed_dev_dataset.jsonl \
    --use_ptuning True \
    --pre_seq_len 128 \
    --batch_size 1 \
    --num_train_epochs 2 \
    --save_freq 200 \
    --learning_rate 2e-4 \
    --logging_steps 100 \
    --max_source_seq_len 400 \
    --max_target_seq_len 300 \
    --save_dir checkpoints/ptuning \
    --img_log_dir "log/fintune_log" \
    --img_log_name "ChatGLM P-Tuning" \
    --device cuda:0

# 推理 / Playground
python inference.py
streamlit run playground_local.py --server.port 8001
```

> 显存不足时，可给上述命令追加 `--quantization_bit 8`（脚本已支持），并下调 `max_source_seq_len` / `max_target_seq_len`。

### 7.2 LLM zero-shot

```sh
cd LLM/zero-shot
pip install -r requirements.txt
python llm_classification.py        2>&1 | tee logs_zero_shot_cls.log
python llm_text_matching.py         2>&1 | tee logs_zero_shot_match.log
python llm_information_extraction.py 2>&1 | tee logs_zero_shot_ie.log

# prompt 调试平台
streamlit run playground.py --server.port 8001
```

### 7.3 UIE

```sh
cd UIE
pip install -r ../requirements.txt

python train.py \
    --pretrained_model "uie-base-zh" \
    --save_dir "checkpoints/DuIE" \
    --train_path "data/DuIE/train.txt" \
    --dev_path "data/DuIE/dev.txt" \
    --img_log_dir "logs/" \
    --img_log_name "UIE Base" \
    --batch_size 32 \
    --max_seq_len 256 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device cuda:0

python inference.py
# 数据增强平台（需先填入 filling 模型路径）
streamlit run web_da.py --server.port 8904
```

### 7.4 LLMsTrainer 与 llms_mbti

```sh
# ---- LLMsTrainer ----
cd LLM/LLMsTrainer
pip install -r requirements.txt

# 先压缩数据（默认只压缩 pretrain；SFT 需在 compress_data.py 中切换函数）
cd data && python compress_data.py && cd ..

# 继续预训练
sh train_llms.sh configs/accelerate_configs/ds_stage1.yaml \
    configs/pretrain_configs/llama.yaml \
    openlm-research/open_llama_7b_v2

# 指令微调
sh train_llms.sh configs/accelerate_configs/ds_stage1.yaml \
    configs/sft_configs/llama.yaml \
    openlm-research/open_llama_7b_v2

# 奖励模型
sh train_reward_model.sh configs/accelerate_configs/ds_stage1.yaml \
    configs/reward_model_configs/llama.yaml

# ---- llms_mbti ----
cd ../llms_mbti
pip install -r requirements.txt
python get_llms_mbti.py
streamlit run web.py --server.port 8001
```
