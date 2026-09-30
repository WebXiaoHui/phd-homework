# 任务二：从零实现 mini-GPT —— BPE + RoPE + KV Cache 实验报告

> 实验对象：`llm-beginner-master/task-2-mini-gpt/`
> 数据来源：本报告全部指标取自 `task-2-mini-gpt/eval/result.json` 及实际存在的训练产物，未编造任何数值。
> 说明：仓库根 `日志.md` 记录的是**代码产出阶段**（当时未运行），本文以运行后落盘的 `eval/result.json` 为准。

---

## 1. 任务目标与 DoD

### 1.1 一句话目标

用 PyTorch 从零搭一个 decoder-only mini-GPT：手写 BPE 分词器、集成 RoPE 旋转位置编码、实现 KV cache，在中文小语料（唐诗）上预训练到困惑度达标（< 50），并能自回归生成连贯文本。

### 1.2 任务约束

- 禁止使用 `tiktoken` / `sentencepiece`，分词器自己写；
- 位置编码必须用 RoPE，不用绝对位置编码；
- 推理必须带 KV cache，并能解释它省了什么；
- 实现 greedy / top-k / top-p / temperature 四种采样。

### 1.3 Definition of Done（必做 5 项）

| 编号 | DoD 内容 | 自检项 | 通过标准 |
|---|---|---|---|
| M1 | 手写简化版 BPE tokenizer | `tokenizer_roundtrip` | encode→decode 能还原中文/英文 |
| M2 | decoder-only 模型 + RoPE，前向形状正确 | 由 KV cache 与 PPL 项间接验证 | 前向不报错 |
| M3 | 实现 KV cache | `kv_cache_equivalence` | 开/关 cache 的 logits 最大误差 < 1e-4 |
| M4 | 预训练，困惑度达标 | `perplexity_on_dev` | 唐诗档 < 50（TinyStories 档 < 10） |
| M5 | 四种采样策略，能生成连贯文本 | 无自检项，人工交付 | 生成样例对比 |

加分项：S1（参数量扫描 10M/50M/100M）、S2（绝对位置编码 vs RoPE 外推）、S3（KV cache 开关速度对比）、S4（TinyStories 涌现叙事能力）。

---

## 2. 方法与实现要点

### 2.1 项目结构

```
task-2-mini-gpt/
├── src/tokenizer.py   字节级简化 BPE（256 字节基表 + 贪心 merge）
├── src/rope.py        RoPE（half-split，base=10000，动态位置）
├── src/attention.py   causal MHA + KV cache（沿序列维拼接）
├── src/sampling.py    greedy / top-k / top-p / temperature
├── src/model.py       MiniGPT（weight tying）+ load_for_eval
├── train.py           预训练（BPE + 切窗 + next-token + warmup/cosine + grad clip）
├── demo_generate.py   采样策略对比 + --check-cache 自检
├── data/{train.txt,dev.txt,dataset_info.json}
└── ckpt/{best.pt,tokenizer.json}
```

### 2.2 关键设计决策

来自 `日志.md` 第 5 节的设计备忘：

1. **字节级 BPE**：以 256 个字节为基表，通过贪心统计相邻 token 对频率并迭代 merge。字节级方案对中文天然覆盖（任意 Unicode 都由字节拼出），且 roundtrip 有数学保证——merge 只是把相邻字节拼成新 token，任何合法切分解码回字节串再按 UTF-8 展开都能还原原文，不存在未登录字丢失问题。这直接规避了 README 中"中文按字符级会丢未登录字"的坑。
2. **RoPE half-split**：与 LLaMA 约定一致，频率动态生成、不预设上限表，长序列不会出现 shape 报错；RoPE 只作用于 Q/K，不加在 V 上。
3. **KV cache 结构**：每层一份 `(k, v)`，模型 forward 接收长度为 `n_layer` 的 cache 列表，返回拼接后的新 cache；生成时先用整段 prompt 预热 cache，再逐 token 增量解码。
4. **增量位置接续**（最关键的实现点）：增量解码时新 token 的 RoPE 位置必须接续历史长度，offset 由 `cache[0].shape[2]` 取得。若位置从 0 重新计数，RoPE 角度错误，`kv_cache_equivalence` 会直接失败。
5. **cache 沿序列维拼接**：`torch.cat(..., dim=2)`，而非 batch 或 head 维。
6. **weight tying + GPT-2 风格残差缩放**：输入 embedding 与输出 lm_head 共享权重，作为工程加分点。
7. **采样实现顺序**：temp → top-k → softmax → top-p → multinomial；`temperature <= 0` 退化为 greedy，避免除零。

### 2.3 训练脚本默认超参

`train.py` argparse 默认值（实际运行是否采用默认未在产物中留痕，此处仅说明脚本默认）：

| 参数 | 默认值 |
|---|---|
| `vocab_size` | 1024（256 字节 + merges） |
| `n_embd` | 256 |
| `n_head` | 4 |
| `n_layer` | 4 |
| `block_size` | 128 |
| `dropout` | 0.1 |
| `epochs` | 20 |
| `batch_size` | 64 |
| `lr` | 3e-4 |
| `warmup_frac` | 0.05 |
| `clip` | 1.0 |

数据集信息（`data/dataset_info.json`）：`dataset = poetry`，`train = train.txt`，`dev = dev.txt`，`ppl_threshold = 50`。语料体量：`train.txt` 44,087 B、`dev.txt` 4,916 B、`poetry.txt` 48,133 B（约 49 KB 的唐诗 quick-start 档）。

---

## 3. 实验结果

以下为 `task-2-mini-gpt/eval/result.json` 的完整内容整理：

| 测试名 | 结果 | 关键指标 | 阈值 | 对应 DoD |
|---|---|---|---|---|
| `tokenizer_roundtrip` | **通过** | `failures = []` | 无还原失败 | M1 |
| `kv_cache_equivalence` | **通过** | `max_abs_diff = 3.814697265625e-06`（≈ 3.8e-6） | < 1e-4 | M3 |
| `perplexity_on_dev` | **未通过（失败）** | `perplexity = 65.48`；`n_tokens = 3156`；`dataset = poetry` | < 50.0 | M4 |

产物核查：

| 产物 | 状态 | 体量 |
|---|---|---|
| `ckpt/best.pt` | 存在 | 13,689,290 B（约 13.1 MB） |
| `ckpt/tokenizer.json` | 存在 | 9,261 B |
| `data/train.txt` / `data/dev.txt` | 存在 | 44,087 B / 4,916 B |
| PPL 曲线 / 生成样例落盘 | 未发现 | 训练脚本 `--save-curve` 默认 `None`，故未生成 |

### 3.1 DoD 完成情况

| 编号 | 结论 | 依据 |
|---|---|---|
| M1 | ✅ 达标 | tokenizer roundtrip 对中文与英文样例均无失败 |
| M2 | ✅ 达标 | 模型完成训练并做前向与困惑度评估，说明 decoder-only + RoPE 前向、形状、梯度链路可用 |
| M3 | ✅ 达标 | KV cache 开关 logits 误差 3.8e-6，比 1e-4 阈值小约 26 倍 |
| M4 | ❌ **未达标** | dev 困惑度 65.48 > 阈值 50 |
| M5 | ⚠️ 无法确认 | `sampling.py` 与 `demo_generate.py` 已实现且通过语法检查，但无生成样例落盘，自检不覆盖 |

加分项 S1–S4 均未在 `eval/result.json` 中体现。

---

## 4. 结果分析

### 4.1 已达标项分析

- **M1（BPE）**：三个测试样例 `床前明月光`、`Hello, world!`、`深度学习需要数学基础` 全部无损还原。字节级设计的价值在此体现：中文多字节字符在 merge 表下可被稳定合并与还原，不存在字符级分词器的 OOV 问题。
- **M2（decoder-only + RoPE）**：模型能完成训练与推理，且 `block_size` 属性被自检正确读取用于切窗——自检的 `perplexity_on_dev` 按模型上下文长度非重叠分块累加 NLL，说明接口契约完全对齐。
- **M3（KV cache）**：3.8e-6 的误差量级说明增量解码与全量前向在数值上等价。这条自检是任务二"最容易挂"的点（见 `日志.md`），通过它意味着 RoPE 的增量位置接续正确、cache 拼接维度正确。误差来源是浮点累加顺序差异而非逻辑错误。

### 4.2 失败项 `perplexity_on_dev` 的根因分析

实测困惑度 **65.48**，阈值 **50**，差距约 31%。结合语料与实现，可能原因如下（按可能性排序）：

1. **语料规模过小（最可能）**：唐诗档仅约 49 KB，`train.txt` 44 KB。这是设计上的 quick-start 档，README 明确说明它的用途是"5 分钟跑通 pipeline、验证代码正确性"，而非追求困惑度。数据量不足时，模型很难把 1024 词表的字节级 BPE 分布学到足够平滑，dev 与 train 同源但样本极少，困惑度天然偏高。
2. **字节级分词使有效上下文变短**：`block_size = 128`（默认）。中文一个字在 UTF-8 下占 3 字节，若 merge 尚未把常见汉字合并为单 token，128 个 byte-token 仅覆盖约 40 个汉字，next-token 任务可用的上下文语义非常有限，长距离依赖难以建模。
3. **过拟合与早停策略的相互影响**：`train.py` 按最低 dev ppl 保存 best。小语料上 dev ppl 容易先降后升，若 best 落在较早的 epoch，会保存一个"还没充分训练"的模型；反之若训练过久则过拟合。日志中也提示"唐诗小语料易过拟合：困惑度先降后升就该早停"。
4. **评测样本量小、估计噪声大**：`n_tokens = 3156`（自检最多取 dev 文本前 4096 个 token）。仅 3156 个 token 的 PPL 估计方差较大，单次运行的 65.48 不宜过度解读为稳定的 65 左右。
5. **词表规模与语料的匹配**：`vocab_size = 1024` 对 49 KB 语料偏大，可能导致低频 merge 学得不够充分。

### 4.3 改进方向

1. **换用更充分的语料**：按 README 三档渐进，改用 TinyStories 或中文故事语料（阈值相应变为 < 10），或把 SkyPile 子集纳入。语料规模是提高 PPL 上限最直接的手段。
2. **增大 `block_size`**：从 128 提到 256/512，让模型看到更长上下文；同时保证训练与推理的上下文长度一致。
3. **提高 `vocab_size` 或改混合分词**：扩大 merge 数量（如 4096/8192），减少中文单字所需 token 数，等价于加长有效上下文。
4. **延长训练并结合早停**：增加 `epochs`、观察 dev ppl 拐点；`--save-curve` 落盘 PPL 曲线以判断是否欠训练或过拟合。
5. **补充 M5 与加分项**：跑 `demo_generate.py` 落盘四种采样策略的生成对比，并做 S1（参数量扫描）、S3（KV cache 开关计时）、S2（RoPE vs 绝对位置外推）。README 要求的"困惑度曲线 + 几段生成样例"目前都缺少落盘证据。

---

## 5. 踩坑与经验

| 坑 | 现象 | 本质与规避 |
|---|---|---|
| BPE merge 顺序错 / decode 跨 UTF-8 边界 | roundtrip 失败 | 每轮必须取当前频率最高的对；字节级 decode 先整体展开字节串再 `utf-8` 解码 |
| 增量解码位置没接续 | `kv_cache_equivalence` 直接挂 | 新 token 的 RoPE 角度必须用 `offset = 历史长度`，由 cache 的序列维取得 |
| cache 拼接维度搞反 | logits 对不上 | 应在序列维 `dim=2` 拼接 |
| RoPE 加到了 V 上 | 数值与预期不符 | RoPE 只作用于 Q/K |
| `head_dim` 为奇数 | RoPE half-split 报错 | 代码内置偶数断言；`n_embd / n_head` 必须为偶数 |
| 一次性喂超长序列算 PPL | shape 报错或 PPL 失真 | 自检已按 `block_size` 非重叠分窗，长序列落进外推区会使 PPL 虚高 |
| loss 偶发 spike | 训练不稳 | 已加 gradient clipping；仍不稳则调小 lr |

**经验**：KV cache 的"等价性自检"是整条 pipeline 中最有价值的设计——它把最容易写错的 RoPE 位置偏移问题转化成一个可量化的数值阈值（1e-4），使错误在训练之前就被暴露，而不必等到生成质量变差才发现。

---

## 6. 结论

任务二的三项自检呈现"两过一挂"：手写字节级 BPE 的 roundtrip 无损（M1 达标）、KV cache 与全量前向数值等价（误差 3.8e-6，M3 达标）、decoder-only + RoPE 前向链路可用（M2 达标）；但 **dev 困惑度 65.48 未达到 50 的阈值（M4 未达标）**，评测集为 poetry，仅有 3156 个 token 参与估计。

从工程实现看，任务二真正困难的两点——KV cache 的位置接续与 RoPE 的数值一致性——已经做对，说明核心机制掌握到位；未达标项更可能来自小语料 quick-start 档的固有上限（约 49 KB 唐诗），而非实现缺陷。建议按 4.3 节换用更大语料、加大 `block_size` 与训练量后重测，同时补齐生成样例与消融实验，即可形成完整交付。

---

## 7. 复现命令

```bash
source .venv/bin/activate && cd task-2-mini-gpt
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com        # 国内镜像，必设

# 1. 数据（默认唐诗 quick-start 档；也可换 tinystories / skypile）
python data/download.py
python data/download.py --dataset tinystories
python data/download.py --dataset skypile

# 2. 训练：内部先训 BPE 并做 roundtrip 自检，再预训练并按最低 dev ppl 存 best.pt
python train.py
python train.py --save-curve figures/ppl_curve.png

# 3. 自检：三项（roundtrip / KV cache 等价 / dev 困惑度）
python eval/run.py

# 4. 生成对比（M5）
python demo_generate.py --prompt "床前明月光"
python demo_generate.py --prompt "春眠不觉晓" --check-cache
```

> 若中途缺 ckpt/数据，自检显示 `[跳过]`（不是错误），补齐后重跑即可。

---

**附：`eval/result.json` 原始内容**

```json
[
  { "test": "tokenizer_roundtrip", "pass": true, "failures": [] },
  { "test": "kv_cache_equivalence", "pass": true, "max_abs_diff": 3.814697265625e-06 },
  { "test": "perplexity_on_dev", "pass": false, "perplexity": 65.48,
    "threshold": 50.0, "n_tokens": 3156, "dataset": "poetry" }
]
```
