# 任务一：熟悉 Transformer —— 手写 Self-Attention 与 Transformer Encoder 实验报告

> 实验对象：`llm-beginner-master/task-1-transformer/`
> 数据来源：本报告全部指标取自 `task-1-transformer/eval/result.json` 与工作区中实际存在的训练产物，未做任何数值编造。
> 说明：仓库根 `日志.md` 记录的是**代码产出阶段**的状态（当时受"只写代码不运行"约束，标注"均未运行验证"）。本文以运行后落盘的 `eval/result.json` 为准。

---

## 1. 任务目标与 DoD

### 1.1 一句话目标

用约 300 行 PyTorch 从零实现 Transformer encoder，在 ChnSentiCorp 中文情感二分类上取得 dev 准确率 ≥ 0.80（参考基线约 0.85），并用注意力热图解释模型"在看什么"。

### 1.2 任务约束

- 禁止调用 `nn.MultiheadAttention` 或任何高层封装；
- 禁止加载预训练模型；
- 手写 scaled dot-product attention（缩放、softmax、mask）；
- 手写 multi-head attention 与完整 Transformer encoder block（attention + FFN + residual + LayerNorm）；
- 用 padding mask 做文本分类，用 causal mask 做 toy 语言模型。

### 1.3 Definition of Done（必做 5 项）

| 编号 | DoD 内容 | 自检项 | 通过标准 |
|---|---|---|---|
| M1 | 手写 `scaled_dot_product_attention` | `attention_correctness` | 与 `F.scaled_dot_product_attention` 最大绝对误差 < 1e-5 |
| M2 | 手写 `MultiHeadAttention` + `TransformerBlock` | 前向不报错、形状正确（被 M3 的训练间接验证） | 前向正常 |
| M3 | ChnSentiCorp 上训练分类器 | `classifier_accuracy` | dev 准确率 ≥ 0.80 |
| M4 | causal mask 不泄漏 | `causal_mask` | 篡改未来位置 V 后，过去位置输出变化 < 1e-6 |
| M5 | ≥ 3 张注意力热图 | 无自检项，属人工交付 | `figures/` 下 ≥ 3 张 |

加分项：S1（head/层数消融）、S2（拆 residual / LayerNorm）、S3（dev acc > 0.88）、S4（绝对位置编码换 RoPE）。

---

## 2. 方法与实现要点

### 2.1 项目结构

```
task-1-transformer/
├── src/attention.py        手写 scaled_dot_product_attention + MultiHeadAttention
├── src/block.py            TransformerBlock（Pre-LN）+ PositionwiseFFN
├── src/tokenizer.py        字符级 tokenizer（<pad>=0, <unk>=1，定长截断补齐）
├── src/model.py            TransformerClassifier + load_for_eval
├── train.py                训练脚本（warmup + cosine、梯度裁剪、按 dev acc 存 best）
├── causal_demo.py          causal mask 泄漏自检 + 可选 toy LM
├── visualize_attention.py  注意力热图
├── data/{train,validation,test}.parquet
├── ckpt/best.pt
└── eval/figures/train_curve.png
```

### 2.2 关键设计决策

设计决策记录于 `日志.md` 第 3 节，概括如下：

1. **字符级分词**：中文情感分类不依赖预训练词表，自包含、省下载；词表随 ckpt 一起保存（`cfg`/`vocab` 存入 ckpt dict），使 `load_for_eval` 只读 ckpt 即可重建 tokenizer，保证训练与评测的分词严格一致。
2. **定长 + masked-mean pooling**：tokenizer 输出固定 `max_len`（默认 200），右侧补 PAD；分类头前的 mean pooling 只对非 PAD 位置求均值，避免 PAD 污染句向量——这直接对应 README 列出的"mean pooling 没排除 padding"这一常见坑。
3. **Pre-LN 结构**：`TransformerBlock` 采用 Pre-LayerNorm（residual 在子层之外），与主流现代 decoder 结构一致；FFN 为 `Linear → GELU → Dropout → Linear → Dropout`，中间维度 `4 × d_model`。
4. **mask 语义统一**：全项目约定 `True = 被屏蔽`。padding mask 形状 `(B,1,1,T)`，屏蔽 key 侧的 PAD 列；causal mask 为 `(T,T)` 上三角。该约定与 `torch.nn.functional.scaled_dot_product_attention` 的语义一致，因此自检可直接比对数值。
5. **mask 实现用 `-inf` 而非乘 0**：被屏蔽位置在 softmax 之前填 `-inf`，从根本上避免概率泄漏；这也是 `causal_mask` 自检能得到 `leaked_diff = 0.0` 的原因。
6. **缩放因子为 `1/sqrt(d_k)`**（`d_k` 为单头维度），multi-head reshape 顺序为 `(B,T,D) → (B,T,H,d_k) → (B,H,T,d_k)`，拼回前调用 `.contiguous()`。
7. **注意力权重旁路记录**：自检要求 `scaled_dot_product_attention` 返回单个张量，为不破坏契约，注意力权重通过 `MultiHeadAttention.last_attn_weights` 单独记录，专供可视化脚本读取。
8. **训练工程**：AdamW + 线性 warmup + cosine 退火 + 梯度裁剪 + 按 dev accuracy 保存 best checkpoint。

### 2.3 训练脚本默认超参

`train.py` 的 argparse 默认值（实际运行时是否全部采用默认值未在产物中留痕，故此处仅说明脚本默认）：

| 参数 | 默认值 |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `max_len` | 200 |
| `dropout` | 0.1 |
| `epochs` | 8 |
| `batch_size` | 64 |
| `lr` | 3e-4 |
| `warmup_frac` | 0.1 |
| `clip` | 1.0 |

数据方面，ChnSentiCorp 已下载为 `data/{train,validation,test}.parquet`，自检读取 `validation.parquet` 计算 dev 准确率。

---

## 3. 实验结果

以下为 `task-1-transformer/eval/result.json` 的完整内容整理：

| 测试名 | 结果 | 关键指标 | 阈值 | 对应 DoD |
|---|---|---|---|---|
| `attention_correctness` | **通过** | `max_abs_diff = 9.5367431640625e-07`（≈ 9.5e-7） | < 1e-5 | M1 |
| `causal_mask` | **通过** | `leaked_diff = 0.0` | < 1e-6 | M4 |
| `classifier_accuracy` | **通过** | `accuracy = 0.8608` | ≥ 0.80（参考基线 0.85） | M3 |

产物核查（仅看存在性与体量，不读取大文件内容）：

| 产物 | 状态 | 体量 |
|---|---|---|
| `ckpt/best.pt` | 存在 | 5,538,157 B（约 5.3 MB，落在 README 预期 5–50 MB 区间） |
| `eval/figures/train_curve.png` | 存在 | 46,407 B |
| `data/{train,validation,test}.parquet` | 存在 | 2,142,673 / 271,249 / 270,218 B |
| attention 热图（M5） | **未在工作区发现** | 工作区内仅检索到 `eval/figures/train_curve.png` 一张 PNG |

### 3.1 DoD 完成情况

| 编号 | 结论 | 依据 |
|---|---|---|
| M1 | ✅ 达标 | `attention_correctness` 通过，误差 9.5e-7，比阈值小一个数量级 |
| M2 | ✅ 达标 | 分类器完成训练并达到 0.8608 dev acc，说明 `MultiHeadAttention` + `TransformerBlock` 前向、形状、梯度链路全部可用 |
| M3 | ✅ 达标 | dev acc = 0.8608 ≥ 0.80，且高于参考基线 0.85 |
| M4 | ✅ 达标 | `causal_mask` 通过，`leaked_diff = 0.0`；`causal_demo.py` 已提供同性质的独立复现 |
| M5 | ⚠️ 无法确认 | `eval/result.json` 不含该项；工作区内未发现注意力热图文件，仅发现训练曲线 |

加分项：`eval/result.json` 未记录 S1–S4 的消融结果；S3（> 0.88）从 0.8608 看未达成。

---

## 4. 结果分析

### 4.1 达标情况

三项自检全部通过，M1–M4 均达标，核心实现正确性得到数值级验证：

- **attention 数值正确**：9.5e-7 的误差量级说明缩放因子、softmax 维度、mask 语义、multi-head 拼接全部与官方实现对齐。若缩放因子误写为 `sqrt(d_model)`，误差会远大于 1e-5；若 mask 用乘 0 而非 `-inf`，`causal_mask` 也无法得到 0 泄漏。
- **causal mask 零泄漏**：`leaked_diff` 精确为 `0.0`，表明 mask 在 softmax 之前生效且未来位置对过去输出的梯度/数值通路被彻底切断。这同时是任务二的预热验证。
- **分类准确率超基线**：0.8608 高于 0.80 的通过线与 0.85 的参考基线，说明 padding mask 与 masked-mean pooling 都被正确使用；若 PAD 参与了 attention 或 mean pooling，通常难以稳定超过 0.85。

### 4.2 未达标/无法确认项的根因

**M5（注意力热图）无法确认**。从产物看，只落盘了训练曲线 `eval/figures/train_curve.png`，没有注意力热图 PNG。可能原因是可视化脚本虽已写好（`visualize_attention.py`，默认取验证集正面/负面/最长句各一张），但本次运行未执行该步骤，或产图写到了工作区之外。由于 M5 不进入 `eval/result.json` 的自动判定，属于"自检覆盖不到的人工交付项"，容易被遗漏。建议补跑：

```bash
python visualize_attention.py
python visualize_attention.py --head 0
python visualize_attention.py --layer 1
```

并确认 `figures/`（README 约定路径）或脚本实际输出目录下存在 ≥ 3 张热图，同时按 README 的问题清单做人工解读：正面样本上"不错/赞"类词、负面样本上"差/失望"类词的注意力权重是否被抬高。

### 4.3 可改进之处

1. **补齐 M5 与消融实验**：S1（`--n-layers 1/2/4`、`--n-heads 1/2/4`）、S2（拆 residual/LayerNorm）、S4（绝对位置编码 vs RoPE）均未在结果中体现，补跑后才能形成 README 要求的"配置 | dev acc | 观察"表。
2. **冲击 S3（> 0.88）**：可增大 `d_model` 至 192/256、增加层数、延长 `epochs`，并检查 `max_len=200` 是否截断了长评论。
3. **热图解读与准确率的联合论证**：仅报准确率不能说明模型"看对了关键词"，需把热图与误判样本对齐分析。

---

## 5. 踩坑与经验

来源为 `日志.md` 记录的代码产出阶段与 `操作流程.md` 的排错表，结合实测结果复核：

| 坑 | 现象 | 本质与规避 |
|---|---|---|
| mask 用乘 0 | softmax 后仍有概率泄漏 | 必须在 softmax 前填 `-inf`；本次 `causal_mask` = 0.0 即该点做对 |
| 缩放因子写错 | 与官方实现误差超线 | 应为 `1/sqrt(d_k)`；实测 9.5e-7 证明写对 |
| multi-head reshape 漏 `.contiguous()` | `view` 报错 | 拼接回 `(B,T,D)` 前必须连续化 |
| pooling 未排除 PAD | dev acc 上不去 | 使用 masked-mean pooling，只对非 PAD 位置求均值 |
| padding mask 未逐层传入 | dev acc 偏低 | 每层 block 都要接收同一 mask |
| 在任务目录外运行 `eval/run.py` | import 失败 | 自检依赖仓库根 `_eval_harness.py`，必须在仓库内运行 |
| Windows 控制台中文乱码 | 输出乱码 | 运行壳已用 `sys.stdout.reconfigure` 处理；`result.json` 始终 UTF-8 |

**工程经验**：把 mask 语义（`True = 屏蔽`）在项目内统一成一条约定，可以让 padding 分类与 causal toy LM 复用同一份 attention 实现，既减少重复代码，也让自检的数值比对成为可能。把"看不见的错"变成"看得见的数值"（`max_abs_diff`、`leaked_diff`）是本任务最有价值的工程习惯。

---

## 6. 结论

任务一的必做项 M1–M4 已用实测数值验证通过：attention 与官方实现误差 9.5e-7、causal mask 泄漏 0.0、ChnSentiCorp dev 准确率 0.8608（超过 0.80 通过线与 0.85 参考基线）。这证明从零手写的 scaled dot-product attention、multi-head attention、Pre-LN Transformer encoder block 以及 padding mask 分类流程在数学与工程上都正确。

唯一不能从现有数据确认的是 M5（≥ 3 张注意力热图）——工作区内只有训练曲线，没有热图产物；建议按第 4.2 节的命令补跑并补做 S1–S4 消融，即可形成完整提交。

后续任务二（mini-GPT）直接复用本任务验证过的 causal mask 实现，衔接顺畅。

---

## 7. 复现命令

```bash
# 0. 环境（建议仓库根建 venv，六个任务可共用）
python3 -m venv .venv && source .venv/bin/activate

cd task-1-transformer
pip install -r requirements.txt

# 1. 国内镜像（本机实测 huggingface.co 直连不通，hf-mirror.com 可达）
export HF_ENDPOINT=https://hf-mirror.com

# 2. 下载数据，应生成 data/{train,validation,test}.parquet
python data/download.py

# 3. 单元测试（不依赖 checkpoint）：前两关通过，第三关在无 ckpt 时跳过
python eval/run.py

# 4. 训练分类器（保存 ckpt/best.pt，可顺带存训练曲线）
python train.py
python train.py --save-curve figures/train_curve.png

# 5. causal mask 演示（M4）
python causal_demo.py
python causal_demo.py --train-toy

# 6. 注意力热图（M5，需先有 ckpt/best.pt）
python visualize_attention.py
python visualize_attention.py --head 0
python visualize_attention.py --layer 1

# 7. 最终自检（三项全绿并写入 eval/result.json）
python eval/run.py
```

> ⚠️ 必须在仓库内运行 `python eval/run.py`，它依赖仓库根 `_eval_harness.py`。

---

**附：`eval/result.json` 原始内容**

```json
[
  { "test": "attention_correctness", "pass": true, "max_abs_diff": 9.5367431640625e-07 },
  { "test": "causal_mask", "pass": true, "leaked_diff": 0.0 },
  { "test": "classifier_accuracy", "pass": true, "accuracy": 0.8608, "baseline_reference": 0.85 }
]
```
