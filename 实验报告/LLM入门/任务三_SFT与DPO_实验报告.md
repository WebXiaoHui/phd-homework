# 任务三：指令微调与偏好对齐（SFT + DPO）实验报告

> 实验对象：`llm-beginner-master/task-3-sft-dpo/`
> 数据来源：自检指标取自 `task-3-sft-dpo/eval/result.json`；定性对比取自 `task-3-sft-dpo/eval/compare_results.json`；训练规模取自 `ckpt/sft/adapter_config.json` 与 `ckpt/dpo/adapter_config.json`。未编造任何数值。
> 说明：仓库根 `日志.md` 记录的是**代码产出阶段**（当时未运行），本文以运行后落盘的产物为准。

---

## 1. 任务目标与 DoD

### 1.1 一句话目标

在 Qwen2.5-0.5B 上手写 LoRA，跑通 SFT + DPO 两阶段对齐：SFT 用 MOSS 中英双语对话数据，DPO 在 SFT 之上继续训练，自检三项全过，并能在同一批指令上观察到 base / SFT / DPO 的可解释差异。

### 1.2 任务约束

- 不许调用 PEFT 现成 LoRA，低秩注入必须自己写；
- chat template 套 Qwen 官方格式，loss 只算 assistant turn；
- 需要 freeze 的 reference model，DPO 用 `-log σ(β·margin)`。

### 1.3 Definition of Done（必做 4 项）

| 编号 | DoD 内容 | 自检项 | 通过标准 |
|---|---|---|---|
| M1 | 手写 LoRA 低秩注入 | `lora_param_count` | 注入后可训参数占比 < 5% |
| M2 | Qwen chat template + loss masking | `loss_masking` | mock 多轮对话中 -100 占比落在 20%–90%，user/system 全 -100 |
| M3 | MOSS-003-sft 跑 SFT，产出非空 `ckpt/sft/` | `sft_vs_base` | `ckpt/sft/` 非空（仅校验产物存在，质量人工对比） |
| M4 | SFT 之上跑 DPO，产出 `ckpt/dpo/`，并做 base/SFT/DPO 对比 | 无自动自检，人工交付 | `src/compare.py` 输出的定性对比 |

关键提示：**`sft_vs_base` 这一项只是"校验 `ckpt/sft` 非空"的存在性检查**，它并不衡量输出质量；输出质量必须看 `compare_results.json` 的定性对比。本报告第 4 节按此口径展开。

加分项：S1（全量微调 vs LoRA）、S2（LoRA rank 消融 4/8/16/32）、S3（灾难性遗忘 C-Eval）、S4（SFT-only vs SFT+DPO 偏好差异 + reward margin 曲线）、S5（moss-003-sft-plugin 贯通任务五）。

---

## 2. 方法与实现要点

### 2.1 项目结构

```
task-3-sft-dpo/
├── src/lora.py       手写 LoRA：LoRALinear / inject_lora / merge_lora / save_adapter / load_adapter
├── src/chat.py       format_messages（Qwen 模板）+ build_labels（loss masking）
├── src/dataset.py    MOSS SFT / DPO 偏好数据的容错解析与 token 化
├── src/compare.py    base / SFT / DPO 同指令对比
├── train_sft.py      SFT（LoRA + assistant-only loss）
├── train_dpo.py      DPO（policy + freeze ref）
├── ckpt/sft/adapter.{bin,config.json}
├── ckpt/dpo/adapter.{bin,config.json}
├── models/Qwen2.5-0.5B/
└── eval/{result.json,compare_results.json}
```

### 2.2 LoRA 实现要点

来自 `日志.md` 第 6.4 节的设计决策：

1. **A/B 形状**：A 为 `(in, r)`、B 为 `(out, r)`，增量 `delta = (x @ A) @ Bᵀ · (alpha/r)`，与 `y = Wx + scaling·B(Ax)` 数值等价。
2. **初始化**：A 用 kaiming、B 全零，保证训练初始前向恰好等于原权重（零扰动的 warm start），这也是 README 强调的坑点。
3. **scaling = alpha / r**：换 rank 时等效学习率不会漂移。
4. **冻结策略**：`inject_lora` 先把全部参数设为 `requires_grad=False`，再单独放行 `lora_A / lora_B`；embedding 与 lm_head 一并冻结，因此可训占比远低于 5%。
5. **merge_lora**：把 `scaling·(B @ Aᵀ)` 合回原权重并关闭低秩分支，用于推理落盘。
6. **adapter 产物结构**：`adapter_config.json`（target_modules / r / alpha / meta）+ `adapter.bin`（仅含 `*.lora_A / *.lora_B`）。

### 2.3 Chat template 与 loss masking

1. `format_messages` 套 Qwen 官方 `<|im_start|>role\n...<|im_end|>` 模板。
2. `build_labels` 在完整 `input_ids` 上对每个 assistant 内容做**连续子序列定位**，不依赖 offset map，从而兼容快/慢 tokenizer（Qwen 为 GPT-2 风格 BPE，空白与特殊 token 不跨段 merge，保证内容 token 是整段 ids 的连续子序列）。
3. user / system / 角色头全部 -100；assistant 内容及其收尾 `<|im_end|>` 保留监督（否则模型学不会自己结束回合）；多轮对话中**每个 assistant turn 都参与训练**。

### 2.4 DPO 实现要点

1. policy = 基座 + SFT adapter；ref = 另一个基座 + 同一份 SFT adapter，`requires_grad=False` 且 `no_grad` 只跑 forward。
2. 每个 batch 共 4 次 forward（policy × chosen/rejected，ref × chosen/rejected）。
3. 损失 `-log σ(β·margin)`，默认 `β = 0.1`；聚合用论文口径的 sum-of-logp。
4. 监控 margin > 0 的比例，而不只看 loss。

### 2.5 实际训练规模（来自 adapter_config.json，非估计）

这部分是评估结论时必须知道的事实前提：

| 阶段 | r | alpha | target_modules | 样本数 | epochs | steps | 最终 loss | 其他 |
|---|---|---|---|---|---|---|---|---|
| SFT | 8 | 16.0 | `q_proj`, `v_proj` | **59** | 1 | 7 | 1.3887 | max_len = 1024 |
| DPO | 8 | 16.0 | `q_proj`, `v_proj` | **800** | — | 100 | 0.5394 | β = 0.1 |

即：录得的 SFT 适配器来自一次**最小规模 pipeline 验证**（59 条样本），DPO 使用了 800 对偏好数据、100 步。两个 `adapter.bin` 均约 1.06 MB。数据侧已就位：`data/moss-sft/moss-003-sft-data_1w.jsonl`（约 76 MB）与 `data/dpo-en-zh-20k/`（含 `dpo_en.jsonl` ≈ 49.7 MB、`dpo_zh.jsonl` ≈ 27.7 MB）——说明数据规模本身充足，但本次训练只用了其中很小的一部分。

---

## 3. 实验结果

### 3.1 自检指标（`eval/result.json`）

| 测试名 | 结果 | 关键指标 | 阈值 | 对应 DoD |
|---|---|---|---|---|
| `lora_param_count` | **通过** | `trainable = 540,672`，`total = 494,573,440`，`ratio = 0.00109` | < 0.05 | M1 |
| `loss_masking` | **通过** | `mask_ratio = 0.511` | ∈ (0.2, 0.9) | M2 |
| `sft_vs_base` | **通过** | 仅校验 `ckpt/sft` 非空 | 非空 | M3 |

产物核查：

| 产物 | 状态 | 体量 |
|---|---|---|
| `ckpt/sft/adapter.bin` + `adapter_config.json` | 存在 | 1,111,491 B + 328 B |
| `ckpt/dpo/adapter.bin` + `adapter_config.json` | 存在 | 1,111,491 B + 197 B |
| `models/Qwen2.5-0.5B/model.safetensors` | 存在 | 约 988 MB |
| `eval/compare_results.json` | 存在 | 5,363 B |

### 3.2 定性对比（`eval/compare_results.json` 四道同题）

四道题依次为：①"什么是机器学习？请用三句话解释。" ②"把下面这句话翻译成英文：深度学习让我学到了很多有意思的知识。" ③"写一个 Python 函数，判断一个字符串是不是回文。" ④"用一句话介绍 LoRA 的原理。"

以下逐题摘要（引用为 `compare_results.json` 原文片段，均做了截断）。

#### 题目① 什么是机器学习？（三句话解释）

- **base**：把 prompt 原样吐回，并夹带明显噪声 token，形成退化复读：

  > `什么是机器学习？请用三句话解释。})();\nticauser\n什么是机器学习？请用三句话解释。})();\nticauser\n…`

- **SFT**：给出连贯的三句解释，覆盖监督/无监督/强化学习与应用领域，指令遵循成立。

- **DPO**：同样成句、结构更紧凑，先定义再讲机制再讲应用：

  > `机器学习是一种人工智能技术，它使计算机能够通过数据和经验来自动改进和优化其性能。…可以应用于各种领域，包括自然语言处理、图像识别、推荐系统和金融分析等。`

**结论**：base → SFT 完成从"复读"到"回答"的质变；DPO 在"三句话"这一约束上比 SFT 更收敛、更有条理。

#### 题目② 中译英

- **base**：复读 + `ticauser` 噪声。
- **SFT**：直接原样输出中文，**未完成翻译**：

  > `深度学习让我学到了很多有意思的知识。`

- **DPO**：仍是中文，仅把"有意思"换成"有趣"：

  > `深度学习让我学到了很多有趣的知识。`

**结论**：翻译这一能力在 SFT/DPO 两阶段都没有真正学到。原因是训练数据为中文对话（MOSS 中文为主 + 800 对 DPO 偏好），模型没有获得稳定的中英对齐信号；DPO 只做了同义改写，没有改变任务类型。这是典型的"数据分布决定能力边界"。

#### 题目③ 回文函数

- **base**：确实吐出了一段函数，但随后迅速退化为模板/控制符复读：

  > `def is_palindrome(s):\n    s = s.lower()\n    s = s[::-1]\n    return s == s[::-1]\nitosuser\n…UAGEassistant\ndef is_palindrome(s):…`

  注意该实现先 lower 再取反，逻辑上等价于比较 `s.lower() == s.lower()[::-1]`，勉强可运行，但没有任何输入清洗。

- **SFT**：以"当然可以！"起头，给出**完整、可运行、带边界处理**的实现：

  > `cleaned_s = ''.join(c.lower() for c in s if c.isalnum())\n    \n    # 比较字符串的前半部分和后半部分是否相同\n    return cleaned_s == cleaned_s[::-1]`

- **DPO**：与 SFT 同构，变量名改为 `char`，代码风格一致：

  > `cleaned_s = ''.join(char.lower() for char in s if char.isalnum())\n    \n    # 比较原始字符串和其反转版本\n    return cleaned_s == cleaned_s[::-1]`

**结论**：这是 SFT 最实在的收益——从"半退化但碰巧能跑"变成"结构完整、有注释、处理非字母数字字符"的工程级回答。DPO 在此题上与 SFT 基本持平，说明 DPO 主要改变偏好风格而非新学知识。

#### 题目④ 用一句话介绍 LoRA（关键失败案例）

- **base**：把 LoRA 说成"基于自适应学习的神经网络架构"，属幻觉。

- **SFT**：换成另一套幻觉——"自适应学习器"：

  > `LoRA 是一种用于深度学习模型的自适应学习方法，…通过在模型的每个层上添加一个自适应学习器…`

- **DPO**：**错误进一步升级为概念混淆**，把 LoRA 直接等同于 LSTM，并且整段自我重复：

  > `LoRA（Long Short Term Memory）是一种神经网络架构，用于处理长时记忆数据，通过引入长期依赖性来提高模型的泛化能力和性能。…（后半段几乎逐句重复）`

**结论**：LoRA 原理这一知识型问题，三个阶段**全部答错**，且 DPO 后错误更"自信"、更具体（主动补上了一个错误的英文全称）。这与 DPO 的数据规模直接相关：本次 DPO 仅 800 对偏好样本、100 步，偏好对齐只能微调"偏好哪种表达风格"，无法注入新的参数高效微调知识；同时 SFT 只用了 59 条样本、1 个 epoch，也不足以覆盖该概念。偏好优化在此把"不确定性"推向了一个更流畅但更错误的答案，这是小数据 DPO 的典型风险。

### 3.3 DoD 完成情况

| 编号 | 结论 | 依据 |
|---|---|---|
| M1 | ✅ 达标 | 可训参数 540,672 / 494,573,440 = 0.109%，约为 5% 上限的 1/46 |
| M2 | ✅ 达标 | mask 比例 0.511，落在 20%–90% 中位附近；形状与 ids 一致（自检未报 shape 错误） |
| M3 | ✅ 达标（存在性） | `ckpt/sft/` 非空；但请注意该项**不衡量质量** |
| M4 | ⚠️ 部分达标 | `ckpt/dpo/` 已产出、`compare_results.json` 已完成三模型对比；但对比结果显示翻译与 LoRA 知识两项仍失败，仅风格/格式层面有可观察改进 |

---

## 4. 结果分析

### 4.1 达标情况

- **LoRA 注入正确**：0.109% 的可训占比说明冻结策略彻底（embedding 与 lm_head 也冻结），且 A/B 形状与 scaling 实现正确。对比 README 给出的 `["q_proj","v_proj"], r=8, alpha=16` 固定自检参数，接口完全对齐。
- **loss masking 正确**：0.511 的 -100 占比正处在"一半监督、一半屏蔽"的合理位置。若 user/system 未全屏蔽，占比会显著偏低；若 assistant 内容漏标，占比会接近 1。
- **SFT 带来真实的指令遵循能力**：从"复读 prompt + `ticauser`/`UAGE` 噪声"退化状态，变为能成段回答、能给出可运行代码。`ckpt/sft` 的存在性通过只是形式，真正的证据是 compare 输出。
- **DPO 带来风格层面的偏好改进**：在"什么是机器学习"上更简洁有序；在代码题上与 SFT 持平。

### 4.2 失败项的根因

1. **训练规模过小是首要根因**。SFT 仅 59 条、1 epoch；DPO 仅 800 对、100 步。`adapter_config.json` 的 meta 明确记录了这两个数字。数据文件其实已下载（MOSS 1w 子集 76 MB、DPO 中英各 50 MB / 28 MB），说明不是数据缺失，而是本次运行只做了小规模验证。
2. **0.5B 基座 + 小数据无法补足知识**。"LoRA 是什么"属于事实性知识，SFT/DPO 都无法从偏好数据中习得；要解决必须让 SFT 数据覆盖相关概念，或改用更大基座。
3. **DPO 会放大流畅度而未必提升正确性**。题目④中 DPO 给出了比 SFT 更"确定"的错误答案（补上错误英文全称并重复），说明偏好对齐在没有足够知识支撑时可能只优化了"看起来像好回答"的表面特征。
4. **翻译能力缺失源于数据分布**。MOSS 中文对话与 DPO 数据中缺少稳定的中英平行/翻译监督信号，SFT 与 DPO 都只做了中文改写。

### 4.3 可改进之处

1. **放大训练规模**：SFT 用 `--max-samples 5000 --max-len 512 --epochs 2`（README/操作流程已给出），DPO 提到 2000 对以上，并观察 margin 正例占比曲线。
2. **做 rank 消融（S2）**：`--r 4/8/16/32` 固定 seed 重跑，形成质量对照表。
3. **做灾难性遗忘评估（S3）**：在 C-Eval/MMLU 子集上对比 base 与 SFT，验证小数据 SFT 是否损伤通用能力。
4. **补 reward margin 曲线（S4）**：脚本已打印 margin acc，可落盘成图，判断 DPO 是否真的拉开了 chosen/rejected。
5. **翻译能力单独处理**：如需中译英，应在 SFT 数据中显式加入翻译样本，而不是指望 DPO 修复。

---

## 5. 踩坑与经验

| 坑 | 现象 | 规避方式 |
|---|---|---|
| A/B 形状对调或 B 未置零 | 训练一开始就改变原输出 | A `(in,r)` kaiming、B `(out,r)` 全零 |
| scaling 忘记 `alpha/r` | 换 rank 后等效学习率漂移 | scaling 统一写为 `alpha/r` |
| 原权重未冻结 | `lora_param_count` 占比 > 5% | 先全冻结再放行 A/B |
| loss mask 写反/漏标 | -100 占比越界 | 用 `--inspect` 查看解析；确认 user/system/控制符全 -100 |
| 只训最后一个 assistant turn | 多轮对话前面轮次浪费 | 每个 assistant turn 都参与监督 |
| 把 `<|im_start|>assistant` 计入 loss | 模型学会复读控制符 | 角色头 -100，仅内容与收尾 `<|im_end|>` 保留 |
| MOSS/DPO 数据 schema 不确定 | 解析为 0 条、一直跳过 | 代码做容器键/角色键容错；先跑 `--inspect` 体检 |
| DPO ref 参与反向 | 显存翻倍、DPO 退化 | ref `requires_grad=False` + `no_grad` |
| 只看 DPO loss 不看 margin | loss 降但偏好没拉开 | 同时监控 margin 正例占比，必要时换 per-token 均值变体 |
| 误把 `sft_vs_base` 当质量指标 | 以为通过就说明 SFT 有效 | 该项只校验目录非空，质量必须看 `compare_results.json` |

**经验**：本任务最值得记录的一条是"自检绿灯 ≠ 效果好"。`sft_vs_base` 通过只代表 `ckpt/sft/` 目录里有文件；真正说明问题的是 compare 输出里 base 的复读退化与 DPO 把 LoRA 混淆成 LSTM。评测设计上，存在性检查适合做 pipeline 冒烟，但不能替代定性/定量质量评估。

---

## 6. 结论

任务三的三项自动自检全部通过：LoRA 可训参数占比 0.109%（M1）、loss mask 比例 0.511（M2）、`ckpt/sft` 非空（M3）；DPO 产物 `ckpt/dpo/` 与 base/SFT/DPO 对比文件均已生成（M4 的产物要求满足）。

但若以**输出质量**为口径，结论要更谨慎：录得的 SFT 仅用 59 条样本、DPO 仅用 800 对偏好数据、100 步，属小规模 pipeline 验证。定性对比显示：base 是"复读机 + 噪声"（原样吐回 prompt、夹带 `ticauser` 等 token）；SFT 后能正常成段回答，回文函数给出了完整可运行实现；DPO 后在"什么是机器学习"上更简洁有条理，但中译英仍只做中文改写，而"用一句话介绍 LoRA"三个阶段全错，DPO 甚至把 LoRA 说成 LSTM——说明小规模 DPO 会优化表达风格而无法注入知识，甚至可能让错误答案更流畅自信。要真正提升质量，必须扩大 SFT/DPO 数据规模并针对知识型问题补充监督。

---

## 7. 复现命令

```bash
source .venv/bin/activate && cd task-3-sft-dpo
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com

# 1. 基座 + 数据
python data/download.py                       # 拉 Qwen2.5-0.5B -> models/Qwen2.5-0.5B
huggingface-cli download OpenMOSS-Team/moss-003-sft-data \
  moss-003-sft-no-tools.jsonl.zip --repo-type dataset --local-dir ./data/moss-sft
unzip data/moss-sft/moss-003-sft-no-tools.jsonl.zip -d data/moss-sft/
huggingface-cli download hiyouga/DPO-En-Zh-20k --repo-type dataset --local-dir ./data/dpo

# 2. 数据体检（确认能解析出消息 / 偏好对）
python train_sft.py --data data/moss-sft/moss-003-sft-no-tools.jsonl --inspect
python train_dpo.py --data data/dpo/*.jsonl --inspect

# 3. SFT（小批试跑 → 正式）
python train_sft.py --data data/moss-sft/moss-003-sft-no-tools.jsonl \
    --max-samples 200 --max-len 256 --epochs 1
python train_sft.py --data data/moss-sft/moss-003-sft-no-tools.jsonl \
    --max-samples 5000 --max-len 512 --epochs 2 \
    --batch-size 2 --grad-accum 8 --lr 2e-4

# 4. DPO（从 ckpt/sft 起步）
python train_dpo.py --data data/dpo/xxx.jsonl --max-samples 2000 --batch-size 1 --grad-accum 8

# 5. 三模型对比 + 自检
python src/compare.py
python src/compare.py --prompt "什么是机器学习？" --max-new-tokens 80
python eval/run.py
```

---

**附：`eval/result.json` 原始内容**

```json
[
  { "test": "lora_param_count", "pass": true, "trainable": 540672,
    "total": 494573440, "ratio": 0.00109 },
  { "test": "loss_masking", "pass": true, "mask_ratio": 0.511,
    "note": "若不在 (0.2, 0.9) 范围，请检查 user/system 是否全部 -100" },
  { "test": "sft_vs_base", "pass": true,
    "note": "仅校验 ckpt/sft 非空；输出质量请手动跑 src/compare.py 对比 base 与 SFT 并附在提交里" }
]
```
