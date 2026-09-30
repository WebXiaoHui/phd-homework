# 入学任务 · 实验报告总目录

本目录是三个作业部分的**实验报告**，所有指标都可在仓库里找到原始出处（`results.jsonl`、
`eval/result.json`、训练曲线图、终端截图、checkpoint 目录），没有估算或补造的数字。

```
实验报告/
├── README.md                    ← 本文件（总目录 + 完成度与缺口清单）
├── 图神经网络/                    4 篇（graph_beginner-main）
├── LLM入门/                      6 篇（llm-beginner-master）
├── Transformers实战/              7 篇（transformers_tasks-main）
└── _原始数据/                     结果汇总与读图脚本（可复现报告的每个数字）
```

---

## 一、总览：三个部分、17 个任务的完成情况

| 部分 | 任务数 | 数据来源 | 一句话结论 |
|---|---|---|---|
| 图神经网络 | 4 | `任务*/results/results.jsonl`（92 条实验台账） | 四类 GNN 全部跑通；**子图采样省显存不省时间**；任务三/任务四的消融实验有大片缺口 |
| LLM 入门 | 6 | `task-*/eval/result.json` | 6 个任务的代码全部落地并自检：**19 项自检中 15 项通过、2 项失败（mini-GPT 困惑度、RAG 端到端）、2 项跳过** |
| Transformers 实战 | 10 个跑过 + 4 个未跑 | `logs/*.png` 曲线图 + 终端截图 + `checkpoints/` | 10 个子任务有结果留存，**ChatGLM 微调/LLM zero-shot/UIE/LLMsTrainer 只有代码没有结果** |

---

## 二、图神经网络（`graph_beginner-main/`）

作业原文与框架说明见 [`../graph_beginner-main/README.md`](../graph_beginner-main/README.md)，
实验前的运行手册见 [`../graph_beginner-main/日志.md`](../graph_beginner-main/日志.md)（注意它写于正式实验之前）。

| 报告 | 完成度 | 关键指标（best test） |
|---|---|---|
| [任务一 节点分类](图神经网络/任务一_节点分类_实验报告.md) | 35 条 / 34 个唯一配置（主实验 + 学习率 + 层数**全部完成**） | 全台账最优 **Cora + GraphSAGE + 采样 = 0.8180**；主实验最优 Cora GraphSAGE 0.7840 / Citeseer GraphSAGE 0.6830 / Flickr GAT 0.5237 |
| [任务二 链路预测](图神经网络/任务二_链路预测_实验报告.md) | 24 条（**只有主实验，学习率/层数消融一条未跑**） | 全台账最优 **Cora + GCN + 全图 = 0.9398**；GIN 在 Flickr 上 AUC≈0.0014~0.0027，是明确的负面结果 |
| [任务三 图分类](图神经网络/任务三_图分类_实验报告.md) | 22 条 / 21 个唯一配置（**池化对比、全图 vs 分批、层数/学习率消融均大面积缺失**） | 分类最优 **MUTAG + GIN + avg = 0.8333**；ZINC 回归（MAE 越小越好）最优 **GIN = 0.5372** |
| [任务四 知识图谱](图神经网络/任务四_知识图谱_实验报告.md) | 11 条 / 9 条有效（**学习率消融无有效数据**） | WN18RR **RotatE(dim=500) MRR 0.3128 / H@10 0.4655**；FB15k-237 **TransE(dim=500) MRR 0.2100 / H@10 0.3681** |

**这个部分最值得写进报告的实测发现**

1. **采样训练省的是显存，不是时间**：Flickr 上 GCN 采样 2.03 s/轮 vs 全图 0.015 s/轮（**慢 139 倍**），
   因为瓶颈是 PyG 在 CPU 上搭子图；比较两者必须同时看 `steps_per_epoch`。
2. **池化后接 BatchNorm 会让 MaxPooling 完全失效**（任务三）：验证准确率卡在多数类占比 0.4054，
   换成 LayerNorm 后回到 0.72~0.77。
3. **知识图谱的权重衰减必须为 0**：同配置下 `wd=5e-4` 的验证 MRR 只有 0.0086，`wd=0` 是 0.0936（约 11 倍）。
4. **ZINC 是回归任务**：指标是 MAE（`HIGHER_IS_BETTER["regression"]=False`），
   汇总时不能按"越大越好"排序，否则会得出错误的"最优模型"。

---

## 三、LLM 入门（`llm-beginner-master/`）

作业要求、DoD 与加分项见 [`../llm-beginner-master/README.md`](../llm-beginner-master/README.md)；
各任务的代码产出过程见 [`../llm-beginner-master/日志.md`](../llm-beginner-master/日志.md)
（该日志记录的是"只写代码、未运行"阶段，**实际运行结果以各任务 `eval/result.json` 为准**）。

| 报告 | 自检结果 | 关键指标 |
|---|---|---|
| [任务一 Transformer](LLM入门/任务一_Transformer_实验报告.md) | M1~M4 ✅ | 手写注意力 max_abs_diff **9.5e-7**；causal mask 泄漏 **0.0**；分类 dev acc **0.8608**（基线 0.85）；⚠️ 未发现注意力热图，M5 无法确认 |
| [任务二 mini-GPT](LLM入门/任务二_miniGPT_实验报告.md) | M1~M3 ✅ / **M4 ❌** | BPE roundtrip 无损；KV cache 与全量前向差 **3.8e-6**；dev **困惑度 65.48 > 阈值 50**（poetry，3,156 token） |
| [任务三 SFT + DPO](LLM入门/任务三_SFT与DPO_实验报告.md) | M1~M3 ✅ / M4 部分 | 可训练参数占比 **0.109%**；label mask 比例 **0.511**；base 是"复读机"，SFT 能正常作答，DPO 更简洁但**知识类问题仍会幻觉** |
| [任务四 RAG](LLM入门/任务四_RAG_实验报告.md) | M1~M3 ✅ / **M4 ❌** | 17 个 chunk / 均长 241.9；**Recall@10 0.867、MRR 0.527**；⚠️ 端到端问答未通过（Ollama 未启动） |
| [任务五 工具 Agent](LLM入门/任务五_工具Agent_实验报告.md) | M1 ✅ / M4 ✅ | 4 个工具单测通过（wiki 因网络超时跳过）；10 题**成功率 0.8 > 0.6**；失败 2 题（关键词口径 + wiki 超时） |
| [任务六 Coding Agent](LLM入门/任务六_CodingAgent_实验报告.md) | M1~M4 ✅ | MCP 暴露 **9 个工具**；扫到 **3 个 Skill**；toy-repo 修复后 **pytest 3 passed**；SWE-bench Lite 因数据缺失跳过 |

**这个部分最值得写进报告的实测发现**

1. **小规模 DPO 只改风格、不注入知识**：800 对偏好数据 / 100 步训练后，
   "用一句话介绍 LoRA"依然答错（把 LoRA 说成 LSTM），只有表达变得更流畅。
2. **关键词口径的自检会双向失真**：任务五里既误杀过正确答案（"位数是 6"不含关键词"6位"），
   也放过过没用对工具的答案，因此必须结合 `used_tools/expected_tools` 一起看。
3. **"不静默失败"的设计**：RAG 在没有生成服务时明确报错并给出补救指令，
   而不是返回空答案——这正好让未启动 Ollama 的问题暴露出来。

---

## 四、Transformers 实战（`transformers_tasks-main/`）

该目录**没有 `日志.md`**，结果以「训练曲线图 PNG + 终端截图 + checkpoint 目录」为准；
曲线指标由 [`_原始数据/extract_all.py`](_原始数据/extract_all.py) 做刻度标定后像素反解得到
（与终端截图交叉验证一致，见 [`_原始数据/transformers_结果提取.md`](_原始数据/transformers_结果提取.md)）。
其中 SimCSE 的 **F1 0.70541 / spearman 0.56527 / recall 0.99546**、p-tuning 的 **F1 0.64000**
都能与终端截图的原始打印对上；PET 无终端截图，只有曲线一个来源。

| 报告 | 完成度 | 关键指标 |
|---|---|---|
| [01 文本分类 BERT](Transformers实战/01_文本分类_BERT_实验报告.md) | 跑过，但只在 step 200 评测 1 次 | 8 分类 dev：acc 0.32 / precision 0.34 / recall 0.32 / **F1 0.26**（欠训练 + 单点评测） |
| [02 文本匹配（有监督）](Transformers实战/02_文本匹配_有监督_实验报告.md) | 3 个模型都跑通 | **PointWise（单塔）F1 0.90** > Sentence-BERT 双塔 0.818 > DSSM 双塔 0.61 |
| [03 文本匹配（无监督 SimCSE）](Transformers实战/03_文本匹配_无监督SimCSE_实验报告.md) | 跑通（47.7 万对语料） | **F1 0.7054、spearman 0.5653**（终端截图原始值）；precision 0.546 / recall 0.995 |
| [04 Prompt 学习 PET / p-tuning](Transformers实战/04_Prompt学习_PET与p-tuning_实验报告.md) | 两者都跑通（小样本 61 条） | PET **F1 峰值 0.771 / 末端 0.750**；p-tuning **F1 0.640**（截图原始值）；⚠️ train 只有 8 类而 dev 有 10 类，`电器` 类 F1=0 |
| [05 RLHF 奖励模型](Transformers实战/05_RLHF_奖励模型_实验报告.md) | 只完成第一阶段 | 奖励模型 eval/acc ≈ **0.666**；⚠️ **PPO 阶段未运行** |
| [06 文本生成：问答与 Filling](Transformers实战/06_文本生成_问答与Filling_实验报告.md) | 问答跑通 / filling 无指标 | DuReaderQG：BLEU-1~4 = **0.097 / 0.063 / 0.035 / 0.023**；filling 只有截图，**无可用指标** |
| [07 未运行子任务说明](Transformers实战/07_未运行子任务说明_LLM微调与应用_实验报告.md) | 4 个子任务未运行 | ChatGLM-6B 微调、LLM zero-shot、UIE、LLMsTrainer/llms_mbti：**只有代码，没有结果** |

---

## 五、尚未完成的实验（如实清单，答辩/补跑时按此执行）

| 部分 | 缺口 | 影响 | 补跑入口 |
|---|---|---|---|
| 图神经网络 · 任务二 | `--stage lr`（8 个）与 `--stage layers`（6 个）**一条未跑** | 报告要求 3.1 的"学习率/层数影响"在该任务无实测数据 | `cd graph_beginner-main/任务二_链路预测/code && python run_all.py --stage all`（`--skip_done` 默认开启，只补跑缺的） |
| 图神经网络 · 任务三 | 130 个实验只完成 22 个：池化对比 2/54、**全图 vs 分批 0/24**、层数 0/16、学习率 0/16 | 作业第 3 条（池化方法对比）与报告要求 3.2 严重不足 | `cd graph_beginner-main/任务三_图分类/code && python run_all.py --stage all`（预计 3~5 小时，建议挂过夜） |
| 图神经网络 · 任务四 | `dim` 9→3、学习率 12→**0 条有效** | 报告要求 3.1 的"参数影响"缺学习率一维 | `cd graph_beginner-main/任务四_知识图谱/code && python run_all.py --stage all` |
| LLM 入门 · 任务一 | 未找到注意力热图 | DoD M5 无法确认 | `cd llm-beginner-master/task-1-transformer && python visualize_attention.py` |
| LLM 入门 · 任务二 | dev 困惑度 65.48 > 50 | DoD M4 未达标 | 加长语料/训练轮数后 `python eval/run.py` |
| LLM 入门 · 任务四 | 未启动生成服务 | DoD M4（端到端问答）未达标 | `ollama serve && ollama pull qwen2.5:7b-instruct` 后重跑 `eval/run.py` |
| LLM 入门 · 任务六 | SWE-bench Lite 数据缺失 | 加分项 S4 未做 | `python data/download.py --with-swebench` |
| Transformers · 文本分类 | 只在 step 200 评测 1 次 | 结果不能代表收敛水平 | 调大评测覆盖、跑满 20 epoch 后重跑 `train.sh` |
| Transformers · 05/06/07 | PPO 阶段、filling 指标、ChatGLM 微调等未完成 | 相关章节只能写"未运行" | 见各报告第 7 节命令 |

---

## 六、原始数据与复现

`_原始数据/` 目录：

| 文件 | 用途 |
|---|---|
| `图神经网络_结果汇总.md` | 由四个任务的 `results.jsonl` 统计出的完整结果表 + 自动统计 |
| `transformers_结果提取.md` | Transformers 部分的配置表、曲线反解指标、终端截图逐字记录、交叉验证，以及 F 节记录的"一次已修正的读图事故" |
| `solve_values.py` | 读图核心：定位子图（含曲线贴顶时的行带校正）→ 检测白色网格线 → 用"漂亮步长 × 候选值"二维搜索反解**单点子图**的数值 |
| `extract_all.py` | 按人工读出的刻度标签，输出本目录 transformers 报告所用的**全部指标**（单点值 / 曲线 min-max-末端） |
| `check_edges.py` | 质量检查：逐条曲线报告它离坐标轴上/下边缘的距离，用来判断"曲线是否真的被裁剪" |
| `analyze_charts.py` | 结构探针：列出每张图有几个子图、子图框位置、橙色像素分布 |
| `build_strips.py` | 把各子图的 y 轴刻度标签区裁剪拼图（输出 `strips_*.png`），便于人工读刻度 |

复现方式（Windows Python，需 matplotlib + numpy + Pillow）：

```bash
python _原始数据/solve_values.py "<曲线图路径>"   # 看子图结构与单点反解
python _原始数据/extract_all.py                  # 输出全部指标
python _原始数据/check_edges.py                  # 检查曲线是否被坐标轴裁剪
```

> ⚠️ 读图方法有一个已修正的坑，见 `transformers_结果提取.md` F 节：
> 若用"每行背景像素占比"判定坐标轴范围，遇到**长距离贴顶的曲线**会把坐标轴截短，
> 从而把正常曲线误判成"被裁剪"。修正后所有曲线距轴边界都是 16~19px（= 默认 5% 边距）。
