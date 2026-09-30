# 任务四：RAG 文档问答 —— PDF 切分、BGE+FAISS 召回与端到端生成实验报告

> 实验对象：`llm-beginner-master/task-4-rag/`
> 数据来源：自检指标与逐题明细取自 `task-4-rag/eval/result.json`；索引参数取自 `task-4-rag/data/index/meta.json`。未编造任何数值。
> 说明：仓库根 `日志.md` 记录的是**代码产出阶段**（当时未运行），本文以运行后落盘的产物为准。

---

## 1. 任务目标与 DoD

### 1.1 一句话目标

搭一套端到端中文 RAG（PDF chunking → BGE embedding + FAISS → reranker → Qwen 生成），在随任务提供的 30 条 NNDL gold QA 上做到 Recall@10 > 0.6，并把检索质量与生成忠实性拆开量化。

### 1.2 任务约束

- 知识库必须是《神经网络与深度学习（第二版）》**PDF**（`data/kb.pdf`），不允许直接索引 LaTeX 源；
- 手写 RAG 流水线，不使用 LlamaIndex / LangChain 的高层封装；
- 评测必须用固定 `data/gold_qa.jsonl`（30 条，每条含 `source_file` 与 `gold_anchors`），命中口径为"召回 chunk 文本去空白后包含任一 gold anchor"。

### 1.3 Definition of Done（必做 4 项）

| 编号 | DoD 内容 | 自检项 | 通过标准 |
|---|---|---|---|
| M1 | 实现 `chunk_text` | `chunking_sanity` | chunk 数 > 10；平均长度 ∈ (chunk_size×0.5, chunk_size×1.2) |
| M2 | 从 `data/kb.pdf` 建 embedding + FAISS 索引，实现 `Retriever.retrieve` | 由 M3 间接验证 | 索引来自 PDF |
| M3 | 在 gold QA 上评测召回 | `nndl_gold_recall_at_10` | Recall@10 > 0.6；同时报 Recall@1/3/5/10 与 MRR |
| M4 | 串起端到端 `answer` | `rag_end_to_end` | 返回非空 answer + 非空 sources |

加分项：S1（chunk_size 扫描 128/256/512/1024）、S2（有无 reranker）、S3（Query rewriting / HyDE）、S4（RAGAS 端到端打分）。

---

## 2. 方法与实现要点

### 2.1 项目结构

```
task-4-rag/
├── src/paths.py       统一路径与默认参数（KB_PDF / GOLD_QA / INDEX_DIR / chunk_size 等，可环境变量覆盖）
├── src/chunker.py     chunk_text（字符计、句末断句、overlap）+ PDF 抽文本
├── src/embed.py       BGE embedding（query 检索前缀 + L2 归一化）
├── src/indexer.py     chunk → embed → FAISS IndexFlatIP
├── src/retriever.py   Retriever.retrieve(query, k)
├── src/reranker.py    bge-reranker-base cross-encoder 精排
├── src/generator.py   OpenAI 兼容本地服务客户端
├── src/rag.py         answer(query) 端到端
├── build_index.py     一键建索引
├── recall_report.py   Recall@1/3/5/10 + MRR + 未命中 id
├── query_demo.py      示例问题问答
└── data/{kb.pdf,gold_qa.jsonl,index/}
```

### 2.2 关键设计决策

来自 `日志.md` 第 7.4 节：

1. **知识库只用 `data/kb.pdf`**：抽文本后把行间换行合成空格、清掉孤立页码行；gold anchor 判定在**去除所有空白后**做子串匹配，从而等价容忍页缝处的断行。
2. **句子意识 + overlap 切块**：在窗口内 `[chunk_size//2, chunk_size]` 找最后一个句末标点（`。！？!?；;`），找不到也在窗口尾切；由此保证单块长度恒 ≤ chunk_size×1.2，同时用 overlap 保住跨切缝的 anchor。
3. **BGE query 前缀 + L2 归一化 + FAISS 内积**：三者缺一召回即坏。统一在 embed 层完成，调用方不感知——query 侧加 `为这个句子生成表示以用于检索相关文章：`，文档侧不加；归一化后内积等价于余弦相似度。
4. **召回 k（20）≫ 精排 k（5）**：先宽召回再 rerank，检索与精排两段分开，可分别量化。
5. **生成与检索解耦报错**：没起 LLM 时 `answer` 明确抛错并给出起服务方式（而非静默返回空串）；`RAG_OFFLINE=1` 让纯检索 demo 不依赖生成模型。
6. **不静默失败**：索引缺失时 `Retriever` 抛 `FileNotFoundError` 并提示先跑 `build_index.py`。

### 2.3 实际索引参数（来自 `data/index/meta.json`）

| 项 | 值 |
|---|---|
| chunk_size | 512（字符） |
| overlap | 128 |
| embedding 维度 | 512 |
| chunk 总数 | 2137 |
| embedding 模型 | bge-small-zh-v1.5 |

索引文件齐全：`chunks.json`（1,884,618 B）、`embeddings.npy`（4,376,704 B）、`index.faiss`（4,376,621 B）、`meta.json`。知识库 `data/kb.pdf` 为 7,587,144 B。

---

## 3. 实验结果

以下为 `task-4-rag/eval/result.json` 的完整整理。

| 测试名 | 结果 | 关键指标 | 阈值 | 对应 DoD |
|---|---|---|---|---|
| `chunking_sanity` | **通过** | `chunks = 17`，`avg_len = 241.9`，期望区间 `[128.0, 307.2]` | 数 > 10 且均值 ∈ (0.5·size, 1.2·size) | M1 |
| `nndl_gold_recall_at_10` | **通过** | `n = 30`；Recall@1 = **0.333**、Recall@3 = **0.7**、Recall@5 = **0.867**、Recall@10 = **0.867**；MRR = **0.527** | Recall@10 > 0.6 | M3 |
| `rag_end_to_end` | **失败** | 连不上本地生成服务 `http://localhost:11434/v1`（WinError 10061 目标计算机积极拒绝） | answer 与 sources 均非空 | M4 |

> 注：`chunking_sanity` 使用 `chunk_size = 256, overlap = 32` 的合成文本（`"这是一段测试文本。"×400`）单独核验 chunker；索引侧的实际参数为 512/128（见 2.3）。

### 3.1 召回的逐题明细

30 道题中命中 26 题、未命中 **4** 题（26/30 = 0.867，与 `recall_at_10` 一致）：

| 未命中 id | 来源文件 |
|---|---|
| `nndl-inductive-bias` | `chap-绪论/chap-绪论.tex` |
| `nndl-learning-types` | `chap-机器学习概述/chap-机器学习概述.tex` |
| `nndl-causal-mask` | `chap-注意力机制与Transformer/chap-注意力机制与Transformer.tex` |
| `nndl-pca-autoencoder` | `chap-无监督学习/chap-无监督学习.tex` |

命中题的排名分布也值得注意：Recall@5 与 Recall@10 完全相同（均为 0.867），说明**没有任何一道题是在第 6–10 名才被首次命中**——`recall@10` 相对 `recall@5` 的增量为零。

### 3.2 DoD 完成情况

| 编号 | 结论 | 依据 |
|---|---|---|
| M1 | ✅ 达标 | 17 个 chunk、平均 241.9 字符，落在 128.0–307.2 区间内 |
| M2 | ✅ 达标 | 索引 2137 个 chunk 由 `data/kb.pdf` 建立，`bge-small-zh-v1.5` + FAISS 落盘完整 |
| M3 | ✅ 达标 | Recall@10 = 0.867 > 0.6，且 MRR = 0.527 提供了排序质量信息 |
| M4 | ❌ **未达标** | `rag_end_to_end` 因本地生成服务未启动而失败，错误为连接被拒 |

加分项 S1–S4 均未在结果中体现。

---

## 4. 结果分析

### 4.1 已达标项分析

**M1（chunking）**：合成样本上得到 17 个 chunk、均值 241.9，接近 `chunk_size = 256` 的上界内水平，说明"句子意识断点 + 保留 overlap"的策略有效——块既没有短到丢语义，也没有超过 1.2 倍上限。实际索引用的 512/128 参数下共切出 2137 个块。

**M3（召回）**：Recall@10 = 0.867 通过。更细的指标给出了有价值的诊断信息：

- **Recall@1 只有 0.333**：三分之二的题目首命中不在第 1 位，说明单靠 BGE 向量相似度，query 与正确 chunk 之间的语义排序不够锐利。这正是 reranker（cross-encoder 精排）应当发力的地方——但本次结果中第 5 名之后不再有新增命中，提示瓶颈可能同时来自召回集本身与排序。
- **MRR = 0.527**：平均倒数排名约 0.53，介于"总是排第 2"（0.5）与"总是排第 1"（1.0）之间，与 Recall@1 = 0.333、Recall@3 = 0.7 的分布自洽。
- **Recall@5 == Recall@10**：26 道命中题全部落在前 5 名。这意味着扩大 k 从 5 到 10 对本题集没有任何收益，检索的上限问题出在"排名前 5 之外就再也召回不到"，而非排序退化。若要提升，方向应是增大召回宽度之外的手段（chunk 策略、query 改写、reranker 换序）。

### 4.2 未命中项的根因分析

4 道未命中题有一个共同特征：其 gold anchor 位于**概念定义/分类枚举类段落**（绪论中的归纳偏置、机器学习概述中的学习类型分类、注意力机制章节中的 causal mask 说明、无监督学习中的 PCA 与自编码器），这些内容在 PDF 中往往以列表、小标题或跨页段落形式出现。可能原因：

1. **锚点被切缝截断或跨页**：尽管有 128 字符的 overlap，若 anchor 恰好落在 512 字符窗口的边界附近、且 PDF 抽取时该处存在页眉页脚噪声，仍可能被切断。
2. **PDF 版式损失**：列表项、公式与图注在抽文本后常变成碎片，导致 anchor 所在的语义单元不完整，BGE 无法把它与问句对齐。
3. **术语与问句的表面差异**：如"归纳偏置""学习类型"这类抽象概念，问句用词与正文用词差异大，向量召回容易偏向同章的其他段落。
4. **chunk 粒度**：512 字符的块可能把"定义 + 展开"混在一起，稀释了锚点句的权重。

### 4.3 `rag_end_to_end` 失败的原因与修复

失败原因是**生成服务未启动**，不是流水线逻辑缺陷：错误信息为 `[WinError 10061] 由于目标计算机积极拒绝，无法连接`，指向 `http://localhost:11434/v1`（Ollama 默认端点）。`answer` 按设计"连不上就抛带指引的 RuntimeError"，因此该项被判失败。要补齐 M4：

```bash
ollama pull qwen2.5:7b-instruct && ollama serve
python src/rag.py --query "什么是反向传播？请结合资料回答。"
python eval/run.py
```

或在只想验证检索链路时用 `RAG_OFFLINE=1` 走抽取式兜底（但该模式不满足 M4 对真实生成的要求）。

需要强调：**M4 未通过不应被其他三项绿灯掩盖**。`rag_end_to_end` 是唯一检验"检索→拼 prompt→调用模型→返回 answer+sources"整链路的项；它失败意味着端到端交付尚未成立。

### 4.4 可改进之处

1. **补齐端到端验证**：启动 Ollama 后重跑 `eval/run.py`，并人工抽查答案是否被返回 sources 支持、有无上下文外断言、不知道时是否拒答。
2. **做 chunk_size 扫描（S1）**：重建 128/256/512/1024 四档索引，用 `recall_report.py` 对比 Recall@10，验证本次 4 个未命中是否由切分粒度导致。
3. **做 reranker 消融（S2）**：对比有无 bge-reranker-base 时的 top-k 命中与 MRR。当前 Recall@1 = 0.333 明显偏低，reranker 理论上应显著改善首命中率。
4. **Query rewriting / HyDE（S3）**：针对概念类问题（本次 4 个未命中全部属此类）改写问句，使其更接近教材表述。
5. **针对未命中题定向分析**：用 `recall_report.py` 打印 4 题的实际 top-10 文本，确认是"切缝截断"还是"语义不匹配"，再决定调 overlap 还是换 chunk 策略。

---

## 5. 踩坑与经验

| 坑 | 现象 | 规避方式 |
|---|---|---|
| chunk_size 按 token 而非字符 | `chunking_sanity` 平均长度越界 | `chunk_size/overlap` 一律以字符计 |
| 无句子边界硬切 | anchor 被拦腰切断 | 在窗口 `[size//2, size]` 内找最后一个句末标点 |
| overlap 太小 | 跨缝 anchor 两侧都命中不了 | 实际索引使用 512/128 |
| BGE 漏加 query 检索前缀 | 召回显著下降 | query 侧加 `为这个句子生成表示以用于检索相关文章：`，doc 侧不加 |
| 忘记 L2 归一化 | FAISS 内积分数全乱 | 归一化后内积即余弦 |
| 索引缺失时静默返回空 | 召回全 0 却难定位 | `Retriever` 主动抛 `FileNotFoundError` 并提示建索引 |
| 没起 LLM 时静默给空答案 | 端到端看似通过实则无效 | `answer` 抛带中文指引的 RuntimeError；本次正是因此暴露失败 |
| 用流畅答案掩盖召回失败 | 报告结论失真 | 检索指标（Recall/MRR）与生成忠实性分开报告 |

**经验**：本任务最有价值的设计是"报错而非静默降级"。正因为 `answer` 在连不上 LLM 时直接抛错，`rag_end_to_end` 如实显示为失败，避免了"看起来三项全绿、其实生成环节从未运行"的假通过。另一个经验是 Recall@5 与 Recall@10 相等这一细节——它比单一 Recall@10 更能指出瓶颈所在（不是 k 不够大，而是候选集里根本没有正确答案）。

---

## 6. 结论

任务四的检索侧表现良好：chunking 自检通过（17 块、均值 241.9），在 30 条 NNDL gold QA 上 Recall@1/3/5/10 = 0.333 / 0.7 / 0.867 / 0.867、MRR = 0.527，Recall@10 = 0.867 明显超过 0.6 的通过线；索引确由 `data/kb.pdf` 建立（2137 个 chunk，chunk_size 512 / overlap 128，bge-small-zh-v1.5，512 维）。

同时必须如实记录两点不足：其一，**端到端 `rag_end_to_end` 失败**，原因是本地生成服务 `localhost:11434` 未启动，M4 未达标，需启动 Ollama 后重跑；其二，**4 道题未命中**（`nndl-inductive-bias`、`nndl-learning-types`、`nndl-causal-mask`、`nndl-pca-autoencoder`），均为教材中的概念定义/分类枚举类段落，且 Recall@5 与 Recall@10 相等说明这些题在前 5 名之外也召回不到，瓶颈在 chunk 粒度与语义对齐而非 k 值。补齐生成服务并完成 chunk_size / reranker 消融后，本任务即可完整交付。

---

## 7. 复现命令

```bash
source .venv/bin/activate && cd task-4-rag
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com

# 1. 数据（BGE 模型 + NNDL v2 PDF + 校验 gold QA）
python data/download.py
python data/download.py --skip-models      # 只拿 PDF 并校验 gold QA

# 2. 建索引（默认 chunk_size=512, overlap=128 -> data/index/）
python build_index.py
python build_index.py --chunk-size 256 --overlap 64 --force   # 消融重建

# 3. 自检（chunking / gold 召回 / 端到端）
python eval/run.py
python recall_report.py                    # Recall@1/3/5/10 + MRR + 未命中 id

# 4. 端到端生成（需先起本地 LLM）
ollama pull qwen2.5:7b-instruct && ollama serve
python src/rag.py --query "什么是反向传播？请结合资料回答。"
python query_demo.py

# 5. 无生成模型时的检索链路验证
RAG_OFFLINE=1 python query_demo.py
```

> ⚠️ `eval/run.py` 必须在仓库内运行（依赖仓库根 `_eval_harness.py`）。

---

**附：`eval/result.json` 关键字段（完整 details 见原文件）**

```json
[
  { "test": "chunking_sanity", "pass": true, "chunks": 17, "avg_len": 241.9,
    "expected_avg_range": [128.0, 307.2] },
  { "test": "nndl_gold_recall_at_10", "pass": true, "n": 30,
    "recall_at_1": 0.333, "recall_at_3": 0.7, "recall_at_5": 0.867,
    "recall_at_10": 0.867, "mrr": 0.527,
    "details": [ "…30 条逐题 hit/rank/matched_anchor/source_file…" ] },
  { "test": "rag_end_to_end", "pass": false,
    "error": "连不上本地生成服务 http://localhost:11434/v1：…请先启动模型服务，例如：ollama serve && ollama pull qwen2.5:7b-instruct" }
]
```
