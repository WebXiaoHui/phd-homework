# 任务五：工具调用 Agent —— 手写 ReAct 循环与四类工具实验报告

> 实验对象：`llm-beginner-master/task-5-tool-agent/`
> 数据来源：全部指标与逐题明细取自 `task-5-tool-agent/eval/result.json`；任务集取自 `task-5-tool-agent/data/tasks.json`。未编造任何数值。
> 说明：仓库根 `日志.md` 记录的是**代码产出阶段**（当时未运行），本文以运行后落盘的 `eval/result.json` 为准。

---

## 1. 任务目标与 DoD

### 1.1 一句话目标

手写约 200 行 ReAct 循环（Thought / Action / Action Input / Observation），让本地 Qwen2.5-7B-Instruct 自主调度 calculator / python_sandbox / file_search / wiki 四类工具，在自建 10 题任务集上答案关键词命中率 > 60%（按关键词校验，不信任 agent 自报 `success`）。

### 1.2 任务约束

- 不许直接用框架的 agent 封装，循环自己写；
- 工具要齐 4 类，每类一个文件、各带 OpenAI function calling schema；
- 工具抛异常时必须捕获并塞回 Observation 让模型自我纠错，单次失败不得 crash 循环。

### 1.3 Definition of Done（必做 4 项）

| 编号 | DoD 内容 | 自检项 | 通过标准 |
|---|---|---|---|
| M1 | 实现 4 个工具（各带 `TOOL_SCHEMA` + `run(args)`） | `tools_individual` | 4 个工具单测全过；wiki 因网络异常按"跳过"处理 |
| M2 | 手写 ReAct 循环（路由、步数上限、Final Answer 终止） | 由 M4 的 trace 间接验证 | 循环可运行 |
| M3 | 工具异常捕获并塞回 Observation | `error_recovery`（本次未判定） | 注入错误后仍完成比例 > 40%（可选 S4） |
| M4 | 10 题任务集成功率 | `multi_tool_success_rate` | 关键词命中率 > 60% |

加分项：S1（Qwen-Agent 对照）、S2（1.5B/7B/14B 尺寸对比）、S3（prompt 模板对比）、S4（`inject_error` 错误恢复）。

---

## 2. 方法与实现要点

### 2.1 项目结构

```
task-5-tool-agent/
├── src/tools/calculator.py     AST 白名单求值（非 eval）
├── src/tools/python_sandbox.py 受限 exec（import 黑名单 + builtins 白名单 + 超时 + stdout 捕获）
├── src/tools/file_search.py    文件名/内容检索（路径越界保护 + 命中预览）
├── src/tools/wiki.py           MediaWiki API（含 CJK 走 zh，否则 en；端点可 env 覆盖）
├── src/tools/__init__.py       工具注册表 TOOLS + all_schemas()
├── src/agent.py                ReActAgent + parse_step + inject_error 钩子
├── agent_demo.py               单题/多题运行 + 完整 trace + 命中汇总
├── tools_check.py              4 工具独立快速自检
├── error_recovery_demo.py      错误恢复验证（M3 / S4）
└── data/{tasks.json,agent-fixtures/}
```

### 2.2 关键设计决策

来自 `日志.md` 第 8.4 节：

1. **工具 = schema + run 双导出**：`TOOL_SCHEMA` 只用于生成 prompt 与元信息，真正执行全走 `run(args)`；`TOOLS` 注册表统一路由，未知工具也走 Observation 纠错路径。
2. **calculator 用 AST 白名单而非 `eval`**：只放行数字常量、白名单数学函数与纯运算符节点，其余语法节点一律拒绝，从设计上消除"计算器即任意代码执行"的风险。整数结果附"N 位整数"提示、浮点附 6 位小数双写法，方便下游引用。
3. **错误消息即 Observation（M3 的实现）**：工具异常、Action 解析失败、未知工具三种失败统一折叠成一条 `工具 X 执行出错：…` 文本塞回模型，循环不 crash；只有步数上限兜底终止。
4. **Action 解析三层容错**：正则优先匹配行首标签 → 无冒号的 `Final Answer` 也接受（MULTILINE）→ JSON 解析失败时把原始串也送回 Observation 让模型重写，而不是静默丢步。
5. **评测链路独立**：自检不看 agent 自报的 `success`，只看 `final_answer` 的关键词命中（归一化去逗号/空白/小写，同义变体列表任一命中即可）；`eval/result.json` 同时记录 `used_tools` 与 `expected_tools`，便于定位工具路由问题。
6. **安全边界**：`python_sandbox` 的黑白名单被明确定位为**教学级**防护（README 明确警告可由 `__globals__` 等路径逃逸），`file_search` 路径 resolve 后必须落在允许根内。

### 2.3 评测任务集

`data/tasks.json` 共 10 题，覆盖 4 类工具：

| id | 任务要点 | 期望工具 |
|---|---|---|
| 1 | 计算 (123+456)×789 并说出结果位数 | calculator |
| 2 | Python 计算 100 以内质数之和 | python_sandbox |
| 3 | 统计 `data/agent-fixtures` 下 .md 文件数 | file_search |
| 4 | 查维基百科"图灵机"发明者 | wiki |
| 5 | 查 Hinton 出生年份并算到 2026 年的岁数 | wiki, calculator |
| 6 | 写回文函数并测试 level / world | python_sandbox |
| 7 | 查找含 TODO 的文件路径 | file_search |
| 8 | 计算 sqrt(2026) 小数点后 6 位 | calculator, python_sandbox |
| 9 | 查 Transformer 论文发表年份并算到 2026 年多少年 | wiki, calculator |
| 10 | 读取 README.md 第一段内容 | file_search |

其中 id 4/5/9 依赖维基百科网络。

---

## 3. 实验结果

### 3.1 自检指标（`eval/result.json`）

| 测试名 | 结果 | 关键指标 | 阈值 | 对应 DoD |
|---|---|---|---|---|
| `tools_individual` | **通过** | `calculator = true`、`python_sandbox = true`、`file_search = true`；`wiki = "skip(网络不可用？)"`；`network_skipped = ["wiki"]` | 非网络工具全过 | M1 |
| `multi_tool_success_rate` | **通过** | `rate = 0.8`（10 题中 8 题命中） | > 0.6 | M4 |
| `error_recovery` | **跳过（未判定）** | `pass = null`，`skip = "需要学生实现 inject_error 测试钩子；可选实验"` | > 0.4（可选） | M3（S4） |

### 3.2 四类工具单测明细

| 工具 | 调用参数 | 期望子串 | 结果 |
|---|---|---|---|
| calculator | `{"expression": "2 + 3 * 4"}` | `14` | ✅ true |
| python_sandbox | `{"code": "print(sum(range(10)))"}` | `45` | ✅ true |
| file_search | `{"pattern": "README.md", "dir": <ROOT>}` | `README.md` | ✅ true |
| wiki | `{"query": "Alan Turing"}` | 响应长度 > 50 | ⏭️ skip |

wiki 跳过的底层错误为连接超时：

> `访问维基百科失败（请检查网络）：ConnectTimeout: HTTPSConnectionPool(host='en.wikipedia.org', port=443) … 'Connection to en.wikipedia.org timed out. (connect timeout=15)'`

### 3.3 10 题逐题结果

| id | 结果 | 期望工具 | 实际使用工具 | 是否用对工具 | 最终答案预览 |
|---|---|---|---|---|---|
| 1 | ❌ 失败 | calculator | calculator | ✅ | `(123 + 456) * 789 = 456831，结果的位数是 6。` |
| 2 | ✅ 成功 | python_sandbox | python_sandbox ×12 | ✅ | （多轮修正代码后得到 1060） |
| 3 | ✅ 成功 | file_search | file_search | ✅ | `在 data/agent-fixtures 目录下共有 2 个 .md 文件。` |
| 4 | ✅ 成功 | wiki | wiki×2, python_sandbox, calculator, … | ✅ | （含 Turing / 图灵） |
| 5 | ✅ 成功 | wiki, calculator | wiki, python_sandbox, file_search, calculator | ✅ | `Geoffrey Hinton 出生年份是 1947 年，到 2026 年他 79 岁。` |
| 6 | ✅ 成功 | python_sandbox | python_sandbox ×12 | ✅ | （含 level / True / world / False） |
| 7 | ✅ 成功 | file_search | file_search | ✅ | `data/agent-fixtures/todo_note.md` |
| 8 | ✅ 成功 | calculator, python_sandbox | calculator | ❌（未用 python_sandbox） | `sqrt(2026) 的小数点后 6 位结果是 45.011110。` |
| 9 | ❌ 失败 | wiki, calculator | wiki | ❌ | `维基百科查询失败，请稍后再试。` |
| 10 | ✅ 成功 | file_search | file_search, python_sandbox, file_search | ✅ | `这是任务五的本地文件检索测试文件。` |

统计：命中 8 题、失败 2 题，成功率 **0.8**；10 题中仅 **id 8** 出现"答案正确但未使用全部期望工具"的情况；id 1 工具用对但答案措辞不符合关键词；id 9 工具用对（wiki）但工具执行失败。

### 3.4 DoD 完成情况

| 编号 | 结论 | 依据 |
|---|---|---|
| M1 | ✅ 达标 | 3 个非网络工具单测全过；wiki 按设计在离线时跳过，不拖累判定 |
| M2 | ✅ 达标的间接证据 | 10 题全部跑出 trace 与 `final_answer`，说明 ReAct 循环、工具路由、终止条件可用；多步题（5、8、10）也产生了多轮工具调用 |
| M3 | ⚠️ 未判定 | `error_recovery` 返回 skip；实现层面 `src/agent.py` 已带 `inject_error` 钩子，但自检脚本对这项硬编码为"可选实验，跳过" |
| M4 | ✅ 达标 | 命中率 0.8 > 0.6 |

---

## 4. 结果分析

### 4.1 达标情况

- **M1（工具可用）**：三个非网络工具的单元测试全部通过，说明 `TOOL_SCHEMA` + `run(args)` 的契约与固定参数键（`{"expression"}` / `{"code"}` / `{"pattern","dir"}` / `{"query"}`）完全对齐。
- **M4（成功率）**：0.8 超过 0.6 的通过线。分类型看，纯计算/代码题（id 2、6、8）全过，文件检索题（id 3、7、10）全过，wiki 类题（id 4、5、9）2 过 1 败。
- **多步与自我纠错在运行中确有发生**：id 2 与 id 6 都出现了 12 次 `python_sandbox` 调用，trace 预览里能看到"看起来代码中存在语法错误。我需要修正代码并重新运行。""看起来代码语法有误，可能是由于缩进问题。我将修正缩进并重新运行。"这类 Thought。这说明 M3 的"错误消息当 Observation"机制在真实运行中生效，agent 确实基于失败反馈迭代了代码，而不是一次崩溃退出——这是比 `error_recovery` 那项 skip 更有说服力的 M3 证据。

### 4.2 失败项根因分析

**id 1 —— 纯措辞导致的判定失败，工具与计算都正确。**

- 最终答案：`(123 + 456) * 789 = 456831，结果的位数是 6。`
- 期望关键词：`"456831"` 且 `["6位","6 位","六位","6 digits"]` 之一。
- 自检的归一化会去掉逗号与空白，因此 `"6 位"` 会被视为 `"6位"`；但答案里写的是"位数是 6"，**不含"6位"这个连续子串**，故判失败。
- 根因不在推理能力（456831 完全正确、位数 6 也完全正确），而在**答案表述与评测关键词的错位**。有趣的是，calculator 工具按设计会在整数结果后附"N 位整数"提示（`日志.md` 第 8.2 节），但 agent 在 Final Answer 里改写成自己的口语表述，丢掉了评测依赖的字面形式。这暴露出"关键词判定"对同义表述的鲁棒性有限，也说明 prompt 中应明确要求"数值结果请连同单位/位数的标准写法一起给出"。

**id 9 —— 依赖网络，wiki 工具超时。**

- 最终答案：`维基百科查询失败，请稍后再试。`
- 期望关键词：`"2017"` 与 `"9"`（Transformer 论文年份及到 2026 年的年数）。wiki 不可达，agent 只调用了一次 wiki 就停止，没有降级到"用已有知识作答 + 计算"，也没有尝试替代路径，因此判失败。
- 根因是**网络不可用**（`tools_individual` 已记录 `en.wikipedia.org` 连接超时 15 s），叠加 agent 缺少"工具失败后的降级策略"。

**id 8 —— 答案对但工具路由不完整。**

- 期望 `calculator` + `python_sandbox`，实际只用 `calculator`，`used_expected_tools = false`。
- 但最终答案 `45.011110` 命中了关键词，所以该项仍判成功。这说明**当前口径只看答案、不看路由**：工具路由偏差不会扣分。对"任务成功率"而言这是合理的（结果导向），但对"多工具协同能力"的评估而言是盲区。

**关于 id 4 / id 5 的一个需要说明的疑点。**

id 4、5 都依赖维基百科，且单测中英文维基已确认超时，但这两题仍然成功（id 5 给出了 1947 与 79 岁）。可能的原因是：wiki 工具对含 CJK 的查询走 `zh.wikipedia.org` 端点，而本次失败的是英文端点，故中文查询可能仍可达；另一种可能是模型用了参数化知识作答。区分二者需要逐题查看完整 trace（`agent_demo.py --ids 4`）。在不能确认的前提下，不应把 id 4/5 的成功归因于真实检索。本报告如实标注这一不确定性。

### 4.3 加分项与改进方向

1. **补做 S4 / 让 `error_recovery` 可判定**：`src/agent.py` 已实现 `inject_error` 钩子，`error_recovery_demo.py` 也提供了离线与 `--live` 两种验证方式，但 `eval/run.py` 里的 `test_error_recovery` 目前无论实现与否都直接返回 skip。建议补一个真实的注入式评测项（例如向 calculator 注入 1 次故障后统计仍完成任务的占比），把 M3 从"间接证据"升级为"实测指标"。
2. **提高 wiki 题的鲁棒性**：给 agent 加"工具失败降级"策略（重试其他端点、或用已有知识作答并标注不确定性），或在评测前配置代理 / `WIKI_API_ZH`、`WIKI_API_EN` 镜像。
3. **改善答案格式约束**：针对 id 1 这类"位数/单位"题，在 system prompt 中要求复用工具返回的标准写法（如 `6 位整数`），减少关键词错位。
4. **做 S1/S2/S3**：Qwen-Agent 对照、1.5B/7B/14B 尺寸对比、few-shot 模板消融，这些在 `eval/result.json` 中均无记录。

---

## 5. 踩坑与经验

| 坑 | 现象 | 规避方式 |
|---|---|---|
| calculator 直接 `eval` | 任意代码执行风险 | 改为 AST 白名单求值 |
| python_sandbox 被当成真正隔离 | 可经 `__globals__` 等逃逸 | 明确定位为教学级防护，只跑可信/自产代码 |
| file_search 无路径校验 | `../../` 读到工作区外 | 路径 resolve 后校验落在允许根内 |
| Action Input 键名不符 | KeyError 直接打断 | 固定键名 + 异常折叠成 Observation |
| Action 解析靠脆弱正则 | 7B 偶尔不守格式 | 三层容错 + 解析失败要求重写 + 步数上限防死循环 |
| 工具异常冒泡 | 单次失败 crash 循环 | `except Exception` 包成 Observation（本次 id 2/6 多轮修正即其效果） |
| 用 agent 自报 `success` 计分 | 成绩虚高 | 自检只看 `final_answer` 关键词 |
| wiki 依赖网络 | 离线时单测与题目失败 | 单测按 skip 处理；跑 M4 前配代理或镜像 |
| 只看答案不看工具路由 | 如 id 8 答对但没用第二个工具 | 同时记录 `used_tools` vs `expected_tools` |

**经验**：本次最具启发的一点是"关键词判定"的双向误差。一方面它可能误杀正确答案（id 1：位数 6 写对了，只是没写成"6位"）；另一方面它可能掩盖能力缺陷（id 8：没用 python_sandbox 也照样得分）。因此 `eval/result.json` 中 `used_tools` / `expected_tools` 字段与 `final_answer_preview` 必须与命中率一起解读，单看 0.8 不能完整反映 agent 的多工具调度水平。此外，多步题里 12 次工具调用与显式的"修正后重跑"Thought，是 M3 机制真实生效的最好证据。

---

## 6. 结论

任务五的必做项达成情况为"M1 通过、M2 有间接证据、M4 通过、M3 未判定"：

- **M1 达标**：calculator / python_sandbox / file_search 三个工具单测全部通过；wiki 因 `en.wikipedia.org` 连接超时被记为 skip（`network_skipped = ["wiki"]`），符合自检设计，不算代码错误。
- **M4 达标**：10 题命中 8 题，`multi_tool_success_rate = 0.8`，超过 0.6 的通过线。
- **两题失败且原因不同**：id 1 是"答案正确但措辞未被关键词命中"（写"位数是 6"而非"6 位"），id 9 是"wiki 网络超时后缺少降级策略"。
- **M3（错误恢复）为 skip / 未判定**：`eval/result.json` 中 `error_recovery` 的 `pass` 为 `null`，自检脚本将其定义为可选实验；但 trace 显示 id 2、id 6 在语法/缩进报错后仍通过多轮 Observation 自我修正并最终答对，可作为 M3 机制生效的间接证据。
- **一处口径盲区**：id 8 答案正确但未使用期望的 python_sandbox，仍判成功，说明命中率不反映工具路由的完整性。

补齐 wiki 网络（或加降级策略）、把 `error_recovery` 变成可实测项、并完成 Qwen-Agent / 模型尺寸消融后，本任务可形成完整交付。

---

## 7. 复现命令

```bash
source .venv/bin/activate && cd task-5-tool-agent
pip install -r requirements.txt

# 1. 本地模型（默认客户端指向 Ollama 的 OpenAI 兼容口）
ollama pull qwen2.5:7b-instruct && ollama serve
export AGENT_LLM_BASE_URL=http://localhost:11434/v1
export AGENT_LLM_API_KEY=ollama
export AGENT_LLM_MODEL=qwen2.5:7b-instruct
export AGENT_MAX_STEPS=12

# 2. 生成 10 题任务集与本地检索夹具
python data/download.py

# 3. 工具独立自检（M1，不需要模型服务；wiki 离线可跳过）
python tools_check.py

# 4. 端到端评测（M2 + M4）
python eval/run.py

# 5. 单题 / 全量 trace 调试
python agent_demo.py --ids 5
python agent_demo.py --all
python agent_demo.py --task "在 data/agent-fixtures 下找出所有 .md 文件，统计总数。"

# 6. 错误恢复（M3 / S4）
python error_recovery_demo.py
python error_recovery_demo.py --live
```

> wiki 需联网；大陆网络常不通，可 `export HTTPS_PROXY=http://127.0.0.1:7890`，或按 `wiki.py` 用 `WIKI_API_ZH` / `WIKI_API_EN` 指向可达镜像。

---

**附：`eval/result.json` 关键字段（完整 details 见原文件）**

```json
[
  { "test": "tools_individual", "pass": true,
    "results": { "calculator": true, "python_sandbox": true,
                 "file_search": true, "wiki": "skip(网络不可用？): …ConnectTimeout…" },
    "network_skipped": ["wiki"] },
  { "test": "multi_tool_success_rate", "pass": true, "rate": 0.8, "n": 10,
    "details": [ "…10 题逐题 success / final_answer_preview / expected_tools / used_tools / used_expected_tools…" ] },
  { "test": "error_recovery", "pass": null,
    "skip": "需要学生实现 inject_error 测试钩子；可选实验" }
]
```
