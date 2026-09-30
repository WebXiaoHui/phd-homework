# 任务六：Mini Coding Agent —— MCP + Skills + Subagents 三层栈实验报告

> 实验对象：`llm-beginner-master/task-6-coding-agent/`
> 数据来源：全部指标取自 `task-6-coding-agent/eval/result.json`；toy-repo 内容取自 `data/toy-repo/`。未编造任何数值。
> 说明：仓库根 `日志.md` 记录的是**代码产出阶段**（当时未运行），本文以运行后落盘的 `eval/result.json` 为准。

---

## 1. 任务目标与 DoD

### 1.1 一句话目标

用本地 Qwen2.5-Coder-7B-Instruct 复刻一个极简版 Claude Code：手写 MCP server（暴露 ≥ 5 个工具）、约 50 行的 Skill 加载器、1–2 个 Subagent 和一条 agentic loop，能在 `data/toy-repo` 上自主读懂 issue、改代码、跑测试直到 `python -m pytest` 全绿。

### 1.2 任务范围

- 输入：本地 Git 仓库 + 一个 issue 描述；
- 输出：能自动定位代码、修改、跑测试、生成 patch 的完整 trace；
- 进阶评测：SWE-bench Lite 抽样。

### 1.3 Definition of Done（必做 4 项）

| 编号 | DoD 内容 | 自检项 | 通过标准 |
|---|---|---|---|
| M1 | 手写 MCP server，顶层导出 `list_tools()` | `mcp_server_lists_tools` | 枚举到 ≥ 5 个工具 |
| M2 | Skill 加载器 + 2–3 个带 YAML front-matter 的 `SKILL.md` | `skill_loader_metadata` | ≥ 2 个 skill 且每个都有 name + description |
| M3 | 实现 agent loop，修好 `calculator.add` 并让 pytest 全绿 | `toy_repo_patch` | 外部 `python -m pytest -q` 返回 0 |
| M4 | Trace 含 `steps` / `patch` / `tests_passed` | 由 `toy_repo_patch` 的 `trace_has_patch` 间接验证 | Trace 字段完整 |

加分项：S1（Q4_K_M 量化 vs FP16）、S2（单 agent vs 加 Subagent 的词元消耗与成功率）、S3（纯 prompt vs 加 Skill）、S4（SWE-bench Lite 抽样 ≥ 1 题 `tests_passed`）。

---

## 2. 方法与实现要点

### 2.1 项目结构

```
task-6-coding-agent/
├── src/toolkit.py            原子工具层：9 个工具 + ToolServer(repo)
├── src/mcp_server.py         MCP stdio server：list_tools()/run_tool()/FastMCP 注册
├── src/skill_loader.py       SkillLoader（front-matter 索引 + 命中才 load 正文）
├── src/skills/{test-runner,code-review,pr-description-writer}/SKILL.md
├── src/subagents/__init__.py SubAgent 基类 + explore / test_repro
├── src/agent.py              CodingAgent + Trace（agentic loop）
├── run_agent.py              CLI：单 repo+issue、可 --with-subagents/--json-out
├── smoke_offline.py          离线冒烟：M1/M2/工具层（无需模型）
├── data/toy-repo/            calculator.py / calculator.py.orig / test_calculator.py / ISSUE.md
└── eval/result.json
```

### 2.2 能力三层栈

| 层 | 概念 | 本实验的实现 |
|---|---|---|
| 底层 | **Tools / MCP** | `toolkit.py` 9 个原子工具：`read_file` / `write_file` / `edit_file` / `list_files` / `grep_search` / `run_command` / `run_tests` / `git_diff` / `git_apply`，经 `mcp_server.py` 以 MCP stdio 暴露 |
| 中层 | **Skills** | `SkillLoader` 扫描 `*/SKILL.md` 的 YAML front-matter 形成轻量索引，issue 命中 description 后才把正文注入 system context（渐进式披露）；共 3 个 Skill |
| 顶层 | **Subagents** | `explore` / `test_repro` 两个只读子 agent，各自独立 message 列表、独立步数上限、独立工具子集，主 agent 只接收摘要 |

### 2.3 关键设计决策

来自 `日志.md` 第 9.4 节：

1. **一套工具实现两路暴露**：`toolkit` 的函数同时供 MCP server 与进程内 `ToolServer` 调用，agent 不依赖子进程 stdio 握手，离线也能跑；而 MCP 协议能力对自检与外部 client 仍然成立。这对应 README "两条路都得通"的要求。
2. **安全三条线**：
   - 路径 resolve 后必须落在 repo 内，防 `../` 越界；
   - 子进程一律用 list 形式参数 + 指定 `cwd`，防 shell 注入；
   - `write_file` / `edit_file` 直接拒绝 `test_*`、`conftest` 与 `*.orig`，防改测试绕过判分。
3. **Skills 渐进式披露**：front-matter 只被扫成轻量索引，issue 命中 description 才 load 正文；toy issue 含 "pytest" 触发词，恰好命中 `test-runner`，把"run_tests 全绿才算完成、以 DONE 收尾"写进 agent 的即时上下文。
4. **停机双保险**：显式 DONE / Final Answer 尽早停；即便 7B 忘记输出 DONE，循环在步数耗尽后也会用一次独立 pytest 终验得出真实 `tests_passed`，使 M3 判定不被格式漂移拖垮。
5. **context 压缩**：预估 token 超阈值时丢弃中段、保留 system + 最近一轮并补一行摘要（约 28k 字符触发），避免 7B 长任务撑满上下文。

### 2.4 toy-repo 的 bug 与约束

`data/toy-repo/ISSUE.md` 内容为：

> 修复 `calculator.add` 的实现，使 `python -m pytest` 全部通过。不要改测试文件。

按 README，bug 是 `add(a, b)` 被写成 `return a - b`，正确实现应为 `a + b`。自检每次先从 `calculator.py.orig` 快照恢复 buggy 版本再让 agent 修，避免上一轮已修好导致本轮空跑通过——这一点很重要，它保证了 `toy_repo_patch` 的通过是真实的。

---

## 3. 实验结果

以下为 `task-6-coding-agent/eval/result.json` 的完整整理：

| 测试名 | 结果 | 关键指标 | 阈值 | 对应 DoD |
|---|---|---|---|---|
| `mcp_server_lists_tools` | **通过** | 枚举到 **9** 个工具 | ≥ 5 | M1 |
| `skill_loader_metadata` | **通过** | `count = 3`，`missing_meta = []` | ≥ 2 且无缺元数据 | M2 |
| `toy_repo_patch` | **通过** | `tests_passed = true`；`trace_has_patch = true`；pytest 输出 `... [100%]\n3 passed in 0.01s\n` | pytest returncode == 0 | M3 / M4 |
| `swebench_lite_sample` | **跳过** | `pass = null`，`skip = "data/swebench-lite-sample.parquet 不存在；需要时跑 data/download.py --with-swebench"` | ≥ 1 题通过（可选 S4） | S4 |

MCP 枚举到的 9 个工具（按 `result.json` 顺序）：

```
read_file, write_file, edit_file, list_files, grep_search,
run_command, run_tests, git_diff, git_apply
```

### 3.1 DoD 完成情况

| 编号 | 结论 | 依据 |
|---|---|---|
| M1 | ✅ 达标 | `list_tools()` 返回 9 个工具，为 5 个下限的 1.8 倍 |
| M2 | ✅ 达标 | 扫描到 3 个 Skill，`missing_meta` 为空，说明每个 SKILL.md 的 name + description 都齐全 |
| M3 | ✅ 达标 | 从 `.orig` 复位 buggy 版本后，agent 修通，外部独立 pytest 3 passed |
| M4 | ✅ 达标 | `trace_has_patch = true`，Trace 含 patch；`tests_passed` 为独立 pytest 终验结果 |
| S4 | ⏭️ 未执行 | SWE-bench Lite 抽样数据文件不存在，按设计跳过 |

### 3.2 产物核查

| 产物 | 状态 |
|---|---|
| `data/toy-repo/calculator.py` / `calculator.py.orig` / `test_calculator.py` / `ISSUE.md` | 均存在 |
| `data/toy-repo/.pytest_cache/` | 存在（说明 pytest 确实在仓库内被执行过） |
| `data/toy-repo/__pycache__/` | 含 `calculator.cpython-312.pyc`、`test_calculator.cpython-312-pytest-9.1.1.pyc` |
| `data/swebench-lite-sample.parquet` | 不存在（故 S4 跳过） |

---

## 4. 结果分析

### 4.1 达标情况

- **M1（MCP / Tools）**：9 个工具全部被 `list_tools()` 枚举，覆盖文件读写、内容检索、命令执行、测试运行与 git 操作，满足 README "MCP server 暴露 ≥ 5 个工具"的要求。自检通过 `import list_tools` 完成，不依赖真实 stdio 握手，因此即使 MCP SDK 版本导致 `python src/mcp_server.py` 起不来，也不影响该项判定——这是"独立可跑"与"协议可用"两条路解耦的设计收益。
- **M2（Skills）**：3 个 Skill（`test-runner`、`code-review`、`pr-description-writer`）全部带完整 front-matter 元数据。`test-runner` 的 description 与 toy issue（含 pytest）匹配，会在 agent 运行时被命中并把正文注入上下文，这正是 README 要求的"description 匹配 + 按需加载正文"的渐进式披露。
- **M3 / M4（agent loop）**：这是本任务最核心也最容易"假通过"的一项，而本次结果可信度较高，原因有两点：
  1. 自检**每次先用 `calculator.py.orig` 覆盖回 buggy 版本**，排除了"上一轮残留修好的文件导致空跑通过"的可能；
  2. `tests_passed` 来自**独立的外部 pytest 终验**（`subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=toy_repo)`），而非 agent 自报。

  pytest 输出为：

  > `...                                                                      [100%]`
  > `3 passed in 0.01s`

  三条测试全部通过，说明 `calculator.add` 被真正改回 `a + b`，且 agent 没有（也无法）修改测试文件——工具层对 `test_*` / `conftest` / `*.orig` 的写入拒绝约束在此可视为生效。
- **M4（Trace 完整性）**：`trace_has_patch = true` 表明 `Trace` 字典中存在非空 `patch` 字段；配合 `tests_passed` 字段，README 要求的 `steps` / `patch` / `tests_passed` 三项已具备可验证证据。

### 4.2 未执行项分析

**S4（SWE-bench Lite）跳过**，原因是 `data/swebench-lite-sample.parquet` 不存在，需要额外执行 `python data/download.py --with-swebench`，并按提示把对应仓库 clone 到 `data/repos/`。自检脚本对此做了三层防护，值得肯定：

1. 先检查数据文件是否存在，不存在直接返回 skip——避免学生没打算跑 SWE-bench 时被 `src.agent` 的 ImportError 抢先报失败；
2. 即使元数据已下载，若 `data/repos/` 下没有本地仓库，也会返回 skip 并给出 details，而不是误判失败；
3. 只有真正尝试了 ≥ 1 题才有 `pass = passed >= 1` 的判定。

同时 README 与操作流程都明确：SWE-bench 真实判题需官方 harness 应用 golden test_patch 再跑 FAIL_TO_PASS，本 agent 只保证"改对并让自己跑到的测试通过"。因此 S4 的难度定位应理解为 bonus，不影响 M1–M4。

### 4.3 可改进之处

1. **补跑 S4**：`python data/download.py --with-swebench`，clone 对应仓库到 `data/repos/`，再跑 `eval/run.py`；README 说明跑通 1 题即合格。
2. **做消融实验**：S1（`qwen2.5-coder:7b-instruct` vs GGUF q4_k_m，换 `CODING_AGENT_LLM_MODEL` 重跑）、S2（`--with-subagents` 开关对比词元消耗与成功率）、S3（`--no-skills` 对比 Skill 增益）。当前 `eval/result.json` 未记录任何消融数据。
3. **补充 trace 落盘**：用 `python run_agent.py --with-subagents --json-out trace.json` 保存一条完整 trace，作为 README 提交清单要求的"toy-repo patch / 一条完整 trace"证据。
4. **验证 MCP stdio 真实握手**：`python src/mcp_server.py` 能否在 8GB 环境与当前 MCP SDK 版本下成功启动，这一条自检并不覆盖，建议手动确认一次。

---

## 5. 踩坑与经验

| 坑 | 现象 | 规避方式 |
|---|---|---|
| `list_tools` 未在模块顶层导出 | 自检 ImportError | `mcp_server.py` 顶层定义 `list_tools()` |
| SKILL.md 缺 name / description | `skill_loader_metadata` 挂 | front-matter 必写 name + description |
| description 过泛（"处理代码"） | 匹配不到任务 | 写清"何时加载" |
| 一上来读全部 SKILL.md 正文 | 失去渐进式披露意义 | 只索引 front-matter，命中才 load 正文 |
| Trace 不是 dict 或缺 `patch` / `tests_passed` | 自检 `.get()` 取不到即失败 | `Trace` 用 dict，字段齐全 |
| 停机写成"步数到了硬停" | 早停或空转 | 显式 DONE + 独立 pytest 终验双保险 |
| 假设上一轮已修好 | 空跑也能通过，判定失真 | 自检每轮从 `calculator.py.orig` 复位 |
| 长任务上下文撑满 | 7B 中途开始胡言 | 内置简易 context 压缩（约 28k 字符触发） |
| 改测试文件绕过 | 假通过 | 工具层拒写 `test_*` / `conftest` / `*.orig` |
| 用 `shell=True` 拼命令 | shell 注入 | subprocess 用 list 形式 + `cwd` |
| Subagent 共用同一份 messages | 隔离失去意义 | 独立 message 列表 + 独立步数 + 工具子集 |
| 依赖 git 状态复位 | 新机器未配 git 身份时初始 commit 静默失败 | 改用 `calculator.py.orig` 快照复位，不依赖 git |

**经验**：本任务最有价值的工程模式是"**不信任自报、外部终验**"。agent 可以自己说 DONE、自己说修好了，但 `toy_repo_patch` 只认外部独立 pytest 的 returncode；同时每轮从 `.orig` 复位，杜绝了残留状态造成的假通过。这两条一起，把一个容易被"看起来跑通了"蒙混的评测变成了可信的判定。另一条经验是"一套工具两路暴露"——把 agent 的实际执行路径与 MCP 协议暴露解耦，既保证了离线可跑，又不牺牲协议完整性。

---

## 6. 结论

任务六的必做项 **M1–M4 全部通过**：MCP server 顶层 `list_tools()` 枚举到 9 个工具（≥ 5）；SkillLoader 扫描到 3 个带完整 name + description 的 Skill；toy-repo 上从 `.orig` 复位 buggy 版本后，agent 自主修好 `calculator.add`，外部独立 pytest 结果为 `3 passed in 0.01s`；Trace 含非空 `patch`，`tests_passed` 由独立终验给出。

加分项 S4（SWE-bench Lite 抽样）**跳过**，原因是 `data/swebench-lite-sample.parquet` 不存在（需 `python data/download.py --with-swebench` 并 clone 对应仓库），其余三项加分实验（量化对照、Subagent 对照、Skill 对照）也未在结果中记录。

作为全系列子系统最多、且超出教材覆盖范围的任务，本任务在 Tools / Skills / Subagents 三层栈的搭建与 agentic loop 的可靠停机上都给出了可验证的实现，安全约束（路径越界、shell 注入、禁改测试）也已内建。补齐 S4 与消融实验、落盘一条完整 trace 后，即可形成完整交付。

---

## 7. 复现命令

```bash
source .venv/bin/activate && cd task-6-coding-agent
pip install -r requirements.txt

# 1. 本地模型（任选其一；推荐 GGUF q4_k_m 或 AWQ 省显存）
ollama pull qwen2.5-coder:7b-instruct && ollama serve
export CODING_AGENT_LLM_BASE_URL=http://localhost:11434/v1
export CODING_AGENT_LLM_MODEL=qwen2.5-coder:7b-instruct
export CODING_AGENT_LLM_API_KEY=ollama
export CODING_AGENT_REPO=$(pwd)/data/toy-repo

# 2. 生成本地 toy-repo
python data/download.py
python data/download.py --with-swebench      # 可选：额外下载 SWE-bench Lite 抽样元数据

# 3. 离线冒烟（M1 + M2 + 工具层，无需模型）
python smoke_offline.py

# 4. 跑通 toy-repo（M3 + M4）
python run_agent.py
python run_agent.py --verbose
python run_agent.py --with-subagents --json-out trace.json

# 5. 官方自检（M1/M2/M3 + 默认跳过 S4）
python eval/run.py
```

> SWE-bench Lite 需按 `data/download.py` 提示把对应仓库 clone 到 `data/repos/` 后再跑 `eval/run.py`。

---

**附：`eval/result.json` 原始内容**

```json
[
  { "test": "mcp_server_lists_tools", "pass": true,
    "tools": ["read_file", "write_file", "edit_file", "list_files", "grep_search",
              "run_command", "run_tests", "git_diff", "git_apply"] },
  { "test": "skill_loader_metadata", "pass": true, "count": 3, "missing_meta": [] },
  { "test": "toy_repo_patch", "pass": true, "tests_passed": true,
    "pytest_output": "...                                                                      [100%]\n3 passed in 0.01s\n",
    "trace_has_patch": true },
  { "test": "swebench_lite_sample", "pass": null,
    "skip": "data/swebench-lite-sample.parquet 不存在；需要时跑 data/download.py --with-swebench" }
]
```
