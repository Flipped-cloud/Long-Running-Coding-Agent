# longrun-agent

`longrun-agent` 是一个面向长时编程任务的 Coding Agent Runtime。项目不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen 或 CrewAI，而是直接实现最小原生闭环：

```text
模型决策 -> 原生工具调用 -> 本地环境执行 -> 结构化观察 -> 下一轮模型决策
```

系统在此闭环上加入持久化任务图、断点续跑、上下文生命周期、证据化知识、契约验证、隐藏测试、完整性检查和独立 Oracle，适合在本地仓库中执行可验证的多阶段任务。

## 核心能力

- OpenAI-compatible 模型接口与原生 tool calling。
- 确定性 Fake Provider，用于基础测试和短演示。
- `read_file`、`write_file`、`bash` 三个本地工具。
- Project / Task / Session 分层，以及带依赖关系的持久化任务 DAG。
- 静态计划、自适应分解和有界恢复候选选择。
- 上下文预算、确定性裁剪和同一 Session 内的结构化交接。
- 基于工具轨迹证据的 episode、memory 与 skill 生命周期。
- 冻结 Verification Contract、公共/隐藏检查、生成测试校验和失败后 reopen。
- ProjectState、Session、审计事件、指标和 JSONL telemetry 落盘。
- 普通模式的工作区边界、密钥环境隔离与遥测脱敏。
- Ubuntu/WSL2 下可选 Bubblewrap 隔离。

## 架构

```text
ProjectOrchestrator
  -> 持久化任务 DAG 与依赖调度
  -> AgentLoop
       -> ModelProvider
       -> ToolRouter(read_file / write_file / bash)
       -> ContextBuffer 与 structured handoff
       -> KnowledgeStore
  -> CompletionCandidate
  -> frozen VerificationContract
  -> isolated VerificationReport
  -> VERIFIED 或 REOPENED
```

`Project` 表示完整目标，`Task` 表示可独立执行的子目标，`Session` 表示某个任务的一次 AgentLoop 尝试。正常状态路径为：

```text
PENDING -> READY -> IN_PROGRESS -> CANDIDATE_COMPLETE
        -> VERIFICATION_PENDING -> VERIFIED
```

验证失败时任务可以进入 `REOPENED`；阻塞、分解和失败也有独立状态。普通 `FinalAnswer` 不会被当作项目任务完成，项目模式必须通过结构化控制信号申请完成、报告阻塞或请求分解。

## 安装

要求 Python 3.11 及以上，推荐 Ubuntu 或 WSL2：

```bash
python -m pip install -e ".[dev]"
```

## 模型配置

程序不会自动加载 `.env`。在 Ubuntu/WSL2 中显式导出变量：

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="your-model"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

`configs/baseline.yaml` 用于普通真实模型运行，`configs/fake.yaml` 使用 Fake Provider，`configs/comprehensive_real_api.yaml` 由综合案例脚本配置其工作区、计划与验证资产。密钥只从配置指定的环境变量读取，不应写入仓库。

## 基础闭环演示

```bash
git restore examples/toy_repo/calculator.py
longrun-agent run \
  --config configs/fake.yaml \
  --fake-provider \
  --task "Fix the implementation bug in calculator.py so that all tests pass."
pytest -q examples/toy_repo/tests
```

PowerShell 用户也可运行 `./examples/toy_repo/reset_toy_repo.ps1`。Fake Provider 路径不读取 API 密钥。

真实模型的普通运行示例：

```bash
longrun-agent run \
  --config configs/baseline.yaml \
  --task "Fix the implementation bug in calculator.py so that all tests pass."
```

## 约 30 分钟的综合案例

综合案例位于 `examples/long_horizon_repo`，使用五任务依赖计划，覆盖：

- DAG 依赖调度与生命周期转换；
- 多 Session 持久化、恢复和结构化 handoff；
- Agent 生成并注册 F2P 测试；
- evidence episode、memory/skill 数据与使用记录；
- 冻结公共/隐藏契约、隔离验证和失败 reopen；
- 公共测试、隐藏测试、回归、完整性与变更范围检查；
- 独立 Oracle、审计指标、重复操作和 token 指标。

每次使用新的项目 ID：

```bash
PROJECT_ID="comprehensive-$(date -u +%Y%m%dT%H%M%SZ)" \
CASE_PROFILE=comprehensive \
bash scripts/run_long_horizon_real_api.sh
```

该案例按约 30 分钟的工作量设计。配置中的 55 分钟项目预算和脚本的 70 分钟 watchdog 是故障安全上限，不是人为延时或 30 分钟硬限制。成功结束时应同时看到：

```text
status: verified
hidden_tests_passed: true
integrity_passed: true
oracle_verified: true
violations: []
```

中断后可按脚本末尾打印的命令恢复，或执行：

```bash
PROJECT_ID="<原项目ID>" \
RESUME_PROJECT=1 \
CASE_PROFILE=comprehensive \
MINIMUM_SECONDS=0 \
bash scripts/run_long_horizon_real_api.sh
```

详细说明见 `evals/comprehensive/README.md`；脚本会输出 workspace、state、telemetry、console log、oracle 和 summary 的路径。

## CLI

查看工具：

```bash
longrun-agent tools --config configs/fake.yaml
```

项目与上下文命令要求显式传入配置文件：

```bash
longrun-agent project status --config <config.yaml> --project-id <project-id>
longrun-agent project tree --config <config.yaml> --project-id <project-id>
longrun-agent project metrics --config <config.yaml> --project-id <project-id>
longrun-agent context inspect --config <config.yaml> --project-id <project-id> --session-id <session-id>
longrun-agent verify report --config <config.yaml> --project-id <project-id> --report-id <report-id>
```

项目状态默认保存在配置的 `state.root/<project_id>/` 下，主要包括：

```text
project_state.json
project_events.jsonl
sessions.jsonl
project_metrics.json
plan_revisions/
context/
knowledge/
verification/
```

## 上下文与知识

上下文模式包括 `full_history`、`recent_window`、`deterministic_prune` 和 `structured_reset`。重置只重建当前 Session 的有效输入，不会创建新任务、增加任务尝试次数或伪造进度。

知识模式包括 `disabled`、`raw_episode`、`reflection`、`verified_memory` 和 `memory_skill`。记忆必须引用实际工具轨迹证据；skill 只在已验证成功后晋升，且不会自动执行。

## 验证边界

`CompletionCandidate` 只是 Agent 的完成申请；只有冻结契约产生的权威 `VerificationReport` 才能使项目进入 `VERIFIED`。隐藏资产位于 Agent 工作区之外，只注入验证副本，不进入提示、handoff、公开 telemetry 或知识记录。

完整性检查会验证受保护文件、可信测试、契约、允许变更范围、符号链接和隐藏资产泄漏。生成测试是运行时证据，不能替代 Oracle 契约。综合案例的最终结论由独立结果校验和 Oracle 共同确认。

## Bash 与遥测安全

- 文件路径经解析后必须位于配置的 workspace 内，拒绝父目录穿越、工作区外绝对路径和符号链接逃逸。
- 普通 `bash` 以 workspace 为 cwd，默认不启用 shell 语法，并只继承少量运行环境变量，因此 API key 不会传给子进程。
- 普通模式不是操作系统级沙箱；被执行程序仍拥有当前用户权限，只应用于可信本地代码。
- Bash 输出、事件、提示和汇总在落盘前会脱敏已知密钥值。
- 不需要完整数据时可关闭 `telemetry.save_prompts` 或 `telemetry.save_full_tool_outputs`。
- 隔离隐藏数据时，在 Ubuntu/WSL2 使用 Bubblewrap 模式。

Bubblewrap 环境检查：

```bash
sudo apt-get install bubblewrap
unshare -Ur true
longrun-agent sandbox doctor --config configs/comprehensive_real_api.yaml
```

## 测试

```bash
pytest -q
pytest -q --cov=longrun_agent --cov-report=term-missing --cov-fail-under=85
python -m compileall -q src tests scripts
ruff check .
ruff format --check .
git diff --check
```

综合案例资产的本地回归测试：

```bash
pytest -q tests/test_long_horizon_experiment.py
```

Windows 可运行大部分确定性测试；Bubblewrap 原生隔离检查由 Linux CI 执行。

## 仓库卫生

不要提交 `.runs/`、`.coverage`、`coverage.xml`、`htmlcov/`、日志、缓存、`.env`、API key、隐藏契约私有数据或 Oracle 私有输出。完整案例需要的源码、公开/隐藏测试、计划、契约、脚本和校验器必须保留在仓库中。
