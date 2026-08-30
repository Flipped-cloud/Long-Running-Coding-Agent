LongRun Agent：长时编程智能体

Git 仓库：https://github.com/Flipped-cloud/Long-Running-Coding-Agent

本项目从零实现一个可在本地代码仓库中自主工作的 Coding Agent，未使用 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等 Agent 框架。系统通过 OpenAI 兼容接口调用支持原生 tool calling 的模型，形成“模型决策—工具调用—环境观察—下一轮决策”的闭环。

运行环境：Python 3.11 及以上，推荐 Ubuntu/WSL2。安装命令：
python -m pip install -e ".[dev]"

运行前通过环境变量设置 OPENAI_API_KEY、OPENAI_BASE_URL 和 MODEL_NAME，不要把密钥写入仓库。核心运行命令：
longrun-agent run --config configs/baseline.yaml --task "修复目标仓库中的代码并运行测试"

普通运行模式的 Bash 以工作区为 cwd，默认不启用 shell 语法，拒绝显式越界路径，并且子进程只继承少量运行环境变量，不继承 API 密钥。它不是操作系统级文件沙箱：被执行程序仍具有当前用户的主机文件权限，因此只应用于可信本地代码；需要隔离隐藏数据时，应在 Ubuntu/WSL2 使用 Bubblewrap 评测模式。事件、保存的提示和 Bash 完整输出在落盘前会替换已知密钥值；不需要时可将 telemetry.save_prompts 或 telemetry.save_full_tool_outputs 设为 false。

预计约 30 分钟的完整检测案例（每次使用新的 PROJECT_ID；55 分钟项目预算和 60 分钟 watchdog 仅是安全上限）：
PROJECT_ID=assessment-check-01 MINIMUM_SECONDS=0 bash scripts/run_long_horizon_real_api.sh

覆盖范围更完整的新案例还会要求每个任务生成并注册 F2P 测试，并启用自适应恢复搜索；预计同样约 30 分钟，55 分钟项目预算和 70 分钟 watchdog 只是安全上限：
PROJECT_ID=comprehensive-agent-01 CASE_PROFILE=comprehensive bash scripts/run_long_horizon_real_api.sh

特色功能包括：工作区受限的文件读写与命令执行；结构化模型协议和错误观察；步数、时间与连续错误终止条件；持久化任务图和断点续跑；上下文预算、裁剪与结构化交接；基于证据的记忆；冻结验证契约、隐藏测试、完整性检查和独立 Oracle。测试与静态检查由 GitHub Actions 自动执行。
