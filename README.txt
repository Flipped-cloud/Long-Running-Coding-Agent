coding-agent：编程智能体

Git 仓库：https://github.com/Flipped-cloud/Long-Running-Coding-Agent

本项目从零实现一个可在本地代码仓库中自主工作的 coding agent，未使用现成的 agent 框架。系统通过 OpenAI 兼容接口调用支持原生 tool calling 的模型，形成“模型决策—工具调用—环境观察—下一轮决策”的闭环。

运行环境：Python 3.11 及以上，推荐 Ubuntu/WSL2。

安装：
python -m pip install -e ".[dev]"

运行前设置模型名称、接口地址和 API 密钥，密钥只放在环境变量中，不写入仓库：
export MODEL_NAME="模型名称"
export OPENAI_BASE_URL="OpenAI 兼容接口地址"
export OPENAI_API_KEY="API 密钥"

运行 coding agent：
longrun-agent run --config configs/baseline.yaml --task "修复目标仓库中的代码并运行测试"

安全说明：文件工具只允许访问指定工作区，命令默认不启用管道、重定向等 shell 语法，也不会把 API 密钥传给子进程。普通模式不是操作系统级沙箱，应只用于可信代码；评测时可在 Ubuntu/WSL2 中使用 Bubblewrap 隔离隐藏数据。日志中的已知密钥会被替换。

运行完整案例：
PROJECT_ID=assessment-check-01 MINIMUM_SECONDS=0 bash scripts/run_long_horizon_real_api.sh

运行包含生成测试和自适应恢复的综合案例：
PROJECT_ID=comprehensive-agent-01 CASE_PROFILE=comprehensive bash scripts/run_long_horizon_real_api.sh

特色功能：工作区内文件读写与命令执行；结构化模型协议和错误反馈；步数、时间及连续错误终止条件；持久化任务图和断点续跑；上下文预算、裁剪与结构化交接；基于执行证据的记忆；冻结验证契约、隔离隐藏测试、完整性检查和独立结果校验。测试与静态检查由 GitHub Actions 自动执行。
