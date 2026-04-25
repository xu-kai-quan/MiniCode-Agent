# MiniCode Agent

> 一个**最小化编码 agent**, 单文件实现, 没有任何框架封装, 适合用来理解 "AI agent 到底是怎么工作的". 从最小可跑的 30 行 ReAct 循环, 一路迭代到流式 + 双后端 + token 可见的实用体验, 每一步都摊开给你看.
>
> *A minimal coding agent in a single Python file. ReAct loop, atomic tools, no framework. Five iterations from "it runs" to "it's actually usable", documented step by step.*

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![model](https://img.shields.io/badge/model-Qwen2.5%2FMiniMax--M2.7-orange)
![framework](https://img.shields.io/badge/framework-none-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

📖 [English](README_EN.md) · 中文

---

## 为什么有这个项目

市面上 "AI agent" 教程几乎都把东西包进 **langchain / autogen / crewai** 里, 结果你花了大半时间在读框架文档, 而不是在搞懂 agent 循环本身. 这个 repo 反过来:

- 每个项目都是**一个 Python 文件**, 没有任何抽象
- 整个 ReAct 循环大约 **30 行代码**, 摊开就能看完
- 五个迭代版本展示**一个真实 agent 是怎么从 MVP 长出来的** — 一路加上测试、沙箱、乐观锁、流式、云端切换

看遍了 langchain / autogen 的教程, 还是不知道 agent 内部到底怎么转? 读源码, 不要读框架文档.

## 这个项目不是什么

- ❌ 不是框架 — 别 `pip install`, 直接读代码
- ❌ 不适合生产 — 没有重试 / 可观测 / 多租户
- ❌ 不是多 agent — 单 agent, 单会话 (子 agent 在 v6 路线图里)
- ❌ 不绑定特定后端 — 默认本地 Ollama 跑 7B, 也可一行切到云端 MiniMax-M2.7

---

## 五个版本

每一版只引入一两个新概念, 改动可控, 跟着版本号往后读就能看清"agent 是怎么从能跑长到能用的".

| 目录 | 这一版的核心新东西 | 适合谁读 |
|---|---|---|
| [01-bash-only/](01-bash-only/) | 最小 ReAct + 5 个工具 (bash / read / write / edit / todo), 一次性任务模式, 跑完就退出 | 第一次接触 agent, 想看清**最小骨架** |
| [02-sandboxed/](02-sandboxed/) | 加交互式 REPL + bash 沙箱探测 (排除 WSL System32, 避免 UTF-16 输出冲突) + 大文件分片写入 | 想看 agent 怎么从一次性脚本演化成**可用的日常工具** |
| [03-atomic-tools/](03-atomic-tools/) | 三层工具架构 (LS/Glob/Grep/Read 原子层) + 读后写乐观锁 (mtime+size cache, NOT_READ/CONFLICT) + 42 个 pytest + GitHub Actions CI | 想看怎么把 agent **做扎实**, 经得起测试 |
| [04-structured-tool-calls/](04-structured-tool-calls/) | 后端换 Ollama HTTP (默认 qwen2.5-coder:7b, 能力跳档) + OpenAI 结构化 tool_calls + `apply_patch` 跨文件 unified diff (两阶段锁 + 原子回滚) + 66 个 pytest | 想看 agent 怎么从"能跑"演进到**跨文件原子改动** |
| [05-session-and-streaming/](05-session-and-streaming/) | Session 状态管理 + 流式输出 + Ctrl-C 中断 + 双后端 (Ollama / MiniMax-M2.7 二选一) + token/成本可见 + 代码块伪 tool_call 系统层兜底 + 安全脚手架 (.env / pre-commit hook) | 想看 agent 怎么从"能跑"长出**能用的体感** |

每一版都只有一个 `todo.py` 文件 — **你看到的就是全部真相**, 没有任何框架包装.

---

## 学习路径 — 怎么挑该读哪一版

**完全新手, 第一次看 AI agent**: 直接 [01-bash-only](01-bash-only/) 开始. 整个 todo.py 大约 300 行, ReAct 循环就 30 行. 看懂这个再往后跳.

**已经知道 ReAct 是什么, 想看 agent 怎么"做扎实"**: 跳到 [03-atomic-tools](03-atomic-tools/). 工具分层 / 读后写乐观锁 / pytest 这些工程实践集中在这一版.

**关心生产级体验 (流式、云端切换、成本可见)**: 直接看 [05-session-and-streaming](05-session-and-streaming/). 这是当前活跃版本, README 写得最详细 (~700 行, 每个能力的"为什么做、怎么做、起没起效果、中间踩的坑"都讲了).

**对比"小模型 vs 大模型"在 agent 场景下的真实差异**: 看 [05 的 §3 双后端](05-session-and-streaming/README.md#3-双后端--ollama-和-minimax-之间切) — 同一个 query 给 7B 和 M2.7 跑, 失败模式和成功路径放在一起.

**关心架构演化**: 按版本号顺序读 README — 不读源码也能看清"工具单一返回 → 三层架构 → 流式 + Session"这条线.

**只想把它跑起来用**: 跳到下面的"快速开始", 选 Ollama (本地零成本) 或 MiniMax (云端任务连续性好).

## 快速开始

```bash
git clone https://github.com/xu-kai-quan/MiniCode-Agent.git
cd MiniCode-Agent
```

**01 / 02 用 torch + transformers** (本地加载 Qwen safetensors):

```bash
pip install torch transformers safetensors
cd 01-bash-only
python todo.py "create hello.py that prints hi"
```

需要本地放一份 [Qwen2.5](https://huggingface.co/Qwen) 模型权重, 默认路径在每个 `todo.py` 顶部, 改成你自己的.

**03 / 04 / 05 用 Ollama** (零 Python 依赖, 跑得起 7B):

```bash
# 装 Ollama 桌面版: https://ollama.com/
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

cd 04-structured-tool-calls       # 或 05-session-and-streaming
python todo.py
```

**05 还可以一行切到云端 MiniMax-M2.7** (任务连续性显著好):

```bash
cd 05-session-and-streaming
cp .env.example .env
# 编辑 .env: MINICODE_BACKEND=minimax + MINIMAX_API_KEY=<你的 key>
python todo.py
```

`.env` 在 `.gitignore` 里, 永远不会被 commit; 仓库还有 pre-commit hook 扫常见密钥前缀挡误操作 (见 [05 README §8](05-session-and-streaming/README.md#8-安全脚手架--env-和-pre-commit-hook)).

## 技术栈

- **模型 (按版本)**:
  - 01 / 02: Qwen 系列, 本地 torch + safetensors 加载
  - 03 / 04 / 05 (Ollama 后端): qwen2.5-coder:7b-instruct-q4_K_M (默认), 跑在本地 GPU
  - 05 (MiniMax 后端): MiniMax-M2.7 (~204K context, 云端)
- **运行环境**: Python 3.10+, 仅 stdlib (03 起不再依赖 torch — Ollama HTTP 协议)
- **工具调用格式**:
  - 01-03: Qwen XML `<tool_call>...</tool_call>`, 自己正则解析
  - 04-05: OpenAI 结构化 `tool_calls` 字段 (后端 Ollama 或 MiniMax)
- **核心范式**: ReAct (Thought → Action → Observation) 循环 + 工具表分发
- **沙箱**: 基于路径, 在工具层强制, 不依赖 prompt
- **测试**: pytest, 03 起 conftest.py 把 torch / transformers stub 掉. CI 矩阵跑全部 5 个版本

## 测试

```bash
cd 03-atomic-tools && pip install pytest && pytest tests/   # 42 个, ~0.6s
cd 04-structured-tool-calls && pytest tests/                # 66 个
cd 05-session-and-streaming && pytest tests/                # 66 个 (与 04 同, 加新功能未加新测试)
```

五个版本都有 CI, 每次 push 自动跑全部 — 见 [.github/workflows/test.yml](.github/workflows/test.yml). 03 起测试不需要 torch (conftest.py stub 掉).

## 设计理念

1. **循环就是 agent 的全部本质** — 模型输出 → 解析工具调用 → 执行 → 把结果喂回去, 直到模型不再调工具。仅此而已。
2. **工具是查表分发, 不是硬编码** — 加新工具 = 加一个函数 + 登记表项, 循环本身不动。
3. **沙箱在工具层做, 不在 prompt 里写** — 小模型不会乖乖听 "不要读 cwd 外的文件", 但 `safe_path()` 函数会拦住。
4. **prompt 是软约束, 代码是硬约束** — 关键不变量必须在代码里兜底, 因为小模型不一定听 prompt。
5. **日志要"可审查"而非"好看"** — 每一步: 模型在想什么 / 调了什么工具 / 拿到什么结果, 全部打印出来, 方便 debug 和学习。

## 相关项目

如果你想要的是**生产级**的本地编码 agent (而不是用来学习的), 去看 [aider](https://github.com/paul-gauthier/aider) / [continue.dev](https://continue.dev) / [Claude Code](https://claude.com/claude-code)。这个 repo 是为了让你**搞懂原理**, 不是日常工具。

## Keywords

`qwen` `qwen3` `qwen2.5-coder` `minimax` `minimax-m2` `ollama` `local-llm` `coding-agent` `react-agent` `ai-agent` `tool-use` `function-calling` `llm-agent` `no-framework` `educational` `single-file` `react-loop` `python` `from-scratch` `streaming` `apply-patch` `unified-diff` `optimistic-locking` `session-management`
