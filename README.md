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
- ❌ 不是多 agent — 当前单 agent 单会话 (子 agent 在 05 §10 列为 v6 候选, 没承诺)

**关于后端**: 默认本地 Ollama 跑 7B, 也可一行切到云端 MiniMax-M2.7. 这是**能力**, 不是"不是什么".

---

## 五个版本 — 一个活跃, 四个封存快照

每一版只引入一两个新概念, 改动可控, 跟着版本号往后读就能看清"agent 是怎么从能跑长到能用的". **04 之前都已封存** (作为历史快照保留, 不再改), 当前活跃版本是 **05-session-and-streaming**.

| 目录 | 状态 | 这一版的核心新东西 | 适合谁读 |
|---|---|---|---|
| [01-bash-only/](01-bash-only/) | 封存 | 最小 ReAct + 5 个工具 (bash / read / write / edit / todo), 一次性任务模式, 跑完就退出 | 第一次接触 agent, 想看清**最小骨架** |
| [02-sandboxed/](02-sandboxed/) | 封存 | 加交互式 REPL + bash 沙箱探测 (排除 WSL System32, 避免 UTF-16 输出冲突) + 大文件分片写入 | 想看 agent 怎么从一次性脚本演化成**可用的日常工具** |
| [03-atomic-tools/](03-atomic-tools/) | 封存 | 三层工具架构 (LS/Glob/Grep/Read 原子层) + 读后写乐观锁 (mtime+size cache, NOT_READ/CONFLICT) + 42 个 pytest + GitHub Actions CI | 想看怎么把 agent **做扎实**, 经得起测试 |
| [04-structured-tool-calls/](04-structured-tool-calls/) | 封存 | 后端换 Ollama HTTP (默认 qwen2.5-coder:7b, 能力跳档) + OpenAI 结构化 tool_calls + `apply_patch` 跨文件 unified diff (两阶段锁 + 原子回滚) + 66 个 pytest | 想看 agent 怎么从"能跑"演进到**跨文件原子改动** |
| [05-session-and-streaming/](05-session-and-streaming/) | ✅ **活跃** | Session 状态管理 + 流式输出 + Ctrl-C 中断 + 双后端 (Ollama / MiniMax-M2.7 二选一) + token/成本可见 + 代码块伪 tool_call 系统层兜底 + 安全脚手架 (.env / pre-commit hook) | 想看 agent 怎么从"能跑"长出**能用的体感** — 也是想跑起来实际用的默认推荐 |

每一版都只有一个 `todo.py` 文件 — **你看到的就是全部真相**, 没有任何框架包装.

---

## 学习路径 — 怎么挑该读哪一版

**默认推荐 (想跑起来用 / 想看最完整的 README)**: 直接去 [05-session-and-streaming](05-session-and-streaming/). 这是唯一活跃版本, README 736 行, 每个能力的"为什么做、怎么做、起没起效果、中间踩的坑"都讲了.

**完全新手, 第一次看 AI agent**: 从 [01-bash-only](01-bash-only/) 开始读源码. 整个 todo.py 369 行, ReAct 循环就 30 行——这是看清"agent 到底是什么"最快的路. 但**实际跑用 04 / 05**, 01 是教学快照.

**已经知道 ReAct, 想看 agent 怎么"做扎实"**: 跳到 [03-atomic-tools](03-atomic-tools/). 工具分层 / 读后写乐观锁 / pytest / CI 这些工程实践集中在这一版引入.

**对比"小模型 vs 大模型"在 agent 场景下的真实差异**: 看 [05 §3 双后端](05-session-and-streaming/README.md#3-双后端--ollama-和-minimax-之间切) — 同一个 query 给 7B 和 M2.7 跑, 失败模式和成功路径放在一起对比.

**关心架构演化**: 按版本号顺序读各版 README — 不读源码也能看清"工具单一返回 → 三层架构 → 流式 + Session + 双后端"这条线. 每个 README 都有"上一版痛点 / 这一版怎么改 / 起没起效果"段, 互相交叉引用.

## 快速开始 — 默认走 05

```bash
git clone https://github.com/xu-kai-quan/MiniCode-Agent.git
cd MiniCode-Agent

# 装 Ollama 桌面版: https://ollama.com/
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

cd 05-session-and-streaming
python todo.py
```

**没有 Python 依赖** (运行时仅 stdlib). 装 Ollama + 拉模型大约 5 分钟. 启动后给你一个 `>` 提示符, 边生成边显示输出, Ctrl-C 中断不丢 partial.

**想用云端大模型** (MiniMax-M2.7, 任务连续性显著好):

```bash
cd 05-session-and-streaming
cp .env.example .env
# 编辑 .env: MINICODE_BACKEND=minimax + MINIMAX_API_KEY=<你的 key>
python todo.py
```

`.env` 在 `.gitignore` 里, 永远不会被 commit; 仓库还有 pre-commit hook 扫常见密钥前缀挡误操作 (见 [05 README §8](05-session-and-streaming/README.md#8-安全脚手架--env-和-pre-commit-hook)).

**想跑老版本** (历史教学用, 不推荐日常使用):

- **01 / 02**: 还是 torch + transformers + 本地 Qwen safetensors. `pip install torch transformers safetensors` 后 cd 进去 `python todo.py`. 需要本地放一份 Qwen2.5 模型权重, 默认路径在 todo.py 顶部
- **03**: 同 01/02 (torch 路径). 但工具层做扎实了, 是工程基础的转折点
- **04**: 跟 05 一样用 Ollama, 但没有流式 / Session / 云端 / token 显示

## 技术栈

- **模型 (按版本)**:
  - 01 / 02 / 03: Qwen 系列, 本地 torch + safetensors 加载
  - 04 / 05 (Ollama 后端): qwen2.5-coder:7b-instruct-q4_K_M (默认), 跑在本地 GPU
  - 05 (MiniMax 后端): MiniMax-M2.7 (~204K context, 云端 OpenAI 兼容端点)
- **运行环境**: Python 3.10+. 04 / 05 仅 stdlib (Ollama HTTP 协议在另一进程里), 01-03 要 torch + transformers
- **工具调用格式**:
  - 01-03: Qwen XML `<tool_call>...</tool_call>`, 自己正则解析
  - 04 / 05: OpenAI 结构化 `tool_calls` 字段 (后端 Ollama 或 MiniMax)
- **核心范式**: ReAct (Thought → Action → Observation) 循环 + 工具表分发
- **沙箱**: 基于路径, 在工具层强制, 不依赖 prompt
- **测试**: pytest. 03 引入 (42 个), 04 / 05 各 66 个. conftest.py 把 torch / transformers stub 掉, CI 不需要装 ML 栈

## 测试

CI 矩阵每次 push 自动跑全部 5 个版本 — 见 [.github/workflows/test.yml](.github/workflows/test.yml).

本地跑 (任选一个, 不需要全跑):

```bash
cd 05-session-and-streaming                                  # 推荐: 当前活跃版本
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/               # 66 个, ~0.8s
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 是绕过本地环境里冲突的 pytest 插件 (比如 langsmith). 没冲突就不用加.

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
