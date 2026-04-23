# MiniCode Agent

> 用本地 Qwen3.5-2B 驱动的**最小化编码 agent**, 单文件实现, 没有任何框架封装, 适合用来理解 "AI agent 到底是怎么工作的"。
>
> *A minimal local-Qwen coding agent in a single Python file. ReAct loop, atomic tools, no framework. Educational.*

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![model](https://img.shields.io/badge/model-Qwen3.5--2B-orange)
![framework](https://img.shields.io/badge/framework-none-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

📖 [English](README_EN.md) · 中文

---

## 为什么有这个项目

市面上 "AI agent" 教程几乎都把东西包进 **langchain / autogen / crewai** 里, 结果你花了大半时间在读框架文档, 而不是在搞懂 agent 循环本身。这个 repo 反过来:

- 每个项目都是**一个 Python 文件**, 没有任何抽象
- 整个 ReAct 循环大约 **30 行代码**, 摊开就能看完
- 三个迭代版本展示**一个真实 agent 是怎么从 MVP 长出来的** — 一路加上测试、沙箱、乐观锁

看遍了 langchain / autogen 的教程, 还是不知道 agent 内部到底怎么转?读源码, 不要读框架文档。

## 这个项目不是什么

- ❌ 不是框架 — 别 `pip install`, 直接读代码
- ❌ 不适合生产 — 模型只有 2B, 没有重试 / 可观测 / 多租户
- ❌ 不是多 agent — 单 agent, 单会话
- ❌ 不依赖云 API — 100% **本地** 跑, 用本地的 Qwen safetensors 权重

---

## 四个版本

| 目录 | 这一版的核心新东西 | 适合谁读 |
|---|---|---|
| [01-bash-only/](01-bash-only/) | 最小 ReAct + 5 个工具 (bash / read / write / edit / todo), 一次性任务模式, 跑完就退出 | 第一次接触 agent, 想看清最小骨架 |
| [02-sandboxed/](02-sandboxed/) | 加交互式 REPL + bash 沙箱探测 (排除 WSL System32, 避免 UTF-16 输出冲突) + 大文件分片写入 | 想看 agent 怎么演化成可用的日常工具 |
| [03-atomic-tools/](03-atomic-tools/) | 三层工具架构 (LS/Glob/Grep/Read 原子层) + 读后写乐观锁 (mtime + size cache, NOT_READ / CONFLICT) + 42 个 pytest + GitHub Actions CI | 想看怎么把 agent 做扎实, 经得起测试 |
| [04-atomic-tools/](04-atomic-tools/) | 后端换 Ollama HTTP (默认 `qwen2.5-coder:7b`, 能力跳档) + OpenAI 结构化 tool_calls + `apply_patch` 跨文件 unified diff (两阶段锁 + 原子回滚) + 66 个 pytest | 想看 agent 怎么从"能跑"演进到"跨文件原子改动" |

每一版都只有一个 `todo.py` 文件 — **你看到的就是全部真相**, 没有任何框架包装。

## 快速开始

```bash
git clone https://github.com/xu-kai-quan/MiniCode-Agent.git
cd MiniCode-Agent

# 装依赖 (按你 PyTorch 平台选 wheel: https://pytorch.org)
pip install torch transformers safetensors

# 跑最简单的版本
cd 01-bash-only && python todo.py "create hello.py that prints hi"
```

需要本地放一份 [Qwen3.5-2B](https://huggingface.co/Qwen) 模型 (默认路径在每个 `todo.py` 顶部, 改成你自己的)。`03-atomic-tools` 也支持读 `MINICODE_MODEL_DIR` 环境变量。

## 技术栈

- **模型:** Qwen3.5-2B (本地, 无 API 调用)
- **运行环境:** Python 3.10+, PyTorch, transformers
- **工具调用格式:** Qwen XML
  ```
  <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
  ```
- **核心范式:** ReAct (Thought → Action → Observation) 循环 + 工具表分发
- **沙箱:** 基于路径, 在工具层强制, 不依赖 prompt
- **测试:** pytest, conftest.py 把 torch / transformers stub 掉, CI 跑完不到 1 秒

## 测试

```bash
cd 03-atomic-tools
pip install pytest
pytest tests/        # 42 个测试, ~0.6 秒, 不需要 torch
```

四个版本都有 CI, 每次 push 自动跑 — 见 [.github/workflows/test.yml](.github/workflows/test.yml)。

## 设计理念

1. **循环就是 agent 的全部本质** — 模型输出 → 解析工具调用 → 执行 → 把结果喂回去, 直到模型不再调工具。仅此而已。
2. **工具是查表分发, 不是硬编码** — 加新工具 = 加一个函数 + 登记表项, 循环本身不动。
3. **沙箱在工具层做, 不在 prompt 里写** — 小模型不会乖乖听 "不要读 cwd 外的文件", 但 `safe_path()` 函数会拦住。
4. **prompt 是软约束, 代码是硬约束** — 关键不变量必须在代码里兜底, 因为小模型不一定听 prompt。
5. **日志要"可审查"而非"好看"** — 每一步: 模型在想什么 / 调了什么工具 / 拿到什么结果, 全部打印出来, 方便 debug 和学习。

## 相关项目

如果你想要的是**生产级**的本地编码 agent (而不是用来学习的), 去看 [aider](https://github.com/paul-gauthier/aider) / [continue.dev](https://continue.dev) / [Claude Code](https://claude.com/claude-code)。这个 repo 是为了让你**搞懂原理**, 不是日常工具。

## Keywords

`qwen` `qwen3` `local-llm` `coding-agent` `react-agent` `ai-agent` `tool-use` `function-calling` `llm-agent` `no-framework` `educational` `single-file` `react-loop` `python` `from-scratch`
