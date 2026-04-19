# MiniCode Agent

> **A minimal local-Qwen coding agent in a single Python file.** ReAct loop, atomic tools, no framework. Three iterations from MVP to tested. Educational.
>
> 用本地 Qwen3.5-2B 驱动的**最小化编码 agent**, 单文件实现, 没有任何框架封装, 适合用来理解 "AI agent 到底是怎么工作的"。

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![model](https://img.shields.io/badge/model-Qwen3.5--2B-orange)
![framework](https://img.shields.io/badge/framework-none-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

📖 [English](README_EN.md) · 中文

---

## Why this exists / 为什么有这个项目

Most "AI agent" tutorials wrap everything in **langchain / autogen / crewai**, so you end up reading framework docs instead of understanding the agent loop itself. This repo does the opposite:

- Every project is **one Python file**, no abstractions
- The entire ReAct loop is **visible in ~30 lines**
- Three iterations show **how a real agent grows** from MVP to something with tests, sandboxing, and optimistic locks

> 看遍了 langchain / autogen 的教程, 还是不知道 agent 内部到底怎么转?这个 repo 把所有概念摊开 — 一个文件, 30 行循环, 三个迭代版本, 看完就懂。

## What this is NOT

- ❌ Not a framework — don't `pip install minicode-agent`, just read the code
- ❌ Not for production — small 2B model, no retry / observability / multi-tenancy
- ❌ Not multi-agent — single agent, single conversation
- ❌ Not cloud-API based — runs **100% locally** with a Qwen safetensors checkpoint

---

## The three iterations / 三个版本

| 目录 | What's new in this step | 适合谁读 |
|---|---|---|
| [01-bash-only/](01-bash-only/) | **Minimal ReAct + 5 tools** (bash / read / write / edit / todo). One-shot, no REPL. | 第一次接触 agent, 想看清最小骨架 |
| [02-sandboxed/](02-sandboxed/) | **Interactive REPL** + **bash sandbox detection** (excludes WSL System32) + chunked file writes for long output | 想看 agent 怎么演化成可用的日常工具 |
| [03-atomic-tools/](03-atomic-tools/) | **Three-layer tool architecture** (LS/Glob/Grep/Read atomic layer) + **read-before-write optimistic lock** (mtime + size cache) + **42 pytest tests** + GitHub Actions CI | 想看怎么把 agent 做扎实, 经得起测试 |

每一版都只有一个 `todo.py` 文件 — **你看到的就是全部真相**, 没有任何框架包装。

## Quick start / 快速开始

```bash
git clone https://github.com/xu-kai-quan/MiniCode-Agent.git
cd MiniCode-Agent

# 装依赖 (按你 PyTorch 平台选 wheel: https://pytorch.org)
pip install torch transformers safetensors

# 跑最简单的版本
cd 01-bash-only && python todo.py "create hello.py that prints hi"
```

需要本地放一份 [Qwen3.5-2B](https://huggingface.co/Qwen) 模型 (默认路径在每个 `todo.py` 顶部, 改成你自己的)。

## Tech stack / 技术栈

- **Model:** Qwen3.5-2B (local, no API)
- **Runtime:** Python 3.10+, PyTorch, transformers
- **Format:** Qwen XML tool-call format (`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`)
- **Pattern:** ReAct (Thought → Action → Observation) loop with tool dispatch
- **Sandbox:** Path-based, enforced in tool layer (not in prompt)
- **Tests:** pytest, ML deps stubbed in `conftest.py` so CI runs in <1 second

## Tests / 测试

```bash
cd 03-atomic-tools
pip install pytest
pytest tests/        # 42 tests, ~0.6s, no torch needed
```

CI runs all three versions on every push — see [.github/workflows/test.yml](.github/workflows/test.yml).

## Design principles / 设计理念

1. **The loop IS the agent.** Model output → parse tool calls → execute → feed result back. That's it.
2. **Tools are dispatch-table entries, not hardcoded.** Add a tool = add a function + register it. Loop doesn't change.
3. **Sandbox in the tool layer, never in the prompt.** A small model will not respect "don't read files outside cwd". The `safe_path()` function will.
4. **Prompts are soft constraints, code is hard.** Critical invariants must be enforced in code because small models don't reliably follow instructions.
5. **Logs should be auditable, not pretty.** Every step prints what the model thought, what tool it called, and what came back — so you can debug AND learn.

## Related / 相关

If you want a production-grade local coding agent (not educational), look at [aider](https://github.com/paul-gauthier/aider), [continue.dev](https://continue.dev), or [Claude Code](https://claude.com/claude-code). This repo is for **understanding**, not for daily use.

## Keywords

`qwen` `qwen3` `local-llm` `coding-agent` `react-agent` `ai-agent` `tool-use` `function-calling` `llm-agent` `no-framework` `educational` `single-file` `react-loop` `python`
