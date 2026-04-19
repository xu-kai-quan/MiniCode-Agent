# MiniCode Agent

> A minimal local-Qwen coding agent in a single Python file. ReAct loop, atomic tools, no framework. Three iterations from MVP to tested. Educational.

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![model](https://img.shields.io/badge/model-Qwen3.5--2B-orange)
![framework](https://img.shields.io/badge/framework-none-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

📖 English · [中文](README.md)

---

## Why this exists

Most "AI agent" tutorials wrap everything in **langchain / autogen / crewai**, so you spend more time reading framework docs than understanding the agent loop itself. This repo is the opposite:

- Every project is **one Python file**, no abstractions
- The entire ReAct loop is **visible in ~30 lines**
- Three iterations show **how a real agent grows** from MVP to something with tests, sandboxing, and optimistic locks

If you've ever asked "how does an AI coding agent actually work under the hood?" — read the source, not the framework docs.

## What this is NOT

- ❌ Not a framework — don't `pip install minicode-agent`, just read the code
- ❌ Not for production — small 2B model, no retry / observability / multi-tenancy
- ❌ Not multi-agent — single agent, single conversation
- ❌ Not cloud-API based — runs **100% locally** with a Qwen safetensors checkpoint

## The three iterations

| Directory | What's new in this step | Read it if... |
|---|---|---|
| [01-bash-only/](01-bash-only/) | **Minimal ReAct + 5 tools** (bash / read / write / edit / todo). One-shot, no REPL. | You're new to agents and want to see the smallest possible working skeleton. |
| [02-sandboxed/](02-sandboxed/) | **Interactive REPL** + **bash sandbox detection** (excludes WSL System32 to avoid UTF-16 issues) + chunked file writes for output too large for one turn | You want to see how an MVP grows into something usable day-to-day. |
| [03-atomic-tools/](03-atomic-tools/) | **Three-layer tool architecture** (LS/Glob/Grep/Read atomic layer) + **read-before-write optimistic lock** (mtime + size cache, NOT_READ / CONFLICT error codes) + **42 pytest tests** + GitHub Actions CI | You want to see how to make an agent solid enough to test. |

Every version is **a single `todo.py` file** — no langchain, no autogen, no hidden abstractions. **What you see is everything.**

## Quick start

```bash
git clone https://github.com/xu-kai-quan/MiniCode-Agent.git
cd MiniCode-Agent

# Install deps (pick your PyTorch wheel from https://pytorch.org)
pip install torch transformers safetensors

# Run the simplest version
cd 01-bash-only && python todo.py "create hello.py that prints hi"
```

You need a local copy of [Qwen3.5-2B](https://huggingface.co/Qwen) — the default path is at the top of each `todo.py` (`MODEL_DIR`), edit it for your machine. `03-atomic-tools` also reads `MINICODE_MODEL_DIR` env var.

## Tech stack

- **Model:** Qwen3.5-2B (local, no API)
- **Runtime:** Python 3.10+, PyTorch, transformers
- **Format:** Qwen XML tool-call format
  ```
  <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
  ```
- **Pattern:** ReAct (Thought → Action → Observation) loop with tool dispatch
- **Sandbox:** Path-based, enforced in tool layer (not in prompt)
- **Tests:** pytest, ML deps (torch / transformers) stubbed in `conftest.py` so CI runs in <1 second

## Tests

```bash
cd 03-atomic-tools
pip install pytest
pytest tests/        # 42 tests, ~0.6s, no torch needed
```

All three versions have CI on every push — see [.github/workflows/test.yml](.github/workflows/test.yml).

## Design principles

1. **The loop IS the agent.** Model output → parse tool calls → execute → feed result back. That's it.
2. **Tools are dispatch-table entries, not hardcoded.** Add a tool = add a function + register it. The loop never changes.
3. **Sandbox in the tool layer, never in the prompt.** A small model will not reliably respect "don't read files outside cwd". The `safe_path()` function will.
4. **Prompts are soft constraints, code is hard.** Critical invariants must be enforced in code, because small models don't reliably follow instructions.
5. **Logs should be auditable, not pretty.** Every step prints what the model thought, what tool it called, and what came back — so you can debug AND learn.

## When to read this vs. real frameworks

| Use this repo when you want to... | Use [aider](https://github.com/paul-gauthier/aider) / [continue.dev](https://continue.dev) / [Claude Code](https://claude.com/claude-code) when... |
|---|---|
| Understand how a coding agent works internally | You want a coding agent that just works |
| Build your own agent from scratch | You don't want to reinvent anything |
| Teach / learn agents in a workshop | You're shipping product |
| Run 100% locally with a small open model | You have API budget or a strong local GPU |

## Keywords

`qwen` `qwen3` `local-llm` `coding-agent` `react-agent` `ai-agent` `tool-use` `function-calling` `llm-agent` `no-framework` `educational` `single-file` `react-loop` `python` `from-scratch`
