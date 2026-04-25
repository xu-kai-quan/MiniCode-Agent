# MiniCode Agent

> A minimal coding agent in a single Python file. ReAct loop, atomic tools, no framework. Eight iterations from MVP to "actually usable" — local Qwen or cloud MiniMax-M2.7, sub-agents, resumable sessions, automatic context compression. Educational.

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![model](https://img.shields.io/badge/model-Qwen2.5%2FMiniMax--M2.7-orange)
![framework](https://img.shields.io/badge/framework-none-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

📖 English · [中文](README.md)

---

## Why this exists

Most "AI agent" tutorials wrap everything in **langchain / autogen / crewai**, so you spend more time reading framework docs than understanding the agent loop itself. This repo is the opposite:

- Every project is **one Python file**, no abstractions
- The entire ReAct loop is **visible in ~30 lines**
- Eight iterations show **how a real agent grows** from MVP to something with tests, sandboxing, optimistic locks, streaming, dual backends, sub-agents, resumable sessions, and automatic context compression

If you've ever asked "how does an AI coding agent actually work under the hood?" — read the source, not the framework docs.

## What this is NOT

- ❌ Not a framework — don't `pip install minicode-agent`, just read the code
- ❌ Not for production — no retry / observability / multi-tenancy
- ❌ Not a real "multi-agent system" — 06 added `spawn_agent` (parent dispatches a child in an isolated worktree), but **depth limit 1, synchronous blocking**. No parallel multi-agent / distributed coordination.

**About backends**: defaults to local Ollama running 7B; one line of config switches to cloud MiniMax-M2.7. That's a **capability**, not a "what this is not".

## The eight iterations — one active, seven archived snapshots

Each version introduces one or two new concepts. Read them in order to see how a working agent grows. **07 and below are archived snapshots** (kept frozen for reference, no longer modified). Active version: **08-context-compression**.

| Directory | Status | What's new in this step | Read it if... |
|---|---|---|---|
| [01-bash-only/](01-bash-only/) | archived | **Minimal ReAct + 5 tools** (bash / read / write / edit / todo). One-shot, no REPL. | You're new to agents and want to see the smallest possible working skeleton. |
| [02-sandboxed/](02-sandboxed/) | archived | **Interactive REPL** + **bash sandbox detection** (excludes WSL System32 to avoid UTF-16 issues) + chunked file writes for output too large for one turn | You want to see how an MVP grows into something usable day-to-day. |
| [03-atomic-tools/](03-atomic-tools/) | archived | **Three-layer tool architecture** (LS/Glob/Grep/Read atomic layer) + **read-before-write optimistic lock** (mtime + size cache, NOT_READ / CONFLICT error codes) + **42 pytest tests** + GitHub Actions CI | You want to see how to make an agent solid enough to test. |
| [04-structured-tool-calls/](04-structured-tool-calls/) | archived | **Ollama HTTP backend** (defaults to `qwen2.5-coder:7b`, big capability jump) + OpenAI-style structured `tool_calls` + **`apply_patch`** — cross-file unified diff with two-phase locking and atomic rollback + **66 pytest tests** | You want to see how an agent evolves from "it works" to "cross-file atomic edits". |
| [05-session-and-streaming/](05-session-and-streaming/) | archived | **Session state management** + **streaming output** with Ctrl-C interrupt + **dual backend** (Ollama or MiniMax-M2.7) + **token/cost reporting** + system-layer fallback against the model dodging tool_calls with code blocks | You want to see an agent grow from "it works" into something that actually feels usable. |
| [06-sub-agents/](06-sub-agents/) | archived | **`spawn_agent` tool** — when user says "try X", parent spawns a sub-agent in an isolated git worktree; sub-agent's diff comes back to the parent who asks the user whether to apply it to the main workspace (which never gets touched) + **76 pytest tests** (66 inherited + 10 new) + 5 real-REPL fix iterations documented in §4 | You want to see how the sub-agent paradigm lands in practice / want the "try a change without polluting main" workflow. |
| [07-session-persistence/](07-session-persistence/) | archived | **Cross-process session save/load** — `/save <name>` / `/load <name>` / `/sessions` / `/del <name>` REPL commands persist history + todos + cumulative tokens to `~/.minicode/sessions/<name>.json`. Long task interrupted by reboot or end-of-day? Resume it tomorrow. + **82 pytest tests** (76 inherited + 6 new) | You want session persistence — but v8 inherits all of it. Read 07's README for the design rationale, then run 08. |
| [08-context-compression/](08-context-compression/) | ✅ **active** | **Automatic history compression** — when a turn's real `prompt_tokens` crosses `CONTEXT_LIMIT × COMPRESS_AT` (default 70%), the next turn starts with the LLM summarizing middle messages into a single `<compressed>...</compressed>` user message; system prompt and the last `KEEP_TAIL` (default 10) messages are kept verbatim. Split point is forced outside any tool_call/result pair. LLM failure falls back to a mechanical structured summary. Manual `/compress` command also available. + **104 pytest tests** (82 inherited + 22 new) + end-to-end verified against real MiniMax | You want long sessions that don't blow the context window — **default recommendation**. |

Every version is **a single `todo.py` file** — no langchain, no autogen, no hidden abstractions. **What you see is everything.**

## Quick start — defaults to 08

```bash
git clone https://github.com/xu-kai-quan/MiniCode-Agent.git
cd MiniCode-Agent

# Install Ollama: https://ollama.com/
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

cd 08-context-compression
python todo.py
```

**No Python deps** at runtime (stdlib only). Ollama install + model pull is ~5 minutes. Once running, you get a `>` prompt with streaming output (Ctrl-C drops to interrupt without losing partial output). For long tasks, use `/save my-task` to checkpoint and `/load my-task` to resume; once prompt tokens cross 70% of the context limit, history auto-compresses, or you can `/compress` manually.

**For the cloud backend** (MiniMax-M2.7, noticeably better task continuity):

```bash
cd 08-context-compression
cp .env.example .env
# Edit .env: MINICODE_BACKEND=minimax + MINIMAX_API_KEY=<your key>
python todo.py
```

`.env` is gitignored; the repo also ships a pre-commit hook that scans for common secret prefixes (see [05 README §8](05-session-and-streaming/README.md#8-安全脚手架--env-和-pre-commit-hook)).

**Older versions** (educational, not for daily use):

- **01 / 02 / 03**: still use torch + transformers + local Qwen safetensors. `pip install torch transformers safetensors`, then cd in and `python todo.py`. Needs a local Qwen2.5 checkpoint; default path is at the top of each `todo.py`.
- **04**: same Ollama backend as 05+, but no streaming / Session / cloud / token display
- **05**: 04 + Session / streaming / dual backend / token visibility, no spawn_agent or persistence
- **06**: 05 + spawn_agent, no session persistence
- **07**: 06 + cross-process `/save` `/load`, no automatic history compression

## Tech stack

- **Models** (per version):
  - 01 / 02 / 03: Qwen series, local torch + safetensors
  - 04 / 05 (Ollama): qwen2.5-coder:7b-instruct-q4_K_M (default), runs on local GPU
  - 05 (MiniMax): MiniMax-M2.7 (~204K context, OpenAI-compatible cloud endpoint)
- **Runtime:** Python 3.10+. 04+ uses stdlib only (Ollama runs in another process); 01-03 require torch + transformers.
- **Tool-call format:**
  - 01-03: Qwen XML `<tool_call>...</tool_call>`, parsed with regex
  - 04+: OpenAI-style structured `tool_calls` (Ollama or MiniMax backend)
- **Pattern:** ReAct (Thought → Action → Observation) loop with tool-table dispatch
- **Sandbox:** Path-based, enforced in tool layer (not in prompt)
- **Tests:** pytest. 03 introduced 42, 04/05 have 66 each, 06 added spawn_agent tests (76), 07 added session-persistence tests (82), 08 added context-compression tests (104). `conftest.py` stubs torch/transformers so CI doesn't need the multi-GB ML stack.

## Tests

CI matrix runs all 8 versions on every push — see [.github/workflows/test.yml](.github/workflows/test.yml).

Run locally (pick one, no need for all):

```bash
cd 08-context-compression                                    # recommended: active version
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/               # 104 tests, ~7s
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` works around a flaky third-party pytest plugin in the user's local env (e.g. langsmith). Skip if you don't have such conflicts.

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

`qwen` `qwen3` `qwen2.5-coder` `minimax` `minimax-m2` `ollama` `local-llm` `coding-agent` `react-agent` `ai-agent` `tool-use` `function-calling` `llm-agent` `no-framework` `educational` `single-file` `react-loop` `python` `from-scratch` `streaming` `apply-patch` `unified-diff` `optimistic-locking` `session-management` `sub-agent` `spawn-agent` `git-worktree` `session-persistence` `resumable-sessions` `context-compression` `history-summarization`
