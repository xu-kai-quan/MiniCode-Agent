"""MiniCode Agent v6 — sub-agents for "explore-without-pollution" workflow.

继承 v5 的 Session / 流式 / 双后端 / token 可见, 新增:

  spawn_agent(task, ...) — 派子 agent 在隔离的 git worktree 里跑探索式任务.
  返回结果包含子 agent 总结 + 改动的 unified diff. 主 agent 看到 diff 后决定
  是否 apply 到主 workspace. 失败的 worktree 默认保留供调试.

  典型工作流:
    用户: "把 foo.py 里 X 改成 Y, 看测试还能不能过"
    主 agent: spawn_agent(task="apply this diff, run pytest, return outcome")
              → [子 agent 在临时 worktree apply_patch + bash pytest]
              → 收到 {test_passed: True, diff: "..."}
              → 主 agent 报告并询问是否采纳
    用户: "采纳"
    主 agent: apply_patch(patch=spawn_result.diff) 到主 workspace
"""
from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import typing
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Union, get_args, get_origin
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


# ---------- .env 加载 (零依赖, 50 行内) ----------

def _load_dotenv(path: Path) -> None:
    """读 .env 文件, 把 KEY=value 填进 os.environ — 但不覆盖已存在的环境变量.

    支持: KEY=value, 行首/末空白, # 注释, KEY="quoted value".
    不支持: 多行值, ${VAR} 插值, export KEY=... — 简单足够.

    设计选择: 已存在的环境变量优先 — 让 shell 里 `set MINICODE_BACKEND=ollama` 能临时
    覆盖 .env 里的设置. .env 是默认, 命令行/shell 是覆盖.
    """
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 剥引号 (双或单)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass  # .env 读失败就当没有, 不阻塞启动


# 在读任何 MINICODE_* 环境变量前加载 .env. 项目根 (todo.py 同级) 的 .env 优先.
_load_dotenv(Path(__file__).parent / ".env")


# ---------- 配置 ----------

# Backend 选择: "ollama" (默认本地) 或 "minimax" (云端 OpenAI 兼容)
BACKEND = os.environ.get("MINICODE_BACKEND", "ollama").lower()

# Ollama 配置
OLLAMA_BASE_URL = os.environ.get("MINICODE_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("MINICODE_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

# MiniMax 配置 (OpenAI 兼容端点 — 见 README §backends).
# base_url 不要带 /v1 — client 内部拼 /v1/chat/completions, 跟 OllamaClient 一致.
MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

# 兼容旧名: MINICODE_MODEL 在 minimax backend 下也允许 (优先级低于 MINIMAX_MODEL)
if BACKEND == "minimax" and "MINIMAX_MODEL" not in os.environ:
    MINIMAX_MODEL = os.environ.get("MINICODE_MODEL", MINIMAX_MODEL)

# MODEL_NAME 是个老变量名, 留着供下游 print 兼容 — 实际选择在 load_model() 里
MODEL_NAME = MINIMAX_MODEL if BACKEND == "minimax" else OLLAMA_MODEL

REQUEST_TIMEOUT = int(os.environ.get("MINICODE_TIMEOUT", "300"))  # 秒, 首次加载模型可能慢
# 流式输出开关. REPL 默认开 (体验), pytest 默认关 (确定性).
# 设 MINICODE_STREAM=0 强制关.
USE_STREAM = os.environ.get("MINICODE_STREAM", "1") != "0"
# 流式 buffer 阈值: 累积到 \n 或超过这么多字符就刷一次. 太小屏幕会卡.
STREAM_FLUSH_CHARS = 80

# ----- 价格 (¥/M token) -----
# 估算用 — 以 MiniMax 官网最新价为准. 改这两个数字一处即可.
# (M2.7 当前估价: input ~¥1/M, output ~¥8/M; 我们取保守值, 略高一点不至于低估)
MINIMAX_PRICE_INPUT_PER_M = float(os.environ.get("MINIMAX_PRICE_IN", "2.0"))
MINIMAX_PRICE_OUTPUT_PER_M = float(os.environ.get("MINIMAX_PRICE_OUT", "8.0"))

# 流式后端不返回 usage 时, 用字符数粗估 token. 中英混合大致 1 token ≈ 3 字符.
# 估算误差 ±30%, 但 REPL 体验级别够用.
CHARS_PER_TOKEN_EST = 3

# ----- 子 agent 配置 (v6 新增) -----
# 子 agent 单次最多跑这么多轮 (主 agent 是 MAX_ROUNDS=20)
SUB_AGENT_MAX_ROUNDS = int(os.environ.get("MINICODE_SUB_MAX_ROUNDS", "10"))
# 子 agent 整体超时 (秒). 防止跑死锁住主流程.
SUB_AGENT_TIMEOUT = int(os.environ.get("MINICODE_SUB_TIMEOUT", "300"))
# spawn_agent 返回 text 里 diff 截断阈值. 太大撑爆 context, 太小用户看不全.
# 真 diff 完整保留在 data["diff"] 里, 这里只截 text 显示部分.
SUB_AGENT_DIFF_MAX_CHARS = int(os.environ.get("MINICODE_SUB_DIFF_MAX", "4000"))

# spawn_agent 创建 worktree 后写入 .git/info/exclude 的 patterns —
# 排除子 agent 跑命令时常生成的"垃圾", 让 git diff 只显示真改动.
# 注意: .git/info/exclude 是 worktree 局部的, 不污染主仓库的 .gitignore.
_WORKTREE_EXCLUDE_PATTERNS = [
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "node_modules/",
    ".venv/",
    "venv/",
    # .env 在主 repo 应当已经在 .gitignore 里, 但 worktree 这里再加一层
    # 双保险 — 防止 spawn 期间 .env 被列入 diff
    ".env",
    ".env.local",
]

WORKDIR = Path.cwd().resolve()
SYSTEM = textwrap.dedent(f"""\
    You are a coding agent.

    WORKSPACE: {WORKDIR}
    All tool path arguments are RELATIVE to this workspace. Pass paths
    like 'todo.py', 'src/foo.py', '.'. Absolute paths (/path/...,
    C:\\..., /usr/...) get rejected with PATH_ESCAPE. When unsure what's
    here, call LS first.

    TOOL PREFERENCE: prefer atomic tools (LS, Glob, Grep, Read) over
    bash — they return structured, predictable results. Use bash only
    for git, scripts, package managers, anything atomic tools don't cover.

    READ BEFORE WRITE: edit_file / apply_patch / write_file / append_file
    on an EXISTING file requires a prior Read on it. Brand-new files
    don't. CONFLICT means the file changed externally — Read again then
    retry.

    LINE RANGES: for "lines 800-900 of foo.py" call Read(path='foo.py',
    offset=800, limit=100). Read returns line numbers built-in. Do NOT
    cat the whole file — large files truncate and waste tokens.

    MULTI-FILE / MULTI-EDIT: prefer apply_patch (unified diff, atomic
    all-or-nothing). edit_file is for one localized replacement.

    PLANNING: tasks with more than one step — call todo first, keep it
    updated, exactly one item in_progress at a time. After a successful
    write/edit, if more steps remain, continue — don't stop with a
    summary mid-task.

    TOOL-CALL PROTOCOL (critical): when invoking a tool, emit a real
    structured tool_call. NEVER write the call as visible text — not as
    ```json, ```diff, ```patch, ```bash, ```sh, or any other code block
    containing what should have been a tool argument. The system treats
    text replies as final answers; nothing runs. RUN means tool_call;
    code blocks only SHOW example code to the user. Have a diff ready?
    Call apply_patch. Want to run a shell command? Call bash. Don't print
    them.

    SUB-AGENT (spawn_agent) — MANDATORY for exploratory changes:

    When the user asks you to TRY a change to see what happens, you MUST
    use spawn_agent. Do NOT directly edit_file/write_file/apply_patch the
    main workspace for an exploratory request. The user explicitly wants
    to see the diff and decide whether to apply it.

    Trigger words that mean "use spawn_agent":
      Chinese: "试一下", "试试", "试一试", "试着", "试验", "看看能不能",
               "看通不通过", "测一下", "尝试"
      English: "try", "experiment with", "see if", "check whether",
               "verify if", "test if X works after Y"

    Combined signal: trigger word + change/verification request
    (e.g. "试一下把 X 改成 Y, 跑测试" / "try modifying X to Y and run tests").

    The sub-agent runs in an isolated git worktree (temp checkout of HEAD).
    It returns a summary plus a unified diff. You then report to the user
    and ask whether to apply the diff to the main workspace via apply_patch.

    EXAMPLE — correct sub-agent use:
      user: "试一下把 todo.py 第 1 行改成 hhhhhhhh, 跑 pytest 看通不通过"
      assistant: [tool_call: spawn_agent(task="In foo.py replace line 1
                  with 'hhhhhhhh' (use edit_file), then run pytest and
                  report PASS/FAIL")]
      → sub-agent runs in /tmp/minicode-sub-XXX, returns diff + outcome
      → you tell user the result and ask "apply this diff to your
         workspace?" (do NOT auto-apply)
      → if user says yes: apply_patch(patch=<sub_agent's diff>)

    EXAMPLE — DO NOT use sub-agent (normal task):
      user: "把 hello.py 改成带类型提示的版本"
      assistant: [direct: Read(hello.py), then edit_file or apply_patch]
      (No "试" / "try" — user wants the change committed directly.)

    SHOWING THE DIFF TO USER: when reporting spawn_agent's diff back
    to the user, DO NOT wrap it in a ```diff or ```patch code block
    — that triggers the system's dodge-detection (which thinks you're
    pasting a diff instead of calling apply_patch). Show the diff as
    plain indented text or quote it line-by-line with leading "│ ".
    The user can read it just fine without code-block syntax highlighting.

    When NOT to use sub-agent: user wants the change committed (no
    "试/try"), pure read/explore tasks (Read/Grep/LS suffice), or
    multi-step planning tasks (use todo).

    Skip the mandate when user explicitly says "直接改" / "no sandbox" /
    "in main" — that's an opt-out signal.
""").strip()

# 子 agent 的 SYSTEM — 跟主 agent 完全不同. 强调:
#  - 它在隔离 worktree 里, 改动不会污染主 workspace
#  - 单一目标, 做完就总结, 不要规划/装腔
#  - 输出最终是要给主 agent 的 tool result, 简洁直接
SUB_AGENT_SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are a sub-agent spawned by the main agent to explore one specific change.

    ISOLATED WORKSPACE: {workdir}
    This is a git worktree of the main project. Any change you make stays
    here — the main agent will see your final diff and decide whether to
    apply it to the real workspace. So feel free to try things; you cannot
    break the user's actual code.

    YOUR TASK IS BOUNDED. Do exactly what the parent asked, then stop.
    - You do NOT have a `todo` tool — single goal, no planning needed.
    - You do NOT have a `spawn_agent` tool — no recursion.
    - All other tools are available (LS, Glob, Grep, Read, write_file,
      append_file, edit_file, apply_patch, bash).

    All path arguments are RELATIVE to the worktree above. Same rules as
    a normal agent for read-before-write, atomic apply_patch, etc.

    YOU ARE ALREADY AT THE WORKTREE — your cwd is exactly the workspace
    above. Do NOT `cd` into it from bash; just call commands directly:
    `pytest`, `python script.py`, etc. The bash tool runs with cwd = the
    worktree. Never use `cd /tmp/minicode-sub-...` — the system already
    placed you there.

    OUTPUT: keep your final reply terse — it becomes a tool_result for
    the parent. State the outcome in one or two sentences:
      - "PASS — N/M tests passed, change applied cleanly"
      - "FAIL — test_xyz failed: <reason>"
      - "DONE — changes made, no test run"
    The parent will get your message + the diff your changes produced.
    Don't repeat the diff in your reply; the system extracts it.

    No code blocks of `bash`/`diff`/`patch`/`json`/`shell` for fake tool
    calls — same critical rule as the parent. Use real tool_calls.
""").strip()

MAX_ROUNDS = 20
MAX_NEW_TOKENS = 4096

# 各工具的截断/输出阈值. 集中在此, 改一处即可.
GREP_MAX_MATCHES = 200
GREP_MAX_FILES_SCAN = 5000
LS_MAX_ENTRIES = 500
GLOB_MAX_PATHS = 500
READ_MAX_CHARS = 8000
BASH_MAX_CHARS = 8000


# =================================================================
# ToolResult — 工具调用的统一返回结构
# =================================================================

@dataclass
class ToolResult:
    """工具调用的统一返回结构.

    - status: 三态.
        success — 任务完成且没有信息被丢弃.
        partial — 任务完成但部分内容被截断 (用户预期 vs 实际之间有差).
        error   — 任务失败. text 描述错误, data["code"] 给机器读的错误码.
    - data:   结构化结果 (entries / paths / matches / content...). 只进日志.
    - text:   自然语言摘要. 喂给模型的就是这段.
    """
    status: str           # "success" | "partial" | "error"
    text: str
    data: dict = field(default_factory=dict)

    @classmethod
    def success(cls, text: str, **data: Any) -> "ToolResult":
        return cls("success", text, data)

    @classmethod
    def partial(cls, text: str, **data: Any) -> "ToolResult":
        return cls("partial", text, data)

    @classmethod
    def error(cls, code: str, message: str, **extra: Any) -> "ToolResult":
        data = {"code": code, "message": message, **extra}
        return cls("error", f"error [{code}]: {message}", data)


# =================================================================
# Message — 统一的会话消息数据结构
# =================================================================

@dataclass
class Message:
    """一条对话消息. 替代裸 dict, 让字段约定显式化."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None  # role=tool 时对应的调用 id

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls("system", text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str, tool_calls: list[dict] | None = None) -> "Message":
        return cls("assistant", text, tool_calls or None)

    @classmethod
    def tool(cls, output: str, tool_call_id: str | None = None) -> "Message":
        return cls("tool", output, tool_call_id=tool_call_id)


# =================================================================
# ToolRegistry — 工具的注册与调用分发
# =================================================================

@dataclass
class Tool:
    """一个工具 = 名字 + JSON schema + handler."""
    name: str
    description: str
    parameters: dict
    handler: Callable[..., ToolResult]

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_function(
        cls,
        func: Callable[..., ToolResult],
        description: str,
        *,
        name: str | None = None,
        handler: Callable[..., ToolResult] | None = None,
        param_docs: dict[str, str] | None = None,
        overrides: dict[str, dict] | None = None,
        exclude: set[str] | None = None,
    ) -> "Tool":
        """从 handler 签名反射出 JSON schema — 默认值/类型/required 都不用手写.

        - func: 用它的 signature + type hints 生成 schema. 也默认当 handler.
        - handler: 真正被 dispatch 调用的函数. 需要闭包额外依赖时才传 (如 apply_patch).
        - param_docs: {参数名: 描述}. 模型看到的参数说明写这里, 不靠 docstring 解析.
        - overrides: {参数名: JSON schema 片段}. 反射搞不定的参数 (如 list[TypedDict]) 手工覆盖.
        - exclude: 从 schema 里剔掉的参数名 (模型不该看见, 由 handler 闭包注入).
        """
        params = _schema_from_signature(
            func, param_docs or {}, overrides or {}, exclude or set(),
        )
        return cls(
            name=name or func.__name__,
            description=description,
            parameters=params,
            handler=handler or func,
        )


# Python 类型 → JSON Schema 基本类型. 没列出的类型(如 Path)走 object, 但目前没这需求.
_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _type_to_schema(tp: Any) -> dict:
    """把 Python 类型注解翻成 JSON Schema 片段. 只覆盖当前工具用到的形态."""
    origin = get_origin(tp)

    # X | None / Optional[X] / Union[X, None]: 剥掉 None, 取另一边.
    if origin is Union or origin is UnionType:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        # 多选一的 Union 当前没工具用; 真要用再扩 oneOf.
        return {}

    # list[X] / dict[...] 等带参数的泛型: 只认外壳类型.
    if origin in _PY_TO_JSON_TYPE:
        return {"type": _PY_TO_JSON_TYPE[origin]}

    if isinstance(tp, type) and tp in _PY_TO_JSON_TYPE:
        return {"type": _PY_TO_JSON_TYPE[tp]}

    return {}  # 兜底: 不声明类型, schema 仍合法, 只是信息更少


def _schema_from_signature(
    func: Callable[..., Any],
    param_docs: dict[str, str],
    overrides: dict[str, dict],
    exclude: set[str],
) -> dict:
    """inspect.signature + get_type_hints 反射出 {properties, required}."""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # *args/**kwargs 不进 schema
        if pname.startswith("_"):
            continue  # _session 等内部参数: 由 dispatch 注入, 不暴露给模型
        if pname in exclude:
            continue

        if pname in overrides:
            properties[pname] = dict(overrides[pname])
        else:
            schema = _type_to_schema(hints.get(pname, Any))
            if pname in param_docs:
                schema["description"] = param_docs[pname]
            properties[pname] = schema

        if p.default is inspect.Parameter.empty:
            required.append(pname)

    return {"type": "object", "properties": properties, "required": required}


class ReadCache:
    """记录每个文件被 Read 时的 (mtime_ns, size). 给乐观锁用.

    - 模型 Read 一次, cache 一条记录.
    - 模型再次 Write/Edit/Append 该文件时, dispatch 用这条记录验证文件没被外部修改过.
    - 模型不需要知道 cache 存在 — 它只在 NOT_READ / CONFLICT 错误的 text 里看到反馈.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, int]] = {}

    def record(self, abs_path: Path) -> None:
        try:
            st = abs_path.stat()
            self._entries[str(abs_path)] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass

    def get(self, abs_path: Path) -> tuple[int, int] | None:
        return self._entries.get(str(abs_path))

    def forget(self, abs_path: Path) -> None:
        self._entries.pop(str(abs_path), None)

    def clear(self) -> None:
        self._entries.clear()


class TodoManager:
    """todo 列表的状态. 之前是模块级单例 TODO, 现在挂在 Session 上, 每个 session 独立."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, items: list[dict]) -> ToolResult:
        in_progress = sum(1 for it in items if it.get("status") == "in_progress")
        if in_progress > 1:
            return ToolResult.error(
                "INVALID_STATE",
                "Only one task can be in_progress at a time",
            )
        self.items = [
            {"id": it["id"], "text": it["text"], "status": it.get("status", "pending")}
            for it in items
        ]
        return ToolResult.success(self.render(), items=self.items)

    def render(self) -> str:
        mark = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        if not self.items:
            return "(no todos)"
        return "\n".join(f"{mark[it['status']]} {it['text']}" for it in self.items)


@dataclass
class Session:
    """一次 agent run 的全部可变状态.

    把 history / todo / read_cache / 计数器集中到一个对象, 替代 04 散在
    `agent.history` + 模块级 `TODO` + `registry.read_cache` 三处的状态.

    new_session() 重新构造这个对象就等于"开新会话" — reset 所有状态一步到位.
    """
    history: list[Message]
    todo: TodoManager = field(default_factory=TodoManager)
    read_cache: ReadCache = field(default_factory=ReadCache)
    rounds_since_todo: int = 0
    # 连续多少轮 "无 tool_call 但 visible 里有疑似工具调用代码块" — 干预计数器.
    # 触达 CODEBLOCK_NAG_LIMIT (在 ReActAgent 上) 后 hard DONE, 防死循环.
    codeblock_nag_count: int = 0
    # token 累计 — 每个 turn 的 usage 累加进来 (供 REPL 显示成本).
    # 流式模式下后端可能不返回 usage, 这时是 _StreamRenderer 估算的值, 标 estimated.
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    # 跟踪有多少 turn 的 token 数是估算的 (对 ollama 流式恒为 True, 对 minimax 看运气)
    estimated_turns: int = 0
    exact_turns: int = 0
    # v6: 标识这个 session 是否是子 agent 的. 子 agent 不该被 todo NAG 打扰
    # (它根本没 todo 工具, NAG 提醒是噪音). 主 agent 此值默认 False.
    is_subagent: bool = False

    @classmethod
    def new(cls, system_prompt: str, is_subagent: bool = False) -> "Session":
        return cls(history=[Message.system(system_prompt)], is_subagent=is_subagent)


# 写类工具集中在此. dispatch 用这个集合决定是否启用乐观锁检查.
WRITE_TOOLS = frozenset({"write_file", "edit_file", "append_file"})


class ToolRegistry:
    """工具表. 注册时同时登记 schema 和 handler — 加工具只改一处.

    dispatch 还兼两个职责:
      1. 对写类工具做"读后写 + 乐观锁"检查 (用 self.session.read_cache);
      2. 给 handler 注入 _session — handler 通过签名声明 `_session: Session`
         即可拿到当前 session 引用 (apply_patch / todo 用得上, 其他工具不声明就不注入).
    """

    def __init__(self, session: Session | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        # session 没传就 new 一个空 session — 单元测试里很常见
        self.session = session if session is not None else Session.new("")
        # 缓存 handler 是否需要 _session, 避免每次 dispatch 都 inspect 一遍
        self._handler_takes_session: dict[str, bool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        # 一次性查清楚 handler 要不要 _session
        try:
            sig = inspect.signature(tool.handler)
            self._handler_takes_session[tool.name] = "_session" in sig.parameters
        except (TypeError, ValueError):
            self._handler_takes_session[tool.name] = False

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def _check_read_before_write(self, name: str, arguments: dict) -> ToolResult | None:
        """写类工具的前置检查. 返回 ToolResult 表示拦截, None 表示放行.

        规则:
        - 新建文件 (路径不存在) — 放行. 模型不需要先 Read 一个不存在的文件.
        - 覆盖/追加/编辑已存在的文件 — 必须先 Read. 否则 NOT_READ.
        - 已 Read 但 mtime/size 变了 — CONFLICT, 模型必须重新 Read.
        """
        if name not in WRITE_TOOLS:
            return None
        path_str = arguments.get("path")
        if not isinstance(path_str, str):
            return None  # 缺 path, 让 handler 自己报 BAD_ARGS
        try:
            abs_path = _safe_path(WORKDIR, path_str)
        except ValueError:
            return None  # 路径越界, 让 handler 报 PATH_ESCAPE

        if not abs_path.exists():
            return None  # 新文件, 不需要先 Read

        cached = self.session.read_cache.get(abs_path)
        if cached is None:
            return ToolResult.error(
                "NOT_READ",
                f"File '{path_str}' must be read before {name}. "
                f"Call Read first to load it into context.",
            )
        try:
            st = abs_path.stat()
        except OSError as e:
            return ToolResult.error("STAT_FAILED", str(e))
        if (st.st_mtime_ns, st.st_size) != cached:
            return ToolResult.error(
                "CONFLICT",
                f"File '{path_str}' changed since last read "
                f"(was {cached[1]} bytes, now {st.st_size}). "
                f"Re-read it before retrying.",
            )
        return None

    def _post_write_refresh(self, name: str, arguments: dict) -> None:
        """写成功后用新 stat 刷新 cache.

        语义: 这次写产生的新内容现在算"已知最新". 模型可以接着 edit/append 同一文件
        而不必重新 Read. 但若外部进程修改, cache stat 就对不上 → 下次写会 CONFLICT.
        """
        if name not in WRITE_TOOLS:
            return
        path_str = arguments.get("path")
        if not isinstance(path_str, str):
            return
        try:
            abs_path = _safe_path(WORKDIR, path_str)
        except ValueError:
            return
        self.session.read_cache.record(abs_path)

    def dispatch(self, name: str, arguments: dict) -> ToolResult:
        """执行工具, 返回 ToolResult. 异常被吃掉转成 error result."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error("UNKNOWN_TOOL", f"Unknown tool: {name}")

        guard = self._check_read_before_write(name, arguments)
        if guard is not None:
            return guard

        # _session 注入: handler 声明了就传, 没声明就不传.
        # 这样 8/10 的纯函数 handler 不感知 session, 只有 apply_patch/todo 等需要时才取.
        try:
            if self._handler_takes_session.get(name):
                result = tool.handler(_session=self.session, **arguments)
            else:
                result = tool.handler(**arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult.success(str(result))
        except TypeError as e:
            return ToolResult.error("BAD_ARGS", f"{e}")
        except Exception as e:
            return ToolResult.error("EXEC_FAILED", f"{type(e).__name__}: {e}")

        # Read 通过 data["_stat"] 把 (mtime_ns, size) 回传给 dispatch — 写入 cache 后剥掉,
        # 这是工具→registry 的内部约定, 既不污染 handler 也不进日志面板.
        stat_info = result.data.pop("_stat", None) if result.data else None
        if stat_info is not None:
            path_str = arguments.get("path")
            if isinstance(path_str, str):
                try:
                    abs_path = _safe_path(WORKDIR, path_str)
                    self.session.read_cache._entries[str(abs_path)] = stat_info
                except ValueError:
                    pass

        if result.status != "error":
            self._post_write_refresh(name, arguments)
        return result


# =================================================================
# 路径沙箱
# =================================================================

def _safe_path(workdir: Path, p: str) -> Path:
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        # 错误消息直接告诉模型怎么修 — 比单纯说"escapes" 让模型不至于放弃.
        # README §三 4 提过 7B 在错误后容易"任务放弃", 把可操作建议放进错误本身是
        # 最直接的兜底.
        raise ValueError(
            f"Path '{p}' is outside the workspace ({workdir}). "
            f"Use a path RELATIVE to the workspace — e.g. 'todo.py', "
            f"'src/foo.py', or '.' for the workspace root. "
            f"If you don't know what's there, call LS with no arguments first."
        )
    return path


# =================================================================
# 原子工具: LS / Glob / Grep / Read / Write / Append / Edit
# =================================================================

def tool_ls(path: str = ".") -> ToolResult:
    """列目录. 不递归 — 递归请用 Glob."""
    try:
        target = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    if not target.exists():
        return ToolResult.error("NOT_FOUND", f"Path '{path}' does not exist")
    if not target.is_dir():
        return ToolResult.error("NOT_A_DIR", f"Path '{path}' is not a directory")

    entries: list[dict] = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            stat = child.stat()
            entries.append({
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": stat.st_size if child.is_file() else None,
            })
        except OSError:
            continue

    truncated = len(entries) > LS_MAX_ENTRIES
    shown = entries[:LS_MAX_ENTRIES]
    if not shown:
        return ToolResult.success(
            f"Directory '{path}' is empty",
            entries=[], path=path,
        )
    lines = [
        f"{e['name']}/" if e["type"] == "dir"
        else f"{e['name']}  ({e['size']} bytes)"
        for e in shown
    ]
    text = f"{len(entries)} entries in '{path}':\n" + "\n".join(lines)
    if truncated:
        return ToolResult.partial(
            text + f"\n... (truncated, {len(entries) - LS_MAX_ENTRIES} more)",
            entries=shown, path=path, total=len(entries),
        )
    return ToolResult.success(text, entries=shown, path=path)


def tool_glob(pattern: str, path: str = ".") -> ToolResult:
    """按文件名模式找文件 (递归). 用 Path.rglob, 跨平台."""
    try:
        base = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    if not base.exists():
        return ToolResult.error("NOT_FOUND", f"Path '{path}' does not exist")

    paths: list[str] = []
    try:
        for p in base.rglob(pattern):
            if p.is_file():
                paths.append(str(p.relative_to(WORKDIR)).replace("\\", "/"))
                if len(paths) >= GLOB_MAX_PATHS:
                    break
    except (OSError, ValueError) as e:
        return ToolResult.error("GLOB_FAILED", str(e))

    if not paths:
        return ToolResult.success(
            f"No files matching '{pattern}' under '{path}'",
            paths=[], pattern=pattern,
        )

    truncated = len(paths) >= GLOB_MAX_PATHS
    text = f"Found {len(paths)} file(s) matching '{pattern}':\n" + "\n".join(paths)
    if truncated:
        return ToolResult.partial(
            text + f"\n... (capped at {GLOB_MAX_PATHS}; refine pattern)",
            paths=paths, pattern=pattern,
        )
    return ToolResult.success(text, paths=paths, pattern=pattern)


# Grep: 优先 rg, 没有就退到纯 Python.
_RG_PATH = shutil.which("rg")


def _grep_with_rg(pattern: str, path: str, ignore_case: bool, file_pattern: str | None) -> ToolResult:
    target = _safe_path(WORKDIR, path)
    cmd = [_RG_PATH, "--no-heading", "--with-filename", "--line-number",
           "--max-count", str(GREP_MAX_MATCHES + 1), "--color=never"]
    if ignore_case:
        cmd.append("-i")
    if file_pattern:
        cmd.extend(["-g", file_pattern])
    cmd.extend(["--", pattern, str(target)])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ToolResult.error("TIMEOUT", "rg timed out after 15s")

    # rg: 0=有匹配, 1=无匹配, >1=出错
    if r.returncode > 1:
        return ToolResult.error("GREP_FAILED", (r.stderr or "rg failed").strip())

    matches: list[dict] = []
    for line in (r.stdout or "").splitlines():
        # format: path:lineno:content
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, lineno, content = parts
        try:
            rel = str(Path(file_path).resolve().relative_to(WORKDIR)).replace("\\", "/")
        except (ValueError, OSError):
            rel = file_path
        matches.append({"file": rel, "line": int(lineno), "text": content})
    return _grep_format(pattern, matches)


def _grep_with_python(pattern: str, path: str, ignore_case: bool, file_pattern: str | None) -> ToolResult:
    """纯 Python 兜底. 慢但跨平台无依赖."""
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return ToolResult.error("BAD_REGEX", f"invalid regex: {e}")

    base = _safe_path(WORKDIR, path)
    matches: list[dict] = []
    capped = False

    def _matches_filter(name: str) -> bool:
        return file_pattern is None or fnmatch.fnmatch(name, file_pattern)

    targets: list[Path] = []
    if base.is_file():
        if _matches_filter(base.name):
            targets.append(base)
    elif base.is_dir():
        for root, dirs, files in os.walk(base):
            # 跳过常见的大型/无用目录, 否则跑得慢
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", ".venv", "venv")]
            for name in files:
                if not _matches_filter(name):
                    continue
                targets.append(Path(root) / name)
                if len(targets) >= GREP_MAX_FILES_SCAN:
                    capped = True
                    break
            if capped:
                break

    for fp in targets:
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        try:
                            rel = str(fp.resolve().relative_to(WORKDIR)).replace("\\", "/")
                        except (ValueError, OSError):
                            rel = str(fp)
                        matches.append({"file": rel, "line": i, "text": line.rstrip("\n")})
                        if len(matches) > GREP_MAX_MATCHES:
                            return _grep_format(pattern, matches, scan_capped=capped)
        except (OSError, UnicodeError):
            continue

    return _grep_format(pattern, matches, scan_capped=capped)


def _grep_format(pattern: str, matches: list[dict], scan_capped: bool = False) -> ToolResult:
    if not matches:
        return ToolResult.success(
            f"No matches for '{pattern}'",
            matches=[], pattern=pattern,
        )

    truncated = len(matches) > GREP_MAX_MATCHES
    shown = matches[:GREP_MAX_MATCHES]
    lines = [f"{m['file']}:{m['line']}: {m['text']}" for m in shown]
    text = f"{len(shown)} match(es) for '{pattern}':\n" + "\n".join(lines)

    if truncated or scan_capped:
        suffix_parts = []
        if truncated:
            suffix_parts.append(f"capped at {GREP_MAX_MATCHES} matches")
        if scan_capped:
            suffix_parts.append(f"scanned only first {GREP_MAX_FILES_SCAN} files")
        return ToolResult.partial(
            text + f"\n... ({'; '.join(suffix_parts)}; refine pattern or path)",
            matches=shown, pattern=pattern,
        )
    return ToolResult.success(text, matches=shown, pattern=pattern)


def tool_grep(pattern: str, path: str = ".", ignore_case: bool = False,
              file_pattern: str | None = None) -> ToolResult:
    """按内容搜索. 有 rg 用 rg, 没有用纯 Python — 同一个开箱即用承诺.
    file_pattern 限定文件名 glob (如 '*.py'), 让模型不用先 Glob 再 Grep."""
    try:
        _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    target = (WORKDIR / path).resolve()
    if not target.exists():
        return ToolResult.error("NOT_FOUND", f"Path '{path}' does not exist")

    if _RG_PATH:
        return _grep_with_rg(pattern, path, ignore_case, file_pattern)
    return _grep_with_python(pattern, path, ignore_case, file_pattern)


def tool_read(path: str, limit: int | None = None, offset: int = 0) -> ToolResult:
    """带行号读取. limit 是用户契约 → success; 字符上限是系统兜底 → partial.

    流式实现 (优化 G): 用 file.iter() 逐行读, 只在内存持有 limit 行的窗口.
    无 limit 时仍要扫全文 — 模型可能真要全文; 但极少触发, 触发也是模型有意为之.
    用这种实现, 50MB 文件 + Read(limit=100) 只占 100 行内存, 不再 OOM.
    """
    try:
        p = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    if not p.exists():
        return ToolResult.error("NOT_FOUND", f"File '{path}' does not exist")
    if not p.is_file():
        return ToolResult.error("NOT_A_FILE", f"Path '{path}' is not a file")

    start = max(0, offset)
    selected: list[str] = []   # 落在 [start, start+limit) 窗口里的行
    total = 0                  # 文件总行数 — 一直数到 EOF

    try:
        st = p.stat()
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, raw in enumerate(f):
                total += 1
                line = raw.rstrip("\r\n")  # 跨平台: Windows \r\n / Unix \n 都剥掉
                if i < start:
                    continue
                if limit is not None and len(selected) >= limit:
                    # 窗口已满, 但要继续读到 EOF 才能给出 total — 不再 append
                    continue
                selected.append(line)
    except OSError as e:
        return ToolResult.error("READ_FAILED", str(e))

    end = start + len(selected)

    numbered = "\n".join(f"{i + 1:6d}\t{line}" for i, line in enumerate(selected, start=start))
    char_capped = len(numbered) > READ_MAX_CHARS
    if char_capped:
        numbered = numbered[:READ_MAX_CHARS]

    header = f"{path} (lines {start + 1}-{end} of {total})"
    text = f"{header}\n{numbered}"
    # _stat 是内部约定: dispatch 看到这个 key 会回写 read_cache. 模型看不到 (data 不喂模型).
    data = {
        "path": path, "content": numbered,
        "lines": (start + 1, end), "total_lines": total,
        "_stat": (st.st_mtime_ns, st.st_size),
    }

    # 用户传了 limit 拿到 limit 行 → success (契约).
    # 没传 limit 但行数被字符上限砍掉 → partial (没拿全).
    if char_capped:
        return ToolResult.partial(
            text + "\n... (output truncated by char limit; pass smaller limit/offset)",
            **data,
        )
    return ToolResult.success(text, **data)


def tool_write(path: str, content: str) -> ToolResult:
    try:
        p = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return ToolResult.error("WRITE_FAILED", str(e))
    return ToolResult.success(
        f"wrote {len(content)} chars to {path}",
        path=path, bytes_written=len(content),
    )


def tool_append(path: str, content: str) -> ToolResult:
    try:
        p = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        size = p.stat().st_size
    except OSError as e:
        return ToolResult.error("APPEND_FAILED", str(e))
    return ToolResult.success(
        f"appended {len(content)} chars to {path} (now {size} bytes)",
        path=path, total_bytes=size,
    )


def tool_edit(path: str, old_text: str, new_text: str) -> ToolResult:
    try:
        p = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    if not p.exists():
        return ToolResult.error("NOT_FOUND", f"File '{path}' does not exist")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult.error("READ_FAILED", str(e))

    count = text.count(old_text)
    if count == 0:
        return ToolResult.error("NO_MATCH", f"old_text not found in {path}")
    if count > 1:
        return ToolResult.error(
            "AMBIGUOUS", f"old_text matches {count} times in {path}; make it unique",
            occurrences=count,
        )
    try:
        p.write_text(text.replace(old_text, new_text), encoding="utf-8")
    except OSError as e:
        return ToolResult.error("WRITE_FAILED", str(e))
    return ToolResult.success(f"edited {path}", path=path)


# =================================================================
# apply_patch — 多文件 unified diff, 两阶段锁 + 原子回滚
# =================================================================
#
# 为什么单独做这个而不是循环调 edit_file:
#   - edit_file 一次一处, 跨文件修改要 N 次 tool_call, 模型易半路放弃.
#   - 这里做"全部预检通过才开始写, 任一写失败全部回滚", 语义接近 git apply.
# 不信任 hunk 头的行号 (@@ -N,M +N,M @@) — 小模型经常算错, 只用上下文行定位.

@dataclass
class _Hunk:
    old_block: str   # ' ' + '-' 行拼接 (用来在原文里定位)
    new_block: str   # ' ' + '+' 行拼接 (替换成这段)


@dataclass
class _FilePatch:
    path: str                  # 仓内相对路径 (来自 +++ b/path, 除非是删除)
    op: str                    # "modify" | "create" | "delete"
    hunks: list[_Hunk]         # create 时整个新文件塞进 hunks[0].new_block; delete 时为空


def _strip_ab_prefix(p: str) -> str:
    """把 'a/foo.py' / 'b/foo.py' 剥成 'foo.py'. /dev/null 原样返回."""
    if p == "/dev/null":
        return p
    if p.startswith(("a/", "b/")):
        return p[2:]
    return p


def _parse_unified_diff(patch: str) -> list[_FilePatch]:
    """解析 unified diff. 出错抛 ValueError, 由调用方包成 PARSE_FAILED."""
    lines = patch.splitlines()
    i = 0
    files: list[_FilePatch] = []
    while i < len(lines):
        line = lines[i]
        if not line.startswith("--- "):
            i += 1
            continue
        old_path = _strip_ab_prefix(line[4:].split("\t", 1)[0].strip())
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise ValueError(f"'--- {old_path}' not followed by '+++ ' line")
        new_path = _strip_ab_prefix(lines[i + 1][4:].split("\t", 1)[0].strip())
        i += 2

        if old_path == "/dev/null" and new_path == "/dev/null":
            raise ValueError("patch header has /dev/null on both sides")
        if old_path == "/dev/null":
            op, path = "create", new_path
        elif new_path == "/dev/null":
            op, path = "delete", old_path
        else:
            # 不强制 old == new (允许未来扩展重命名), 但目前按 new_path 作为目标路径
            op, path = "modify", new_path

        hunks: list[_Hunk] = []
        while i < len(lines) and lines[i].startswith("@@"):
            i += 1  # 跳过 @@ 头, 不解析数字
            old_buf: list[str] = []
            new_buf: list[str] = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                row = lines[i]
                if not row:
                    # 裸空行是 hunk 终止符 (分隔两个文件段), 不属于 hunk 内容.
                    # unified diff 里 hunk 内真正的空行形如 ' ' (一个空格 + 空内容),
                    # 会走下面的 row[0] == ' ' 分支.
                    break
                if row[0] == " ":
                    old_buf.append(row[1:])
                    new_buf.append(row[1:])
                elif row[0] == "-":
                    old_buf.append(row[1:])
                elif row[0] == "+":
                    new_buf.append(row[1:])
                elif row.startswith("\\ No newline"):
                    pass  # 忽略尾行标记
                else:
                    raise ValueError(f"bad hunk line (expected ' ', '-', '+'): {row!r}")
                i += 1
            hunks.append(_Hunk(old_block="\n".join(old_buf), new_block="\n".join(new_buf)))

        if op == "create":
            if len(hunks) != 1 or hunks[0].old_block:
                raise ValueError(f"create patch for {path} must have exactly one hunk with no '-' lines")
        elif op == "delete":
            if hunks and any(h.new_block for h in hunks):
                raise ValueError(f"delete patch for {path} must not have '+' lines")
        else:
            if not hunks:
                raise ValueError(f"modify patch for {path} has no hunks")

        files.append(_FilePatch(path=path, op=op, hunks=hunks))
    if not files:
        raise ValueError("no file headers ('--- '/'+++ ') found in patch")
    return files


def _apply_hunks_to_text(text: str, hunks: list[_Hunk]) -> str:
    """按顺序对 text 应用每个 hunk. old_block 必须唯一出现, 否则抛 ValueError."""
    cursor = 0
    out_parts: list[str] = []
    for idx, h in enumerate(hunks):
        if not h.old_block:
            # 纯插入 hunk 没上下文无法定位 — 在真实 patch 里极少见, 拒绝
            raise ValueError(f"hunk #{idx + 1} has no context/removal lines to locate")
        hay = text[cursor:]
        first = hay.find(h.old_block)
        if first < 0:
            raise ValueError(f"hunk #{idx + 1}: context not found in file")
        second = hay.find(h.old_block, first + 1)
        if second >= 0:
            raise ValueError(f"hunk #{idx + 1}: context matches multiple places; add more context")
        out_parts.append(hay[:first])
        out_parts.append(h.new_block)
        cursor += first + len(h.old_block)
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def tool_apply_patch(patch: str, _session: Session) -> ToolResult:
    """应用 unified diff. 两阶段: 预检锁 + 解析 → 构造新内容 → 一次性写盘 + 回滚.

    _session 由 dispatch 自动注入 (handler 签名声明就给, 名字以下划线开头不进 schema).
    """
    read_cache = _session.read_cache
    try:
        file_patches = _parse_unified_diff(patch)
    except ValueError as e:
        return ToolResult.error("PARSE_FAILED", str(e))

    # 路径越界一并在 phase 1 拦下
    resolved: list[tuple[_FilePatch, Path]] = []
    for fp in file_patches:
        try:
            abs_path = _safe_path(WORKDIR, fp.path)
        except ValueError as e:
            return ToolResult.error("PATH_ESCAPE", str(e), path=fp.path)
        resolved.append((fp, abs_path))

    # Phase 1: 锁 + 预读每个 modify/delete 的原内容 (也当回滚 backup).
    # create: 文件必须不存在; modify/delete: 必须存在, 必须已 Read, mtime/size 没变.
    originals: dict[Path, str | None] = {}  # None 表示原本不存在 (create 时)
    for fp, abs_path in resolved:
        if fp.op == "create":
            if abs_path.exists():
                return ToolResult.error(
                    "ALREADY_EXISTS",
                    f"create patch targets existing file: {fp.path}",
                    path=fp.path,
                )
            originals[abs_path] = None
            continue

        if not abs_path.exists():
            return ToolResult.error("NOT_FOUND", f"file not found: {fp.path}", path=fp.path)

        cached = read_cache.get(abs_path)
        if cached is None:
            return ToolResult.error(
                "NOT_READ",
                f"file must be read before apply_patch: {fp.path}",
                path=fp.path,
            )
        try:
            st = abs_path.stat()
        except OSError as e:
            return ToolResult.error("STAT_FAILED", str(e), path=fp.path)
        if (st.st_mtime_ns, st.st_size) != cached:
            return ToolResult.error(
                "CONFLICT",
                f"file changed since last read: {fp.path}",
                path=fp.path,
            )
        try:
            originals[abs_path] = abs_path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult.error("READ_FAILED", str(e), path=fp.path)

    # Phase 2: 在内存里构造所有新内容. 任一失败直接返回, 磁盘没动过.
    planned: dict[Path, str | None] = {}  # None 表示要删
    for fp, abs_path in resolved:
        if fp.op == "create":
            planned[abs_path] = fp.hunks[0].new_block
        elif fp.op == "delete":
            planned[abs_path] = None
        else:
            try:
                planned[abs_path] = _apply_hunks_to_text(originals[abs_path] or "", fp.hunks)
            except ValueError as e:
                return ToolResult.error(
                    "HUNK_FAILED",
                    f"{fp.path}: {e}",
                    path=fp.path,
                )

    # Phase 3: 落盘. 到这里任何 IO 错误都要把已写的全部回滚.
    written: list[Path] = []
    try:
        for _, abs_path in resolved:
            new_content = planned[abs_path]
            if new_content is None:
                abs_path.unlink()
            else:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(new_content, encoding="utf-8")
            written.append(abs_path)
    except OSError as e:
        # 回滚: 已写的恢复成 originals 里的内容
        for path in written:
            orig = originals[path]
            try:
                if orig is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_text(orig, encoding="utf-8")
            except OSError:
                pass  # 回滚失败就吞掉 — 主错误信息更重要
        return ToolResult.error("WRITE_FAILED", str(e))

    # 成功: 刷新 read_cache, 让后续 edit 不需要再 Read.
    summary_items = []
    for fp, abs_path in resolved:
        if fp.op == "delete":
            read_cache.forget(abs_path)
        else:
            read_cache.record(abs_path)
        summary_items.append(f"{fp.op} {fp.path}")

    return ToolResult.success(
        f"applied patch: {len(resolved)} file(s) — " + ", ".join(summary_items),
        files=[{"path": fp.path, "op": fp.op, "hunks": len(fp.hunks)} for fp, _ in resolved],
    )


# =================================================================
# Bash (降级为兜底)
# =================================================================

def _find_bash() -> str | None:
    """优先 Git for Windows 的 bash; 排除 WSL 的 System32\\bash.exe."""
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"E:\software\Git\usr\bin\bash.exe",
        r"E:\software\Git\bin\bash.exe",
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


class TerminalTool:
    """边角需求兜底: git, 跑脚本, 包管理. 不是主链路."""

    def __init__(self, workdir: Path, bash_exe: str | None = None, timeout: int = 30) -> None:
        self.workdir = workdir
        self.bash_exe = bash_exe if bash_exe is not None else _find_bash()
        self.timeout = timeout

    def run(self, command: str) -> ToolResult:
        try:
            if self.bash_exe:
                r = subprocess.run(
                    [self.bash_exe, "-lc", command],
                    cwd=str(self.workdir),
                    capture_output=True, text=True, timeout=self.timeout,
                    encoding="utf-8", errors="replace",
                )
            else:
                r = subprocess.run(
                    command, shell=True,
                    cwd=str(self.workdir),
                    capture_output=True, text=True, timeout=self.timeout,
                    encoding="gbk", errors="replace",
                )
        except subprocess.TimeoutExpired:
            return ToolResult.error("TIMEOUT", f"command timed out after {self.timeout}s")
        except Exception as e:
            return ToolResult.error("EXEC_FAILED", f"{type(e).__name__}: {e}")

        full = (r.stdout or "") + (r.stderr or "")
        truncated = len(full) > BASH_MAX_CHARS
        shown = full[:BASH_MAX_CHARS] if truncated else full
        if not shown:
            shown = "(no output)"

        data = {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        text = f"exit={r.returncode}\n{shown}"
        if truncated:
            return ToolResult.partial(
                text + f"\n... (output truncated, {len(full) - BASH_MAX_CHARS} more chars)",
                **data,
            )
        # 命令跑了但 exit!=0 也算 success — error 只留给"工具本身没跑成"的情况.
        # 模型能从 exit_code 看出命令是否成功.
        return ToolResult.success(text, **data)


# =================================================================
# Todo handler (TodoManager 类已在 Session 之前定义, 此处只是 tool handler)
# =================================================================

def tool_todo(items: list[dict], _session: Session) -> ToolResult:
    """更新 todo 列表. _session 由 dispatch 注入, 取出 session 上的 TodoManager."""
    return _session.todo.update(items)


# =================================================================
# spawn_agent (v6 新增) — 在隔离 git worktree 里跑子 agent
# =================================================================
#
# 设计选择 (跟用户讨论后定的):
#  - 隔离机制: git worktree (要求 main workspace 是 git repo)
#  - worktree 路径: tempfile.mkdtemp(prefix='minicode-sub-')
#  - 同步等 (主 agent 阻塞), 加 SUB_AGENT_TIMEOUT 兜底
#  - 子 agent 继承主 agent backend (主用 MiniMax, 子也用 MiniMax)
#  - 失败 worktree 保留供调试, 成功删除
#  - 深度限制 1: 子 agent 工具集不含 spawn_agent, 没法递归

# 主 agent 启动时设置 — 子 agent 用同一个 LLM 客户端 (复用连接, 共享 backend 选择)
_SPAWN_LLM: "LLMClient | None" = None

# 子 agent 跑完了, 这些信息回到主 agent: ToolResult 的 data 里包括 diff / worktree 路径
# 用于主 agent 后续决策 (apply diff / 调试时去看 worktree)


def _git_worktree_create(main_workdir: Path) -> Path:
    """在临时目录创建 main_workdir HEAD 的 worktree. 返回 worktree 路径.

    要求 main_workdir 是 git repo (有 .git 且至少一个 commit). 否则抛 RuntimeError
    并附带可操作的引导 (含 .gitignore 模板防止 .env 被 commit).

    创建成功后**不动 worktree 的 .gitignore** — 排除 build 垃圾的工作放在
    _git_worktree_diff 里用 pathspec exclude 做. 这样 spawn_agent 不留下任何
    "我们自己造的"修改 (之前往 .gitignore 追加导致 .gitignore 修改本身出现在
    diff 里, 用户看到一坨跟任务无关的 .gitignore 变更, 是个真烦的瑕疵).
    """
    if not (main_workdir / ".git").exists():
        # 升级为可操作错误 — 教用户怎么做, 而不是只说"missing .git"
        # 关键: 引导里的 git init 步骤必须先建 .gitignore 排掉 .env, 否则
        # 模型按错误消息照做时会把密钥 commit 进根 commit (上次实测的真坑).
        gitignore_template = "\n  ".join(_WORKTREE_EXCLUDE_PATTERNS)
        raise RuntimeError(
            f"spawn_agent requires {main_workdir} to be a git repository.\n"
            f"\n"
            f"To enable spawn_agent here, run THESE EXACT COMMANDS (the order matters — "
            f".gitignore MUST exist before `git add .` so secrets don't get committed):\n"
            f"\n"
            f"  cd {main_workdir}\n"
            f"  git init\n"
            f"  cat > .gitignore << 'EOF'\n"
            f"  {gitignore_template}\n"
            f"  EOF\n"
            f"  git add .\n"
            f"  git commit -m 'init for minicode'\n"
            f"\n"
            f"WARNING: do NOT skip the .gitignore step. Without it, `git add .` "
            f"will stage .env (with your API keys) and __pycache__/*.pyc into the "
            f"root commit. Once committed, secrets are hard to scrub."
        )
    sub_dir = Path(tempfile.mkdtemp(prefix="minicode-sub-"))
    # git worktree add <path> HEAD: 在 path checkout 当前 HEAD
    # --detach: 不创建分支, worktree 处于 detached HEAD (我们不需要分支)
    r = subprocess.run(
        ["git", "worktree", "add", "--detach", str(sub_dir), "HEAD"],
        cwd=str(main_workdir), capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        # worktree 创建失败 — 清理临时目录, 抛出原始 stderr 给上层
        try:
            sub_dir.rmdir()  # mkdtemp 创的空目录, 还能 rmdir
        except OSError:
            pass
        raise RuntimeError(f"git worktree add failed: {(r.stderr or '').strip()}")

    # 注意: 不写 .gitignore. 之前版本会追加 _WORKTREE_EXCLUDE_PATTERNS 到
    # worktree 的 .gitignore, 但那个"追加"本身是改动, 会出现在 diff 里 ——
    # 用户看到一段跟任务无关的 .gitignore 变更, 真烦. 现在改用 pathspec
    # exclude 在 _git_worktree_diff 里过滤, 不动磁盘上任何文件.
    return sub_dir


def _build_pathspec_excludes() -> list[str]:
    """把 _WORKTREE_EXCLUDE_PATTERNS 转成 git pathspec 的 exclude 形式.

    git pathspec exclude 语法: ':(exclude,glob)<pattern>'
    - exclude: 反向匹配
    - glob: 用 shell 风格的 glob (* / **) 匹配, 而不是 git 自家的 fnmatch

    注意 _WORKTREE_EXCLUDE_PATTERNS 里有的是目录 ('__pycache__/'), 有的是 glob
    ('*.pyc'). 都按 glob 处理, 目录用 '**/dirname/**' 形式让所有层级生效.
    """
    specs = []
    for p in _WORKTREE_EXCLUDE_PATTERNS:
        if p.endswith("/"):
            # 目录: 任意层级下 dirname/* 都排除
            dirname = p.rstrip("/")
            specs.append(f":(exclude,glob)**/{dirname}/**")
            specs.append(f":(exclude,glob){dirname}/**")  # 顶层
        else:
            # 文件 glob (e.g. *.pyc / .env)
            specs.append(f":(exclude,glob)**/{p}")
            specs.append(f":(exclude,glob){p}")  # 顶层
    return specs


def _git_worktree_diff(worktree: Path) -> str:
    """跑 git diff HEAD 拿到子 agent 的全部改动 (含 untracked 新建文件).

    要让新文件出现在 diff 里, 必须 git add -N (intent-to-add). 但 `git add -N .`
    会绕过 .gitignore — 把所有 untracked 都强制加进去. 用 ls-files 拿"非 ignored
    untracked", 再 add -N. 同时**两个步骤都加 pathspec exclude**, 显式排除
    build/cache/secrets — 不依赖磁盘上 .gitignore 文件 (避免我们造的 .gitignore
    修改本身污染 diff).
    """
    excludes = _build_pathspec_excludes()
    # 1) 列出 untracked + 不是 ignored + 不在我们的 exclude 列表里
    ls = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--"] + excludes,
        cwd=str(worktree), capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    if ls.returncode == 0:
        files_to_add = [f for f in (ls.stdout or "").splitlines() if f.strip()]
        if files_to_add:
            # 2) 逐个 git add -N (intent-to-add) — 显式列文件, 不会被 ignore 拦
            subprocess.run(
                ["git", "add", "-N", "--"] + files_to_add,
                cwd=str(worktree), capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
    # 3) git diff HEAD — 同样加 pathspec exclude, 防 tracked 的 .pyc 之类的被显示
    r = subprocess.run(
        ["git", "diff", "HEAD", "--"] + excludes,
        cwd=str(worktree), capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return ""  # 容错: 拿不到 diff 也别抛, 让 result text 报告任务结果即可
    return r.stdout or ""


def _git_worktree_remove(worktree: Path, main_workdir: Path) -> None:
    """正常清理: git worktree remove 然后 rmtree 临时目录."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=str(main_workdir), capture_output=True, text=True, timeout=30,
        )
    except Exception:
        pass  # 即便 worktree remove 失败, 后面 rmtree 兜底
    shutil.rmtree(worktree, ignore_errors=True)


def tool_spawn_agent(task: str, _session: Session) -> ToolResult:
    """派子 agent 在隔离 git worktree 里跑探索式任务.

    返回:
      ToolResult.success(text="<子 agent 摘要> + outcome 标记",
        data={
          "diff": "<unified diff string>",   # 主 agent 可 apply_patch 这个
          "worktree": "<path or null>",      # 失败时保留, 成功 None
          "sub_history_len": int,            # 子 agent 跑了几条消息
          "sub_tokens": (in, out),           # 子 agent 烧的 token
        })
    """
    # global 声明必须在函数开头, 即使后面才用. Python 语法要求.
    global WORKDIR

    if _SPAWN_LLM is None:
        return ToolResult.error(
            "SPAWN_NOT_AVAILABLE",
            "spawn_agent is not configured (no _SPAWN_LLM). "
            "This usually means you're calling spawn_agent from a context "
            "without a parent ReActAgent setup.",
        )

    main_workdir = WORKDIR

    # 1) 创建 worktree
    try:
        worktree = _git_worktree_create(main_workdir)
    except RuntimeError as e:
        return ToolResult.error("WORKTREE_FAILED", str(e))

    print(f"\n{C.META}┌─ 🚀 spawn_agent 启动 ─{'─' * 50}{C.RESET}")
    print(f"{C.META}│ task     : {_short(task, 200)}{C.RESET}")
    print(f"{C.META}│ worktree : {worktree}{C.RESET}")
    print(f"{C.META}└{'─' * 71}{C.RESET}\n")

    # 2) 给子 agent 准备 SYSTEM (含真实 worktree 路径)
    sub_system = SUB_AGENT_SYSTEM_TEMPLATE.format(workdir=worktree)

    # 3) 创建子 agent 的 Session / Registry / 工具集
    #    关键: 子 agent 的 WORKDIR 必须是 worktree, 不是主 WORKDIR
    #    简化处理: monkey-patch 模块级 WORKDIR, 跑完恢复. 不优雅但可工作.
    #    (彻底解决要把 WORKDIR 进 Session, 那是 v7 的事 — 见 README §10)
    WORKDIR = worktree
    sub_session = Session.new(sub_system, is_subagent=True)
    try:
        # 子 agent 的 terminal 也要在 worktree 里跑命令
        sub_terminal = TerminalTool(worktree)
        sub_registry = build_subagent_registry(sub_terminal, sub_session)
        sub_agent = ReActAgent(_SPAWN_LLM, sub_registry, system_prompt=sub_system)
        sub_agent.run(task, max_rounds=SUB_AGENT_MAX_ROUNDS)
    finally:
        WORKDIR = main_workdir  # 恢复, 一定要做

    # 4) 提取子 agent 的最终 visible 文本 + diff
    final_msg = next(
        (m for m in reversed(sub_session.history) if m.role == "assistant" and m.content),
        None,
    )
    summary = (final_msg.content or "").strip() if final_msg else "(子 agent 没产出可见文本)"
    # 剥 think 标签 (子 agent 也可能有 <think>)
    _, summary = split_think(summary) if "<think>" in summary else ("", summary)

    diff = _git_worktree_diff(worktree)
    diff_lines = len([ln for ln in diff.splitlines() if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))])

    # 5) 决定 worktree 是删还是留 — 看子 agent 的 outcome 关键字
    is_failure = any(kw in summary.upper() for kw in ("FAIL", "ERROR", "GAVE UP"))
    if is_failure:
        keep_path: str | None = str(worktree)
    else:
        _git_worktree_remove(worktree, main_workdir)
        keep_path = None

    # 6) 累计子 agent token 到主 session (成本可见)
    sub_tokens = (
        sub_session.prompt_tokens_total,
        sub_session.completion_tokens_total,
    )
    _session.prompt_tokens_total += sub_tokens[0]
    _session.completion_tokens_total += sub_tokens[1]

    # 7) 组合返回 — text 给主 agent 看 (摘要 + 关键统计 + 真 diff), data 给程序读
    # 关键: 真 diff 必须进 text. 之前只放 data 里, 模型看不到, 跟用户报告时只能编 diff.
    file_count = diff.count("+++") if diff else 0
    text_parts = [
        f"sub-agent finished. {summary}",
        f"diff: {diff_lines} line(s) changed across {file_count} file(s).",
    ]
    if diff:
        # 截断到 SUB_AGENT_DIFF_MAX_CHARS 防止巨 diff 撑爆 context
        diff_preview = diff[:SUB_AGENT_DIFF_MAX_CHARS]
        if len(diff) > SUB_AGENT_DIFF_MAX_CHARS:
            diff_preview += (
                f"\n... (diff truncated, {len(diff) - SUB_AGENT_DIFF_MAX_CHARS} "
                f"more chars in result.data.diff)"
            )
        text_parts.append(f"\nFull diff (use this verbatim if user wants to apply):\n{diff_preview}")
    if keep_path:
        text_parts.append(f"\nworktree retained for inspection: {keep_path}")
    if sub_tokens[0] + sub_tokens[1] > 0:
        text_parts.append(
            f"sub-agent tokens: {_fmt_k(sub_tokens[0])} in + {_fmt_k(sub_tokens[1])} out"
        )

    return ToolResult.success(
        "\n".join(text_parts),
        diff=diff,
        worktree=keep_path,
        sub_history_len=len(sub_session.history),
        sub_tokens=sub_tokens,
    )


def build_subagent_registry(terminal: "TerminalTool", session: "Session") -> "ToolRegistry":
    """子 agent 的工具集 — 主 agent 工具集去掉 spawn_agent 和 todo.

    去掉 spawn_agent: 防递归 (深度限制 1)
    去掉 todo: 子 agent 是单一目标, 不需要规划工具污染它的认知
    """
    reg = ToolRegistry(session=session)
    # 直接复用主工具的注册逻辑, 但跳过 spawn_agent / todo
    reg.register(Tool.from_function(
        tool_ls, name="LS",
        description="List entries in a directory (non-recursive).",
        param_docs={"path": "Directory path. Default '.'"},
    ))
    reg.register(Tool.from_function(
        tool_glob, name="Glob",
        description="Find files by name pattern (recursive).",
        param_docs={"pattern": "Glob pattern, e.g. '**/*.py'", "path": "Base path. Default '.'"},
    ))
    reg.register(Tool.from_function(
        tool_grep, name="Grep",
        description="Search file contents by regex.",
        param_docs={
            "pattern": "Regex pattern", "path": "File or directory. Default '.'",
            "ignore_case": "Case-insensitive. Default false",
            "file_pattern": "Only search files matching this glob (e.g. '*.py')",
        },
    ))
    reg.register(Tool.from_function(
        tool_read, name="Read",
        description="Read a text file with line numbers. Pass limit/offset for partial reads.",
        param_docs={"limit": "Max lines to return", "offset": "Skip this many lines from start"},
    ))
    reg.register(Tool.from_function(
        tool_write, name="write_file",
        description="Create or overwrite a file with the given content.",
    ))
    reg.register(Tool.from_function(
        tool_append, name="append_file",
        description="Append content to the end of a file (creates it if missing).",
    ))
    reg.register(Tool.from_function(
        tool_edit, name="edit_file",
        description="Replace a unique old_text with new_text inside a file.",
    ))
    reg.register(Tool.from_function(
        tool_apply_patch, name="apply_patch",
        description="Apply a unified-diff patch across files atomically.",
        param_docs={"patch": "Unified diff text."},
    ))
    reg.register(Tool.from_function(
        terminal.run, name="bash",
        description="Fallback shell. Use only when atomic tools don't cover the operation.",
    ))
    return reg


# =================================================================
# 工具注册
# =================================================================

_TODO_ITEMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
            },
        },
        "required": ["id", "text"],
    },
}


def build_default_registry(terminal: TerminalTool, session: Session | None = None) -> ToolRegistry:
    reg = ToolRegistry(session=session)

    # ---- 原子工具 (主链路) ----
    reg.register(Tool.from_function(
        tool_ls, name="LS",
        description="List entries in a directory (non-recursive). Use Glob for recursive.",
        param_docs={"path": "Directory path. Default '.'"},
    ))
    reg.register(Tool.from_function(
        tool_glob, name="Glob",
        description="Find files by name pattern (recursive). Pattern is glob syntax like '**/*.py'.",
        param_docs={
            "pattern": "Glob pattern, e.g. '**/*.py'",
            "path": "Base path. Default '.'",
        },
    ))
    reg.register(Tool.from_function(
        tool_grep, name="Grep",
        description="Search file contents by regex. Returns file:line:text matches.",
        param_docs={
            "pattern": "Regex pattern",
            "path": "File or directory. Default '.'",
            "ignore_case": "Case-insensitive. Default false",
            "file_pattern": "Only search files matching this glob (e.g. '*.py')",
        },
    ))
    reg.register(Tool.from_function(
        tool_read, name="Read",
        description="Read a text file with line numbers. Pass limit/offset for partial reads.",
        param_docs={
            "limit": "Max lines to return",
            "offset": "Skip this many lines from start",
        },
    ))
    reg.register(Tool.from_function(
        tool_write, name="write_file",
        description="Create or overwrite a file with the given content.",
    ))
    reg.register(Tool.from_function(
        tool_append, name="append_file",
        description=(
            "Append content to the end of a file (creates it if missing). "
            "Use this to write files too large for a single write_file call."
        ),
    ))
    reg.register(Tool.from_function(
        tool_edit, name="edit_file",
        description="Replace a unique old_text with new_text inside a file.",
    ))
    reg.register(Tool.from_function(
        tool_apply_patch, name="apply_patch",
        description=(
            "Apply a unified-diff patch across one or more files atomically. "
            "Supports modify (--- a/path +++ b/path), create (--- /dev/null +++ b/path), "
            "and delete (--- a/path +++ /dev/null). Hunk line numbers (@@ -N,M @@) are "
            "ignored — context lines locate the change. All target files must have been "
            "Read first (except brand-new files). If any hunk fails to apply, no file is "
            "modified. Prefer this over multiple edit_file calls when changes span files "
            "or a single file has multiple edits."
        ),
        # _session 由 dispatch 注入, 反射默认排除 _ 前缀参数, 模型看不到.
        param_docs={"patch": "Unified diff text. Paths may use 'a/' and 'b/' prefixes."},
    ))
    reg.register(Tool.from_function(
        tool_todo, name="todo",
        description=(
            "Write the full todo list. Pass every item each call. "
            "Statuses: pending, in_progress (at most one), completed."
        ),
        overrides={"items": _TODO_ITEMS_SCHEMA},
    ))

    # ---- spawn_agent (v6 新增, 主 agent 专属 — 子 agent 工具集没有这个防递归) ----
    reg.register(Tool.from_function(
        tool_spawn_agent, name="spawn_agent",
        description=(
            "*** USE THIS for ANY 'try / 试一下 / 试试 / 试着 / experiment / "
            "see if' request from the user. *** This is the ONLY safe way to "
            "apply experimental changes — it runs in an isolated git worktree, "
            "so the main workspace is untouched no matter what happens. "
            "Returns a summary + unified diff; you then ask the user whether "
            "to apply the diff via apply_patch.\n"
            "\n"
            "DO NOT call edit_file/write_file/apply_patch directly on the main "
            "workspace when the user's request includes 'try/试' wording — "
            "that defeats the purpose. The user wants to REVIEW before "
            "committing.\n"
            "\n"
            "Requires the main workspace to be a git repository. Don't use "
            "spawn_agent for committed (non-exploratory) tasks — those want "
            "direct edit_file/apply_patch + optional todo planning."
        ),
        param_docs={
            "task": "Bounded task for the sub-agent. State BOTH the change AND "
                    "the verification step (e.g. 'replace line 1 of foo.py with "
                    "X using edit_file, then run pytest tests/ and report PASS/"
                    "FAIL with the failing test names if any').",
        },
    ))

    # ---- bash (兜底, 不是主链路) ----
    reg.register(Tool.from_function(
        terminal.run, name="bash",
        description=(
            "Fallback shell. Prefer LS/Glob/Grep/Read for file operations. "
            "Use bash only for things atomic tools don't cover: git, "
            "running scripts, package managers, etc."
        ),
    ))
    return reg


# =================================================================
# 输出后处理 — 模型若用 <think> 包裹推理, 仅剥离不喂给后续轮
# =================================================================

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# 检测"模型用代码块伪装工具调用"的语言列表. 这 6 个 lang 的代码块在 agent 场景下
# 几乎只有"想绕过 tool_calls" 一种解释 — 用户不会问"教我写 diff/patch/json".
# 故意不含 python/js/go/c++/sql 等 — 那些在"展示代码示例"场景下太常见, 误伤率高.
_DODGED_TOOL_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell|diff|patch|json)\b",
    re.IGNORECASE,
)

_CODEBLOCK_REMINDER = (
    "<reminder>Your last reply contained a ```bash/```sh/```diff/```patch/```json "
    "code block, but no structured tool_call. The system does not execute code "
    "blocks — only tool_calls reach the disk. If you intended to RUN that "
    "content, call the corresponding tool now (`bash` for shell, `apply_patch` "
    "for diffs). If you only meant to SHOW example code to the user, ignore "
    "this reminder and reply normally without any of those code blocks.</reminder>"
)


def looks_like_dodged_tool_call(text: str) -> bool:
    """模型在 visible 里写了 ```bash/```diff 等代码块 — 大概率是想绕过 tool_calls."""
    if not text:
        return False
    return bool(_DODGED_TOOL_BLOCK_RE.search(text))


def split_think(text: str) -> tuple[str, str]:
    """返回 (think, visible). think 仅供日志; visible 是真正放回历史的."""
    if not text:
        return "", ""
    m = THINK_RE.search(text)
    think = m.group(1).strip() if m else ""
    visible = THINK_RE.sub("", text).strip()
    return think, visible


# =================================================================
# 模型加载 + 生成 (Ollama HTTP, OpenAI 兼容)
# =================================================================

class _OpenAICompatClient:
    """OpenAI-兼容 HTTP 客户端基类 — Ollama 和 MiniMax 共享这套.

    两者都暴露 `/v1/chat/completions`, 同样的 SSE 流式协议, 同样的 tool_calls 形态.
    唯一差异: base_url, 认证 header (有/无), 错误消息措辞.

    子类只需提供:
      - provider_name (用在错误消息里)
      - 构造时填 base_url, model, optional headers (auth)
      - 可选 hint(): 连接失败时给用户的引导 (Ollama 是"is daemon running",
        MiniMax 是"check API key & quota")
    """

    provider_name: str = "OpenAI-compat"

    def __init__(
        self, base_url: str, model: str, timeout: int = REQUEST_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def _hint_unreachable(self, reason: str) -> str:
        """子类可覆盖, 给用户更针对性的引导."""
        return f"Can't reach {self.provider_name} at {self.base_url} ({reason})"

    def _build_headers(self, accept_sse: bool = False) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if accept_sse:
            h["Accept"] = "text/event-stream"
        return h

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url, data=body, method="POST", headers=self._build_headers(),
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} HTTP {e.code}: {detail}") from e
        except URLError as e:
            raise RuntimeError(self._hint_unreachable(str(e.reason))) from e

    def chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = self._post_json("/v1/chat/completions", payload)
        try:
            return resp["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected {self.provider_name} response shape: {resp}") from e

    def chat_stream(self, messages: list[dict], tools: list[dict] | None):
        """流式版 chat — 按 SSE 协议逐行解析 OpenAI 兼容的增量 delta.

        yield 出每个 delta 的 message 片段 (dict): 上层 _StreamRenderer 负责拼回完整 message.
        典型 delta 形态:
          {"content": "hello "}            # 文本 token
          {"tool_calls": [{"index": 0, "id": "...", "function": {"name":"LS"}}]}
          {"tool_calls": [{"index": 0, "function": {"arguments": "{\\""}}]}
        Ollama 习惯把整个 tool_call 一次性塞进单 delta; MiniMax 应该会更细粒度
        (是真流式), 但 reassembler 都能吃 — 一片也是拼, N 片也是拼.

        usage: 流结束时 self.last_usage 会被设置 — 来自顶层 chunk["usage"] (OpenAI
        spec 规定在 stream_options.include_usage=True 时, 最后一个 chunk 顶层带
        usage). 后端不支持就是 None, 调用方走估算分支.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stream": True,
            # 让后端在最后一个 chunk 顶层带 usage. Ollama 不识别这个字段直接忽略,
            # MiniMax / OpenAI 会照办. 不是所有兼容实现都支持 — 没 usage 就估算.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url, data=body, method="POST", headers=self._build_headers(accept_sse=True),
        )
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} HTTP {e.code}: {detail}") from e
        except URLError as e:
            raise RuntimeError(self._hint_unreachable(str(e.reason))) from e

        self.last_usage: dict | None = None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # OpenAI spec: stream_options.include_usage=True 时, 最后一个 chunk
                # 的顶层会带 usage (此 chunk 的 choices 通常是空数组).
                if isinstance(chunk.get("usage"), dict):
                    self.last_usage = chunk["usage"]
                try:
                    delta = chunk["choices"][0].get("delta") or {}
                except (KeyError, IndexError):
                    continue
                if delta:
                    yield delta


class OllamaClient(_OpenAICompatClient):
    """Ollama 本地后端 — 无认证, 默认 localhost."""

    provider_name = "Ollama"

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL,
                 timeout: int = REQUEST_TIMEOUT) -> None:
        super().__init__(base_url=base_url, model=model, timeout=timeout)

    def _hint_unreachable(self, reason: str) -> str:
        return (f"Can't reach Ollama at {self.base_url}. "
                f"Is the Ollama app running? ({reason})")


class MinimaxClient(_OpenAICompatClient):
    """MiniMax 云端后端 — Bearer 认证, OpenAI-兼容端点."""

    provider_name = "MiniMax"

    def __init__(self, api_key: str, base_url: str = MINIMAX_BASE_URL,
                 model: str = MINIMAX_MODEL, timeout: int = REQUEST_TIMEOUT) -> None:
        if not api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is empty. Set it in .env (gitignored) or your shell env. "
                "Get a key at https://platform.minimaxi.com/"
            )
        super().__init__(
            base_url=base_url, model=model, timeout=timeout,
            extra_headers={"Authorization": f"Bearer {api_key}"},
        )

    def _hint_unreachable(self, reason: str) -> str:
        return (f"Can't reach MiniMax at {self.base_url}. "
                f"Check network, key validity, and quota. ({reason})")


# 类型别名: 上层只关心"有 chat/chat_stream 的客户端", 不关心是哪家
LLMClient = _OpenAICompatClient


def load_model() -> LLMClient:
    """根据 MINICODE_BACKEND 环境变量构造客户端."""
    if BACKEND == "minimax":
        client = MinimaxClient(api_key=MINIMAX_API_KEY)
    elif BACKEND == "ollama":
        client = OllamaClient()
    else:
        raise RuntimeError(
            f"Unknown MINICODE_BACKEND={BACKEND!r}. Use 'ollama' or 'minimax'."
        )
    print(
        f"Using {client.provider_name} at {client.base_url}, "
        f"model={client.model} (timeout={client.timeout}s)",
        file=sys.stderr,
    )
    return client


# =================================================================
# 流式渲染: 把 SSE delta 流拼成完整 message + 实时打印
# =================================================================

class _StreamRenderer:
    """流式渲染器: 累积 content / tool_calls 同时往终端 flush 文本.

    渲染策略 (与 Claude Code 行为对齐):
    - content 文本累积到行边界 (\\n) 或 STREAM_FLUSH_CHARS 字符上限再刷, 避免逐 token
      闪屏 (Windows 终端 + 中文 + ANSI 颜色组合下尤其卡).
    - tool_calls 按 OpenAI 标准 reassembly (按 index 累加 arguments 分片), 完整后留给
      上层一次性 _box 渲染. 不做 "分片预览" — 半截 JSON 显示意义不大.
    - <think>...</think> 在流式期不做特殊染色 (跨 chunk 边界处理太脆), 等 finalize
      后由 split_think 一次性剥离, 跟非流式语义一致.
    - Ctrl-C: 让外层 (step) 捕获并把已收到的 partial 写进历史, 渲染器只负责 flush.
    """

    def __init__(self) -> None:
        self.content_parts: list[str] = []        # 完整 content (供历史)
        self.tool_calls: list[dict] = []          # 累积的 tool_calls (按 index)
        self._buffer = ""                         # 待 flush 的文本

    def feed(self, delta: dict) -> None:
        """喂一个 delta. 文本按规则 flush; tool_calls 累积."""
        text = delta.get("content")
        if text:
            self.content_parts.append(text)
            self._buffer += text
            self._flush_lines()

        tcs = delta.get("tool_calls")
        if tcs:
            for tc in tcs:
                self._merge_tool_call(tc)

    def _merge_tool_call(self, tc: dict) -> None:
        """OpenAI 流式 tool_calls: 按 index 合并 — name 一次给, arguments 分片拼."""
        idx = tc.get("index", 0)
        while len(self.tool_calls) <= idx:
            self.tool_calls.append({"id": None, "function": {"name": None, "arguments": ""}})
        slot = self.tool_calls[idx]
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] = fn["name"]
        if "arguments" in fn:
            slot["function"]["arguments"] += fn["arguments"] or ""

    def _flush_lines(self) -> None:
        """从 buffer 里把"完整行"或"超长片段"刷出去, 余下留给下次."""
        while True:
            nl = self._buffer.find("\n")
            if nl >= 0:
                chunk, self._buffer = self._buffer[: nl + 1], self._buffer[nl + 1:]
                self._write(chunk)
                continue
            if len(self._buffer) >= STREAM_FLUSH_CHARS:
                self._write(self._buffer)
                self._buffer = ""
                continue
            break

    def _write(self, s: str) -> None:
        sys.stdout.write(C.ASSIST + s + C.RESET)
        sys.stdout.flush()

    def finalize(self) -> dict:
        """流结束: 把残留 buffer 全 flush, 返回完整 message dict (与非流式 chat 同形)."""
        if self._buffer:
            self._write(self._buffer)
            self._buffer = ""
        sys.stdout.write("\n")
        sys.stdout.flush()

        full_content = "".join(self.content_parts)
        msg: dict[str, Any] = {"role": "assistant", "content": full_content}
        valid_tool_calls = [
            tc for tc in self.tool_calls
            if tc.get("function", {}).get("name")
        ]
        if valid_tool_calls:
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": tc["function"]}
                for tc in valid_tool_calls
            ]
        return msg


def _history_to_openai(history: list[Message]) -> list[dict]:
    """把 Message 列表转成 Ollama / OpenAI 兼容的消息格式."""
    out: list[dict] = []
    for m in history:
        d: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
            if not d["content"]:
                d["content"] = None
        if m.role == "tool" and m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def generate(client: LLMClient, history: list[Message],
             tool_schemas: list[dict]) -> dict:
    """调用 LLM (Ollama / MiniMax) 的 chat completion, 返回 assistant message dict.

    返回形如:
      {"role": "assistant", "content": "...",
       "tool_calls": [{"id": "...", "type": "function",
                       "function": {"name": "...", "arguments": "json-str"}}, ...]}
    """
    return client.chat(_history_to_openai(history), tool_schemas or None)


# =================================================================
# 日志辅助
# =================================================================

_USE_COLOR = os.environ.get("NO_COLOR") is None and sys.stdout.isatty()
if sys.platform == "win32":
    # Windows 控制台默认 GBK, box 用的 ─ │ └ ▶ 这些字符会 UnicodeEncodeError.
    # 强制 UTF-8 输出 — Win10+ 的现代终端都支持.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if _USE_COLOR:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


class C:
    RESET   = "\033[0m"   if _USE_COLOR else ""
    DIM     = "\033[2m"   if _USE_COLOR else ""
    BOLD    = "\033[1m"   if _USE_COLOR else ""
    USER    = "\033[1;36m" if _USE_COLOR else ""
    THOUGHT = "\033[2;37m" if _USE_COLOR else ""
    ASSIST  = "\033[0;37m" if _USE_COLOR else ""
    ACTION  = "\033[1;33m" if _USE_COLOR else ""
    OK      = "\033[0;32m" if _USE_COLOR else ""
    PARTIAL = "\033[0;33m" if _USE_COLOR else ""
    ERR     = "\033[0;31m" if _USE_COLOR else ""
    WARN    = "\033[1;31m" if _USE_COLOR else ""
    DONE    = "\033[1;32m" if _USE_COLOR else ""
    META    = "\033[2;36m" if _USE_COLOR else ""
    DATA    = "\033[2;35m" if _USE_COLOR else ""  # 暗紫 — data 面板


def _box(title: str, body: str = "", char: str = "─", width: int = 72,
         color: str = "") -> None:
    print()
    border = char * max(0, width - len(title) - 4)
    print(f"{color}┌─ {title} {border}{C.RESET}")
    for line in (body or "").splitlines() or [""]:
        print(f"{color}│{C.RESET} {line}")
    print(f"{color}└{char * (width - 1)}{C.RESET}")


def _short(s: str, n: int = 400) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f" …(+{len(s) - n} chars)"


def _fmt_args(args: dict) -> str:
    out = []
    for k, v in args.items():
        vs = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        out.append(f"  {k} = {_short(vs, 200)}")
    return "\n".join(out) if out else "  (no arguments)"


def _fmt_data(data: dict) -> str:
    """data 面板. 小字典直接展开, 大数组只显示形状."""
    if not data:
        return "  (empty)"
    out = []
    for k, v in data.items():
        if isinstance(v, list):
            out.append(f"  {k}: list[{len(v)}]")
        elif isinstance(v, dict):
            out.append(f"  {k}: dict[{len(v)} keys]")
        elif isinstance(v, str) and len(v) > 200:
            out.append(f"  {k}: str[{len(v)} chars]")
        else:
            out.append(f"  {k}: {json.dumps(v, ensure_ascii=False, default=str)}")
    return "\n".join(out)


def _fmt_k(n: int) -> str:
    """格式化 token 数: 1234 -> '1.2K'; <1000 显示原值."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}K"


def _estimate_cost(prompt_t: int, comp_t: int) -> float:
    """按 MiniMax-M2.7 价格估算成本 (¥). 仅 minimax backend 调用."""
    return (prompt_t * MINIMAX_PRICE_INPUT_PER_M
            + comp_t * MINIMAX_PRICE_OUTPUT_PER_M) / 1_000_000


def _fmt_tokens_turn(usage: tuple[int, int, bool]) -> str:
    """每 turn 末尾的一行 token 报告."""
    p, c, exact = usage
    suffix = "" if exact else " (estimated)"
    line = f"tokens: {_fmt_k(p)} in + {_fmt_k(c)} out = {_fmt_k(p + c)}{suffix}"
    if BACKEND == "minimax":
        line += f"  ≈ ¥{_estimate_cost(p, c):.4f}"
    return line


def _fmt_tokens_session(sess: "Session") -> str:
    """run() 收尾的总计行."""
    p, c = sess.prompt_tokens_total, sess.completion_tokens_total
    accuracy = ""
    if sess.estimated_turns and not sess.exact_turns:
        accuracy = " (all estimated)"
    elif sess.estimated_turns:
        accuracy = f" ({sess.exact_turns} exact + {sess.estimated_turns} estimated)"
    line = f"session total: {_fmt_k(p)} in + {_fmt_k(c)} out = {_fmt_k(p + c)}{accuracy}"
    if BACKEND == "minimax":
        line += f"  ≈ ¥{_estimate_cost(p, c):.4f}"
    return line


# =================================================================
# ReActAgent — Thought → Action → Observation 循环
# =================================================================

@dataclass
class StepResult:
    thought: str
    visible: str
    actions: list[dict]
    observations: list[ToolResult]
    truncated: bool
    used_todo: bool


def _parse_tool_arguments(raw: Any) -> dict:
    """llama-cpp 的 tool_call.function.arguments 通常是 JSON 字符串, 也可能已是 dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


class ReActAgent:
    NAG_THRESHOLD = 3
    # 系统层兜底: 模型连续这么多轮"用代码块伪装工具调用", 就硬退出 — 防止
    # SYSTEM 禁令被绕过后陷入 reminder 死循环.
    CODEBLOCK_NAG_LIMIT = 2

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        system_prompt: str = SYSTEM,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        # registry 自带 session — agent 不另起一个, 防止 dispatch 注入的 session
        # 跟 agent.history 不同步 (单一权威源). agent 把自己的 system_prompt
        # 写进 session.history[0], 替换掉 registry 默认 session 里的占位 system.
        registry.session.history = [Message.system(system_prompt)]

    @property
    def session(self) -> Session:
        return self.registry.session

    @property
    def history(self) -> list[Message]:
        return self.session.history

    def new_session(self) -> None:
        """重新构造 session, 一步清空所有可变状态 (history / todo / read_cache)."""
        self.registry.session = Session.new(self.system_prompt)

    def _call_llm(self) -> tuple[dict, bool, tuple[int, int, bool]]:
        """调用模型, 返回 (assistant_message, interrupted, (prompt_t, comp_t, exact)).

        USE_STREAM=True 时走 chat_stream + 实时打印; 否则走老的 chat 一次性返回.
        Ctrl-C 期间已收到的 partial 也会作为 message 返回, 不丢.

        usage 三元组 (prompt_tokens, completion_tokens, is_exact):
          - 后端给了 usage 字段 → exact=True
          - 没给 (Ollama 流式 / 部分兼容实现) → 按字符数估算, exact=False
        """
        messages = _history_to_openai(self.history)
        tools = self.registry.schemas() or None

        # 预估 prompt token (用于估算 fallback) — 拼所有 message content 的字符数
        prompt_chars = sum(len(m.get("content") or "") for m in messages)

        if not USE_STREAM:
            full = self.llm.chat(messages, tools)  # 这里 chat 仍只返回 message dict
            # 非流式时 usage 在 self.llm._post_json 的 resp 里 — 但 chat() 抛掉了
            # 顶层 wrapper. 简化处理: 非流式也走估算 (REPL 默认流式, 这条分支冷)
            content = full.get("content") or ""
            comp_chars = len(content)
            usage = (
                max(1, prompt_chars // CHARS_PER_TOKEN_EST),
                max(1, comp_chars // CHARS_PER_TOKEN_EST),
                False,
            )
            return full, False, usage

        renderer = _StreamRenderer()
        interrupted = False
        try:
            for delta in self.llm.chat_stream(messages, tools):
                renderer.feed(delta)
        except KeyboardInterrupt:
            interrupted = True
        msg = renderer.finalize()
        if interrupted:
            print(f"{C.WARN}(interrupted by user — partial output kept){C.RESET}")

        # 取 usage: chat_stream 把 last_usage 留在 client 上
        last_usage = getattr(self.llm, "last_usage", None)
        if isinstance(last_usage, dict) and "prompt_tokens" in last_usage:
            usage = (
                int(last_usage.get("prompt_tokens", 0)),
                int(last_usage.get("completion_tokens", 0)),
                True,
            )
        else:
            # 后端没给 usage — 按字符数估算
            content = msg.get("content") or ""
            tcs_args = sum(
                len((tc.get("function") or {}).get("arguments") or "")
                for tc in (msg.get("tool_calls") or [])
            )
            comp_chars = len(content) + tcs_args
            usage = (
                max(1, prompt_chars // CHARS_PER_TOKEN_EST),
                max(1, comp_chars // CHARS_PER_TOKEN_EST),
                False,
            )
        return msg, interrupted, usage

    def step(self) -> StepResult:
        msg, interrupted, usage = self._call_llm()
        # 累加到 session, 给 run() 收尾用
        prompt_t, comp_t, exact = usage
        sess = self.session
        sess.prompt_tokens_total += prompt_t
        sess.completion_tokens_total += comp_t
        if exact:
            sess.exact_turns += 1
        else:
            sess.estimated_turns += 1
        # 把 usage 挂在 step result 上, 让 run() 显示本 turn 数据
        self._last_usage = usage  # 简单临时通信, 不进 StepResult dataclass
        raw_content = msg.get("content") or ""
        thought, visible = split_think(raw_content)
        if interrupted:
            # 给历史里的可见文本打个标记, 让模型下次知道上一轮被打断了.
            visible = (visible + "\n[interrupted by user]").strip()

        raw_tool_calls = msg.get("tool_calls") or []
        actions: list[dict] = []
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            actions.append({
                "id": tc.get("id"),
                "name": name,
                "arguments": _parse_tool_arguments(fn.get("arguments")),
            })

        # 被中断时不执行已经收到一半的 tool_calls — 安全第一, 不知道 arguments 是否完整.
        if interrupted:
            actions = []

        # 把 assistant 消息原样放回历史 — 保留 llama-cpp 给的 id, 后续 tool 消息配对.
        assistant_tool_calls: list[dict] | None = None
        if actions:
            assistant_tool_calls = []
            for a in actions:
                entry = {
                    "type": "function",
                    "function": {
                        "name": a["name"],
                        "arguments": json.dumps(a["arguments"], ensure_ascii=False),
                    },
                }
                if a["id"] is not None:
                    entry["id"] = a["id"]
                assistant_tool_calls.append(entry)

        self.history.append(Message.assistant(visible, tool_calls=assistant_tool_calls))

        observations: list[ToolResult] = []
        used_todo = False
        for a in actions:
            result = self.registry.dispatch(a["name"], a["arguments"])
            observations.append(result)
            # 关键: 只把 text 喂给模型, data 留给日志.
            self.history.append(Message.tool(result.text, tool_call_id=a["id"]))
            if a["name"] == "todo":
                used_todo = True

        return StepResult(
            thought=thought, visible=visible,
            actions=actions, observations=observations,
            truncated=False, used_todo=used_todo,
        )

    def run(self, query: str, max_rounds: int = MAX_ROUNDS) -> None:
        # 每次 run 都把 codeblock 嘴硬计数归零 — 上一个 task 的累计不应惩罚下一个.
        # (rounds_since_todo 同理由 turn 内逻辑自然管理, 这里不动)
        self.session.codeblock_nag_count = 0
        self.history.append(Message.user(query))
        _box("▶ USER", query, char="═", color=C.USER)

        for turn in range(1, max_rounds + 1):
            print(f"\n{C.META}══════════ Turn {turn} ══════════{C.RESET}")
            print(f"{C.META}→ [1] 调用模型 {'(streaming)' if USE_STREAM else '...'}{C.RESET}")
            res = self.step()

            # 流式: content 已经实时打印过 (含 <think>), 不再用 _box 重复展示.
            # 非流式: 一次性返回, 这时 _box 是唯一展示路径.
            if not USE_STREAM:
                if res.thought:
                    _box("Thought (模型内部推理)", _short(res.thought, 1200), color=C.THOUGHT)
                if res.visible:
                    _box("Assistant (可见输出)", res.visible, color=C.ASSIST)
            for i, a in enumerate(res.actions, 1):
                _box(f"Action {i}/{len(res.actions)}: {a['name']}",
                     _fmt_args(a["arguments"]), color=C.ACTION)
            for i, obs in enumerate(res.observations, 1):
                color = {"success": C.OK, "partial": C.PARTIAL, "error": C.ERR}[obs.status]
                _box(f"Observation {i}/{len(res.observations)} ({obs.status}) → text",
                     _short(obs.text, 800), color=color)
                if obs.data:
                    _box(f"Observation {i} → data (日志面板, 不喂模型)",
                         _fmt_data(obs.data), color=C.DATA)

            # 本 turn token 报告 (放在 actions/observations 之后, 紧贴 turn 收尾)
            print(f"{C.META}→ [2] {_fmt_tokens_turn(self._last_usage)}{C.RESET}")

            sess = self.session
            if not res.actions:
                # 系统层兜底: 无 tool_call + visible 里有 ```bash/```diff 等代码块,
                # 八成是模型用文本伪装工具调用 (SYSTEM 已禁但 7B 偶尔嘴硬).
                # 注入提醒, 给一次重做的机会; 累计 CODEBLOCK_NAG_LIMIT 次硬退出.
                #
                # 关键: counter 在本次 run() 内只增不减. 之前的版本"调任何 tool 都
                # 重置", 结果模型用 dodge ↔ todo 摆烂的振荡把计数永远拖在 1, 跑满
                # 20 轮上限. 现在 counter 累积 — 振荡也会触顶, GAVE UP 必生效.
                if looks_like_dodged_tool_call(res.visible):
                    sess.codeblock_nag_count += 1
                    if sess.codeblock_nag_count > self.CODEBLOCK_NAG_LIMIT:
                        _box(
                            f"⛔ GAVE UP (本会话累计 {sess.codeblock_nag_count} 次"
                            f"用代码块代替 tool_call, 硬退出)",
                            _fmt_tokens_session(sess), char="═", color=C.WARN,
                        )
                        return
                    _box(
                        f"⚠️  CODEBLOCK NAG ({sess.codeblock_nag_count}"
                        f"/{self.CODEBLOCK_NAG_LIMIT})",
                        "检测到 ```bash/```diff 等代码块但无 tool_call, 注入提醒",
                        color=C.WARN,
                    )
                    self.history.append(Message.user(_CODEBLOCK_REMINDER))
                    continue  # 跳过 DONE, 进下一轮重试
                _box("✅ DONE (无工具调用, 本轮结束)",
                     _fmt_tokens_session(sess), char="═", color=C.DONE)
                return

            # 有 tool_call 走正常路径. 注意: 这里不再清 codeblock_nag_count —
            # 一次合法 tool_call 不能"洗白"前面的嘴硬, 否则模型用 todo 摆烂就能绕过.
            sess.rounds_since_todo = 0 if res.used_todo else sess.rounds_since_todo + 1
            print(f"{C.META}→ [3] rounds_since_todo = {sess.rounds_since_todo}"
                  f"{'  (本轮调用了 todo, 已重置)' if res.used_todo else ''}{C.RESET}")
            # 子 agent 没 todo 工具 — todo NAG 对它毫无意义, 跳过.
            if sess.rounds_since_todo >= self.NAG_THRESHOLD and not sess.is_subagent:
                _box("⚠️  NAG", f"连续 {self.NAG_THRESHOLD} 轮未调用 todo, 注入提醒",
                     color=C.WARN)
                self.history.append(Message.user(
                    "<reminder>Update your todos.</reminder>"
                ))
                sess.rounds_since_todo = 0

        _box("⛔ MAX ROUNDS", f"达到 {max_rounds} 轮上限, 强制退出本轮\n"
             f"{_fmt_tokens_session(self.session)}",
             char="═", color=C.WARN)


# =================================================================
# REPL 外壳
# =================================================================

BANNER = r"""
 ███╗   ███╗██╗███╗   ██╗██╗ ██████╗ ██████╗ ██████╗ ███████╗
 ████╗ ████║██║████╗  ██║██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ██╔████╔██║██║██╔██╗ ██║██║██║     ██║   ██║██║  ██║█████╗
 ██║╚██╔╝██║██║██║╚██╗██║██║██║     ██║   ██║██║  ██║██╔══╝
 ██║ ╚═╝ ██║██║██║ ╚████║██║╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
              MiniCode Agent v4 — llama.cpp + atomic tools
"""

HELP = """\
可用斜杠命令:
  /help      显示本帮助
  /exit      退出 (Ctrl-D / Ctrl-C 同效)
  /clear     清空对话历史 (保留 system, todo 状态一并重置)
  /todos     显示当前 todo 列表
  /history   显示消息条数统计
其他任何非 / 开头的输入都会作为 user 消息送入 agent。
"""


def _print_banner(agent: ReActAgent, terminal: TerminalTool) -> None:
    print(f"{C.BOLD}{C.USER}{BANNER}{C.RESET}")
    print(f"  {C.DIM}backend:{C.RESET} {agent.llm.provider_name} @ {agent.llm.base_url}")
    print(f"  {C.DIM}model  :{C.RESET} {agent.llm.model}")
    print(f"  {C.DIM}cwd    :{C.RESET} {WORKDIR}")
    print(f"  {C.DIM}shell  :{C.RESET} {terminal.bash_exe or '(cmd.exe 回退)'}")
    print(f"  {C.DIM}grep   :{C.RESET} {_RG_PATH or '(纯 Python 兜底, 装 ripgrep 更快)'}")
    print(f"  {C.DIM}tools  :{C.RESET} {C.ACTION}{', '.join(agent.registry.names())}{C.RESET}")
    print(f"  {C.DIM}输入 /help 查看命令, /exit 退出{C.RESET}")
    print(f"{C.DIM}{'─' * 72}{C.RESET}")


def repl() -> None:
    llm = load_model()
    # 子 agent 共享主 agent 的 llm 实例 (复用连接, 共享 backend 选择)
    global _SPAWN_LLM
    _SPAWN_LLM = llm
    terminal = TerminalTool(WORKDIR)
    registry = build_default_registry(terminal)
    agent = ReActAgent(llm, registry)

    _print_banner(agent, terminal)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if not line:
            continue

        if line.startswith("/"):
            cmd = line[1:].split(maxsplit=1)[0].lower()
            if cmd in ("exit", "quit", "q"):
                print("bye.")
                return
            if cmd == "help":
                print(HELP)
                continue
            if cmd == "clear":
                agent.new_session()
                print("(history cleared)")
                continue
            if cmd == "todos":
                print(agent.session.todo.render())
                continue
            if cmd == "history":
                counts: dict[str, int] = {}
                for m in agent.history:
                    counts[m.role] = counts.get(m.role, 0) + 1
                print(f"messages: {len(agent.history)}  {counts}")
                continue
            print(f"unknown command: /{cmd} (try /help)")
            continue

        try:
            agent.run(line)
        except KeyboardInterrupt:
            print("\n(turn interrupted)")


# =================================================================
# 入口
# =================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        llm = load_model()
        _SPAWN_LLM = llm  # 一次性模式也支持 spawn_agent
        terminal = TerminalTool(WORKDIR)
        registry = build_default_registry(terminal)
        agent = ReActAgent(llm, registry)
        agent.run(q)
    else:
        repl()
