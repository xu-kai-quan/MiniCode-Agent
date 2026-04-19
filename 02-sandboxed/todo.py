"""s03 过程 2: TodoWrite MVP with local Qwen3.5-2B (基于 v1 的工作副本, 用于后续修改).

Agent loop + tool dispatch + todo planning + nag reminder, running against
the local Qwen model at E:/MYSELF/model/qwen/Qwen3.5-2B.

Qwen's chat template emits tool calls as XML:
  <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
and expects tool outputs back as messages with role="tool".
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("E:/MYSELF/model/qwen/Qwen3.5-2B")
WORKDIR = Path.cwd().resolve()
SYSTEM = (
    "You are a coding agent. For any task with more than one step, call the "
    "`todo` tool first to plan, then keep it updated as you work. Exactly one "
    "item should be `in_progress` at a time. "
    "For files longer than ~200 lines, write in chunks: start with "
    "`write_file` for the first chunk, then call `append_file` repeatedly for "
    "each remaining chunk (roughly 150 lines per chunk), preserving newlines. "
    "Reply with a final message only when no more tool calls are needed."
)
MAX_ROUNDS = 20
MAX_NEW_TOKENS = 4096


# ---------- tool handlers ----------

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _find_bash() -> str | None:
    """探测 Git for Windows 的 bash.exe. 排除 WSL 的 System32\\bash.exe — 它输出 UTF-16
    会和 Python 默认解码冲突, 而且工作区路径语义也不同 (WSL 里 E:\\ → /mnt/e/)."""
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
    # 也允许 PATH 上非 System32 的 bash (例如用户自己装的 MSYS2)
    import shutil
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


BASH_EXE = _find_bash()


def run_bash(command: str) -> str:
    try:
        if BASH_EXE:
            r = subprocess.run(
                [BASH_EXE, "-lc", command],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
        else:
            # 找不到 bash, 退回系统 shell (Windows 下是 cmd.exe, GBK 编码)
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30,
                encoding="gbk", errors="replace",
            )
        return ((r.stdout or "") + (r.stderr or ""))[:8000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "error: command timed out after 30s"
    except Exception as e:
        return f"error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:8000]


def run_write(path: str, content: str) -> str:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def run_append(path: str, content: str) -> str:
    """追加文本到文件末尾, 不存在则创建. 用于分片写入超出单轮 token 预算的大文件."""
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    total = p.stat().st_size
    return f"appended {len(content)} chars to {path} (now {total} bytes)"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    p = safe_path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(old_text) != 1:
        return f"error: old_text must match exactly once (found {text.count(old_text)})"
    p.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"edited {path}"


# ---------- todo manager ----------

class TodoManager:
    def __init__(self):
        self.items: list[dict] = []

    def update(self, items: list[dict]) -> str:
        in_progress = sum(1 for it in items if it.get("status") == "in_progress")
        if in_progress > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = [
            {"id": it["id"], "text": it["text"], "status": it.get("status", "pending")}
            for it in items
        ]
        return self.render()

    def render(self) -> str:
        mark = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        if not self.items:
            return "(no todos)"
        return "\n".join(f"{mark[it['status']]} {it['text']}" for it in self.items)


TODO = TodoManager()


def _coerce(value: str):
    """Qwen emits all XML parameter values as strings; coerce JSON-ish ones."""
    import json
    s = value.strip()
    if (s and s[0] in "[{") or s in ("true", "false", "null"):
        try:
            return json.loads(s)
        except Exception:
            return value
    if s.isdigit():
        return int(s)
    return value


TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "append_file": lambda **kw: run_append(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":        lambda **kw: TODO.update(kw["items"]),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": (
                "Append content to the end of a file (creates it if missing). "
                "Use this to write files that are too large for a single write_file "
                "call — start with write_file for the first chunk, then call "
                "append_file repeatedly for the remaining chunks, preserving newlines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a unique old_text with new_text inside a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": (
                "Write the full todo list. Pass every item each call. "
                "Statuses: pending, in_progress (at most one), completed. "
                "`items` must be a JSON array of {id, text, status}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
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
                },
                "required": ["items"],
            },
        },
    },
]


# ---------- Qwen tool-call parsing ----------

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for tc in TOOL_CALL_RE.findall(text):
        fm = FUNCTION_RE.search(tc)
        if not fm:
            continue
        name = fm.group(1).strip()
        args = {k: _coerce(v) for k, v in PARAM_RE.findall(fm.group(2))}
        calls.append({"name": name, "arguments": args})
    return calls


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


# ---------- pretty logging ----------

def _box(title: str, body: str = "", char: str = "─", width: int = 72) -> None:
    print()
    print(f"┌─ {title} " + char * max(0, width - len(title) - 4))
    for line in (body or "").splitlines() or [""]:
        print(f"│ {line}")
    print("└" + char * (width - 1))


def _short(s: str, n: int = 400) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f" …(+{len(s) - n} chars)"


def _fmt_args(args: dict) -> str:
    import json
    out = []
    for k, v in args.items():
        vs = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        out.append(f"  {k} = {_short(vs, 200)}")
    return "\n".join(out) if out else "  (no arguments)"


# ---------- agent loop ----------

def load_model():
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    print(f"Loading Qwen from {MODEL_DIR} on {device}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, dtype=dtype
    )
    mdl.to(device).eval()
    return tok, mdl, device


def _stop_token_ids(tok) -> list[int]:
    """Qwen 在一轮 assistant 结束时写 <|im_end|>; 把它作为额外的停止 token,
    防止模型脑补下一轮的 `user`/`<tool_response>` 继续写满 max_new_tokens."""
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tok.convert_tokens_to_ids(marker)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return list(ids)


def generate(tok, mdl, device, messages: list[dict]) -> str:
    prompt = tok.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=_stop_token_ids(tok),
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def run_turn(state: dict, query: str) -> None:
    """把一条用户输入跑到 assistant 不再调用工具为止, 共享 state['messages']."""
    tok, mdl, device = state["tok"], state["mdl"], state["device"]
    messages = state["messages"]
    messages.append({"role": "user", "content": query})

    _box("▶ USER", query, char="═")

    for turn in range(1, MAX_ROUNDS + 1):
        print(f"\n══════════ Turn {turn} ══════════")

        # 1) 模型生成
        print("→ [1] 调用模型 …")
        raw = generate(tok, mdl, device, messages)
        think = extract_think(raw)
        visible = strip_think(raw)
        if think:
            _box("🧠 THINK (模型内部推理)", _short(think, 1200))
        if visible:
            _box("💬 ASSISTANT (可见输出)", visible)

        # 2) 解析工具调用
        calls = parse_tool_calls(visible)
        messages.append({
            "role": "assistant",
            "content": visible,
            "tool_calls": [
                {"type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ] or None,
        })

        # 2b) 截断检测: 有 <tool_call> 开头但 parse_tool_calls 拿不到 = 被 MAX_NEW_TOKENS 截掉了
        truncated = visible.count("<tool_call>") > len(calls)
        if not calls:
            if truncated:
                _box("⚠️  TRUNCATED (工具调用被截断, 让模型重来并改用分片)",
                     "输出里有 <tool_call> 但没闭合, 说明撞到了 MAX_NEW_TOKENS 上限。\n"
                     "向模型注入提醒, 让它用 write_file + append_file 分片重写。")
                messages.append({
                    "role": "user",
                    "content": (
                        "<reminder>Your previous tool call was cut off before "
                        "</tool_call>. Do not repeat the same oversized write_file. "
                        "Instead: call write_file with only the FIRST ~150 lines, "
                        "then call append_file repeatedly for the remaining chunks.</reminder>"
                    ),
                })
                continue
            _box("✅ DONE (无工具调用, 本轮结束)", "", char="═")
            return

        print(f"→ [2] 解析到 {len(calls)} 个 tool_call")

        # 3) 依次执行
        used_todo = False
        for i, c in enumerate(calls, 1):
            name = c["name"]
            _box(f"🔧 TOOL CALL {i}/{len(calls)}: {name}", _fmt_args(c["arguments"]))
            if name == "todo":
                used_todo = True
            handler = TOOL_HANDLERS.get(name)
            try:
                output = handler(**c["arguments"]) if handler else f"Unknown tool: {name}"
                status = "ok"
            except Exception as e:
                output = f"error: {e}"
                status = "error"
            _box(f"📤 RESULT {i}/{len(calls)} ({status})", _short(str(output), 800))
            messages.append({"role": "tool", "content": str(output)})

        # 4) todo 计数器 + nag
        state["rounds_since_todo"] = 0 if used_todo else state["rounds_since_todo"] + 1
        rst = state["rounds_since_todo"]
        print(f"→ [3] rounds_since_todo = {rst}"
              f"{'  (已重置, 本轮调用了 todo)' if used_todo else ''}")
        if rst >= 3:
            _box("⚠️  NAG", "已连续 3 轮未调用 todo, 注入 <reminder>Update your todos.</reminder>")
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos.</reminder>",
            })
            state["rounds_since_todo"] = 0

    _box("⛔ MAX ROUNDS", f"达到 {MAX_ROUNDS} 轮上限, 强制退出本轮", char="═")


# ---------- REPL ----------

BANNER = r"""
 ██████╗  ██╗    ██╗███████╗███╗   ██╗       ██╗      ██████╗  ██████╗ ██████╗
██╔═══██╗ ██║    ██║██╔════╝████╗  ██║       ██║     ██╔═══██╗██╔═══██╗██╔══██╗
██║   ██║ ██║ █╗ ██║█████╗  ██╔██╗ ██║ ████╗ ██║     ██║   ██║██║   ██║██████╔╝
██║▄▄ ██║ ██║███╗██║██╔══╝  ██║╚██╗██║       ██║     ██║   ██║██║   ██║██╔═══╝
╚██████╔╝ ╚███╔███╔╝███████╗██║ ╚████║       ███████╗╚██████╔╝╚██████╔╝██║
 ╚══▀▀═╝   ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝       ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝
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


def _print_banner(device: str) -> None:
    print(BANNER)
    print(f"  model   : {MODEL_DIR}")
    print(f"  device  : {device}")
    print(f"  cwd     : {WORKDIR}")
    print(f"  shell   : {BASH_EXE or '(cmd.exe 回退)'}")
    print(f"  tools   : {', '.join(TOOL_HANDLERS)}")
    print("  输入 /help 查看命令, /exit 退出")
    print("─" * 72)


def _reset_state(state: dict) -> None:
    state["messages"] = [{"role": "system", "content": SYSTEM}]
    state["rounds_since_todo"] = 0
    TODO.items.clear()


def repl() -> None:
    tok, mdl, device = load_model()
    state = {"tok": tok, "mdl": mdl, "device": device}
    _reset_state(state)
    _print_banner(device)

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
                _reset_state(state)
                print("(history cleared)")
                continue
            if cmd == "todos":
                print(TODO.render())
                continue
            if cmd == "history":
                msgs = state["messages"]
                counts = {}
                for m in msgs:
                    counts[m["role"]] = counts.get(m["role"], 0) + 1
                print(f"messages: {len(msgs)}  {counts}")
                continue
            print(f"unknown command: /{cmd} (try /help)")
            continue

        try:
            run_turn(state, line)
        except KeyboardInterrupt:
            print("\n(turn interrupted)")


if __name__ == "__main__":
    # 支持一次性模式: `python todo.py "your query"` 仍跑单轮然后退出
    # 不带参数则进入交互式 REPL
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        tok, mdl, device = load_model()
        state = {"tok": tok, "mdl": mdl, "device": device,
                 "messages": [{"role": "system", "content": SYSTEM}],
                 "rounds_since_todo": 0}
        run_turn(state, q)
    else:
        repl()
