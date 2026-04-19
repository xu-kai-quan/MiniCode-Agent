"""MiniCode Agent v3 — ReAct agent with atomic tools.

设计要点 (相对 v2 的变化):
  1. 工具结果统一为 ToolResult(status, data, text):
     - status ∈ {"success", "partial", "error"}
     - data: 结构化结果 (list/dict). 仅供日志/UI, 不喂给模型.
     - text: 自然语言摘要. 真正进入对话历史的就是这段.
     模型只读 text — 给 JSON 会让小模型以为要 parse, 反而干扰自然交互.
  2. 高频操作拆成原子工具: LS / Glob / Grep / Read.
     bash 降级为兜底 — 留给 git, 跑脚本, 以及原子工具不覆盖的边角.
  3. partial 的判据: 用户预期 vs 实际返回之间有事实被丢弃.
     - read 截断, grep 命中过多, bash 输出爆 8000 字符 → partial
     - 用户主动 limit=N 拿到 N 条 → success (这是契约不是截断)

Qwen 的 chat_template 把工具调用编码成 XML:
  <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
工具结果以 role="tool" 的消息回传.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- 配置 ----------

MODEL_DIR = Path(os.environ.get(
    "MINICODE_MODEL_DIR",
    "E:/MYSELF/model/qwen/Qwen3.5-2B",
))
WORKDIR = Path.cwd().resolve()
SYSTEM = (
    "You are a coding agent. Prefer the dedicated atomic tools "
    "(LS, Glob, Grep, Read) over `bash` — they return structured, "
    "predictable results. Use `bash` only for operations no atomic tool "
    "covers: running git, executing scripts, package managers, etc. "
    "Read-before-write: before edit_file or before write_file/append_file "
    "on an EXISTING file, you must call Read first. Creating a brand-new "
    "file with write_file does not require a prior Read. If a write fails "
    "with CONFLICT, the file changed externally — Read it again, then retry. "
    "For any task with more than one step, call the `todo` tool first to "
    "plan, then keep it updated as you work. Exactly one item should be "
    "`in_progress` at a time. "
    "For files longer than ~200 lines, write in chunks: start with "
    "`write_file` for the first chunk, then call `append_file` repeatedly "
    "for each remaining chunk (roughly 150 lines per chunk), preserving "
    "newlines. Reply with a final message only when no more tool calls "
    "are needed."
)
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

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
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
    def tool(cls, output: str) -> "Message":
        return cls("tool", output)


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


# 写类工具集中在此. dispatch 用这个集合决定是否启用乐观锁检查.
WRITE_TOOLS = frozenset({"write_file", "edit_file", "append_file"})


class ToolRegistry:
    """工具表. 注册时同时登记 schema 和 handler — 加工具只改一处.

    dispatch 还兼一个职责: 对写类工具做"读后写 + 乐观锁"检查.
    检查在 handler 之前进行, 不污染 handler 的纯函数性.
    """

    def __init__(self, read_cache: ReadCache | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.read_cache = read_cache or ReadCache()

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

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

        cached = self.read_cache.get(abs_path)
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
        self.read_cache.record(abs_path)

    def dispatch(self, name: str, arguments: dict) -> ToolResult:
        """执行工具, 返回 ToolResult. 异常被吃掉转成 error result."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error("UNKNOWN_TOOL", f"Unknown tool: {name}")

        guard = self._check_read_before_write(name, arguments)
        if guard is not None:
            return guard

        try:
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
                    self.read_cache._entries[str(abs_path)] = stat_info
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
        raise ValueError(f"Path escapes workspace: {p}")
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
    """带行号读取. limit 是用户契约 → success; 字符上限是系统兜底 → partial."""
    try:
        p = _safe_path(WORKDIR, path)
    except ValueError as e:
        return ToolResult.error("PATH_ESCAPE", str(e))
    if not p.exists():
        return ToolResult.error("NOT_FOUND", f"File '{path}' does not exist")
    if not p.is_file():
        return ToolResult.error("NOT_A_FILE", f"Path '{path}' is not a file")

    try:
        st = p.stat()
        all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return ToolResult.error("READ_FAILED", str(e))

    total = len(all_lines)
    start = max(0, offset)
    end = total if limit is None else min(total, start + limit)
    selected = all_lines[start:end]

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
        if r.returncode != 0:
            # 命令跑了但失败 — 用 success 比 error 更准确, 模型能从 exit_code 看出.
            # error 留给"工具本身没跑成"的情况.
            return ToolResult.success(text, **data)
        return ToolResult.success(text, **data)


# =================================================================
# Todo (规划工具)
# =================================================================

class TodoManager:
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


TODO = TodoManager()


# =================================================================
# 工具注册
# =================================================================

def build_default_registry(terminal: TerminalTool) -> ToolRegistry:
    reg = ToolRegistry()

    # ---- 原子工具 (主链路) ----
    reg.register(Tool(
        name="LS",
        description="List entries in a directory (non-recursive). Use Glob for recursive.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path. Default '.'"}},
            "required": [],
        },
        handler=lambda path=".": tool_ls(path),
    ))
    reg.register(Tool(
        name="Glob",
        description="Find files by name pattern (recursive). Pattern is glob syntax like '**/*.py'.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                "path": {"type": "string", "description": "Base path. Default '.'"},
            },
            "required": ["pattern"],
        },
        handler=lambda pattern, path=".": tool_glob(pattern, path),
    ))
    reg.register(Tool(
        name="Grep",
        description="Search file contents by regex. Returns file:line:text matches.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "File or directory. Default '.'"},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive. Default false"},
                "file_pattern": {"type": "string", "description": "Only search files matching this glob (e.g. '*.py')"},
            },
            "required": ["pattern"],
        },
        handler=lambda pattern, path=".", ignore_case=False, file_pattern=None:
            tool_grep(pattern, path, ignore_case, file_pattern),
    ))
    reg.register(Tool(
        name="Read",
        description="Read a text file with line numbers. Pass limit/offset for partial reads.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "description": "Max lines to return"},
                "offset": {"type": "integer", "description": "Skip this many lines from start"},
            },
            "required": ["path"],
        },
        handler=lambda path, limit=None, offset=0: tool_read(path, limit, offset),
    ))
    reg.register(Tool(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=lambda path, content: tool_write(path, content),
    ))
    reg.register(Tool(
        name="append_file",
        description=(
            "Append content to the end of a file (creates it if missing). "
            "Use this to write files too large for a single write_file call."
        ),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=lambda path, content: tool_append(path, content),
    ))
    reg.register(Tool(
        name="edit_file",
        description="Replace a unique old_text with new_text inside a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        handler=lambda path, old_text, new_text: tool_edit(path, old_text, new_text),
    ))
    reg.register(Tool(
        name="todo",
        description=(
            "Write the full todo list. Pass every item each call. "
            "Statuses: pending, in_progress (at most one), completed."
        ),
        parameters={
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
        handler=lambda items: TODO.update(items),
    ))

    # ---- bash (兜底, 不是主链路) ----
    reg.register(Tool(
        name="bash",
        description=(
            "Fallback shell. Prefer LS/Glob/Grep/Read for file operations. "
            "Use bash only for things atomic tools don't cover: git, "
            "running scripts, package managers, etc."
        ),
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=lambda command: terminal.run(command),
    ))
    return reg


# =================================================================
# Qwen 输出解析
# =================================================================

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _coerce(value: str) -> Any:
    """XML 参数值都是字符串; 把 JSON-ish 的还原成对象."""
    s = value.strip()
    if (s and s[0] in "[{") or s in ("true", "false", "null"):
        try:
            return json.loads(s)
        except Exception:
            return value
    if s.isdigit():
        return int(s)
    return value


def parse_tool_calls(text: str) -> list[dict]:
    calls: list[dict] = []
    for tc in TOOL_CALL_RE.findall(text):
        fm = FUNCTION_RE.search(tc)
        if not fm:
            continue
        name = fm.group(1).strip()
        args = {k: _coerce(v) for k, v in PARAM_RE.findall(fm.group(2))}
        calls.append({"name": name, "arguments": args})
    return calls


def split_think(text: str) -> tuple[str, str]:
    """返回 (think, visible). think 仅供日志; visible 是真正放回历史的."""
    m = THINK_RE.search(text)
    think = m.group(1).strip() if m else ""
    visible = THINK_RE.sub("", text).strip()
    return think, visible


# =================================================================
# ContextBuilder
# =================================================================

class ContextBuilder:
    """把 (system + history + tool schemas) 拼成模型的输入 prompt."""

    def __init__(self, tokenizer, registry: ToolRegistry) -> None:
        self.tokenizer = tokenizer
        self.registry = registry

    def build(self, history: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            [m.to_dict() for m in history],
            tools=self.registry.schemas(),
            tokenize=False,
            add_generation_prompt=True,
        )


# =================================================================
# 模型加载 + 生成
# =================================================================

def load_model() -> tuple[Any, Any, str]:
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    print(f"Loading Qwen from {MODEL_DIR} on {device}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), trust_remote_code=True, dtype=dtype)
    mdl.to(device).eval()
    return tok, mdl, device


def _stop_token_ids(tok) -> list[int]:
    ids: set[int] = set()
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


def generate(tok, mdl, device: str, prompt: str) -> str:
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


class ReActAgent:
    NAG_THRESHOLD = 3

    def __init__(
        self,
        tok,
        mdl,
        device: str,
        registry: ToolRegistry,
        context: ContextBuilder,
        system_prompt: str = SYSTEM,
    ) -> None:
        self.tok = tok
        self.mdl = mdl
        self.device = device
        self.registry = registry
        self.context = context
        self.history: list[Message] = [Message.system(system_prompt)]
        self.rounds_since_todo = 0

    def reset(self) -> None:
        self.history = [Message.system(self.history[0].content)]
        self.rounds_since_todo = 0
        TODO.items.clear()
        self.registry.read_cache.clear()

    def step(self) -> StepResult:
        prompt = self.context.build(self.history)
        raw = generate(self.tok, self.mdl, self.device, prompt)
        thought, visible = split_think(raw)
        actions = parse_tool_calls(visible)
        truncated = visible.count("<tool_call>") > len(actions)

        self.history.append(Message.assistant(
            visible,
            tool_calls=[
                {"type": "function",
                 "function": {"name": a["name"], "arguments": a["arguments"]}}
                for a in actions
            ] or None,
        ))

        observations: list[ToolResult] = []
        used_todo = False
        for a in actions:
            result = self.registry.dispatch(a["name"], a["arguments"])
            observations.append(result)
            # 关键: 只把 text 喂给模型, data 留给日志.
            self.history.append(Message.tool(result.text))
            if a["name"] == "todo":
                used_todo = True

        return StepResult(
            thought=thought, visible=visible,
            actions=actions, observations=observations,
            truncated=truncated, used_todo=used_todo,
        )

    def run(self, query: str, max_rounds: int = MAX_ROUNDS) -> None:
        self.history.append(Message.user(query))
        _box("▶ USER", query, char="═", color=C.USER)

        for turn in range(1, max_rounds + 1):
            print(f"\n{C.META}══════════ Turn {turn} ══════════{C.RESET}")
            print(f"{C.META}→ [1] 调用模型 …{C.RESET}")
            res = self.step()

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

            if not res.actions:
                if res.truncated:
                    _box("⚠️  TRUNCATED (工具调用被截断, 让模型改用分片重写)",
                         "输出里有 <tool_call> 但没闭合, 撞到了 MAX_NEW_TOKENS。",
                         color=C.WARN)
                    self.history.append(Message.user(
                        "<reminder>Your previous tool call was cut off before "
                        "</tool_call>. Do not repeat the same oversized write_file. "
                        "Instead: call write_file with only the FIRST ~150 lines, "
                        "then call append_file repeatedly for the remaining chunks.</reminder>"
                    ))
                    continue
                _box("✅ DONE (无工具调用, 本轮结束)", "", char="═", color=C.DONE)
                return

            self.rounds_since_todo = 0 if res.used_todo else self.rounds_since_todo + 1
            print(f"{C.META}→ [3] rounds_since_todo = {self.rounds_since_todo}"
                  f"{'  (本轮调用了 todo, 已重置)' if res.used_todo else ''}{C.RESET}")
            if self.rounds_since_todo >= self.NAG_THRESHOLD:
                _box("⚠️  NAG", f"连续 {self.NAG_THRESHOLD} 轮未调用 todo, 注入提醒",
                     color=C.WARN)
                self.history.append(Message.user(
                    "<reminder>Update your todos.</reminder>"
                ))
                self.rounds_since_todo = 0

        _box("⛔ MAX ROUNDS", f"达到 {max_rounds} 轮上限, 强制退出本轮",
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
              MiniCode Agent v3 — atomic tools edition
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
    print(f"  {C.DIM}model  :{C.RESET} {MODEL_DIR}")
    print(f"  {C.DIM}device :{C.RESET} {agent.device}")
    print(f"  {C.DIM}cwd    :{C.RESET} {WORKDIR}")
    print(f"  {C.DIM}shell  :{C.RESET} {terminal.bash_exe or '(cmd.exe 回退)'}")
    print(f"  {C.DIM}grep   :{C.RESET} {_RG_PATH or '(纯 Python 兜底, 装 ripgrep 更快)'}")
    print(f"  {C.DIM}tools  :{C.RESET} {C.ACTION}{', '.join(agent.registry.names())}{C.RESET}")
    print(f"  {C.DIM}输入 /help 查看命令, /exit 退出{C.RESET}")
    print(f"{C.DIM}{'─' * 72}{C.RESET}")


def repl() -> None:
    tok, mdl, device = load_model()
    terminal = TerminalTool(WORKDIR)
    registry = build_default_registry(terminal)
    context = ContextBuilder(tok, registry)
    agent = ReActAgent(tok, mdl, device, registry, context)

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
                agent.reset()
                print("(history cleared)")
                continue
            if cmd == "todos":
                print(TODO.render())
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
        tok, mdl, device = load_model()
        terminal = TerminalTool(WORKDIR)
        registry = build_default_registry(terminal)
        context = ContextBuilder(tok, registry)
        agent = ReActAgent(tok, mdl, device, registry, context)
        agent.run(q)
    else:
        repl()
