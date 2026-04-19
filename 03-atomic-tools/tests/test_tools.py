"""工具层测试 — 覆盖 ToolResult 三态、读后写、乐观锁、各 error code、Grep 双路径.

不加载模型, 只测 ToolRegistry / 工具 handler / 乐观锁 dispatch 守卫.
跑法: cd 03-atomic-tools/ && pytest tests/ -v
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import todo as M


# ---------- fixtures ----------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """每个测试一个干净的临时工作目录, 并把 todo.WORKDIR 指过去."""
    monkeypatch.setattr(M, "WORKDIR", tmp_path.resolve())
    return tmp_path.resolve()


@pytest.fixture
def registry(workspace):
    """新建一个 registry — read_cache 自然是空的."""
    terminal = M.TerminalTool(workspace)
    return M.build_default_registry(terminal)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# =================================================================
# ToolResult 三态
# =================================================================

class TestToolResult:
    def test_success_constructor(self):
        r = M.ToolResult.success("done", paths=["a.py"])
        assert r.status == "success"
        assert r.text == "done"
        assert r.data == {"paths": ["a.py"]}

    def test_partial_constructor(self):
        r = M.ToolResult.partial("truncated", matches=[1, 2])
        assert r.status == "partial"
        assert r.data["matches"] == [1, 2]

    def test_error_constructor_carries_code(self):
        r = M.ToolResult.error("NOT_FOUND", "missing file")
        assert r.status == "error"
        assert r.data["code"] == "NOT_FOUND"
        assert r.data["message"] == "missing file"
        assert "[NOT_FOUND]" in r.text


# =================================================================
# 高频原子层: LS / Glob / Grep / Read
# =================================================================

class TestLS:
    def test_empty_dir(self, registry, workspace):
        r = registry.dispatch("LS", {"path": "."})
        assert r.status == "success"
        assert "empty" in r.text

    def test_lists_files_and_dirs(self, registry, workspace):
        _write(workspace / "a.py", "x")
        (workspace / "sub").mkdir()
        r = registry.dispatch("LS", {"path": "."})
        assert r.status == "success"
        assert any(e["name"] == "a.py" and e["type"] == "file" for e in r.data["entries"])
        assert any(e["name"] == "sub" and e["type"] == "dir" for e in r.data["entries"])

    def test_not_found(self, registry):
        r = registry.dispatch("LS", {"path": "ghost"})
        assert r.status == "error"
        assert r.data["code"] == "NOT_FOUND"

    def test_path_escape(self, registry):
        r = registry.dispatch("LS", {"path": "../.."})
        assert r.status == "error"
        assert r.data["code"] == "PATH_ESCAPE"

    def test_not_a_dir(self, registry, workspace):
        _write(workspace / "a.py", "x")
        r = registry.dispatch("LS", {"path": "a.py"})
        assert r.status == "error"
        assert r.data["code"] == "NOT_A_DIR"


class TestGlob:
    def test_finds_files_recursive(self, registry, workspace):
        _write(workspace / "a.py", "x")
        _write(workspace / "sub" / "b.py", "y")
        r = registry.dispatch("Glob", {"pattern": "**/*.py"})
        assert r.status == "success"
        assert "a.py" in r.text and "sub/b.py" in r.text

    def test_no_match_is_success(self, registry, workspace):
        r = registry.dispatch("Glob", {"pattern": "*.xyz"})
        assert r.status == "success"  # 空数组就是事实, 不是错误
        assert r.data["paths"] == []


class TestGrep:
    def test_match(self, registry, workspace):
        _write(workspace / "a.py", "def foo():\n    pass\n")
        r = registry.dispatch("Grep", {"pattern": "def foo", "path": "."})
        assert r.status == "success"
        assert "a.py:1" in r.text

    def test_no_match_is_success(self, registry, workspace):
        _write(workspace / "a.py", "x\n")
        r = registry.dispatch("Grep", {"pattern": "ZZZ_xyz", "path": "."})
        assert r.status == "success"
        assert r.data["matches"] == []

    def test_file_pattern_filters(self, registry, workspace):
        _write(workspace / "a.py", "secret\n")
        _write(workspace / "b.txt", "secret\n")
        r = registry.dispatch("Grep", {"pattern": "secret", "path": ".", "file_pattern": "*.py"})
        assert r.status == "success"
        assert "a.py" in r.text
        assert "b.txt" not in r.text

    def test_python_fallback_path(self, registry, workspace, monkeypatch):
        """显式走 Python 路径 — 即便机器装了 rg 也要保证兜底版能跑."""
        monkeypatch.setattr(M, "_RG_PATH", None)
        _write(workspace / "a.py", "hello\n")
        r = registry.dispatch("Grep", {"pattern": "hello", "path": "."})
        assert r.status == "success"
        assert "a.py:1" in r.text

    def test_python_fallback_skips_noise_dirs(self, registry, workspace, monkeypatch):
        monkeypatch.setattr(M, "_RG_PATH", None)
        _write(workspace / ".git" / "secret.txt", "needle\n")
        _write(workspace / "node_modules" / "x.js", "needle\n")
        _write(workspace / "real.py", "needle\n")
        r = registry.dispatch("Grep", {"pattern": "needle", "path": "."})
        assert r.status == "success"
        assert "real.py" in r.text
        assert ".git" not in r.text
        assert "node_modules" not in r.text


class TestRead:
    def test_reads_with_line_numbers(self, registry, workspace):
        _write(workspace / "a.py", "one\ntwo\nthree\n")
        r = registry.dispatch("Read", {"path": "a.py"})
        assert r.status == "success"
        assert "1\tone" in r.text
        assert "3\tthree" in r.text
        assert r.data["total_lines"] == 3

    def test_limit_is_contract_not_partial(self, registry, workspace):
        """用户主动 limit=2 拿到 2 行 -> success, 不是 partial."""
        _write(workspace / "a.py", "\n".join(f"line{i}" for i in range(100)) + "\n")
        r = registry.dispatch("Read", {"path": "a.py", "limit": 2})
        assert r.status == "success"

    def test_char_overflow_is_partial(self, registry, workspace):
        """没限制但内容爆 READ_MAX_CHARS -> partial."""
        _write(workspace / "huge.txt", "x" * (M.READ_MAX_CHARS * 2))
        r = registry.dispatch("Read", {"path": "huge.txt"})
        assert r.status == "partial"

    def test_not_found(self, registry):
        r = registry.dispatch("Read", {"path": "ghost.py"})
        assert r.status == "error"
        assert r.data["code"] == "NOT_FOUND"

    def test_stat_field_stripped_from_data(self, registry, workspace):
        """_stat 是工具→registry 的内部约定, 不应泄漏到 data."""
        _write(workspace / "a.py", "x\n")
        r = registry.dispatch("Read", {"path": "a.py"})
        assert "_stat" not in r.data


# =================================================================
# 中频受控层: 读后写 + 乐观锁
# =================================================================

class TestReadBeforeWrite:
    def test_edit_existing_without_read_blocked(self, registry, workspace):
        _write(workspace / "a.py", "hello\n")
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "hello", "new_text": "hi"})
        assert r.status == "error"
        assert r.data["code"] == "NOT_READ"

    def test_append_existing_without_read_blocked(self, registry, workspace):
        _write(workspace / "a.py", "hello\n")
        r = registry.dispatch("append_file", {"path": "a.py", "content": "x"})
        assert r.status == "error"
        assert r.data["code"] == "NOT_READ"

    def test_write_overwrite_existing_without_read_blocked(self, registry, workspace):
        _write(workspace / "a.py", "old\n")
        r = registry.dispatch("write_file", {"path": "a.py", "content": "new"})
        assert r.status == "error"
        assert r.data["code"] == "NOT_READ"

    def test_write_new_file_allowed(self, registry):
        r = registry.dispatch("write_file", {"path": "fresh.py", "content": "x"})
        assert r.status == "success"

    def test_edit_after_read_works(self, registry, workspace):
        _write(workspace / "a.py", "hello\n")
        registry.dispatch("Read", {"path": "a.py"})
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "hello", "new_text": "hi"})
        assert r.status == "success"

    def test_consecutive_edits_use_refreshed_cache(self, registry, workspace):
        """写完 cache 自动刷新, 不需要再 Read 才能改第二次."""
        _write(workspace / "a.py", "v1\n")
        registry.dispatch("Read", {"path": "a.py"})
        r1 = registry.dispatch("edit_file", {"path": "a.py", "old_text": "v1", "new_text": "v2"})
        assert r1.status == "success"
        r2 = registry.dispatch("edit_file", {"path": "a.py", "old_text": "v2", "new_text": "v3"})
        assert r2.status == "success"

    def test_write_fresh_then_edit_no_read_needed(self, registry):
        """新建文件后立刻 edit — cache 应该有了, 不应 NOT_READ."""
        registry.dispatch("write_file", {"path": "fresh.py", "content": "v1\n"})
        r = registry.dispatch("edit_file", {"path": "fresh.py", "old_text": "v1", "new_text": "v2"})
        assert r.status == "success"


class TestOptimisticLock:
    def test_external_modification_triggers_conflict(self, registry, workspace):
        path = workspace / "a.py"
        _write(path, "v1\n")
        registry.dispatch("Read", {"path": "a.py"})
        time.sleep(0.05)  # 确保 mtime_ns 变化
        path.write_text("v1_extern\n", encoding="utf-8")
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "v1_extern", "new_text": "v2"})
        assert r.status == "error"
        assert r.data["code"] == "CONFLICT"

    def test_re_read_resolves_conflict(self, registry, workspace):
        path = workspace / "a.py"
        _write(path, "v1\n")
        registry.dispatch("Read", {"path": "a.py"})
        time.sleep(0.05)
        path.write_text("v1_extern\n", encoding="utf-8")
        # 第一次撞 CONFLICT
        registry.dispatch("edit_file", {"path": "a.py", "old_text": "v1_extern", "new_text": "v2"})
        # 重新 Read 后应该恢复
        registry.dispatch("Read", {"path": "a.py"})
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "v1_extern", "new_text": "v2"})
        assert r.status == "success"


class TestEditErrors:
    def test_edit_not_found(self, registry):
        r = registry.dispatch("edit_file", {"path": "ghost.py", "old_text": "x", "new_text": "y"})
        assert r.status == "error"
        # 不存在文件: 走 read-before-write 守卫前的"_safe_path 不抛异常但 .exists() False"
        # 守卫放行 (按"新文件"对待), handler 自己检测 NOT_FOUND
        assert r.data["code"] == "NOT_FOUND"

    def test_edit_no_match(self, registry, workspace):
        _write(workspace / "a.py", "hello\n")
        registry.dispatch("Read", {"path": "a.py"})
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "ZZZ", "new_text": "x"})
        assert r.status == "error"
        assert r.data["code"] == "NO_MATCH"

    def test_edit_ambiguous(self, registry, workspace):
        _write(workspace / "a.py", "x\nx\n")
        registry.dispatch("Read", {"path": "a.py"})
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "x", "new_text": "y"})
        assert r.status == "error"
        assert r.data["code"] == "AMBIGUOUS"
        assert r.data["occurrences"] == 2


# =================================================================
# 规划工具: todo
# =================================================================

class TestTodo:
    def test_basic_update(self, registry):
        r = registry.dispatch("todo", {"items": [
            {"id": "1", "text": "step a", "status": "in_progress"},
            {"id": "2", "text": "step b", "status": "pending"},
        ]})
        assert r.status == "success"
        assert "[>]" in r.text
        assert "[ ]" in r.text

    def test_two_in_progress_rejected(self, registry):
        r = registry.dispatch("todo", {"items": [
            {"id": "1", "text": "a", "status": "in_progress"},
            {"id": "2", "text": "b", "status": "in_progress"},
        ]})
        assert r.status == "error"
        assert r.data["code"] == "INVALID_STATE"


# =================================================================
# Dispatch 层: 异常处理 / 路径沙箱 / 未知工具
# =================================================================

class TestDispatch:
    def test_unknown_tool(self, registry):
        r = registry.dispatch("NoSuchTool", {})
        assert r.status == "error"
        assert r.data["code"] == "UNKNOWN_TOOL"

    def test_bad_args_caught(self, registry):
        r = registry.dispatch("LS", {"path": ".", "bogus_kwarg": 1})
        assert r.status == "error"
        assert r.data["code"] == "BAD_ARGS"

    def test_path_escape_takes_priority_over_not_read(self, registry):
        """路径越界要优先于 NOT_READ — 别让 NOT_READ 把真正的安全问题盖了."""
        r = registry.dispatch("edit_file", {
            "path": "../../etc/passwd", "old_text": "x", "new_text": "y",
        })
        assert r.status == "error"
        assert r.data["code"] == "PATH_ESCAPE"


# =================================================================
# 解析层: 模型输出的 <tool_call> XML
# =================================================================

class TestParser:
    def test_simple_tool_call(self):
        text = (
            "<tool_call><function=Read>"
            "<parameter=path>a.py</parameter>"
            "</function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        assert calls == [{"name": "Read", "arguments": {"path": "a.py"}}]

    def test_coerces_int(self):
        text = (
            "<tool_call><function=Read>"
            "<parameter=path>a.py</parameter>"
            "<parameter=limit>10</parameter>"
            "</function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        assert calls[0]["arguments"]["limit"] == 10

    def test_coerces_json_array(self):
        text = (
            "<tool_call><function=todo>"
            "<parameter=items>[{\"id\":\"1\",\"text\":\"a\"}]</parameter>"
            "</function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        assert calls[0]["arguments"]["items"] == [{"id": "1", "text": "a"}]

    def test_split_think_strips_thought(self):
        thought, visible = M.split_think("<think>plan</think>actual reply")
        assert thought == "plan"
        assert visible == "actual reply"

    def test_no_think_block(self):
        thought, visible = M.split_think("just a reply")
        assert thought == ""
        assert visible == "just a reply"
