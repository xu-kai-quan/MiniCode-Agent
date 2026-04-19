"""01-bash-only 工具层测试.

只覆盖纯逻辑: 路径沙箱 / 4 个文件工具 / TodoManager / Qwen XML 解析.
跑模型和 bash 子进程不在这一层 — 模型靠手动跑, bash 子进程在 CI 环境太脆
(Windows/Linux 行为差异), 也不是 v1 想突出的能力。
"""
from __future__ import annotations

import pytest

import todo as M


@pytest.fixture(autouse=True)
def _isolate_workdir(tmp_path, monkeypatch):
    """每个测试拿一个干净的临时 WORKDIR + 全新 TodoManager."""
    monkeypatch.setattr(M, "WORKDIR", tmp_path.resolve())
    monkeypatch.setattr(M, "TODO", M.TodoManager())
    return tmp_path


# ---------- safe_path ----------

class TestSafePath:
    def test_relative_path_resolves_inside_workdir(self, _isolate_workdir):
        p = M.safe_path("a/b.txt")
        assert p == _isolate_workdir / "a" / "b.txt"

    def test_dot_resolves_to_workdir(self, _isolate_workdir):
        assert M.safe_path(".") == _isolate_workdir

    def test_parent_escape_raises(self):
        with pytest.raises(ValueError, match="escapes workspace"):
            M.safe_path("../outside.txt")

    def test_deep_parent_escape_raises(self):
        with pytest.raises(ValueError, match="escapes workspace"):
            M.safe_path("a/b/../../../etc/passwd")

    def test_absolute_path_outside_raises(self, tmp_path):
        # 绝对路径如果不在 WORKDIR 下也要拦住
        outside = tmp_path.parent / "definitely_outside.txt"
        with pytest.raises(ValueError, match="escapes workspace"):
            M.safe_path(str(outside))


# ---------- run_read / run_write / run_edit ----------

class TestReadWrite:
    def test_write_creates_file_and_parents(self, _isolate_workdir):
        msg = M.run_write("nested/dir/hello.txt", "hi")
        assert "wrote 2 chars" in msg
        assert (_isolate_workdir / "nested/dir/hello.txt").read_text(encoding="utf-8") == "hi"

    def test_write_overwrites_existing(self, _isolate_workdir):
        f = _isolate_workdir / "x.txt"
        f.write_text("old", encoding="utf-8")
        M.run_write("x.txt", "new")
        assert f.read_text(encoding="utf-8") == "new"

    def test_read_roundtrip(self, _isolate_workdir):
        M.run_write("a.txt", "line1\nline2\nline3\n")
        assert M.run_read("a.txt") == "line1\nline2\nline3"

    def test_read_with_limit_truncates(self, _isolate_workdir):
        M.run_write("a.txt", "\n".join(f"L{i}" for i in range(10)))
        out = M.run_read("a.txt", limit=3)
        assert out == "L0\nL1\nL2"

    def test_read_caps_at_8000_chars(self, _isolate_workdir):
        big = "x" * 20000
        M.run_write("big.txt", big)
        assert len(M.run_read("big.txt")) == 8000

    def test_read_escape_raises(self):
        with pytest.raises(ValueError):
            M.run_read("../etc/passwd")


class TestEdit:
    def test_edit_unique_match(self, _isolate_workdir):
        M.run_write("a.py", "x = 1\ny = 2\n")
        msg = M.run_edit("a.py", "x = 1", "x = 99")
        assert "edited" in msg
        assert (_isolate_workdir / "a.py").read_text(encoding="utf-8") == "x = 99\ny = 2\n"

    def test_edit_zero_matches_returns_error_string(self, _isolate_workdir):
        M.run_write("a.py", "x = 1\n")
        msg = M.run_edit("a.py", "NOPE", "z")
        assert msg.startswith("error: old_text must match exactly once (found 0)")

    def test_edit_multiple_matches_returns_error_string(self, _isolate_workdir):
        M.run_write("a.py", "x\nx\n")
        msg = M.run_edit("a.py", "x", "z")
        assert "found 2" in msg
        # 文件不应被改动
        assert (_isolate_workdir / "a.py").read_text(encoding="utf-8") == "x\nx\n"


# ---------- TodoManager ----------

class TestTodoManager:
    def test_empty_render(self):
        assert M.TodoManager().render() == "(no todos)"

    def test_update_and_render(self):
        t = M.TodoManager()
        out = t.update([
            {"id": "1", "text": "do A", "status": "completed"},
            {"id": "2", "text": "do B", "status": "in_progress"},
            {"id": "3", "text": "do C"},  # 默认 pending
        ])
        assert "[x] do A" in out
        assert "[>] do B" in out
        assert "[ ] do C" in out

    def test_two_in_progress_raises(self):
        t = M.TodoManager()
        with pytest.raises(ValueError, match="Only one task"):
            t.update([
                {"id": "1", "text": "a", "status": "in_progress"},
                {"id": "2", "text": "b", "status": "in_progress"},
            ])

    def test_zero_in_progress_ok(self):
        t = M.TodoManager()
        out = t.update([{"id": "1", "text": "a", "status": "pending"}])
        assert "[ ] a" in out


# ---------- _coerce ----------

class TestCoerce:
    def test_passthrough_string(self):
        assert M._coerce("hello") == "hello"

    def test_digits_to_int(self):
        assert M._coerce("42") == 42

    def test_negative_stays_string(self):
        # _coerce 只看 isdigit, "-1" 不算
        assert M._coerce("-1") == "-1"

    def test_json_array(self):
        assert M._coerce('[1, 2, 3]') == [1, 2, 3]

    def test_json_object(self):
        assert M._coerce('{"a": 1}') == {"a": 1}

    def test_true_false_null(self):
        assert M._coerce("true") is True
        assert M._coerce("false") is False
        assert M._coerce("null") is None

    def test_invalid_json_falls_back_to_original(self):
        # 以 [ 开头但不是合法 JSON, 应该原样返回 (注意: 返回 value, 不是 strip 后的 s)
        assert M._coerce("[not json") == "[not json"


# ---------- parse_tool_calls / strip_think / extract_think ----------

class TestParser:
    def test_single_tool_call(self):
        text = (
            "<tool_call><function=read_file>"
            "<parameter=path>a.py</parameter>"
            "</function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        assert calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call><function=read_file><parameter=path>a</parameter></function></tool_call>"
            "<tool_call><function=read_file><parameter=path>b</parameter></function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        assert [c["arguments"]["path"] for c in calls] == ["a", "b"]

    def test_int_param_coerced(self):
        text = (
            "<tool_call><function=read_file>"
            "<parameter=path>a.py</parameter>"
            "<parameter=limit>10</parameter>"
            "</function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        assert calls[0]["arguments"]["limit"] == 10

    def test_json_array_param_coerced(self):
        text = (
            "<tool_call><function=todo>"
            '<parameter=items>[{"id":"1","text":"a","status":"pending"}]</parameter>'
            "</function></tool_call>"
        )
        calls = M.parse_tool_calls(text)
        items = calls[0]["arguments"]["items"]
        assert items == [{"id": "1", "text": "a", "status": "pending"}]

    def test_no_tool_calls_returns_empty(self):
        assert M.parse_tool_calls("just some prose, no tools") == []

    def test_malformed_tool_call_skipped(self):
        # <tool_call> 没有 <function=...>, 应该跳过而不是抛
        assert M.parse_tool_calls("<tool_call>garbage</tool_call>") == []

    def test_strip_think_removes_block(self):
        assert M.strip_think("<think>internal</think>visible") == "visible"

    def test_strip_think_handles_no_think(self):
        assert M.strip_think("plain") == "plain"

    def test_strip_think_multiline(self):
        text = "<think>line1\nline2</think>\nactual reply"
        assert M.strip_think(text) == "actual reply"

    def test_extract_think(self):
        assert M.extract_think("<think>内部</think>对外") == "内部"

    def test_extract_think_empty_when_missing(self):
        assert M.extract_think("no think tag") == ""


# ---------- TOOL_HANDLERS dispatch table ----------

class TestToolHandlers:
    def test_all_five_tools_registered(self):
        assert set(M.TOOL_HANDLERS) == {"bash", "read_file", "write_file", "edit_file", "todo"}

    def test_handler_runs_via_kwargs(self, _isolate_workdir):
        out = M.TOOL_HANDLERS["write_file"](path="a.txt", content="hi")
        assert "wrote 2 chars" in out

    def test_todo_handler_routes_to_global(self):
        out = M.TOOL_HANDLERS["todo"](items=[{"id": "1", "text": "a", "status": "pending"}])
        assert "[ ] a" in out
        assert M.TODO.items[0]["text"] == "a"
