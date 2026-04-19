"""02-sandboxed 工具层测试.

覆盖: 路径沙箱 / read+write+append+edit / TodoManager / Qwen XML 解析 /
bash 探测的纯逻辑分支. 不跑真实 bash 子进程 (CI 跨平台不稳, 而且 v2 的卖点
是分片写 + REPL, 不是 bash 本身).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import todo as M


@pytest.fixture(autouse=True)
def _isolate_workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "WORKDIR", tmp_path.resolve())
    monkeypatch.setattr(M, "TODO", M.TodoManager())
    return tmp_path


# ---------- safe_path ----------

class TestSafePath:
    def test_relative_path_resolves_inside_workdir(self, _isolate_workdir):
        assert M.safe_path("a/b.txt") == _isolate_workdir / "a" / "b.txt"

    def test_parent_escape_raises(self):
        with pytest.raises(ValueError, match="escapes workspace"):
            M.safe_path("../outside.txt")

    def test_absolute_path_outside_raises(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        with pytest.raises(ValueError, match="escapes workspace"):
            M.safe_path(str(outside))


# ---------- read / write / append / edit ----------

class TestReadWrite:
    def test_write_creates_parents(self, _isolate_workdir):
        msg = M.run_write("a/b/c.txt", "hi")
        assert "wrote 2 chars" in msg
        assert (_isolate_workdir / "a/b/c.txt").read_text(encoding="utf-8") == "hi"

    def test_read_with_limit(self, _isolate_workdir):
        M.run_write("x.txt", "L1\nL2\nL3\nL4")
        assert M.run_read("x.txt", limit=2) == "L1\nL2"

    def test_read_caps_at_8000(self, _isolate_workdir):
        M.run_write("big.txt", "x" * 20000)
        assert len(M.run_read("big.txt")) == 8000


class TestAppend:
    """v2 新增: append_file 是为了分片写大文件."""

    def test_append_creates_when_missing(self, _isolate_workdir):
        msg = M.run_append("new.txt", "hello")
        assert "appended 5 chars" in msg
        assert "now 5 bytes" in msg
        assert (_isolate_workdir / "new.txt").read_text(encoding="utf-8") == "hello"

    def test_append_extends_existing(self, _isolate_workdir):
        M.run_write("x.txt", "part1\n")
        M.run_append("x.txt", "part2\n")
        M.run_append("x.txt", "part3\n")
        assert (_isolate_workdir / "x.txt").read_text(encoding="utf-8") == "part1\npart2\npart3\n"

    def test_append_creates_parent_dirs(self, _isolate_workdir):
        M.run_append("deep/nested/x.txt", "hi")
        assert (_isolate_workdir / "deep/nested/x.txt").read_text(encoding="utf-8") == "hi"

    def test_append_escape_raises(self):
        with pytest.raises(ValueError):
            M.run_append("../evil.txt", "x")

    def test_append_byte_count_in_message(self, _isolate_workdir):
        M.run_write("x.txt", "abc")
        msg = M.run_append("x.txt", "def")
        # 总字节数 = 6
        assert "now 6 bytes" in msg


class TestEdit:
    def test_edit_unique_match(self, _isolate_workdir):
        M.run_write("a.py", "x = 1\n")
        M.run_edit("a.py", "x = 1", "x = 99")
        assert (_isolate_workdir / "a.py").read_text(encoding="utf-8") == "x = 99\n"

    def test_edit_zero_matches_returns_error_string(self, _isolate_workdir):
        M.run_write("a.py", "x = 1\n")
        msg = M.run_edit("a.py", "NOPE", "z")
        assert "found 0" in msg

    def test_edit_multiple_matches_returns_error_string(self, _isolate_workdir):
        M.run_write("a.py", "x\nx\n")
        msg = M.run_edit("a.py", "x", "z")
        assert "found 2" in msg
        assert (_isolate_workdir / "a.py").read_text(encoding="utf-8") == "x\nx\n"


# ---------- _find_bash ----------

class TestFindBash:
    """_find_bash 的逻辑核心: 优先 Git 候选路径, 兜底 PATH 但排除 System32."""

    def test_picks_first_existing_candidate(self, tmp_path, monkeypatch):
        fake_bash = tmp_path / "bash.exe"
        fake_bash.write_text("#!/bin/sh\n")
        # 让所有原候选都不存在, 通过 PATH 兜底命中 fake_bash
        monkeypatch.setattr(M, "Path", Path)  # 没必要换, 留给清晰
        monkeypatch.setattr("shutil.which", lambda name: str(fake_bash))
        # 模拟所有硬编码候选都不存在
        original_is_file = Path.is_file
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        result = M._find_bash()
        # 恢复
        monkeypatch.setattr(Path, "is_file", original_is_file)
        assert result == str(fake_bash)

    def test_excludes_system32_bash(self, monkeypatch):
        """WSL 的 C:\\Windows\\System32\\bash.exe 输出 UTF-16, 必须排除."""
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr("shutil.which", lambda name: r"C:\Windows\System32\bash.exe")
        assert M._find_bash() is None

    def test_returns_none_when_nothing_found(self, monkeypatch):
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert M._find_bash() is None


# ---------- TodoManager ----------

class TestTodoManager:
    def test_empty_render(self):
        assert M.TodoManager().render() == "(no todos)"

    def test_update_and_render(self):
        t = M.TodoManager()
        out = t.update([
            {"id": "1", "text": "a", "status": "completed"},
            {"id": "2", "text": "b", "status": "in_progress"},
            {"id": "3", "text": "c"},
        ])
        assert "[x] a" in out and "[>] b" in out and "[ ] c" in out

    def test_two_in_progress_raises(self):
        with pytest.raises(ValueError, match="Only one task"):
            M.TodoManager().update([
                {"id": "1", "text": "a", "status": "in_progress"},
                {"id": "2", "text": "b", "status": "in_progress"},
            ])


# ---------- _coerce ----------

class TestCoerce:
    def test_passthrough_string(self):
        assert M._coerce("hello") == "hello"

    def test_digits_to_int(self):
        assert M._coerce("42") == 42

    def test_json_array(self):
        assert M._coerce('[1, 2]') == [1, 2]

    def test_json_object(self):
        assert M._coerce('{"a": 1}') == {"a": 1}

    def test_true_false_null(self):
        assert M._coerce("true") is True
        assert M._coerce("false") is False
        assert M._coerce("null") is None

    def test_invalid_json_falls_back(self):
        assert M._coerce("[bad") == "[bad"


# ---------- parser ----------

class TestParser:
    def test_single_call(self):
        text = "<tool_call><function=read_file><parameter=path>a.py</parameter></function></tool_call>"
        assert M.parse_tool_calls(text) == [{"name": "read_file", "arguments": {"path": "a.py"}}]

    def test_multiple_calls(self):
        text = (
            "<tool_call><function=read_file><parameter=path>a</parameter></function></tool_call>"
            "<tool_call><function=read_file><parameter=path>b</parameter></function></tool_call>"
        )
        assert [c["arguments"]["path"] for c in M.parse_tool_calls(text)] == ["a", "b"]

    def test_int_param_coerced(self):
        text = (
            "<tool_call><function=read_file>"
            "<parameter=path>a.py</parameter><parameter=limit>5</parameter>"
            "</function></tool_call>"
        )
        assert M.parse_tool_calls(text)[0]["arguments"]["limit"] == 5

    def test_no_calls(self):
        assert M.parse_tool_calls("just prose") == []

    def test_malformed_skipped(self):
        assert M.parse_tool_calls("<tool_call>garbage</tool_call>") == []

    def test_strip_think(self):
        assert M.strip_think("<think>internal</think>visible") == "visible"

    def test_strip_think_multiline(self):
        assert M.strip_think("<think>a\nb</think>\nreply") == "reply"

    def test_extract_think(self):
        assert M.extract_think("<think>x</think>y") == "x"

    def test_extract_think_missing(self):
        assert M.extract_think("plain") == ""


# ---------- TOOL_HANDLERS ----------

class TestToolHandlers:
    def test_six_tools_registered(self):
        """v2 比 v1 多了 append_file."""
        assert set(M.TOOL_HANDLERS) == {
            "bash", "read_file", "write_file", "append_file", "edit_file", "todo"
        }

    def test_write_handler_via_kwargs(self, _isolate_workdir):
        out = M.TOOL_HANDLERS["write_file"](path="a.txt", content="hi")
        assert "wrote 2 chars" in out

    def test_append_handler_via_kwargs(self, _isolate_workdir):
        M.TOOL_HANDLERS["write_file"](path="a.txt", content="part1\n")
        out = M.TOOL_HANDLERS["append_file"](path="a.txt", content="part2\n")
        assert "appended" in out


# ---------- _stop_token_ids (v2 新增) ----------

class TestStopTokenIds:
    """v2 新增: 防止模型脑补下一轮 user/tool_response 写满 max_new_tokens."""

    def test_includes_eos(self):
        class FakeTok:
            eos_token_id = 7
            def convert_tokens_to_ids(self, m):
                return -1
        ids = M._stop_token_ids(FakeTok())
        assert 7 in ids

    def test_includes_im_end_when_present(self):
        class FakeTok:
            eos_token_id = 7
            def convert_tokens_to_ids(self, m):
                return {"<|im_end|>": 100, "<|endoftext|>": 101}.get(m, -1)
        ids = set(M._stop_token_ids(FakeTok()))
        assert ids == {7, 100, 101}

    def test_skips_negative_ids(self):
        class FakeTok:
            eos_token_id = 7
            def convert_tokens_to_ids(self, m):
                return -1
        ids = M._stop_token_ids(FakeTok())
        assert ids == [7]

    def test_handles_convert_exception(self):
        class FakeTok:
            eos_token_id = 7
            def convert_tokens_to_ids(self, m):
                raise RuntimeError("boom")
        ids = M._stop_token_ids(FakeTok())
        assert ids == [7]
