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
# 解析层: llama-cpp 的 tool_call.function.arguments 转 dict + <think> 剥离
# =================================================================

class TestParser:
    def test_arguments_json_string(self):
        assert M._parse_tool_arguments('{"path": "a.py"}') == {"path": "a.py"}

    def test_arguments_already_dict(self):
        assert M._parse_tool_arguments({"path": "a.py"}) == {"path": "a.py"}

    def test_arguments_empty_string(self):
        assert M._parse_tool_arguments("") == {}

    def test_arguments_invalid_json(self):
        assert M._parse_tool_arguments("not json") == {}

    def test_arguments_non_object_json(self):
        # 顶层是数组而非 object — 按契约应退化为空 dict
        assert M._parse_tool_arguments("[1,2,3]") == {}

    def test_split_think_strips_thought(self):
        thought, visible = M.split_think("<think>plan</think>actual reply")
        assert thought == "plan"
        assert visible == "actual reply"

    def test_no_think_block(self):
        thought, visible = M.split_think("just a reply")
        assert thought == ""
        assert visible == "just a reply"


# =================================================================
# apply_patch — 多文件 unified diff, 锁 + 回滚
# =================================================================

class TestApplyPatchParse:
    def test_modify_single_hunk(self):
        patch = (
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1,2 +1,2 @@\n"
            " keep\n"
            "-old\n"
            "+new\n"
        )
        files = M._parse_unified_diff(patch)
        assert len(files) == 1 and files[0].op == "modify"
        assert files[0].hunks[0].old_block == "keep\nold"
        assert files[0].hunks[0].new_block == "keep\nnew"

    def test_create_file(self):
        patch = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+line1\n"
            "+line2\n"
        )
        files = M._parse_unified_diff(patch)
        assert files[0].op == "create" and files[0].path == "new.py"
        assert files[0].hunks[0].new_block == "line1\nline2"

    def test_delete_file(self):
        patch = (
            "--- a/gone.py\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-bye\n"
        )
        files = M._parse_unified_diff(patch)
        assert files[0].op == "delete" and files[0].path == "gone.py"

    def test_no_header_raises(self):
        with pytest.raises(ValueError):
            M._parse_unified_diff("just some text\nnothing here\n")

    def test_missing_plus_line_raises(self):
        with pytest.raises(ValueError):
            M._parse_unified_diff("--- a/x.py\nno plus line follows\n")

    def test_blank_line_between_file_sections_is_separator(self):
        """模型常在两个文件段之间插空行 — 必须当分隔符而不是前一 hunk 的上下文."""
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
            "\n"  # <-- 裸空行分隔
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-y = 2\n"
            "+y = 3\n"
        )
        files = M._parse_unified_diff(patch)
        assert len(files) == 2
        assert files[0].path == "a.py" and files[0].hunks[0].old_block == "x = 1"
        assert files[1].path == "b.py" and files[1].hunks[0].old_block == "y = 2"

    def test_hunk_internal_blank_line_preserved(self):
        """hunk 内的真空行写作 ' ' (空格+空内容), 应保留为上下文."""
        patch = (
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ @@\n"
            " line1\n"
            " \n"        # <-- hunk 内真空行, 空格前缀
            "-old\n"
            "+new\n"
        )
        files = M._parse_unified_diff(patch)
        assert files[0].hunks[0].old_block == "line1\n\nold"


class TestApplyPatchModify:
    def test_single_file_multi_hunk(self, registry, workspace):
        _write(workspace / "a.py", "one\ntwo\nthree\nfour\nfive\n")
        registry.dispatch("Read", {"path": "a.py"})
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            " one\n"
            "-two\n"
            "+TWO\n"
            " three\n"
            "@@ @@\n"
            " four\n"
            "-five\n"
            "+FIVE\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "success", r.text
        assert (workspace / "a.py").read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\nFIVE\n"

    def test_multi_file(self, registry, workspace):
        _write(workspace / "a.py", "alpha\n")
        _write(workspace / "b.py", "beta\n")
        registry.dispatch("Read", {"path": "a.py"})
        registry.dispatch("Read", {"path": "b.py"})
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-alpha\n"
            "+ALPHA\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ @@\n"
            "-beta\n"
            "+BETA\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "success", r.text
        assert (workspace / "a.py").read_text(encoding="utf-8") == "ALPHA\n"
        assert (workspace / "b.py").read_text(encoding="utf-8") == "BETA\n"

    def test_real_model_style_patch_no_trailing_newline(self, registry, workspace):
        """E2E 回归: 复现真实 Ollama/Qwen 生成的 patch 格式 —
        文件内容无尾换行, 且两文件段之间插了空行. 之前会 HUNK_FAILED."""
        (workspace / "a.py").write_text("x = 1", encoding="utf-8")  # 无 \n
        (workspace / "b.py").write_text("y = 2", encoding="utf-8")
        registry.dispatch("Read", {"path": "a.py"})
        registry.dispatch("Read", {"path": "b.py"})
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
            "\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-y = 2\n"
            "+y = 3\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "success", r.text
        assert (workspace / "a.py").read_text(encoding="utf-8") == "x = 2"
        assert (workspace / "b.py").read_text(encoding="utf-8") == "y = 3"

    def test_context_mismatch_rolls_back_everything(self, registry, workspace):
        """第二个文件的 hunk 对不上 — 第一个文件也不能写."""
        _write(workspace / "a.py", "alpha\n")
        _write(workspace / "b.py", "beta\n")
        registry.dispatch("Read", {"path": "a.py"})
        registry.dispatch("Read", {"path": "b.py"})
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-alpha\n"
            "+ALPHA\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ @@\n"
            "-NO_SUCH_LINE\n"
            "+whatever\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "HUNK_FAILED"
        assert r.data["path"] == "b.py"
        # 关键: a.py 必须还是原样
        assert (workspace / "a.py").read_text(encoding="utf-8") == "alpha\n"

    def test_not_read_blocks_whole_patch(self, registry, workspace):
        _write(workspace / "a.py", "alpha\n")
        _write(workspace / "b.py", "beta\n")
        registry.dispatch("Read", {"path": "a.py"})  # 故意只 Read 一个
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-alpha\n"
            "+ALPHA\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ @@\n"
            "-beta\n"
            "+BETA\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "NOT_READ"
        assert r.data["path"] == "b.py"
        assert (workspace / "a.py").read_text(encoding="utf-8") == "alpha\n"

    def test_conflict_blocks_whole_patch(self, registry, workspace):
        path = workspace / "a.py"
        _write(path, "v1\n")
        registry.dispatch("Read", {"path": "a.py"})
        time.sleep(0.05)
        path.write_text("v1_extern\n", encoding="utf-8")
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-v1_extern\n"
            "+v2\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "CONFLICT"
        # 文件保持外部的样子, apply_patch 没动
        assert path.read_text(encoding="utf-8") == "v1_extern\n"


class TestApplyPatchCreateDelete:
    def test_create_new_file(self, registry, workspace):
        patch = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ @@\n"
            "+hello\n"
            "+world\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "success", r.text
        assert (workspace / "new.py").read_text(encoding="utf-8") == "hello\nworld"

    def test_create_existing_file_rejected(self, registry, workspace):
        _write(workspace / "existing.py", "x\n")
        patch = (
            "--- /dev/null\n"
            "+++ b/existing.py\n"
            "@@ @@\n"
            "+new content\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "ALREADY_EXISTS"

    def test_delete_file(self, registry, workspace):
        _write(workspace / "gone.py", "bye\n")
        registry.dispatch("Read", {"path": "gone.py"})
        patch = (
            "--- a/gone.py\n"
            "+++ /dev/null\n"
            "@@ @@\n"
            "-bye\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "success", r.text
        assert not (workspace / "gone.py").exists()

    def test_post_success_allows_subsequent_edit(self, registry, workspace):
        """apply_patch 成功后 cache 应已刷新, edit_file 不需要再 Read."""
        _write(workspace / "a.py", "v1\n")
        registry.dispatch("Read", {"path": "a.py"})
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-v1\n"
            "+v2\n"
        )
        registry.dispatch("apply_patch", {"patch": patch})
        r = registry.dispatch("edit_file", {"path": "a.py", "old_text": "v2", "new_text": "v3"})
        assert r.status == "success"


class TestApplyPatchErrors:
    def test_parse_failed(self, registry):
        r = registry.dispatch("apply_patch", {"patch": "not a diff\n"})
        assert r.status == "error"
        assert r.data["code"] == "PARSE_FAILED"

    def test_path_escape(self, registry):
        patch = (
            "--- a/../etc/passwd\n"
            "+++ b/../etc/passwd\n"
            "@@ @@\n"
            "-x\n"
            "+y\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "PATH_ESCAPE"

    def test_modify_missing_file(self, registry):
        patch = (
            "--- a/ghost.py\n"
            "+++ b/ghost.py\n"
            "@@ @@\n"
            "-x\n"
            "+y\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "NOT_FOUND"

    def test_hunk_context_ambiguous(self, registry, workspace):
        """old_block 在文件里出现多次 — 模型给的上下文不够区分, 报 HUNK_FAILED."""
        _write(workspace / "a.py", "x\ny\nx\n")
        registry.dispatch("Read", {"path": "a.py"})
        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-x\n"
            "+X\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "HUNK_FAILED"
        assert "multiple places" in r.data["message"]
        # 文件未被修改
        assert (workspace / "a.py").read_text(encoding="utf-8") == "x\ny\nx\n"

    def test_disk_write_failure_rolls_back(self, registry, workspace, monkeypatch):
        """Phase 3 中途写盘失败: 第一个文件已写, 第二个文件抛 OSError -> 第一个必须回滚."""
        _write(workspace / "a.py", "alpha\n")
        _write(workspace / "b.py", "beta\n")
        registry.dispatch("Read", {"path": "a.py"})
        registry.dispatch("Read", {"path": "b.py"})

        # 让 b.py 的 write_text 抛 OSError, a.py 正常走.
        original_write_text = Path.write_text

        def fake_write_text(self, data, *args, **kwargs):
            if self.name == "b.py":
                raise OSError("simulated: disk full")
            return original_write_text(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fake_write_text)

        patch = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ @@\n"
            "-alpha\n"
            "+ALPHA\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ @@\n"
            "-beta\n"
            "+BETA\n"
        )
        r = registry.dispatch("apply_patch", {"patch": patch})
        assert r.status == "error"
        assert r.data["code"] == "WRITE_FAILED"
        # 关键: a.py 已经被写入 "ALPHA\n", 但回滚必须把它恢复成 "alpha\n"
        assert (workspace / "a.py").read_text(encoding="utf-8") == "alpha\n"
        assert (workspace / "b.py").read_text(encoding="utf-8") == "beta\n"


# =================================================================
# spawn_agent (v6 新增) — 子 agent 在隔离 git worktree 跑探索
# =================================================================

@pytest.fixture
def git_workspace(tmp_path, monkeypatch):
    """带 git init 的 workspace — spawn_agent 要求 git repo."""
    import subprocess
    workdir = tmp_path.resolve()
    monkeypatch.setattr(M, "WORKDIR", workdir)
    # 初始化 git repo + 至少一个 commit (worktree 要求)
    subprocess.run(["git", "init"], cwd=workdir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=workdir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=workdir, capture_output=True, check=True)
    (workdir / "seed.txt").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=workdir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=workdir, capture_output=True, check=True)
    return workdir


class _FakeSubAgentLLM:
    """模拟子 agent 的 LLM — 返回预设的 tool_calls 序列, 然后说 "DONE"."""

    def __init__(self, scripted_calls):
        """scripted_calls: list of (name, args_dict) — 子 agent 按顺序调用这些工具.
        最后一轮无 tool_call, content 设为 'DONE — done.' 触发 ReActAgent DONE 退出.
        """
        self.scripted = list(scripted_calls)
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}

    def chat(self, messages, tools):
        if self.scripted:
            name, args = self.scripted.pop(0)
            return {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{name}", "type": "function",
                    "function": {"name": name, "arguments": __import__("json").dumps(args)},
                }],
            }
        return {"role": "assistant", "content": "DONE — sub-agent finished.", "tool_calls": []}

    def chat_stream(self, messages, tools):
        # 测试不走流式; 但 ReActAgent 在 USE_STREAM=True 时会调它. 所以也实现一下,
        # yield 同样的内容做成流式形态.
        msg = self.chat(messages, tools)
        if msg.get("content"):
            yield {"content": msg["content"]}
        for tc in (msg.get("tool_calls") or []):
            yield {"tool_calls": [{"index": 0, **tc}]}


class TestSpawnAgent:
    def test_requires_git_repo(self, monkeypatch, registry, workspace):
        """非 git 目录 — spawn_agent 应当 WORKTREE_FAILED."""
        # 必须先设 _SPAWN_LLM 否则会先报 SPAWN_NOT_AVAILABLE
        monkeypatch.setattr(M, "_SPAWN_LLM", _FakeSubAgentLLM([]))
        # workspace fixture 不带 git init
        r = registry.dispatch("spawn_agent", {"task": "anything"})
        assert r.status == "error"
        assert r.data["code"] == "WORKTREE_FAILED"
        assert "git" in r.data["message"].lower()

    def test_requires_spawn_llm_set(self, monkeypatch, git_workspace):
        """_SPAWN_LLM 没设, 应当 SPAWN_NOT_AVAILABLE."""
        monkeypatch.setattr(M, "_SPAWN_LLM", None)
        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)
        r = registry.dispatch("spawn_agent", {"task": "anything"})
        assert r.status == "error"
        assert r.data["code"] == "SPAWN_NOT_AVAILABLE"

    def test_subagent_changes_extracted_as_diff(self, monkeypatch, git_workspace):
        """子 agent 创建一个文件 → spawn_agent 返回应当包含含该文件的 diff."""
        # 子 agent 脚本: 创建 new.txt
        fake = _FakeSubAgentLLM([
            ("write_file", {"path": "new.txt", "content": "from sub-agent\n"}),
        ])
        monkeypatch.setattr(M, "_SPAWN_LLM", fake)
        monkeypatch.setattr(M, "USE_STREAM", False)  # 测试走非流式分支

        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)
        r = registry.dispatch("spawn_agent", {"task": "create new.txt with 'from sub-agent'"})

        assert r.status == "success", f"got error: {r.text}"
        assert "diff" in r.data
        diff = r.data["diff"]
        assert "new.txt" in diff
        assert "from sub-agent" in diff
        # 主 workspace 应该没动 (只有 seed.txt)
        assert (git_workspace / "new.txt").exists() is False
        assert (git_workspace / "seed.txt").read_text() == "hello\n"

    def test_subagent_failure_keeps_worktree(self, monkeypatch, git_workspace):
        """子 agent 报告 FAIL 时 worktree 应当保留供调试."""
        # 让子 agent 只跑一轮就说 FAIL (无 tool_call, content 含 FAIL)
        class FailLLM:
            last_usage = None
            def chat(self, m, t):
                return {"role": "assistant", "content": "FAIL — could not determine root cause.",
                        "tool_calls": []}
            def chat_stream(self, m, t):
                yield {"content": "FAIL — could not determine root cause."}
        monkeypatch.setattr(M, "_SPAWN_LLM", FailLLM())
        monkeypatch.setattr(M, "USE_STREAM", False)

        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)
        r = registry.dispatch("spawn_agent", {"task": "do something that fails"})

        assert r.status == "success"  # tool 本身成功执行了, 子任务失败信息在 text
        assert "FAIL" in r.text
        # 失败 → worktree 路径应该被保留
        assert r.data["worktree"] is not None
        assert Path(r.data["worktree"]).exists()
        # 清理 (测试结尾手工清, 避免 /tmp 累积)
        import shutil as _sh
        _sh.rmtree(r.data["worktree"], ignore_errors=True)

    def test_subagent_success_removes_worktree(self, monkeypatch, git_workspace):
        """子 agent 成功 (无 FAIL/ERROR/GAVE UP 关键字) → worktree 应当清掉."""
        fake = _FakeSubAgentLLM([
            ("write_file", {"path": "ok.txt", "content": "ok\n"}),
        ])
        monkeypatch.setattr(M, "_SPAWN_LLM", fake)
        monkeypatch.setattr(M, "USE_STREAM", False)

        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)
        r = registry.dispatch("spawn_agent", {"task": "ok task"})

        assert r.status == "success"
        # 成功 → worktree 应该被清掉, data["worktree"] = None
        assert r.data["worktree"] is None

    def test_main_workdir_restored_after_spawn(self, monkeypatch, git_workspace):
        """spawn_agent 跑完, 主 WORKDIR 必须恢复 — 不能停留在 worktree."""
        fake = _FakeSubAgentLLM([
            ("write_file", {"path": "x.txt", "content": "x\n"}),
        ])
        monkeypatch.setattr(M, "_SPAWN_LLM", fake)
        monkeypatch.setattr(M, "USE_STREAM", False)

        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)

        before = M.WORKDIR
        r = registry.dispatch("spawn_agent", {"task": "task"})
        after = M.WORKDIR

        assert before == after, f"WORKDIR shifted: {before} -> {after}"
        assert r.status == "success"

    def test_subagent_tokens_accumulate_to_main_session(self, monkeypatch, git_workspace):
        """子 agent 烧的 token 应该累计到主 session — 成本可见性."""
        fake = _FakeSubAgentLLM([
            ("write_file", {"path": "tk.txt", "content": "x\n"}),
        ])
        monkeypatch.setattr(M, "_SPAWN_LLM", fake)
        monkeypatch.setattr(M, "USE_STREAM", False)

        terminal = M.TerminalTool(git_workspace)
        # 用一个我们能拿到 session 引用的 registry
        main_session = M.Session.new("test main")
        registry = M.build_default_registry(terminal, session=main_session)

        before_in = main_session.prompt_tokens_total
        before_out = main_session.completion_tokens_total
        r = registry.dispatch("spawn_agent", {"task": "task"})
        assert r.status == "success"

        # 子 agent 跑了 >= 2 轮, 必然有 token 消耗. 估算路径下数字小, 但必然 > 0.
        # 关键断言: 主 session 累加了子 agent 的 token, 不为 0.
        assert main_session.prompt_tokens_total > before_in, \
            "main session prompt_tokens_total didn't accumulate sub-agent's"
        assert main_session.completion_tokens_total > before_out, \
            "main session completion_tokens_total didn't accumulate sub-agent's"
        # data["sub_tokens"] 应该跟累加的差额一致
        sub_in, sub_out = r.data["sub_tokens"]
        assert main_session.prompt_tokens_total - before_in == sub_in
        assert main_session.completion_tokens_total - before_out == sub_out

    def test_diff_excludes_pyc_and_pycache(self, monkeypatch, git_workspace):
        """子 agent 写 *.pyc 和 __pycache__/ → spawn_agent 返回 diff 不该包含它们."""
        # 子 agent 脚本: 写一个 fake .pyc + 一个 .pytest_cache 文件 + 一个真改动 (real.txt)
        fake = _FakeSubAgentLLM([
            ("write_file", {"path": "real.txt", "content": "real change\n"}),
            ("write_file", {"path": "__pycache__/foo.cpython-312.pyc", "content": "fake bytecode\n"}),
            ("write_file", {"path": ".pytest_cache/CACHEDIR.TAG", "content": "Signature: ...\n"}),
        ])
        monkeypatch.setattr(M, "_SPAWN_LLM", fake)
        monkeypatch.setattr(M, "USE_STREAM", False)

        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)
        r = registry.dispatch("spawn_agent", {"task": "make changes including pyc and pycache"})

        assert r.status == "success", f"got error: {r.text}"
        diff = r.data["diff"]
        # 真改动应该在 diff 里
        assert "real.txt" in diff, "real change should appear in diff"
        # 但 .pyc / __pycache__ / .pytest_cache 不应该
        assert "__pycache__" not in diff, f"__pycache__ should be excluded:\n{diff}"
        assert ".pyc" not in diff, f".pyc files should be excluded:\n{diff}"
        assert ".pytest_cache" not in diff, f".pytest_cache should be excluded:\n{diff}"

    def test_error_message_includes_gitignore_template(self, monkeypatch, registry, workspace):
        """non-git workspace 的错误消息应当含 .gitignore 安全引导."""
        monkeypatch.setattr(M, "_SPAWN_LLM", _FakeSubAgentLLM([]))
        r = registry.dispatch("spawn_agent", {"task": "anything"})
        assert r.status == "error"
        assert r.data["code"] == "WORKTREE_FAILED"
        msg = r.data["message"]
        # 错误消息应当: (a) 教用户跑 git init, (b) 含 .gitignore 模板, (c) 警告 .env 风险
        assert "git init" in msg, "error msg should suggest git init"
        assert ".gitignore" in msg, "error msg should include .gitignore step"
        assert ".env" in msg, "error msg should warn about .env"
        assert "__pycache__" in msg or "pyc" in msg, "error msg should mention build artifacts"

    def test_spawn_agent_does_not_modify_gitignore(self, monkeypatch, git_workspace):
        """spawn_agent 不该往 worktree 的 .gitignore 写东西.

        之前实测发现: 老版本会在 worktree 的 .gitignore 追加 exclude patterns,
        但那个追加本身是改动, 出现在 diff 里 — 用户看到一段跟任务无关的
        .gitignore 修改, 真烦. 修法: 改用 pathspec exclude, 完全不动磁盘文件.
        """
        # 让主 workspace 已经有一个 .gitignore (常见场景)
        gi = git_workspace / ".gitignore"
        gi.write_text("# user's existing gitignore\n*.log\n")
        import subprocess as _sp
        _sp.run(["git", "add", ".gitignore"], cwd=git_workspace, capture_output=True, check=True)
        _sp.run(["git", "commit", "-m", "add gitignore"], cwd=git_workspace, capture_output=True, check=True)

        # 子 agent 只做一个真改动
        fake = _FakeSubAgentLLM([
            ("write_file", {"path": "x.txt", "content": "x\n"}),
        ])
        monkeypatch.setattr(M, "_SPAWN_LLM", fake)
        monkeypatch.setattr(M, "USE_STREAM", False)

        terminal = M.TerminalTool(git_workspace)
        registry = M.build_default_registry(terminal)
        r = registry.dispatch("spawn_agent", {"task": "make x.txt"})

        assert r.status == "success", f"got error: {r.text}"
        diff = r.data["diff"]
        # 真改动应当在
        assert "x.txt" in diff
        # .gitignore 不应当出现在 diff 里 (我们没动它)
        assert ".gitignore" not in diff, \
            f".gitignore should NOT appear in diff (we use pathspec, not file edits):\n{diff}"


# =================================================================
# Session persistence (v7 新增) — to_dict / from_dict / round-trip
# =================================================================

class TestSessionPersistence:
    def test_round_trip_empty_session(self):
        """空 session (只有 system message) 序列化-反序列化."""
        s = M.Session.new("test system")
        d = s.to_dict()
        loaded = M.Session.from_dict(d)
        assert len(loaded.history) == 1
        assert loaded.history[0].role == "system"
        assert loaded.history[0].content == "test system"
        assert loaded.todo.items == []
        assert loaded.prompt_tokens_total == 0
        assert loaded.completion_tokens_total == 0
        # read_cache 永远是新空的, 不持久化
        assert isinstance(loaded.read_cache, M.ReadCache)
        # is_subagent 总是 False
        assert loaded.is_subagent is False

    def test_round_trip_with_history_and_todos(self):
        """有 user/assistant/tool 三种消息 + todos + token 累计."""
        s = M.Session.new("sys")
        s.history.append(M.Message.user("do X"))
        s.history.append(M.Message.assistant(
            "ok", tool_calls=[{"id": "c1", "type": "function",
                                "function": {"name": "LS", "arguments": "{}"}}]
        ))
        s.history.append(M.Message.tool("3 entries", tool_call_id="c1"))
        s.todo.items = [{"id": "1", "text": "task one", "status": "in_progress"}]
        s.prompt_tokens_total = 1234
        s.completion_tokens_total = 567

        d = s.to_dict()
        # JSON serializable check
        json_str = __import__("json").dumps(d, ensure_ascii=False)
        assert "do X" in json_str
        assert "task one" in json_str

        loaded = M.Session.from_dict(d)
        assert len(loaded.history) == 4
        assert loaded.history[1].role == "user"
        assert loaded.history[1].content == "do X"
        assert loaded.history[2].role == "assistant"
        assert loaded.history[2].tool_calls is not None
        assert loaded.history[2].tool_calls[0]["function"]["name"] == "LS"
        assert loaded.history[3].role == "tool"
        assert loaded.history[3].tool_call_id == "c1"
        assert loaded.todo.items == [{"id": "1", "text": "task one", "status": "in_progress"}]
        assert loaded.prompt_tokens_total == 1234
        assert loaded.completion_tokens_total == 567

    def test_read_cache_not_persisted(self):
        """read_cache 不该跟 session 一起存. load 后必须是空的."""
        s = M.Session.new("sys")
        # 模拟一些 read_cache 记录
        from pathlib import Path as _P
        s.read_cache._entries[str(_P("/some/file"))] = (12345, 100)
        d = s.to_dict()
        # to_dict 不该含 read_cache
        assert "read_cache" not in d
        loaded = M.Session.from_dict(d)
        assert loaded.read_cache._entries == {}, \
            "loaded session's read_cache should be empty (forces re-read)"

    def test_from_dict_rejects_unknown_version(self):
        """schema 版本不识别 → 抛 RuntimeError, 不静默吃."""
        import pytest as _pt
        with _pt.raises(RuntimeError, match="version"):
            M.Session.from_dict({"version": 999, "history": []})

    def test_save_load_round_trip_via_filesystem(self, tmp_path, monkeypatch):
        """完整链路: to_dict → JSON file → 读回 → from_dict."""
        import json as _json
        s = M.Session.new("sys")
        s.history.append(M.Message.user("hello"))
        s.completion_tokens_total = 42

        f = tmp_path / "test.json"
        f.write_text(_json.dumps(s.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")

        # 读回
        loaded = M.Session.from_dict(_json.loads(f.read_text(encoding="utf-8")))
        assert len(loaded.history) == 2
        assert loaded.history[1].content == "hello"
        assert loaded.completion_tokens_total == 42

    def test_validate_session_name(self):
        """文件名安全检查 — 防路径穿越."""
        # 合法
        assert M._validate_session_name("foo") is None
        assert M._validate_session_name("my-task_1") is None
        assert M._validate_session_name("a") is None
        # 非法
        assert M._validate_session_name("") is not None
        assert M._validate_session_name("a/b") is not None  # 路径
        assert M._validate_session_name("../etc") is not None
        assert M._validate_session_name("a b") is not None  # 空格
        assert M._validate_session_name("a.json") is not None  # 不该带扩展
        assert M._validate_session_name("x" * 100) is not None  # 太长


# =================================================================
# auto-save (v7 第二轮新增) — 限速 / force / 保留名 / 真文件 round-trip
# =================================================================

class TestAutoSave:
    @pytest.fixture
    def auto_workspace(self, tmp_path, monkeypatch):
        """每个测试一个干净的 SESSIONS_DIR + 重置 _LAST_AUTOSAVE_AT."""
        sd = tmp_path / "sessions"
        monkeypatch.setattr(M, "SESSIONS_DIR", sd)
        monkeypatch.setattr(M, "_LAST_AUTOSAVE_AT", None)
        return sd

    def _make_agent(self, tmp_path, monkeypatch):
        """造一个 ReActAgent 用于测试. fake LLM 不会被调用."""
        monkeypatch.setattr(M, "WORKDIR", tmp_path.resolve())
        terminal = M.TerminalTool(tmp_path.resolve())
        registry = M.build_default_registry(terminal)
        return M.ReActAgent(llm=None, registry=registry, system_prompt="test")

    def test_autosave_skips_empty_session(self, tmp_path, monkeypatch, auto_workspace):
        """history 只有 system message 时不该写盘."""
        agent = self._make_agent(tmp_path, monkeypatch)
        # 此时 history = [system message] 长度 1
        M._autosave(agent, force=True)
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        assert not autosave_file.exists(), "empty session should not be auto-saved"

    def test_autosave_writes_when_history_exists(self, tmp_path, monkeypatch, auto_workspace):
        """history 有内容时, force=True 写盘."""
        agent = self._make_agent(tmp_path, monkeypatch)
        agent.session.history.append(M.Message.user("hello"))
        agent.session.history.append(M.Message.assistant("hi back"))
        M._autosave(agent, force=True)
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        assert autosave_file.exists()
        # 文件内容是合法 JSON 且含我们写的消息
        d = __import__("json").loads(autosave_file.read_text(encoding="utf-8"))
        assert d["version"] == 1
        contents = [m.get("content", "") for m in d["history"]]
        assert "hello" in contents
        assert "hi back" in contents

    def test_autosave_skips_subagent(self, tmp_path, monkeypatch, auto_workspace):
        """is_subagent=True 的 session 不该被 auto-save."""
        agent = self._make_agent(tmp_path, monkeypatch)
        # 强制把 session 改成子 agent
        agent.session.is_subagent = True
        agent.session.history.append(M.Message.user("sub task"))
        M._autosave(agent, force=True)
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        assert not autosave_file.exists()

    def test_autosave_rate_limited(self, tmp_path, monkeypatch, auto_workspace):
        """两次连续 auto-save (force=False) 间隔 < AUTOSAVE_MIN_INTERVAL 应当跳过第二次."""
        agent = self._make_agent(tmp_path, monkeypatch)
        agent.session.history.append(M.Message.user("first"))
        # 第一次写
        M._autosave(agent, force=False)
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        assert autosave_file.exists()
        first_mtime = autosave_file.stat().st_mtime_ns

        # 改 history 但立即再 autosave (无 force) — 应该被限速跳过
        agent.session.history.append(M.Message.assistant("more"))
        import time as _time
        _time.sleep(0.05)  # 让 mtime 有机会变 (如果真写的话)
        M._autosave(agent, force=False)
        # mtime 应该没变 (没真写)
        assert autosave_file.stat().st_mtime_ns == first_mtime

    def test_autosave_force_bypasses_rate_limit(self, tmp_path, monkeypatch, auto_workspace):
        """force=True 应当无视限速."""
        agent = self._make_agent(tmp_path, monkeypatch)
        agent.session.history.append(M.Message.user("first"))
        M._autosave(agent, force=False)
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        first_mtime = autosave_file.stat().st_mtime_ns

        agent.session.history.append(M.Message.assistant("more"))
        import time as _time
        _time.sleep(0.05)
        M._autosave(agent, force=True)  # force 跳过限速
        # 应该真写了, mtime 变了
        assert autosave_file.stat().st_mtime_ns != first_mtime

    def test_save_rejects_reserved_autosave_name(self, tmp_path, monkeypatch, auto_workspace, capsys):
        """用户 /save _autosave 应当被拒, 不会污染 auto-save slot."""
        agent = self._make_agent(tmp_path, monkeypatch)
        agent.session.history.append(M.Message.user("real session"))

        M._cmd_save(agent, M.AUTOSAVE_NAME)
        captured = capsys.readouterr()
        # 应当打印 reserved 错误
        assert "reserved" in captured.out.lower() or "reserved" in captured.err.lower()
        # 文件不该存在
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        assert not autosave_file.exists()

    def test_full_round_trip_via_real_file(self, tmp_path, monkeypatch, auto_workspace):
        """端到端: agent 跑了一些 turn → auto-save → 新 agent /load → history 完全一致.

        这是 v7 'session 真能跨进程恢复'的 ground-truth 测试.
        模拟"REPL → Ctrl-C → 重启 → /load _autosave → 继续"的全流程.
        """
        # 第一阶段: 模拟 REPL 跑了几轮, 然后 force-save (相当于 Ctrl-C 触发 auto-save)
        agent1 = self._make_agent(tmp_path, monkeypatch)
        agent1.session.history.append(M.Message.user("explain X"))
        agent1.session.history.append(M.Message.assistant(
            "X is...", tool_calls=[{"id": "c1", "type": "function",
                                     "function": {"name": "Read", "arguments": '{"path":"foo.py"}'}}]
        ))
        agent1.session.history.append(M.Message.tool("foo.py contents", tool_call_id="c1"))
        agent1.session.todo.items = [
            {"id": "1", "text": "explain X", "status": "completed"},
            {"id": "2", "text": "explain Y", "status": "in_progress"},
        ]
        agent1.session.prompt_tokens_total = 1500
        agent1.session.completion_tokens_total = 300
        M._autosave(agent1, force=True)
        autosave_file = auto_workspace / f"{M.AUTOSAVE_NAME}.json"
        assert autosave_file.exists()

        # 第二阶段: 模拟新进程启动 — 创建全新 agent, /load _autosave
        # (实际 REPL 走 _cmd_load, 我们直接调它确认逻辑链路)
        agent2 = self._make_agent(tmp_path, monkeypatch)
        # 新 agent 的 history 只有 system, 跟 agent1 不同
        assert len(agent2.session.history) == 1
        assert agent2.session.completion_tokens_total == 0

        # 调 /load
        import io as _io, sys as _sys
        old_stdout = _sys.stdout
        _sys.stdout = _io.StringIO()
        try:
            M._cmd_load(agent2, M.AUTOSAVE_NAME)
        finally:
            _sys.stdout = old_stdout

        # 验证: agent2 的 session 现在跟 agent1 当时一致
        assert len(agent2.session.history) == 4, "history not restored"
        # 检查内容字段
        msgs = [(m.role, m.content[:30]) for m in agent2.session.history]
        assert ("user", "explain X") in msgs
        assert ("tool", "foo.py contents") in msgs
        # tool_calls 也要恢复
        assistant_msg = agent2.session.history[2]
        assert assistant_msg.tool_calls is not None
        assert assistant_msg.tool_calls[0]["function"]["name"] == "Read"
        # tool message 的 tool_call_id 也要保留
        tool_msg = agent2.session.history[3]
        assert tool_msg.tool_call_id == "c1"
        # todos 也要恢复
        assert len(agent2.session.todo.items) == 2
        assert agent2.session.todo.items[1]["status"] == "in_progress"
        # token 累计要保留
        assert agent2.session.prompt_tokens_total == 1500
        assert agent2.session.completion_tokens_total == 300
        # read_cache 必须是空的 (不持久化)
        assert agent2.session.read_cache._entries == {}


# =================================================================
# Context 压缩 (v8) — 切分边界 / LLM 总结 / 机械兜底 / 自动触发
# =================================================================


class _FakeLLM:
    """最小可用的 LLMClient 替身 — 单测压缩逻辑用.

    压缩调的是 chat_stream (跟主 REPL 同路径, 非流式在 minimax 上冷). chat()
    留着是为了 SubAgent / 主 step 可能会调到 — 单测里也别去掉.
    """

    def __init__(self, response_content: str = "fake summary of prior conversation"):
        self.response_content = response_content
        self.calls: list[list[dict]] = []
        self.should_fail = False

    def chat(self, messages, tools):
        self.calls.append(messages)
        if self.should_fail:
            raise RuntimeError("simulated llm failure")
        return {"role": "assistant", "content": self.response_content}

    def chat_stream(self, messages, tools):
        self.calls.append(messages)
        if self.should_fail:
            raise RuntimeError("simulated llm failure")
        # 把 response_content 一次性吐出来 — 真后端会切成多个 delta, 我们的
        # 收集逻辑无所谓 chunk 数, 1 个就够测.
        yield {"content": self.response_content}


def _make_history(*pairs: tuple[str, str, dict | None]) -> list[M.Message]:
    """方便地构造 history: list of (role, content, tool_call_id_or_extra).

    这个 helper 直接返回 Message 列表, 调用方按需要自己拼.
    """
    out = []
    for role, content, extra in pairs:
        if role == "system":
            out.append(M.Message.system(content))
        elif role == "user":
            out.append(M.Message.user(content))
        elif role == "assistant":
            tcs = (extra or {}).get("tool_calls")
            out.append(M.Message.assistant(content, tool_calls=tcs))
        elif role == "tool":
            tcid = (extra or {}).get("tool_call_id")
            out.append(M.Message.tool(content, tool_call_id=tcid))
    return out


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": __import__("json").dumps(args)},
    }


class TestFindCompressionSplit:
    """切分点选择 — 确保不会把 tool_call/tool_result 配对拆开."""

    def test_too_short_returns_none(self):
        h = _make_history(
            ("system", "sys", None),
            ("user", "hi", None),
            ("assistant", "hello", None),
        )
        assert M._find_compression_split(h, keep_tail=10) is None

    def test_split_avoids_orphan_tool_result(self):
        """初始 split 落在一条 tool 消息上 — 必须往前推, 让 tool_call/result 都进尾部."""
        # 12 条: system, 然后 5 个完整轮 (user + assistant_with_tc + tool)
        # keep_tail=4 会让 split 落在某个位置 — 我们构造让它正好落在 tool 上
        tc = _make_tool_call("c1", "Read", {"path": "x"})
        h = _make_history(
            ("system", "sys", None),
            ("user", "q1", None),
            ("assistant", "", {"tool_calls": [tc]}),
            ("tool", "result1", {"tool_call_id": "c1"}),
            ("user", "q2", None),
            ("assistant", "", {"tool_calls": [_make_tool_call("c2", "Read", {"path": "y"})]}),
            ("tool", "result2", {"tool_call_id": "c2"}),
            ("user", "q3", None),
            ("assistant", "", {"tool_calls": [_make_tool_call("c3", "Read", {"path": "z"})]}),
            ("tool", "result3", {"tool_call_id": "c3"}),
        )
        # n=10, keep_tail=4 → 初始 split=6, 但 history[6]=assistant_with_tcs
        # 紧跟着 tool_result, 切在 6 会把 assistant 留尾、用户 q2 进摘要 — 没破坏配对.
        # 不过 history[5] 是 tool — split=5 会把 result2 留下但 assistant_with_tc 进摘要.
        # 我们直接验证: 返回的 split 处不是 tool, 且 split-1 处不是带 tool_calls 的 assistant.
        split = M._find_compression_split(h, keep_tail=4)
        assert split is not None
        assert h[split].role != "tool", "split must not start at a tool message"
        prev = h[split - 1]
        assert not (prev.role == "assistant" and prev.tool_calls), \
            "split must not strand a tool_call from its results"

    def test_split_returns_none_when_only_system_left(self):
        # 大 keep_tail 让 split 推到 1 之前
        h = _make_history(
            ("system", "sys", None),
            ("user", "u1", None),
            ("assistant", "a1", None),
        )
        assert M._find_compression_split(h, keep_tail=100) is None


class TestCompressHistoryLLM:
    """LLM 总结路径 — 看 fake LLM 被调用 + history 形态正确."""

    def test_compress_inserts_marker_message(self):
        h = [M.Message.system("sys")]
        for i in range(20):
            h.append(M.Message.user(f"q{i}"))
            h.append(M.Message.assistant(f"a{i}"))
        llm = _FakeLLM("brief summary text")
        result = M._compress_history(h, llm, todo_items=None, keep_tail=10)
        assert result is not None
        new_h, stats = result
        assert stats["used_llm"] is True
        assert stats["before_count"] == len(h)
        assert stats["after_count"] == len(new_h)
        assert new_h[0].role == "system"
        assert new_h[1].role == "user"
        assert M._COMPRESSION_MARKER in new_h[1].content
        assert M._COMPRESSION_MARKER_END in new_h[1].content
        assert "brief summary text" in new_h[1].content
        # 尾部最近 10 条原样保留
        assert new_h[-1].content == h[-1].content
        assert new_h[-2].content == h[-2].content
        # 调了一次 LLM
        assert len(llm.calls) == 1

    def test_compress_includes_outstanding_todos(self):
        h = [M.Message.system("sys")]
        for i in range(15):
            h.append(M.Message.user(f"q{i}"))
            h.append(M.Message.assistant(f"a{i}"))
        llm = _FakeLLM("summary")
        todos = [
            {"id": "1", "text": "do thing A", "status": "completed"},
            {"id": "2", "text": "do thing B", "status": "in_progress"},
            {"id": "3", "text": "do thing C", "status": "pending"},
        ]
        result = M._compress_history(h, llm, todo_items=todos, keep_tail=8)
        assert result is not None
        new_h, _ = result
        marker_msg = new_h[1].content
        # in_progress / pending 项要被塞进摘要
        assert "do thing B" in marker_msg
        assert "do thing C" in marker_msg
        # completed 项不必出现
        assert "do thing A" not in marker_msg

    def test_compress_empty_or_no_system_returns_none(self):
        # 没 system header 直接拒
        h = [M.Message.user("u")]
        assert M._compress_history(h, _FakeLLM(), keep_tail=10) is None
        # 空也拒
        assert M._compress_history([], _FakeLLM(), keep_tail=10) is None


class TestCompressHistoryFallback:
    """LLM 调用失败时降级到机械裁剪."""

    def test_llm_failure_falls_back_to_mechanical(self):
        h = [M.Message.system("sys")]
        h.append(M.Message.user("Original task: explain X"))
        for i in range(15):
            tc = _make_tool_call(f"c{i}", "Read", {"path": f"file{i}.py"})
            h.append(M.Message.assistant("", tool_calls=[tc]))
            h.append(M.Message.tool(f"contents of file{i}", tool_call_id=f"c{i}"))
        llm = _FakeLLM()
        llm.should_fail = True
        result = M._compress_history(h, llm, keep_tail=8)
        assert result is not None
        new_h, stats = result
        assert stats["used_llm"] is False
        marker = new_h[1].content
        assert M._COMPRESSION_MARKER in marker
        assert "mechanical" in marker.lower()
        # 用户原始 query 应当出现 (机械摘要会抠 user 消息)
        assert "Original task: explain X" in marker
        # Tool 计数应当出现
        assert "Read" in marker

    def test_no_llm_uses_mechanical_directly(self):
        """llm=None 时不调用, 直接走机械路径."""
        h = [M.Message.system("sys")]
        h.append(M.Message.user("first request"))
        for i in range(12):
            h.append(M.Message.assistant(f"a{i}"))
            h.append(M.Message.user(f"q{i}"))
        result = M._compress_history(h, llm=None, keep_tail=6)
        assert result is not None
        _, stats = result
        assert stats["used_llm"] is False


class TestSplitNeverBreaksToolCallPair:
    """属性式: 不管 history 长什么样, 压缩后 history 里 tool_call 一定能配上 tool_result."""

    def test_pairs_preserved_after_compression(self):
        h = [M.Message.system("sys")]
        for i in range(12):
            tc = _make_tool_call(f"c{i}", "Read", {"path": f"f{i}.py"})
            h.append(M.Message.user(f"q{i}"))
            h.append(M.Message.assistant("calling", tool_calls=[tc]))
            h.append(M.Message.tool(f"r{i}", tool_call_id=f"c{i}"))
        result = M._compress_history(h, _FakeLLM(), keep_tail=10)
        assert result is not None
        new_h, _ = result
        # 收集尾部所有 tool_call_id 和 tool_result id
        call_ids: set[str] = set()
        result_ids: set[str] = set()
        for m in new_h:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("id"):
                        call_ids.add(tc["id"])
            if m.role == "tool" and m.tool_call_id:
                result_ids.add(m.tool_call_id)
        # 每个保留的 tool_call 都应该有对应 tool_result, 反之亦然
        assert call_ids == result_ids


class TestCompressInAgent:
    """ReActAgent.compress_now / _maybe_compress_before_turn 行为."""

    def _make_agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(M, "WORKDIR", tmp_path.resolve())
        terminal = M.TerminalTool(tmp_path.resolve())
        registry = M.build_default_registry(terminal)
        # llm 给个 fake — _maybe_compress 触发时会真去调
        return M.ReActAgent(llm=_FakeLLM("agent-side summary"),
                            registry=registry, system_prompt="test sys")

    def test_compress_now_short_history_noop(self, tmp_path, monkeypatch):
        agent = self._make_agent(tmp_path, monkeypatch)
        agent.session.history.append(M.Message.user("hi"))
        before = len(agent.session.history)
        stats = agent.compress_now(verbose=False)
        assert stats is None
        assert len(agent.session.history) == before

    def test_compress_now_long_history_runs(self, tmp_path, monkeypatch):
        agent = self._make_agent(tmp_path, monkeypatch)
        for i in range(20):
            agent.session.history.append(M.Message.user(f"q{i}"))
            agent.session.history.append(M.Message.assistant(f"a{i}"))
        before = len(agent.session.history)
        stats = agent.compress_now(verbose=False)
        assert stats is not None
        assert stats["used_llm"] is True
        assert len(agent.session.history) < before
        # system 仍在头部
        assert agent.session.history[0].role == "system"
        assert agent.session.history[0].content == "test sys"

    def test_should_compress_threshold(self, tmp_path, monkeypatch):
        agent = self._make_agent(tmp_path, monkeypatch)
        # 没跑过 turn — _last_usage 不存在, 不该压
        assert agent._should_compress() is False
        # 设上一次 prompt 数远低于阈值
        agent._last_usage = (100, 50, True)
        assert agent._should_compress() is False
        # 设到阈值之上 — CONTEXT_LIMIT * COMPRESS_AT
        threshold = int(M.CONTEXT_LIMIT * M.COMPRESS_AT)
        agent._last_usage = (threshold + 1, 50, True)
        assert agent._should_compress() is True

    def test_should_compress_skips_subagent(self, tmp_path, monkeypatch):
        agent = self._make_agent(tmp_path, monkeypatch)
        agent.session.is_subagent = True
        threshold = int(M.CONTEXT_LIMIT * M.COMPRESS_AT)
        agent._last_usage = (threshold + 1, 50, True)
        # 子 agent 即便撞阈值也不压
        assert agent._should_compress() is False

    def test_maybe_compress_before_turn_first_turn_skips(self, tmp_path, monkeypatch):
        """第一个 turn 还没设 _last_usage, 不该压也不该崩."""
        agent = self._make_agent(tmp_path, monkeypatch)
        # 不设 _last_usage, 直接调
        agent._maybe_compress_before_turn()  # 不抛就行
        assert len(agent.session.history) == 1  # 只有 system, 没动


class TestCompressionRoundTripsThroughOpenAI:
    """压缩后的 history 仍能 _history_to_openai 转出来 — 不留半空 tool_calls 字段."""

    def test_compressed_history_serializes(self):
        h = [M.Message.system("sys")]
        for i in range(15):
            tc = _make_tool_call(f"c{i}", "Read", {"path": f"f{i}.py"})
            h.append(M.Message.user(f"q{i}"))
            h.append(M.Message.assistant("", tool_calls=[tc]))
            h.append(M.Message.tool(f"r{i}", tool_call_id=f"c{i}"))
        result = M._compress_history(h, _FakeLLM(), keep_tail=10)
        assert result is not None
        new_h, _ = result
        # _history_to_openai 不抛
        out = M._history_to_openai(new_h)
        assert isinstance(out, list)
        # 第一条是 system
        assert out[0]["role"] == "system"
        # 第二条是含 marker 的 user 消息
        assert out[1]["role"] == "user"
        assert M._COMPRESSION_MARKER in out[1]["content"]
