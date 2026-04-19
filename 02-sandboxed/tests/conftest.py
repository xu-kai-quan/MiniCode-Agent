"""Stub heavy ML deps before importing todo.py.

todo.py 顶部会 `import torch` / `from transformers import ...`, 单这两行就要拉
2GB+ 的轮子。CI 跑工具层的纯逻辑测试用不上模型, 所以在 import 前用占位模块
顶替掉。production code 不需要为测试改一行。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _stub_heavy_deps() -> None:
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = type("x", (), {"is_available": staticmethod(lambda: False)})()
        torch_mod.float16 = "float16"
        torch_mod.float32 = "float32"
        torch_mod.inference_mode = lambda: _NullCtx()
        sys.modules["torch"] = torch_mod
    if "transformers" not in sys.modules:
        tr_mod = types.ModuleType("transformers")
        tr_mod.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
        tr_mod.AutoTokenizer = type("AutoTokenizer", (), {})
        sys.modules["transformers"] = tr_mod


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


_stub_heavy_deps()

# 把 02-sandboxed/ 加进 sys.path, 让 `import todo` 找得到
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
