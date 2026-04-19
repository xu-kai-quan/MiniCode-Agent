"""pytest 配置: 在 import todo.py 之前屏蔽 torch / transformers,
让工具层测试不必加载 2B 模型权重."""
import sys
import types
import contextlib
from pathlib import Path


def _stub_heavy_deps() -> None:
    """伪造 torch / transformers, 让 todo.py 顶部的 import 不报错."""
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = type("x", (), {"is_available": staticmethod(lambda: False)})()
        torch_mod.float16 = None
        torch_mod.float32 = None
        torch_mod.inference_mode = lambda: contextlib.nullcontext()
        sys.modules["torch"] = torch_mod

    if "transformers" not in sys.modules:
        tf_mod = types.ModuleType("transformers")
        tf_mod.AutoModelForCausalLM = None
        tf_mod.AutoTokenizer = None
        sys.modules["transformers"] = tf_mod


_stub_heavy_deps()

# 把 03-atomic-tools/ 加进 sys.path, 让 `import todo` 找得到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
