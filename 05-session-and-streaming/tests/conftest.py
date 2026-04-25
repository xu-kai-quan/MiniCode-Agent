"""pytest 配置: 让 `import todo` 找得到.

v4 (Ollama 后端) 只依赖 stdlib, 不再需要 stub 重依赖.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
