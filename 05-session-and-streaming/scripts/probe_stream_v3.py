"""探针 v3: 强制长回复, 验 Ollama 是真流式还是单次 batch dump.

v2 发现 Ollama 第二轮把 254 字符的中文回复一次性塞进单个 delta — 看不到流式. 但
也可能是因为输出太短, 触发了内部缓冲策略. 这里换成"生成 ~1500 字符的长回复"看会
不会真分多个 delta.

修复 v2 的两个小问题:
  1. Windows stdout 强制 UTF-8 (避免中文乱码)
  2. 用英文 prompt 兜底 (即使 reconfigure 失败, 至少协议层数据是干净的 ASCII)
"""
from __future__ import annotations

import json
import os
import sys
import time
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

# Windows 终端: 把 stdout 改成 UTF-8 输出, 避免中文乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OLLAMA_URL = os.environ.get("MINICODE_OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MINICODE_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
TIMEOUT = 300

# 英文 prompt — 协议层不依赖中文 codec
SYSTEM = (
    "You are a coding agent. When asked to explain code, give a thorough, "
    "structured explanation with multiple paragraphs. Aim for ~1500 characters."
)
USER_QUERY = (
    "Explain in detail how a Python ReAct agent works: the loop structure, "
    "tool dispatch, message history, why tool_calls are structured, "
    "common pitfalls. Be thorough — at least 4 paragraphs."
)


def main() -> int:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_QUERY},
        ],
        "temperature": 0.0,
        "stream": True,
    }

    url = f"{OLLAMA_URL.rstrip('/')}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    print(f"[probe v3] POST {url}")
    print(f"[probe v3] model={MODEL}")
    print(f"[probe v3] strategy: force long-form English answer (~1500 chars)")
    print(f"[probe v3] expect: many small deltas if truly streaming")
    print("=" * 72)

    try:
        resp = urlrequest.urlopen(req, timeout=TIMEOUT)
    except HTTPError as e:
        print(f"[probe v3] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        return 1
    except URLError as e:
        print(f"[probe v3] URLError: {e.reason}")
        return 1

    t_start = time.monotonic()
    t_first_delta: float | None = None
    deltas: list[dict] = []
    finish_reason: str | None = None

    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]" or not data:
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                choice = chunk["choices"][0]
            except (KeyError, IndexError):
                continue
            delta = choice.get("delta") or {}
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            now = time.monotonic() - t_start
            text = delta.get("content")
            if text:
                if t_first_delta is None:
                    t_first_delta = now
                deltas.append({"t": now, "text": text})
                # 不实时回显 — 避免被乱码干扰判断, 最后一次性回显

    t_end = time.monotonic() - t_start

    print(f"[probe v3] FULL TEXT FOLLOWS (then stats):")
    print("-" * 72)
    full_text = "".join(d["text"] for d in deltas)
    print(full_text)
    print("-" * 72)

    print()
    print(f"[probe v3] TIMING")
    print(f"  total wall time         : {t_end:.2f}s")
    if t_first_delta is not None:
        print(f"  time to first delta     : {t_first_delta:.2f}s")
        gen_window = max(0.001, deltas[-1]['t'] - t_first_delta)
        print(f"  generation window       : {gen_window:.2f}s")
        print(f"  avg delta interval      : {gen_window / max(1, len(deltas) - 1) * 1000:.1f}ms")
    print(f"  text deltas count       : {len(deltas)}")
    print(f"  finish_reason           : {finish_reason!r}")

    sizes = [len(d["text"]) for d in deltas]
    if sizes:
        sizes_sorted = sorted(sizes)
        n = len(sizes)
        print()
        print(f"[probe v3] DELTA SIZE DISTRIBUTION (chars)")
        print(f"  count   : {n}")
        print(f"  min     : {min(sizes)}")
        print(f"  max     : {max(sizes)}")
        print(f"  mean    : {sum(sizes)/n:.1f}")
        print(f"  median  : {sizes_sorted[n//2]}")
        print(f"  p90     : {sizes_sorted[int(n*0.9)]}")
        # 直方图
        buckets = {(1, 5): 0, (6, 20): 0, (21, 50): 0, (51, 100): 0, (101, 500): 0, (501, 9999): 0}
        for s in sizes:
            for (lo, hi) in buckets:
                if lo <= s <= hi:
                    buckets[(lo, hi)] += 1
                    break
        print(f"  size distribution:")
        for (lo, hi), c in buckets.items():
            bar = "#" * min(40, c)
            print(f"    {lo:4d}-{hi:<4d}  : {c:4d}  {bar}")

    print()
    print(f"[probe v3] FIRST 10 DELTAS (timing + text):")
    for i, d in enumerate(deltas[:10], 1):
        snippet = d['text'][:60] + ('...' if len(d['text']) > 60 else '')
        print(f"  #{i:2d}  t={d['t']*1000:6.0f}ms  len={len(d['text']):3d}  text={snippet!r}")

    print()
    print("=" * 72)
    print("[probe v3] VERDICT")
    if len(deltas) <= 2:
        print("  [BATCH] Ollama is NOT streaming for this model — single dump.")
        print("  Implication: streaming code paths in todo.py work but provide no UX benefit.")
        print("  Possible cause: model/Ollama version doesn't stream, or client buffering at urllib level.")
    elif sizes and max(sizes) > 200:
        print("  [PARTIAL] Some streaming, but at least one delta is large (>200 chars).")
        print("  STREAM_FLUSH_CHARS=80 will flush early on those large chunks — fine.")
    else:
        print("  [STREAMING] Many small deltas — true token-ish streaming.")
        print("  STREAM_FLUSH_CHARS=80 + \\n boundary will give smooth output.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
