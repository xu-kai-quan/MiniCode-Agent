"""探针: 验证 MiniMax 端点真能跑, 且 SSE/tool_calls 行为符合 _StreamRenderer 的假定.

跟 probe_stream_v3 同精神, 但调真 MiniMax API. 跑这个之前:
  1. 在 .env (项目根目录, gitignored) 里填 MINIMAX_API_KEY=<your key>
  2. 确认 MINICODE_BACKEND=minimax (默认就是)
  3. python scripts/probe_minimax.py

不验证模型生成质量 — 只验协议层.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 让 import todo 能找到模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import todo as M

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> int:
    if M.BACKEND != "minimax":
        print(f"[probe] MINICODE_BACKEND={M.BACKEND} — set it to 'minimax' in .env first.")
        return 1
    if not M.MINIMAX_API_KEY:
        print("[probe] MINIMAX_API_KEY is empty. Fill it in .env (gitignored).")
        print(f"        .env path: {Path(__file__).resolve().parent.parent / '.env'}")
        return 1

    client = M.MinimaxClient(api_key=M.MINIMAX_API_KEY)
    print(f"[probe] using {client.provider_name} @ {client.base_url}")
    print(f"[probe] model: {client.model}")
    print("=" * 72)

    # ===== Round 1: tool_call =====
    LS_SCHEMA = {
        "type": "function",
        "function": {
            "name": "LS",
            "description": "List entries in a directory (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    }
    messages = [
        {"role": "system", "content": "You are an agent. Always call the LS tool first to look around."},
        {"role": "user", "content": "What's in the current directory?"},
    ]
    print("\n[probe] Round 1: triggering tool_call...")
    t0 = time.monotonic()
    deltas = []
    tcs_acc = {}
    text_acc = []
    for delta in client.chat_stream(messages, [LS_SCHEMA]):
        deltas.append(delta)
        if delta.get("content"):
            text_acc.append(delta["content"])
        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            slot = tcs_acc.setdefault(idx, {"id": None, "name": None, "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if "arguments" in fn:
                slot["args"] += fn["arguments"] or ""
    elapsed = time.monotonic() - t0
    print(f"[probe] elapsed: {elapsed:.2f}s, deltas: {len(deltas)}")
    print(f"[probe] text accumulated: {''.join(text_acc)!r}")
    print(f"[probe] tool_calls accumulated:")
    for idx, slot in tcs_acc.items():
        print(f"  slot[{idx}]: id={slot['id']!r} name={slot['name']!r} args={slot['args']!r}")
        if slot["args"]:
            try:
                json.loads(slot["args"])
                print(f"    -> args is valid JSON")
            except json.JSONDecodeError as e:
                print(f"    !! args not valid JSON: {e}")

    if not tcs_acc:
        print("[probe] WARNING: no tool_calls received. Model may have answered with text.")

    # ===== Round 2: 长文本流式 =====
    print()
    print("=" * 72)
    print("[probe] Round 2: long-form streaming text (no tools)...")
    long_messages = [
        {"role": "system", "content": "Give thorough technical explanations."},
        {"role": "user", "content": "Explain in 4+ paragraphs how a Python ReAct agent loop works."},
    ]
    t0 = time.monotonic()
    sizes = []
    full = []
    first_t = None
    for delta in client.chat_stream(long_messages, None):
        text = delta.get("content")
        if text:
            now = time.monotonic() - t0
            if first_t is None:
                first_t = now
            sizes.append(len(text))
            full.append(text)
    elapsed = time.monotonic() - t0
    full_text = "".join(full)
    print(f"[probe] elapsed: {elapsed:.2f}s")
    print(f"[probe] time-to-first-delta: {first_t:.2f}s" if first_t else "  (no deltas)")
    print(f"[probe] delta count: {len(sizes)}")
    if sizes:
        print(f"[probe] delta sizes — min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f}")
    print(f"[probe] full text length: {len(full_text)} chars")
    print(f"[probe] first 200 chars: {full_text[:200]!r}")

    # ===== Verdict =====
    print()
    print("=" * 72)
    print("[probe] VERDICT")
    issues = []
    if not tcs_acc:
        issues.append("Round 1: no tool_calls — model may not be using structured calling")
    else:
        for idx, slot in tcs_acc.items():
            if not slot["name"]:
                issues.append(f"Round 1 slot {idx}: no name accumulated")
            if slot["args"]:
                try:
                    json.loads(slot["args"])
                except json.JSONDecodeError:
                    issues.append(f"Round 1 slot {idx}: args don't form valid JSON")

    if len(sizes) <= 2:
        issues.append(f"Round 2: only {len(sizes)} delta(s) — batch dump, not streaming")
    if sizes and max(sizes) > 200:
        issues.append(f"Round 2: largest delta is {max(sizes)} chars — _StreamRenderer's "
                       "80-char flush threshold is fine but bigger chunks happen")

    if not issues:
        print("  ALL CHECKS PASS — MiniMax client should drop into todo.py REPL cleanly.")
    else:
        print(f"  {len(issues)} observation(s):")
        for i in issues:
            print(f"    - {i}")
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
