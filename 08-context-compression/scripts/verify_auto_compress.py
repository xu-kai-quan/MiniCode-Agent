"""非交互式验证: 自动触发 context 压缩的端到端测试.

跑法:
    cd 08-context-compression
    python scripts/verify_auto_compress.py

做了什么:
    1. 临时把 CONTEXT_LIMIT 调低 (默认 5000), COMPRESS_AT 不变
    2. 启动一个 ReActAgent (真 minimax LLM)
    3. 发一个会读多个文件的 query, 让 prompt token 累积到撞阈值
    4. 验证 _maybe_compress_before_turn 真的被调到 — history 长度变了 + 中间出
       现 <compressed>...</compressed> marker
    5. 输出结论 + 真实成本

跟 REPL 交互式跑的区别:
    - 不用人手动敲 input
    - 限制 max_rounds 防失控
    - 显式断言压缩发生了, 没发生就报错退出
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 import todo 之前设环境变量 — module-level 读 os.environ 的常量定了就定了
os.environ.setdefault("MINICODE_CONTEXT_LIMIT", "5000")
os.environ.setdefault("MINICODE_COMPRESS_AT", "0.7")
os.environ.setdefault("MINICODE_COMPRESS_KEEP_TAIL", "6")
# 流式还是默认开 (跟 REPL 一致, 走真路径)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import todo as M  # noqa: E402


def main() -> int:
    print("=" * 72)
    print("VERIFY AUTO-COMPRESS  (non-interactive end-to-end)")
    print("=" * 72)
    print(f"  backend           : {M.BACKEND}")
    print(f"  CONTEXT_LIMIT     : {M.CONTEXT_LIMIT}")
    print(f"  COMPRESS_AT       : {M.COMPRESS_AT}")
    print(f"  threshold         : {int(M.CONTEXT_LIMIT * M.COMPRESS_AT)} prompt tokens")
    print(f"  COMPRESS_KEEP_TAIL: {M.COMPRESS_KEEP_TAIL}")
    print("=" * 72)
    print()

    if M.BACKEND != "minimax":
        print("WARNING: backend is not minimax — this script is calibrated for minimax")
        print("         (ollama 也能跑, 但 token 估算可能不准)")
        print()

    # 准备 agent — 跟 REPL 一样的 build 路径
    llm = M.load_model()
    M._SPAWN_LLM = llm  # 万一模型派 spawn_agent (这个 query 不会, 但保险)
    terminal = M.TerminalTool(M.WORKDIR)
    registry = M.build_default_registry(terminal)
    agent = M.ReActAgent(llm, registry)

    # 选个保证撞阈值的 query — 让模型读一两个长函数就够了
    # 5000 limit * 0.7 = 3500 tokens, Read 一次 todo.py 的某个区间就 ~2K, 跑 2-3
    # 轮就该撞.
    query = (
        "请仔细阅读 todo.py 第 1100-1400 行 (apply_patch 整个实现), "
        "然后再读第 2400-2700 行 (ReActAgent run loop), "
        "最后告诉我这两块代码各自在做什么."
    )

    print(f">>> QUERY: {query}")
    print()

    # 跑 — 加上 max_rounds 兜底, 防长任务跑飞
    history_lengths_per_turn: list[int] = []
    has_compressed_marker = False

    # 我们需要在 run() 里间接知道压缩有没有触发 — agent.run() 直接打印, 不返回
    # 中间状态. 简单办法: 跑完看最终 history 是否含 marker 消息.
    # 更精细: monkey-patch _maybe_compress_before_turn 拦截.
    real_check = agent._maybe_compress_before_turn
    compress_calls: list[dict] = []

    def patched():
        before = len(agent.session.history)
        real_check()
        after = len(agent.session.history)
        if after != before:
            compress_calls.append({
                "turn_history_before": before,
                "turn_history_after": after,
                "last_prompt_tokens": agent._last_prompt_tokens(),
            })

    agent._maybe_compress_before_turn = patched  # type: ignore

    try:
        agent.run(query, max_rounds=12)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 2
    except Exception as e:
        print(f"\n[unexpected error during run: {type(e).__name__}: {e}]")
        return 3

    # 收尾分析
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)

    # 1. 看 history 里有没有 compressed marker
    for m in agent.session.history:
        if m.role == "user" and M._COMPRESSION_MARKER in (m.content or ""):
            has_compressed_marker = True
            break

    threshold = int(M.CONTEXT_LIMIT * M.COMPRESS_AT)
    final_prompt = agent._last_prompt_tokens()
    sess = agent.session

    print(f"  threshold              : {threshold} tokens "
          f"(= {M.CONTEXT_LIMIT} × {M.COMPRESS_AT})")
    print(f"  final turn prompt_t    : {final_prompt}")
    print(f"  threshold ever crossed : "
          f"{'YES' if final_prompt >= threshold or compress_calls else 'NO'}")
    print(f"  compression triggered  : "
          f"{'YES (' + str(len(compress_calls)) + 'x)' if compress_calls else 'NO'}")
    if compress_calls:
        for i, c in enumerate(compress_calls, 1):
            print(f"    [{i}] history {c['turn_history_before']} → "
                  f"{c['turn_history_after']} msgs  "
                  f"(prior turn prompt={c['last_prompt_tokens']})")
    print(f"  marker in history      : "
          f"{'YES' if has_compressed_marker else 'NO'}")
    print(f"  final history len      : {len(sess.history)} messages")
    print(f"  total cost             : "
          f"{M._fmt_k(sess.prompt_tokens_total)} in + "
          f"{M._fmt_k(sess.completion_tokens_total)} out  "
          f"≈ ¥{M._estimate_cost(sess.prompt_tokens_total, sess.completion_tokens_total):.4f}")
    print()

    if compress_calls and has_compressed_marker:
        print("RESULT: PASS — auto-compression hit the threshold and reshaped history.")
        return 0
    if final_prompt < threshold and not compress_calls:
        print("RESULT: INCONCLUSIVE — never crossed threshold (lower CONTEXT_LIMIT and retry).")
        print("        try: MINICODE_CONTEXT_LIMIT=2500 python scripts/verify_auto_compress.py")
        return 4
    print("RESULT: FAIL — threshold crossed but compression did not run / marker missing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
