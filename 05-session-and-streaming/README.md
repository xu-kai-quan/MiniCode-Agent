# 05-session-and-streaming — 让本地 agent 用起来不别扭

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **alpha**, 但比 04 实用. 单人项目, 接口可能继续变. 工具层有 66 个 pytest 测试, 主链路稳定.

这是 MiniCode Agent 第 5 版. 一个文件 [todo.py](todo.py), 大约 2000 行 Python. 跑起来给你一个 `>` 提示符, 你跟它说"帮我建个项目"或者"看一下 todo.py 第 800 行解释一下", 它就动手干。

## 它跟 04 的区别 — 一句话版本

**04 能跑, 05 能用**.

04 工程上已经合格 (原子工具、apply_patch 跨文件原子改动、66 个测试). 但实际用起来有几个让人想关掉它的地方:

- 等了 30 秒屏幕没动, 不知道是死了还是在思考
- 7B 模型在多步任务里经常脱轨, 死循环 19 轮才超时退出
- 本地 7B 跑长任务的连续性确实不如云端大模型, 但 04 没法切换
- 跑一次任务不知道用了多少 token, 云端按 token 收费时心里没底
- 多 agent 实例共享模块级全局变量, 状态串台

05 一个一个修. 这份 README 重点讲**每件事是为什么做、怎么做、起没起效果**. 不是技术参考, 是过程记录.

---

## 目录

- [1. Session — 把散落的状态收回来](#1-session--把散落的状态收回来)
- [2. 流式输出 — 看见模型在思考](#2-流式输出--看见模型在思考)
- [3. 双后端 — Ollama 和 MiniMax 之间切](#3-双后端--ollama-和-minimax-之间切)
- [4. Token 和成本可见 — 不再黑盒烧钱](#4-token-和成本可见--不再黑盒烧钱)
- [5. 代码块伪 tool_call — 系统层把 prompt 兜不住的接住](#5-代码块伪-tool_call--系统层把-prompt-兜不住的接住)
- [6. PATH_ESCAPE 错误信息 — 不让模型在错误后放弃](#6-path_escape-错误信息--不让模型在错误后放弃)
- [7. 流式 Read 和 SYSTEM 精简 — 两个事后优化](#7-流式-read-和-system-精简--两个事后优化)
- [8. 安全脚手架 — .env 和 pre-commit hook](#8-安全脚手架--env-和-pre-commit-hook)
- [9. 怎么跑起来](#9-怎么跑起来)
- [10. 已知问题 / 还能优化的地方](#10-已知问题--还能优化的地方)
- [11. 给后来者: 如果你想读源码](#11-给后来者-如果你想读源码)

---

## 1. Session — 把散落的状态收回来

### 04 是什么样

04 的可变状态散在三个地方:

```python
# 模块顶层全局
TODO = TodoManager()

# agent 实例上
class ReActAgent:
    def __init__(self, ...):
        self.history: list[Message] = [...]
        self.rounds_since_todo = 0

# registry 实例上
class ToolRegistry:
    def __init__(self, ...):
        self.read_cache = ReadCache()
```

要"清空当前会话从头开始", 得三处都 reset:

```python
def reset(self):
    self.history = [Message.system(...)]
    self.rounds_since_todo = 0
    TODO.items.clear()              # 全局变量, 别忘了
    self.registry.read_cache.clear() # 另一个对象上的, 也别忘了
```

### 这有什么问题

不是"代码风格不好"那种问题, 是**真问题**:

1. **模块级 `TODO` 是全进程共享的** — 同一个 Python 进程跑两个 agent 实例, 它们的 todo 列表会互相污染. 单进程 REPL 看不出来, 但写测试的时候就麻烦
2. **状态散三处, reset 必漏** — 加新的状态字段时, 你得记得改 reset(). 反例: 加个 codeblock_nag_count 在 agent 上, reset 时容易忘记清, 串到下次任务
3. **没法表达"不同会话"** — 想做"这次跑用这个 system prompt, 下次跑换一个", 没有清晰的边界

### 怎么改

把所有可变状态封进一个 dataclass:

```python
@dataclass
class Session:
    history: list[Message]
    todo: TodoManager = field(default_factory=TodoManager)
    read_cache: ReadCache = field(default_factory=ReadCache)
    rounds_since_todo: int = 0
    codeblock_nag_count: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    # ... 等等

    @classmethod
    def new(cls, system_prompt: str) -> "Session":
        return cls(history=[Message.system(system_prompt)])
```

`ReActAgent` 不再持有 history/计数器, 通过 property 代理到 session:

```python
class ReActAgent:
    @property
    def session(self) -> Session:
        return self.registry.session

    @property
    def history(self) -> list[Message]:
        return self.session.history

    def new_session(self) -> None:
        # reset 一行解决: 重建 Session, 所有字段默认值清零
        self.registry.session = Session.new(self.system_prompt)
```

### 起没起效果

起了, 而且**红利不止于 reset 干净**:

- **写测试更顺** — 之前 audit 脚本里我会 build_default_registry 三次开三个独立 registry 来绕开 read_cache 污染. 现在新 session 一行就独立
- **token 计数有了归宿** — 加 prompt_tokens_total/completion_tokens_total 这两个新字段时, 直接进 Session, 不需要再决定"挂在哪里". 见 [§4](#4-token-和成本可见--不再黑盒烧钱)
- **后面所有新状态字段都进 Session** — codeblock_nag_count、estimated_turns/exact_turns 都是这次设计后顺手加的, 没纠结过"放哪"

更重要的是**它没引入新复杂度**. 改完 ReActAgent.__init__ 那 7 行代码, dispatch / step / run 全部不需要动 — 因为 history 这些通过 property 代理, 用法没变.

### 中间的尝试

我在动手前讨论过三个方案 (具体见 commit `a156839` 之前的对话):

- **方案 A**: 所有 handler 签名加 `session` 参数 — 侵入太大, 9 个工具有 8 个根本不需要 session
- **方案 B**: dispatch 检测 handler 签名里有没有 `_session: Session`, 有就注入 — 优雅, 选了这个
- **方案 C**: registry 持有 session 引用 — 之前以为破坏"无状态 registry", 后来发现 04 的 registry 已经持有 read_cache 状态了, 升级为 session 不算新耦合, **最后用的是这个**

最终落地: registry 持有 session, dispatch 通过反射给声明了 `_session: Session` 的 handler 自动注入. 反射默认排除 `_` 前缀参数 (避免 `_session` 跑进给模型看的 schema). 工具表里只有 `tool_apply_patch` 和 `tool_todo` 用了这个机制 — 其他 8 个 handler 是纯函数, 跟 session 完全脱耦.

---

## 2. 流式输出 — 看见模型在思考

### 痛点

04 默认非流式. 你输入 query, 看到 `→ [1] 调用模型 …` 然后屏幕静止 5-30 秒. 第一次用以为它挂了. 7B 在 4060 上的速度大约 40 token/秒, 一个稍长的回答就要等十几秒.

### 怎么做

OpenAI 兼容协议的流式很标准 — `payload["stream"] = True`, response 是 SSE (Server-Sent Events), 一行行 JSON delta. 但**陷阱比看起来多**.

我先调研了 Claude Code (CLI) 的流式行为 (因为它是同类产品, 它做对的我抄, 它做错的我避). 重点观察:

- **不是 token-by-token 输出** — Claude Code 也是 chunk-based buffering. 原因: 终端逐 token print 闪屏严重, 中文 + ANSI 颜色组合下尤其卡 (有人提过 issue #29213)
- **tool_call 等完整 arguments 才显示** — 不做"半截 JSON 预览". 不然显示个 `{"path":"`没了, 用户也看不懂
- **Ctrl-C 应当干净中断** — 但 Claude Code 自己的 issue (#26802, #17466) 抱怨 Ctrl-C 不可靠. 我们要做对

我们的实现 (在 [todo.py 的 `_StreamRenderer`](todo.py)):

1. **OllamaClient.chat_stream** 是个 generator, yield 出每个 delta
2. **`_StreamRenderer.feed(delta)`** 累积 content / tool_calls. 文本累积到行边界 (`\n`) 或 80 字符上限再 flush
3. **tool_call reassembly** — OpenAI 协议把 `tool_calls[i].function.arguments` 拆成多个分片, 按 `index` 累加. 但 Ollama 实际把整个 tool_call 一次性塞进单 delta, 跟 spec 不一致 (但对我们更友好, reassembly 一片或 N 片都能吃)
4. **think 标签不在流式期分色** — Qwen 的 `<think>...</think>` 跨 chunk 边界处理太脆 (考虑 chunk 切在 `<thi` + `nk>` 中间), 流式期间整段普通色, 等 finalize 用 `split_think` 一次性剥离进 history
5. **Ctrl-C 处理** — 流式 loop 里 try/except KeyboardInterrupt, 已收到的 partial 入 history (带 `[interrupted by user]` 标记让模型下次知道), 半截 tool_call **不执行** (避免拿不完整 arguments 调工具)

### 起没起效果

起了, 体验差异巨大. 但**也暴露了一个真相**: 第一轮模型生成 tool_call 时, 大部分内容是 tool_call 而不是 content, 所以"流式视觉体验"那一轮其实不明显. 真正闪光的是**第二轮 — 模型拿到工具结果后生成给用户的可见答案**, 那时候才是大段 token 流出.

跑过真 Ollama 探针 (`scripts/probe_stream_v3.py`) 验证: 长回复 (~6000 字符) 拆成 1168 个 deltas, 平均每 5 字符一个 delta, 每 23ms 一次 — 真 token 级流式.

### 中间的尝试

写了三个版本探针 (probe_stream.py / v2 / v3, 后两个保留为 `scripts/probe_stream_v3.py`). v1 测短回复 + tool_call, v2 测中等长度, v3 强制长英文回复. 结论:

- **短回复时 Ollama 会 batch dump** (1 个 delta 包全部) — 看起来像没流式, 实际是 Ollama 内部缓冲优化, 不是 bug
- **长回复时 Ollama 真流式** — token 级 delta
- **MiniMax 流式粒度更粗** — 平均 130 字符/delta (按句/段而不是 token), 视觉上"成段成段"出, 仍然流式

### Ctrl-C 这件事我们做对了

Claude Code 的 issue 26802 / 17466 抱怨 escape 不可靠. 我们的实现:

```python
try:
    for delta in self.llm.chat_stream(messages, tools):
        renderer.feed(delta)
except KeyboardInterrupt:
    interrupted = True
msg = renderer.finalize()  # 先 flush partial 内容
if interrupted:
    print(f"(interrupted by user — partial output kept)")
```

- partial content 立刻进 history, 不丢
- 半截 tool_call (假设 arguments 流到一半) **不执行** — 这是关键安全策略. 我们宁可错失一次 tool 调用, 也不拿不确定的 arguments 去调工具

测试用 fake LLM 模拟"流到一半 raise KeyboardInterrupt", 验证: visible 内容入历史 + `[interrupted by user]` 标记 + actions 为空. 通过.

---

## 3. 双后端 — Ollama 和 MiniMax 之间切

### 为什么要做

04 一直绑死 Ollama 跑本地 7B. 跑下来发现 7B 在多步任务上**有能力上限**:

实际跑过 query "解释一下 todo.py 第 800-900 行 apply_patch 的两阶段锁怎么工作的", 7B 的表现:

- Turn 1: 调 `LS('/path/to/todo.py')` — 把训练数据里的占位符当真实路径
- Turn 2: 收到 PATH_ESCAPE, 不重试, 直接说"please provide a path"
- 改了 SYSTEM 加 workspace 后, 第二次跑:
- Turn 1: 调对了 Read
- Turn 2: 莫名其妙调 apply_patch 传空 patch
- Turn 3: 拿到 PARSE_FAILED, 在 todo 里写"Review the provided unified diff" — 幻觉了一个 diff
- Turn 4: 给用户回答"please ensure your unified diff includes proper headers" — 把自己当客服了

7B 的"任务连续性"问题在 README §3.4 (04 时代) 早就记录过, 这是模型能力上限, prompt 治不好.

### 想清楚再动手

我先讨论了三个方向:

1. **接受现状** — 把"7B 跑长解释问题不可靠"写进已知局限, 不再为这个改代码
2. **换更大模型** — 验证是不是模型能力问题
3. **加更狠的系统层约束** — 比如 query intent 分类禁用某些工具. 这条违背"不过度设计"原则, 否决

选了 1+2: 接受 7B 的极限 + 提供云端切换.

### 用 MiniMax 还是别的

研究了一下 (调研报告归档在 commit `079b7d3` 的对话里). MiniMax-M2.7:

- 200K context, 131K output (M2 系列)
- 公开标称为"自主编码 agent"打造, SWE-Bench 分数高
- 国内可访问 (`api.minimaxi.com`)
- **OpenAI 兼容端点** — 这一条是关键, 我们的 OllamaClient 是 OpenAI 形态, 复用代码量极大

也研究了它的 Anthropic 兼容端点 (`/anthropic/v1/messages`), 但 litellm #18834 等 issue 有已知 bug, 选 OpenAI 路径稳得多.

### 怎么实现

抽出一个共享基类:

```python
class _OpenAICompatClient:
    """OpenAI-兼容 HTTP 客户端基类 — Ollama 和 MiniMax 共享这套."""

    def __init__(self, base_url, model, timeout, extra_headers=None):
        self.base_url = base_url
        self.model = model
        self.extra_headers = extra_headers or {}

    def chat(self, messages, tools): ...
    def chat_stream(self, messages, tools): ...


class OllamaClient(_OpenAICompatClient):
    provider_name = "Ollama"
    # 默认 base_url 本地, 不需要 auth header

class MinimaxClient(_OpenAICompatClient):
    provider_name = "MiniMax"
    def __init__(self, api_key, ...):
        super().__init__(extra_headers={"Authorization": f"Bearer {api_key}"})
```

`load_model()` 工厂按 `MINICODE_BACKEND` 环境变量选:

```python
def load_model() -> LLMClient:
    if BACKEND == "minimax":
        return MinimaxClient(api_key=MINIMAX_API_KEY)
    elif BACKEND == "ollama":
        return OllamaClient()
    else:
        raise RuntimeError(f"Unknown MINICODE_BACKEND={BACKEND!r}")
```

切换方法: 在项目根 `.env` 里写 `MINICODE_BACKEND=minimax`, 启动时自动读. 见 [§9](#9-怎么跑起来).

### 中间踩的坑

跑探针验证 MiniMax 端点时碰到 **HTTP 404**. 直接 curl 同样 payload 是 200. 排查后发现:

```python
final stream URL: https://api.minimaxi.com/v1/v1/chat/completions
                                          ^^^^^^^^
                                          /v1 重复了
```

`base_url = "https://api.minimaxi.com/v1"` (来自 .env.example 默认值), 然后 chat_stream 里又拼 `/v1/chat/completions`. 修法: 跟 Ollama 一样, base_url 只到主机名 `https://api.minimaxi.com`, client 内部统一拼 `/v1/chat/completions`. 在 .env.example 里加了显式注释 "do NOT include /v1 in base_url" 防止下次再犯. (commit `d91f227`)

### 起没起效果

跑同一个 query (解释 apply_patch 锁) 对比:

| | 7B (Ollama) | M2.7 (MiniMax) |
|---|---|---|
| Turn 1 | LS 错误路径 | Read 正确 offset/limit |
| Turn 2 | 调 apply_patch 传空 patch | Grep 找 def tool_apply_patch 真实位置 |
| Turn 3 | todo 摆烂 + 让用户给 patch | Read 1050-1250 行 |
| Turn 4 | 教用户怎么写 diff 格式 | 流式生成完整结构化解释 |
| 结果 | 没回答原问题 | 完整回答, 含三阶段拆解 + 表格 + 总结 |

M2.7 还**自我纠错** — Turn 2 它说"读到的 lines 800-900 是 grep 工具, apply_patch 不在这个范围", 主动 Grep 找真实位置. 这种"我读错了, 再来"的元认知能力, 7B 完全做不到.

---

## 4. Token 和成本可见 — 不再黑盒烧钱

### 为什么要做

切到 MiniMax 后第一次跑, 跑完一脸懵 — 烧了多少钱? 不知道. 跑第二次还是不知道. 长任务跑下来更没数. 云端按 token 计费, 不能黑盒.

### 怎么做

**取数据**: OpenAI 协议规定流式的 usage 在最后一个 chunk 顶层 (不在 delta 里), 但需要在请求里加 `stream_options: {include_usage: true}`. Ollama 不识别这个字段直接忽略 (无害), MiniMax 会照办.

```python
payload = {
    "model": self.model,
    "messages": messages,
    "stream": True,
    "stream_options": {"include_usage": True},  # 关键
}
```

`chat_stream` 解析 chunk 时除了取 delta, 也看顶层有没有 `usage`:

```python
if isinstance(chunk.get("usage"), dict):
    self.last_usage = chunk["usage"]  # 留给调用方取
```

**优雅降级**: 如果后端没给 usage (Ollama 流式不给), 用字符数估算 (中英混合大致 1 token ≈ 3 字符), 标 `(estimated)` 让用户知道是估的.

**累加到 Session**: 上面 §1 已经有 Session 了, 加 `prompt_tokens_total` / `completion_tokens_total` 字段, 每个 turn 累加, run() 退出时显示总计.

**显示成本**: MiniMax-M2.7 当前定价大约 input ¥1/M, output ¥8/M. 我们用保守值 (¥2/M, ¥8/M). 价格放在模块顶常量, 改一处即可:

```python
MINIMAX_PRICE_INPUT_PER_M = float(os.environ.get("MINIMAX_PRICE_IN", "2.0"))
MINIMAX_PRICE_OUTPUT_PER_M = float(os.environ.get("MINIMAX_PRICE_OUT", "8.0"))
```

### 长什么样

每个 turn 末尾一行:

```
→ [2] tokens: 2.1K in + 819 out = 2.9K  ≈ ¥0.0107
```

DONE / GAVE UP / MAX ROUNDS 时给总计:

```
┌─ ✅ DONE ════════════════════════════════════════════════
│ session total: 5.3K in + 2.1K out = 7.4K  ≈ ¥0.0274
└═══════════════════════════════════════════════════════════
```

如果是 Ollama (本地零成本), 不显示 `≈ ¥` 部分; 如果是估算, 标 `(estimated)`.

### 中间的尝试

最初想"非流式调一次额外的请求取 usage"——浪费一次 API call. 否决.

也想过"加 tiktoken 算精确 token"——破坏"零依赖"卖点, 而且 tiktoken 也算不准 MiniMax 的 token. 否决.

最后选了"include_usage 优先, 估算兜底"的方案 — 不引依赖, 后端给真值就用真值, 没有就估算并标注. 实测 MiniMax 给的是真值, Ollama 流式给不出但没关系 (本地零成本无所谓).

---

## 5. 代码块伪 tool_call — 系统层把 prompt 兜不住的接住

### 04 已经踩过的坑

04 README §3.2 记录过: 7B 经常不走 `tool_calls` 字段, 而是在 content 里写 ```json 块假装调工具. 系统收不到工具调用, 任务卡死. 04 在 SYSTEM 里加了 CRITICAL 段堵 ```json 和 ```diff/```patch.

但 05 实际跑发现: **堵了 ```json/```diff, 模型改用 ```bash 或 ```shell**. 你压一处, 它从另一处冒出来.

### 两层防线

**Prompt 层** (软约束) — SYSTEM 里把 ```bash/```sh/```shell 也加进禁令, 并且讲清楚 RUN vs SHOW 的区别 (怕误伤"展示代码"的合法用法):

```
TOOL-CALL PROTOCOL (critical): when invoking a tool, emit a real
structured tool_call. NEVER write the call as visible text — not as
```json, ```diff, ```patch, ```bash, ```sh, or any other code block
containing what should have been a tool argument. ...
```

**系统层** (硬约束) — 不指望 prompt 全靠. 在 `run()` 主循环加检测:

```python
_DODGED_TOOL_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell|diff|patch|json)\b",
    re.IGNORECASE,
)

def looks_like_dodged_tool_call(text: str) -> bool:
    """模型在 visible 里写了 ```bash/```diff 等代码块 — 大概率是想绕过 tool_calls."""
    return bool(_DODGED_TOOL_BLOCK_RE.search(text))
```

如果一轮没 tool_call 但 visible 里命中这个正则, **不立刻 DONE**, 而是注入一条 reminder user message 重来:

```python
if not res.actions:
    if looks_like_dodged_tool_call(res.visible):
        sess.codeblock_nag_count += 1
        if sess.codeblock_nag_count > self.CODEBLOCK_NAG_LIMIT:  # 默认 2
            _box(f"⛔ GAVE UP", ...); return
        _box(f"⚠️  CODEBLOCK NAG ({sess.codeblock_nag_count}/2)", ...)
        self.history.append(Message.user(_CODEBLOCK_REMINDER))
        continue
```

### 为什么禁这 6 个语言不禁别的

`bash / sh / shell / diff / patch / json` 这 6 个在 agent 场景下**几乎只在"绕过 tool_calls"时出现**. 用户不会问"教我写 diff" 或 "解释下这个 json schema". 误伤率极低.

`python / js / go / c++ / sql` 这些在"展示代码示例"场景里高频合法 (回答"教我写一段 Python"必然会用 ```python). 不在触发列表里, 防止误伤.

### 第一次实现踩的坑

最初我写的版本: **任何 tool_call 都重置 codeblock_nag_count**. 结果跑出灾难性振荡:

- Turn 1: dodge ```bash → NAG 1/2
- Turn 2: 模型调 todo (摆烂凑数, 不解决任何问题) → counter 重置为 0
- Turn 3: dodge ```bash → NAG 1/2 (又是 1/2!)
- Turn 4: 摆烂 todo → 重置
- ... 死循环到第 19 轮 MAX ROUNDS 才退出

原因: 一次合法 tool_call 不应"洗白"前面的嘴硬, 否则模型用 todo 摆烂就能永远绕过. 修了之后 (commit `580e9bb`):

- counter 在一次 `run()` 内**只增不减**
- 只在每个 `run()` **入口归零** (上个任务的累计不该惩罚下个任务)

实测后振荡场景在第 5 轮触发 GAVE UP, 不再 19 轮死循环.

---

## 6. PATH_ESCAPE 错误信息 — 不让模型在错误后放弃

### 现象

7B 模型有个倾向: 工具返回错误后, 把它当**任务终止信号**. 比如:

- 调 LS('/path/to/todo.py') → PATH_ESCAPE
- 不重试, 直接对用户说"please provide a path"

**模型把 PATH_ESCAPE 解读成了"用户给的路径错"而不是"我自己用错了"**.

### 修法

错误消息本身是引导. 改 `_safe_path` 抛的 ValueError 文案:

```python
raise ValueError(
    f"Path '{p}' is outside the workspace ({workdir}). "
    f"Use a path RELATIVE to the workspace — e.g. 'todo.py', "
    f"'src/foo.py', or '.' for the workspace root. "
    f"If you don't know what's there, call LS with no arguments first."
)
```

四个动作:
1. 说出 workspace 在哪
2. 说"用相对路径"
3. 给三个具体例子
4. 给恢复路径 (call LS 看看)

加上 SYSTEM 顶部已经有 WORKSPACE 段说明工作目录, 模型有两次接触到正确做法的机会 (启动时一次 + 出错时一次).

### 中间的尝试

讨论过加第三层"SYSTEM 加错误后必须重试的强制约束". 否决, 理由: SYSTEM 已经够长了, 加新的禁令时会摊薄前面已有的. **错误消息引导是最直接的兜底, 不引入新 prompt 噪音**.

### 起没起效果

7B 的失败模式不再是"放弃", 改成"乱试别的工具" (后来又催生了 §5 的 dodge 检测). 错误消息对 M2.7 几乎无影响 (它本来就会重试).

---

## 7. 流式 Read 和 SYSTEM 精简 — 两个事后优化

这俩不是"新功能", 是 v5 大改完成后扫一遍代码做的清洁工作. 但都修了真问题, 不是为改而改.

### 优化 G: 流式 Read

**问题**: `tool_read` 把整个文件读进内存:

```python
all_lines = p.read_text(...).splitlines()  # 全文加载
selected = all_lines[start:end]              # 然后切片
```

对 50MB 日志文件 + `Read(limit=100, offset=0)`, 仍然全文加载. Ollama 时代不显眼 (本地内存便宜), 接 MiniMax 后**输入 token 变贵** — 真不能浪费.

**修法**: 流式读到 `start + limit` 行就**停止 append**, 但**继续读到 EOF 计数行数** (拿到精确 total_lines):

```python
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, raw in enumerate(f):
        total += 1
        line = raw.rstrip("\r\n")  # 跨平台行尾
        if i < start:
            continue
        if limit is not None and len(selected) >= limit:
            continue  # 窗口满了不存, 但继续数 total
        selected.append(line)
```

50MB 文件 + Read(limit=100) 现在只占 100 行内存, 不是 50MB. total_lines 仍精确 (模型需要它来选下次 offset).

跨平台行尾问题: Windows 的 `\r\n` 用 `rstrip("\n")` 会留 `\r`. 改 `rstrip("\r\n")`. (这是 04 splitlines() 隐式处理的, 流式版本得显式).

### 优化 H: SYSTEM 精简

**问题**: SYSTEM 累积到 3635 字符 ≈ 1200 token. 每次 LLM 调用都重复传, MiniMax 输入端按 ¥2/M 算, 每 turn 烧 ¥0.0024 在 SYSTEM 上.

**怎么压**:

- 多句陈述句改成单句祈使句 ("READ BEFORE WRITE: edit_file / apply_patch / ... requires a prior Read on it.")
- 删掉"超过 200 行分块写"的段落 — 那是给 7B 写的, 对 M2.7 (输出预算大得多) 没用. 而且系统层有 chunked write 兜底, prompt 没必要再讲
- (a)/(b)/(c)/(d) 四点禁令压成一行列出语言. 真正起约束作用的是系统层 detection, 不是 SYSTEM 文字

**结果**: 3635 → 1739 字符, **砍 52%**, 每 turn 省 ~632 token. 5 turn 任务省 ¥0.006, 100 个任务/天省 ¥0.6.

更重要的是 SYSTEM 现在**可读**了 — 下次再加新约束时, 不会因为它已经 3635 字符就不敢动.

audit 验证关键短语没丢: WORKSPACE / PATH_ESCAPE / READ BEFORE WRITE / CONFLICT / apply_patch / todo / in_progress / TOOL-CALL PROTOCOL / ```bash / ```diff / RUN — 都在.

---

## 8. 安全脚手架 — .env 和 pre-commit hook

接了 MiniMax 就要管密钥. 几个原则:

### .env 模式

- 所有密钥从 `os.environ.get(...)` 读, 不出现在源码里
- `.env` 在 `.gitignore` 里, 永远不会 commit
- `.env.example` 进 git, 给协作者看格式 (占位符, 没真值)
- 写了一个 50 行的 `_load_dotenv()` (零依赖, 启动时自动读), 不引 `python-dotenv`

```python
def _load_dotenv(path: Path) -> None:
    """读 .env 文件, 把 KEY=value 填进 os.environ — 但不覆盖已存在的环境变量."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        ...
        if key and key not in os.environ:
            os.environ[key] = value
```

设计选择: 已存在的环境变量优先 — shell 里 `set MINICODE_BACKEND=ollama` 能临时覆盖 .env 里的设置.

### pre-commit hook

`.git/hooks/pre-commit` 是个 bash 脚本, 扫 staged diff 里有没有常见密钥前缀:

```bash
PATTERN='(sk-api-|sk-ant-|sk-or-|sk-proj-|sk-svcacct-|AIza[0-9A-Za-z_-]{20,}|ghp_[0-9A-Za-z]{20,}|gho_[0-9A-Za-z]{20,}|xox[bp]-)'
```

发现就拒绝 commit, 给修法指引. 测试用 fake `sk-api-FAKE...` 验证过, 正确挡住.

注意: hook 在 `.git/hooks/` 里, 是**每个 clone 独立**的. 别人 clone 你的仓库不会自动得到 — 这是 git 的设计 (出于安全考虑, hook 不进 repo). 如果想分发, 要么用 `husky` 类工具 (引依赖), 要么写个 `install_hooks.sh` 让用户主动跑. 我们目前接受 per-clone, 因为这是单人项目.

### 这事的背景

老实说, 这套防线是**从一次错误中学到的**: 我跑了 v5 接 MiniMax 那次, 用户在对话里**贴了真实密钥**两次. 我没办法把已经发出的对话撤回, 但可以保证未来不再发生 — `.env` + pre-commit hook + 代码永远从 env 读, 三层兜底.

---

## 9. 怎么跑起来

### 默认 Ollama (本地, 零成本)

需要先装 Ollama 桌面版 (`https://ollama.com/`), 拉默认模型:

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

然后:

```bash
cd 05-session-and-streaming/
python todo.py
```

启动 banner 显示 `backend: Ollama @ http://localhost:11434`.

### 切到 MiniMax (云端, 任务连续性好)

```bash
# 1. 复制 .env 模板:
cp .env.example .env

# 2. 编辑 .env, 填:
#    MINICODE_BACKEND=minimax
#    MINIMAX_API_KEY=<你的 key, 从 https://platform.minimaxi.com/ 拿>

# 3. (可选) 跑探针验证端点能用:
python scripts/probe_minimax.py

# 4. 跑:
python todo.py
```

启动 banner 显示 `backend: MiniMax @ https://api.minimaxi.com`.

`.env` 在 `.gitignore` 里, **永远不会被 commit**. pre-commit hook 也会扫密钥前缀挡住误操作.

切回 Ollama: 改 `.env` 的 `MINICODE_BACKEND=ollama` (或干脆删那行 — 默认就是 ollama).

### 环境变量一览

```
MINICODE_BACKEND=ollama|minimax    # 选后端, 默认 ollama
MINICODE_STREAM=1                  # 流式开关, 默认开 (设 0 强制非流式)
MINICODE_TIMEOUT=300               # HTTP 超时秒数, 首次加载模型可能慢

# Ollama
MINICODE_OLLAMA_URL=http://localhost:11434
MINICODE_MODEL=qwen2.5-coder:7b-instruct-q4_K_M

# MiniMax
MINIMAX_API_KEY=                   # 必填 (走 minimax 时)
MINIMAX_BASE_URL=https://api.minimaxi.com   # 不要带 /v1, client 内部拼
MINIMAX_MODEL=MiniMax-M2.7
MINIMAX_PRICE_IN=2.0               # ¥/M token, 用于成本估算
MINIMAX_PRICE_OUT=8.0
```

### REPL 命令

```
/help      显示帮助
/exit      退出
/clear     清空对话历史 (相当于 new_session)
/todos     显示当前 todo 列表
/history   显示消息条数统计
```

---

## 10. 已知问题 / 还能优化的地方

诚实交代当前状态. 这些是真实存在但暂时没修的:

### 已知问题

**A. NAG_THRESHOLD = 3 对 M2.7 偏严**

老规则 "连续 3 轮没调 todo 就注入提醒" 是给 7B 的. M2.7 单步任务能力强, 不需要 todo, 被 NAG 注入提醒后没被带偏但提醒本身是噪声. 可改成"按 backend 区分阈值"或"模型 N 轮里有 1 轮调过 todo 就不 nag".

**B. 模型选错工具时无系统层兜底**

7B 跑 query "解释 800-900 行" 时第一轮调了 `apply_patch` 传空 patch (用户从没要求修改). 系统层没拦, 空 patch 走 PARSE_FAILED 浪费一轮. 可加: tool_call 前置 sanity (apply_patch 的 patch 字段空 → 直接拒).

**C. todo 状态机没强约束**

一个 todo item 可以从 completed 变回 pending. 之前 7B 跑出"摆烂复读"的根因之一. state transitions 应当是单向 DAG: pending → in_progress → completed, 不能回退. 加状态转换校验即可.

**D. 错误码命名不统一**

`NO_MATCH` vs `NOT_FOUND`、`BAD_ARGS` vs `BAD_REGEX`、`AMBIGUOUS` 含义不显. 模型容易搞混. 但改起来要同步改 README + 测试断言, 收益不大, 一直没做.

### 真新功能 (v6 候选)

**E. Session 持久化**

跑 1 小时关掉 CLI 进度全丢. 加 `Session.to_dict()` / `from_dict()` + `/save name` / `/load name` 命令. ~80 行可做最简版.

**F. Context 压缩**

M2.7 有 200K context 暂时不紧, 但 Ollama 7B 是 32K, 长会话会撞墙. 滑动窗口最简单 (~50 行), 摘要复杂得多.

**G. 子 agent**

主 agent 派任务给子 agent, 子 agent 独立 session, 跑完返回摘要. Session 设计好了直接就能做, 但 7B 不适合做"派工的元认知". 留给云端模型 + 真实需求驱动时再做.

**H. prompt caching**

MiniMax 是否支持 cache_control / 隐式自动 caching, 没确认. 如果支持, SYSTEM 重复传的成本能再砍一半. 需要专门测.

---

## 11. 给后来者: 如果你想读源码

[todo.py](todo.py) 一个文件 ~2000 行. 推荐阅读顺序:

1. **顶部配置 + .env loader** (~150 行) — 看懂 BACKEND / 环境变量怎么进来的
2. **`ToolResult` + `Message` + `Tool` + `Session`** (~250 行) — 4 个核心数据结构
3. **`_OpenAICompatClient` + `OllamaClient` + `MinimaxClient`** (~150 行) — HTTP + SSE + reassembly
4. **`_StreamRenderer`** (~80 行) — 流式渲染细节, 关键陷阱注释里都有
5. **`ToolRegistry.dispatch`** (~50 行) — _session 注入 + read-before-write 守卫
6. **每个 tool_xxx 函数** — 平铺, 一个一个看, LS / Glob / Grep / Read / Write / Append / Edit / apply_patch
7. **`apply_patch` 三阶段** — 这是工具层最复杂的一段. 见 04 README 也讲过
8. **`ReActAgent.run` 主循环** — 把所有东西串起来

每个段落顶部都有 `# ===` 边框 + 中文标题. 跟着边框扫就能看清结构.

测试: [tests/test_tools.py](tests/test_tools.py) 66 个测试, 全部不依赖网络 / 模型, 跑 `pytest tests/` 一秒内出结果. 看测试比看代码更快理解工具的契约.

探针: [scripts/](scripts/) 有 `probe_stream_v3.py` 测 SSE 协议、`probe_minimax.py` 测真 MiniMax 端点. 不进生产代码, 但**有用**——以后 backend 行为变了 (比如 Ollama 升级了流式协议), 跑探针就知道.

---

## 历史包袱: 04 / 03 / 02 / 01 都做了什么

不在这份 README 重写 — 各自 README 仍然有效:

- [04-structured-tool-calls/](../04-structured-tool-calls/) — Ollama 后端 + structured tool_calls + apply_patch + 66 个测试. 是当前 v5 的直接前身
- [03-atomic-tools/](../03-atomic-tools/) — 三层工具架构 + 读后写乐观锁 + 42 个测试 + CI. 工具层第一次做扎实
- [02-sandboxed/](../02-sandboxed/) — REPL + bash 沙箱探测 + 大文件分片写入. 从一次性脚本进化成日常工具
- [01-bash-only/](../01-bash-only/) — 最小 ReAct + 5 个工具. 看清骨架最快

按版本号往回读, 能看清 agent 是怎么从"能跑"长到"能用"的. 每一版只引入一两个新概念, 改动可控.

---

## 一句话总结

**v5 不是加了什么新酷炫能力, 是把"用着别扭"的几个具体地方一个一个修了.** Session 让状态收口, 流式让等待变成观察, 双后端让 7B 不再是天花板, token 可见让花钱有数, 系统层兜底让 prompt 不再是孤军奋战.

每件事都是有理由地做, 也都留了"做得不完美的地方"在 §10. 这是一个真实的迭代记录, 不是产品宣传.
