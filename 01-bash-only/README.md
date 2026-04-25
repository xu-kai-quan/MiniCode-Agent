# 01-bash-only — 最小骨架, 30 行循环看清"agent 是怎么转的"

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **封存** (作为 v1 快照保留, 不再迭代). 后续工作看 [02-sandboxed](../02-sandboxed/) → [03](../03-atomic-tools/) → [04](../04-structured-tool-calls/) → [05](../05-session-and-streaming/).
>
> 一个文件 [todo.py](todo.py), **369 行**. 是整个仓库**最小**的 agent 实现.

这是 MiniCode Agent 第 1 版. 一句话: 给它一个任务, 它自己规划步骤、调工具 (执行命令 / 读写文件), 跑完就退. 适合**第一次接触 AI agent 概念**的人——所有东西摊在台面上, 没有任何框架封装.

## 读这个版本能学到什么

- **agent 循环到底是什么**——30 行代码看完, 没有黑盒
- **工具是怎么注册和调度的**——一个 dict, 一个 for 循环
- **模型怎么"请求"调用工具**——它输出一段文本, 系统正则解析
- **路径沙箱是怎么做的**——一个 `safe_path` 函数, 文件类工具调用前都过它
- **todo 工具为什么需要**——给小模型一份"外部记忆"

如果以上你都觉得"看一眼就懂"——直接跳去 [03](../03-atomic-tools/) 看怎么把工具层做扎实, 或者 [05](../05-session-and-streaming/) 看 agent 怎么变成"能用的体感".

---

## 目录

- [1. agent 循环 — 30 行就是全部本质](#1-agent-循环--30-行就是全部本质)
- [2. 5 个工具 — 一个 dict, 一个 for 循环](#2-5-个工具--一个-dict-一个-for-循环)
- [3. 工具调用解析 — Qwen XML + 正则](#3-工具调用解析--qwen-xml--正则)
- [4. 路径沙箱 — 一个函数, 三个工具用](#4-路径沙箱--一个函数-三个工具用)
- [5. todo 工具 — 给小模型一份外部记忆](#5-todo-工具--给小模型一份外部记忆)
- [6. nag 机制 — 强制模型保持规划习惯](#6-nag-机制--强制模型保持规划习惯)
- [7. 跑起来长什么样](#7-跑起来长什么样)
- [8. 怎么跑](#8-怎么跑)
- [9. 读源码的建议路线](#9-读源码的建议路线)
- [10. 已知局限 (有的后续版本已修)](#10-已知局限-有的后续版本已修)
- [11. v1 → v2 的演化方向](#11-v1--v2-的演化方向)

---

## 1. agent 循环 — 30 行就是全部本质

普通调用大模型: 你问一句, 它答一句, 结束.

Agent 是这样: 你问一句, 模型答一句**带着工具调用**, 系统真的去执行那个工具, 把结果再喂回模型, 模型看到结果再决定下一步…… 直到模型说"我做完了".

简单说就是一个 `while` 循环. 在 [todo.py](todo.py) 的 `agent_loop` 函数里:

```python
def agent_loop(query: str):
    tok, mdl, device = load_model()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    for turn in range(1, MAX_ROUNDS + 1):
        # 1) 模型生成
        raw = generate(tok, mdl, device, messages)
        visible = strip_think(raw)   # 剥掉 <think>...</think>

        # 2) 解析工具调用
        calls = parse_tool_calls(visible)
        messages.append({"role": "assistant", "content": visible,
                         "tool_calls": [...]})

        if not calls:
            return    # 模型不再调工具, 完事

        # 3) 依次执行工具
        for c in calls:
            handler = TOOL_HANDLERS[c["name"]]
            output = handler(**c["arguments"])
            messages.append({"role": "tool", "content": str(output)})
```

**这就是 agent 的全部本质**. 没有 langchain, 没有 autogen, 没有"agent 框架". 一个 while 循环, 一个 if-else 早退, 一个 for 调 handler.

后面所有版本都是在这个基础上加东西——v2 加 REPL 让你能连续提问, v3 把工具返回标准化, v4 换更大模型 + 跨文件原子改动, v5 加流式 + 状态管理 + 双后端. **但循环本身一直是这 30 行**.

---

## 2. 5 个工具 — 一个 dict, 一个 for 循环

工具就是"模型可以请求系统帮它做的事". v1 内置 5 个:

| 工具名 | 干什么 | handler 大约多少行 |
|---|---|---|
| `bash` | 执行任意 shell 命令, 30 秒 timeout, 输出截断到 8000 字符 | 3 行 |
| `read_file` | 读取一个文本文件, 可选 `limit` 行数上限 | 5 行 |
| `write_file` | 创建或覆盖文件, 自动建父目录 | 5 行 |
| `edit_file` | 把文件里某段唯一文本精确替换 (歧义时拒绝) | 7 行 |
| `todo` | 写/更新一个待办列表, 强制最多一个 in_progress | 见 §5 |

**注册方式**——一个 dict 就完事:

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}
```

**调度也是一行**:

```python
handler = TOOL_HANDLERS.get(name)
output = handler(**c["arguments"])
```

模型并不能直接动你的电脑——它只能"输出一段表示工具调用的文本", 由系统真正执行. 这是一道很重要的安全边界.

### 工具返回的是裸字符串

注意 v1 的工具 handler 都返回 `str`, 不像 v3+ 用结构化的 `ToolResult`:

```python
def run_write(path: str, content: str) -> str:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"
```

`str` 简单——直接喂回模型. 但**不能区分** "成功 / 截断了 / 失败了"——只能靠模型读 text 里的关键字猜. v3 加 ToolResult 就是为了解决这个 (见 [03 README §2](../03-atomic-tools/README.md#2-toolresult-统一返回结构)).

工具里所有异常**都被 try/except 吞掉**, 转成 `f"error: {e}"` 返回给模型——模型看到 "error" 字样就会自己想办法. 直接抛异常会让 agent_loop 崩, 不优雅.

### 这套工具集是怎么选出来的

5 个工具, 不多不少. 标准是"覆盖一个最简单编码任务所需的最小操作集":

- 想"改一行代码" → `edit_file` 够
- 想"建一个新文件" → `write_file` 够
- 想"看现有代码" → `read_file` 够
- 想"运行测试 / git status / pip install" → `bash` 兜底
- 想"分多步做" → `todo` 规划

但这套**有缺**:
- 没文件搜索 (LS / Glob / Grep)——找文件要靠 `bash ls` / `bash find` / `bash grep`, **工具名在骗模型** (跨平台不一样, 输出格式不固定). v3 补了原子搜索三件套
- write_file 不分片——大文件写不下. v2 加了 append_file
- edit_file 一次只改一处——多文件改动要多次调用. v4 加了 apply_patch

这些都是**真实跑出来才发现的**, v1 时还不知道.

---

## 3. 工具调用解析 — Qwen XML + 正则

模型不能直接"调函数"——它输出文本, 文本里嵌着调用请求. Qwen 用 XML 表达:

```xml
<tool_call>
  <function=write_file>
    <parameter=path>hello.py</parameter>
    <parameter=content>print("hi")</parameter>
  </function>
</tool_call>
```

我们用正则三件套抠出来:

```python
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for tc in TOOL_CALL_RE.findall(text):
        fm = FUNCTION_RE.search(tc)
        if not fm:
            continue
        name = fm.group(1).strip()
        args = {k: _coerce(v) for k, v in PARAM_RE.findall(fm.group(2))}
        calls.append({"name": name, "arguments": args})
    return calls
```

### `<think>...</think>` 也要剥

Qwen 会在正式回答前写一段 `<think>...</think>`, 那是它的内部推理. 我们打印它给你看 (帮助理解), 但**不存回对话历史**——否则历史会被前几轮的 think 越塞越重:

```python
def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

### 为什么需要 `_coerce`

XML 把所有 parameter 值当字符串. 但 `todo` 的 `items` 参数应该是 list, `limit` 应该是 int. 我们做一个轻度类型推断:

```python
def _coerce(value: str):
    s = value.strip()
    if (s and s[0] in "[{") or s in ("true", "false", "null"):
        try:
            return json.loads(s)    # JSON-shaped, 试着 parse
        except Exception:
            return value
    if s.isdigit():
        return int(s)
    return value
```

这是 XML 解析的硬伤之一——v4 换 OpenAI 结构化 `tool_calls` 后, arguments 直接是 JSON 字符串 (`"{\"path\":\"a.py\"}"`), 一个 `json.loads` 就完事. 见 [04 README §2](../04-structured-tool-calls/README.md#2-工具调用走-openai-结构化协议--扔掉-xml-解析).

### XML 的脆弱性

XML 解析靠模型严格闭合标签. 模型偶尔少个 `</function>`、parameter 里嵌套了 `<`/`>` 字符没转义, 整个 tool_call 就解析失败. 这种问题 v1 时还不严重 (任务简单), v2 跑长任务才暴露——见 02 README.

---

## 4. 路径沙箱 — 一个函数, 三个工具用

模型理论上可以请求写到任何路径. 万一它想 `write_file("C:/Windows/system32/...")` 怎么办?

文件类工具 (`read_file` / `write_file` / `edit_file`) 都先过一次:

```python
WORKDIR = Path.cwd().resolve()      # 启动时锁定

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

不在 WORKDIR 之下的, 直接抛异常 (在 agent_loop 里被 try/except 转成 error 字符串返回给模型). 模型的破坏范围被锁在启动时的 cwd 里.

### bash 工具是个例外

`bash` 不过 `safe_path`——既然叫 bash, 就把权力交给 shell 了, 它能 `cd ..` 出去. **这是有意的取舍**: bash 是"兜底工具", 既然要用就接受它的危险. 想要更安全的话, 别用 bash (后续版本都有更结构化的工具).

### 一个原则: 沙箱在工具层做, 不在 prompt 里写

不要在 SYSTEM 里写"请不要读 cwd 外的文件". 小模型不会乖乖听 prompt——但 `safe_path()` 函数会拦住. **prompt 是软约束, 代码是硬约束**——这是整个仓库的核心设计原则, 后续每一版都遵守.

---

## 5. todo 工具 — 给小模型一份外部记忆

### 为什么需要

小模型 (2B 参数) 在多步任务里很容易**走神**:
- 对话越长, 它越会忘记最初的目标
- 它会重复做已经做过的事
- 它做完一半就突然"觉得自己完事了"

### 怎么解决

让模型**先把步骤写下来**. 写下来的东西会留在对话历史里, 之后每一轮模型都能看到, 相当于一份"外部记忆":

```python
class TodoManager:
    def __init__(self):
        self.items: list[dict] = []

    def update(self, items: list[dict]) -> str:
        in_progress = sum(1 for it in items if it.get("status") == "in_progress")
        if in_progress > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = [
            {"id": it["id"], "text": it["text"], "status": it.get("status", "pending")}
            for it in items
        ]
        return self.render()

    def render(self) -> str:
        mark = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        return "\n".join(f"{mark[it['status']]} {it['text']}" for it in self.items)
```

两条规则:

- **任何超过一步的任务, 必须先调 `todo` 列出步骤**——SYSTEM prompt 里写
- **同一时刻只能有一个步骤是 `in_progress`**——TodoManager 在系统层强制 (违反就抛异常)

### "强制单 in_progress" 是个软规则但很有用

为什么强制单 in_progress? 因为这逼模型"一件一件来". 它没法同时把 5 件事都标 in_progress 然后混着做——必须明确"现在我做的是哪一件".

**这是 prompt 教不会的——模型会"为了进度好看"把多个标 in_progress**. 在系统层强制就靠谱.

---

## 6. nag 机制 — 强制模型保持规划习惯

光有 todo 工具还不够——模型可能调一次 todo 之后就把它**忘了**, 后面 5 轮全在 bash 折腾, todo 列表停留在 "all pending" 状态.

agent_loop 里有个计数器:

```python
rounds_since_todo = 0

for turn in range(1, MAX_ROUNDS + 1):
    ...
    used_todo = any(c["name"] == "todo" for c in calls)
    rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

    if rounds_since_todo >= 3:
        # 注入一条 user 消息: 该更新规划了
        messages.append({
            "role": "user",
            "content": "<reminder>Update your todos.</reminder>",
        })
        rounds_since_todo = 0
```

连续 3 轮没调 todo → 系统插一条 user role 的提醒消息, 强迫模型回到"看一下 todo 列表 + 更新状态"的节奏.

**这又是"prompt 是软约束, 系统是硬约束"的一个体现**. SYSTEM prompt 里写"keep todos updated" 模型听不到三轮, 系统层注入 reminder 才听得见.

---

## 7. 跑起来长什么样

```sh
python todo.py "在 mypkg/ 下建一个包, 包含 __init__.py 和一个 add(a,b) 函数"
```

它会自动:

1. 先思考一下 (`<think>` 块——"我应该先建目录, 再写两个文件")
2. 调用 `todo` 工具把步骤列出来
3. 调用 `bash` 创建目录
4. 调用 `write_file` 写入 `__init__.py`
5. 调用 `write_file` 写入 `utils.py`
6. 把 todo 项全部标记为完成
7. 退出 (没有 `>` 提示符——这一版是一次性模式)

整个过程你能在终端里**看见每一步**: 模型在想什么、决定做什么、工具实际跑出了什么. 用了一组朴素的 Unicode box 字符画框框.

**注意没有 REPL**——v1 跑完就退. v2 才加了交互模式 (REPL + 状态保持).

---

## 8. 怎么跑

### 准备

1. **下载模型**到本地 (Qwen3.5-2B). 默认查找路径是 `E:/MYSELF/model/qwen/Qwen3.5-2B/`. 改路径直接改 [todo.py](todo.py) 顶部的 `MODEL_DIR` 常量 (v1 还没支持 env var, v3 才支持)
2. **装依赖**:
   ```sh
   # 先按平台/CUDA 装合适的 torch wheel: https://pytorch.org/get-started/locally/
   pip install torch transformers safetensors
   ```
3. **有 NVIDIA 显卡更好** (CUDA), 没有也能 CPU 跑, 慢很多

### 启动

进到**你希望它工作的目录** (所有文件操作都被锁在启动时的 cwd 里), 然后:

```sh
# 用内置默认任务跑一次
python path/to/01-bash-only/todo.py

# 自己写个任务
python path/to/01-bash-only/todo.py "把 hello.py 改成带类型提示和 main guard 的版本"
```

模型不再调用工具时自动退出, 或者跑满 20 轮强制退出. **没有交互式对话**——一次只跑一个任务, 跑完就退.

### 想调点什么

[todo.py](todo.py) 顶部几行就是全部旋钮:

```python
MODEL_DIR = Path("E:/MYSELF/model/qwen/Qwen3.5-2B")
WORKDIR = Path.cwd().resolve()    # 工作目录, 启动时锁定
SYSTEM = "..."                     # 给模型看的系统提示词
MAX_ROUNDS = 20                    # 最多跑几轮就强制退出
MAX_NEW_TOKENS = 1024              # 单轮最多生成多少 token
```

`MAX_NEW_TOKENS = 1024` 是 v1 的痛点之一——单轮 token 预算不够时, 模型生成的 `<tool_call>` 会被半截截掉, 看起来像"完成"了但文件根本没写. v2 加了截断检测.

---

## 9. 读源码的建议路线

[todo.py](todo.py) 369 行, 按这个顺序看最高效:

1. **`agent_loop` 函数** (文件末尾): 30 行循环就是 agent 的全部本质
2. **`TOOL_HANDLERS` 字典**: 工具是怎么登记和分发的——一个 dict, 一行 lambda
3. **`run_xxx` 一族函数**: 一个具体工具长什么样. 全部 3-7 行
4. **`safe_path`**: 路径沙箱的全部实现, 4 行
5. **`parse_tool_calls`**: 模型输出的 XML 是怎么变成结构化调用的
6. **`TodoManager`**: 规划机制怎么实现, 为什么强制单 in_progress
7. **`extract_think` / `strip_think`**: think 块怎么剥
8. **`load_model` / `generate`**: torch + transformers 加载和推理 (后续 v4 换 Ollama 整段不用了)

整个文件没有用任何 agent 框架——你看到的就是真相, 没有黑盒.

---

## 10. 已知局限 (有的后续版本已修)

| 局限 | 后续状态 |
|---|---|
| 跑完就退, 没 REPL | ✅ v2 加了交互模式 + 状态持久 |
| Windows 下 `bash` 工具其实是 `cmd.exe`, `ls` 报错 | ✅ v2 探测 Git for Windows 的 bash |
| 写长文件单轮 token 不够会被截断 | ✅ v2 加 `append_file` + 截断检测 |
| 工具返回裸 str, 无法区分 success/partial/error | ✅ v3 加 `ToolResult` 三态结构 |
| 没文件搜索 (LS / Glob / Grep) | ✅ v3 加原子搜索三件套 |
| 写文件没读后写保护, 模型可能凭记忆乱改 | ✅ v3 加读后写 + 乐观锁 |
| 没自动测试, 改动靠手动跑 REPL 验证 | ✅ v3 加 42 个 pytest |
| XML 解析脆弱 (模型少个尾标签就崩) | ✅ v4 换 OpenAI 结构化 `tool_calls` |
| 多文件改动要多次 edit_file 容易半路丢 | ✅ v4 加 `apply_patch` |
| 本地 torch 跑 2B 已经吃力, 换更大模型困难 | ✅ v4 换 Ollama, v5 加 MiniMax 云端切换 |
| 等模型几秒看不到状态 | ✅ v5 加流式输出 |
| 模型不知道工作目录在哪 (LS 用占位符路径) | ✅ v5 把 cwd 注入 SYSTEM |
| 跑云端不知道烧了多少钱 | ✅ v5 每 turn 显示 token + 估算成本 |
| 小模型用代码块伪装 tool_call 死循环 | ✅ v5 加系统层检测 + GAVE UP |
| `bash` 不在路径沙箱内, 能 `cd ..` 出工作区 | ⏸ 设计权衡, 后续版本都这样 |
| 小模型 (2B) 在多步任务能力上限 | ⚠️ v4 换 7B 改善, v5 加云端 M2.7 解决 |

---

## 11. v1 → v2 的演化方向

v1 把"agent 循环 + 工具表 + 沙箱 + todo + nag"凑齐了, 跑起来. 但实际跑下来, 有几个具体的"用着别扭"的地方:

- **跑完就退, 想接着问就要重新跑** → v2 加 REPL, 模型加载一次反复用
- **Windows 下 `ls` 报错**——工具名叫 bash 但实际是 cmd → v2 探测 Git Bash
- **写 500 行 HTML 单轮写不完, 半截截断** → v2 加 `append_file` + 截断检测

每一条都是 v1 实际跑过才发现的问题. v2 README 把这三件事的"为什么、怎么、起没起效果"都写下来了——见 [02-sandboxed/README.md](../02-sandboxed/README.md).

再往后的演化:

- **v3** ([03-atomic-tools](../03-atomic-tools/README.md)) — 把工具层做扎实: 三层架构 / ToolResult 统一返回 / LS-Glob-Grep / 读后写乐观锁 / 42 个 pytest
- **v4** ([04-structured-tool-calls](../04-structured-tool-calls/README.md)) — 换 Ollama + OpenAI 结构化 tool_calls + apply_patch 跨文件原子改动
- **v5** ([05-session-and-streaming](../05-session-and-streaming/README.md)) — Session 状态管理 + 流式输出 + 双后端 + token/成本可见 + 系统层兜底

按版本号顺序读, 能看清"agent 是怎么从最小骨架长到能用的工具"的每一步.
