# 02-sandboxed — 从一次性脚本变成"能开机就用"的工具

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **封存** (作为 v2 快照保留, 不再迭代). 后续工作看 [03-atomic-tools](../03-atomic-tools/) → [04](../04-structured-tool-calls/) → [05](../05-session-and-streaming/).
>
> 一个文件 [todo.py](todo.py), **570 行** (v1 是 369 行).

这是 MiniCode Agent 第 2 版. v1 的最小骨架跑得起来, 但实际用了几次发现"用着别扭"——v2 把这些别扭的地方一个一个修, 让 agent 从"跑一次的脚本"变成"打开窗口能连续干活的助手".

## 它跟 v1 的区别 — 一句话版本

**v1 是脚本, v2 是工具**.

v1 的痛点全是**重复使用时**才发现的:

- **跑完就退**——想接着问就要重新启动, 模型重新加载几十秒
- **Windows 下 `bash ls` 报"不是内部命令"**——工具名在骗模型, 跨平台体验崩
- **写 500 行 HTML 单轮 token 不够, 半截截断**——模型以为完成了, 文件实际没写完
- **模型脑补下一轮**——回答完一轮后继续生成 `<|im_start|>user...` 把 max_new_tokens 用满

v2 一个一个修. 这份 README 主要面向想看清"agent 怎么从能跑长成能用"的过程的读者.

---

## 目录

- [1. REPL — 让模型加载一次反复用](#1-repl--让模型加载一次反复用)
- [2. bash 找 Git for Windows — 不让工具名骗模型](#2-bash-找-git-for-windows--不让工具名骗模型)
- [3. append_file 分片写 — 大文件不再半截截断](#3-append_file-分片写--大文件不再半截截断)
- [4. 截断检测 — 系统兜底捞回半截 tool_call](#4-截断检测--系统兜底捞回半截-tool_call)
- [5. `_stop_token_ids` — 不让模型脑补下一轮](#5-_stop_token_ids--不让模型脑补下一轮)
- [6. 跑起来长什么样](#6-跑起来长什么样)
- [7. 核心概念 (写给新读者)](#7-核心概念-写给新读者)
- [8. 怎么跑](#8-怎么跑)
- [9. 读源码的建议路线](#9-读源码的建议路线)
- [10. 已知局限 (有的后续版本已修)](#10-已知局限-有的后续版本已修)
- [11. v2 → v3 的演化方向](#11-v2--v3-的演化方向)

---

## 1. REPL — 让模型加载一次反复用

### v1 是什么样

```python
# v1 的入口
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "..."
    agent_loop(q)
```

`agent_loop` 里面 `load_model()` 加载模型, 跑完任务后函数返回, 进程退出. 想接着问就得重新跑——又要等几十秒加载模型.

### v2 怎么改

把 agent_loop 拆成两层:

- **内层** `run_turn(state, query)` — 把一条用户输入跑到模型不再调工具为止. 接收 state dict 共享 messages 和已加载的 tok/mdl/device
- **外层** `repl()` — 持续读用户输入, 维护 state, 拦截斜杠命令

```python
def repl():
    state = {
        "tok": ..., "mdl": ..., "device": ...,    # 加载一次
        "messages": [{"role": "system", "content": SYSTEM}],
        "rounds_since_todo": 0,
    }
    while True:
        line = input("> ").strip()
        if line.startswith("/"):
            # 拦截斜杠命令: /exit /clear /todos /history /help
            ...
        run_turn(state, line)    # 共享 state, 模型不重新加载
```

**关键点**: state dict 持有 `messages` 列表, 每次 `run_turn` 调用追加新内容. 模型记得之前所有对话. 你可以问"刚才那个文件再加个函数", 它知道"刚才那个文件"是哪个.

### 斜杠命令

```
/help     显示帮助
/exit     退出 (Ctrl-D / Ctrl-C 同效)
/clear    清空对话历史和 todo, 从头开始
/todos    看当前待办列表
/history  看消息条数统计
```

`/clear` 重置 state 但不重新加载模型——比退出再启快得多.

### 一次性模式仍保留

```python
if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        run_turn(state, q)    # 跑一次就退
    else:
        repl()                # 进交互模式
```

CI / 脚本场景仍能用 `python todo.py "task"` 一次性跑完.

### 起没起效果

显著. v1 用一次重启一次的痛感消失了. 但**新问题来了**——长会话会让 messages 越塞越大, 模型生成越来越慢 (每轮重新 tokenize 完整历史). v3 / v4 / v5 一直在跟这个问题斗 (v5 加了 token 显示让你看到成本压力).

---

## 2. bash 找 Git for Windows — 不让工具名骗模型

### v1 是什么样

```python
def run_bash(command: str) -> str:
    r = subprocess.run(command, shell=True, ...)
```

`shell=True` 在 Windows 上调 `cmd.exe`. 模型如果写 `ls -la` 或 `cat foo.py` 或 `grep ...`——全部失败:

```
'ls' 不是内部或外部命令、可运行的程序或批处理文件.
```

**工具叫 bash, 实际不是 bash**——模型输出的命令按 bash 习惯写, 系统按 cmd 解释, 结果常常是错的. 模型很困惑, 因为它"明明用 bash 命令啊".

### v2 怎么改

启动时探测 Git for Windows 的 `bash.exe`:

```python
def _find_bash() -> str | None:
    """探测 Git for Windows 的 bash.exe. 排除 WSL 的 System32\\bash.exe — 它输出 UTF-16
    会和 Python 默认解码冲突, 而且工作区路径语义也不同 (WSL 里 E:\\ → /mnt/e/)."""
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"E:\software\Git\usr\bin\bash.exe",
        r"E:\software\Git\bin\bash.exe",
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    # 也允许 PATH 上非 System32 的 bash (例如用户自己装的 MSYS2)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None

BASH_EXE = _find_bash()
```

找到就用 Git Bash, 找不到就退回 cmd. **横幅里明确显示当前用的是哪个 shell**, 让用户一眼看见:

```
shell  : C:\Program Files\Git\bin\bash.exe
# 或者
shell  : (cmd.exe 回退)
```

### 为什么故意不用 WSL bash

WSL (Windows Subsystem for Linux) 也有 bash, 在 `C:\Windows\System32\bash.exe`. 但**故意不用**:

1. **输出 UTF-16 编码**——Python `subprocess` 默认按 UTF-8 解码, 会乱码
2. **路径语义不同**——`E:\` 在 WSL 里是 `/mnt/e/`, 工作目录对不上
3. **环境隔离**——你 Windows 装的 git / python 在 WSL 里看不见

WSL 是好东西, 但跟 agent 沙箱组合在一起, **引入的麻烦比解决的多**. 显式排除掉它.

### 起没起效果

模型用 bash 命令的成功率从经常崩 → 在装了 Git Bash 的机器上几乎不崩. 没装 Git Bash 的也能 fallback 到 cmd 跑——只是模型的 Unix 命令仍会失败, 它会从错误信息看出来重试.

后续 v3 / v4 / v5 沿用这套探测. v4 加了 Ollama 后, 跨平台问题相对小一些 (后端跑在另一个进程里), 但 bash 工具本身仍依赖 Git Bash 才能跑 Unix 命令.

---

## 3. append_file 分片写 — 大文件不再半截截断

### 痛点

模型一次生成的 token 数量有上限 (`MAX_NEW_TOKENS`, v1 是 1024, v2 提到 4096). 要写一个 500 行的 HTML, 一轮可能写不完.

v1 的表现: write_file 调用被截在中间, 看起来像"完成"了 (parse 出 1 个 tool_call), 但 file content 里少了下半截. 落盘文件**就是个半截**, 模型也不知道.

### v2 加了第 6 个工具: append_file

```python
def run_append(path: str, content: str) -> str:
    """追加文本到文件末尾, 不存在则创建. 用于分片写入超出单轮 token 预算的大文件."""
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    total = p.stat().st_size
    return f"appended {len(content)} chars to {path} (now {total} bytes)"
```

工具表里多一行:

```python
TOOL_HANDLERS = {
    "bash":        ...,
    "read_file":   ...,
    "write_file":  ...,
    "append_file": lambda **kw: run_append(kw["path"], kw["content"]),    # 新
    "edit_file":   ...,
    "todo":        ...,
}
```

### SYSTEM 教模型怎么用

```
For files longer than ~200 lines, write in chunks: start with
write_file for the first chunk, then call append_file repeatedly
for each remaining chunk (roughly 150 lines per chunk), preserving
newlines.
```

模型遇到大文件时, 调 write_file 写第一段, 然后多次调 append_file 加剩下的. 每次都是单独的 tool_call, 不会被一次 token 限制截断.

### 但 prompt 是软约束——模型可能不听

模型可能仍然忘了用 append_file, 直接 write_file 写大文件. **怎么办?** —— 见下一节 §4 截断检测.

---

## 4. 截断检测 — 系统兜底捞回半截 tool_call

### 现象

模型尝试 `write_file('long.html', '...')` 写一个长内容. 单轮 token 写不下, 输出形如:

```
<tool_call>
  <function=write_file>
    <parameter=path>long.html</parameter>
    <parameter=content><html>...
... (这里被 MAX_NEW_TOKENS 砍掉, 没有 </parameter></function></tool_call>
```

`parse_tool_calls` 用正则匹配 `<tool_call>...</tool_call>` 闭合对——这个不闭合所以拿不到. 系统看到 "0 个 tool_call" → 以为模型完事了 → 退出循环. 文件**根本没写**.

### v2 怎么修

在 `run_turn` 里, **当解析不到 tool_call 时**, 多看一眼: 文本里有没有 `<tool_call>` 开头?

```python
calls = parse_tool_calls(visible)
truncated = visible.count("<tool_call>") > len(calls)

if not calls:
    if truncated:
        # 注入提醒, 不算"完成"
        messages.append({
            "role": "user",
            "content": (
                "<reminder>Your previous tool call was cut off before "
                "</tool_call>. Do not repeat the same oversized write_file. "
                "Instead: call write_file with only the FIRST ~150 lines, "
                "then call append_file repeatedly for the remaining chunks.</reminder>"
            ),
        })
        continue   # 不退出, 让模型重来
    return    # 真的完事了
```

这个机制是**系统层的硬约束**——不管 SYSTEM 怎么教, 只要检测到截断, 系统就强制重来.

### 这体现一个更大的设计原则

**prompt 是软约束, 系统是硬约束**.

- 教模型用 append_file → 在 SYSTEM 里写 (软)
- 检测到 tool_call 被截断 → 在系统层捕捉并注入 reminder (硬)

**关键不变量必须在系统层兜底**, 因为小模型不一定听 prompt. 这条原则贯穿后续所有版本——v3 的读后写乐观锁、v4 的 apply_patch 原子回滚、v5 的代码块伪 tool_call 检测, 都是同一个路数.

### 起没起效果

显著. 长文件场景下, 模型写一半被截 → 系统注入提醒 → 模型改用分片 → 文件完整落盘. 用户看到的是中间一个 `⚠️ TRUNCATED` 框框, 然后任务正常完成.

---

## 5. `_stop_token_ids` — 不让模型脑补下一轮

### v1 时的怪现象

模型回答完一轮 (assistant 角色), 应该停下让系统执行 tool_call. 但 v1 经常看到:

```
<tool_call>...</tool_call>
<|im_start|>user
请确认上述操作.
<|im_start|>assistant
好的我已经...
```

模型在自己的回复里**继续脑补下一轮的 user 和 assistant**, 把 max_new_tokens 占满. 解析时这些"幻觉对话"被 strip_think + parse_tool_calls 抠掉一部分, 但仍然污染对话历史.

### 根因

Qwen 用 `<|im_start|>` / `<|im_end|>` 作为对话边界. `<|im_end|>` 标记一轮结束. v1 的 generate() 只把 `tok.eos_token_id` 当停止标记, **不包括** `<|im_end|>`——所以模型生成完 `<|im_end|>` 不会停, 继续往下写.

### v2 怎么修

```python
def _stop_token_ids(tok) -> list[int]:
    """Qwen 在一轮 assistant 结束时写 <|im_end|>; 把它作为额外的停止 token,
    防止模型脑补下一轮的 `user`/`<tool_response>` 继续写满 max_new_tokens."""
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tok.convert_tokens_to_ids(marker)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return list(ids)

def generate(tok, mdl, device, messages):
    ...
    out = mdl.generate(
        ...,
        eos_token_id=_stop_token_ids(tok),    # 关键这里
    )
```

把 `<|im_end|>` 和 `<|endoftext|>` 加进 stop tokens. 模型生成到 `<|im_end|>` 就停, 不再脑补.

### 起没起效果

显著. 模型不再生成"幻觉下一轮", token 预算实际用在该用的地方 (一轮的 tool_call), 长输出时 MAX_NEW_TOKENS 利用率高很多.

后续 v4 换 Ollama 后, 这个问题不复存在——Ollama 的 chat completions 协议自己处理边界, 不需要我们指定 stop tokens.

---

## 6. 跑起来长什么样

启动:

```sh
python todo.py
```

看到横幅 (模型 / 设备 / 工作目录 / shell):

```
┌─────────────────────────────────────────────────────────────────────┐
│   model   : qwen3.5-2b
│   device  : cuda
│   workdir : E:\my-project
│   shell   : C:\Program Files\Git\bin\bash.exe
└─────────────────────────────────────────────────────────────────────┘

> 帮我建一个 mypkg, 里面放 __init__.py 和一个 add(a,b) 函数
```

它会:

1. 先思考 (`<think>` 块——"我应该先建目录, 再写两个文件")
2. 调 `todo` 工具列出步骤
3. 调 `bash` 建目录
4. 调 `write_file` 写文件
5. 把 todo 项标记完成

干完之后**回到 `>` 提示符**, 你可以继续问别的, 它**记得**刚才发生了什么 (这是 v1 没有的). 输入 `/exit` 退出.

---

## 7. 核心概念 (写给新读者)

如果你跳着读到这里——v2 的核心概念跟 v1 一样, 因为 agent 循环和工具表的本质没变. 这里只放最小集合, 详细解释在 [01 README](../01-bash-only/README.md):

### 7.1 什么叫 "agent"

普通调用大模型: 你问一句, 它答一句, 结束.

Agent 是这样: 你问一句, 模型答一句**带着工具调用**, 系统真的去执行那个工具, 把结果再喂回模型, 模型看到结果再决定下一步…… 直到模型说"我做完了".

简单说就是一个 `while` 循环:

```python
while True:
    回复 = 模型(对话历史)
    工具调用列表 = 解析(回复)
    if 没有工具调用:
        break
    for 调用 in 工具调用列表:
        结果 = 执行(调用)
        对话历史.append(结果)
```

v2 在外面**多套了一层 REPL** 让你能连续提问, 但内层循环还是同样的.

### 7.2 工具是什么

工具就是"模型可以请求系统帮它做的事". v2 内置 6 个 (v1 是 5 个, 加了 append_file):

| 工具 | 干什么 |
|---|---|
| `bash` | 执行 shell 命令 (优先用 Git Bash, 没有才退回 cmd) |
| `read_file` | 读取一个文本文件 |
| `write_file` | 创建或覆盖文件 |
| `append_file` | 追加内容到文件末尾 (用来分片写大文件) — **v2 新增** |
| `edit_file` | 把文件里某段文本精确替换 |
| `todo` | 写/更新一个待办列表 |

模型并不能直接动你的电脑——它只能"输出一段表示工具调用的文本", 由系统真正执行. 这是一道很重要的安全边界.

### 7.3 todo 工具 — 给小模型一份外部记忆

模型在多步任务里很容易**走神**. 让它**先把步骤写下来**, 写下来的东西会留在对话历史里, 之后每一轮都能看到, 相当于"外部记忆":

- 任何超过一步的任务, 必须先调 `todo` 列出步骤
- 同一时刻只能有一个步骤是 `in_progress`, 强制它**一件一件来**

如果模型连续 3 轮没更新 todo, 系统主动注入提醒 ("该更新规划了"). 这个机制叫 **nag**.

### 7.4 沙箱 — 把破坏范围锁在工作目录

文件类工具都先做一次**路径检查**:

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

不在工作目录之下的, 直接拒绝. 模型的破坏范围被锁在启动时的 cwd 里.

`bash` 工具是个例外——既然叫 bash, 就把权力交给 shell 了, 它能 `cd ..` 出去. 这是有意的取舍.

### 7.5 `<think>` 和 `<tool_call>`

Qwen 会在正式回答前写一段 `<think>...</think>`, 那是它的内部推理. 我们打印它给你看 (帮助理解), 但**不存回对话历史**——否则历史会越塞越重.

`<tool_call>...</tool_call>` 才是真正请求系统执行的部分, 系统用正则把它抠出来, 找到对应的工具函数执行. (这套 XML 协议在 v4 换成了 OpenAI 结构化 `tool_calls` 字段, 见 [04 README §2](../04-structured-tool-calls/README.md#2-工具调用走-openai-结构化协议--扔掉-xml-解析).)

---

## 8. 怎么跑

### 准备

1. **下载模型**到本地 (Qwen3.5-2B). 默认放在 `E:/MYSELF/model/qwen/Qwen3.5-2B/`. 要改路径, 编辑 [todo.py](todo.py) 顶部的 `MODEL_DIR` (v3 起支持 env var)
2. **装依赖**:
   ```sh
   # 先按平台/CUDA 装合适的 torch wheel: https://pytorch.org/get-started/locally/
   pip install torch transformers safetensors
   ```
3. **推荐装 Git for Windows**——bash 工具会自动找到它. 不装也能跑, 退到 cmd
4. **有 NVIDIA 显卡更好** (CUDA), 没有就 CPU, 慢很多

### 启动 (推荐: 交互模式)

进到**你希望它工作的目录** (所有文件操作都被锁在启动时的 pwd 里):

```sh
python path/to/02-sandboxed/todo.py
```

横幅出现后, 在 `>` 后面输入你想让它做的事. 输入 `/exit` 退出.

### 启动 (一次性模式)

```sh
python path/to/02-sandboxed/todo.py "把 hello.py 改成带类型提示的版本"
```

跑完就退, 跟 v1 行为一样.

### REPL 命令

| 命令 | 作用 |
|---|---|
| 任意非 `/` 开头的输入 | 作为消息发给 agent |
| `/help` | 显示帮助 |
| `/exit` | 退出 (Ctrl-D / Ctrl-C 同效) |
| `/clear` | 清空对话历史和 todo |
| `/todos` | 看当前待办列表 |
| `/history` | 看消息条数统计 |
| 生成中按 `Ctrl-C` | 中断当前轮但不退出 |

### 想调点什么

[todo.py](todo.py) 顶部几行就是全部旋钮:

```python
MODEL_DIR = Path("E:/MYSELF/model/qwen/Qwen3.5-2B")
WORKDIR = Path.cwd().resolve()      # 工作目录, 启动时锁定
SYSTEM = "..."                       # 给模型看的系统提示词
MAX_ROUNDS = 20                      # 单条用户消息最多跑几轮
MAX_NEW_TOKENS = 4096                # v1 是 1024, 提到 4096 减少截断
```

---

## 9. 读源码的建议路线

[todo.py](todo.py) 570 行, 按这个顺序看最高效:

1. **`repl` 函数** (文件末尾): 整个程序的入口, 你能看见 REPL 怎么读输入、怎么拦截斜杠命令
2. **`run_turn`**: 这就是 agent 循环的核心, 把一条用户输入跑完整. 注意比 v1 的 `agent_loop` 多了截断检测段
3. **`_find_bash`**: 探测 Git for Windows 的 bash, 排除 WSL System32. v2 新增, 解决跨平台体验
4. **`run_append`**: append_file 的实现. 5 行. v2 新增, 配合 SYSTEM 教模型分片写大文件
5. **`_stop_token_ids`**: 把 `<|im_end|>` 加进停止 token, 防止模型脑补下一轮. v2 新增的关键细节
6. **`TOOL_HANDLERS` 字典**: 工具是怎么登记和分发的——v2 比 v1 多了 append_file 一行
7. **`run_xxx` 一族函数**: 一个具体工具长什么样, 注意它怎么把所有异常封装成文本返回
8. **`parse_tool_calls`**: 模型输出的 XML 是怎么变成结构化调用的 (跟 v1 一样, v4 换成 OpenAI 结构化)

整个文件没有用任何 agent 框架——你看到的就是真相, 没有黑盒.

---

## 10. 已知局限 (有的后续版本已修)

| 局限 | 后续状态 |
|---|---|
| 小模型 (2B) 长对话 / 多步任务能力上限 | ✅ v4 换 7B (Ollama Q4 量化), v5 加 MiniMax-M2.7 |
| 每轮重新 tokenize 完整历史, 没 KV cache 复用 | ✅ v4 换 Ollama 后由它管 |
| `input()` 没箭头键回看历史 | ⏸ 仍未做 |
| bash 工具不在路径沙箱内, 能 `cd ..` 出工作区 | ⏸ 设计权衡, 后续也这样 |
| 工具返回裸 str, 无法区分 success / partial / error | ✅ v3 加 `ToolResult` 三态 |
| 没文件搜索 (LS / Glob / Grep), 模型靠 `bash ls` | ✅ v3 加原子搜索三件套 |
| 写文件没读后写保护, 模型可能凭记忆乱改 | ✅ v3 加读后写 + 乐观锁 |
| 没自动测试 | ✅ v3 加 42 个 pytest |
| XML 解析脆弱 (模型少个尾标签就崩) | ✅ v4 换 OpenAI 结构化 `tool_calls` |
| 多文件改动要多次 edit_file | ✅ v4 加 `apply_patch` |
| 等模型几秒看不到状态 | ✅ v5 加流式输出 |
| 跑云端不知道烧了多少钱 | ✅ v5 每 turn 显示 token + 估算成本 |

---

## 11. v2 → v3 的演化方向

v2 让 agent 可用了 (REPL + 大文件分片 + bash 跨平台). 但跑久了发现工程基础**薄**:

- **6 个工具平铺, 没结构**——bash 跟 read_file 在模型眼里没区别, 它就什么顺手用什么. 模型经常用 `bash ls` 而不是 read_file
- **工具返回裸字符串**——模型看到的、UI 显示的、错误信息全混在一个 str 里, 程序无法机器读"是错误吗"
- **没文件搜索能力**——找代码靠 `bash find` / `bash grep`, 跨平台输出格式不固定
- **写文件没保护**——模型可以直接覆盖你正在编辑的文件, 没机会发现冲突
- **没有任何自动测试**——改动只能手动跑 REPL 验证

v3 一个一个修. v3 README 把每件事的"为什么、怎么、起没起效果"都写下来了——见 [03-atomic-tools/README.md](../03-atomic-tools/README.md).

再往后:

- **v4** ([04-structured-tool-calls](../04-structured-tool-calls/README.md)) — 换 Ollama + OpenAI 结构化 tool_calls + apply_patch 跨文件原子改动
- **v5** ([05-session-and-streaming](../05-session-and-streaming/README.md)) — Session 状态管理 + 流式输出 + 双后端 + token/成本可见 + 系统层兜底

按版本号顺序读, 能看清"agent 是怎么从可用长成实用工具"的每一步.
