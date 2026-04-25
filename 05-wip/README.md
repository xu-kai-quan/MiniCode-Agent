# 05-wip — 下一轮迭代的起点 (派生自 04-structured-tool-calls)

> 这一份是 04 的精确副本, 作为 v5 迭代的工作基线. 还没有任何 04 之外的改动.
> 下文仍是 v4 的内容 (本地 Ollama 驱动的交互式编码助手, MiniCode Agent v4),
> 等 v5 主题确定后再批量更新.

---


[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **alpha**. 单人项目, 接口可能变. 工具层有 66 个 pytest 测试, 主链路稳定.

用本地 Ollama (默认 `qwen2.5-coder:7b-instruct-q4_K_M`) 驱动的**交互式编码助手**: 你打开它, 它给你一个 `>` 提示符, 你可以连续提问, 它记得前面聊过什么, 会调用工具帮你干活。

整个程序就一个文件 [todo.py](todo.py), 约 1500 行 Python。

## v4 相对 v3 的主要变化

| 主题 | v3 | v4 |
|---|---|---|
| 后端 | 本地 torch + transformers 加载 Qwen3.5-2B safetensors | Ollama HTTP (OpenAI 兼容 `/v1/chat/completions`), 默认 `qwen2.5-coder:7b-instruct-q4_K_M` |
| 工具调用协议 | Qwen 自家的 `<tool_call>` XML, 自己正则解析 | OpenAI 结构化 `tool_calls`, 扔掉 XML 解析 |
| 多文件/多点修改 | 只能多次 `edit_file` | 新增 `apply_patch` — unified diff, 跨文件两阶段锁 + 原子回滚 |
| 依赖 | torch / transformers / safetensors | 仅 stdlib (前置装 Ollama 桌面版) |

`apply_patch` 的设计要点: 先把所有目标文件的读后写锁一次性预检通过, 再在内存里算好每个文件的新内容, 最后一次性落盘; 中途任一步失败, 已写的文件立刻回滚到修改前状态。等价于小号的 `git apply --atomic`。

## v3 相对 v2 的主要变化

| 主题 | v2 | v3 |
|---|---|---|
| 工具分层 | 6 个工具平铺 | 三层: 高频原子 / 中频受控 / 低频兜底 |
| 返回结构 | 裸字符串 | `ToolResult(status, data, text)` 统一协议 |
| 文件搜索 | 没有 | `LS` / `Glob` / `Grep` (rg→Python 兜底) |
| Grep 过滤 | — | `file_pattern` 限定文件类型 |
| 写文件保护 | 无 | 读后写 + 乐观锁 (NOT_READ / CONFLICT) |
| 模型路径 | 硬编码 | 支持 `MINICODE_MODEL_DIR` 环境变量 |
| 测试 | 无 | `pytest tests/` (42 个测试, ~0.6s) |
| Windows 终端编码 | GBK 撞 box 字符崩溃 | 强制 stdout UTF-8 |

---

## 一、跑起来长什么样

启动:

```sh
python todo.py
```

你会看到一个横幅, 显示 Ollama 后端地址、当前模型、工作目录、bash 路径、grep 模式 (rg 或 Python 兜底)、已注册的工具列表, 然后是 `>` 提示符:

```
> 帮我建一个 mypkg, 里面放 __init__.py 和一个 add(a,b) 函数
```

它会:

1. 先思考 ("我应该先建目录, 再写两个文件")
2. 调用 `todo` 工具列出步骤
3. 调用 `bash` 建目录
4. 调用 `write_file` 写文件
5. 把 todo 项标记为完成

干完之后回到 `>`, 你可以继续问别的, 它**记得**刚才发生了什么。

输入 `/exit` 退出, `/clear` 清空对话从头开始, `/help` 看所有命令。

---

## 二、核心概念

### 1) 什么叫 "agent"?

普通调用大模型: 你问一句, 它答一句, 结束。

Agent 是这样: 你问一句, 模型答一句**带着工具调用**, 系统真的去执行那个工具, 把结果再喂回模型, 模型看到结果再决定下一步…… 直到模型说"我做完了"。

简单说就是一个 `while` 循环:

```python
while True:
    回复 = 模型(对话历史)
    工具调用列表 = 解析(回复)
    if 没有工具调用:
        break          # 模型不再想干活, 退出
    for 调用 in 工具调用列表:
        结果 = 执行(调用)
        对话历史.append(结果)
```

**这个循环就是 agent 的全部本质**。本项目里, 它在 [todo.py](todo.py) 的 `run_turn` 函数里。

外面再套一层 REPL ( read-eval-print loop, 就是那个 `>` 提示符), 让你能连续提问。这两层是分开的:
- **内层** (`run_turn`): 把"一条用户输入"跑到"模型不再调工具"。
- **外层** (`repl`): 持续读取你的输入, 维护对话历史, 拦截斜杠命令。

### 2) 工具是什么?

工具就是"模型可以请求系统帮它做的事"。按"频率 × 确定性"分三层:

**高频原子层** (主力, 强约束):

| 工具名 | 干什么 | data 字段 |
|---|---|---|
| `LS` | 列目录 (不递归) | `entries` |
| `Glob` | 按文件名模式找文件 (递归) | `paths` |
| `Grep` | 按正则搜文件内容, 可加 `file_pattern` 限制文件类型 | `matches` |
| `Read` | 带行号读取文件, 支持 `limit`/`offset` | `content`, `lines`, `total_lines` |

**中频受控层** (有副作用, 加保险 — 见 §2.2):

| 工具名 | 干什么 | data 字段 |
|---|---|---|
| `write_file` | 创建或覆盖文件 | `bytes_written` |
| `append_file` | 追加内容到文件末尾 (分片写大文件) | `total_bytes` |
| `edit_file` | 把文件里某段唯一文本精确替换 | — |
| `apply_patch` | 应用 unified diff, 可跨多文件, 两阶段锁 + 任一失败全部回滚 | `files` (每文件的 op/hunks) |
| `todo` | 写/更新待办列表 (规划步骤) | `items` |

**低频兜底层** — `bash`: 留给原子工具不覆盖的边角需求 (git, 跑脚本, 包管理)。**明确不是主链路**, prompt 和 description 都这么说。

模型并不能直接动你的电脑 — 它只能"输出一段表示工具调用的文本", 由系统来真正执行。这是一道很重要的安全边界。

### 2.1) 工具返回的统一结构 ToolResult

每个工具都返回同一种结构:

```python
ToolResult(
  status: "success" | "partial" | "error",
  text:   str,    # 自然语言摘要 — 真正喂给模型的就是这段
  data:   dict,   # 结构化结果 — 只进日志面板, 不喂模型
)
```

**为什么 data 不喂模型?** 给小模型一坨 JSON, 它会精神分裂 — 一会儿当对话, 一会儿当数据要 parse。两条管道分开: text 是给模型看的自然语言, data 是给程序读的结构化事实。日志面板会把它们分开显示, UI 也只用 data。

**status 三态怎么分?**

- `success`: 任务完成, 没有信息被默默丢弃。
  - `Glob` 没匹配到 → success (返回空数组就是事实)
  - `Read` 用户传了 `limit=100` 拿到 100 行 → success (这是契约)
- `partial`: 任务完成但内容被截断, 用户预期 vs 实际之间有信息丢失。
  - `Read` 没传 limit, 被字符上限砍掉 → partial
  - `Grep` 命中过多, 只返回前 N 条 → partial
  - `bash` 输出超过 8000 字符 → partial
- `error`: 工具本身没跑成。`data["code"]` 给机器读的错误码 (`NOT_FOUND` / `PATH_ESCAPE` / `BAD_REGEX` / `NOT_READ` / `CONFLICT` / ...), `text` 给模型读的描述。

注意: bash 命令 **跑了但 exit code 非 0** 算 `success`, 不是 `error` — 因为工具本身正常工作, 模型自己能从 `exit=N` 看出命令失败了。`error` 严格留给"工具自己崩了"。

### 2.2) 中频受控层: 读后写 + 乐观锁

写类工具 (`write_file` / `append_file` / `edit_file`) 有两道保险, **dispatch 层强制**, 模型 handler 看不见这层逻辑:

**1. 读后写 (read-before-write)** — 改一个**已存在**的文件之前, 必须先 `Read` 它。否则 dispatch 直接返回:

```
error [NOT_READ]: File 'core/llm.py' must be read before edit_file.
Call Read first to load it into context.
```

新建文件不需要 (强制 Read 一个不存在的文件没意义)。这条规则防止模型"凭记忆"乱改 — 它必须先把当前内容拉进上下文, 看到了, 才能改。

**2. 乐观锁** — `Read` 时记录文件的 `(mtime_ns, size)`, 写之前验证没变。变了就拒:

```
error [CONFLICT]: File 'core/llm.py' changed since last read
(was 4217 bytes, now 4380). Re-read it before retrying.
```

这个 race 真实存在: IDE 自动保存、其他终端进程、git 切分支, 都可能在 Read 和 Write 之间动文件。乐观锁让模型察觉而不是默默覆盖别人的工作。

**写成功后** cache 自动刷新成新 stat — 模型可以连续 `edit_file` 同一个文件而不必重新 Read, 但若期间外部有改动, 下一次仍会 CONFLICT。

实现细节: `ReadCache` 是 `ToolRegistry` 的成员, 工具 handler 仍是纯函数。`Read` 把 stat 通过约定的 `data["_stat"]` 回传, dispatch 写入 cache 后剥掉, 模型既看不到这字段也不需要管它。

### 3) 为什么要有 todo (规划) 工具?

模型在多步任务里很容易**走神**: 对话越长, 它越会忘记最初的目标, 或者重复做已经做过的事。

解决办法是让它**先把步骤写下来**, 写下来的东西会留在对话历史里, 之后每一轮都能看到, 相当于一份"外部记忆"。规则有两条:

- 任何超过一步的任务, 必须先调 `todo` 列出步骤。
- 同一时刻只能有一个步骤是 `in_progress`, 强制它**一件一件来**。

如果模型连续 3 轮没更新 todo, 系统会主动注入一条提醒 ("该更新规划了"), 这个机制叫 **nag**。

### 4) 为什么要"沙箱"?

模型理论上可以请求写到任何路径。万一它想 `write_file("C:/Windows/...")` 怎么办?

文件类工具都先做一次**路径检查**: 把传入的路径解析成绝对路径, 如果它不在启动时的当前目录之下, 直接拒绝。这样模型的破坏范围就被锁在工作目录里了。

(`bash` 工具是个例外 — 既然叫 bash, 就把权力交给 shell 了, 它能 `cd ..` 出去。想更安全的话别用 bash。)

### 5) 为什么 bash 优先用 Git Bash?

在 Windows 上, Python 默认调用的"shell"其实是 `cmd.exe`, 模型如果写 `ls -la`, 会得到"`ls` 不是内部命令"。这相当于**工具名在骗模型** — 工具叫 bash, 实际不是 bash。

所以启动时会去找 Git for Windows 自带的 `bash.exe`, 找到就用它。横幅里会明确显示当前用的是哪个 shell, 让你一眼看见。

注意: 我们**故意不用** WSL 的 bash, 因为它输出 UTF-16 编码、路径语义 (`E:\` 在 WSL 里是 `/mnt/e/`) 也对不上工作目录, 引入的麻烦比解决的多。

### 5.1) Grep 为什么"有 rg 用 rg, 没有用 Python"?

诚实问题, 诚实答案: rg (ripgrep) 在几万行代码库上比纯 Python 快两个数量级, 用户真实项目一定会踩到这个差距。但**硬依赖 rg 违反了"开箱即用"的承诺** — 跟 bash 工具"找 Git Bash, 找不到退 cmd"是同一个原则。所以启动时检测一次 `rg`, 有就用, 没有就退到 Python 兜底版。横幅里会显示当前走哪条路。

Python 兜底版会跳过 `.git` / `node_modules` / `__pycache__` / `.venv` / `venv` 这种大型无用目录, 否则在大仓库里慢得不可接受。

### 6) 为什么需要"分片写文件"?

模型一次生成的 token 数量有上限。要写一个 500 行的 HTML, 一轮可能写不完, 半截被截断, 文件就没真正落盘。

解决办法两层:

- **多给一个工具**: `append_file`, 模型可以"先 write_file 写第一块, 再多次 append_file 加后面的块"。
- **系统兜底检测**: 如果模型输出里有 `<tool_call>` 开头但没闭合, 说明被截断了。系统不会假装"完成", 而是主动告诉模型"你被截了, 改用分片重来"。

这体现一个更大的设计原则: **prompt 是软约束, 系统是硬约束**。能在 prompt 里提示就提示, 但关键的不变量必须在系统层兜底, 因为小模型不一定听 prompt。

### 7) 模型是怎么"请求调用工具"的?

v4 走 **OpenAI 兼容的结构化 `tool_calls`** (Ollama 的 `/v1/chat/completions` 原生支持)。模型返回的 assistant message 形如:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "Read", "arguments": "{\"path\":\"a.py\"}"}}
  ]
}
```

系统从 `tool_calls` 字段直接读出工具名和 JSON 参数, 不再像 v3 那样用正则从文本里抠 `<tool_call>` XML。好处: 少一层脆弱的自解析, 而且能原生支持"并行调用多个工具"。

某些模型 (包括 qwen2.5-coder) 会在正式回答前写 `<think>...</think>` 段, 那是内部推理。我们打印它给你看, 但**不存回对话历史** — 否则历史会越塞越重。

---

## 三、怎么跑

### 准备工作

1. **装 Ollama**: 从 https://ollama.com/download 装桌面版, 启动它 (默认监听 `http://localhost:11434`)。
2. **拉模型**:
   ```sh
   ollama pull qwen2.5-coder:7b-instruct-q4_K_M
   ```
   想换模型或换 Ollama 地址, 用环境变量:
   ```sh
   export MINICODE_MODEL="qwen2.5-coder:14b"             # macOS/Linux
   set MINICODE_OLLAMA_URL=http://192.168.1.10:11434     # Windows cmd
   $env:MINICODE_TIMEOUT="600"                           # PowerShell (首载慢可调大)
   ```
3. **Python 依赖**: v4 运行时**只用标准库**, 不需要 `pip install` 任何东西。
   ```sh
   # 跑测试才需要 pytest
   pip install -r requirements-dev.txt
   ```
4. **可选, 推荐**: 装 Git for Windows (bash 工具会用), 装 ripgrep (Grep 会快两个数量级)。
5. **GPU**: Ollama 自己管 CUDA/Metal, 你不用操心; 显存够就自动上 GPU, 不够就落 CPU。

### 启动 (推荐: 交互模式)

进到**你希望它工作的目录** (所有文件操作都被锁在启动时的 pwd 里), 然后:

```sh
python path/to/05-wip/todo.py
```

横幅出现后, 在 `>` 后面输入你想让它做的事。

### 启动 (一次性模式)

如果你只想跑一个任务就退出, 直接传命令行参数:

```sh
python path/to/05-wip/todo.py "把 hello.py 改成带类型提示的版本"
```

### 跑测试

```sh
pip install -r requirements-dev.txt
cd 05-wip/
pytest tests/
```

如果 pytest 在 entrypoint 加载阶段崩 (我们环境里 langsmith 插件就这样), 加这个环境变量绕过自动加载:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

### 会话中能用的命令

| 命令 | 作用 |
|---|---|
| 任意非 `/` 开头的输入 | 作为消息发给 agent |
| `/help` | 显示帮助 |
| `/exit` | 退出 (等同于 Ctrl-D / Ctrl-C) |
| `/clear` | 清空对话历史和 todo, 从头开始 |
| `/todos` | 看当前待办列表 |
| `/history` | 看消息条数统计 |
| 生成中按 `Ctrl-C` | 中断当前轮但不退出 |

### 想调点什么

[todo.py](todo.py) 顶部几行就是全部旋钮:

```python
OLLAMA_BASE_URL = os.environ.get("MINICODE_OLLAMA_URL", "http://localhost:11434")
MODEL_NAME      = os.environ.get("MINICODE_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
REQUEST_TIMEOUT = int(os.environ.get("MINICODE_TIMEOUT", "300"))   # 秒
WORKDIR         = Path.cwd()    # 工作目录 (启动时锁定)
SYSTEM          = "..."         # 给模型看的系统提示词
MAX_ROUNDS      = 20            # 单条用户消息最多跑几轮
MAX_NEW_TOKENS  = 4096          # 单轮最多生成多少 token
GREP_MAX_MATCHES = 200          # Grep 单次最多返回的匹配数
LS_MAX_ENTRIES   = 500          # LS 单次最多列的条目数
READ_MAX_CHARS   = 8000         # Read 单次最多返回的字符数
```

---

## 四、读源码的建议路线

如果想搞懂内部, 按这个顺序看 [todo.py](todo.py) 最高效:

1. **`repl` 函数** (文件接近末尾): 程序入口, REPL 怎么读输入、怎么拦截斜杠命令。
2. **`ReActAgent.run` / `ReActAgent.step`**: agent 循环的核心 — Thought → Action → Observation, 把一条用户输入跑完整。
3. **`ToolRegistry.dispatch`**: 工具调度 + 读后写守卫 (NOT_READ / CONFLICT). 锁的事全在这一个函数里。
4. **`build_default_registry`**: 工具是怎么登记的 — 加新工具只改这里。
5. **任意一个工具 handler** (比如 `tool_grep` 或 `tool_apply_patch`): 一个具体工具长什么样, 注意它怎么把所有异常封装成 `ToolResult.error(...)` 而不是抛出去。`apply_patch` 是 v4 新增, 演示了两阶段锁 + 原子回滚。
6. **`OllamaClient` + `generate` + `_history_to_openai`**: HTTP 后端怎么把内部 `Message` 转成 Ollama 的 OpenAI 兼容请求, 又怎么把 `tool_calls` 拿回来。
7. **`_parse_tool_arguments` / `split_think`**: 结构化 tool_calls 的 `arguments` 字段 Ollama 有时给 JSON 字符串、有时给 dict, 这里统一成 dict; `split_think` 负责剥离模型的 `<think>` 推理。

整个文件没有用任何 agent 框架 — 你看到的就是真相, 没有黑盒。

对应的测试在 [tests/test_tools.py](tests/test_tools.py): 想知道某个机制 (例如乐观锁) 该怎么表现, 直接看测试用例最快。

---

## 五、测试

工具层有 66 个 pytest, 覆盖 `ToolResult` / 4 个原子工具 / 读后写 + 乐观锁 / `edit_file` 错误分支 / `apply_patch` (解析 / 多文件 / 回滚 / NOT_READ / CONFLICT) / `todo` / `dispatch` 守卫 / 解析器。

本地跑:

```bash
cd 05-wip
pip install -r requirements-dev.txt
pytest tests/
```

CI 在 GitHub Actions 上每次 push / PR 自动跑, 见 [.github/workflows/test.yml](../.github/workflows/test.yml). v4 的运行时本来就只依赖 stdlib (Ollama 在另一个进程里), 所以 CI 不需要装任何 ML 栈, 只装 `pytest`。

如果本地遇到 `ModuleNotFoundError: requests_toolbelt` 之类的 pytest 插件报错, 用:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

---

## 六、开发中踩到的坑

这里记四条在 v4 开发里吃过亏、最后留下经验教训的真实问题。单元测试没抓住的,都是 E2E 跑起来才暴露的。

### 1) unified diff 的空行是 hunk 结束符, 不是 context

**现象**: 模型生成的 patch 形如 `--- a/a.py ... +x=2\n\n--- a/b.py ...` — 两个文件段之间有空行。apply_patch 返回 `HUNK_FAILED`, 文件一个都没改, 甚至新文件也没创建。

**根因**: 解析器把文件之间的空行当成了"一行空的 context"追加给前一个 hunk 的 `old_buf`, 于是期望匹配 `"x = 1\n"`, 而文件里实际是 `"x = 1"` (无尾换行), 匹配失败。

**修法**: unified diff 规范里, 空行是 hunk 的**结束符**, 不是内容。把 `old_buf.append(""); new_buf.append("")` 改成 `break` 终结当前 hunk。

**教训**: 解析器一定要按格式规范写, 不要按"我觉得它应该怎样"写。单元测试当时用的都是"干净"的 diff (hunk 之间紧挨), 真实模型输出带空行才暴露。已补 3 个回归测试, 其中一个直接复刻 E2E 失败输入。

### 2) 小模型会绕过 tool_calls 协议, SYSTEM 得一条一条堵

**现象**: 模型不走 `tool_calls` 字段, 而是在 `content` 里写 ```` ```json\n{"name":"todo","arguments":[...]}\n``` ```` — 看起来在调工具, 实际是纯文本, 系统收不到任何工具调用, 流程卡住。堵掉 JSON 后, 模型又改用 ```` ```diff\n--- a/a.py\n... ``` ```` 直接贴 patch。

**修法**: SYSTEM 里加一节 **CRITICAL**, 明确禁止 (a) 用 ```` ```json ```` 块伪造工具调用, (b) 用 ```` ```diff ```/` ```` ```patch ```` 块贴补丁, (c) 任何其他把工具调用写成文本的形式; 并给正向指令"有 unified diff 就把它当作 `apply_patch` 的 `patch` 参数调用"。

**教训**: prompt 是软约束, 但对小模型这是唯一能加的约束 (结构层 Ollama 管不住 content)。禁令要具体、要带正向替代, 光说"不要那样做"模型会找别的通路绕。

### 3) 7B 在多步任务里会提前收手, 用 SYSTEM 兜底但不完美

**现象**: todo 规划了 5 步, 模型做完第 2 步就回复"已完成第二步, 请确认"后停下, 不继续第 3 步。人工 "/继续" 后模型反而乱调 `Grep \d+` 把自己的源码刷屏 12000+ 字符, 再回复"I can't assist"。

**修法**: SYSTEM 里加 MUST-continue 条款 — "每次成功 write/edit 之后, 如果原始用户请求还有剩余步骤, 必须继续; 不要只回复状态摘要, 直到所有 todo 项都 completed 再停"。

**教训**: 这条**并不能根除**, 是模型能力上限问题 (上下文越长 7B 越容易脱轨)。SYSTEM 把触发率降下来是值得做的; 根治得换 14B 或 32B。记在 §六 已知局限里, 诚实告知。

### 4) apply_patch 的锁, 为什么挂在 handler 闭包里而不是 dispatch 层

**背景**: 其他写类工具 (`write_file` / `edit_file`) 的读后写 + 乐观锁都在 `ToolRegistry.dispatch` 里统一做 — 它从 `arguments["path"]` 拿路径, 查 `read_cache`, 验 NOT_READ / CONFLICT。一个地方, 一次到位。

**问题**: `apply_patch` 的参数是 `patch` (一整坨 unified diff), **没有单一的 `path`**。diff 里可能涉及 N 个文件, 每个文件是 create/modify/delete 里的一种。dispatch 层的通用钩子挂不上来。

**修法**: 把 `read_cache` 通过闭包注入 handler —

```python
reg.register("apply_patch", handler=lambda patch: tool_apply_patch(patch, reg.read_cache))
```

handler 自己对 diff 里**每个被修改的文件**单独查 cache、验锁、然后两阶段提交 (先全部在内存里算出新内容并校验, 再统一落盘, 任一步失败就原子回滚已写文件)。

**教训**: 通用机制有覆盖不到的特例时, 别硬套通用机制把代码搞丑; 让特例在自己的 handler 里做, 用闭包传它需要的上下文就够。dispatch 层仍然只做它能做的那份工作 — 单文件路径的统一守卫。

---

## 七、已知局限

诚实告知:

- **小模型 (默认 7B) 对长对话和复杂多步任务的稳定性有限**。如果它开始胡言乱语、跳步、或把工具调用写成文本代码块, 用 `/clear` 重开; 换更大的模型 (如 `qwen2.5-coder:14b`) 会明显改善多步规划稳定性。
- **`input()` 是原生的**, 没有箭头键回看历史等舒适特性。
- **bash 工具不在路径沙箱内**, 它能 `cd ..` 出工作区。原子工具 (LS/Glob/Grep/Read/write/append/edit) 都强制路径沙箱。
- **Grep 的 Python 兜底版**对大仓库慢, 装 `ripgrep` 会快两个数量级。
- **写后 cache 自动刷新**意味着模型可以连续 `edit_file` 同一文件而不必每次重新 Read — 这是体验权衡, 不是漏洞。仍然防外部并发修改 (CONFLICT 一样会触发)。
- **乐观锁实际触发率不高**: 端到端测试中我们故意诱导模型不重 Read, 模型仍会下意识先 Read 一次, 让 cache 自动刷新。所以 CONFLICT 在真实使用里更多是"兜底防 IDE 自动保存 / git checkout / 其他终端"这种**模型不知道的外部修改**, 而不是防模型自己。pytest 里 `TestOptimisticLock` 直接调 dispatch 验证了机制本身。
- **Windows 终端编码**: 我们强制 stdout 为 UTF-8, 但下游管道 / `tee` / 重定向到文件可能仍按系统默认 (GBK) 解码, 中文/box 字符会乱。需要时 `chcp 65001` 切码页或用 `> output.log` 后用 UTF-8 编辑器打开。
- **`apply_patch` 不信任 hunk 头行号** (`@@ -N,M +N,M @@`), 仅用上下文行定位。好处是模型算错行号也能应用; 代价是上下文极少的纯插入 hunk 无法定位, 会报 `HUNK_FAILED`。
- **每轮都重新把完整对话历史发给 Ollama**, 没做 prompt 缓存复用, 对话变长会让 Ollama 那边重新 prefill, 速度会下降。
- **7B 模型的"任务规划连续性"有时会跳步**: 比如已经 `apply_patch` 成功了, 下一轮又去 `bash` 重做一次相同的修改。结果一般是幂等无害的, 但确实浪费一轮。这是模型能力问题, 不是循环 bug。
