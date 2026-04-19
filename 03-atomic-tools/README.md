# 03-atomic-tools — 本地 Qwen 驱动的交互式编码助手 (MiniCode Agent v3)

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **alpha**. 单人项目, 接口可能变. 工具层有 42 个 pytest 测试, 主链路稳定.

用本地 Qwen3.5-2B 模型驱动的**交互式编码助手**: 你打开它, 它给你一个 `>` 提示符, 你可以连续提问, 它记得前面聊过什么, 会调用工具帮你干活。

整个程序就一个文件 [todo.py](todo.py), 约 1300 行 Python。

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

你会看到一个横幅, 显示当前用的模型、设备 (CPU/GPU)、工作目录、以及 shell 类型, 然后是 `>` 提示符:

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

**为什么 data 不喂模型?** 给小模型 (2B) 一坨 JSON, 它会精神分裂 — 一会儿当对话, 一会儿当数据要 parse。两条管道分开: text 是给模型看的自然语言, data 是给程序读的结构化事实。日志面板会把它们分开显示, UI 也只用 data。

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

### 7) 为什么模型输出里有 `<think>` 又有 `<tool_call>`?

Qwen 会在正式回答前写一段 `<think>...</think>`, 那是它的内部推理。我们打印它给你看 (帮助理解), 但**不存回对话历史** — 否则历史会越塞越重。

`<tool_call>...</tool_call>` 才是真正请求系统执行的部分, 系统用正则把它抠出来, 找到对应的工具函数执行。

---

## 三、怎么跑

### 准备工作

1. **下载模型**到本地 (Qwen3.5-2B). 默认查找路径是 `E:/MYSELF/model/qwen/Qwen3.5-2B/`。
   想换路径, 设环境变量:
   ```sh
   export MINICODE_MODEL_DIR=/your/path/to/Qwen3.5-2B    # macOS/Linux
   set MINICODE_MODEL_DIR=D:\models\Qwen3.5-2B           # Windows cmd
   $env:MINICODE_MODEL_DIR="D:\models\Qwen3.5-2B"        # PowerShell
   ```
2. **装依赖**:
   ```sh
   # 先按平台/CUDA 装合适的 torch wheel: https://pytorch.org/get-started/locally/
   pip install -r requirements.txt
   ```
3. **可选, 推荐**: 装 Git for Windows (bash 工具会用), 装 ripgrep (Grep 会快两个数量级)。
4. **有 NVIDIA 显卡更好** (CUDA), 没有就 CPU, 慢很多。

### 启动 (推荐: 交互模式)

进到**你希望它工作的目录** (所有文件操作都被锁在启动时的 pwd 里), 然后:

```sh
python path/to/03-atomic-tools/todo.py
```

横幅出现后, 在 `>` 后面输入你想让它做的事。

### 启动 (一次性模式)

如果你只想跑一个任务就退出, 直接传命令行参数:

```sh
python path/to/03-atomic-tools/todo.py "把 hello.py 改成带类型提示的版本"
```

### 跑测试

```sh
pip install -r requirements-dev.txt
cd 03-atomic-tools/
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
MODEL_DIR = Path(os.environ.get("MINICODE_MODEL_DIR", "E:/.../Qwen3.5-2B"))
WORKDIR = Path.cwd()    # 工作目录 (启动时锁定)
SYSTEM = "..."          # 给模型看的系统提示词
MAX_ROUNDS = 20         # 单条用户消息最多跑几轮
MAX_NEW_TOKENS = 4096   # 单轮最多生成多少 token
GREP_MAX_MATCHES = 200  # Grep 单次最多返回的匹配数
LS_MAX_ENTRIES = 500    # LS 单次最多列的条目数
READ_MAX_CHARS = 8000   # Read 单次最多返回的字符数
```

---

## 四、读源码的建议路线

如果想搞懂内部, 按这个顺序看 [todo.py](todo.py) 最高效:

1. **`repl` 函数** (文件接近末尾): 程序入口, REPL 怎么读输入、怎么拦截斜杠命令。
2. **`ReActAgent.run` / `ReActAgent.step`**: agent 循环的核心 — Thought → Action → Observation, 把一条用户输入跑完整。
3. **`ToolRegistry.dispatch`**: 工具调度 + 读后写守卫 (NOT_READ / CONFLICT). 是 v3 最有意思的一段。
4. **`build_default_registry`**: 工具是怎么登记的 — 加新工具只改这里。
5. **任意一个工具 handler** (比如 `tool_grep` 或 `tool_edit`): 一个具体工具长什么样, 注意它怎么把所有异常封装成 `ToolResult.error(...)` 而不是抛出去。
6. **`parse_tool_calls` / `split_think`**: 模型输出的 XML 是怎么变成结构化调用的。
7. **`_stop_token_ids`**: 一个看似不起眼但很关键的细节 — 告诉模型"在哪儿该停"。

整个文件没有用任何 agent 框架 — 你看到的就是真相, 没有黑盒。

对应的测试在 [tests/test_tools.py](tests/test_tools.py): 想知道某个机制 (例如乐观锁) 该怎么表现, 直接看测试用例最快。

---

## 五、测试

工具层有 42 个 pytest, 覆盖 `ToolResult` / 4 个原子工具 / 读后写 + 乐观锁 / `edit_file` 错误分支 / `todo` / `dispatch` 守卫 / 解析器。

本地跑:

```bash
cd 03-atomic-tools
pip install -r requirements-dev.txt
pytest tests/
```

CI 在 GitHub Actions 上每次 push / PR 自动跑, 见 [.github/workflows/test.yml](../.github/workflows/test.yml). 测试不依赖 torch/transformers — `tests/conftest.py` 把它们 stub 掉了, 所以 CI 不需要拉几个 GB 的 ML 栈。

如果本地遇到 `ModuleNotFoundError: requests_toolbelt` 之类的 pytest 插件报错, 用:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

---

## 六、已知局限

诚实告知:

- **小模型 (2B) 对长对话和复杂多步任务的稳定性有限**。如果它开始胡言乱语, 用 `/clear` 重开。
- **每轮都重新 tokenize 完整历史**, 没做 KV cache 复用, 对话变长会明显变慢。
- **`input()` 是原生的**, 没有箭头键回看历史等舒适特性。
- **bash 工具不在路径沙箱内**, 它能 `cd ..` 出工作区。原子工具 (LS/Glob/Grep/Read/write/append/edit) 都强制路径沙箱。
- **Grep 的 Python 兜底版**对大仓库慢, 装 `ripgrep` 会快两个数量级。
- **写后 cache 自动刷新**意味着模型可以连续 `edit_file` 同一文件而不必每次重新 Read — 这是体验权衡, 不是漏洞。仍然防外部并发修改 (CONFLICT 一样会触发)。
- **乐观锁实际触发率不高**: 端到端测试中我们故意诱导模型不重 Read, 模型仍会下意识先 Read 一次, 让 cache 自动刷新。所以 CONFLICT 在真实使用里更多是"兜底防 IDE 自动保存 / git checkout / 其他终端"这种**模型不知道的外部修改**, 而不是防模型自己。pytest 里 `TestOptimisticLock` 直接调 dispatch 验证了机制本身。
- **Windows 终端编码**: 我们强制 stdout 为 UTF-8, 但下游管道 / `tee` / 重定向到文件可能仍按系统默认 (GBK) 解码, 中文/box 字符会乱。需要时 `chcp 65001` 切码页或用 `> output.log` 后用 UTF-8 编辑器打开。
- **没有 CI**. 测试要手动跑 `pytest`。
- **MultiEdit 还没做** — 多点原子修改要么拆成多次 `edit_file`, 要么 `Read → write_file 整文件`。
