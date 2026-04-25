# 03-atomic-tools — 把工具层做扎实, 经得起测试

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **封存** (作为 v3 快照保留, 不再迭代). 后续工作看 [04-structured-tool-calls](../04-structured-tool-calls/) 和 [05-session-and-streaming](../05-session-and-streaming/).
>
> 工具层有 42 个 pytest 测试. 一个文件 [todo.py](todo.py), 约 1300 行.

这是 MiniCode Agent 第 3 版. 用本地 Qwen3.5-2B 模型驱动的**交互式编码助手**: 给你一个 `>` 提示符, 你连续提问, 它记得前面聊过什么, 会调用工具帮你干活.

## 它跟 v2 的区别 — 一句话版本

**v2 能跑, v3 经得起测试**.

v2 把"agent + REPL + 沙箱"凑齐了, 跑得起, 但**工程基础薄**:

- 6 个工具平铺, 没有结构. 加新工具靠到处加 if/elif
- 工具返回裸字符串. 模型看到的、UI 显示的、错误信息——全都混在一个 string 里
- 没有文件搜索能力 (LS/Glob/Grep). 模型要看文件结构得用 `bash ls`, 工具名在骗模型
- 写文件没保护. 模型可以直接覆盖你正在编辑的文件, 没机会发现冲突
- **没有任何测试**. 改动靠手动跑 REPL 验证

v3 一个一个修. 这份 README 是**给想看清"agent 工具层是怎么做扎实的"的人看的**.

---

## 目录

- [1. 工具分层 — 高频 / 中频 / 低频](#1-工具分层--高频--中频--低频)
- [2. ToolResult 统一返回结构](#2-toolresult-统一返回结构)
- [3. 文件搜索三件套 LS / Glob / Grep](#3-文件搜索三件套-ls--glob--grep)
- [4. 读后写 + 乐观锁 — 防止模型乱改](#4-读后写--乐观锁--防止模型乱改)
- [5. 测试 — 工具层第一次有了 pytest](#5-测试--工具层第一次有了-pytest)
- [6. 跑起来长什么样](#6-跑起来长什么样)
- [7. 核心概念 (写给新读者)](#7-核心概念-写给新读者)
- [8. 怎么跑](#8-怎么跑)
- [9. 读源码的建议路线](#9-读源码的建议路线)
- [10. 已知局限 (有的 v4 / v5 已修)](#10-已知局限-有的-v4--v5-已修)
- [11. v3 → v4 的演化方向](#11-v3--v4-的演化方向)

---

## 1. 工具分层 — 高频 / 中频 / 低频

### v2 是什么样

```python
# v2: 6 个工具平铺
TOOLS = {
    "bash": run_bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "todo": update_todo,
    ...
}
```

加新工具就加一行. 看起来简单, 但**所有工具地位平等**——bash 跟 read_file 在模型眼里没区别, 它就什么顺手用什么. 实际表现:

- 模型用 `bash cat foo.py` 读文件 (而不是 read_file), 输出**没行号**
- 模型用 `bash ls` 看目录, 输出格式不固定 (Linux vs Windows 不一样)
- 模型用 `bash grep ...` 搜内容, 错误信息是 shell 的, 模型看不懂

工具能力相当, 但**确定性差异巨大**: read_file 返回的格式我们能保证, bash 返回什么完全看 shell 心情.

### 怎么改

按"频率 × 确定性"分三层, **prompt 里明确说每层什么时候用**:

**高频原子层** (主力, 强约束) — 这是模型应该**默认用**的:

| 工具 | 干什么 | data 字段 |
|---|---|---|
| `LS` | 列目录 (不递归) | `entries` |
| `Glob` | 按文件名模式找文件 (递归) | `paths` |
| `Grep` | 按正则搜文件内容, 可加 `file_pattern` | `matches` |
| `Read` | 带行号读取, 支持 `limit`/`offset` | `content`, `lines`, `total_lines` |

**中频受控层** (有副作用, 加保险):

| 工具 | 干什么 |
|---|---|
| `write_file` | 创建或覆盖文件 |
| `append_file` | 追加内容 (分片写大文件) |
| `edit_file` | 把文件里某段唯一文本精确替换 |
| `todo` | 写/更新待办列表 |

**低频兜底层** — `bash`: 留给原子工具不覆盖的边角需求 (git, 跑脚本, 包管理). **明确不是主链路**.

SYSTEM 里说:

```
Prefer the dedicated atomic tools (LS, Glob, Grep, Read) over `bash` —
they return structured, predictable results. Use `bash` only for
operations no atomic tool covers: running git, executing scripts, etc.
```

### 起没起效果

模型默认用 LS / Read 而不是 `bash ls` / `bash cat`. 输出有行号, 格式稳定. bash 退回到"边角工具"的位置.

但**只有 prompt 是软约束**——模型偶尔仍会用 bash 干原子工具能做的事 (尤其是 7B 在 v4 时代). 我们没在系统层强制 (会限制灵活性), 但分层让"什么时候该用什么"在 prompt 里有了清晰表达.

---

## 2. ToolResult 统一返回结构

### v2 是什么样

每个工具返回什么由 handler 自己决定, 大部分是 `str`:

```python
def read_file(path):
    try:
        return open(path).read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
```

模型看到的、UI 显示的、错误判断——都是这一坨 string. 问题:

- 程序无法机器读"是错误吗"——只能正则匹配 "Error" 前缀, 脆弱
- 截断信息 (e.g. "只显示前 100 行") 跟正常输出混在一起, 模型分不清
- UI 想显示进度 (匹配数 / 字节数), 没结构化数据可拿

### 怎么改

定义一个**三态 + 双管道**的统一结构:

```python
@dataclass
class ToolResult:
    status: str        # "success" | "partial" | "error"
    text: str          # 自然语言摘要 — 真正喂给模型的就是这段
    data: dict         # 结构化结果 — 只进日志面板, 不喂模型

    @classmethod
    def success(cls, text, **data) -> "ToolResult":
        return cls("success", text, data)

    @classmethod
    def partial(cls, text, **data) -> "ToolResult":
        return cls("partial", text, data)

    @classmethod
    def error(cls, code, message, **extra) -> "ToolResult":
        data = {"code": code, "message": message, **extra}
        return cls("error", f"error [{code}]: {message}", data)
```

### 三个原则

**1. text 给模型, data 给程序——双管道分开**

为什么 data 不喂模型? 给小模型 (2B) 一坨 JSON, 它会精神分裂——一会儿当对话, 一会儿当数据要 parse. 两条管道分开:
- text 是给模型看的自然语言摘要
- data 是给 UI / 测试 / 日志面板读的结构化事实

模型的对话历史里只有 text. data 留在内存里, 日志面板单独显示.

**2. status 三态分清, 不混淆**

- `success`: 任务完成, **没有信息被默默丢弃**.
  - `Glob` 没匹配到 → success (返回空数组就是事实)
  - `Read` 用户传了 `limit=100` 拿到 100 行 → success (这是契约)
- `partial`: 任务完成但被截断, **用户预期 vs 实际之间有信息丢失**.
  - `Read` 没传 limit, 被字符上限砍掉 → partial
  - `Grep` 命中过多, 只返回前 N 条 → partial
  - `bash` 输出超 8000 字符 → partial
- `error`: **工具本身没跑成**.
  - `data["code"]` 给机器读的错误码 (`NOT_FOUND` / `PATH_ESCAPE` / `BAD_REGEX` / `NOT_READ` / `CONFLICT`)
  - `text` 给模型读的描述

**3. bash 命令跑了但 exit code 非 0 算 success, 不是 error**

工具本身正常工作了 (subprocess 跑起来了, 拿到 exit code), 模型自己能从 `exit=N` 看出命令失败. error 严格留给"工具自己崩了" (subprocess 启动不了 / timeout / 进程 OOM).

### 起没起效果

显著. 一个例子:

```python
# 测试代码可以这样写, 不再 if "error" in text 之类的脆弱匹配
def test_grep_no_match():
    r = registry.dispatch("Grep", {"pattern": "ZZZ_xyz", "path": "."})
    assert r.status == "success"   # 没匹配到不是错误
    assert r.data["matches"] == []
```

UI 也好做了——日志面板按 status 染色 (绿/黄/红), data 里的 `matches` / `entries` 直接渲染表格.

---

## 3. 文件搜索三件套 LS / Glob / Grep

### v2 没有这层

模型要找代码, 只能 `bash ls -R` 或 `bash find . -name '*.py'`. 输出杂、跨平台不一样、没有截断保护——大仓库一搜炸屏.

### v3 加了三个原子工具

**LS** — 列目录 (非递归):

```
8 entries in '.':
src/
tests/
README.md  (12340 bytes)
todo.py  (52000 bytes)
...
```

返回 `entries: list[dict]` (含 type/name/size), 截断到 500 条.

**Glob** — 按文件名模式找文件 (递归), 用 `Path.rglob` 跨平台:

```
Found 12 file(s) matching '**/*.py':
src/foo.py
src/bar.py
tests/test_foo.py
...
```

**Grep** — 按正则搜文件内容, 双路径实现:

- 启动时检测 `rg`, 找到就用 ripgrep (快两个数量级)
- 找不到退到纯 Python (`re.search` + `os.walk`), 跳过 `.git` / `node_modules` / `__pycache__` / `.venv` / `venv`

每条 match 形如 `path:lineno:content`, 截断到 200 条.

加 `file_pattern='*.py'` 可以限定文件类型——模型不用先 Glob 找 *.py 再 Grep, 一步到位.

### 双路径实现的取舍

```python
_RG_PATH = shutil.which("rg")

def tool_grep(pattern, path=".", ignore_case=False, file_pattern=None):
    ...
    if _RG_PATH:
        return _grep_with_rg(pattern, path, ignore_case, file_pattern)
    return _grep_with_python(pattern, path, ignore_case, file_pattern)
```

**为什么不硬依赖 rg**? 因为这违背"开箱即用"承诺——跟 bash 工具"找 Git Bash, 找不到退 cmd"是同一个原则. 装 rg 是性能优化, 不是必要条件.

横幅里会显示当前走的是 rg 还是 Python:

```
grep: /usr/local/bin/rg
grep: (纯 Python 兜底, 装 ripgrep 更快)
```

---

## 4. 读后写 + 乐观锁 — 防止模型乱改

### v2 是什么样

模型可以直接 `write_file('important.py', '...')` 覆盖任何文件, **不需要先读**. 也没机制检测"两次读之间被外部改了". 后果:

- 模型凭记忆改文件, 改错了
- IDE 自动保存了, 模型基于"几分钟前的内容"写 patch, 默默覆盖你的工作

### v3 加了两道保险

**1. 读后写 (read-before-write)**

写类工具 (`write_file` / `append_file` / `edit_file`) 改一个**已存在**的文件之前, 必须先 `Read` 它. 否则 dispatch 直接返回:

```
error [NOT_READ]: File 'core/llm.py' must be read before edit_file.
Call Read first to load it into context.
```

新建文件不需要 (强制 Read 一个不存在的文件没意义). 这条规则防止模型"凭记忆"乱改——它必须先把当前内容拉进上下文, 看到了, 才能改.

**2. 乐观锁**

`Read` 时记录文件的 `(mtime_ns, size)`, 写之前验证没变. 变了就拒:

```
error [CONFLICT]: File 'core/llm.py' changed since last read
(was 4217 bytes, now 4380). Re-read it before retrying.
```

这个 race 真实存在: IDE 自动保存、其他终端进程、git 切分支, 都可能在 Read 和 Write 之间动文件. 乐观锁让模型察觉而不是默默覆盖别人的工作.

### 实现细节: 工具 handler 仍是纯函数

锁的逻辑放在 `ToolRegistry.dispatch` 层, 不在 handler 里. 设计原则: handler 应该是**可独立测试的纯函数**, 不依赖外部状态.

```python
class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self.read_cache = ReadCache()   # 锁的状态在这里

    def dispatch(self, name, arguments):
        # 写类工具的前置检查, 在 handler 之前做
        guard = self._check_read_before_write(name, arguments)
        if guard is not None:
            return guard
        # 调 handler (handler 不知道有 read_cache 这回事)
        return self._tools[name].handler(**arguments)
```

`Read` 怎么把 stat 回传给 cache? 通过约定: Read 在 `ToolResult.data["_stat"]` 里塞 `(mtime_ns, size)`, dispatch 写入 cache 后**剥掉**这个 key, 不让模型看到也不让日志面板显示.

```python
def tool_read(path, limit=None, offset=0) -> ToolResult:
    ...
    return ToolResult.success(
        text,
        path=path, content=numbered, lines=(start+1, end), total_lines=total,
        _stat=(st.st_mtime_ns, st.st_size),   # 给 dispatch 的私下信号
    )
```

这是工具层和 registry 层的内部约定, 既不污染 handler 的纯函数性, 也不让模型看到这种内部簿记.

### 写成功后 cache 自动刷新

模型可以连续 `edit_file` 同一文件而不必每次重新 Read——dispatch 在写成功后用新 stat 刷新 cache. 但若期间外部有改动, 下一次仍会 CONFLICT.

### 起没起效果

测试里有个专门的 `TestOptimisticLock` 类, 直接调 dispatch 验证机制本身——通过率 100%.

实际跑 REPL 时, 模型很少触发 NOT_READ (它下意识会先 Read, 这是个意外的"无心收益"). CONFLICT 触发更少, 但**真触发的时候**几乎都是 IDE 自动保存或 git checkout 的时候, 救了我们的工作.

---

## 5. 测试 — 工具层第一次有了 pytest

### v2 没有任何自动测试

每次改动靠手动跑 REPL: 输入"建一个 hello.py", 看屏幕输出对不对, 看 git status 看有没有副作用. 改一处可能挂三处, 没人告诉你.

### v3 加了 42 个 pytest

覆盖:
- `ToolResult` 三态构造器
- 4 个原子工具的成功 / 失败 / 截断分支
- 读后写 + 乐观锁的 NOT_READ / CONFLICT / 写后刷新
- `edit_file` 的歧义 (AMBIGUOUS) / 找不到 (NO_MATCH)
- `todo` 的 `INVALID_STATE` (两个 in_progress)
- `dispatch` 守卫: UNKNOWN_TOOL / BAD_ARGS / EXEC_FAILED / PATH_ESCAPE
- 解析器: `<tool_call>` XML 解析、`<think>` 剥离

### 测试不依赖 torch/transformers

`tests/conftest.py` 把 `torch` / `transformers` / `safetensors` 全部 stub 掉, 测试只跑工具层逻辑, 不加载模型. CI 跑完 0.6 秒, 不需要装几个 GB 的 ML 栈.

```python
# conftest.py
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("transformers", types.ModuleType("transformers"))
...
```

跑测试:

```bash
cd 03-atomic-tools
pip install pytest
pytest tests/        # 42 个, ~0.6s
```

### 起没起效果

显著. v3 之后所有改动 (包括 v4 / v5 的所有 commit) 都先跑 pytest. 工具层的回归被自动化抓住.

也设了 GitHub Actions CI, 每次 push 自动跑——`.github/workflows/test.yml`. 后面 v4 / v5 都把自己的目录加进 matrix, 现在 5 个版本的测试一起跑.

---

## 6. 跑起来长什么样

启动:

```sh
python todo.py
```

看到横幅 (模型 / 设备 / 工作目录 / shell / grep 模式 / 工具列表), 然后是 `>`:

```
> 帮我建一个 mypkg, 里面放 __init__.py 和一个 add(a,b) 函数
```

它会:

1. 先思考 (`<think>` 块)
2. 调 `todo` 工具列出步骤
3. 调 `bash` 建目录
4. 调 `write_file` 写文件
5. 把 todo 项标记完成

干完之后回到 `>`, 你可以继续问别的, 它**记得**刚才发生了什么.

`/exit` 退出, `/clear` 清空对话从头开始, `/help` 看所有命令.

---

## 7. 核心概念 (写给新读者)

如果你跳着读到这里, 不知道什么是 ReAct agent / 什么是工具表 / 为什么要沙箱——这一节给你最小心智模型. 已经懂的可以跳过.

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

**这个循环就是 agent 的全部本质**. 在 [todo.py](todo.py) 的 `ReActAgent.run` 函数里, 30 行代码看完.

外面再套一层 REPL ( read-eval-print loop, 就是那个 `>` 提示符), 让你能连续提问. 两层分开.

### 7.2 工具调用是怎么解析的

v3 用 Qwen 自家的 XML 格式:

```xml
<tool_call>
  <function=write_file>
    <parameter=path>hello.py</parameter>
    <parameter=content>print("hi")</parameter>
  </function>
</tool_call>
```

我们用正则从模型 content 里抠这一坨, 解析 function 名 + 参数. 工作但**脆**——v4 换成 Ollama 的 OpenAI 结构化 `tool_calls` 字段就稳得多 (见 [04 README §2](../04-structured-tool-calls/README.md#2-工具调用走-openai-结构化协议--扔掉-xml-解析)).

`<think>...</think>` 是 Qwen 的内部推理. 我们打印它给你看 (帮助理解), 但**不存回对话历史**——否则历史会越塞越重.

### 7.3 todo 工具 — 给模型一份外部记忆

模型在多步任务里很容易**走神**: 对话越长, 它越会忘记最初的目标.

解决办法: 让它**先把步骤写下来**, 写下来的东西会留在对话历史里, 之后每一轮都能看到, 相当于"外部记忆":

- 任何超过一步的任务, 必须先调 `todo` 列出步骤
- 同一时刻只能有一个步骤是 `in_progress`, 强制它**一件一件来**

如果模型连续 3 轮没更新 todo, 系统主动注入提醒 ("该更新规划了"). 这个机制叫 **nag**.

### 7.4 沙箱 — 把破坏范围锁在工作目录

模型理论上可以请求写到任何路径. 万一它想 `write_file("C:/Windows/...")` 怎么办?

文件类工具都先做一次**路径检查**:

```python
def _safe_path(workdir, p):
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

不在工作目录之下的, 直接拒绝 (`PATH_ESCAPE` 错误). 模型的破坏范围被锁在启动时的 cwd 里.

(`bash` 工具是个例外——既然叫 bash, 就把权力交给 shell 了, 它能 `cd ..` 出去. 想更安全的话别用 bash.)

### 7.5 bash 优先用 Git Bash

在 Windows 上, Python 默认调用的"shell"其实是 `cmd.exe`, 模型如果写 `ls -la`, 会得到"`ls` 不是内部命令". 这相当于**工具名在骗模型**——工具叫 bash, 实际不是 bash.

启动时去找 Git for Windows 自带的 `bash.exe`, 找到就用. 横幅里显示当前用的是哪个 shell.

故意不用 WSL 的 bash——它输出 UTF-16、路径语义 (`E:\` → `/mnt/e/`) 跟工作目录对不上.

### 7.6 分片写大文件

模型一次生成的 token 数量有上限. 写 500 行 HTML, 一轮可能写不完, 半截被截断, 文件没真正落盘.

两层方案:
- **多给一个工具** `append_file`——模型可以"先 write_file 写第一块, 再多次 append_file 加后面的块"
- **系统兜底检测**——如果模型输出里有 `<tool_call>` 开头但没闭合, 说明被截断了, 系统主动告诉模型"你被截了, 改用分片重来"

这体现一个更大的原则: **prompt 是软约束, 系统是硬约束**. 关键不变量必须在系统层兜底, 因为小模型不一定听 prompt.

---

## 8. 怎么跑

### 准备

1. **下载模型**到本地 (Qwen3.5-2B). 默认查找路径是 `E:/MYSELF/model/qwen/Qwen3.5-2B/`.
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
3. **可选, 推荐**: 装 Git for Windows (bash 工具会用), 装 ripgrep (Grep 会快两个数量级)
4. **有 NVIDIA 显卡更好** (CUDA), 没有就 CPU, 慢很多

### 启动

进到**你希望它工作的目录** (所有文件操作都被锁在启动时的 pwd 里):

```sh
python path/to/03-atomic-tools/todo.py
```

横幅出现后, 在 `>` 后面输入你想让它做的事.

一次性模式:

```sh
python path/to/03-atomic-tools/todo.py "把 hello.py 改成带类型提示的版本"
```

### 跑测试

```sh
pip install -r requirements-dev.txt
cd 03-atomic-tools/
pytest tests/
```

如果 pytest 在 entrypoint 加载阶段崩 (langsmith 等插件冲突):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

### REPL 命令

| 命令 | 作用 |
|---|---|
| 任意非 `/` 开头的输入 | 作为消息发给 agent |
| `/help` | 显示帮助 |
| `/exit` | 退出 |
| `/clear` | 清空对话历史和 todo |
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
GREP_MAX_MATCHES = 200
LS_MAX_ENTRIES = 500
READ_MAX_CHARS = 8000
```

---

## 9. 读源码的建议路线

按这个顺序看 [todo.py](todo.py) 最高效:

1. **`repl` 函数** (文件末尾): 程序入口, REPL 怎么读输入、怎么拦截斜杠命令
2. **`ReActAgent.run` / `ReActAgent.step`**: agent 循环的核心——Thought → Action → Observation
3. **`ToolRegistry.dispatch`**: 工具调度 + 读后写守卫 (NOT_READ / CONFLICT). v3 最有意思的一段
4. **`build_default_registry`**: 工具是怎么登记的——加新工具只改这里
5. **任意一个工具 handler** (比如 `tool_grep` 或 `tool_edit`): 一个具体工具长什么样, 注意它怎么把所有异常封装成 `ToolResult.error(...)` 而不是抛出去
6. **`parse_tool_calls` / `split_think`**: 模型输出的 XML 是怎么变成结构化调用的
7. **`_stop_token_ids`**: 一个看似不起眼但很关键的细节——告诉模型"在哪儿该停"

整个文件没有用任何 agent 框架——你看到的就是真相, 没有黑盒.

测试在 [tests/test_tools.py](tests/test_tools.py): 想知道某个机制 (例如乐观锁) 该怎么表现, 直接看测试用例最快.

---

## 10. 已知局限 (有的 v4 / v5 已修)

| 局限 | 后续状态 |
|---|---|
| 小模型 (2B) 长对话 / 多步任务能力上限 | ✅ v4 换 7B (Ollama Q4 量化), v5 加 MiniMax-M2.7 云端选项 |
| 每轮重新 tokenize 完整历史, 没 KV cache 复用 | ✅ v4 换 Ollama 后, KV cache 由 Ollama 管, 不用我们操心 |
| `input()` 没箭头键回看历史 | ⏸ 仍未做 |
| bash 工具不在路径沙箱内, 能 `cd ..` 出工作区 | ⏸ 仍这样 (设计权衡) |
| Grep Python 兜底版对大仓库慢 | ⏸ 装 ripgrep 即可 |
| 写后 cache 自动刷新 (体验权衡) | ⏸ 同 |
| Windows 终端编码下游管道 / `tee` 仍按 GBK | ⏸ 同 |
| 没有 CI | ✅ v3 末尾加了 GitHub Actions, v4/v5 沿用 |
| MultiEdit 还没做, 多点修改要拆 | ✅ v4 加了 `apply_patch`——unified diff, 跨多文件原子 |
| 工具调用走 XML, 自己正则解析, 偶尔崩 | ✅ v4 换 OpenAI 结构化 `tool_calls` |
| 模块级状态散乱, 多 agent 串台 | ✅ v5 引入 Session 集中状态 |
| 等模型几秒看不到状态 | ✅ v5 加流式输出 |

---

## 11. v3 → v4 的演化方向

v3 工程上做扎实了 (分层 / ToolResult / 锁 / 测试), 但还有几个能力上的痛点:

- **本地 torch 跑 2B 已经吃力**, 想换更大模型基本没办法 → v4 换 Ollama, 8G 显存能跑 7B Q4
- **XML 工具调用要自己解析**, 偶尔崩 → v4 换 OpenAI 结构化 `tool_calls`
- **跨文件改动要多次 edit_file**, 容易半路丢 → v4 加 `apply_patch` 跨文件原子

每一条都是 v3 实际跑过才发现的问题. v4 README 把这三件事的"为什么、怎么、起没起效果"都写下来了, 想看 agent 怎么从"能跑 + 经得起测试"长到"能改多文件代码"的, 直接去 [04-structured-tool-calls](../04-structured-tool-calls/README.md).

再往后 v5 又加了流式 / 双后端 / token 可见 / 系统层兜底等"能用的体感"——见 [05-session-and-streaming](../05-session-and-streaming/README.md).
