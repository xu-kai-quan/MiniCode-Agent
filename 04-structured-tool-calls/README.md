# 04-structured-tool-calls — 把 agent 做成"能跑 + 能改多文件"

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **封存** (作为 v4 快照保留, 不再迭代). 后续工作看 [05-session-and-streaming](../05-session-and-streaming/).
>
> 工具层有 66 个 pytest 测试, 主链路稳定. 一个文件 [todo.py](todo.py), 约 1500 行.

这是 MiniCode Agent 第 4 版. 用本地 Ollama (默认 `qwen2.5-coder:7b-instruct-q4_K_M`) 驱动的**交互式编码助手**: 给你一个 `>` 提示符, 你连续提问, 它记得前面聊过什么, 会调用工具帮你干活.

## 它跟 v3 的区别 — 一句话版本

**v3 是"看代码"的工具, v4 是"改多文件代码"的工具**.

v3 用本地 torch 加载 Qwen3.5-2B safetensors, 工具调用走 Qwen 自家的 `<tool_call>` XML, 跨文件改动只能多次 `edit_file`. 跑得起来但有几个让人想换的地方:

- **本地 torch 跑 2B 已经吃力**——CPU 慢、GPU 显存抠. 想换更大模型基本没办法
- **XML 工具调用要自己正则解析**——脆弱, 模型偶尔少个尾标签整套就崩
- **跨文件改动一次 edit_file 一个文件**——五个文件相关的改动, 模型容易做完三个就忘掉后两个
- **Qwen safetensors 要求本地占好几个 G 磁盘**, 还要管模型路径

v4 一个一个修. 这份 README 把"为什么、怎么、起没起效果"都写下来了——主要面向想看清"agent 长出能用形态"过程的读者.

---

## 目录

- [1. 后端换 Ollama HTTP — 摆脱 torch + safetensors](#1-后端换-ollama-http--摆脱-torch--safetensors)
- [2. 工具调用走 OpenAI 结构化协议 — 扔掉 XML 解析](#2-工具调用走-openai-结构化协议--扔掉-xml-解析)
- [3. 新工具 apply_patch — 跨文件原子改动](#3-新工具-apply_patch--跨文件原子改动)
- [4. 跑起来长什么样](#4-跑起来长什么样)
- [5. 核心概念 (写给新读者)](#5-核心概念-写给新读者)
- [6. 怎么跑](#6-怎么跑)
- [7. 读源码的建议路线](#7-读源码的建议路线)
- [8. 开发中踩到的真实坑 (有图有真相)](#8-开发中踩到的真实坑-有图有真相)
- [9. 已知局限 (有的 v5 已修)](#9-已知局限-有的-v5-已修)
- [10. v4 → v5 的演化方向](#10-v4--v5-的演化方向)

---

## 1. 后端换 Ollama HTTP — 摆脱 torch + safetensors

### v3 是什么样

```python
# v3 启动时:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# 然后每次推理都走 model.generate(...)
```

要装的东西:
- `torch` (按平台选 CUDA/CPU wheel, 几 GB)
- `transformers` + `safetensors`
- 本地 Qwen3.5-2B 权重 (~4 GB)
- 自己管 `MODEL_DIR` 环境变量

显存 / 内存的事 transformers 不替你想——加载就是把整个模型读进显存. 4060 8G 跑 2B FP16 卡得很, 跑不动 7B.

### 怎么改

把推理整个外包给 [Ollama](https://ollama.com/). Ollama 是个本地的模型服务进程:

- 一个安装包搞定 GPU 推理 + GGUF 量化加载 + HTTP API
- 默认监听 `http://localhost:11434`, 提供 OpenAI 兼容的 `/v1/chat/completions` 端点
- 模型用 `ollama pull` 拉, 它帮你管文件位置和 GPU 显存
- Q4_K_M 混合量化让 8G 显存能塞下 7B (~4.7G), 比 2B FP16 强一个量级

我们的代码相应**砍掉**:
- `import torch` / `import transformers` 全删
- 模型加载、显存管理、采样温度——Ollama 全管
- 自己 detokenize、stop-sequence 处理——OpenAI 协议层抽象掉

**剩下的 client 代码 80 行, 只用 stdlib** ([todo.py 的 OllamaClient](todo.py)):

```python
@dataclass
class OllamaClient:
    base_url: str = OLLAMA_BASE_URL
    model: str = MODEL_NAME

    def chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        # urllib.request.Request 一次 POST, 解析 JSON, 返回 message 字典
        ...
```

### 为什么不用 llama-cpp-python (本来更轻)

试过. Windows + 中文环境下 llama-cpp-python 没有预编译 wheel, 源码编译时 MSVC 把头文件中文注释按 GBK 解析挂掉. Ollama 是更省心的选择.

### 起没起效果

显著. 4060 8G 现在跑 7B Q4 流畅, 多步任务的能力比 2B FP16 高一个量级. 但 7B 仍然有上限——见 §9 / §10.

---

## 2. 工具调用走 OpenAI 结构化协议 — 扔掉 XML 解析

### v3 是什么样

Qwen 自家用 XML 表达工具调用:

```xml
<tool_call>
  <function=write_file>
    <parameter=path>hello.py</parameter>
    <parameter=content>print("hi")</parameter>
  </function>
</tool_call>
```

v3 用正则从模型 content 里抠这一坨, 解析 function 名 + 参数. 问题:

- 模型偶尔少个尾标签 (`</function>` 漏了), 整个 tool_call 解析失败
- `<parameter=content>` 里的内容如果含 XML 字符 (`<`, `>`, `&`), 嵌套就乱
- 多个 tool_call 并行调用要自己解析多个 `<tool_call>` 块

### 怎么改

Ollama 的 `/v1/chat/completions` 端点原生支持 OpenAI 的**结构化** `tool_calls` 字段. 模型返回的 assistant message 形如:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {"name": "Read", "arguments": "{\"path\":\"a.py\"}"}
    }
  ]
}
```

我们直接读 `tool_calls` 字段, **不再正则解析 content**. 一处省心:

```python
raw_tool_calls = msg.get("tool_calls") or []
for tc in raw_tool_calls:
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"])  # 标准 JSON, 不会少字符
    actions.append({"id": tc["id"], "name": name, "arguments": args})
```

### 起没起效果

少一层脆弱的自解析, 而且能原生支持"并行调用多个工具" (返回的 `tool_calls` 是个数组).

但**模型不一定听话**——见 §8.2: 7B 经常**绕过** tool_calls 字段, 在 content 里写 ```json 块伪装工具调用. 我们在 SYSTEM 里加了 CRITICAL 段堵这个洞, 但 v5 在系统层加了硬约束兜底 (检测到代码块伪 tool_call 就硬退出, 不再死循环).

---

## 3. 新工具 apply_patch — 跨文件原子改动

### 为什么需要

v3 的写类工具:

- `write_file` — 整文件覆盖
- `edit_file` — 一段唯一文本替换

要做"5 个文件相关改动" (比如重命名一个函数), 模型必须连调 5 次 edit_file. 三个问题:

1. **模型在第 3 个文件改完后容易忘剩下两个**——多步任务连续性差
2. **半路失败**: 第 3 个文件 edit 成功了, 第 4 个 NO_MATCH 了, 系统已经留下半改的状态. 模型不知道怎么收拾
3. **没原子性**: 改到一半被 Ctrl-C 打断, 部分文件已写, 部分没写

### apply_patch 的设计

参考 `git apply --atomic` 的语义. 模型给一个 unified diff, 可以涉及任意数量的文件、任意 op:

```diff
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hi")
+    print("hi v2")
 hello()

--- /dev/null
+++ b/bar.py
@@ -0,0 +1,1 @@
+print("brand new file")

--- a/old.py
+++ /dev/null
@@ -1,1 +0,0 @@
-print("delete me")
```

dispatch 端**原子化**:

```
Phase 1: 全部预检 (路径沙箱 / 文件存在性 / 已 Read 过 / mtime 没变)
  任何一项失败 → 立即返回, 磁盘没动过

Phase 2: 内存里构造每个文件的新内容 (apply hunks)
  任何一个 hunk 上下文匹配失败 → 立即返回, 磁盘没动过

Phase 3: 一次性落盘 (按文件顺序写)
  写到一半失败 (磁盘满 / 权限) → 把已写的全部回滚到 Phase 1 备份的原内容
```

**用户视角的保证**: 要么 N 个文件全改, 要么一个都没动. 不存在"改了 3 个的中间状态".

### 实现细节: hunk 上下文定位 (不信任行号)

unified diff 的 `@@ -10,5 +10,5 @@` 行号头**有可能算错** (小模型生成 diff 时常见). 我们的解析器**只用上下文行定位**, 忽略行号头:

```python
def _apply_hunks_to_text(text, hunks):
    cursor = 0
    out = []
    for h in hunks:
        # 在 text[cursor:] 里搜 hunk 的 old_block (含 ' ' 和 '-' 行)
        idx = text[cursor:].find(h.old_block)
        if idx < 0:
            raise ValueError("context not found in file")
        # 必须唯一 — 多处匹配就歧义
        if text[cursor:].find(h.old_block, idx + 1) >= 0:
            raise ValueError("context matches multiple places")
        out.append(text[cursor:cursor+idx])
        out.append(h.new_block)
        cursor += idx + len(h.old_block)
    out.append(text[cursor:])
    return "".join(out)
```

代价: 上下文极少的纯插入 hunk 无法定位 (报 `HUNK_FAILED`). 收益: 模型行号算错也没事——更重要.

### 起没起效果

显著. 之前模型连改 5 个文件失败率高, 现在一个 patch 一次到位. v5 跑实际任务时, M2.7 给出的 patch 一次性 apply 成功率非常高.

§8.1 记录了一个真实的 bug: 模型生成的 patch 在两个文件段之间多了空行, 我们的解析器误把空行当 context, 失败. 修了之后加了回归测试.

---

## 4. 跑起来长什么样

启动:

```sh
python todo.py
```

看到横幅 (Ollama 后端、当前模型、工作目录、bash 路径、grep 模式、工具列表), 然后是 `>`:

```
> 帮我建一个 mypkg, 里面放 __init__.py 和一个 add(a,b) 函数
```

它会:

1. 先思考 (`<think>` 块——"我应该先建目录, 再写两个文件")
2. 调 `todo` 工具列出步骤
3. 调 `bash` 建目录
4. 调 `write_file` 写文件
5. 把 todo 项标记完成

干完之后回到 `>`, 你可以继续问别的, 它**记得**刚才发生了什么.

`/exit` 退出, `/clear` 清空对话从头开始, `/help` 看所有命令.

---

## 5. 核心概念 (写给新读者)

如果你跳着读到这里, 不知道什么是 ReAct agent / 什么是工具表 / 为什么要沙箱——这一节给你最小心智模型. 已经懂的可以跳过.

### 5.1 什么叫 "agent"?

普通调用大模型: 你问一句, 它答一句, 结束.

Agent 是这样: 你问一句, 模型答一句**带着工具调用**, 系统真的去执行那个工具, 把结果再喂回模型, 模型看到结果再决定下一步…… 直到模型说"我做完了".

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

**这个循环就是 agent 的全部本质**. 在 [todo.py](todo.py) 的 `ReActAgent.run` 函数里, 30 行代码看完.

外面再套一层 REPL (read-eval-print loop, 就是那个 `>` 提示符), 让你能连续提问. 两层分开:
- **内层** (`run`): 把"一条用户输入"跑到"模型不再调工具"
- **外层** (`repl`): 持续读取你的输入, 维护对话历史, 拦截斜杠命令

### 5.2 工具是什么 / 为什么分三层

工具就是"模型可以请求系统帮它做的事". 按"频率 × 确定性"分三层:

**高频原子层** (主力, 强约束):

| 工具 | 干什么 | 返回的 data |
|---|---|---|
| `LS` | 列目录 (不递归) | `entries` |
| `Glob` | 按文件名模式找文件 (递归) | `paths` |
| `Grep` | 按正则搜文件内容, 可加 `file_pattern` | `matches` |
| `Read` | 带行号读取, 支持 `limit`/`offset` | `content`, `lines`, `total_lines` |

**中频受控层** (有副作用, 加保险):

| 工具 | 干什么 | 备注 |
|---|---|---|
| `write_file` | 创建或覆盖文件 | 已存在文件需先 Read (NOT_READ) |
| `append_file` | 追加内容 (分片写大文件) | 同上 |
| `edit_file` | 把文件里某段唯一文本精确替换 | 同上 |
| `apply_patch` | 应用 unified diff, 跨多文件原子 | 见 §3 |
| `todo` | 写/更新待办列表 (规划步骤) | 见 §5.4 |

**低频兜底层** — `bash`: 留给原子工具不覆盖的边角需求 (git, 跑脚本, 包管理). **明确不是主链路**, prompt 和 description 都这么说.

模型并不能直接动你的电脑——它只能"输出一段表示工具调用的文本", 由系统真正执行. 这是一道很重要的安全边界.

### 5.3 ToolResult 三态: success / partial / error

每个工具都返回同一种结构:

```python
ToolResult(
  status: "success" | "partial" | "error",
  text:   str,    # 自然语言摘要 — 真正喂给模型的就是这段
  data:   dict,   # 结构化结果 — 只进日志面板, 不喂模型
)
```

**为什么 data 不喂模型?** 给小模型一坨 JSON, 它会精神分裂——一会儿当对话, 一会儿当数据要 parse. 两条管道分开: text 是给模型看的自然语言, data 是给程序读的结构化事实.

**status 三态**:

- `success`: 任务完成, 没有信息被默默丢弃.
  - `Glob` 没匹配到 → success (返回空数组就是事实)
  - `Read` 用户传了 `limit=100` 拿到 100 行 → success (这是契约)
- `partial`: 任务完成但被截断, 用户预期 vs 实际之间有信息丢失.
  - `Read` 没传 limit, 被字符上限砍掉 → partial
  - `Grep` 命中过多, 只返回前 N 条 → partial
  - `bash` 输出超 8000 字符 → partial
- `error`: 工具本身没跑成. `data["code"]` 给机器读的错误码 (`NOT_FOUND` / `PATH_ESCAPE` / `BAD_REGEX` / `NOT_READ` / `CONFLICT` / ...), `text` 给模型读的描述.

注意: bash 命令 **跑了但 exit code 非 0** 算 `success`, 不是 `error`——因为工具本身正常工作, 模型自己能从 `exit=N` 看出命令失败. `error` 严格留给"工具自己崩了".

### 5.4 读后写 + 乐观锁

写类工具 (`write_file` / `append_file` / `edit_file`) 有两道保险, **dispatch 层强制**, 模型 handler 看不见这层逻辑:

**1. 读后写 (read-before-write)** — 改一个**已存在**的文件之前, 必须先 `Read` 它:

```
error [NOT_READ]: File 'core/llm.py' must be read before edit_file.
Call Read first to load it into context.
```

新建文件不需要 (强制 Read 一个不存在的文件没意义). 这条规则防止模型"凭记忆"乱改——它必须先把当前内容拉进上下文, 看到了, 才能改.

**2. 乐观锁** — `Read` 时记录文件的 `(mtime_ns, size)`, 写之前验证没变. 变了就拒:

```
error [CONFLICT]: File 'core/llm.py' changed since last read
(was 4217 bytes, now 4380). Re-read it before retrying.
```

这个 race 真实存在: IDE 自动保存、其他终端进程、git 切分支, 都可能在 Read 和 Write 之间动文件. 乐观锁让模型察觉而不是默默覆盖别人的工作.

**写成功后 cache 自动刷新成新 stat**——模型可以连续 `edit_file` 同一个文件而不必重新 Read, 但若期间外部有改动, 下一次仍会 CONFLICT.

实现细节: `ReadCache` 是 `ToolRegistry` 的成员, 工具 handler 仍是纯函数. `Read` 把 stat 通过约定的 `data["_stat"]` 回传, dispatch 写入 cache 后剥掉, 模型既看不到这字段也不需要管它.

### 5.5 为什么要有 todo (规划工具)

模型在多步任务里很容易**走神**: 对话越长, 它越会忘记最初的目标, 或者重复做已经做过的事.

解决办法是让它**先把步骤写下来**, 写下来的东西会留在对话历史里, 之后每一轮都能看到, 相当于一份"外部记忆". 规则有两条:

- 任何超过一步的任务, 必须先调 `todo` 列出步骤
- 同一时刻只能有一个步骤是 `in_progress`, 强制它**一件一件来**

如果模型连续 3 轮没更新 todo, 系统会主动注入一条提醒 ("该更新规划了"), 这个机制叫 **nag**.

### 5.6 为什么要"沙箱"

模型理论上可以请求写到任何路径. 万一它想 `write_file("C:/Windows/...")` 怎么办?

文件类工具都先做一次**路径检查**: 把传入的路径解析成绝对路径, 如果它不在启动时的当前目录之下, 直接拒绝. 这样模型的破坏范围就被锁在工作目录里了.

(`bash` 工具是个例外——既然叫 bash, 就把权力交给 shell 了, 它能 `cd ..` 出去. 想更安全的话别用 bash.)

### 5.7 为什么 bash 优先用 Git Bash

在 Windows 上, Python 默认调用的"shell"其实是 `cmd.exe`, 模型如果写 `ls -la`, 会得到"`ls` 不是内部命令". 这相当于**工具名在骗模型**——工具叫 bash, 实际不是 bash.

所以启动时会去找 Git for Windows 自带的 `bash.exe`, 找到就用它. 横幅里会明确显示当前用的是哪个 shell.

注意: 我们**故意不用** WSL 的 bash, 因为它输出 UTF-16 编码、路径语义 (`E:\` 在 WSL 里是 `/mnt/e/`) 也对不上工作目录, 引入的麻烦比解决的多.

### 5.8 Grep 的 rg → Python 兜底

rg (ripgrep) 在几万行代码库上比纯 Python 快两个数量级. 但**硬依赖 rg 违反了"开箱即用"的承诺**——跟 bash 工具"找 Git Bash, 找不到退 cmd"是同一个原则. 启动时检测一次 `rg`, 有就用, 没有就退到 Python 兜底版.

Python 兜底版会跳过 `.git` / `node_modules` / `__pycache__` / `.venv` / `venv` 这种大型无用目录, 否则在大仓库里慢得不可接受.

---

## 6. 怎么跑

### 准备

1. **装 Ollama**: 从 https://ollama.com/download 装桌面版, 启动它 (默认 `http://localhost:11434`)
2. **拉模型**: `ollama pull qwen2.5-coder:7b-instruct-q4_K_M`
3. **Python 依赖**: 运行时**只用标准库**, 不需要 pip install. 跑测试才需要 pytest
4. **可选**: 装 Git for Windows (bash 工具会用), 装 ripgrep (Grep 会快两个数量级)
5. **GPU**: Ollama 自己管 CUDA/Metal, 你不用操心

环境变量调整:

```sh
export MINICODE_MODEL="qwen2.5-coder:14b"             # macOS/Linux
set MINICODE_OLLAMA_URL=http://192.168.1.10:11434     # Windows cmd
$env:MINICODE_TIMEOUT="600"                           # PowerShell
```

### 启动

进到**你希望它工作的目录** (所有文件操作都被锁在启动时的 pwd 里):

```sh
python path/to/04-structured-tool-calls/todo.py
```

横幅出现后, 在 `>` 后面输入你想让它做的事.

一次性模式 (跑完就退):

```sh
python path/to/04-structured-tool-calls/todo.py "把 hello.py 改成带类型提示的版本"
```

### 跑测试

```sh
cd 04-structured-tool-calls/
pip install -r requirements-dev.txt
pytest tests/
```

如果 pytest 在 entrypoint 加载阶段崩 (langsmith 等插件冲突), 用:

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
| 生成中按 `Ctrl-C` | 中断当前轮但不退出 (v4 中断不太干净, v5 修了) |

### 想调点什么

[todo.py](todo.py) 顶部几行就是全部旋钮:

```python
OLLAMA_BASE_URL = os.environ.get("MINICODE_OLLAMA_URL", "http://localhost:11434")
MODEL_NAME      = os.environ.get("MINICODE_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
REQUEST_TIMEOUT = int(os.environ.get("MINICODE_TIMEOUT", "300"))
WORKDIR         = Path.cwd()    # 工作目录 (启动时锁定)
SYSTEM          = "..."         # 给模型看的系统提示词
MAX_ROUNDS      = 20            # 单条用户消息最多跑几轮
GREP_MAX_MATCHES = 200
LS_MAX_ENTRIES   = 500
READ_MAX_CHARS   = 8000
```

---

## 7. 读源码的建议路线

按这个顺序看 [todo.py](todo.py) 最高效:

1. **`repl` 函数** (文件末尾): 程序入口, REPL 怎么读输入、怎么拦截斜杠命令
2. **`ReActAgent.run` / `ReActAgent.step`**: agent 循环的核心——Thought → Action → Observation
3. **`ToolRegistry.dispatch`**: 工具调度 + 读后写守卫 (NOT_READ / CONFLICT). 锁的事全在这一个函数里
4. **`build_default_registry`**: 工具是怎么登记的——加新工具只改这里
5. **任意一个工具 handler** (比如 `tool_grep` 或 `tool_apply_patch`): 一个具体工具长什么样, 注意它怎么把所有异常封装成 `ToolResult.error(...)` 而不是抛出去. `apply_patch` 是 v4 新增, 演示了两阶段锁 + 原子回滚
6. **`OllamaClient` + `generate` + `_history_to_openai`**: HTTP 后端怎么把内部 `Message` 转成 Ollama 的 OpenAI 兼容请求, 又怎么把 `tool_calls` 拿回来
7. **`_parse_tool_arguments` / `split_think`**: 结构化 tool_calls 的 `arguments` 字段 Ollama 有时给 JSON 字符串、有时给 dict, 这里统一成 dict; `split_think` 负责剥离模型的 `<think>` 推理

整个文件没有用任何 agent 框架——你看到的就是真相, 没有黑盒.

测试在 [tests/test_tools.py](tests/test_tools.py): 想知道某个机制 (例如乐观锁) 该怎么表现, 直接看测试用例最快.

---

## 8. 开发中踩到的真实坑 (有图有真相)

这里记四条 v4 开发里吃过亏、最后留下经验教训的真实问题. 单元测试没抓住的, 都是 E2E 跑起来才暴露的.

### 8.1 unified diff 的空行是 hunk 结束符, 不是 context

**现象**: 模型生成的 patch 形如 `--- a/a.py ... +x=2\n\n--- a/b.py ...`——两个文件段之间有空行. apply_patch 返回 `HUNK_FAILED`, 文件一个都没改, 甚至新文件也没创建.

**根因**: 解析器把文件之间的空行当成了"一行空的 context"追加给前一个 hunk 的 `old_buf`, 于是期望匹配 `"x = 1\n"`, 而文件里实际是 `"x = 1"` (无尾换行), 匹配失败.

**修法**: unified diff 规范里, 空行是 hunk 的**结束符**, 不是内容. 把 `old_buf.append(""); new_buf.append("")` 改成 `break` 终结当前 hunk.

**教训**: 解析器一定要按格式规范写, 不要按"我觉得它应该怎样"写. 单元测试当时用的都是"干净"的 diff (hunk 之间紧挨), 真实模型输出带空行才暴露. 已补 3 个回归测试, 其中一个直接复刻 E2E 失败输入.

### 8.2 小模型会绕过 tool_calls 协议, SYSTEM 得一条一条堵

**现象**: 模型不走 `tool_calls` 字段, 而是在 `content` 里写 `` ```json\n{"name":"todo","arguments":[...]}\n``` `` ——看起来在调工具, 实际是纯文本, 系统收不到任何工具调用, 流程卡住. 堵掉 JSON 后, 模型又改用 `` ```diff\n--- a/a.py\n... `` 直接贴 patch.

**修法**: SYSTEM 里加一节 **CRITICAL**, 明确禁止 (a) 用 ```` ```json ```` 块伪造工具调用, (b) 用 ```` ```diff ```/```` ```patch ```` 块贴补丁, (c) 任何其他把工具调用写成文本的形式; 并给正向指令"有 unified diff 就把它当作 `apply_patch` 的 `patch` 参数调用".

**教训**: prompt 是软约束, 但对小模型这是唯一能加的约束 (结构层 Ollama 管不住 content). 禁令要具体、要带正向替代, 光说"不要那样做"模型会找别的通路绕.

**v5 后续**: prompt 还是堵不住——v5 跑实际任务时模型继续用 ```bash 块绕, 我们在系统层加了硬检测 + 累计 nag + 硬退出. 见 [05 README §5](../05-session-and-streaming/README.md#5-代码块伪-tool_call--系统层把-prompt-兜不住的接住).

### 8.3 7B 在多步任务里会提前收手, 用 SYSTEM 兜底但不完美

**现象**: todo 规划了 5 步, 模型做完第 2 步就回复"已完成第二步, 请确认"后停下, 不继续第 3 步. 人工 "/继续" 后模型反而乱调 `Grep \d+` 把自己的源码刷屏 12000+ 字符, 再回复"I can't assist".

**修法**: SYSTEM 里加 MUST-continue 条款——"每次成功 write/edit 之后, 如果原始用户请求还有剩余步骤, 必须继续; 不要只回复状态摘要, 直到所有 todo 项都 completed 再停".

**教训**: 这条**并不能根除**, 是模型能力上限问题 (上下文越长 7B 越容易脱轨). SYSTEM 把触发率降下来是值得做的; 根治得换 14B 或 32B.

**v5 后续**: 接了 MiniMax-M2.7, 7B 在多步任务的失败模式几乎消失. 见 [05 README §3](../05-session-and-streaming/README.md#3-双后端--ollama-和-minimax-之间切) 同一个 query 在 7B vs M2.7 上的 4 行对比.

### 8.4 apply_patch 的锁, 为什么挂在 handler 闭包里而不是 dispatch 层

**背景**: 其他写类工具 (`write_file` / `edit_file`) 的读后写 + 乐观锁都在 `ToolRegistry.dispatch` 里统一做——它从 `arguments["path"]` 拿路径, 查 `read_cache`, 验 NOT_READ / CONFLICT. 一个地方, 一次到位.

**问题**: `apply_patch` 的参数是 `patch` (一整坨 unified diff), **没有单一的 `path`**. diff 里可能涉及 N 个文件, 每个文件是 create/modify/delete 里的一种. dispatch 层的通用钩子挂不上来.

**修法**: 把 `read_cache` 通过闭包注入 handler:

```python
reg.register("apply_patch", handler=lambda patch: tool_apply_patch(patch, reg.read_cache))
```

handler 自己对 diff 里**每个被修改的文件**单独查 cache、验锁、然后两阶段提交.

**教训**: 通用机制有覆盖不到的特例时, 别硬套通用机制把代码搞丑; 让特例在自己的 handler 里做, 用闭包传它需要的上下文就够.

**v5 后续**: v5 引入了 Session, 不再需要闭包 hack——dispatch 通过反射给声明了 `_session: Session` 参数的 handler 自动注入. apply_patch 改写为 `def tool_apply_patch(patch: str, _session: Session)`, handler 直接通过 `_session.read_cache` 取. 见 [05 README §1](../05-session-and-streaming/README.md#1-session--把散落的状态收回来).

---

## 9. 已知局限 (有的 v5 已修)

| 局限 | v5 状态 |
|---|---|
| 小模型 (7B) 长对话 / 多步任务能力上限 | ⚠️ 模型问题, 仍存在; v5 加了双后端可切 MiniMax-M2.7 |
| `input()` 没箭头键回看历史 | ⏸ 仍未做 |
| bash 工具不在路径沙箱内, 能 `cd ..` 出工作区 | ⏸ 仍这样 (设计权衡) |
| Grep Python 兜底版对大仓库慢 | ⏸ 装 ripgrep 即可 |
| 写后 cache 自动刷新 (体验权衡, 非漏洞) | ⏸ 同 |
| Windows 终端编码下游管道 / `tee` 仍按 GBK | ⏸ 同 |
| `apply_patch` 不信任 hunk 头行号, 上下文极少的纯插入会失败 | ⏸ 设计权衡 |
| 每轮重传完整 history, 没做 prompt cache 复用 | ⏸ MiniMax 是否支持 cache 待查 |
| 每轮等模型 5-30 秒看不到状态 | ✅ v5 加了流式输出 |
| 7B 用代码块伪装 tool_call 死循环 | ✅ v5 加了系统层检测 + GAVE UP |
| 模型不知道工作目录在哪 (LS 用占位符路径) | ✅ v5 把 cwd 注入 SYSTEM |
| PATH_ESCAPE 错误模型不会重试 | ✅ v5 改了错误消息为可操作引导 |
| 跑 MiniMax 不知道烧了多少钱 | ✅ v5 每 turn 显示 token + 估算成本 |
| 模块级 TODO 单例多 agent 串台 | ✅ v5 改成 Session 状态封装 |

---

## 10. v4 → v5 的演化方向

v4 工程上已经合格, 但实际跑下来有几个具体的"用着别扭"的地方:

- 等 30 秒屏幕没动, 不知道是死了还是在思考 → v5 加流式
- 7B 多步任务上限治不好 → v5 加云端切换
- 跑一次任务不知道烧了多少钱 → v5 加 token + 成本可见
- 模块级全局变量在多 agent 间串台 → v5 加 Session
- prompt 堵不住的伪 tool_call → v5 加系统层硬检测

每一条都是 v4 实际跑过才发现的问题. v5 README 把这些"为什么、怎么、起没起效果"都写下来了, 想看 agent 怎么从"能跑"长到"能用"的, 直接去 [05-session-and-streaming](../05-session-and-streaming/README.md).
