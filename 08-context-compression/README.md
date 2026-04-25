# 08-context-compression — 长会话不再撞 context 上限

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

> 状态: **alpha / wip**. 当前活跃版本. 在 07 基础上加 history 压缩 — 自动 (turn 开始前看 prompt token 是否撞阈值) + 手动 (`/compress`). 切分点保证不拆 tool_call/tool_result 配对. LLM 总结失败时降级到机械裁剪. 工具层 104 个 pytest (82 继承 07 + 22 压缩专项). 07 已封存.
>
> v8 跟 07 的区别 — 一句话: **跑久了不会因为 context 撑爆而崩**. v7 的 session 持久化 + v6 的 spawn_agent + 流式 / 双后端 / token 可见全部继承.

## 8.1 v8 新增 — Context 压缩

### 痛点

agent 跑久了 history 永不裁剪, 累积撑爆 context — 真正吃 token 的不是对话, 是 tool_results: `Read` 一个 1500 行文件 ~30K 字符, `bash pytest 2>&1` 几百行, `Grep` 把整个 repo 扫出来, `spawn_agent` 返回 4000 字符 diff. 每个都进 history 永不消失.

M2.7 的 200K 不算紧但跑长任务也会撞; 7B 32K 一次大 Read 就够你慌.

### 怎么做

```
触发: 本 turn prompt_tokens ≥ CONTEXT_LIMIT × COMPRESS_AT (默认 70%)
切分:
  [0]      system prompt           → 保留
  [1..k]   早期对话 + tool_results  → 让 LLM 总结成一条 <compressed>...</compressed> user 消息
  [k+1..]  最近 N 条 (默认 10)      → 原样保留
```

关键约束:
- **tool_call 和 tool_result 永不拆开**. split 落在配对中间会自动往前推
- **tool_results 全文不进摘要** — 模型如果还需要会重 Read (跟 v7 read_cache load 后清空一个思路)
- **outstanding todo 显式塞进摘要** — `in_progress` / `pending` 项保留, `completed` 不进
- **LLM 总结失败 (RuntimeError / 空内容) 降级机械裁剪** — 抠 user 消息 + tool 计数 + path 字段, 不调 LLM, 不会失败
- **子 agent 不压缩** — 短任务跑完就丢, 压缩是过度设计

### 命令

```
/compress     # 手动压一次 (调试用, 或者觉得马上要超了想提前压)
```

自动触发不需要命令, turn 开始前会自动检查并报告:

```
┌─ ⚠️  CONTEXT THRESHOLD ───────────
│ prompt_tokens=145000 ≥ 140000 (70% of 200000). compressing...
└──────────────

┌─ 🗜️  CONTEXT COMPRESSED ═══════════
│ mode             : LLM summary
│ messages before  : 87
│ messages after   : 12
│ summarized       : 76 messages
│ kept tail        : 10 messages
│ summary length   : 1342 chars
└═══════════════════
```

### 配置

```
MINICODE_CONTEXT_LIMIT       # 模型 context 上限. 默认: minimax=200000, ollama=32000
MINICODE_COMPRESS_AT         # 触发比例. 默认 0.7
MINICODE_COMPRESS_KEEP_TAIL  # 保尾消息数. 默认 10
MINICODE_COMPRESS_SUMMARY_MAX # LLM 摘要字符上限提示. 默认 2000
```

### 跟 session 持久化的关系

`/save` 存的是当前 history — 如果是已经压缩过的, 存下来就是压缩后的. `/load` 同样. 这是有意的:
- 压缩过的 session 更小, 加载更快
- 压缩丢失的细节本来也丢了, 不会因为存盘多一份恢复机会
- read_cache 反正 load 时已清, 重 Read 是常态

### 实测路径 (尚未跑)

跟 06/07 一样的实测节奏: 真跑长任务直到撞 70%, 看自动压缩是否触发, 摘要质量如何, 压完模型能否继续干活. 跑过会把发现的问题更新到这里.

---

## 8.2 (以下为继承 07 的内容)

## 7.1 v7 新增 — Session 持久化

### 痛点

跑 1 小时 / 2 小时的长任务, 电脑要关 / 重启 / 系统更新, **进度全丢**. 下次只能重头开始. v6 之前完全没救.

### 怎么做

Session 已经是个 dataclass, 加 `to_dict` / `from_dict` 序列化为 JSON, 存到 `~/.minicode/sessions/<name>.json`. 4 个 REPL 命令操作:

```
/save my-task       # 把当前 session 存到 ~/.minicode/sessions/my-task.json
/load my-task       # 加载, 替换当前 session (替换前不会自动 save 当前的)
/sessions           # 列出所有保存的 session
/del my-task        # 删除一个保存的 session
```

session 文件存的:
- `history` (所有对话消息含 tool_calls)
- `todo_items` (TodoManager 的当前列表)
- `prompt_tokens_total` / `completion_tokens_total` (跨 session 累计成本)

不存的 (有意):
- `read_cache` — 文件 mtime/size 在 load 时大概率已变, 强制模型重 Read 更安全. load 后 read_cache 永远是新空的
- `is_subagent` — 子 agent 不该被独立保存
- 各种瞬时计数器 (rounds_since_todo / codeblock_nag_count / estimated_turns)

### 安全

session 名经过校验: `^[A-Za-z0-9_-]+$`, 长度 ≤ 64, 防路径穿越 (`/load ../../etc/passwd` 之类被直接拒).

session 存到 `~/.minicode/sessions/`, **不在仓库内**, 不会被 git track. 但 session 内容含完整对话历史, **可能有敏感信息** — 用户自己保管 `~/.minicode/` 的访问权限. 如果 session 含 .env 内容或工具结果里有密钥, 删 session 文件就完事.

可设 `MINICODE_SESSIONS_DIR` 环境变量改存储位置.

### 跟 v6 spawn_agent 的关系

无关. 子 agent 的 session 不持久化 (上面 `is_subagent` 那条). spawn_agent 跑期间产生的内容仍在主 session 的 history 里 (`spawn_agent` 工具的 tool_call + tool_result 都进 history), 所以 `/save` 也保留了 spawn_agent 跑过什么.

### 实测路径 (尚未跑)

跟 06 一样的实测节奏: 真跑一次 long task, 中间 `/save`, 退出, 重启, `/load`, 继续. 跑过会把发现的问题更新到 README §4.

---

## 下面是继承 06 的全部内容 (v6 的 spawn_agent + 流式 + 双后端 / token 可见)

> 说明: 06 已封存, 它的整套能力 (spawn_agent / 流式 / 双后端 / token 可见 / 代码块兜底) 在 07 全部保留. 下面这段从 06 README 全文继承, 仍然适用 — 省得读者跳来跳去.

---

# (以下原 06 README 内容)

# 06-sub-agents — 派子 agent 在隔离 worktree 里"试一下"

> 状态: **alpha**, 当前活跃版本. 工具层 76 个 pytest (66 继承 05 + 10 spawn_agent 专项). 一个文件 [todo.py](todo.py), 约 2300 行.

这是 MiniCode Agent 第 6 版. 在 05 的基础上加一件事: **`spawn_agent` 工具** —— 让主 agent 派子 agent 在隔离的 git worktree 里跑探索式任务, 不污染主 workspace.

---

## 它跟 05 的区别 — 一句话版本

**05 已经能用. 06 解决"想试个改动但不愿先在主 workspace 改"的具体痛点.**

05 跑得好, 但有个特定场景它解不好: 你想"试试这个改动通不通过测试", 又不想真改主代码. 用户必须自己手动:
1. `git stash` 或者备份当前状态
2. 跟 agent 说"改 foo.py 第 5 行, 跑 pytest"
3. 看结果
4. 如果不行, `git checkout .` 或手动恢复
5. 如果行, 留下改动

**这个 dance 烦, 而且容易在第 4 步漏一些 .pyc 之类的临时文件**. 06 的 `spawn_agent` 把整个 dance 移到一个工具调用里:

```
用户: "试一下把 foo.py 第 5 行改成 X, 跑 pytest 看通不通过"
主 agent: spawn_agent(task="apply this change, run pytest, report PASS/FAIL")
  ↓ [子 agent 在 /tmp/minicode-sub-XXX (HEAD 的 git worktree) 里跑]
  ↓ edit_file → bash pytest → 报告
主 agent: 拿到子 agent 的报告 + unified diff
  → 询问用户 "结果是 FAIL — 要把这个 diff apply 到主 workspace 吗?"
用户: "不要" / "要"
```

**主 workspace 全程没动**. 子 agent 跑完, worktree 自动清掉 (失败时保留供调试).

---

## 目录

- [1. 为什么不直接在主 workspace 改](#1-为什么不直接在主-workspace-改)
- [2. 怎么实现 — 5 个核心设计选择](#2-怎么实现--5-个核心设计选择)
- [3. 工作流走一遍 — 实测真录](#3-工作流走一遍--实测真录)
- [4. 修复迭代 — 5 次实测发现的真问题](#4-修复迭代--5-次实测发现的真问题)
- [5. 跑起来长什么样](#5-跑起来长什么样)
- [6. spawn_agent 的工具层细节](#6-spawn_agent-的工具层细节)
- [7. 怎么跑](#7-怎么跑)
- [8. 读源码的建议路线](#8-读源码的建议路线)
- [9. 已知局限](#9-已知局限)
- [10. v6 → v7 候选方向](#10-v6--v7-候选方向)

---

## 1. 为什么不直接在主 workspace 改

agent 默认行为是直接改 disk. 这对"我要写个新文件" / "把这个 bug 修了" 这种**已决策的任务**完全合适. 但有一类任务**本质上是探索式的**:

- "试试用 X 替换 Y 看测试还能不能通过"
- "看看把这个函数拆成两个会怎样"
- "尝试用 A 库代替 B 库, 跑一下"

这些任务**用户希望先看结果再决定要不要**. 如果直接改主 workspace:

- 失败了要回滚, agent 必须自己记住改了什么 (它**会忘**)
- 中间状态可能让其他工具 (IDE / git / pytest) 看到一坨半成品
- 用户没机会"看一眼 diff 再决定"

**spawn_agent 的设计意图就是把这个"探索 → 看结果 → 决策" 的循环显式化**. 不再依赖 agent 的记忆和自律, 改成结构化的:

1. 派子 agent 在**隔离的 git worktree** 里干
2. 子 agent 跑完, 系统自动**算 diff** 返回
3. 主 agent 把 diff 给用户看, **询问是否采纳**
4. 用户决策 → 主 agent 用 `apply_patch` 把 diff 真应用到主 workspace

整个流程**主 workspace 全程没动**, 失败也不留痕.

---

## 2. 怎么实现 — 5 个核心设计选择

跟用户讨论后定的 5 个选择. 每个都有真理由:

### 2.1 隔离机制: git worktree, 不是 cp -r

**对比**:

| 方案 | 速度 | 干净度 | 限制 |
|---|---|---|---|
| `cp -r workspace /tmp/sub` | 慢 (大项目几十秒) | 干净 | 无 |
| `git worktree add /tmp/sub HEAD` | **秒级** | 自带 git 状态隔离 | 要求是 git repo |
| Linux overlayfs | 快 | 干净 | Windows 没有 |

选 git worktree. 用户的项目大概率是 git repo, 这个限制可接受. 大项目场景下 cp -r 慢得不能用.

**怎么做**:

```python
sub_dir = Path(tempfile.mkdtemp(prefix="minicode-sub-"))
subprocess.run(["git", "worktree", "add", "--detach", str(sub_dir), "HEAD"], cwd=main_workdir)
```

`--detach` 让 worktree 处于 detached HEAD, 不创建分支. 我们不需要分支管理 — worktree 用完就扔.

### 2.2 同步等, 不是后台轮询

主 agent 调 spawn_agent, 必须**等子 agent 跑完**才返回. 不用"派出去后异步轮询" 那种复杂模式.

理由:
- REPL 体验里"派出去 → 等" 是用户能接受的, 跟普通 tool_call 一样
- 异步会让主 agent 的对话历史变得复杂 (穿插"任务还在跑" 的中间状态)
- 简单的同步阻塞配上**超时兜底** (默认 5 分钟) 就够

```python
SUB_AGENT_TIMEOUT = int(os.environ.get("MINICODE_SUB_TIMEOUT", "300"))
```

### 2.3 子 agent 继承主 agent 的 LLM 实例

主 agent 用 MiniMax-M2.7, 子 agent 也用 MiniMax-M2.7. 复用同一个客户端连接.

**为什么不允许子 agent 用不同 backend** (比如主用 MiniMax, 子用 Ollama 省钱)?

- 实现复杂度高
- 子 agent 任务质量比成本重要 — 用户想"准确知道这个改动通不通过测试", 不想为了省 5 分钱让子 agent 给个不靠谱答案
- 想省钱可以**整体**切到 Ollama (主和子都换)

具体实现: 模块级 `_SPAWN_LLM` 在 REPL 启动时设, spawn_agent 直接复用.

```python
# repl() 启动时
_SPAWN_LLM = llm

# spawn_agent 内部
sub_agent = ReActAgent(_SPAWN_LLM, sub_registry, system_prompt=sub_system)
```

### 2.4 子 agent 不能再派子 agent (深度限制 1)

`build_subagent_registry` 显式**不注册** spawn_agent 工具. 子 agent 工具集 = 主 agent 工具集 - {spawn_agent, todo}.

**为什么深度限制 1**:
- 7B 模型容易递归爆炸 — 派一个子 agent 派一个孙 agent 派一个曾孙 agent ...
- 深度 > 1 的复杂度对应的实用场景几乎没有
- 简单粗暴 = 安全

也去掉 `todo` 工具 — 子 agent 是单一目标的, 不需要规划.

### 2.5 失败 worktree 保留, 成功 worktree 清掉

子 agent 报告 `FAIL` / `ERROR` / `GAVE UP` 时, **不**自动清 worktree. 把路径放进 `result.data["worktree"]` 让用户能去看:

```bash
cd /tmp/minicode-sub-XXX
ls -la  # 看子 agent 留下了什么状态
```

成功时清掉 (`git worktree remove --force` + `rmtree`).

判断关键词: 子 agent 最后一条 assistant 消息含 "FAIL" / "ERROR" / "GAVE UP" → 失败. 否则成功.

---

## 3. 工作流走一遍 — 实测真录

实测对话 (M2.7 backend). 用户输入是中文 "试一下", 触发了 spawn_agent. 输出我精简了一下:

**用户**:
```
试一下把 todo.py 第 1 行改成 hhhhhhhh, 跑 pytest 看通不通过
```

**主 agent (Turn 1)**: 直接派子 agent

```
spawn_agent(task="In todo.py, replace line 1 with 'hhhhhhhh' (use edit_file),
             then run pytest and report PASS/FAIL with the failing test names if any.")
```

**子 agent (在 /tmp/minicode-sub-0jzgy0ag 里, M2.7 跑了 4 轮)**:

```
Turn 1: Read(path='todo.py')
        → 看到第 1 行是 docstring 开头 """MiniCode Agent v6 — ..."""

Turn 2: edit_file(path='todo.py', old_text='"""MiniCode Agent v6 — ...',
                  new_text='hhhhhhhh')
        → 改成功

Turn 3: bash(command='pytest 2>&1 | head -100')
        → 跑 pytest, 收集失败 (SyntaxError on line 5)

Turn 4: 报告 "FAIL — syntax error in todo.py"
```

**主 agent (Turn 2)**: 拿到子 agent 结果

```
sub-agent finished. FAIL — syntax error in todo.py.
diff: 2 line(s) changed across 1 file(s).

Full diff (use this verbatim if user wants to apply):
diff --git a/todo.py b/todo.py
@@ -1,4 +1,4 @@
-"""MiniCode Agent v6 — sub-agents for "explore-without-pollution" workflow.
+hhhhhhhh
 ...
```

**主 agent 的回复给用户**:

> 结果: **FAIL** ❌
>
> 修改后 todo.py 出现 **SyntaxError** — 把多行文档字符串的开头 `"""` 替换成 `hhhhhhhh` 后, Python 把后面的文本当成了普通代码, 导致第 5 行的 em dash `—` 变成了无效字符.
>
> 修改内容:
> ```
> -"""MiniCode Agent v6 — sub-agents for "explore-without-pollution" workflow.
> +hhhhhhhh
> ```
>
> 要我把这份修改应用到主工作区吗?

**用户**: "不要"

**结果**: 主 workspace 的 todo.py **没动过**. worktree 自动清理 (因为子 agent 报告 FAIL → 应该保留, 但 token 在第二次 实测中被路径含 FAIL 关键字检测到, worktree 保留了 — 你能 `cd /tmp/minicode-sub-0jzgy0ag` 去看子 agent 留下的状态).

**关键观察点**:
- 主 agent **不自动 apply**, 只是报告 + 询问. 这是 spawn_agent 设计的核心 — 决策权在用户/主 agent
- 主 agent 看到的 diff 是 **真 diff** (字符级一致, 不是模型 hallucinate 的)
- 子 agent 跑期间没触发 todo NAG (虽然它跑了 4 轮没调 todo, 但**子 session is_subagent=True**, NAG 跳过)
- 整个 spawn_agent 走完用了**约 0.044 元** (MiniMax 真 backend 计费)

---

## 4. 修复迭代 — 5 次实测发现的真问题

spawn_agent 的实现**第一版能跑通**单元测试, 但**真实跑出来一堆问题**, 每个都是新的发现. 5 次实测对应 5 个 commit:

### 4.1 实测 1: 模型完全不用 spawn_agent

**第一版**: 在 `build_default_registry` 注册了 spawn_agent 工具, SYSTEM 里加了一段说明. 跑实测, M2.7 看到"试一下"**直接 edit_file 主 workspace** + bash pytest, 完全无视 spawn_agent. 主 workspace 的 todo.py 真被改成 hhhhhhhh 了几秒, 然后 agent 自己又 edit_file 改回去.

**根因**: 工具存在 ≠ 模型会用. SYSTEM 里的引导太弱, M2.7 没接到"应当用 spawn_agent" 的强信号.

**修法 (commit `481ab1a`)**:

1. SYSTEM 加 MANDATORY 段, 明确触发词:
   ```
   SUB-AGENT (spawn_agent) — MANDATORY for exploratory changes:
   Trigger words: "试一下", "试试", "试着", "try", "see if", "experiment", ...
   ```
2. 加 few-shot 例子 (正反两个):
   - 正例: 用户说"试一下" → 应该 spawn_agent
   - 反例: 用户说"把 X 改成 Y" (没"试") → 直接 edit_file
3. spawn_agent 的工具 description 从被动说明改成主动推销:
   ```
   *** USE THIS for ANY 'try / 试一下 / 试试' request from the user. ***
   This is the ONLY safe way to apply experimental changes.
   ```

**效果**: 实测 2 跑同一个 query, M2.7 立刻派 spawn_agent.

**教训**: prompt 是软约束, 但写法决定听不听. 软约束 + few-shot + 工具 description 主动推销 = 强信号.

### 4.2 实测 2: 5 个连锁问题

第一个修生效了, M2.7 用 spawn_agent 了. 但**新问题暴露**:

1. 主 agent 给用户报告 diff 时**完全是编的** ("class TodoList: ..." 那种 — 真 diff 里完全没有这个类)
2. 子 agent 跑时触发了主 agent 的 todo NAG
3. 子 agent 用 ` cd "C:\Users\..." && pytest ` 多此一举 cd 到 worktree (它已经在 worktree 里了)
4. spawn_agent 调 git init 时把 .env (含密钥) commit 进了根 commit
5. 主 agent 显示 diff 时用 ```diff 块, 触发 codeblock NAG

**根因 (1)**: spawn_agent 的 ToolResult.data 含真 diff, 但 data 不喂模型 (这是 05 设计 — text 给模型, data 给程序). 模型只看到 text 里的 "diff: 2 line(s) changed", 看不到内容, 报告时只能编.

**根因 (2)**: 子 agent 跑时计数器照常 +1, 触达 3 → NAG 注入. 但子 agent 没 todo 工具, NAG 是噪音.

**根因 (3)**: 子 agent 的 SYSTEM 没说"你的 cwd 已经在 worktree 里了".

**根因 (4)**: WORKTREE_FAILED 错误消息太短 ("missing .git, run git init"), 模型按字面照做, git add 把 .env 也加了.

**根因 (5)**: 主 agent 的 SYSTEM 没说"显示 diff 给用户时不要用 ```diff 块".

**修法 (commit `3a97423`)**:

1. spawn_agent 的 ToolResult.text 加上**真 diff 内容** (截断到 4000 字符, 全文仍在 data["diff"]):
   ```python
   text_parts.append(f"\nFull diff (use this verbatim if user wants to apply):\n{diff_preview}")
   ```
2. Session 加字段 `is_subagent: bool = False`. spawn_agent 创建子 session 时设 True. ReActAgent.run() 检查 NAG 时跳过子 session
3. 子 agent SYSTEM 加: "YOU ARE ALREADY AT THE WORKTREE — your cwd is exactly the workspace above. Do NOT cd."
4. WORKTREE_FAILED 错误消息升级为多行引导, 包含 .gitignore heredoc + 警告 .env 风险
5. 主 agent SYSTEM 加: "When showing diffs to the user, DO NOT wrap in ```diff or ```patch — that triggers the codeblock-dodge detector."

**效果**: 实测 3 跑通, 主 agent 报告的 diff 准确, 没 NAG 噪音. 但**子 agent 的 cd 警告没生效**(模型仍 cd), `主 agent` 仍可能用 ```diff (但这次是无 lang 的 ```, 没触发 NAG, 算擦边过).

**教训**: 一次实测往往暴露 N 个问题, 修起来要分清"prompt 能解决的" 和 "需要系统层兜底的".

### 4.3 实测 3: spawn_agent 拿到的 diff 有 2/3 是 .pyc 噪音

实测 3 跑出 spawn_agent **结果对了** (M2.7 派对了, 子 agent 跑对了, 主 agent 报告对了). 但**返回的 diff** 是这样:

```diff
diff --git a/tests/__pycache__/conftest.cpython-312-pytest-7.4.4.pyc ...
Binary files differ
diff --git a/tests/__pycache__/test_tools.cpython-312-pytest-7.4.4.pyc ...
Binary files differ
diff --git a/todo.py ...
@@ -1,4 +1,4 @@
-"""MiniCode Agent v6 — ...
+hhhhhhhh
```

3 个 hunk: 2 个是 `.pyc` 改动 (子 agent 跑 pytest 时 Python 写的字节码缓存), 1 个是真 diff. **真 diff 被噪音稀释**, 而且 text 截断后用户主要看到 .pyc.

**根因**: 我们 `_git_worktree_diff` 用 `git add -N .` 把 untracked 文件都标 intent-to-add 让它们出现在 diff. 但 `add -N` **绕过 .gitignore**, .pyc 也被加进去了.

**修法 (commit `5e52340`, 第一次尝试)**:

1. 不用 `git add -N .`, 改用 `git ls-files --others --exclude-standard` 拿"untracked 但 NOT ignored" 列表, 再逐个 `add -N --` 它们
2. 在 worktree 根写一个 `.gitignore` 文件, 含 `__pycache__/` / `*.pyc` / `.env` 等模式
3. WORKTREE_FAILED 错误消息也升级 — 模板 .gitignore 含同样的 patterns

**实测发现**: `.git/info/exclude` 不被 git 尊重 (worktree 的 gitdir/info/ 不在 git 的 read 路径里). 必须用 worktree 根的 `.gitignore`.

**效果**: .pyc 不再进 diff. 但**新问题**: 我们写的 .gitignore 文件**自己**进了 diff (因为 .gitignore 是 tracked 的, 写它就是改它).

### 4.4 实测 4: spawn_agent 自己造的 .gitignore 修改进了 diff

实测 4 跑完, diff 长这样:

```diff
diff --git a/.gitignore b/.gitignore
@@ -11,3 +11,19 @@ node_modules/
+# minicode spawn_agent: ignore build/cache/secrets in diff
+__pycache__/
+...
diff --git a/todo.py
@@ -1,4 +1,4 @@
-"""MiniCode Agent v6 — ...
+hhhhhhhh
```

两个 hunk: `.gitignore` 加了 16 行 (我们 spawn_agent 自己写的!) + todo.py 真改动. 用户看 diff 看到一坨 .gitignore 变更, 跟任务无关.

**根因**: 任何我们在 worktree 里写的文件都是改动, 都进 diff. 不能"为了不污染 diff" 而**修改 worktree 文件**.

**修法 (commit `abf0113`, 真正的 fix)**:

完全不动 worktree 的 .gitignore. 改用 **git pathspec exclude** — 在 git 命令里直接传:

```python
def _build_pathspec_excludes() -> list[str]:
    specs = []
    for p in _WORKTREE_EXCLUDE_PATTERNS:
        if p.endswith("/"):
            specs.append(f":(exclude,glob)**/{p.rstrip('/')}/**")
        else:
            specs.append(f":(exclude,glob)**/{p}")
    return specs

# ls-files 加 pathspec
git ls-files --others --exclude-standard -- :(exclude,glob)**/__pycache__/** :(exclude,glob)**/*.pyc ...

# git diff 也加 pathspec
git diff HEAD -- :(exclude,glob)**/__pycache__/** ...
```

这样**不动磁盘任何文件**, exclude 完全在调用层控制.

**额外好处**: pathspec 也过滤 *tracked* 文件的修改 (老方案的 .gitignore 只管 untracked).

**效果**: 实测 5 跑出来 diff 干净如初 — **240 字符, 1 个 hunk, 仅 todo.py 4 行**. 无 .pyc / 无 .gitignore 段. 这是 spawn_agent 本应有的样子.

**教训**: 工具不应该为了自己工作而修改用户/worktree 的状态. "在 worktree 里写文件让自己工作" 听起来 OK, 实际是给 diff 引入虚假改动.

### 4.5 实测 5 (最终): 完全闭环

```
diff: str[240 chars]   ← 之前是 842, 含 600+ 字符噪音
worktree: null          ← 任务成功, 自动清掉
sub_history_len: 9
sub_tokens: [14310, 363]
```

主 agent 报告:

> 结果: FAIL ❌
> 修改后 todo.py 出现 SyntaxError ...
> 修改内容: -"""MiniCode... +hhhhhhhh
> 要我把这份修改应用到主工作区吗?

完整工作流闭环. 主 workspace 全程没动. 0.044 元.

---

## 5. 跑起来长什么样

启动:

```sh
python todo.py
```

跟 05 一样的 banner, 多了 `spawn_agent` 在工具列表里:

```
tools  : LS, Glob, Grep, Read, write_file, append_file, edit_file,
         apply_patch, todo, spawn_agent, bash
```

输入"试一下"或"try" 类的 query → spawn_agent 触发.

输入"改 X" 类直接命令 → 直接 edit_file 主 workspace, 不派子 agent.

启动 spawn_agent 时会显示一个 banner:

```
┌─ 🚀 spawn_agent 启动 ───────────────────────────────────────────────────
│ task     : <子 agent 的任务描述>
│ worktree : C:\Users\.../AppData\Local\Temp\minicode-sub-XXX
└───────────────────────────────────────────────────────────────────────
```

子 agent 跑期间显示子 agent 的完整 turn (Read / edit_file / bash / ...). 跑完, 主 agent 拿到结果继续.

---

## 6. spawn_agent 的工具层细节

### 6.1 工具签名

```python
def tool_spawn_agent(task: str, _session: Session) -> ToolResult:
    """派子 agent 在隔离 git worktree 里跑探索式任务."""
```

`_session` 由 dispatch 注入 (反射机制, 见 [05 README §1](../05-session-and-streaming/README.md#1-session--把散落的状态收回来)). 用来累加 token 到主 session.

`task` 是主 agent 给子 agent 的任务描述 — 应当**同时包含改动 + 验证步骤**:

> "In todo.py replace line 1 with 'X', then run pytest and report PASS/FAIL"

### 6.2 返回值结构

```python
ToolResult.success(
    text="""
    sub-agent finished. <子 agent 最后一条 assistant 消息的摘要>
    diff: N line(s) changed across M file(s).

    Full diff (use this verbatim if user wants to apply):
    <unified diff, 截断到 4000 字符>

    [worktree retained for inspection: <path>]    ← 仅失败时
    sub-agent tokens: X in + Y out
    """,
    diff="<full unified diff>",                   # 完整 diff (data, 不喂模型)
    worktree="<path or null>",                    # 失败时保留, 成功 None
    sub_history_len=N,                            # 子 agent 跑了几条消息
    sub_tokens=(prompt_t, completion_t),          # 子 agent 烧的 token
)
```

**关键**: `text` 里有 diff (供模型看), `data["diff"]` 有完整 diff (供主 agent 直接 `apply_patch(patch=data["diff"])` 用).

### 6.3 错误返回

| 错误码 | 触发 | 含义 |
|---|---|---|
| `SPAWN_NOT_AVAILABLE` | `_SPAWN_LLM` 未设 | 通常是测试场景 / 非 REPL 入口. 实际 REPL 启动会自动设 |
| `WORKTREE_FAILED` | 主 workspace 不是 git repo / git worktree add 失败 | 错误消息含完整 git init + .gitignore + commit 引导, 警告 .env 风险 |

### 6.4 子 agent 工具集

`build_subagent_registry` 注册的工具 = 主 agent 工具集 - {`spawn_agent`, `todo`}:

| 子 agent 有 | 子 agent 没有 |
|---|---|
| LS, Glob, Grep, Read | spawn_agent (防递归) |
| write_file, append_file, edit_file | todo (单一目标, 不需规划) |
| apply_patch, bash | |

### 6.5 子 agent 的 SYSTEM (跟主 agent 完全不同)

```
You are a sub-agent spawned by the main agent to explore one specific change.

ISOLATED WORKSPACE: <worktree path>
This is a git worktree of the main project. Any change you make stays here.
You CANNOT break the user's actual code.

YOUR TASK IS BOUNDED. Do exactly what the parent asked, then stop.
- You do NOT have a `todo` tool — single goal, no planning needed.
- You do NOT have a `spawn_agent` tool — no recursion.

YOU ARE ALREADY AT THE WORKTREE — your cwd is exactly the workspace above.
Do NOT `cd` into it from bash.

OUTPUT: keep your final reply terse — it becomes a tool_result for the parent.
State the outcome in one or two sentences:
  - "PASS — N/M tests passed"
  - "FAIL — test_xyz failed: <reason>"
  - "DONE — changes made, no test run"
```

### 6.6 _SPAWN_LLM 的设置

模块级全局变量, 在 `repl()` 启动时设:

```python
def repl():
    llm = load_model()
    global _SPAWN_LLM
    _SPAWN_LLM = llm   # 子 agent 复用主 agent 的 llm 实例
    ...
```

这是个**有意的全局**, 替代方案是给 spawn_agent 通过依赖注入传 llm. 但那会让工具 handler 签名复杂化, 且 LLM 在整个进程里就一个实例, 全局是合理的.

### 6.7 worktree 局部排除 — 用 pathspec, 不动磁盘

最初尝试在 worktree 写 .gitignore (实测 3 → 4 演化, 见 §4.3-4.4). 最终方案用 git pathspec:

```python
_WORKTREE_EXCLUDE_PATTERNS = [
    "__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
    "*.pyc", "*.pyo", "*.pyd", ".DS_Store",
    "node_modules/", ".venv/", "venv/",
    ".env", ".env.local",   # 双保险防密钥
]

# 用法 (在 _git_worktree_diff 里):
excludes = _build_pathspec_excludes()
subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--"] + excludes, ...)
subprocess.run(["git", "diff", "HEAD", "--"] + excludes, ...)
```

每个 pattern 转成 `:(exclude,glob)**/<pattern>` 的 git pathspec. ls-files 和 git diff 都加, 这样**新文件和已 tracked 文件**都过滤.

**好处**: spawn_agent 不动 worktree 任何文件. diff 完全干净.

---

## 7. 怎么跑

### 7.1 默认 Ollama (本地, 零成本)

跟 05 一样:

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

cd 06-sub-agents
python todo.py
```

**警告**: 7B 在 spawn_agent 上的可靠性远不如 M2.7. 它可能:
- 不识别"试一下" 触发词, 直接 edit 主 workspace
- 派 spawn_agent 但任务描述模糊, 子 agent 跑不出来
- 子 agent 卡在某一步, 反复重试

7B 上这个特性**有 demo 价值, 没实用价值**. 想真用 → 切 MiniMax.

### 7.2 MiniMax-M2.7 (云端, spawn_agent 推荐)

```bash
cd 06-sub-agents
cp .env.example .env
# 编辑 .env: MINICODE_BACKEND=minimax + MINIMAX_API_KEY=<你的 key>
python todo.py
```

`.env` gitignored, pre-commit hook 防泄露 (见 [05 README §8](../05-session-and-streaming/README.md#8-安全脚手架--env-和-pre-commit-hook)).

### 7.3 主 workspace 必须是 git repo

spawn_agent 要求**当前 cwd 是 git repo (有 .git 且有至少一个 commit)**. 没有的话第一次调 spawn_agent 会拿到 `WORKTREE_FAILED`, 错误消息含完整 init 步骤:

```
spawn_agent requires <path> to be a git repository.

To enable spawn_agent here, run THESE EXACT COMMANDS (the order matters —
.gitignore MUST exist before `git add .` so secrets don't get committed):

  cd <path>
  git init
  cat > .gitignore << 'EOF'
  __pycache__/
  ...
  .env
  .env.local
  EOF
  git add .
  git commit -m 'init for minicode'

WARNING: do NOT skip the .gitignore step. Without it, `git add .`
will stage .env (with your API keys) and __pycache__/*.pyc into the
root commit. Once committed, secrets are hard to scrub.
```

实测 M2.7 看到这条会**按顺序执行**: 先建 .gitignore, 再 git add, 再 commit. **.env 不会被 commit**.

### 7.4 环境变量

跟 05 一样, 加 spawn_agent 专属几个:

```
# 已有 (来自 05)
MINICODE_BACKEND=ollama|minimax
MINICODE_STREAM=1
MINIMAX_API_KEY=...

# v6 新增
MINICODE_SUB_MAX_ROUNDS=10        # 子 agent 单次最多跑几轮
MINICODE_SUB_TIMEOUT=300          # 子 agent 整体超时 (秒)
MINICODE_SUB_DIFF_MAX=4000        # 返回 text 里 diff 截断阈值
```

---

## 8. 读源码的建议路线

[todo.py](todo.py) ~2300 行, 在 05 的基础上加 spawn_agent 相关 ~200 行. 推荐顺序:

1. **`_WORKTREE_EXCLUDE_PATTERNS` 常量** — 排除列表的 source of truth
2. **`_git_worktree_create`** — git worktree add + 错误消息引导
3. **`_build_pathspec_excludes` + `_git_worktree_diff`** — pathspec exclude 怎么用
4. **`_git_worktree_remove`** — 清理逻辑 (`git worktree remove --force`)
5. **`SUB_AGENT_SYSTEM_TEMPLATE`** — 子 agent 的 system prompt
6. **`tool_spawn_agent`** — 主入口, 整个流程串起来
7. **`build_subagent_registry`** — 子 agent 工具集 (主 - spawn_agent - todo)
8. **`Session.is_subagent` 字段** — 子 session 标记, 让 NAG 跳过

测试: [tests/test_tools.py](tests/test_tools.py) 末尾 `TestSpawnAgent` 类, 10 个测试覆盖:

- 基础: 非 git workspace 报错, _SPAWN_LLM 未设报错
- 隔离: 子 agent 改动不出现在主 workspace
- 清理: 失败 worktree 保留, 成功删除
- 状态: 主 WORKDIR 在 spawn 后正确恢复
- token 累加到主 session
- pathspec exclude 生效 (.pyc/.gitignore 不进 diff)
- 错误消息含 git init 引导

---

## 9. 已知局限

诚实交代:

| 局限 | 严重度 | 备注 |
|---|---|---|
| 7B 在 spawn_agent 上不可靠 | 🟧 设计权衡 | 切 MiniMax-M2.7. 7B 元认知不够派工 |
| 同步阻塞 — 不支持并行多个 spawn_agent | 🟨 简单胜复杂 | 单机 GPU 同时也只能一个推理. 云端能并行但实现复杂度大 |
| 模块级 WORKDIR 在 spawn 期间被 monkey-patch | 🟨 设计权衡 | spawn_agent 跑时把 module-level WORKDIR 临时指 worktree, 跑完恢复. 不优雅但可工作. 真正干净的做法是把 WORKDIR 进 Session, 是 v7 的事 |
| 主 agent 用 ``` 块显示 diff | 🟨 模型行为 | SYSTEM 警告"不要用 ```diff 块", M2.7 改用无 lang 的 ```. 没触发 codeblock NAG, 算擦边过. 不修了 |
| 子 agent 仍可能 cd 到 worktree | 🟨 模型行为 | SYSTEM 写了"不要 cd", M2.7 偶尔仍 cd. 不影响结果, 不修 |
| 要求主 workspace 是 git repo | 🟧 设计前提 | 不在 git repo 里跑直接报 WORKTREE_FAILED 并教用户怎么 init |

---

## 10. v6 → v7 候选方向

按"真痛点 vs 假想需求"排:

**真痛点, v7 应该做** (v7 做了 Session 持久化, v8 做了 Context 压缩):
- ~~**Session 持久化**~~ — v7 完成
- **WORKDIR 进 Session** — 当前模块级 monkey-patch 是真技术债. 跟 spawn_agent 联系起来时尤其明显. v9 候选
- ~~**Context 压缩**~~ — v8 完成 (见上 §8.1)

**假想需求, 不该做**:
- 子 agent 并发 — 单机场景没意义
- 子 agent 跨 backend (主 MiniMax, 子 Ollama) — 用户可以**整体**切 backend, 不需要这种复杂度
- 多层子 agent (深度 2+) — 7B 递归爆炸, M2.7 也没强需求

**真要做但很复杂**:
- 子 agent 失败时的精细恢复策略 — 现在简单粗暴 "失败保留 worktree". 想搞"自动重试"会引入新问题
- prompt caching — MiniMax 是否原生支持没确认, 调研工作量不小

---

## 历史包袱: 07 / 06 / 05 / 04 / 03 / 02 / 01 都做了什么

不在这份 README 重写 — 各自 README 仍然有效:

- [07-session-persistence/](../07-session-persistence/) — Session 跨进程存盘 (`/save` / `/load` / auto-save). v8 继承
- [06-sub-agents/](../06-sub-agents/) — `spawn_agent` 在隔离 git worktree 里跑探索式任务. v8 继承
- [05-session-and-streaming/](../05-session-and-streaming/) — Session 状态管理 + 流式 + 双后端 + token 可见. v8 继承
- [04-structured-tool-calls/](../04-structured-tool-calls/) — Ollama 后端 + structured tool_calls + apply_patch
- [03-atomic-tools/](../03-atomic-tools/) — 三层工具架构 + 读后写乐观锁 + 42 测试 + CI
- [02-sandboxed/](../02-sandboxed/) — REPL + bash 沙箱探测 + 大文件分片
- [01-bash-only/](../01-bash-only/) — 最小 ReAct + 5 工具

按版本号顺序读, 能看清"agent 是怎么从单进程脚本长到能派子任务" 的全过程.

---

## 一句话总结

**v6 不是"加了酷炫的多 agent 能力". v6 解决的是一个具体痛点: 用户说"试一下"时, 怎么不污染主 workspace.** 5 次实测, 5 个 commit, 每次都基于真问题修. 最终 spawn_agent 工作流闭环 — 隔离 worktree → 子 agent 跑 → 真 diff 回到主 agent → 询问用户是否采纳.

技术上不复杂 (~200 行新代码 + 10 个测试). 难的是**让模型真用 spawn_agent**, 不是"工具存在"那么简单. SYSTEM 写法 / 工具 description 写法 / few-shot 例子 — 三处都得到位才行. 跟 04/05 的"prompt 是软约束, 系统是硬约束"原则一脉相承.
