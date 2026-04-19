# MiniCode Agent

一个用本地 Qwen3.5-2B 模型驱动的**最小化编码 agent**, 单文件实现, 没有任何框架封装, 适合用来理解"AI agent 到底是怎么工作的"。

仓库里有两个独立项目, 各有各的 README, 想读哪个就进哪个目录:

| 目录 | 定位 | 适合 |
|---|---|---|
| [v1_mvp/](v1_mvp/) | 一次性任务 agent, 给一句话跑完就退出 | 第一次接触 agent, 想看清最小骨架 |
| [v2_workspace/](v2_workspace/) | 交互式 REPL agent, 类似 Claude Code 风格 | 想看 agent 怎么演化成可用的日常工具 |

两个项目都只有一个 `todo.py` 文件, 不超过 600 行, 没有 langchain / autogen 之类的框架包装 — **你看到的就是全部真相**。

## 快速开始

挑一个目录进去, 按各自 README 跑:

- [v1_mvp/README.md](v1_mvp/README.md)
- [v2_workspace/README.md](v2_workspace/README.md)

## 共同依赖

```sh
pip install torch transformers safetensors
```

需要本地放一份 Qwen3.5-2B 模型 (默认路径 `E:/MYSELF/model/qwen/Qwen3.5-2B/`, 各自 `todo.py` 顶部可改)。

## 设计理念

- **循环就是 agent 的全部本质** — 模型输出 → 解析工具调用 → 执行 → 把结果喂回去, 直到模型不再调工具。
- **工具是查表分发, 不是硬编码** — 加新工具 = 加一个函数 + 登记表项, 循环本身不动。
- **路径沙箱在工具层做, 不靠 prompt** — 不能指望 prompt 拦住模型。
- **prompt 是软约束, 系统是硬约束** — 关键不变量必须在系统层兜底, 因为小模型不一定听 prompt。
- **日志要"可审查"而非"好看"** — 每一步: 模型在想什么 / 调了什么工具 / 拿到什么结果, 全部打印出来, 方便 debug 和学习。
