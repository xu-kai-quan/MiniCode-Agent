# MiniCode Agent

[![tests](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/xu-kai-quan/MiniCode-Agent/actions/workflows/test.yml)

一个用本地 Qwen3.5-2B 模型驱动的**最小化编码 agent**, 单文件实现, 没有任何框架封装, 适合用来理解"AI agent 到底是怎么工作的"。

仓库里有三个独立项目, 按学习顺序排列, 各有各的 README, 想读哪个就进哪个目录:

| 目录 | 定位 | 适合 |
|---|---|---|
| [01-bash-only/](01-bash-only/) | 只有 5 个工具的一次性任务 agent, 跑完一句话就退出 | 第一次接触 agent, 想看清最小骨架 |
| [02-sandboxed/](02-sandboxed/) | 加了交互式 REPL + bash 沙箱探测 + 大文件分片写入 | 想看 agent 怎么演化成可用的日常工具 |
| [03-atomic-tools/](03-atomic-tools/) | 三层工具架构 (LS/Glob/Grep/Read 原子层) + 读后写乐观锁 + 42 个 pytest | 想看怎么把 agent 做扎实, 经得起测试 |

三个项目都只有一个 `todo.py` 文件, 没有 langchain / autogen 之类的框架包装 — **你看到的就是全部真相**。

## 快速开始

挑一个目录进去, 按各自 README 跑:

- [01-bash-only/README.md](01-bash-only/README.md)
- [02-sandboxed/README.md](02-sandboxed/README.md)
- [03-atomic-tools/README.md](03-atomic-tools/README.md)

## 共同依赖

```sh
pip install torch transformers safetensors
```

需要本地放一份 Qwen3.5-2B 模型 (默认路径 `E:/MYSELF/model/qwen/Qwen3.5-2B/`, 各自 `todo.py` 顶部可改)。

## 测试

每个版本都有 pytest 套件, CI 在 GitHub Actions 上每次 push / PR 自动跑 — 见 [.github/workflows/test.yml](.github/workflows/test.yml). 测试不依赖 torch/transformers (`tests/conftest.py` 把它们 stub 掉了), 所以 CI 不需要拉几个 GB 的 ML 栈。

## 设计理念

- **循环就是 agent 的全部本质** — 模型输出 → 解析工具调用 → 执行 → 把结果喂回去, 直到模型不再调工具。
- **工具是查表分发, 不是硬编码** — 加新工具 = 加一个函数 + 登记表项, 循环本身不动。
- **路径沙箱在工具层做, 不靠 prompt** — 不能指望 prompt 拦住模型。
- **prompt 是软约束, 系统是硬约束** — 关键不变量必须在系统层兜底, 因为小模型不一定听 prompt。
- **日志要"可审查"而非"好看"** — 每一步: 模型在想什么 / 调了什么工具 / 拿到什么结果, 全部打印出来, 方便 debug 和学习。
