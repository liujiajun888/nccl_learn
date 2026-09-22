# 从会用到看懂：NCCL 系统学习教程

这不是 API 速查表，而是一条完整的学习链路：**通信的数学含义 → 正确使用 CUDA/NCCL → 初始化与执行机制 → 数据如何跨 GPU/跨机器 → 如何测量、解释和优化**。

教材面向有 C/C++ 和基本 CUDA 经验的读者。第一次学习分布式通信也可以读；涉及网络、拓扑和协议的概念会从例子解释。中文讲解，保留英文术语与源码符号，便于对照实现。

## 版本与适用范围

| 项目 | 本教材使用的本地基线 |
|---|---|
| NCCL | **2.32.3**，commit `12df1a11afad322be5a204a2db890161cbf8131d` |
| nccl-tests | **2.20.0**，commit `b4d5beebca8a76cf01335f724d154b9b9d394d96` |
| 编写日期 | 2026-09-22 |
| GPU 实验环境 | Linux + NVIDIA GPU + 匹配的 CUDA Toolkit/驱动；多机增加网络与 MPI |
| 本地编写环境 | macOS arm64，未安装 CUDA，不能运行 NCCL GPU 实验 |

源码行号针对上述 commit；更新仓库后以**文件路径与符号名**重新定位。以 Ring/Simple 等常规 host API 路径建立模型，再学习本版本的 device API、GIN、对称内存、CE 和新调度路径，不把某一条路径误当成 NCCL 的全部。

特别注意：当前源码已经提供 `ncclAlltoAll`、`ncclGather`、`ncclScatter` 和 per-collective config。网上针对旧版本的“没有这些 API”不能直接套用。

## 目录

| 章 | 内容 | 学完应能回答的问题 |
|---|---|---|
| [00 环境与第一轮实验](00-environment.md) | 源码构建、动态库、第一轮正确性测试 | 我实际运行的是哪一个 NCCL？ |
| [01 建立全局模型](01-mental-model.md) | rank、comm、stream、软硬件分工 | 一次 AllReduce 到底由谁完成？ |
| [02 集合通信的数学与内存布局](02-collectives.md) | 八类 collective、Send/Recv、count 与 in-place | 每个 rank 输入输出多大，结果放哪里？ |
| [03 CUDA 与 API 正确性](03-cuda-semantics.md) | stream、group、事件、异步错误 | CPU 返回成功后能否马上读结果？ |
| [04 Communicator 生命周期](04-communicator.md) | bootstrap、初始化、连接、销毁与恢复 | GPU 在开始通信前要达成什么共识？ |
| [05 Host 执行主线](05-host-execution.md) | API → task → plan → launch → proxy | 一行 ncclAllReduce 如何成为执行任务？ |
| [06 拓扑与图搜索](06-topology.md) | PCIe、NVLink、NUMA、NIC、ring/tree | 为什么换卡位或进程绑定会改变速度？ |
| [07 Transport 与网络](07-transports.md) | P2P、SHM、NET、GDR、CPU/GPU 协作 | 数据经过哪些内存和硬件？ |
| [08 集合通信算法](08-algorithms.md) | Ring、Tree、NVLS、CollNet、PAT | 为什么没有一个永远最快的算法？ |
| [09 Device 与协议](09-device-protocols.md) | channel、chunk、slice、step、LL/LL128/Simple | 数据切成小块后如何有序、无覆盖地流动？ |
| [10 读懂 nccl-tests](10-nccl-tests.md) | 参数、计时、正确性、带宽公式 | busbw 到底测到了什么？ |
| [11 性能分析与调优](11-performance.md) | 小包/大包、对照实验、overlap | 先优化哪个瓶颈，如何证明有效？ |
| [12 调试与故障定位](12-debugging.md) | 挂起、日志、异步错误、系统检查 | 是调用不匹配、GPU 错误还是网络故障？ |
| [13 现代 NCCL 进阶](13-advanced.md) | Graph、注册、device API、插件与新特性 | 哪些优化需要应用与硬件共同配合？ |
| [14 源码阅读方法与索引](14-source-map.md) | 六次阅读任务、模块索引、术语表 | 如何带着问题读源码而不迷路？ |
| [15 分级练习与参考答案](15-exercises.md) | 手算、模拟、编程、测量、源码追踪 | 我是否真正建立了可验证的理解？ |
| [16 分布式训练中的 NCCL](16-training-integration.md) | DDP、分片、张量/流水/专家并行 | 通信优化为什么必须看训练关键路径？ |
| [实验代码](examples/README.md) | 单进程多卡、MPI 多进程、CPU Ring 模拟 | 如何从最小程序开始亲手验证？ |

## 三条学习路线

### 路线 A：先会用，约 3～5 个学习日

`01 → 02 → 00 → 03 → examples → 10 → 12`

目标：自己写出并验证两卡 AllReduce，解释 rank/device/count 的差别，能看懂测试输出并诊断常见挂起。每天约 1～2 小时；编译和机器准备时间另计。

### 路线 B：读懂实现，约 2～3 周

完成 A 后，按 `04 → 05 → 06 → 07 → 08 → 09 → 14` 阅读。

每章先读模型，再只打开推荐的几个函数。每次记录三个问题：**谁持有什么状态？谁推动进度？什么条件允许进入下一步？** 不必从 `init.cc` 第一行一直读到最后一行。

### 路线 C：性能与应用，约 1～2 周

`10 → 11 → 16 → 13 → 15 的综合题`

目标：设计可信的实验，区分软件开销、链路瓶颈与应用等待；知道何时应该调 NCCL，何时应该改 bucket、GPU/NIC 亲和性或计算图。

## 没有 NVIDIA GPU 也能开始

在项目根目录运行：

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

它会展示 Ring 的两阶段以及最终校验；**这是数学/调度教学模型，不模拟 CUDA 并发、协议开销或真实硬件性能**。

前 2 章、算法手算、源码阅读和大部分练习都可离线完成。GPU 实验留到 Linux 机器执行，不能用 macOS 的模拟结果代替 NCCL 实测。

## 如何使用教材

1. 所有 shell 实验默认从同时含 `nccl/`、`nccl-tests/`、`nccl-tutorial/` 的项目根目录开始；个别章节会再次标明。
2. 标注“示意”“伪代码”“手算”的内容用于解释；完整可编译示例集中在 `examples/`。
3. 首轮只要求结果正确、路径可解释。先保留默认算法和协议，不急着设置几十个环境变量。
4. 每次性能实验记录版本、GPU/NIC/CPU 拓扑、进程映射、完整命令与原始日志。不要只保存一张峰值截图。
5. 自测题答案就在章节末；综合练习先自行完成再看第 15 章答案。

## 验证边界

教材的 API、参数与源码锚点以本地源码核对；交付检查包含 Markdown 本地链接、代码结构和 CPU 模拟测试。**CUDA 示例及多机实验没有在本次 macOS 环境编译或执行**，文中也不提供冒充实测的跑分。

拿到 Linux GPU 后，建议按 `00 → 单进程示例 → nccl-tests → MPI 示例` 的顺序验证，不要第一步就上大规模集群。上游仓库保持原样；教材和示例全部位于本目录。
