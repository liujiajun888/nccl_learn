# NCCL 软件栈：从一次 AllReduce 到代码开发

这份教程面向有 CUDA/C++ 基础、希望理解或进入 NCCL 开发的读者。目标是先建立一张能用于读代码和定位问题的软件栈地图，再逐步深入具体实现。

**第一次读 NCCL，请从 [00｜初学者主线](00-mainline.md) 开始，不要先按 01→12 逐行硬读。**先明白一次 AllReduce 怎样从 API 走到 GPU/NIC，再回头补初始化、拓扑、transport 和各种细节。

## 1. 初学者推荐路线

```text
00 初学者主线：先建立整体流程
  ↓
01 API 与执行约定：应用交给 NCCL 什么
  ↓
02 Host 执行：请求怎样变成 task、work、plan 和 launch
  ↓
03 Device 执行：CTA 怎样读取 work 并执行算法
  ↓
04 集合算法：Ring/Tree 怎样得到 AllReduce
  ↓
05 Communicator → 06 拓扑 → 07 Transport：这些资源从哪来
  ↓
08 Proxy 与网络 → 09 内存与生命周期：跨机推进和回收
  ↓
10 开发工作流 → 11 测试与调试：开始做小改动
```

[软件栈全景](appendix/reference/software-stack.md) 用作**目录和名词地图**，不必在第一遍逐节精读。

每个阶段只记录四件事：**输入、动作、产物、下一消费者**。第一次阅读可先跳过递归搜索细节、调度公式、FIFO 容量、特殊硬件和异常分支。

## 2. 读完主线应该能回答什么？

如果“整个软件栈”指一次普通 host collective 从 API 到 GPU/NIC 的完整链路，读完主线后你应该能解释：

| 问题 | 对应内容 |
|---|---|
| 应用的一次调用包含哪些信息？ | buffer、count、datatype、op、comm、stream |
| 初始化准备了哪些可复用资源？ | peerInfo、拓扑、算法图、channel 邻接、transport、devComm |
| algorithm、protocol、transport、channel 有什么区别？ | 数据依赖、分片与进度、连接实现、并行工作分区 |
| task、work、plan 分别给谁使用？ | host 调度、device 工作描述、一次 launch |
| `channelMask` 怎样影响 CUDA grid？ | 置位数决定活动 CTA 数，置位顺序决定 block→channel 映射 |
| launch 成功是否等于结果可读？ | 不是；还要看 stream/event，资源回收另有条件 |

这不等于覆盖所有 NCCL 算法、API、硬件分支，也不包含 NVIDIA KMD、GPU 固件或 NIC 驱动内部实现。**先能讲清一条主线，再按问题扩大范围。**

## 3. 章节按用途查阅

| 章节 | 什么时候读 |
|---|---|
| [00 初学者主线](00-mainline.md) | 第一次建立全栈流程，必须优先读 |
| [01 API 与执行约定](01-api-and-execution.md) | 理解参数、Group、stream/event 和完成语义 |
| [02 Host 执行](02-host-execution.md) | 用一条小案例读懂 API→task→调优→work/plan→launch |
| [03 Device 执行](03-device-execution.md) | 理解 CTA、work、Primitives 和协议推进 |
| [04 集合算法](04-algorithms.md) | 理解 ReduceScatter+AllGather、Ring 与 Tree |
| [05 Communicator](05-communicator.md) | 理解成员会合、初始化阶段和按需连接 |
| [06 拓扑与通信图](06-topology.md) | 理解硬件路径怎样变成 channel 邻接 |
| [07 Transport](07-transports.md) | 理解 P2P/SHM/NET 的缓冲、映射和连接 |
| [08 Proxy 与网络](08-proxy-and-network.md) | 理解 proxy op、网络 request、GPU 可见性与槽位回收 |
| [09 内存与生命周期](09-memory-and-lifetime.md) | 理解用户 buffer、内部缓冲、描述和注册何时回收 |
| [10 开发工作流](10-development-workflow.md) | 做一次不改变行为的源码插桩并建立开发闭环 |
| [11 测试与调试](11-testing-and-debugging.md) | 按初始化、连接、提交、GPU、NET、完成阶段定位问题 |
| [环境准备](appendix/reference/environment.md) | 需要构建本地 NCCL、核对加载库和运行基线时 |
| [软件栈全景](appendix/reference/software-stack.md) | 回查分层职责和 `src/` 目录地图 |

## 4. 用实践确认“看懂”

不要只以读完页数作为标准。无 GPU 时先完成前两项；有 GPU 时再完成第三项，并用第四项检验自己的分析方式。

1. **CPU 算法闭环**：运行 Ring 模拟器，解释每份贡献为何恰好加入一次。
2. **源码阅读闭环**：用 `grad` 这个指针串起 `ncclInfo → task → work → launch`，指出每一步有没有复制 payload。
3. **GPU 正确性闭环**：运行两卡示例，确认输出、异步状态和 stream 完成，再做一次不改变行为的插桩。
4. **调试闭环**：遇到等待时，能说出最后一个已完成阶段，以及下一阶段由谁推进。

在项目根目录运行 CPU 模拟：

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

GPU 构建与运行方法见 [环境准备](appendix/reference/environment.md) 和 [示例说明](examples/README.md)。

## 5. 进阶参考与专题附录

主线看懂以后，先查 [附录阅读指南](appendix/README.md)，再按问题选择，不必全部顺序阅读：

- **Host 与算法实现：**[Host 执行进阶](appendix/deep-dive/host-execution.md)、[Device 执行进阶](appendix/deep-dive/device-execution.md)、[集合算法进阶](appendix/deep-dive/algorithms.md)。
- **初始化与拓扑：**[Communicator 进阶](appendix/deep-dive/communicator.md)、[Topology 进阶](appendix/deep-dive/topology.md)、[ncclCommInitRank 完整初始化精读](appendix/initialization/nccl-comm-init-rank.md)。
- **传输、网络与生命周期：**[Transport 进阶](appendix/deep-dive/transports.md)、[Proxy 与网络进阶](appendix/deep-dive/proxy-and-network.md)、[内存与生命周期进阶](appendix/deep-dive/memory-and-lifetime.md)。
- **开发与验证：**[开发工作流进阶](appendix/deep-dive/development-workflow.md)、[测试与调试进阶](appendix/deep-dive/testing-and-debugging.md)。
- **接口、测试与性能：**[集合接口与布局](appendix/reference/collectives.md)、[nccl-tests](appendix/reference/nccl-tests.md)、[性能分析](appendix/reference/performance.md)。
- **扩展机制：**[现代特性](appendix/features/modern-features.md)、[NVLS](appendix/features/nvls.md)、[CollNet](appendix/features/collnet.md)、[MNNVL](appendix/features/mnnvl.md)、[NVML](appendix/features/nvml.md)。

## 6. 贯穿案例与源码基线

正文的大案例是两台机器、每台四张 GPU、八个 rank，每 rank 对 64 MiB FP32 做 Sum AllReduce。新手导读则先使用两个 rank、四个 float，避免在大数据和复杂拓扑中丢失主线。

主线先理解普通 host collective，再用 Ring/Simple 追踪具体实现；实际算法和传输路径由任务、拓扑及运行配置共同选择。

| 源码 | 版本 | 提交 |
|---|---|---|
| NCCL | 2.32.3 | `12df1a11afad322be5a204a2db890161cbf8131d` |
| nccl-tests | 2.20.0 | `b4d5beebca8a76cf01335f724d154b9b9d394d96` |

源码链接指向项目里的 `nccl/` 和 `nccl-tests/`。行号对应上述提交；换版本时按符号重新定位。

本教材在 macOS 上整理，GPU/CUDA/MPI 示例未在此环境编译或运行。GPU 输出均标为预期，性能算例均标为教学数据；CPU 模拟验证的是算法调度与结果。

**现在从 [00｜初学者主线](00-mainline.md) 开始。**
