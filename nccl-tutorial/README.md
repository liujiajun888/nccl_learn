# NCCL 软件栈：从一次 AllReduce 到代码开发

这份教程面向有 CUDA 基础和 UMD 开发经验、准备进入 NCCL 开发的读者。目标是建立一张能用于读代码、改代码和定位问题的软件栈地图。

全书围绕三个问题展开：**每层负责什么，层与层交接什么，修改后怎样验证。** 用一次 AllReduce 串起初始化、连接、任务提交、GPU 执行、网络推进和资源回收。

## 按 01 → 12 顺读

```text
01 软件栈全景
       ↓
02 API 与执行约定
       ↓
03 Communicator → 04 拓扑 → 05 Transport       建立通信关系
       ↓
06 Host 执行 → 07 Device 执行 → 08 算法        把调用变成数据流
       ↓
09 Proxy 与网络 → 10 内存与生命周期            推进和回收
       ↓
11 开发工作流 → 12 测试与调试                  开始修改代码
```

| 章节 | 核心问题 |
|---|---|
| [01 软件栈全景](01-software-stack.md) | NCCL、CUDA、驱动、GPU 和 NIC 怎样分工？ |
| [02 API 与执行约定](02-api-and-execution.md) | 应用交给 NCCL 的是什么工作，如何接入 CUDA 依赖？ |
| [03 Communicator](03-communicator.md) | 分散的 GPU 怎样建立共同通信关系？ |
| [04 拓扑与通信图](04-topology.md) | 硬件连接怎样变成 ring/tree 邻接？ |
| [05 Transport](05-transports.md) | 两端怎样获得可用的缓冲、映射和连接？ |
| [06 Host 执行](06-host-execution.md) | API 参数怎样变成 task、plan 和设备工作描述？ |
| [07 Device 执行](07-device-execution.md) | GPU 怎样取工作、调用原语并推进协议？ |
| [08 集合算法](08-algorithms.md) | Ring/Tree 怎样分工，如何收齐每份贡献？ |
| [09 Proxy 与网络](09-proxy-and-network.md) | CPU、GPU 和 NIC 怎样共同完成跨机传输？ |
| [10 内存与生命周期](10-memory-and-lifetime.md) | 地址、注册、工作描述和通信对象何时回收？ |
| [11 开发工作流](11-development-workflow.md) | 一个需求应该改哪层，怎样做第一项源码实验？ |
| [12 测试与调试](12-testing-and-debugging.md) | 怎样验证修改，并把问题定位到具体阶段？ |

## 随手查阅

- [00 环境准备](00-environment.md)：构建本地 NCCL、确认动态库、跑通两卡基线。
- [示例代码](examples/README.md)：单进程多卡、MPI 多进程、CPU Ring 模拟器。
- [集合接口与布局](appendix-collectives.md)：count、结果归属、原地关系和 Send/Recv。
- [ncclCommInitRank 源码精读](appendix-nccl-comm-init-rank.md)：从 API、Group 调度、GPU 准备与 bootstrap，到 peerInfo、共同能力、同进程成员、拓扑路径与 CPU affinity，含分节小结和当前阅读断点。
- [nccl-tests](appendix-nccl-tests.md)：参数、计时与输出列。
- [性能分析](appendix-performance.md)：延迟/带宽模型、对照实验、计算通信重叠。
- [现代特性](appendix-modern-features.md)：Graph、NVLS、Device API、GIN、RMA 等扩展入口。

每章保留少量关键源码入口，以及一项能核对结果的阅读或实验任务。先读懂正文中的交接关系，再去源码找对应对象；完整 API 清单与特殊路径留在附录。

## 贯穿案例与源码基线

主线设定两台机器，每台四张 GPU，每进程管理一张卡，共八个 rank。每个 rank 对 **64 MiB FP32** 数组做 Sum AllReduce。小数组和四 rank 手算用于展示同一机制；这些布局和数值是教学设定。

先理解普通 host collective，再用 Ring/Simple 这一具体分支追代码。实际算法和传输路径由任务、拓扑及运行配置共同选择。

| 源码 | 版本 | 提交 |
|---|---|---|
| NCCL | 2.32.3 | `12df1a11afad322be5a204a2db890161cbf8131d` |
| nccl-tests | 2.20.0 | `b4d5beebca8a76cf01335f724d154b9b9d394d96` |

源码链接指向项目里的 `nccl/` 和 `nccl-tests/`。行号对应上述提交；换版本时按符号重新定位。

## 从哪里开始动手

在包含三个目录的项目根目录运行 CPU 模拟：

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

有 Linux NVIDIA GPU 时，按 [00](00-environment.md) 构建，再运行单进程示例与 nccl-tests。源码阅读和 CPU 模拟可以先在本机完成。

本教材在 macOS 上整理，GPU/CUDA/MPI 示例未在此环境编译或运行。GPU 输出均标为预期，性能算例均标为教学数据；CPU 模拟验证的是算法调度与结果。

从 [01 软件栈全景](01-software-stack.md) 开始。
