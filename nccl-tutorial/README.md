# NCCL 软件栈：从一次 AllReduce 到代码开发

这份教程面向有 CUDA/C++ 基础、希望理解或进入 NCCL 开发的读者；有驱动或系统编程经验（尤其是用户态驱动，UMD）会更容易衔接。目标是建立一张能用于读代码、改代码和定位问题的软件栈地图。

全书围绕三个问题展开：**每层负责什么，层与层交接什么，修改后怎样验证。** 用一次 AllReduce 串起初始化、连接、任务提交、GPU 执行、网络推进和资源回收。

## 先说结论：能建立全栈地图，但不能只停留在阅读

如果“整个软件栈”指 **NCCL 一次普通 host collective 从 API 到 GPU/NIC 的完整链路**，本教程可以建立较完整的地图：既能说清每层职责，也能沿源码找到交接对象和验证入口。如果指所有 NCCL 算法、所有 API、所有硬件分支，以及 CUDA KMD、GPU 固件和 NIC 驱动内部实现，则不能把这份教程当作完整替代品。

| 范围 | 教程覆盖的内容 | 读后应该能回答 |
|---|---|---|
| 普通 host collective 主线 | API、Group、communicator、bootstrap、拓扑、图搜索、transport、planner、device kernel、proxy、内存与回收（01–10、初始化精读） | 一次 AllReduce 的参数怎样变成 GPU 和网络上的工作？每层交接什么？ |
| 算法与协议 | Ring/Tree、Simple/LL/LL128；NVLS、CollNet、PAT 等扩展入口（07–09、专题附录） | 数据为什么这样分块、规约、发布和复用？ |
| 扩展软件栈 | CUDA Graph、对称窗口、Device API、GIN、Host RMA、MNNVL（现代特性附录） | 新特性改变了哪一层，增加什么能力和生命周期约束？ |
| 开发与验证 | 构建、源码插桩、CPU 模拟、nccl-tests、性能和调试（00、11、12、示例与性能附录） | 改动落在哪层，怎样证明加载正确、结果正确、问题属于哪个阶段？ |
| 教程边界 | NVIDIA KMD/固件/NIC 驱动内部、训练框架专属实现、每个 NCCL 特殊分支的逐行讲解 | 这些内容需要另查对应项目或继续沿本教程的源码入口深入 |

**“看懂”最低限度应当包含一次动手闭环**：跑通 CPU Ring 模拟器，完成一个 GPU 正确性基线，做一次不改变行为的源码插桩，并按第 12 章的方法解释一条等待或异常。没有这些实践，只能说是获得了阅读地图，还不能算掌握了整个软件栈。

## 按目标选择阅读路线

- **只使用 NCCL：** `00 → 01 → 02 → 示例 → 集合接口附录 → nccl-tests 附录`，先掌握调用约定、参与者映射和完成判断。
- **读懂普通 NCCL 实现：** 按 `01 → 12` 顺读，再用[初始化精读](appendix-nccl-comm-init-rank.md)补齐普通新建 communicator 的完整初始化主线。
- **做 transport/device 开发：** 重点读 `04 → 05 → 07 → 09 → 10 → 11 → 12`，再按需求查 NVLS、CollNet、MNNVL 和现代特性附录。
- **研究新硬件或新接口：** 先读 `01、03、04、05、10`，再读现代特性、NVLS、CollNet、MNNVL 和 NVML 附录，明确新能力接入的是拓扑、资源、执行还是观测层。

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
- [ncclCommInitRank 源码精读](appendix-nccl-comm-init-rank.md)：覆盖普通新建 communicator 从 API、Group 调度、bootstrap、拓扑与图搜索，到全局图拼接、transport 连接、`devCommSetup` 和收尾的完整主线；特殊分支与深层辅助函数另列进阶阅读。
- [nccl-tests](appendix-nccl-tests.md)：参数、计时与输出列。
- [性能分析](appendix-performance.md)：延迟/带宽模型、对照实验、计算通信重叠。
- [现代特性](appendix-modern-features.md)：Graph、NVLS、Device API、GIN、RMA 等扩展入口。
- [NVML：GPU 管理与观测](appendix-nvml.md)：设备标识、状态查询，以及 NCCL 如何获取拓扑与 fabric 信息。
- [CollNet：集合网络通信](appendix-collnet.md)：网络插件的职责，以及 Direct、Chain 两种节点内数据流。
- [MNNVL：跨节点 NVLink](appendix-mnnvl.md)：fabric 分组、远端显存映射与跨 host P2P。
- [NVLS：NVLink 硬件归约](appendix-nvls.md)：multicast/multimem、AllReduce 数据流及与 MNNVL、CollNet 的分工。

每章保留少量关键源码入口，以及一项能核对结果的阅读或实验任务。先读懂正文中的交接关系，再去源码找对应对象；完整 API 清单与特殊路径留在附录。

阅读时建议始终同时记录三项：**当前阶段的输入、产生的对象、下一个消费者**。这比单独背函数名更能帮助你在版本变化后重新定位代码。

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
