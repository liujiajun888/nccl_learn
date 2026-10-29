# 附录：软件栈全景——谁把一次 AllReduce 做完

NCCL 的核心工作，是把一次多 GPU 通信请求，转成各 GPU 可以协同执行的数据流。应用描述“需要什么结果”；NCCL 决定怎样分块、经过哪些连接、由哪些执行者推进。

从开发角度看，NCCL 同时包含 **CPU 上的通信运行时**和 **GPU 上的通信程序**。只看 API 包装层，会漏掉大部分工作；只看 CUDA kernel，也看不到它使用的连接和调度信息从哪里来。

## 1. 先固定一份工作

设两台机器 A、B，各有四张 GPU。每个进程管理一张卡，共同执行一次 Sum AllReduce：每个元素把八张卡的对应位置相加，结果留在每张卡上。

```text
主机 A：r0  r1  r2  r3       主机 B：r4  r5  r6  r7
         各持有一份 64 MiB FP32 输入
                    ↓ Sum AllReduce
         各得到一份 64 MiB FP32 求和结果
```

**rank** 是通信组内的成员编号。**communicator** 是这组成员的通信上下文，保存成员关系、连接、调度和运行状态。每个 rank 都持有自己的本地 `ncclComm_t` 句柄。

缩成两张卡、每卡四个元素，语义就是：

```text
r0：[ 1, 2, 3, 4] ─┐              r0：[11,22,33,44]
                   ├─ 对应位置相加 ─┤
r1：[10,20,30,40] ─┘              r1：[11,22,33,44]
```

训练框架常用它合并梯度。梯度何时生成、是否求平均、何时更新参数由框架安排；NCCL 接到的是缓冲地址、元素类型、数量和规约操作。

## 2. 软件栈有两条执行支路

```text
应用 / 训练框架
  │  ncclAllReduce(buffer, count, type, op, comm, stream)
  ▼
NCCL Host：初始化、拓扑、连接、任务规划
  │                                  │
  │ GPU 工作描述、kernel launch       │ 网络操作描述
  ▼                                  ▼
CUDA Runtime / Driver API          NCCL CPU Proxy
  │                                  │
GPU 驱动的提交与资源管理             NET 插件 / 网络后端
  │                                  │
  ▼                                  ▼
NCCL Device Kernel                 网络驱动与 NIC
  │                                  │
GPU 显存 ←────── NVLink / PCIe / 网络传输 ──────→ 远端显存
```

**Host** 指 CPU 侧；**Device** 指 GPU 侧。**Proxy** 是 NCCL 的 CPU 服务与推进机制，为连接准备以及部分传输路径处理异步工作。**NET 插件**把 NCCL 的网络请求接到具体网络实现。

本机 GPU 通信常由 kernel 经 NVLink 或 PCIe 访问数据。跨机时，常规 NET 路径由 proxy 发起网络请求，NIC 搬运数据，GPU kernel 消费收到的分片并继续规约。

这两条支路是并行协作关系。CPU 可以提交下一批网络请求，NIC 传输当前分片，GPU 同时规约上一片。后续章节会把它们的交接拆开来看。

### 与驱动经验怎样对应

| 层 | 负责的事情 | 主要交接物 |
|---|---|---|
| 应用 | 产生数据，选择通信语义，安排依赖 | 用户 buffer、comm、stream |
| NCCL Host | 建立通信关系，选择执行方案，组织提交 | task、plan、设备工作描述、proxy 操作 |
| CUDA Runtime / UMD | 处理 CUDA 资源与执行接口，把 kernel 纳入 stream | launch 参数、地址映射、执行依赖 |
| KMD | 提供设备资源、内存映射、隔离与提交相关支持 | 设备可用的资源与访问关系 |
| NCCL Device | 执行分块收发、规约与协议同步 | 数据分片、就绪进度、可复用空间 |
| 网络后端与 NIC | 注册内存、建立网络连接、传输并报告请求进度 | 内存注册句柄、请求、完成状态 |

GPU/NIC 的固件和硬件在这些资源与命令之下完成调度、访存和传输。具体驱动及固件实现属于对应软件栈；这份 NCCL 源码提供的是它们上方的调用和协作逻辑。

对 UMD 开发者，最有用的迁移方法是继续追踪“资源 → 工作 → 完成”，并在每一步多问一句：**对端由谁推进？**

## 3. 把准备阶段和执行阶段分开

通信组建立时，NCCL 先完成三件事：

1. **认识成员。** 交换 rank、设备和联系信息。
2. **认识机器。** 发现 GPU、CPU、NIC 的连接关系。
3. **准备通信资源。** 为需要的邻居建立连接、缓冲和映射；部分连接可以延迟到首次使用。

这些信息跨多轮调用复用。随后每次 AllReduce 都走一条较短的执行链：

```text
API 参数
   ↓
Host 任务：这次要通信多少数据，使用什么操作
   ↓
执行计划：选算法和协议，分配并行工作，准备发射
   ├── GPU 工作描述 → kernel 收发、规约 → stream 中的通信完成
   │                        ↕ 数据就绪、空间回收
   └── Proxy 操作  → 网络请求与 NIC 传输
```

设备工作描述保存地址、长度和执行参数；64 MiB 数组本身仍留在用户显存。NCCL 规划的是如何访问这些数据。

读源码时先抓住两种时间尺度：**初始化建立可复用关系，每次提交填入当前工作。** 这能解释为什么 `ncclComm` 很大，而单次 API 的包装很短。

## 4. 四个词描述一次执行方案

| 名称 | 回答的问题 | 一个例子 |
|---|---|---|
| collective | 最终要得到什么结果？ | AllReduce：规约后每卡一份 |
| algorithm | 各 rank 按什么次序交换哪些块？ | Ring：沿环逐步规约，再传播结果 |
| protocol | 一段连接如何分片、发布数据和回收空间？ | Simple：用进度计数协调生产与消费 |
| transport | 相邻参与者通过什么资源访问彼此？ | P2P、SHM、NET |

另一个常见词是 **channel**：NCCL 划分并行通信工作的逻辑通道。一个 channel 有自己的算法邻接和连接状态，负责数据中的一部分。

例如，同一次 Ring AllReduce 可以划成多个 channel；每个 channel 沿自己的 ring 运行。环上本机的一跳使用 P2P，跨机的一跳使用 NET。算法规定整体次序，transport 落实每一跳，protocol 让每一跳持续前进。

后面看到几十种配置时，用这四个问题分类，就能知道某个参数改的是哪层。

## 5. 读代码先抓这几个对象

第一次阅读只需记住三组对象，后文再展开字段：

- **通信关系：** `ncclComm`、channel、connector。回答“谁和谁通信、资源在哪里”。
- **本次工作：** `ncclInfo`、`ncclTaskColl`、`ncclKernelPlan`、`ncclDevWorkColl`。回答“应用请求怎样变成可执行工作”。
- **推进状态：** 协议进度、proxy 请求、CUDA event。回答“做到哪里、何时可以继续”。

它们的连接比字段数量重要。例如 `ncclDevWorkColl` 的输入地址来自应用，channel 的邻居来自初始化，协议中的进度由 GPU 或 proxy 更新。一个 kernel 同时使用这三类信息，才能完成通信。

### 源码目录地图：知道该去哪一层找证据

教程中的“层”最终都落在具体目录里。先看目录职责，再看函数名，可以避免把相邻但不同的机制混在一起。

| 目录或文件 | 主要职责 | 对应阅读 |
|---|---|---|
| `src/nccl.h.in`、`src/collectives.cc` | 公开 API、参数与 collective 入口 | 01、集合接口附录 |
| `src/init.cc`、`src/bootstrap.cc` | communicator 创建、成员会合、初始化和收尾 | 05、初始化精读 |
| `src/graph/` | XML/硬件拓扑、路径、图搜索、Ring/Tree 拼接 | 06 |
| `src/transport.cc`、`src/transport/` | transport 选择、连接协调、P2P/SHM/NET 及专用传输 | 07、08 |
| `src/group.cc`、`src/enqueue/` | Group 调度、task、plan、work 上传、launch 和回收 | 02、09、10 |
| `src/device/`、`src/include/device.h` | GPU 算法、协议原语、device 工作描述和函数分派 | 03、04 |
| `src/proxy.cc` | proxy service、op pool、网络工作推进 | 08 |
| `src/plugin/`、`src/include/plugin/` | NET、Tuner、Profiler、GIN、RMA 等插件契约与加载 | 07、08、现代特性附录 |
| `src/tuning/` | 算法/协议候选筛选、成本模型和资源阈值 | 06、02、04、性能附录 |
| `src/allocator.cc`、`src/register/`、`src/mem_manager.cc` | 分配、用户注册、映射和内部内存管理 | 09、现代特性附录 |
| `src/dev_runtime*.cc`、`src/devcomm/`、`src/gin/`、`src/rma/`、`src/ce_coll.cc` | Device API、对称窗口、设备发起网络、RMA 和 copy-engine 路径 | 现代特性附录 |
| `src/os/`、`src/misc/`、`src/include/` | OS 抽象、公共内部数据结构、同步与工具函数 | 各章按需查阅 |
| `src/device/generate.py`、`src/device/Makefile`、`makefiles/` | 设备组合生成、device link 和主库构建 | 环境附录、10 |

`src/include/` 主要放 NCCL 内部契约，不能整体当作应用公开 API；应用公开声明主要来自 `src/nccl.h.in` 和 `src/include/nccl_device/` 等公开头文件。设备生成源码位于构建目录，不应把生成物当成唯一的手写实现来修改。

一条实用的目录追踪顺序是：

```text
API/collectives
  → init/bootstrap
  → graph/transport
  → group/enqueue
  → device/proxy
  → tuning/register/dev_runtime（按问题深入）
```

### 源码起点

- [nccl/src/collectives.cc:192](../../nccl/src/collectives.cc#L192)：`ncclAllReduceConfigImpl` 把 API 参数放入 `ncclInfo`。
- [nccl/src/include/comm.h:212](../../nccl/src/include/comm.h#L212)：`ncclTaskColl`，主机任务表示。
- [nccl/src/include/comm.h:357](../../nccl/src/include/comm.h#L357)：`ncclKernelPlan`，一次发射的材料。
- [nccl/src/device/all_reduce.h:14](../../nccl/src/device/all_reduce.h#L14)：`runRing`，GPU 侧把收发与规约连成算法。

### 阅读任务：把参数和数据分开

打开 `ncclAllReduceConfigImpl`，找到 `sendbuff`、`recvbuff` 和 `count` 的去向，再打开 `runRing`，找到第一次 `directSend`。

核对点：前者保存指针和参数；后者按工作描述计算数据偏移并调用传输原语。两处之间缺少的任务构造、连接和 launch，正是后面要补齐的软件栈。

下一章：[01 API 与执行约定](../01-api-and-execution.md)。
