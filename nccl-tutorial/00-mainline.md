# 初学者主线：用一次 AllReduce 看懂 NCCL 软件栈

这份导读适合第一次读 NCCL 源码的读者。目标不是记住所有结构体，而是先建立一张能继续查代码的地图：**一次通信请求经过哪些阶段，每个阶段接收什么、留下什么、由谁继续执行。**

先读本文，再按需要进入各章。细节字段、特殊硬件和异常分支暂时不用展开。

## 1. 先缩小问题：两个 rank、四个 float

假设两张 GPU 属于同一个 communicator，每张 GPU 有 4 个 FP32：

```text
r0：[1, 1, 1, 1]
r1：[2, 2, 2, 2]

Sum AllReduce 后，两边都得到：[3, 3, 3, 3]
```

一个 rank 的调用可以写成：

```cpp
ncclAllReduce(grad, grad, 4, ncclFloat, ncclSum, comm, stream);
```

其中：

- `grad` 是 GPU 上的输入和输出地址；这里是原地操作。
- `count=4` 表示 4 个元素，不是 4 字节。
- `comm` 决定与哪些 rank 通信以及复用哪些资源。
- `stream` 决定本 GPU 上通信与前后 CUDA 工作的顺序。

**这个调用不是让 CPU 把四个数逐个加起来。**NCCL 要把请求转换成 GPU 可以执行的工作，必要时还要让 CPU proxy 和 NIC 配合。

## 2. 软件栈里有哪些执行者？

```text
应用 / 训练框架
  │  buffer、count、op、comm、stream
  ▼
NCCL Host（CPU 侧）
  ├─ 初始化成员、拓扑、图和连接
  ├─ 每次调用时组织 task、work、plan 和 launch
  │
  ├───────────────┐
  ▼               ▼
CUDA / GPU       NCCL CPU proxy
  │               │
NCCL Device      NET 插件与网络后端
  │               │
  └──── 协议状态、数据分片与完成进度 ────┘
```

| 执行者 | 主要职责 |
|---|---|
| NCCL Host | 初始化通信关系；把每次 API 请求编成可执行工作 |
| CUDA Runtime / Driver | 提供 stream、event、内存映射和 kernel launch 等能力 |
| NCCL Device | 在 GPU 上执行分块收发、规约和协议同步 |
| CPU proxy | 为需要的连接处理资源服务并推进网络操作 |
| NET 插件与 NIC | 提供网络访问、请求提交和完成状态 |

本机 GPU 之间的工作常由 kernel 经 NVLink 或 PCIe 推进；跨机 NET 路径通常还需要 proxy 和 NIC。**Host、Device 和 proxy 是协作关系，不是三个串行函数调用。**

## 3. 初始化与每次调用是两条时间线

### 初始化：建立可复用的通信关系

 communicator 初始化主要完成：

```text
收集成员身份与能力
  → 建立 bootstrap 控制通信
  → 发现 GPU、CPU、NIC 拓扑
  → 搜索 Ring/Tree 等算法布局
  → 拼接全局 Ring/Tree
  → 准备 channel、连接、proxy 和 device 端 communicator
```

可以用一张表区分几个产物：

| 产物 | 回答的问题 |
|---|---|
| `peerInfo[]` | 有哪些成员，它们使用什么设备、支持什么能力？ |
| `ncclTopoSystem` | GPU、CPU、NIC 怎样连接，有哪些路径？ |
| `ncclTopoGraph` | 某种算法在局部怎样排列 GPU 和网络出口？ |
| `channels[c].ring/tree` | 当前 rank 在每个 channel 中与谁通信？ |
| `comm->devComm` | GPU kernel 后续使用什么 communicator 视图？ |

初始化还分两层：**逻辑邻接关系**和**实际传输资源**。`ring.next=5` 只说明下一跳是谁，P2P/SHM/NET 的映射、缓冲和连接还需要 transport 建立；部分连接可以延迟到算法首次使用。

源码入口可先看 [`initTransportsRank`](../nccl/src/init.cc#L1163)，完整精读放在[初始化附录](appendix/nccl-comm-init-rank.md)，初学阶段不必逐行展开。

### 每次 collective：填入当前工作

初始化完成后，每次 AllReduce 只复用这些关系并填入本次数据：

```text
API 参数
  → 集合任务
  → 选择算法与协议
  → 生成 work 和 plan
  → launch
  → Device 执行；需要时 proxy 推进
```

因此，`comm` 很大而一次 API 包装很短：大量可复用工作已经在初始化完成。

## 4. 先分清 algorithm、protocol、transport 和 channel

| 名称 | 回答的问题 | 例子 |
|---|---|---|
| algorithm | 各 rank 按什么依赖关系交换数据？ | Ring、Tree |
| protocol | 一片数据怎样组织、发布和回收？ | Simple、LL、LL128 |
| transport | 一条 peer 连接通过什么资源实现？ | P2P、SHM、NET |
| channel | 怎样把通信工作拆成可并行的逻辑分区？ | 一个任务使用多个 channel |

一次 Ring/Simple AllReduce 中，本机边可以用 P2P，跨机边可以用 NET。**算法不决定具体网卡，transport 也不改变 AllReduce 的数学语义。**

`plan` 与 channel 也不同：

- **channel**：数据如何并行分工，以及该分区使用哪些邻居和连接。
- **plan**：一次 kernel launch 需要哪些参数、work batches、channelMask 和 proxy 工作。

## 5. 一次 AllReduce 的主线流程

### 第一步：API 请求 → `ncclInfo`

公开入口把 `sendbuff、recvbuff、count、datatype、op、comm、stream` 整理成 `ncclInfo`。

它只回答“用户想做什么”。此时没有选择算法、分配 channel、复制用户数组或启动 kernel。

源码：[`ncclAllReduceConfigImpl`](../nccl/src/collectives.cc#L192)。

### 第二步：请求 → `ncclTaskColl`

`ncclEnqueueCheck` 检查 communicator 和参数，`taskAppend` 再按任务类型分流。普通 collective 进入 `collTaskAppend`，生成可等待调度的 `ncclTaskColl`。

任务保存的是**地址和执行信息，不是 payload 副本**。`ncclInfo` 可以退出，用户 buffer 则必须覆盖通信执行期。

源码：[`ncclEnqueueCheck`](../nccl/src/enqueue/enqueue.cc#L3478)、[`collTaskAppend`](../nccl/src/enqueue/enqueue.cc#L2757)。

### 第三步：GroupEnd → 调优与必要连接

最外层 GroupEnd 开始统一处理任务。`ncclPrepareTasks` 根据任务、拓扑和候选能力，为原始任务补上：

- `algorithm` 和 `protocol`；
- `nMaxChannels` 和 `nWarps`；
- 后续 device 分派使用的 `devFuncId`。

`nMaxChannels` 只是上限，实际 channel 数由后续调度决定。若所选算法采用按需连接，还要先完成 preconnect。

源码：[`ncclPrepareTasks`](../nccl/src/enqueue/enqueue.cc#L423)、[`ncclGetAlgoInfo`](../nccl/src/enqueue/enqueue.cc#L2159)。

### 第四步：任务 → work + plan

按路径需要准备 buffer 访问关系后，NCCL 生成 `ncclDevWorkColl`，其中保存 buffer 地址、规约参数、数据分工和 chunk 粒度。

`ncclLaunchPrepare` 再创建 `ncclKernelPlan`，由 `scheduleCollTasksToPlan` 决定实际 channel 数、每个 channel 的数据范围，并设置 `channelMask`。

可以先记住三种信息的分工：

| 对象 | 主要内容 |
|---|---|
| `ncclTaskColl` | host 侧的集合任务和调优结果 |
| `ncclDevWorkColl` | GPU 要访问的数据和执行参数 |
| `ncclKernelPlan` | 一次 launch 的完整材料 |

源码：[`ncclTasksRegAndEnqueue`](../nccl/src/enqueue/enqueue.cc#L351)、[`scheduleCollTasksToPlan`](../nccl/src/enqueue/enqueue.cc#L644)。

### 第五步：plan → kernel launch

`finishPlan` 准备 `kernelArgs` 和 work batches；`uploadWork` 写入执行描述并修正偏移。**上传的是描述，不是用户数组。**

`ncclLaunchKernel` 根据 `channelMask` 的置位数构造 grid，根据 `threadPerBlock` 构造 block，再通过 CUDA Driver 启动 kernel。

例如：

```text
channelMask=0b0110
  → 活动 channel 为 1、2
  → grid.x=2
  → block 0 对应 channel 1，block 1 对应 channel 2
```

在这条通用 kernel 路径中，一个活动 channel 对应一个 CTA。

源码：[`finishPlan`](../nccl/src/enqueue/enqueue.cc#L236)、[`uploadWork`](../nccl/src/enqueue/enqueue.cc#L1365)、[`ncclLaunchKernel`](../nccl/src/enqueue/enqueue.cc#L1886)。

### 第六步：Device 读取 work 并执行算法

`ncclKernelMain` 先根据 `blockIdx.x` 和 `channelMask` 找出 channel ID，再装载三类信息：

```text
ncclKernelComm：整个通信组和执行环境
ncclDevChannel：当前 channel 的邻居和连接
ncclDevWorkColl：本次数据分工与执行参数
```

随后按 `devFuncId` 分派到具体实现。Ring AllReduce 会进入 `runRing`：

- `ncclCollCbdPart` 恢复当前 channel 的数据起点、数量和 chunk；
- `Primitives` 接入 `ring.prev/next`、用户 buffer 和规约参数；
- ReduceScatter 先让每个 rank 得到一个完整规约块，AllGather 再传播这些块。

两个阶段在同一个 device 工作体内衔接，不需要中间返回 host API。

源码：[`ncclKernelMain`](../nccl/src/device/common.h#L419)、[`runRing`](../nccl/src/device/all_reduce.h#L14)。

### 第七步：需要时由 proxy 推进网络

如果连接需要 CPU progress，plan 还会带 proxy 工作。`hostStreamPlanTask` 将 proxy 操作交给共享工作池并调用 `ncclProxyStart`，proxy 随后推进网络请求和完成状态。

GPU 与 proxy 通过协议状态协作：**发送 GPU 发布数据就绪，send proxy 提交网络请求；接收 proxy 发布数据可读，接收 GPU 消费后归还槽位。**

普通 P2P/SHM 路径不必都经过 NET proxy。

## 6. 最后分清三个完成边界

```text
API/launch 已提交
    ≠
GPU 已完成 AllReduce
    ≠
所有内部资源已经可以回收
```

- **提交完成**：host 已把任务交给 CUDA/NCCL 执行框架。
- **数据完成**：相应 stream/event 表明 GPU 工作结束，输出才可消费。
- **资源回收**：plan、work FIFO、proxy args、注册和映射分别按最后使用者结束回收。

`ncclCommGetAsyncError` 的输出状态与 CUDA stream/event 回答的是不同问题，不能互相替代。

## 7. 接下来怎样读而不被细节淹没？

建议按下面顺序：

```text
01 API 与执行约定
  → 02 Host 执行主线
  → 03 Device 执行
  → 04 集合算法
  → 05 Communicator 初始化
  → 06 拓扑与通信图
  → 07 Transport
  → 08 Proxy 与网络
  → 09 内存与生命周期
  → 10 开发工作流
  → 11 测试与调试
```

[软件栈全景](appendix/software-stack.md) 可作为目录和名词地图随时回查。首次阅读先不展开：

- `ncclTopoCompute` 的递归回溯细节；
- channel 调度的 cell/流量公式；
- work FIFO 的容量计算；
- CUDA Graph、enqueue rearch 和特殊硬件分支；
- NVLS、CollNet、MNNVL、GIN、RMA 等扩展机制。

读每个函数时只记录四项：**输入、动作、产物、下一消费者**。不要一开始就试图读懂所有字段和所有分支。

## 8. 自测：能否讲清这条链？

不看文档，尝试回答：

1. `ncclInfo`、`ncclTaskColl`、`ncclDevWorkColl` 和 `ncclKernelPlan` 分别给谁使用？
2. algorithm、protocol、transport 和 channel 为什么不是同一层概念？
3. 为什么 `nMaxChannels` 不等于最终 channel 数？
4. `uploadWork` 上传的是 payload 吗？
5. `channelMask` 怎样影响 grid？`devFuncId` 又负责什么？
6. 为什么 launch 成功不能直接推出输出已经可读？

如果能用自己的话把这些问题串成一次 AllReduce，就已经具备继续读各章的主线框架。
