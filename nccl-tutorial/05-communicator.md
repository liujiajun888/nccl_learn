# 05｜Communicator 初始化：通信资源从哪里来

前面几章都在使用 `comm`：它提供成员关系、拓扑、channel 和连接。现在回头看，**这些资源是怎样在一次初始化中建立起来的？**

本章只讲普通新建 communicator 的主线。完整 API 配置和逐段源码见 [Communicator 进阶附录](appendix/deep-dive/communicator.md) 与 [初始化精读](appendix/initialization/nccl-comm-init-rank.md)。

## 1. 应用先提供三项输入

以最常用的 `ncclCommInitRank` 为例，应用需要准备：

| 输入 | 作用 |
|---|---|
| `nranks` | 通信组共有多少成员 |
| `rank` | 当前成员是第几号 |
| `uniqueId` | 全组共同的会合令牌 |

同时，应用要先调用 `cudaSetDevice`，决定当前 rank 使用哪张 GPU。

```text
应用决定 rank、GPU 和 uniqueId
  → NCCL 负责成员会合、拓扑、图、连接和 device 状态
```

单进程多卡常用 `ncclCommInitAll`；需要配置时使用 `ncclCommInitRankConfig`；大规模引导另有 Scalable 入口。初学时先理解 InitRank 即可。

## 2. 初始化主线：从本地对象到通信团队

```text
ncclCommInitRankDev：创建本地 comm 和 init job
  ↓
ncclCommInitRankFunc：准备 GPU 与本地资源
  ↓
commAlloc：分配管理结构和后端上下文
  ↓
bootstrapInit：成员会合，建立控制通信
  ↓
initTransportsRank：拓扑、图、channel、连接和 device 状态
```

这里有一个重要边界：

> `ncclComm_t` 已分配，不等于 communicator 已可用于通信。对象骨架、成员关系、拓扑和实际资源是逐步补全的。

## 3. 各阶段留下什么？

| 阶段 | 主要产物 | 下一步为什么需要它 |
|---|---|---|
| 本地对象与任务 | `comm`、init job、配置和终止标志 | 由初始化任务继续执行 |
| `commAlloc` | 内部内存管理、stream/event、插件上下文 | 管理后续资源 |
| bootstrap | 控制环、服务地址表、proxy 初始状态 | 交换身份、图摘要和连接描述 |
| `peerInfo[]` | rank、设备身份、主机/进程关系、共同能力 | 判断谁能用哪条路径 |
| `ncclTopoSystem` | GPU、CPU、NIC 的节点、连接和 paths | 为算法选路提供硬件模型 |
| `ncclTopoGraph` | 局部 Ring/Tree 的 GPU 顺序和网络出口 | 形成每个 channel 的布局 |
| `channels[c]` | 全局 `ring.prev/next`、`tree.up/down` | transport 按邻居建连 |
| `comm->devComm` | device 端 communicator 和 channel 视图 | 后续 kernel 使用 |

前四类是“认识成员和硬件”，中间两类是“规划算法”，后两类是“落实运行资源”。

## 4. 拓扑图与算法图不是一回事

```text
ncclTopoSystem：硬件路网
ncclTopoGraph：某种算法的局部走法
channels[c]：拼接后的全局邻接关系
```

例如，一台机器有 8 张 GPU，但 communicator 只使用其中 4 张，正常自动建图时只把这 4 张作为通信成员，而不是把另外 4 张也加入 Ring。

如果两个使用中的 GPU 不能直连，NCCL 只会在当前拓扑和通信成员允许的范围内寻找路径，不会随意借一张不在 communicator 中的空闲 GPU 做中继。

## 5. 局部图怎样变成全局 Ring/Tree？

各 rank 先在局部域调用 `ncclTopoCompute`，再经过一次全局协调：

```text
ncclTopoPreset
  → 提取本地 Ring/Tree 端点与邻居摘要
  → bootstrapAllGather 交换 graphInfo + topoRanks
  → 对齐 channel 数、带宽和路径约束
  → ncclTopoPostset 拼接跨节点邻接并校验 rings
```

假设两个局部段分别是：

```text
node 0：0 → 1
node 1：2 → 3
```

Postset 补上：

```text
1 → 2，3 → 0
```

得到：

```text
0 → 1 → 2 → 3 → 0
```

`ringRecv/ringSend` 是局部段的入口和出口，**不是当前 rank 自己的 recv/send peer**。当前 rank 的局部前后邻居另由 `ringPrev/ringNext` 表示。

## 6. 逻辑邻接和实际连接也要分开

`ring.next=5` 只说明下一跳是 rank 5，并不意味着已经存在可用的数据传输通道。

还需要：

- 选择 P2P、SHM 或 NET；
- 建立映射、缓冲和必要的注册；
- 交换连接描述并完成握手；
- 准备 proxy 和设备侧连接信息。

所以初始化可以分为：

```text
逻辑关系：channels[c].ring/tree
实际资源：connector、缓冲、MR、映射、proxy 和 device 连接信息
```

## 7. 为什么有些连接会延迟到首次使用？

本版的选择条件是：

```cpp
comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
```

两种策略是：

```text
提前建立：初始化时就建立所需算法连接
按需建立：任务首次选择某算法时再补齐连接
```

`runtimeConn` 只是改变建连时机，不是省略连接。首轮 collective 可能因此多一段准备成本，后续任务继续复用。

## 8. 初始化结束时得到什么？

普通路径完成后，当前 rank 具有：

```text
成员与节点映射
局部硬件拓扑与算法图
全局 Ring/Tree 邻接
channel 与必要的 transport 资源
proxy 和 P2P schedule
调优模型
device 端 communicator、work FIFO 和 profiler 计数器
```

后续每次 AllReduce 只填入当前 buffer 和执行参数，不重新经历整套初始化。

下一步读 [06｜拓扑与通信图](06-topology.md)，重点看 `ncclTopoSystem → paths → ncclTopoGraph → channels` 这条转换链。

初始化 API 的更多选择、任务调度和异常边界见 [Communicator 进阶附录](appendix/deep-dive/communicator.md)。
