# CollNet：通过集合网络完成跨节点通信

[返回教程主页](../README.md) · [相关：NVLS](nvls.md) · [相关：MNNVL](mnnvl.md)

**CollNet 把节点内的数据组织与节点间的集合通信分开：GPU 负责本地归约和分发，网络插件负责节点间的集合操作。** CollNetDirect 和 CollNetChain 是这套机制的两种算法，主要区别在于节点内 GPU 怎样交换数据。

本文以两台机器、每台四张 GPU 的 Sum AllReduce 为例。源码基线为 NCCL 2.32.3（`12df1a11`）；示意图展示的是数据依赖，实际执行会按 channel 和 chunk 流水化。

## 1. CollNet 与普通网络传输的区别

普通 NET 接口提供 `isend`、`irecv` 等点对点操作，NCCL 用这些传输能力组织 Ring、Tree 等算法。CollNet 插件则直接提供 `iallreduce` 这样的集合操作：NCCL 交给插件一组参与者、输入输出缓冲、数据类型和归约运算，由插件完成这一组参与者之间的 AllReduce。

一个跨节点 AllReduce 因而可以拆成：

```text
各节点内：汇聚本地 GPU 的贡献，形成局部结果
                         ↓
节点之间：CollNet 插件对局部结果执行 AllReduce
                         ↓
各节点内：将全局结果分发给所有 GPU
```

典型后端是 **InfiniBand SHARP**，可以利用网络交换设备的归约能力。CollNet 定义的是 NCCL 与集合网络之间的接口和组织方式，SHARP 是一种具体实现技术。只有普通 RDMA 收发能力的网卡，还不足以提供 CollNet 所需的集合操作。

在本基线中，CPU proxy 调用插件的 `iallreduce` 并查询完成状态；GPU kernel 负责准备数据和接收结果。因此，网络侧加速与 GPU、CPU 上的通信工作是协作关系。

## 2. 先认识 head、channel 和 chunk

- **head**：承担网络衔接工作的 GPU rank，负责把某部分本地结果交给网络，并接回全局结果。
- **channel**：NCCL 组织并行通信的一条逻辑通道，包含相应的邻接关系和连接资源。
- **chunk**：在通道中逐块推进的数据片段，用于让归约、网络传输和结果分发相互重叠。

head 是 GPU 在算法中的角色，不是额外的一张卡。不同 channel 可以使用不同的 head；Direct 还会利用多个 head 并行处理不同分片。

## 3. CollNetChain：沿链归约，再沿链分发

对某个 channel，假设每台机器的 GPU 0 是 head，本地链的归约方向为：

```text
节点 A：A3 → A2 → A1 → A0（head）──┐
                                  ├── CollNet 网络 AllReduce
节点 B：B3 → B2 → B1 → B0（head）──┘
```

以节点 A 的一个分片为例，四张 GPU 的输入分别记为 `x0`、`x1`、`x2`、`x3`：

| 位置 | 交给上游的数据 |
|---|---|
| A3 | `x3` |
| A2 | `x2 + x3` |
| A1 | `x1 + x2 + x3` |
| A0 | `x0 + x1 + x2 + x3` |

网络插件合并两个节点的局部结果，随后沿相反方向分发：

```text
网络结果 → A0 → A1 → A2 → A3
网络结果 → B0 → B1 → B2 → B3
```

中间 GPU 在归约方向上执行“接收、累加、转发”，在返回方向上执行“接收、保存、转发”。设备实现为两个方向分配不同的线程组，使不同 chunk 可以同时处在归约和分发阶段。

Chain 的优点是每张 GPU 主要与链上相邻 GPU 通信；代价是一个分片需要经过多级转发。这里画的是一条 channel 的链，并不意味着整个节点只能使用一个网络出口。

## 4. CollNetDirect：按分片直接汇聚到多个 head

Direct 将本地 GPU 与负责相应分片的 head 建立直接的逻辑连接，避免沿 GPU 链逐级转发。

假设每台机器有四个 head，将数组划分为四份：

| Head | 负责的本地归约 |
|---|---|
| GPU 0 | 所有本地 GPU 的第 0 个分片 |
| GPU 1 | 所有本地 GPU 的第 1 个分片 |
| GPU 2 | 所有本地 GPU 的第 2 个分片 |
| GPU 3 | 所有本地 GPU 的第 3 个分片 |

对第 0 个分片，节点内的关系是：

```text
GPU 1 ──┐
GPU 2 ──┼──→ GPU 0（head）──→ 网络集合操作
GPU 3 ──┘         ↑
              GPU 0 的贡献
```

GPU 0 归约这一分片，并通过网络插件与其他节点合并结果；GPU 1、2、3 同时负责各自的分片。收到网络结果后，每个 head 将负责的分片分发给本地 GPU，所有 GPU 最终收齐完整结果。

因此，整体可以理解为：

```text
节点内 Reduce-Scatter 到各 head
                 ↓
各分片执行节点间 AllReduce
                 ↓
节点内 AllGather，收齐所有结果分片
```

这是数据流的等价分解，不是额外调用三次公开 NCCL API。设备实现具体分成 Scatter、Reduce、Broadcast、Gather 几类线程工作，并通过 chunk 流水线衔接。

Direct 的“直接”指 GPU 与 head 之间的逻辑通信关系，物理路径仍可以经过 NVSwitch；它与 GPUDirect RDMA 描述的也不是同一层能力。

### 源码里的几个字段

`ncclDirect` 保存每张 GPU 在这种连接关系中的角色：

| 字段 | 含义 |
|---|---|
| `nHeads` | head 数量，决定一轮并行处理多少组分片 |
| `headRank` | 当前 GPU 在 head 列表中的索引；非 head 为 `-1` |
| `up[]` | 当前 GPU 要连接的其他 head |
| `down[]` | 当前 GPU 作为 head 时需要汇聚、分发的本地 peer |
| `out` | head 对接的集合网络逻辑端点；代码用 `comm->nRanks` 表示它 |
| `shift` | 错开各 GPU 访问 head 的顺序，减少同时集中到同一 head 的竞争 |

`headRank` 不是全局 rank 编号，`out` 也不是新增的应用 rank。先分清这些角色，再读 chunk 偏移计算会更容易。

## 5. 两种算法怎样取舍？

| 维度 | CollNetChain | CollNetDirect |
|---|---|---|
| 节点内连接 | GPU 链 | GPU 与多个 head 直接连接 |
| 分片归约 | 逐级累加 | 在负责该分片的 head 汇聚 |
| 主要并行方式 | 多 channel、chunk 流水线 | 多 head、channel、chunk 并行 |
| 主要代价 | 多级转发和链式依赖 | 多 peer 汇聚、分发带来的资源竞争 |
| 节点间操作 | CollNet 插件 | CollNet 插件 |

Direct 更容易利用高带宽节点内互连与多个网络出口；Chain 的邻接关系较简单。实际优劣取决于拓扑、消息大小和可用资源，由 NCCL 的候选筛选与成本模型共同决定。

在本基线的设备算法候选中，AllReduce 有 Direct 和 Chain 两种；AllGather、ReduceScatter 的 CollNet 候选为 Direct，并且需要插件提供相应回调。不能仅凭插件存在 `iallgather`，就推断 Chain 也实现了 AllGather。

## 6. 使用与观察

跨节点使用 CollNet，需要加载可用的集合网络插件，并满足所选 collective、数据类型、归约运算和拓扑的要求。`NCCL_COLLNET_ENABLE=1` 用于启用相关能力，算法是否可选还要经过后续检查。

本基线还有本地 GPU 数、head 连接关系等约束，例如 CollNet 调优路径会排除超过八个 local rank 的情况。这是当前实现的限制，不是 CollNet 插件接口对所有版本的统一规定。

核对日志时，分别看三个阶段：

1. 插件是否加载成功，是否报告集合通信能力。
2. 是否建立了 CollNetDirect 或 CollNetChain 连接。
3. 具体任务的调优或 profiler 记录是否选择了该算法。

前两项证明资源已准备，不代表每次 AllReduce 都使用 CollNet。本文未在带集合网络的 GPU 集群上实测。

## 7. 源码入口

| 阅读目标 | 入口 |
|---|---|
| 集合网络插件契约 | [src/include/plugin/net/net_v12.h:147](../../nccl/src/include/plugin/net/net_v12.h#L147)，`ncclCollNet_v12_t` |
| Direct 的连接关系 | [src/graph/connect.cc:178](../../nccl/src/graph/connect.cc#L178)，`connectCollNet` |
| Direct 的设备数据流 | [src/device/all_reduce.h:248](../../nccl/src/device/all_reduce.h#L248) |
| Chain 的设备数据流 | [src/device/all_reduce.h:638](../../nccl/src/device/all_reduce.h#L638) |
| CPU proxy 调用网络 AllReduce | [src/transport/coll_net.cc:815](../../nccl/src/transport/coll_net.cc#L815)，`collNetIallreduce` |
| 可用 collective 与调优约束 | [src/device/generate.py:104](../../nccl/src/device/generate.py#L104)、[src/tuning/collnet.cc:11](../../nccl/src/tuning/collnet.cc#L11) |

继续阅读：[NVLS](nvls.md) 解释 NVLink 域内的硬件归约；[MNNVL](mnnvl.md) 解释这个域如何跨越物理节点。
