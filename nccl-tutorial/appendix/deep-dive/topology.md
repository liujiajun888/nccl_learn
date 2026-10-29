# 06｜Topology：从硬件路径到 channel 邻接

> 阅读定位：进阶原文。先读 [06 Topology 主线](../../06-topology.md)，再按路径类型、PXN、rail 或图搜索问题查阅，不必研究所有搜索分支。

拓扑层把“这些 rank 使用哪些 GPU”转化为“每条通信通道怎样经过 GPU 和网卡”。
它先建立硬件模型，再计算端点间路径，随后搜索算法布局，最后生成各 rank 的邻接关系。
这些结果保存在 communicator 中，供传输层建连和任务调度使用。

源码基线：NCCL **2.32.3**，提交 `12df1a11`；本章的图是硬件或通信关系图。

## 1. ncclTopoSystem：把硬件视图装进对象

**`ncclTopoSystem`** 是 NCCL 的拓扑模型，按类型保存节点集合。
**Node（节点）**表示模型中的 GPU、PCIe bridge、CPU 域、网卡等对象；**link（链路）**表示相邻对象之间的连接，记录类型、带宽估计和对端。
本版还区分承载 rank 的逻辑 `GPU` 节点与承载物理互连的 `DEV` 节点。

**NIC（网络接口卡）**是网卡；模型中的 `NIC` 偏向它的本地硬件连接，`NET` 表示网络插件提供的通信端点，带有设备号、端口和带宽等属性。
**NUMA 域**是 CPU/内存的局部访问域，跨域访问会经过额外的系统互连。
这些层次让 NCCL 同时描述 GPU 间距离和 GPU 到网络端点的距离。

沿用八 rank 案例，为解释选路，给每台主机添加如下教学布局，B 与 A 对称：

```text
              主机 A
 NUMA 域 0 ── CPU 系统互连 ── NUMA 域 1
     │                           │
  PCIe 域 0                   PCIe 域 1
   ├ A0、A1                    ├ A2、A3
   └ NIC0                      └ NIC1

 A0、A1、A2、A3 通过 NVLink/NVSwitch 互连
 NIC0 靠近 A0/A1；NIC1 靠近 A2/A3
```

`ncclTopoGetSystem` 以 XML 作为中间表示，读取可用的拓扑描述，并补充 GPU、PCIe、NVLink 和网络插件属性。
每个 rank 贡献自己管理的 GPU 信息，再通过本地成员间的引导交换融合 XML，最后生成 `comm->topo`。
因此每进程仅暴露一张 GPU 时，本例仍能形成包含本机四个成员的模型。

这里详细建模的是本地设备及其网络出口；跨主机通信还要靠各节点交换搜索摘要来拼接。
网络交换机中的实际路由与运行时拥塞，需要由网络侧观测补充。

## 2. paths：先算距离，再加入访问条件

**Path（路径）**是一对端点之间的一串 links，附带路径类型、瓶颈带宽和跳数。
`ncclTopoComputePaths` 将这些结果缓存到节点的 `paths[]` 中，后续搜索可以直接查询。

底层 `ncclTopoSetPaths` 从端点扩展路径。
比较候选时，先比较路径类型，再比较带宽；两者相同则优先更少的跳数。
带宽沿路径取最窄一段：

```text
GPU ── 50 ── PCIe switch ── 25 ── NIC
路径带宽估计 = min(50, 25) = 25
```

数字只是教学单位。两个 channel 若共用后半段链路，搜索就要计算它们对同一份 25 的占用。
**Channel（通道）**是 NCCL 用于并行通信的一组逻辑资源，并行收益取决于这些资源实际经过哪些链路。

常见路径类型可以帮助读日志和字段：

| 类型 | 表达的路径特征 |
| --- | --- |
| `NVL` | 经过 NVLink |
| `PIX/PXB` | 经过至多一个／多个 PCIe bridge |
| `PHB/SYS` | 经过 PCIe host bridge／跨 NUMA 系统互连 |
| `PXN` | GPU 先经另一张 GPU，再访问其附近的 NIC |

在本例中，A0 到 NIC0 的路径较近，到 NIC1 的直接路径则可能跨 NUMA。
如果 A0 能经 NVLink 到 A2，再由 A2 接近 NIC1，PXN 就提供了另一种选择。
**GDR（GPUDirect RDMA）**是网卡直接访问 GPU 内存的能力；这条中转路径要求邻卡到 NIC 的 GDR、GPU 互连和相关策略都满足条件。

可访问性会进一步修正物理路径。
**P2P** 是 GPU 间直接访问的传输方式，**SHM** 是借主机共享内存通信的方式。
GPU peer 访问受限时，路径可改为经过 CPU；两个成员缺少可用的 P2P/SHM 路径时，本地域模型会据此裁剪，成员间仍可通过网络通信。

初始化于是按“计算 paths → 裁剪 system → 重算 paths”推进。
裁剪改变了节点集合和可行域，重算使接下来的搜索使用更新后的路径。

## 3. ncclTopoGraph：给并行通信安排路线

**`ncclTopoGraph`** 保存一种算法模式的搜索结果：通道数、节点内 GPU 顺序、网络入口/出口，以及带宽和路径类型估计。
其中 `intra[]` 记录本地 rank 顺序，`inter[]` 记录各搜索通道的网络端点标识。

**Ring** 是沿环形邻接传递数据的算法组织方式，**Tree** 是按父子关系传递数据的方式。
初始化分别搜索 Ring 和 Tree；本版 Tree 搜索的通道数约束使用前面 Ring 的搜索结果。
同一个 communicator 因而保存多种可复用的通信布局。

搜索会尝试 GPU/NIC 排列，沿候选路径扣减可用带宽，回溯时再恢复。
它还会调整目标带宽、路径限制以及多个 channel 是否采用相同排列，并受搜索预算约束。
这让它能在初始化成本可控的条件下，寻找能并行利用链路的布局。

“GPU 靠近哪张 NIC”在这里成为可计算的成本。
如果所有通道都让 A0/A1 从 NIC0 出站，NIC0 或其 PCIe 上行可能成为瓶颈；把部分出口交给 A2/A3 可以利用 NIC1。
选择时还要计入把数据送到这些出口 GPU 的成本，所以近端距离和整体并行利用率需要一起考虑。

**Rail** 是跨主机按端口、接线和网络组织形成的一组对应通信路径。
本例可把 A/B 的 NIC0 端口组织为 rail 0，把 NIC1 端口组织为 rail 1。
两条 rail 可以共享上游交换设备或链路，物理独立性由实际布线决定。

一个 rail 可以承载多个 channel；选择出口时也要考虑远端端口怎样配对。
`crossNic` 等搜索约束用于适配这种网络组织，避免为使用另一张网卡而引入代价更高的跨 rail 路径。

## 4. 具体交接：搜索结果成为 r1 的邻居

`ncclTopoPreset` 从各算法图中提取本地 rank 顺序、节点入口与出口。
这些摘要经成员交换后，由 `ncclTopoPostset` 拼接跨节点关系，写入 `comm->channels[c]`。
Ring 使用 `ring.prev/next`，Tree 使用 `tree.up/down[]`。

假设某个 channel 得到以下示意 Ring，且两端网络出口都走 rail 0：

```text
主机 A                         主机 B
r0 → r2 → r3 → r1 ── NIC0 ──→ r5 → r7 → r6 → r4
↑                                               │
└──────────────── NIC0 ─────────────────────────┘
```

这是一条用于追踪的布局。对 r1 而言，`prev=3、next=5`；对 r5 而言，`prev=1、next=7`。
应用 rank 保留原有语义，环中的排列则服务于硬件资源利用。

交给下一层时，`ncclTransportRingConnect` 读取每个 channel 的前驱和后继。
在 r1 上，它登记来自 r3 的接收连接和发往 r5 的发送连接，再交给通用传输准备流程。
后者才能为这两条边分别选定资源实现，例如节点内 P2P、跨节点 NET。

**NET** 是通过网络插件实现通信的 transport；transport 的接口和映射过程在下一章展开。
此处 `graph/connect.cc` 负责拼接逻辑邻接，资源建连由 `transport` 层完成。
按需连接启用时，这个资源阶段会推迟到算法首次使用。

## 5. 初始化图搜索与每次任务选择是两步

初始化的输入是成员、硬件路径和资源约束，输出是多种算法布局及估计信息。
它尚未处理本次 64 MiB 用户数据的工作描述。
这些布局可以服务同一 communicator 的许多轮不同大小的通信。

**Task（任务）**记录一次待执行通信的类型、数据大小、数据类型和规约操作等信息。
普通 host collective 的任务准备阶段再把这些信息交给调优逻辑，选择算法、协议和通道使用上限；实际通道分配由后续计划生成完成。
本版可从 `ncclGetAlgoInfo` 看到 `nBytes、datatype、redOp` 等输入以及 `algorithm、protocol、nMaxChannels` 等输出。

**Protocol（协议）**规定分片数据与进度信息如何组织，Simple 是其中一种。
本书选 Ring/Simple 作为 64 MiB AllReduce 的具体追踪分支；实际任务会结合拓扑能力、调优模型及配置完成选择。
初始化得到 Ring 图，只说明这套布局已经准备好供后续使用。

修改 GPU/NIC 路径模型，主要影响初始化布局及其成本信息；修改消息大小相关的选择策略，则要继续追到任务调优。
这种分工也是后续定位代码改动点的依据，详见 [02｜Host execution](../../02-host-execution.md) 和 [04｜算法](../../04-algorithms.md)。

## 6. 源码入口与无 GPU 阅读任务

- [`src/graph/topo.cc:1989` · `ncclTopoGetSystem`](../../../nccl/src/graph/topo.cc#L1989)：从 XML、设备与插件信息生成系统模型；对象布局在 [`graph/topo.h:172`](../../../nccl/src/graph/topo.h#L172)。
- [`src/graph/paths.cc:754` · `ncclTopoComputePaths`](../../../nccl/src/graph/paths.cc#L754)：建立并修正端点路径；[`ncclTopoSetPaths:52`](../../../nccl/src/graph/paths.cc#L52) 中可核对比较次序。
- [`src/graph/search.cc:1151` · `ncclTopoCompute`](../../../nccl/src/graph/search.cc#L1151)：搜索算法布局；输入输出字段见 [`include/graph.h:174` · `ncclTopoGraph`](../../../nccl/src/include/graph.h#L174)。
- [`src/graph/connect.cc:20` · `ncclTopoPreset`](../../../nccl/src/graph/connect.cc#L20) 与 [`380` · `ncclTopoPostset`](../../../nccl/src/graph/connect.cc#L380)：从本地顺序提取摘要，再写入跨节点邻接。
- [`src/transport/generic.cc:14` · `ncclTransportRingConnect`](../../../nccl/src/transport/generic.cc#L14)：消费 channel 邻居，生成资源连接需求。
- [`src/enqueue/enqueue.cc:2159` · `ncclGetAlgoInfo`](../../../nccl/src/enqueue/enqueue.cc#L2159)：在任务阶段输入消息特征，接收调优结果。

**阅读任务：沿第 4 节的示意 Ring，追踪 r1 的 `next=5`。**

先判断该值在本地图摘要和跨节点拼接之后分别处于什么状态，再找到传输层读它的位置。
最后检查 `ncclTopoGraph` 和 `ncclGetAlgoInfo`，回答本次消息大小在哪一步参与决策。

**答案线索：**r1 是主机 A 的本地出口，`ncclTopoPreset` 在 `connect.cc:51` 将本地末尾的 `ringNext` 记为 `-1`。
跨节点拼接后，`ncclTopoPostset` 在 451～456 行把完整邻接写入 channel，此例中 r1 的 `next` 成为 5。
`generic.cc:26` 读取该值登记发送需求。
`ncclTopoGraph` 保存顺序、端点和成本；本次 `nBytes` 则在 `enqueue.cc:2163、2180` 进入任务调优。

下一章：[07｜Transports：把邻接变成可用连接](../../07-transports.md)。
