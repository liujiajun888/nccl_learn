# 06｜Topology：从硬件连接图到每个 rank 的邻居

> 基线：`12df1a11`，NCCL **2.32.3**。本章讨论发现、路径建模与通信图构造，不把拓扑估计当成实测带宽。

## 学习目标

- 区分硬件 system graph、端点间 path、算法 graph 与最终 rank 邻接表。
- 解释 PCIe、NVLink/NVSwitch、NUMA、NIC 与多 rail 如何影响通信安排。
- 读懂 `nvidia-smi topo -m` 能提供什么、不能证明什么。
- 沿 `topo.cc → paths.cc → search.cc → connect.cc` 建立可检验的因果链。

承接 [05 的初始化](05-communicator.md)，本章把 [04 已手算的 Ring/Tree](04-algorithms.md) 放到真实硬件图上，不要求先掌握 task/plan 的实现。核心问题只有一个：同样八张 GPU，为什么放置、连线或 NIC 选择不同，就需要不同的通信安排？

**CUDA/UMD 读者切入点：**一对设备能建立映射或允许 P2P，只回答了局部可达性。NCCL 还要为整个 rank 集合安排同时发生的传输。两条各自可达的 path 可能共享同一个 PCIe 上行或 NIC，所以"单路径允许访问"推不出"多 channel 总带宽相加"。带着这个差别读第 4～8 节：先看路径能力与瓶颈，再看图搜索如何分配共享资源，最后看逻辑邻接如何交给 transport。此处的 graph 是通信拓扑图，不是 CUDA Graph。

## 1. 贯穿案例：八卡 AllReduce 的两类距离

案例沿用两主机、每主机四 GPU、64 MiB FP32 AllReduce。每台主机有两组 NUMA/PCIe locality：NIC0 靠近 GPU0/1，NIC1 靠近 GPU2/3。四张 GPU 通过 NVSwitch fabric 互连，两张 NIC 分别接 rail 0、rail 1——**rail** 指跨节点网络中相互独立的转发平面（可理解成并行的"专用通道"）。这是示意硬件，不代表某一实际服务器产品或搜索结果。

```text
主机 A（主机 B 对称）

 CPU/NUMA0 -------- CPU interconnect -------- CPU/NUMA1
     |                                           |
  PCIe 域 0                                   PCIe 域 1
   /  |  \                                     /  |  \
 A0  A1  NIC0                                  A2  A3  NIC1
  \   \                                         /   /
   +------------- NVSwitch fabric -------------+

 NIC0 -> rail 0 -> B 的 NIC0     NIC1 -> rail 1 -> B 的 NIC1
```

这里有两类不同的"距离"：GPU 之间的近邻关系，和 GPU 到 NIC 的近邻关系。A0 到 A3 走 NVLink 很快，A0 到 NIC1 却可能要跨 NUMA，或借另一张 GPU 中转。如果所有跨节点流量都挤向 NIC0，即使 GPU 间连接理想，另一条 rail 也会闲置。反过来，为了多用 NIC 而制造额外的跨 NUMA 流量，也不一定划算。

所以 NCCL 找的不是"最短的一条路"，而是在多种约束下，找一张能并行利用资源的通信图。

## 2. 四张图不要混为一谈

本节给出全章主线：从硬件信息到实际连接，依次经过四层表示。

```text
硬件/系统信息
     |
     v
system graph：GPU / CPU / PCI / NIC / NVSwitch 等节点与 links
     |
     v
paths：端点对的可达性、路径类型、瓶颈带宽、跳数
     |
     v
algorithm graphs：多 channel 的 Ring / Tree 等候选结构
     |
     v
rank mapping：每个 channel 上本 rank 的 prev/next、parent/children
     |
     v
transport connectors：为这些邻接关系实际建连接
```

每张图回答一个问题。第一张："机器像什么"。第二张："端点间可以怎样走"。第三张："多个参与者如何组织传输"。最后的映射让每个 rank 知道自己的职责。transport 才把可行邻接关系变成可用的访问/网络资源，见 [07](07-transports.md)。

**注意：**bootstrap ring、算法 ring 和 NVLink 物理环是三个不同的东西，分别服务会合、集体通信调度和硬件连接，可以有不同的结构与编号。

## 3. Discovery：NCCL 并不凭 rank 编号猜硬件

本节回答：system graph 从哪里来？`ncclTopoGetSystem` 以 XML 为中间形式。它先尝试读取用户指定的 `NCCL_TOPO_FILE`，否则尝试 topology daemon（可选的拓扑服务进程，提供预先发现的硬件 XML）的默认 XML；再补充本 rank 管理的 GPU、CPU/PCIe 关系、NVLink 信息和网络插件报告的设备属性。实际信息获取还依赖平台辅助层、CUDA/NVML 和 OS 暴露的硬件视图。

各 rank 的局部信息怎样合成一张完整图？本版先检测本 rank 管理的 GPU，再通过本地成员的 bootstrap all-gather 融合 XML。所以"每进程只看见一张 GPU"不等于"NCCL 最终只知道一张 GPU 的拓扑"。

网络插件报告设备能力、PCI 路径、速率等，NCCL 把它们接入 GPU/CPU 周围的图。这不等于 NCCL 自动遍历并测量了整个数据中心的每台交换机；网络 fabric 的 rail 设计、拥塞、路由限制仍需要运维配置与额外观测。

<details><summary>深入：源码入口</summary>

锚点：[topo.cc](../nccl/src/graph/topo.cc)，`nccl/src/graph/topo.cc:1989`，`ncclTopoGetSystem`；[xml.cc](../nccl/src/graph/xml.cc)，`nccl/src/graph/xml.cc:1077`，`ncclTopoFillGpu`。阅读 `ncclTopoGetSystem` 时重点跟踪 `xml → rankXml → 融合后的 system`，不必背每种节点的全部属性。

</details>

## 4. Path：带宽之外还有可用性和路径类型

本节回答：两个端点之间"可以怎样走、走得多好"？`ncclTopoComputePaths` 计算到 CPU、GPU、NET、NVSwitch 等端点的路径；底层 `ncclTopoSetPaths` 从基准节点扩展，传播路径类型、带宽与跳数。本版比较候选的次序是：先比路径类型，类型相同再比带宽，都相同再比跳数。它既不是只有距离的 Dijkstra，也不是永远选原始带宽最大的路。

一条 path 的带宽估计取沿路瓶颈：

```text
GPU -- 50 单位 --> switch -- 25 单位 --> NIC
单条 path 的瓶颈 <= min(50, 25) = 25
```

数字只是解释 `min(path->bw, link->bw)` 的教学单位，不是任何产品的带宽数据。两个 channel 若共享后半段链路，两个 25 不能相加为 50；链路竞争留给后面的图搜索处理。真实吞吐还受协议、包大小和执行开销影响。

常见路径类型可这样理解：

| 类型 | 物理/逻辑含义 | 本例中应问的问题 |
| --- | --- | --- |
| `LOC` | 本地自身 | 这是本端还是通信对端？ |
| `NVL` | 经 NVLink | 是否存在可用的高速 GPU 互连？ |
| `NVB` | 经中间 GPU 的 NVLink 路径 | 是否允许这种 GPU 中转？ |
| `PIX` / `PXB` | 至多一个 / 多个 PCIe bridge | 是否共享上游 PCIe 瓶颈？ |
| `PHB` | 经过 PCIe host bridge | GPU/NIC 是否在合适的 PCIe 域？ |
| `SYS` | 跨 NUMA 的系统互连路径 | 流量是否穿过较远的 CPU/NUMA 边界？ |
| `PXN` | 经另一 GPU 接入 NIC | 多一次 GPU 中转能否换来更好的 NIC locality？ |
| `NET` / `DIS` | 网络路径标记 / 不连通 | 本地可行域是否需要裁剪或改走网络？ |

**注意：**这是类型解释，不是固定速度排行榜；同类型链路也可能有不同宽度和代际。

<details><summary>深入：源码入口</summary>

锚点：[paths.cc](../nccl/src/graph/paths.cc)，`nccl/src/graph/paths.cc:52`，`ncclTopoSetPaths`；`nccl/src/graph/paths.cc:754`，`ncclTopoComputePaths`；[graph.h](../nccl/src/include/graph.h)，`nccl/src/include/graph.h:120` 起，`PATH_*` 定义。

</details>

## 5. 为什么还要改写路径、裁剪、再计算

物理上连着，不等于驱动允许访问，更不等于 NCCL 策略决定使用它。路径算完，还要结合 P2P、SHM、GDR 的能力与策略修正图。这里 P2P 指 GPU 间直接访问，SHM 指主机共享内存路径；**GDR（GPUDirect RDMA）**指网卡不经过 CPU 内存中转、直接读写 GPU 内存的能力。GDR 不可用时可能需要 host 中转，所以物理距离不能单独决定可用的数据路径；连接细节见 [07](07-transports.md)。

举两个改写的例子。GPU peer access 不可用时，GPU 间路径可能被改写为经 CPU 的路径。若两个 ranks 连 P2P/SHM 都不能使用，它们在当前本地域中的关系会被标记并参与裁剪——这不是把 rank 从 communicator 除名，它仍可通过 NET 参与整个团队。

GPU 到 NIC 同样要判断 GDR 支持，而不是只看 PCIe 距离。PXN 还可能把发送方向改为"本 GPU → NVLink 邻居 → 邻居附近 NIC"。这个中转只在能力、locality 与收益条件都满足时建立，不是所有远 NIC 路径的默认解法。

因此初始化执行的是：

```text
发现 system -> compute paths -> trim system -> recompute paths -> search
```

删除或调整节点后，旧路径缓存可能不再适用；重算确保搜索基于当前可行域。本例若容器隔离使 SHM 不可用、驱动又不支持某对 GPU 的 P2P，最终图会与裸机不同。

<details><summary>深入：源码入口</summary>

在 [init.cc](../nccl/src/init.cc) 的 `nccl/src/init.cc:1391`，`initTransportsRank` 中可看到整个顺序。在 [paths.cc](../nccl/src/graph/paths.cc) 的 `nccl/src/graph/paths.cc:900`，`ncclTopoTrimSystem` 中可继续追踪裁剪。

</details>

## 6. Search：不是排一个最短 ring，而是安排多条并行通道

本节回答：候选的算法图是怎么搜出来的？`ncclTopoCompute` 接收 path 信息，以及目标 pattern、最少/最多 channels 等约束。输出图包括 `nChannels`、节点内/节点间带宽估计、路径类型和 `intra/inter` 序列。Ring 和 Tree 分别搜索，Tree 的 channel 数约束会受 Ring 搜索结果影响。这样不同算法就有可比较、可拼接的资源布局，而不是每次 collective 从零枚举整台机器。

搜索递归尝试候选 GPU/NIC 顺序，扣减路径资源，再回溯尝试其他分配；外层还会调整目标带宽、路径限制、same-channels、cross-NIC 等条件，并受搜索预算约束。找到结果不代表数学上的全局最优，它是有限成本下的启发式资源安排。

为什么要多 channel？一条逻辑 ring 未必能用足所有独立链路，多条分工可提高并行度。为什么不能无限增加？channels 会争用链路、SM、协议缓冲与调度资源。拓扑搜索输出的是可用骨架；具体一次 AllReduce 用多少 channels，还要由 [08](08-host-execution.md) 的调度决定。

<details><summary>深入：源码入口</summary>

锚点：[search.cc](../nccl/src/graph/search.cc)，`nccl/src/graph/search.cc:1151`，`ncclTopoCompute`；[graph.h](../nccl/src/include/graph.h)，`nccl/src/include/graph.h:174`，`ncclTopoGraph`。

</details>

## 7. 多 rail：NIC 不是越多越自动变快

本例 rail 0 连接各主机的 NIC0，rail 1 连接 NIC1。若跨 rail 的交换网络很弱或不可达，错误地交叉选择两端 NIC 可能制造瓶颈甚至连接问题。反过来，若 fabric 本来允许任意 NIC 间高效通信，过强的"同 rail"限制又会浪费资源。

**rail** 是跨节点的端口/网络组织关系，不能简单等同于一个 CUDA channel：一个 rail 可承载多个 channels，一个 GPU 也可能有多种 NIC 选择。所以要同时检查本地 GPU→NIC locality 和远端 NIC 配对，而不是只给 GPU 选"距离最近的网卡"。

<details><summary>深入：本版的 cross-NIC 策略</summary>

本版 `NCCL_CROSS_NIC` 默认 **2**，不是强制每条 ring 都跨 NIC；见 [search.cc](../nccl/src/graph/search.cc)，`nccl/src/graph/search.cc:16`，`CrossNic`。`ncclTopoCompute` 先尝试相应约束，再在条件允许时尝试 cross-NIC 方案。`ncclTopoPostset` 对某些 crossNic=2 结果还会交替 ring 布局，以避免不必要的 rail 交叉。不要脱离 fabric 设计，把某个 `CROSS_NIC` 值当成普遍最优配置。

</details>

## 8. Rank mapping：编号不是 ring 次序

应用给出的 rank 是用户语义编号；搜索优化的是物理资源使用，两者没有必然的顺序关系。本例某个示意 channel 可以是：

```text
r0 -> r2 -> r3 -> r1 -> r5 -> r7 -> r6 -> r4 -> r0
       主机 A             主机 B
```

这只是合法次序的示意，不是本硬件必然选出的 ring；另一 channel 可以用不同排列。对 rank 2 来说，这个 channel 的前驱是 0、后继是 3，而不是按模 8 推出来的 1、3。AllGather 等操作仍须按用户 rank 组织结果，算法内部重排不允许改变用户可见的结果布局。

**动手验证（需 GPU）：**设 `NCCL_GRAPH_DUMP_FILE=/tmp/nccl-graph.dot` 后运行一次测试，对照 dump 出的 channel 邻接与 rank 编号：环次序由图搜索决定，不是模 8 递增。

逻辑邻接怎样落成？`ncclTopoPreset` 先提取本地图的邻接和节点出入口。各 ranks 交换摘要后，`ncclTopoPostset` 拼接跨节点 ring/tree，填入每个 channel 的邻接表。所以 `graph/connect.cc` 中的 connect 首先是**连逻辑图**，不是在这里完成全部 IPC/RDMA 建链；实际 connector 建立还会进入 `transport.cc`，并可能按需推迟到运行时。

<details><summary>深入：源码入口</summary>

锚点：[connect.cc](../nccl/src/graph/connect.cc)，`nccl/src/graph/connect.cc:20`，`ncclTopoPreset`；`nccl/src/graph/connect.cc:380`，`ncclTopoPostset`。

</details>

## 9. 实操观察：怎样读 topo -m 而不越界推论

以下命令可在 Linux/NVIDIA 节点执行；本机未运行，也不提供虚构的命令输出。

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv
nvidia-smi topo -m
lspci -tv
numactl --hardware
```

先建立 GPU index、bus ID、应用 rank 的对照表，不要只留一张没有进程映射的矩阵截图。再看 GPU-GPU 与 GPU-NIC 条目，以及 CPU Affinity、NUMA Affinity 等列。`NV#` 表示一组 NVLink 连接，`PIX/PXB/PHB/SYS` 描述 PCIe/系统层级关系，以当前工具输出的 Legend 为准。这些字符不是 NCCL 的最终 transport 判决，也不是一次 RDMA 注册或 peer access 的成功证明。

按案例提四个可验证问题：

1. A0 与 A3 虽有 NVLink，是否分别更接近不同 NIC？
2. proxy 所在 CPU 与 NIC/GPU 的 NUMA 关系是否合理？
3. 应用重新映射 rank 后，最终 channel ring 顺序有没有变化？
4. 增加 channels 是否真的利用另一 rail，还是仍共享同一 PCIe 上行？

NCCL 的建模结果可用日志和 XML 继续观察：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,INIT,NET \
NCCL_TOPO_DUMP_FILE=/tmp/nccl-topology.xml \
NCCL_GRAPH_DUMP_FILE=/tmp/nccl-graphs.xml ./your_nccl_app
```

多 rank 使用 dump 文件时，应通过 rank 包装脚本分配独立路径或限制输出 rank，避免文件覆盖。拓扑 dump 描述硬件模型，graph dump 描述搜索结果，两者不能互相替代。不要一上来就改 XML"调出好看的图"：先检查硬件视图和实际链路，再做一次一变量实验。真实吞吐、消息大小拐点和算法差异在 [10](10-nccl-tests.md) 中验证。

## 10. 本版本的集中分歧与边界

- 本版 system graph 包含逻辑 GPU 与物理 DEV 等层次，不能直接套用旧图中"每 GPU 一个节点"的字段索引。
- 除常见路径外还有 `C2C`、`P2C`，定义见 [graph.h](../nccl/src/include/graph.h) `nccl/src/include/graph.h:129` 起的 `PATH_*`；CPU-GPU coherent interconnect 不能当作普通 PCIe。
- MNNVL 的 XML 融合可以按 clique 跨主机进行；本章常规 node=主机的图解不覆盖它。
- NVSwitch 的存在不自动证明某次通信使用 NVLS；拓扑能力、算法支持和任务调优还要分别满足。
- 网络虚拟设备/多端口融合由插件属性及策略参与建模，图中 NET 数量未必等于机箱里的物理网卡数量。
- 搜索图不测量运行时拥塞，rank 重编号也不能创造不存在的带宽；更多高级路径见 [13](13-advanced.md)。
- 以上 CUDA/NVML 路径不等同于 ROCm 的发现实现，迁移概念后应重新核实 RCCL 的源码和工具。

## 11. 自测与简答

1. `nvidia-smi topo -m` 显示 NVLink，为什么 NCCL 仍可能不选普通 P2P？  
   **答：**物理路径之外还有驱动权限、可访问性、策略与替代路径选择；必须看实际 canConnect 与连接日志。
2. 多 channel 是否意味着总带宽等于单 channel 带宽乘 channel 数？  
   **答：**不是；共享链路和 GPU 资源会限制并行收益，图搜索只提供估计与安排。
3. rank 3 的 ring 后继为什么可能不是 rank 4？  
   **答：**rank 是用户编号，ring 次序由拓扑与搜索决定，后继应读对应 channel 的映射。

主线下一步：[07 Transports](07-transports.md)，把已知的邻接关系变成双方可用的连接资源。
