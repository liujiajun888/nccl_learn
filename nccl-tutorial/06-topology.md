# 06｜Topology：硬件路径怎样变成 channel 邻接

上一章给出了初始化的整体流程。本章只追一条转换链：

```text
ncclTopoSystem → paths → ncclTopoGraph → channels[c].ring/tree
```

目标是分清：**硬件怎样连接、NCCL 怎样选路、最后怎样形成 Ring/Tree 邻居。** 搜索预算、XML 字段和特殊路径类型留到 [Topology 进阶附录](appendix/topology.md)。

## 1. 三类对象分别是什么？

| 对象 | 回答的问题 |
|---|---|
| `ncclTopoSystem` | GPU、CPU、NIC、NVSwitch 等硬件怎样连接？ |
| `ncclTopoGraph` | 某种算法在局部域怎样排列 GPU 和 NET 出口？ |
| `channels[c]` | 拼接后，当前 rank 在每个 channel 中与谁通信？ |

`ncclTopoSystem` 中有节点和直接连接 `links`；`paths` 是端点之间预先计算的路线；`ncclTopoGraph` 则保存算法搜索结果。

**拓扑图里的路径不是已经建立的 CUDA/RDMA 连接。**

## 2. 先用“只用部分 GPU”理解建图范围

假设一台机器有 8 张 GPU，但当前 communicator 只使用 4 张：

```text
使用：GPU 0、2、4、6
不用：GPU 1、3、5、7
```

正常自动建图时，每个 rank 只探测自己使用的 GPU，再融合同机其他 rank 的描述。因此最终作为通信成员的通常就是这 4 张，而不是全部 8 张。

但使用中的 GPU 所依赖的 PCI bridge、NUMA、NVSwitch 和 NIC 仍可能保留在硬件模型里。

同样，路径搜索只使用当前拓扑中的节点和通信成员，不会随便拿一张不在 communicator 中的 GPU 做中转。

## 3. paths：把一串连接压缩成可比较的路径

一条 path 主要记录：

- 路径类型，例如 NVL、PIX、PXB、PHB、SYS；
- 瓶颈带宽；
- 跳数和经过的 links。

带宽按最窄一段估计：

```text
GPU ── 50 ── PCIe switch ── 25 ── NIC
路径瓶颈 = min(50, 25) = 25
```

这些数字是**硬件属性形成的模型值**，不是本阶段测出的实际吞吐量。

候选路径比较可以先记成：

```text
先看路径类型 → 再看带宽 → 最后看跳数
```

两个 channel 若共用后半段链路，后续搜索还要共享这份带宽预算。

## 4. 物理连通不等于可以采用

NCCL 还会根据能力和策略修正路径：

| 机制 | 直观含义 |
|---|---|
| P2P | GPU 之间直接访问，要求相应拓扑和访问能力 |
| SHM | 通过同主机共享内存通信 |
| PXN | 经另一张 GPU 接近其附近 NIC |
| GDR | 网卡直接访问 GPU 内存，仍需路径与注册条件 |

如果直接 P2P 不可用，路径可能改为经过 CPU；P2P 和 SHM 都不可用时，相应成员会按非本地可达处理，后面通过局部域裁剪决定是否仍作为同域 GPU。

因此，最终路径是“**硬件连接 + 能力检查 + 用户策略**”的组合，而不只是看物理连线。

## 5. `ncclTopoGraph`：为算法寻找并行布局

搜索的主要输入包括：

- 节点间 paths；
- 路径类型和带宽估计；
- GPU/NIC 的可用性；
- channel 数范围和算法模式。

`ncclTopoGraph` 中最重要的输出是：

| 字段 | 含义 |
|---|---|
| `nChannels` | 找到的并行 channel 数 |
| `intra[]` | 每个 channel 的局部 rank 顺序 |
| `inter[]` | 各 channel 的网络入口和出口 |
| `bwIntra/bwInter` | 域内与网络方向的带宽估计 |

搜索会尝试 GPU/NIC 排列，沿候选路径扣减可用带宽，失败时回溯并恢复。

```text
提出带宽和路径约束
  → 尝试排列并记账
  → 不够理想就放宽约束或降低单 channel 带宽
  → 保留更好的结果
```

第一遍只需理解“**用剩余带宽判断多个 channel 能否共存**”，不用马上读递归细节。

## 6. Preset/Postset：把局部布局拼成全局邻接

局部图还不能直接表达整个 communicator 的 Ring/Tree，所以还要经过全局协调：

```text
ncclTopoPreset：提取局部端点和邻居
  → AllGather 交换摘要并对齐参数
  → ncclTopoPostset：补跨节点边界、生成 rings
```

以两个局部 Ring 段为例：

```text
node 0：0 → 1，入口 0、出口 1
node 1：2 → 3，入口 2、出口 3
```

`connectRings` 补上：

```text
next[1] = 2，prev[2] = 1
next[3] = 0，prev[0] = 3
```

最终：

```text
0 → 1 → 2 → 3 → 0
```

`ncclBuildRings` 还会沿 `next` 走完整环，检查回到起点且覆盖所有 rank。

Tree 则按父子关系拼接，写入 `tree.up/down`，不是简单首尾相连。

## 7. 最后交给 transport 的是什么？

到这一步，得到的是：

```text
channels[c].ring.prev/next
channels[c].tree.up/down
rings[c][...]
```

这些是**逻辑邻接**，不是实际传输连接。

例如 r1 的 `ring.next=5` 后，transport 才能：

- 为 r1 登记发往 r5 的发送需求；
- 为 r5 登记来自 r1 的接收需求；
- 分别为两条边选择 P2P、SHM 或 NET；
- 建立真正可用的资源。

如果启用了按需连接，这个资源阶段会推迟到算法首次使用。

## 8. 本章自测

1. `ncclTopoSystem` 与 `ncclTopoGraph` 分别描述什么？
2. 为什么只用 4 张 GPU 时，另外 4 张不会自动加入 Ring？
3. `path->bw` 是实测吞吐吗？
4. 为什么图搜索后还需要 Preset/Postset？
5. `ring.next` 与实际 transport 连接有什么区别？

下一步读 [07｜Transports](07-transports.md)，看这些逻辑邻居怎样变成可访问的 buffer、映射和连接。

需要 XML 融合、路径类型、PXN、搜索回溯和 rail/crossNic 细节时，再读 [Topology 进阶附录](appendix/topology.md)。
