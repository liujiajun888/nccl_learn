# MNNVL：把 NVLink 访问域扩展到多个物理节点

[返回教程主页](README.md) · [相关：NVML](appendix-nvml.md) · [相关：NVLS](appendix-nvls.md)

**MNNVL 全称 Multi-Node NVLink，支持不同物理节点上的 GPU 通过 NVLink fabric 通信，并访问已经授权、映射的远端 GPU 显存。** 它扩展的是 GPU 的互连与内存访问范围，集合算法仍由 NCCL 组织。

本文采用 NCCL 2.32.3（`12df1a11`）源码基线，重点说明能力发现、内存映射和传输路径之间的关系。

## 1. 从跨服务器网络到跨节点 NVLink

在常见多机系统中，NVLink 主要连接本机 GPU，跨机数据经过网卡：

```text
节点 A GPU → 网卡 → IB/RoCE 网络 → 网卡 → 节点 B GPU
```

在具备跨节点 NVLink 连接的系统中，数据可以走另一条路径：

```text
节点 A 的 GPU ──┐
节点 B 的 GPU ──┼── NVLink fabric ── 跨节点 GPU 互访
节点 C 的 GPU ──┘
```

这里的 fabric 是由 NVLink 链路和交换设备组成的互连系统。GB200 NVL72 这类机架级 NVLink 系统是典型场景：多台计算节点仍有各自的进程和操作系统，但其 GPU 可以处在更大的 NVLink 访问域中。

**物理服务器边界与 NVLink 域边界因此可以不同。** 域内数据可绕开常规 IB/RoCE 数据路径，初始化控制消息以及域外通信仍可能使用普通网络。

## 2. 跨节点显存怎样变得可访问？

把节点 B 的指针数值发给节点 A，并不能建立访问关系。需要先共享底层内存分配，再在访问方建立映射：

```text
节点 B：分配支持 FABRIC handle 的 GPU 显存
                              ↓
         cuMemExportToShareableHandle
                              ↓
                       交换共享句柄
                              ↓
节点 A：cuMemImportFromShareableHandle
                              ↓
         保留虚拟地址 → cuMemMap → cuMemSetAccess
                              ↓
              GPU A 通过映射地址访问 GPU B 显存
```

几个对象分别承担不同职责：

- **显存分配**：保存真正的数据，仍由分配方管理其生命周期。
- **FABRIC handle**：使用 `CU_MEM_HANDLE_TYPE_FABRIC` 导出的共享句柄，用于在支持的 fabric 中导入内存。
- **本地虚拟地址**：导入方为这份远端内存建立的访问地址，两边的地址数值可以不同。
- **IMEX 服务与通道配置**：支持跨节点内存导入、导出及相应访问授权；CUDA 的映射权限还需单独设置。

映射完成后，GPU 可以用 load/store 访问远端显存。通信协议仍负责同步、可见性和缓冲复用，相关映射与分配也要保留到访问结束。

这是一种显式建立的内存共享关系，不是自动合并全部显存的统一内存池。远端 GPU 也不会因此自动成为本进程可供 `cudaSetDevice` 选择的本地设备。

## 3. NCCL 如何确认哪些 GPU 可以互访？

NCCL 通过 NVML 获取 GPU fabric 信息，并与其他 rank 交换。关键字段包括：

| 字段 | 用途 |
|---|---|
| `state` | 判断 GPU 的 fabric 初始化是否完成 |
| `clusterUuid` | 识别 GPU 所属的 NVLink fabric 域 |
| `cliqueId` | 标识该域内的分组，用于组织拓扑和连接范围 |

在本基线的常规路径中，同一 `clusterUuid`、同一 `cliqueId` 的 rank 被归入一个 clique。`comm->clique.ranks` 保存这些成员，`clique.size` 是其中参与当前 communicator 的 rank 数；它们可以分布在不同物理主机上。

初始化还会检查 cuMem、FABRIC handle 支持与 fabric 状态，并实际测试共享句柄能否导出、导入。成功后设置 `comm->MNNVL`，并将相关共享内存句柄类型切换为 FABRIC。

本基线也有可配置的同域跨 clique P2P 路径。理解主线时先看同一 clique 即可；跨 clique 是否使用 P2P，还要结合对应配置和连接检查。

## 4. 它怎样改变 NCCL 的执行路径？

### 拓扑：从同一主机扩展到同一 clique

普通系统按同一 host 收集本地拓扑。MNNVL 系统则融合 clique 成员的拓扑，因此 NCCL 可以在更大的 NVLink 域内规划通信关系。

这也解释了读代码时容易遇到的现象：**MNNVL 场景中的部分 `localRanks` 表示逻辑域内成员，不一定都是同一操作系统上的进程。** 判断是否能使用主机共享内存等资源时，仍需看真实 host 关系。

### 传输：跨 host 的连接也可以走 P2P

`ncclTopoCheckP2p` 在处理不同 host 的 GPU 时，会进一步检查 MNNVL fabric 关系。通过拓扑和访问条件检查后，P2P transport 导入对端显存映射，设备代码通过 NVLink 推进数据。

建立这种连接后，日志中可以看到类似标记：

```text
via P2P/MNNVL
```

### 算法：继续从可用候选中选择

应用仍调用 `ncclAllReduce`、`ncclAllGather` 等接口。MNNVL 改变可用的连接和成本，NCCL 再据此选择算法；在支持 multicast 与归约的系统上，也可以结合 NVLS 使用。

## 5. 与 NVLS、CollNet 怎样分工？

| 机制 | 负责的问题 |
|---|---|
| MNNVL | 哪些跨物理节点的 GPU 能通过 NVLink 互访，如何建立显存映射？ |
| NVLS | 怎样在支持的 NVLink 域内利用 multicast/multimem 加速集合操作？ |
| CollNet | 怎样把集合操作交给网络插件，例如 InfiniBand SHARP 后端？ |

因此，一个跨越多台物理主机的 MNNVL 域，可以在具备相应能力时使用 NVLS。离开这个域后，NCCL 再按可用拓扑和算法接入其他传输路径。是否使用 CollNet，要单独检查集合网络插件和算法选择。

## 6. 使用条件与排查顺序

MNNVL 需要真实的跨节点 NVLink 硬件连接、配套驱动、就绪的 fabric，以及可工作的 CUDA FABRIC handle 和 IMEX 配置。环境变量只能选择使用策略，不能为普通多机集群增加不存在的 NVLink 链路。

本基线的 `NCCL_MNNVL_ENABLE` 默认值为 `2`，用于自动检测；设置为 `0` 可禁用。需要排查时，按以下顺序看：

1. NVML 是否报告了有效的 fabric UUID、clique 和就绪状态。
2. 当前 communicator 的成员是否处在预期的 fabric 分组中。
3. cuMem FABRIC 分配、共享句柄导出和导入是否成功。
4. 拓扑与 P2P 日志是否建立了预期的 MNNVL 连接。

缺少基本能力时，自动检测可以不启用 MNNVL。若已识别到可用的 MNNVL 域，但 FABRIC 分配或导入、导出失败，本基线会报错并提示检查 IMEX，而不是在所有情况下都静默回退。

本文完成了源码核对，未在跨节点 NVLink 硬件上实测。

## 7. 源码入口

| 阅读目标 | 入口 |
|---|---|
| 能力检测与 clique 成员收集 | [src/mnnvl.cc:14](../nccl/src/mnnvl.cc#L14)，`ncclMnnvlCheck` |
| FABRIC 导入、导出检查 | [src/mnnvl.cc:57](../nccl/src/mnnvl.cc#L57) |
| 按 clique 融合拓扑 | [src/graph/topo.cc:2118](../nccl/src/graph/topo.cc#L2118) |
| 跨 host P2P 与 fabric 匹配 | [src/graph/paths.cc:335](../nccl/src/graph/paths.cc#L335)、[464](../nccl/src/graph/paths.cc#L464) |
| 导入对端显存并设置访问权限 | [src/transport/p2p.cc:267](../nccl/src/transport/p2p.cc#L267)，`ncclP2pImportShareableBuffer` |
| 默认启用策略 | [src/init.cc:1054](../nccl/src/init.cc#L1054) |

继续阅读：[NVML](appendix-nvml.md) 解释设备与 fabric 信息从哪里来；[NVLS](appendix-nvls.md) 解释访问域内怎样利用硬件归约。
