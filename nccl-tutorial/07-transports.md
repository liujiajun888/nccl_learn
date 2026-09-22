# 07｜Transports：谁建立访问，谁搬数据，谁推动完成

> 基线：`12df1a11`，NCCL **2.32.3**。本章以普通 host collective 的 P2P、SHM、NET 为主，单列特殊路径边界。

## 学习目标

- 从连接选择、handle 交换、内存访问权限解释 transport，而非只记三种缩写。
- 精确区分 CPU proxy service、CPU proxy progress、GPU kernel 与 NIC 的工作。
- 解释 UVA、CUDA IPC、cuMem、GDR 和 RDMA memory registration 的关系。
- 用控制状态与 payload 两条线定位“通信不动”的原因。

拓扑怎样决定邻接见 [06](06-topology.md)，任务怎样准备 proxy op 见 [05](05-host-execution.md)。
本章不复述整个初始化过程，而是接着回答：已经知道邻居之后，数据怎样真正到达它？

## 1. 贯穿案例：同一个 AllReduce，边上可以走不同 transport

继续使用两台四 GPU 主机，八个 ranks 对 64 MiB FP32 梯度做 AllReduce。
假设某次使用 Ring/Simple：节点内相邻 rank 可能通过 P2P，跨节点边通过 NET。
若某对同主机 GPU 不能用 P2P，则可能改用 SHM，或根据可行性和策略走 NET。
算法是全局分工，transport 是边的实现，不能把整次 AllReduce 简写成“它就是 RDMA”。

```text
示意数据 ring 的一段：
 A0(r0) --P2P--> A1(r1) --NET--> B1(r5) --P2P--> B0(r4)
     节点内 GPU 互连       跨节点网络        节点内 GPU 互连

若 P2P 不可用：A0 --GPU 可访问的 host shared buffer--> A1
                            SHM
```

上图仅解释不同边的实现，不是完整八 rank 次序或实测 transport 选择。
Transport 名称也不直接指定 payload 一定由 SM、copy engine 或 NIC 中的某一个独占搬运。
应继续追踪对应 connector 的 buffers、权限和 progress callback。

## 2. 选择 transport：先问能否连接，再 setup 和交换描述

`transport.cc` 的候选表依次列出 P2P、SHM、NET、CollNet。
`selectTransport` 调用各自的 `canConnect`，选中第一个允许的候选，然后调用该方向的 `setup`。
这不是无条件“同节点 P2P、跨节点 NET”的 if/else：策略和能力可能让前面的候选主动拒绝。

`setup` 分配或准备本地资源，并形成小型 `ncclConnect` 描述。
双方经 bootstrap 交换描述，再执行 `connect` 导入/映射对端资源或推进网络握手。
成功后，connector 标为 connected，并将 `ncclConnInfo` 拷到设备侧供 kernel 使用。

```text
peer/channel 需求
   -> canConnect -> setup -> ncclConnect 描述
                                 |
                       bootstrap 交换元数据
                                 |
   <- device conn <- connected <- connect/import/map
```

两阶段的原因是：在远端端点/共享对象创建之前，本地无法凭空映射它。
交换几百字节的 handle 和地址信息，与传输 64 MiB 梯度，是完全不同的工作。
函数名 `ncclTransportP2pSetup` 在这里协调多种 peer transport，并不证明最终选择了 P2P transport。

锚点：[transport.cc](../nccl/src/transport.cc)，`nccl/src/transport.cc:23`，`selectTransport`；
`nccl/src/transport.cc:125`，`ncclTransportP2pSetup`；
[transport.h](../nccl/src/include/transport.h)，`nccl/src/include/transport.h:129`，`ncclTransportComm`。

## 3. P2P：UVA 给地址解释，不自动给访问权限

Unified Virtual Addressing（UVA）提供统一的虚拟地址组织和指针属性识别基础。
它既不意味着所有 GPU 自动能读写任意 peer allocation，也不等同于 Unified Memory 的迁移机制。
要访问远端 GPU 内存，还需硬件路径、驱动能力、映射与访问权限共同满足。

本版 `p2pCanConnect` 结合拓扑判断、NET 偏好和 CUDA peer 能力检查。
同进程不同 GPU 的常规路径中，`p2pMap` 会处理 `cudaDeviceEnablePeerAccess`。
不同进程不能直接把一个进程里的指针数值交给另一个进程使用；必须交换可导入的共享描述。
同一物理 allocation 映射到两个进程后，虚拟地址也不要求相等。

P2P 资源导入有两类重要机制：

| 机制 | 交接步骤 | 不应作出的推论 |
| --- | --- | --- |
| Legacy CUDA IPC | 导出 `cudaIpcMemHandle`，对端 `cudaIpcOpenMemHandle` | IPC handle 不是 payload，也不是普通跨进程指针 |
| cuMem/VMM | 导出/导入 allocation handle，reserve VA、map、set access | map 成功不等于所有 GPU 都自动有访问权限 |

POSIX FD 类型的 cuMem handle 需要通过 Unix domain socket 等机制传递真实 FD 权限。
不能把一个进程的整数 fd 原样发过去，就假设它在另一个进程中代表同一对象。
本版相关导入通过 proxy 服务获取 FD，再执行 `cuMemImportFromShareableHandle`。
`cuMemSetAccess` 明确授予本地设备访问映射的权限，正好说明 UVA 与访问授权是两件事。

普通 P2P 数据路径里，GPU kernel 通过已映射的协议 buffer 或适用的用户 buffer 进行 load/store、规约与通知。
这些访问由 GPU 内存系统经 NVLink/PCIe 传递，不必为每一片数据发一个主机 `cudaMemcpy`。
P2P 也可能选择读或写式的推进方式，不应把所有 P2P 固定画成“发送端写对端用户指针”。

锚点：[p2p.cc](../nccl/src/transport/p2p.cc)，`nccl/src/transport/p2p.cc:129`，`p2pCanConnect`；
`nccl/src/transport/p2p.cc:267`，`ncclP2pImportShareableBuffer`；
`nccl/src/transport/p2p.cc:349`，`p2pMap`。

## 4. SHM：同主机共享内存，不是“仅进程内通信”

SHM 使用同主机上的共享 host memory，使两侧 GPU 获得可访问的协议缓冲与同步字段。
它可以跨进程；正因为进程隔离，才需要共享对象的创建和导入。
本版 `shmCanConnect` 检查 host identity、共同的 shared-memory device，以及禁用/NET 偏好等条件。
同机的两个容器若共享内存视图不同，也不能只凭 hostname 相同就认定 SHM 可用。

本版普通 SHM 路径是 GPU 经映射访问 host buffer：

```text
发送 GPU kernel --写入--> host shared buffer --读取--> 接收 GPU kernel
          \---------- head/tail 等协议状态 ----------/
CPU service：创建/导入/释放共享对象，不必逐片执行 payload memcpy
```

Host memory 参与数据路径，不等于 CPU 核心必须复制每一字节。
GPU 对 host memory 的访问仍消耗 PCIe、内存控制器与 NUMA 带宽，性能通常不能按显存带宽估计。
`shmSendConnect`、`shmRecvConnect` 将 host buffer 的设备可见地址接入 `conn.buffs` 与 head/tail。
因此要区分 buffer 的物理位置与执行 load/store 的处理器。

共享 host buffer 的实现可采用传统共享内存映射，也可采用满足条件的 cuMem host allocation。
这改变创建和导入方式，不改变“host memory 作为通信介质”的含义。
不要从旧版本教程搬来不存在于当前分支的 SHM memcpy/progress 流程。

锚点：[shm.cc](../nccl/src/transport/shm.cc)，`nccl/src/transport/shm.cc:61`，`shmCanConnect`；
`nccl/src/transport/shm.cc:153`，`shmSendConnect`；
`nccl/src/transport/shm.cc:467`，`shmTransport`，本版普通 SHM 的 progress callback 为 NULL。

## 5. NET：插件接口把网络实现与 collective 解耦

NET 接入选定的网络实现，可能是外部 plugin，也可能是内置 IB/RoCE 或 Socket 实现。
Bootstrap 使用 socket，不决定 NET bulk 必然使用 TCP；同样，机器有 IB 卡也不证明程序已使用 RDMA。
应检查实际加载的网络实现、选中的设备、连接和 GDR 状态。

主机侧网络接口的核心不是一个阻塞 `send`，而是一组可推进的操作：

| 接口 | 作用 | 调用成功不代表什么 |
| --- | --- | --- |
| `listen/connect/accept` | 建立端点、交换可连接的 handle | 不代表用户数据已传输 |
| `regMr/regMrDmaBuf` | 为已有内存建立网络访问注册 | 不代表分配了新的用户 buffer |
| `isend/irecv` | 尝试发起非阻塞发送/接收 | 不代表网络已经完成；request 还可能暂未产生 |
| `test` | 查询 request 是否完成并推动必要工作 | 不独自代表整个 collective 完成 |
| `iflush` | 在需要时处理接收后的可见性 | 不是每个后端、每种协议必经的固定步骤 |

接口及 ABI 细节要按 plugin 版本核实。
本版接口可从 [net_v12.h](../nccl/src/include/plugin/net/net_v12.h) 阅读：
`nccl/src/include/plugin/net/net_v12.h:104`，`regMr`；`nccl/src/include/plugin/net/net_v12.h:110`，`isend`；
`nccl/src/include/plugin/net/net_v12.h:113`，`irecv`；`nccl/src/include/plugin/net/net_v12.h:120`，`test`。
NCCL 的请求语义不等于指定某一种 verbs opcode；IB 后端可用 RDMA 操作实现这些抽象。

## 6. GDR 与 memory registration：直接访问不等于零准备

**GDR 在这里指 GPUDirect RDMA，不是 Direct GPU Read。**
它使支持的 NIC 可以直接对 GPU memory 发起 RDMA 数据访问，避免 payload 必经 host staging buffer。
“Direct”描述 NIC 到 GPU memory 的数据通路，不表示初始化不需 CPU，也不表示没有协议缓冲。

典型的两种 payload 路线是：

```text
有 GDR： GPU memory <--> NIC === network === NIC <--> GPU memory
无 GDR： GPU <--> host buffer <--> NIC === network === NIC <--> host buffer <--> GPU
```

无 GDR 路线中的 GPU↔host 部分可由 GPU 协议访问完成，不能默认每段都是 CPU memcpy。
有 GDR 时仍可能先使用 NCCL 的 GPU 协议 buffer；满足用户 buffer 注册条件后，才可能减少额外 staging。
CPU proxy 负责请求/完成状态，与 NIC 是否直接访问 GPU memory 是正交问题。

RDMA memory registration 的输入是**已经存在的地址范围**。
注册建立设备访问所需的地址翻译、权限与 MR/key 等元数据；具体固定页或导入映射的机制取决于平台。
它不是 `cudaMalloc`，也不要求把原 buffer 内容复制一遍。
解除注册与释放 allocation 也不是同一件事；未完成请求仍引用该范围时，不能抢先失效或复用。

本版 IB 注册实现带有缓存和引用计数，并可通过 DMA-BUF 注册 GPU 内存。
因此首轮可能承担注册成本，稳态复用注册；仅看首轮耗时会混淆准备成本与链路吞吐。
锚点：[reg.cc](../nccl/src/transport/net_ib/reg.cc)，`nccl/src/transport/net_ib/reg.cc:10`，`ncclIbRegMrDmaBufInternal2`；
`nccl/src/transport/net_ib/reg.cc:98`，`ncclIbRegMr`。

## 7. 精确拆开 CPU proxy service 与 progress

**Proxy service** 处理连接、分配、handle 交换、注册和释放等请求。
**Proxy progress** 持续推进被分配给 CPU 的通信操作，例如检查 GPU 已产生的数据、调用网络接口并查询完成。
它们可以同属 proxy 子系统，却不意味着每条连接都需要两者常驻推进 payload。

本版普通 P2P 和 SHM 有 setup/free 服务，但默认不设置 transport progress callback。
NET 则按连接和 device handle 的能力决定是否需要 host progress。
`proxy.cc` 检查 `connector->proxyConn.proxyProgress == NULL` 时直接跳过操作追加。
因此“看到了 proxy 线程”不是“这次 AllReduce 的所有数据都经 CPU”的证据。

对本例的普通 NET/Simple 边，发送与接收主要交接如下：

| 阶段 | 谁推进 | 交接的数据/状态 |
| --- | --- | --- |
| 发送缓冲可用 | CPU proxy 与 GPU 协议 | step/credit，防止覆盖未完成片段 |
| 发送片段准备好 | GPU kernel | 写 payload，发布 FIFO size/tail 等就绪状态 |
| 网络发送 | CPU proxy → plugin → NIC | `isend` 的 buffer、长度、MR handle 与 request |
| 发送可复用 | CPU proxy 查询完成 | `test` 成功后回收槽位、更新相应 credit |
| 接收投递 | CPU proxy → plugin | `irecv` 的目标地址与注册 handle |
| 接收可供 GPU 用 | 网络完成及必要 flush 后 | 发布接收 tail/可见性状态 |
| 接收缓冲可复用 | GPU 消费后 | 更新 head，proxy 可推进后续片段 |

这不是全局串行流水账：多个 slots、channels 和 requests 可交错推进。
若发送方 GPU 没发布 ready，proxy 无数据可送；若接收方 GPU 不消费，credit 最终耗尽。
若 NIC 已完成写入但可见性步骤未满足，也不能随便提前通知 GPU 读取。
这正是“网络没有报错但 kernel 卡住”需要同时检查两端状态的原因。

锚点：[net.cc](../nccl/src/transport/net.cc)，`nccl/src/transport/net.cc:1324`，`sendProxyProgress`；
`nccl/src/transport/net.cc:1493`，`recvProxyProgress`；
[proxy.cc](../nccl/src/proxy.cc)，`nccl/src/proxy.cc:951`，`ncclProxyProgress`；
`nccl/src/proxy.cc:1789`，`ncclProxyService`；`nccl/src/proxy.cc:567`，`SaveProxy` 的 callback 检查。
更细的 memory ordering 和 LL/LL128 状态语义见 [09](09-device-protocols.md)。

## 8. UMD、KMD、FW 与硬件：不同层次，不是四个搬运线程

下面是常见 GPU/RDMA 软件栈的职责模型，不是说 NCCL 仓库包含全部驱动或固件实现。
具体边界随平台而变，不能仅凭 NCCL 函数名断言某一次 ioctl 或固件指令序列。

| 层次 | 主要职责 | 本例中的对应关系 |
| --- | --- | --- |
| NCCL + 网络 plugin | 算法、连接、协议和请求组织 | 确定这片梯度发给谁、何时可以发 |
| UMD（User-Mode Driver） | CUDA/verbs 用户态接口、命令和队列提交、映射协作 | 接收 kernel launch、访问/注册与网络请求 |
| KMD（Kernel-Mode Driver） | 特权资源、隔离、内存映射/注册协作、设备与故障管理 | 让 GPU/NIC 在授权边界内访问资源 |
| FW（Firmware） | 设备管理与部分控制面/队列行为 | 配合硬件执行设备相关管理，不理解整个用户 AllReduce |
| GPU/NIC/互连硬件 | 执行 load/store、规约、copy、DMA、链路传输等 | 真正读写 payload 并产生硬件完成状态 |

用户态 RDMA 快路径通常不是每个包都进入 KMD；资源创建与数据快路径应分开理解。
NIC 在 RDMA 路径可用 DMA 读写已注册内存；GPU SM 可以执行规约和远端 load/store；copy engine 可承担适用的复制。
因此“所有 NCCL 通信都是 DMA”和“所有数据全由 SM 搬”都不成立。
正确的问题是：**这一条边、这一种协议、这一种 buffer 路径，究竟由谁发起数据访问？**

## 9. 本版本的集中分歧

- [p2p.cc](../nccl/src/transport/p2p.cc) `nccl/src/transport/p2p.cc:122` 的 `P2pUseCudaMemcpy` 默认 **0**；开启时 `initCeOperation` 安装发送 progress，`nccl/src/transport/p2p.cc:844` 的 `p2pSendProxyProgress` 仅为 Simple 执行该 memcpy 流程。
- [shm.cc](../nccl/src/transport/shm.cc) `nccl/src/transport/shm.cc:467` 的 `shmTransport` 默认没有 progress；不要笼统写“SHM 都需要 CPU proxy 搬运”。
- [net.cc](../nccl/src/transport/net.cc) `nccl/src/transport/net.cc:547` 的 `sendConnect` 分支根据 `needsProxyProgress` 设置 callback，接收侧也有对应判断；device-capable 路径不能硬套上述 host proxy 时间线。
- LL/LL128 的 ready/completion 机制与 Simple 不完全一致，本版还存在 optional receive completion；第 7 节只画普通 Simple 主线。
- PXN 可借邻居 GPU/NIC 及其 proxy 资源转发；“本 rank 的网络请求”不一定总由本 rank 自己的 NIC/proxy 处理。
- NVLS、CollNet、GIN/device API、host RMA 与 CE collectives 超出三类基础 transport 图解，见 [13](13-advanced.md)。
- Linux CUDA IPC、cuMem、DMA-BUF 的条件不能直接移植到 ROCm；可迁移职责模型，API 与默认值应重新核实。

## 10. 实操观察：用反事实实验验证路径

以下仅为 Linux/CUDA 读者指引，本机未运行 GPU、RDMA 或网络性能实验。
通过现有启动器运行相同 workload，并把环境一致传到所有 ranks：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET,PROXY,REG \
NCCL_DEBUG_FILE=/tmp/nccl-transport.%h.%p.log ./your_nccl_app
```

先记录实际 transport/网络实现，再分别做“禁用 P2P”和“禁用 P2P+SHM”的对照。
可使用 `NCCL_P2P_DISABLE=1`，以及另一次运行再加 `NCCL_SHM_DISABLE=1`；不要把诊断开关长期留在生产配置。
NET 偏好和能力判断仍可能改变结果，不能预写“禁用 P2P 后必为 SHM”。
对 GDR 做单变量诊断时，可单独试验 `NCCL_NET_GDR_LEVEL=LOC` 并核实日志，不能仅凭吞吐变低判定它已关闭。

在每次结果中分开填写：控制面是否成功、选中哪个 transport、buffer 在哪、谁做 progress、首轮/稳态耗时、正确性。
若只有 bootstrap socket 日志，证据不足以认定 bulk TCP；若只有 NIC 活跃，也不足以认定 GDR。
平台允许时再结合 GPU timeline、NIC 计数器与 CPU 线程负载交叉确认，排错顺序见 [12](12-debugging.md)。

## 11. 自测与简答

1. UVA 下得到对端指针，为什么仍不能直接让 GPU 解引用？  
   **答：**地址统一不授予跨设备/跨进程访问权限；还需可达路径、导入映射与访问授权。
2. 使用 GDR 是否意味着 CPU proxy 消失？  
   **答：**不是；GDR 描述 NIC↔GPU 的 payload 路径，host progress 是否需要由连接能力另外决定。
3. RDMA 注册成功是否表示新显存已分配，或者数据已传完？  
   **答：**都不是；它为已有范围建立网络访问条件，allocation 与请求完成各有独立生命周期。
