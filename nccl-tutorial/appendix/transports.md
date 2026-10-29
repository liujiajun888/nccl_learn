# 07｜Transports：把邻接变成可用连接

> 阅读定位：进阶原文。先读 [07 Transport 主线](../07-transports.md)，再按 P2P、SHM 或 NET 的具体建连问题查阅。

**Transport（传输实现）**负责把“给哪个 peer 收发数据”变成可访问的缓冲、映射和通信端点。
算法确定通信关系，transport 为每条边准备实现；同一次 AllReduce 的不同边可以采用不同传输方式。
连接准备好后，GPU kernel 才能使用其中的地址和协议状态。

源码基线：NCCL **2.32.3**，提交 `12df1a11`；主线为普通 host collective。

## 1. ncclConnector 与 ncclConnInfo 各保存什么

拓扑层已经把邻居写入 **channel**，即一组可并行使用的逻辑通信资源。
**Peer** 是与本 rank 通信的另一个成员。
每个 channel 按 peer 保存发送和接收连接，再用 `connIndex` 区分连接槽位。
**Proxy** 是在 CPU 侧代办资源准备和通信推进的子系统。

**`ncclConnector`** 是某条连接在本端的主机管理对象，持有实现回调、私有资源、proxy 联系状态和 `connected` 标志。
**`ncclConnInfo`** 是嵌在 connector 中、供 GPU 执行使用的连接信息，准备完成后会复制到设备侧。

```text
comm->channels[c]
  └ peers[peer]
      ├ send[connIndex] : ncclConnector
      └ recv[connIndex] : ncclConnector
           ├ transportComm       该方向的实现回调
           ├ transportResources  主机管理的私有资源
           ├ proxyConn           CPU 服务联系状态
           └ conn : ncclConnInfo  GPU 使用的地址与状态
```

`conn.buffs[]` 按协议保存数据缓冲地址，`head/tail` 指向槽位回收和数据就绪的进度计数。
**协议缓冲**是让数据分片轮转、让两端协调生产与消费的存储区。
`stepSize` 描述 Simple 的步长，`connFifo` 则为 GPU 与 proxy 交换每片描述提供队列信息。
**Simple** 是以分片缓冲和进度计数组织传输的一种协议。

对 GPU 来说，关键是这些地址和字段如何使用；回调函数及资源所有权留在主机侧管理。
这样，设备协议可以复用相同的连接信息布局，而不同 transport 各自完成资源准备。

## 2. canConnect、setup、connect、free 的交接

`ncclTransport` 提供 `canConnect`，并分别提供 send/recv 两套 `ncclTransportComm` 回调。
通用选择器依次询问 P2P、SHM、NET 等候选，选中第一个允许当前连接的实现，再执行相应方向的 setup。
**P2P** 使用 GPU 间直接访问，**SHM** 使用同主机共享内存，**NET** 使用网络插件。

| 接口 | 负责的交接 |
| --- | --- |
| `canConnect` | 根据成员身份、拓扑、能力和策略判断是否采用该实现 |
| `setup` | 准备本端资源或监听端点，产生可交换描述 |
| `connect` | 消费对端描述，完成映射或网络握手，填好连接信息 |
| `free` | 回收本端 connector 持有的映射和私有资源 |

**`ncclConnect`** 是 setup 产生的固定大小描述容器，本版容量为 256 字节。
它可以携带共享内存描述或网络监听 handle。
双方先创建可供联系的对象，再交换描述，才能完成对端访问关系。

```text
本端 setup ── 本端描述 ──→ bootstrap ──→ 对端 connect
本端 connect ←─ 对端描述 ── bootstrap ←── 对端 setup
```

**Bootstrap** 是成员间交换建连元数据的引导通道。
`ncclTransportP2pSetup` 负责协调这里的候选选择、描述交换和 connect；它服务多种 peer transport，包括 NET。
网络握手尚未完成时，connect 可返回 `ncclInProgress`，协调层继续推进直到所需连接成功。

成功后，协调层设置 `connected`，把 `conn` 异步复制到设备侧，并安排内部 stream 依赖，使后续执行看到已准备的连接信息。
资源回收由 connector 的 `free` 和服务端的 `proxyFree` 分担；完整释放顺序放在 [09｜内存与生命周期](../09-memory-and-lifetime.md)。

## 3. P2P 与 SHM：谁访问缓冲，缓冲在哪里

沿上一章的示意 Ring，r3→r1 是同主机边，r1→r5 是跨主机边。分别看这两种连接怎样准备。

**P2P 路径**先检查 GPU 间路径与访问能力。
本例的同主机成员分属不同进程，通常需要通过 CUDA IPC 或 cuMem 共享描述导入 GPU 内存。
**IPC（进程间通信）handle** 在这里代表可导入的共享对象；**cuMem** 是 CUDA Driver 的虚拟内存管理接口族。

导入后，NCCL 在当前进程建立映射并授予本地 GPU 访问权限，再把得到的地址放入 `conn`。
在 cuMem 分支中，可以直接看到 import、reserve、map、set access 四步。
它们把同一资源接入不同进程的地址空间，适合用已有的 UMD 映射与授权经验理解。

普通 P2P 的数据访问由 GPU kernel 发起，经 NVLink 或 PCIe 到达相关 GPU 内存。
写式路径可把数据写入对端协议缓冲；读式 Simple 路径则让接收端读取发送端缓冲。
满足直接访问条件时还可使用用户缓冲，这部分注册与寿命关系留到第 09 章。

**SHM 路径**要求成员位于同主机并具备可用的共享内存环境。
setup 创建共享 host buffer，connect 导入对端共享对象，把其 GPU 可见地址接到 `conn.buffs/head/tail`。

```text
普通 P2P：GPU kernel ←── NVLink / PCIe ──→ GPU 内存
普通 SHM：发送 GPU ──写──→ host shared buffer ──读──→ 接收 GPU
```

SHM 的存储介质是主机内存，执行常规 payload 读写的仍是 GPU。
**Payload** 指实际传输的数据内容，此处就是梯度分片。
它消耗 PCIe、主机内存与 NUMA 带宽，因而与显存中的 P2P 路径有不同成本。

**Proxy service** 是 CPU 侧处理分配、共享 handle、建连和释放请求的服务。
P2P 和 SHM 都会使用这类服务准备资源。
**Proxy progress** 是持续推进通信操作的回调机制；本版普通 SHM 的 progress 回调为 `NULL`，普通 P2P 也可由 GPU 自行推进数据协议。

## 4. NET：网络端点与 GPU 缓冲怎样接起来

NET 通过插件接口接入网络实现，既可以使用内置 IB/RoCE 后端，也可以使用 Socket 或外部插件。
其中 IB/RoCE 是常见的 RDMA 网络技术，RDMA 表示网络设备按授权直接访问通信内存。
这里继续取普通主机代理推进的 IB/RoCE NET/Simple 分支。

NET setup 根据拓扑图选择网卡和相关 proxy。
接收侧 service 调用网络接口 `listen`，得到可供发送侧连接的 handle；发送侧准备本端设备与连接参数。
描述经 bootstrap 交换后，双方 service 分别调用网络插件的 `connect/accept`。

连接过程中，service 还会分配协议缓冲、准备网络访问注册，并返回 **`connectMap`**，即描述资源位置及 CPU/GPU 地址的映射表。
主机侧 transport connect 据此导入或定位资源，填入 GPU 要用的 `ncclConnInfo`。
因此资源准备既包含 CPU 服务调用，也包含 CUDA 内存分配、映射和设备侧状态准备。

**GDR（GPUDirect RDMA）**让支持的网卡直接读写 GPU 内存，省去 host buffer 中转。
**MR（Memory Region）注册**为已有内存范围建立网络访问所需的翻译、权限与句柄，供后续网络请求引用。

```text
NET/Simple + GDR：
发送 GPU 协议缓冲 → NIC → 网络 → NIC → 接收 GPU 协议缓冲

NET/Simple 使用 host 缓冲：
GPU → host buffer → NIC → 网络 → NIC → host buffer → GPU
```

有 GDR 时，数据仍可经过 NCCL 的 GPU 协议缓冲；直接使用用户缓冲还需相应注册条件。
使用 host 缓冲时，GPU 可以直接访问其映射，主机 CPU 主要承担网络操作与状态管理。

在这里选定的主机代理分支中，GPU 生产或消费分片，proxy 调用网络发送/接收接口并检查完成，网卡执行网络数据搬运。
详细的请求状态、槽位回收和接收可见性放在 [08｜Proxy 与网络](../08-proxy-and-network.md)。

## 5. 具体交接：把 r1→r5 变成可提交的连接

沿前章教学布局，r1 使用 A1，r5 使用 B1，二者各靠近本机 NIC0。
取 Ring/Simple、NET、GDR 和 NCCL 协议缓冲这一条具体分支，连接使用 `connIndex=0`。
假设本轮首次触发 Ring 的按需连接，算法和协议已在主机准备阶段选定。

1. **拓扑交给 transport。**r1 的 `ring.next=5` 产生发送需求；r5 的 `ring.prev=1` 产生对应接收需求。
   选择器为这条跨主机边选 NET，双方 setup 请求各自 proxy 准备资源。
2. **setup 交给 connect。**r5 的接收 service 创建监听端点，把 handle 放入描述；bootstrap 将其送到 r1。
   r1 的发送 connect 把该 handle 交给发送 service，双方推进网络握手。
3. **service 交给 GPU 连接信息。**握手、缓冲和注册准备完成后，service 返回映射表。
   transport 填入 `buffs、head/tail、stepSize` 等字段，协调层复制设备侧连接信息并建立执行依赖。
4. **连接交给本次执行。**主机执行层为这次 64 MiB AllReduce 分配工作；GPU 与 proxy 再按分片协议使用连接。
   多轮任务可以复用同一连接和缓冲，任务内容由每轮的工作描述携带。

这次交接中，bootstrap 搬的是建连描述，CUDA 映射让 GPU 能访问缓冲，网络注册让 NIC 能访问相应范围。
三者准备的是不同访问关系，最终在同一个 connector 上汇合。

## 6. 源码入口与无 GPU 阅读任务

- [`src/include/transport.h:129` · `ncclTransportComm`](../../nccl/src/include/transport.h#L129)、[`src/include/device.h:136` · `ncclConnInfo`](../../nccl/src/include/device.h#L136) 和 [`device.h:168` · `ncclConnector`](../../nccl/src/include/device.h#L168)：对照实现回调、主机管理对象与设备连接信息。
- [`src/transport.cc:23` · `selectTransport`](../../nccl/src/transport.cc#L23) 与 [`125` · `ncclTransportP2pSetup`](../../nccl/src/transport.cc#L125)：选择、setup、描述交换、connect 和设备复制的协调层。
- [`src/transport/p2p.cc:393` · `p2pSendSetup`](../../nccl/src/transport/p2p.cc#L393)：457～466 行请求 proxy 并映射资源；[`267` · `ncclP2pImportShareableBuffer`](../../nccl/src/transport/p2p.cc#L267) 展示跨进程导入。
- [`src/transport/shm.cc:153` · `shmSendConnect`](../../nccl/src/transport/shm.cc#L153)：导入共享内存并填连接字段；[`467` · `shmTransport`](../../nccl/src/transport/shm.cc#L467) 展示 service 与空 progress 回调。
- [`src/transport/net.cc:296` · `sendSetup`](../../nccl/src/transport/net.cc#L296) 与 [`450` · `sendConnect`](../../nccl/src/transport/net.cc#L450)：选择网络资源、请求 service、接收映射表并填充连接信息。
- [`src/transport/net.cc:793` · `recvProxySetup`](../../nccl/src/transport/net.cc#L793) 与 [`858` · `sendProxyConnect`](../../nccl/src/transport/net.cc#L858)：定位监听 handle、网络连接、缓冲及 MR 注册的准备位置。

**阅读任务：只追 SHM 发送端，解释“有 proxy service，但 GPU 搬 payload”。**

找到共享对象的创建者、对端对象的导入者、设备地址的保存位置，再检查 progress 回调。
最后指出这些地址怎样进入设备侧连接信息。

**答案线索：**`shm.cc:245` 的 `shmSendProxySetup` 创建本端共享对象；153 行的 `shmSendConnect` 导入对端对象。
165～170 行把设备可见地址接到 `conn.buffs/head/tail`；470～471 行保留 setup/free 服务，progress 槽位为 `NULL`。
`transport.cc:249–255` 设置连接标志并复制 `ncclConnInfo`，361 行安排内部 stream 等待关系。
由此可以看出：CPU 准备资源，GPU 使用映射后的地址执行常规 SHM 数据协议。

下一章：[02｜Host execution：从 API 任务到执行计划](../02-host-execution.md)。
