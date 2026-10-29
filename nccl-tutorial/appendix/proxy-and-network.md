# 08 Proxy 与网络：把 GPU 的一片数据送过网卡

> 阅读定位：进阶原文。先读 [08 Proxy 与网络主线](../08-proxy-and-network.md)，再按 proxy op、request 或可见性问题查阅，适合定位“网络完成但 GPU 仍等待”。

[上一章：算法](../04-algorithms.md) · [教程主页](../README.md)

沿用两台机器、每台 4 GPU、8 rank 各一进程的案例：
每 rank 对 64 MiB FP32 数据做 Sum AllReduce，即 `count=16,777,216`。
本章具体追 Ring/Simple 的一条跨机边，算法选择由前面的规划阶段决定。
完整输入由算法分配给多个 channel，再切成逐步传输的片；网卡看到的是这些片的网络请求。

[07](../07-transports.md) 已建立连接，[02](../02-host-execution.md) 已分出 proxyOp，
[03](../03-device-execution.md) 已介绍 Simple 的槽位协议。
现在把这几部分接起来：谁把描述变成网络请求，谁发布可读状态，谁归还槽位。

## 1. Service 管连接，progress 推进在途工作

应用线程提交 AllReduce 后，GPU kernel 与 CPU proxy 可以同时工作。
proxy 内部有两个主要线程角色，围绕同一组连接资源分工。

**Proxy service** 接收控制 RPC，管理连接对象和异步控制请求。
它调用 transport 的 setup、connect、register、deregister 回调，
把连接、映射或注册句柄交回请求方；尚未完成的控制请求留在队列中继续推进。

**Proxy progress** 从共享 op pool 接收通信工作，遍历 active 链表，
调用连接对应的 `proxyProgress`，反复提交、测试网络请求并更新 GPU 可见的控制字。
需要 progress 的连接初始化时，service 建立 op pool 并启动这个线程。

```text
应用线程 ── 控制 RPC ───────────────> service ── 连接 / MR
         └─ ncclProxyOp → op pool ──> progress ── request / step
GPU kernel <──── head、tail、FIFO ────────┘
```

这里区分的是连接管理与稳态数据进展。
service 中也有名为 `proxyProgressAsync` 的函数，它推进的是控制 RPC；
逐片收发走的是 progress 线程调用的 NET progress 函数。
连接建好以后，service 留待后续控制请求，progress 持续处理已经发布的工作。
应用线程因而能继续提交计算，而通信依靠 GPU、代理线程和网卡共同推进。

## 2. 从 ncclProxyOp 到 active args，再到 request

`ncclProxyOp` 描述一次已规划的传输：连接、channel、协议、步数、
切片步长和缓冲信息。一个描述覆盖多片数据，代理按步数逐片生成请求；
因此一次集合、一个代理工作和一个网络请求处在三个不同粒度。
它来自 plan，并沿以下路线移交：

```text
plan.proxyOpQueue
  → ncclProxySaveOp → 复制到共享 op pool
  → ncclProxyStart → 发布这批 ops
  → ncclProxyGetPostedOps → ProxyAppend
  → active ncclProxyArgs → subs[] → requests[]
```

`ProxyAppend` 为同一连接保持顺序。
可聚合的 shared 连接、相同 opCount 会组成一个 args 的多个 sub；
同一连接后续工作挂到 `nextPeer`，轮到它时才进入 active 位置。

args 保存协议与进展函数等公共信息，sub 保存某个 channel/连接的独立进度。
`ncclProxyOpToArgs` 填好 sub 后，共享 pool 中原来的 op 槽就能归还。
此后状态留在 args/sub，网络插件返回的 request 放在 sub 的请求槽中。
每轮遍历只推进当前能执行的阶段：有空间就投递，有完成就确认，条件未齐就保留状态。
下一轮从这些字段继续，因此一个线程可以交替服务多条连接。

首次进入 NET progress 时，状态从 `Ready` 变成 `Progress`：
先确定本次操作的 `base`，再把 `posted`、`transmitted`、`done` 等相对计数清零。
接收方向另有 `received`，用于分开网络完成和 GPU 可读。

所有 sub 完成后，args 变成 `None`；`removeOp` 将其归还 args pool，
并把该连接的 `nextPeer` 接上来。连接与 MR 继续供后续操作复用。
这形成三层寿命：长寿命连接、一次工作 args、若干短寿命网络 request。

## 3. Payload 与 CPU 控制是两条线

下面固定使用普通 NET proxy、dedicated 内部缓冲：
`resources->shared == 0`、`sub->reg == 0`，并观察开启 GDR 的 Simple 路径。
这里的 `reg == 0` 表示使用内部缓冲；内部缓冲本身已有网络 MR。

```text
payload：发送 GPU 内部槽 → NIC → 网络 → NIC → 接收 GPU 内部槽
control：GPU 发布就绪 → send proxy 提交 / test
                           recv proxy test / flush → GPU 消费 / 回 credit
```

GDR 让 NIC 直接访问 GPU memory，CPU 负责推进描述和状态。
CPU 传给插件的是地址、长度、tag、MR handle，而数据载荷由 DMA 搬运。
采用 host staging 时，载荷经过主机缓冲，控制流程仍沿这套交接推进。
本例普通 NET 插件负责传输字节，当前片处于哪轮规约、该与哪段本地输入相加，
由 Ring 的设备工作决定。沿这条边查数值问题，应把 GPU 的规约位置也接回数据流。

NET 插件接口按职责排列即可理解：

- `regMr` / `regMrDmaBuf`：将内存范围变成插件可用的 MR handle。
- `isend` / `irecv`：尝试异步提交，交出非空 request 才算接受了请求。
- `test`：查询 request 是否完成，必要时返回实际传输字节数。
- `iflush`：为收到 GPU memory 的数据建立 GPU 可见性。
- `deregMr`：在使用结束后解除注册。

`isend/irecv` 成功返回但 request 为空时，progress 保持当前计数，下一轮再试。
一个 request 可以代表一组接收；`maxRecvs` 和共同的 recvComm 决定接收聚合。
插件实现从这个契约接入，构建与验证流程放在 [10](../10-development-workflow.md)。

## 4. 一片数据经过四个控制状态

先定义教学标签。`S` 指发送端的 send connector，`R` 指接收端的 recv connector：

- `S_tail = *send->conn.tail`：发送 GPU 发布到哪里；通常映射发送资源的 `recvMem->tail`。
- `S_head = *send->conn.head`：send proxy 归还到哪里；通常映射发送资源的 `sendMem->head`。
- `R_tail = *recv->conn.tail`：recv proxy 发布到哪里；通常映射接收资源的 `recvMem->tail`。
- `R_head = *recv->conn.head`：接收 GPU 消费到哪里；映射接收资源的 `sendMem->head`。

这些名字对应两套 conn、四个状态字。
启用 GDRCopy 同步映射时，CPU 写的 `S_head/R_tail` 改指向 `gdcSync` 对应区域，职责相同。

### 发送 GPU → send proxy → NIC

发送 GPU 的 `waitPeer` 先检查 `S_head + NCCL_STEPS`，确认有可写空间。
它选择 `step % NCCL_STEPS` 对应的内部槽，记录 FIFO 中的字节数，再由搬运线程写入本片数据。
完成数据写入及发布所需的 fence 后，`postPeer` 推进 `S_tail`。

send proxy 先用 `posted` 管理允许推进的深度，随后检查两项：
该槽的 `connFifo.size` 已有效，且 `S_tail > base + transmitted`。
满足后以槽地址、实际长度和 MR 调用 `isend`。
取得 request，才把 `transmitted` 增加一个 `sliceSteps`。

NIC 读取发送槽并发往对端。
send proxy 的 `test` 完成后，先将 FIFO size 复位为 `-1`，
再增加 `done`，写入 `S_head = base + done`，发送 GPU 因而获得复用 credit。
这是发送缓冲的回收边界。

### NIC → recv proxy → 接收 GPU

接收侧可以提前行动：recv proxy 在空闲接收槽上提交 `irecv`，
取得 request 后推进 `posted`。这与发送侧等待 GPU 产出数据可以重叠。
NIC 随后将数据写进这个接收槽。

接收 request 的 `test` 完成后，recv proxy 推进 `received`。
对于需要 flush 的 GDR/Simple 接收，它执行 GDRCopy flush 或插件 `iflush`；
若产生 flush request，继续用 `test` 等它完成。

可见性就绪后，recv proxy 推进 `transmitted`，
写入 `R_tail = base + transmitted`，把本片交给接收 GPU。
接收 GPU 的 `waitPeer` 看到 tail 到达目标，才读取、规约或转发数据。

GPU 完成本片消费后，用 `postPeer` 更新 `R_head`。
recv proxy 观察到 `R_head > base + done`，才推进接收 sub 的 `done`，
按需调用 `irecvConsumed`，并允许该接收槽再次用于网络写入。
发送侧归还的是网卡已读完的源槽，接收侧归还的是 GPU 已用完的目标槽。
两端的回收依据各自消费者，发送完成与远端消费可以在时间线上分开出现。

因此，接收的三个交接点依次是：**网络完成 → GPU 可读 → GPU 已消费**。
网络 `test` 回答第一项；最终槽位回收看的是 `R_head`。

## 5. 用同一种单位读 step

Simple 的 head/tail 是连接上的**绝对 step**。
sub 的 `posted/received/transmitted/done` 是相对本次 `base` 的 **step 数**；
每处理一片增加 `sliceSteps`，字节长度另放在 FIFO/request 中。

用对齐后的 `base=16`、`sliceSteps=2`、`NCCL_STEPS=8` 做纸面例子。
两端在这个例子中都从 16 开始，本片覆盖 `[16,18)`：

```text
发送侧：posted=2 → GPU 发布 S_tail=18 → transmitted=2
        send test 完成 → done=2 → S_head=18
接收侧：posted=2 → recv test 完成 → received=2
        可见性就绪 → transmitted=2 → R_tail=18
        GPU 消费 → R_head=18 → done=2
```

请求起始槽号为 `16 % 8 = 0`，下一片从绝对 step 18 开始。
比较 GPU 控制字时要加 base；计算窗口深度时比较相对计数。
head/tail 持续前进，循环使用的是缓冲槽下标。
绝对位置也把同一个物理槽的前后两轮使用区分开：看到旧一轮的完成值时，
新一轮仍会按自己的目标位置等待。协议用这个单调计数维持有限缓冲的循环流动。

## 6. 每个 rank 同时处于收与发两侧

把镜头移回整个 Ring：每个 rank 从前驱接收，并向后继发送。
一个 NET 发送函数等待 NIC 时，progress 会继续检查其他 active 工作；
接收侧可以预投递下一片，GPU 可以同时消费上一片。

```text
前驱 → recv proxy → 本 rank GPU → send proxy → 后继
         预收 k+1       处理 k       完成 k-1
```

若本 rank 的一侧是机内边，就由该侧 P2P/SHM 连接承担对应角色。
跨机的两个方向与多个 channel 共同形成双向 pipeline，
调试时按 `(rank, channel, direction, base)` 对齐状态最清楚。
观察进展快慢时，也按交接段拆开：发送发布进度反映 GPU 供数，网络请求反映传输，
接收头部进度反映 GPU 消费。哪一段长期落后，就从它的执行者和前置条件继续查。

## 7. 关键源码入口

以下位置按本地 NCCL `12df1a11` / `2.32.3` 核对：

1. **控制服务**：[src/proxy.cc:1789](../../nccl/src/proxy.cc#L1789) `ncclProxyService`；[1626](../../nccl/src/proxy.cc#L1626) `proxyProgressAsync` 分派控制回调。
2. **工作移交**：[src/proxy.cc:368](../../nccl/src/proxy.cc#L368) `ncclProxyOpToArgs`；[830](../../nccl/src/proxy.cc#L830) `ncclProxyGetPostedOps`、[951](../../nccl/src/proxy.cc#L951) `ncclProxyProgress`。
3. **发送状态机**：[src/transport/net.cc:1324](../../nccl/src/transport/net.cc#L1324) `sendProxyProgress`；[526–531](../../nccl/src/transport/net.cc#L526) 是 `sendConnect` 的 conn 映射。
4. **接收状态机**：[src/transport/net.cc:1493](../../nccl/src/transport/net.cc#L1493) `recvProxyProgress`；[603–609](../../nccl/src/transport/net.cc#L603) 是 `recvConnect` 的 conn 映射。
5. **GPU 交接**：[src/device/prims_simple.h:103](../../nccl/src/device/prims_simple.h#L103) `waitPeer`、[167](../../nccl/src/device/prims_simple.h#L167) `postPeer`，结合 `loadRecvConn/loadSendConn`。
6. **插件契约**：[src/include/plugin/net/net_v12.h:102–130](../../nccl/src/include/plugin/net/net_v12.h#L102) `ncclNet_v12_t` 的 MR、收发、完成与消费通知接口。

## 8. 离线任务：给一片数据标出交接证据

沿第 5 节的数例，在上述 send/recv progress 函数中各追一轮。
固定 Simple、内部 dedicated buffer，纸上记录执行条件、修改字段和下一执行者。
将每个 `test` 标成 send、recv 或 flush request，再连到对应 head/tail 写入处。

核对点：

- op pool 槽归还后，进度存放在 args/sub；连接仍然存在。
- send request 为空时 `transmitted` 保持原值。
- recv request 完成时先得到 `received=2`，GPU 可读还需经过可见性阶段。
- `R_head=18` 才让接收 `done=2`，计数与 base 的单位均为 step。

这项任务只需本地源码；本章未执行 GPU 或网络实验。

下一章：[内存与生命周期](../09-memory-and-lifetime.md)。
