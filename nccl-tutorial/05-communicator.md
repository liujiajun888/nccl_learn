# 05｜Communicator：把一组 GPU 组织成可执行的通信团队

> 源码基线：`12df1a11`；`makefiles/version.mk` 为 NCCL **2.32.3**。行号均对应此提交。

**阅读定位：机制解释 → 源码追踪 → 验证实验的主线章节。** 面向已有 CUDA 基础和 UMD 开发经验、初学 NCCL 的读者，直接从分布式初始化进入。本章第 1～6 节回答“通信前怎样建立共同关系”，第 7～9 节的非阻塞、重组与版本分支按需选读；[示例带练](examples/README.md)是可选验证材料，不是前置课程。

## 学习目标

读完本章，你应该能够：

- 分清 `uniqueId`、rank、本地 CUDA device、node 与 local rank。
- 用“身份交换 → 拓扑协商 → 连接准备”解释初始化，而不是把它当作一次设备分配。
- 区分 communicator 可用、主机异步任务完成、GPU 通信完成三个时刻。
- 正确安排 split、shrink、finalize、destroy 与 abort 的生命周期。

承接 [04 的算法核心（4.2～4.7）](04-algorithms.md) 与 [03 的 group 与完成边界](03-cuda-semantics.md)，沿用 [01 的 rank/comm 模型](01-mental-model.md)。
本章只回答“谁与谁组成团队、团队何时可用”；下一站是 [06｜拓扑](06-topology.md)，传输与实际提交分别见 [07｜Transports](07-transports.md) 与 [08｜Host execution](08-host-execution.md)。
[14｜源码地图](14-source-map.md)与 [15｜练习](15-exercises.md)贯穿阅读；[00｜环境](00-environment.md)与 [examples](examples/README.md)并行速查，不要求重走逐行带练。

## 从 UMD 经验切入：本地 handle 怎样成为团队的一员

本节回答一个问题：为什么 rank 0 能独立运行 CUDA kernel，却不能完成这个 communicator 的初始化？

下节八 rank 案例中，A 上 rank 0 已选好 GPU，B 上 rank 7 却尚未进入初始化。可以借用本地 handle“持有资源、管理状态与生命周期”的直觉，但 `ncclComm_t` 保存的是分布式团队在本 rank 的状态。**初始化因果链**是：成员进入同一初始化 → 交换身份与联系信息 → 建立团队及本阶段所需资源 → 发布可用状态。

rank 7 缺席时，本地对象已分配也无法替代它的参与；等它加入并完成所需交换，团队关系才能继续建立。状态由谁发布、供谁消费，以及初始化与首轮连接各需要什么证据，见第 3、6 节和章末任务。

**注意：**`ncclComm_t` 不是 CUDA context，也不是驱动 handle 的跨进程副本。

## 1. 贯穿案例：两台机器上的八个训练进程

设主机 A、B 各有四张 GPU，每个进程管理一张 GPU。每轮训练要对 **64 MiB FP32 梯度**执行 Sum AllReduce，即每个 rank 的 `count = 16 * 1024 * 1024`。第 05～08 章沿用此案例；机器布局是教学假设，不是实测配置。

| 身份 | 主机 A | 主机 B | 谁决定它 |
| --- | --- | --- | --- |
| communicator rank | 0、1、2、3 | 4、5、6、7 | 应用或启动器传入 |
| 物理 GPU | A0、A1、A2、A3 | B0、B1、B2、B3 | 应用放置策略 |
| CUDA ordinal | 若全部可见，通常为 0～3 | 同样通常为 0～3 | 当前进程的可见设备集合 |
| 常规 local rank | 0、1、2、3 | 0、1、2、3 | NCCL 根据成员与节点关系建立映射 |

如果启动器给每个进程只暴露一张卡，八个进程都可能调用 `cudaSetDevice(0)`。这不意味着八个 rank 使用同一张物理 GPU；要结合 PCI bus ID 或 UUID 才能核实身份。反过来，rank 5 也绝不意味着 `cudaSetDevice(5)`。

```text
启动器：分配 global rank、分发引导令牌、设置 GPU 可见性
                  |
        +---------+---------+
        |                   |
 主机 A：r0 r1 r2 r3   主机 B：r4 r5 r6 r7
         |  |  |  |           |  |  |  |
 GPU：  A0 A1 A2 A3          B0 B1 B2 B3
         \______ 各自持有本地 ncclComm_t ______/
                  同一逻辑 communicator
```

一个 communicator 是**分布式一致关系加各 rank 的本地状态**，不是一块大家共同解引用的对象。各进程中的 `ncclComm_t` 地址通常不同，内部分别保存自己的设备、peer 信息、channels、连接及错误状态。成员数量与 rank 编号一致只是必要条件；参与者还必须进入同一初始化，并按兼容顺序调用后续通信。

## 2. uniqueId 是 rendezvous token，不是共享 communicator

**uniqueId** 是初次会合用的令牌（rendezvous token）：某个参与者调用 `ncclGetUniqueId`，应用借助 MPI、启动器或其他控制通道广播其字节。所有成员用同一个 ID、相同 `nranks`、互不重复的 rank 调用初始化。NCCL 不替应用完成这次最初的 ID 分发。

ID 为初次会合提供联系信息与匹配标识。本版本内部 `ncclBootstrapHandle` 包含 `addr`、`magic`、`nRanks`，但用户应把 `ncclUniqueId` 视为不透明数据：不要依赖这些字段布局，也不要把 ID 当成 GPU buffer 地址、共享内存句柄或已完成的通信连接。

这解释了一个常见困惑：**分发 ID 成功，并不意味着 NCCL 初始化成功。** MPI 使用的网络、NCCL bootstrap 选择的接口、后续 NET 使用的 HCA（RDMA 网卡，host channel adapter），可能不是同一条路径。若 B 上的进程无法联系引导端点，GPU 再快也无法加入团队。

源码入口：[bootstrap.h](../nccl/src/include/bootstrap.h)，`nccl/src/include/bootstrap.h:14`，`ncclBootstrapHandle`。
ID 构造：[bootstrap.cc](../nccl/src/bootstrap.cc)，`nccl/src/bootstrap.cc:433`，`bootstrapGetUniqueId`。

## 3. 从 API 走入初始化任务

普通 `ncclCommInitRank` 获取当前 CUDA device，再进入 `ncclCommInitRankDev`。因此应用应在调用前选好设备，不要期待 NCCL 根据 global rank 自动选卡。

`ncclCommInitRankDev` 先检查 rank 范围，创建本地 comm 与 abort flags，解析配置。它将 `initState` 置为 `ncclInProgress`，并把 ID 拷贝到初始化 job。这份拷贝有两个作用：异步任务不再依赖调用者栈上的 ID，同时满足内部对象的对齐要求。本地检查与对象准备成功后，初始化任务持有 `job->comm` 继续执行；它仍依赖启动器提供一致的成员集合、正确的设备放置与同一会合令牌。返回一个非空 handle 的时刻，可能早于团队真正可通信的时刻。

```text
ncclCommInitRank / ncclCommInitRankConfig
   |
   v
ncclCommInitRankDev  ---- 创建 comm、复制 ID、提交 job
   |
   v
ncclCommInitRankFunc ---- cudaSetDevice、设备能力/内核准备
   |
   +--> commAlloc --> bootstrapInit
   |
   +--> initTransportsRank
          peerInfo --> topo/paths --> ring/tree --> channels/connectors
   |
   v
initState = ncclSuccess   （或错误状态）
```

只有完成本分支要求的初始化，任务才发布 `initState = ncclSuccess`。

**注意：**
- `Init COMPLETE` 是该 rank 的初始化完成证据，不是梯度完成证据。
- 非阻塞 API 的后续使用，还须等第 7 节查询到主机异步状态成功，不能只看 handle 或这条初始化日志。
- 初始化成功也不等于所有连接就绪，按需连接的边界见第 6 节。

锚点：[init.cc](../nccl/src/init.cc)，`nccl/src/init.cc:2851`，`ncclCommInitRankDev`；
`nccl/src/init.cc:2105`，`ncclCommInitRankFunc`；`nccl/src/init.cc:2207`，成功状态发布。
阅读时先追踪 `job->comm` 和 `comm->bootstrap`，比从头背所有初始化字段更有效。

## 4. bootstrap 建的是控制关系，不是梯度传输环

`bootstrapInit` 创建会合与监听状态，让 ranks 获取相邻成员的联系信息，连成 bootstrap ring。随后各成员交换 peer 联系地址、proxy 地址等元数据，支撑后面的 all-gather 与点对点控制消息。初始化任务必须等所需交换完成，才能用这些信息建立后续资源关系。rank 7 缺席或会合网络不可达，都不能靠“本地 comm 已分配”来绕过。

**注意：**
- 引导 root 是会合协调者，不是每轮 AllReduce 汇聚全部梯度的参数服务器。
- 这里交换的地址、身份和连接元数据，不是 64 MiB 梯度 payload。

常规分支中的 bootstrap ring 用 socket 传递初始化信息。**它的 rank 次序不等于最终数据 ring 的次序。** 最终数据 ring 要等拓扑发现与图搜索之后才能生成，而且不同 channel 可以使用不同次序。

```text
控制面：ID -> 会合端点 -> bootstrap ring -> 身份/拓扑/连接句柄
                                               |
                                               v
数据面：用户 GPU buffer -> 算法 + channel + transport -> 对端 GPU
```

控制面会参与建立数据面的资源，但两者承载的内容与性能目标不同。看到 bootstrap socket 只能证明控制关系用了 socket，不能据此判定 64 MiB 梯度必走 TCP。本版还存在 NET 辅助 bootstrap 的分支，统一在第 9 节说明。

源码：[bootstrap.cc](../nccl/src/bootstrap.cc)，`nccl/src/bootstrap.cc:754`，`bootstrapInit`；
`nccl/src/bootstrap.cc:885`，`bootstrapInit` 的 ring 建链；`nccl/src/bootstrap.cc:935`，`ringAllInfo` 调用。

## 5. 从“大家是谁”到“我该连接谁”

初始化交换的 `peerInfo` 包含主机/进程身份、设备身份、版本与能力信息。NCCL 据此判断同进程、同主机、设备能力是否兼容，以及哪些资源可以共享。比如同一主机上两个进程误选同一 GPU，不能靠不同 rank 编号掩盖。

接下来发生四次重要的状态交接：

| 生产者 → 消费者 | 交接数据 | 为什么必须先得到它 |
| --- | --- | --- |
| bootstrap → 初始化协调 | `peerInfo`、成员联系信息 | 识别成员，比较能力，发现放置错误 |
| topology → graph search | 设备图、可行 paths、带宽与路径类型 | 算法邻接关系必须适配硬件 |
| graph search → channel setup | ring/tree 次序、节点入口/出口、channel 数 | 每个 rank 才知道自己的邻居 |
| transport setup → GPU 侧 comm | buffer 映射、head/tail、连接描述 | kernel 才能访问正确资源并推进协议 |

这些算法图不是用户的一次 AllReduce 任务；它们是可被多轮任务复用的通信骨架。

<details><summary>深入：初始化各阶段在 init.cc 中的顺序</summary>

实际顺序可在 [init.cc](../nccl/src/init.cc) 对照：

- `nccl/src/init.cc:1391` 调 `ncclTopoGetSystem`，随后计算 paths、裁剪并重算。
- `nccl/src/init.cc:1428` 与 `nccl/src/init.cc:1436` 调 `ncclTopoCompute`，分别生成 ring 和 tree 图。
- 之后交换图摘要、建立 rank/node/local-rank 映射，并拼接跨节点关系。

</details>

在常规主机配置下，node 通常对应同主机成员集合，local rank 是集合内编号。`intraRank` 则是同进程多 GPU 成员的编号，不能与 local rank 混用。本例每进程一 GPU，所以每个进程的 intra-process 成员数是 1，但每台主机有 4 个 local ranks。

## 6. Init COMPLETE 不等于所有潜在连接都已经建好

假设本例前十轮都使用 Ring，后来某种大小的任务选择 Tree。若初始化必须提前为所有算法、所有 peer 建完全部资源，就可能浪费显存、连接对象和启动时间。因此 NCCL 允许把部分工作延迟到真正需要时再做。

本版 `comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect()`。启用该分支时，初始化先建立 channel 结构，并处理 NVLS/CollNet 的相应 setup，不会像 eager 分支那样直接执行全部列出的 Ring/Tree 连接工作。首次使用尚未连接的算法时，enqueue preparation 标记 `algoNeedConnect`，group 消费这个需求并执行连接 job。交给 launcher 的条件是所需连接成功完成，而不是“已标记”；连接仍依赖对端协作与设备/传输资源可访问，成本可能落在首轮而非初始化计时中。Send/Recv 的 peer preconnect 也有独立的按需路径；准备到发射的调用链见 [08｜Host execution](08-host-execution.md)。

因此应该区分三个状态：

- **拓扑邻接确定**：知道要与谁通信。
- **连接资源就绪**：对应 buffer、handle、transport 状态已经可用。
- **本次工作就绪**：工作描述已生成、连接已满足，能够提交。

**注意：**
- “本地 handle 已创建”“bootstrap 可达”“本次算法所需连接就绪”是三份不同证据。
- 引导成功后，若数据路径资源不可用，首轮仍可能停在连接准备，而非 GPU 传输。
- 第一次 AllReduce 慢，不一定是带宽差，也可能包含连接、注册或其他懒初始化成本。
- 但也不能反过来说首次慢必然就是 runtime connect；需要日志与 timeline 证据。

锚点：[init.cc](../nccl/src/init.cc)，`nccl/src/init.cc:1811`，`runtimeConn` 分支；
[enqueue.cc](../nccl/src/enqueue/enqueue.cc)，`nccl/src/enqueue/enqueue.cc:572`，`ncclPrepareTasks` 的连接需求标记。

## 7. 非阻塞生命周期：两种“异步”不要混在一起

`config.blocking = 0` 允许某些主机 API 的内部工作尚未完成就返回 `ncclInProgress`。这与正常 NCCL 调用把 GPU 工作异步提交到 CUDA stream，是两个不同层次。默认 blocking communicator 的 AllReduce 返回成功，也不表示梯度已经可供 CPU 读取。

```text
创建中 --state=success--> 可用 --提交通信--> GPU 工作进行中
  |                      |                 |
  |                      |                 +--stream/event--> 数据可消费
  |                      +--finalize--> 收尾中 --state=success--> 可 destroy
  +--错误-------------------------abort----------------------> 释放/失效
```

这是概念状态图，不是所有内部枚举的逐项转录。`ncclCommGetAsyncError(comm, &state)` 的返回值说明“查询本身是否成功”，`state` 才是被查询状态。当 `state == ncclInProgress` 时，不要向该 comm 继续提交普通通信；本版 `ncclCommEnsureReady` 会把这种提前使用报告为错误，而非自动替应用排队等待。

读者可将以下逻辑嵌入自己的非阻塞管理代码；生产代码还应加 deadline 与故障协调：

```cpp
ncclResult_t waitReady(ncclComm_t comm) {
  for (;;) {
    ncclResult_t state;
    ncclResult_t query = ncclCommGetAsyncError(comm, &state);
    if (query != ncclSuccess) return query;
    if (state != ncclInProgress) return state;
    std::this_thread::yield();  // 需包含 <thread>；不是 GPU 完成检查
  }
}
```

初始化/管理状态就绪后，数据完成仍应按 [03](03-cuda-semantics.md) 使用 stream/event 判断。
锚点：[init.cc](../nccl/src/init.cc)，`nccl/src/init.cc:493`，`ncclCommEnsureReady`。

## 8. 团队重组与退出：不要把所有 API 当作 free

**Split** 按 color 分组，按 key 确定新组内 rank；key 相同时使用父 rank 打破平局。父 communicator 的成员都要参与；`NCCL_SPLIT_NOCOLOR` 表示参与协商但不加入子组。本例可按主机分成两个四卡组，但新 comm 的 rank 已是新命名空间，不能继续拿父 rank 当索引。Split 产生新对象；即使配置允许共享底层资源，也不是原地修改父 comm。

**Shrink** 由保留成员以一致的排除列表建立缩小后的新 comm，被排除成员不参与新组初始化。例如排除父 rank 3、7，剩余六个成员的新 rank 会填补编号空洞。它不负责判断谁失联，也不负责恢复模型状态；故障检测和成员一致性仍由应用协调。`NCCL_SHRINK_ABORT` 会先终止父组进行中的工作，默认模式则不能当作故障逃生按钮。

**Finalize** 是停止提交后的正常收尾：等待已发工作与相关资源安静下来，而不是释放 handle 本身。本版在初始化成功且未 abort 的路径中，`commDestroySync` 停止 proxy 前还会执行 bootstrap host-local barrier：同一 communicator 的相关同主机 ranks 必须进入收尾，避免过早停 proxy 阻碍 PXN（先经本机 NVLink 到邻卡、再由其网卡出站的路径类型）建连。参与集合从 `localRanks` 按 `hostHash` 过滤，包含同主机的其他进程；它不是仅看当前进程，也不是整个跨主机 communicator 的全局屏障。

`ncclCommFinalize` 会把收尾加入 job。若一个 CPU 线程管理多个本地 ranks，逐个调用 blocking Finalize 可能卡死：第一次调用就要等同主机其他 ranks 进入收尾，而线程被阻塞在这次调用里，后续 rank 的 Finalize 再也没机会发起。解决办法有两种：用 `ncclGroupStart/End` 组织该线程负责的各 rank Finalize，让收尾任务一起推进；或者使用非阻塞模式，先发起所有相关 ranks 的 Finalize 再轮询，而不是发起一个就等一个。其他进程也须协调进入正常收尾。非阻塞 group 结束后仍要逐 comm 调用 `ncclCommGetAsyncError`，检查查询返回值与被查询状态，确认所有相关 comm 的主机收尾成功后再 destroy；`ncclInProgress` 不是完成。有 CUDA Graph 引用的 persistent plans 时，还要妥善结束图的使用并释放相关引用；收尾会等待这些引用消失，不能把一次 replay 完成当作图寿命结束。

**Destroy** 回收本地对象和剩余资源；未先 finalize 时，它可能承担等待/收尾工作，不保证立刻返回。但直接 Destroy 不等于逐个 blocking Finalize：`commReclaim` 在同进程成员中最后一次 Destroy/Abort 调用到达时，才为尚未 finalize 的成员安排并发收尾，使它们能够通过上述 barrier。因此现有单线程逐个 Destroy 的示例并不因这个 barrier 就必然死锁；它仍依赖相关同主机成员参与收尾，以及已发工作和图引用得到妥善处理。

**Abort** 设置 host/device abort flags 并进入回收，允许放弃未完成结果；它不是正常完成屏障。故障后不能只在一个进程 abort，然后期待其他 ranks 自动恢复；Destroy/Abort 后也不能继续复用旧 handle。

源码：[init.cc](../nccl/src/init.cc)，`nccl/src/init.cc:2046`，`commGetSplitInfo`；
`nccl/src/init.cc:3653`，`ncclCommShrink`；`nccl/src/init.cc:3877`，`ncclCommSplit`；`nccl/src/init.cc:3122`，`commDestroySync`（3155–3172 行为 host-local barrier）；
`nccl/src/init.cc:3206–3248`，`ncclCommFinalize` 的 job 提交；`nccl/src/init.cc:3260–3307`，`commReclaim` 的并发收尾；
`nccl/src/init.cc:3331`，`ncclCommDestroy`；`nccl/src/init.cc:3489`，`ncclCommAbort`。

## 9. 本版本的集中分歧

以下默认值与分支只属于本基线，第二遍再逐条核对：

<details><summary>深入：默认参数与版本分支（第二遍选读）</summary>

- 默认配置 `blocking = 1`，见 [init.cc](../nccl/src/init.cc) `nccl/src/init.cc:2764`，`parseCommConfig`。
- `NCCL_RUNTIME_CONNECT` 默认参数为 1，但实际还受 cuMem 支持约束，不能只看环境变量。
- `NCCL_OOB_NET_ENABLE` 默认 0，见 [bootstrap.cc](../nccl/src/bootstrap.cc) `nccl/src/bootstrap.cc:106`，`BootstrapNetEnable`；启用后的引导 all-gather 可用 NET，但根会合仍有 socket 逻辑。
- MNNVL（multi-node NVLink，跨节点 NVLink 互连）可使内部 locality/clique 跨物理主机，`localRanks` 不应永远解释为 OS 主机内 ranks；本章案例未启用它。
- 本版还有 Grow、Revoke 等管理能力，不能用传统 init/destroy 模型概括全部 API，扩展见 [13](13-advanced.md)。
- 源码是 CUDA NCCL；ROCm 开发者可迁移“身份—资源—状态”模型，不能直接照搬到 RCCL 的版本与驱动行为。

</details>

## 10. 实操观察：分开量初始化、首轮和稳态

以下仅供读者在 Linux/CUDA 环境执行；本教材未在本机运行 GPU 实验。先让训练程序打印 hostname、global rank、可见 CUDA ordinal 与 bus ID，再启用日志：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,GRAPH,NET \
NCCL_DEBUG_FILE=/tmp/nccl-init.%h.%p.log ./your_nccl_app
```

多机任务应通过现有启动器把这些环境变量传到各节点，不要以单进程命令代替 rank 启动。比较 `Init START/COMPLETE` 与初始化分阶段计时，再单独记录首轮、预热后多轮的 stream 完成时间。若卡在 bootstrap，先查会合地址可达性；若 init 完成而首次通信卡住，继续查 runtime connect 与成员调用匹配。不要把 host API 耗时直接当作 GPU 通信耗时；测量规范见 [10](10-nccl-tests.md)，排错见 [12](12-debugging.md)。

### 源码追踪/验证任务：找出初始化与首轮连接的交接点

只追普通初始化和首轮 Ring AllReduce；静态追踪不需要 GPU，动态观察复用已有应用与调试器，不需新增代码。每一跳记录：**生产者/执行线程 → 持有状态 → 消费者 → ready 条件 → 阶段完成证据 → 尚缺的外部依赖**。

1. 从 `nccl/src/init.cc:2851` 的 `ncclCommInitRankDev` 追 `comm` 与 `job->comm`；在 2893、2910 行标出初始状态和 ID 拷贝，说明非空 handle 还缺什么。
2. 进入 `nccl/src/init.cc:2105` 的 `ncclCommInitRankFunc`，定位 2190 行 `bootstrapInit`、2196 行 `initTransportsRank`；记录谁生产成员元数据、谁消费它建立资源关系，而非把它记成梯度传输。
3. 在 `nccl/src/init.cc:1811` 对照 `runtimeConn` 两支，列出 `setupChannel` 与 Ring/Tree connect 调用的差别；记录实际 `cuMemSupport` 和 `runtimeConn`，不要仅抄环境变量。
4. 在 `nccl/src/init.cc:2207` 找到 `initState` 成功发布。再对照：`ncclCommEnsureReady`（493 行）通过 `ncclCommGetAsyncError`（3934 行）查询的是 `asyncResult`。注意区分两种证据——初始化完成、主机可继续提交；它们都不是 GPU 完成。
5. 从 `nccl/src/enqueue/enqueue.cc:572` 追 `algoNeedConnect` 到 `nccl/src/group.cc:824–828` 的连接 job；再到 873 行 `doLaunches`，写明连接成功为何必须在发射之前。
   注意 `initAlgoChannels` 在标记需求时就会写入；连接完成要看 `asyncJobLaunch`（`nccl/src/group.cc:609`）的 job 结果/等待逻辑，不能只看该标记。
6. 可选 GPU 验证：固定 rank 布局、大小及 `NCCL_ALGO=Ring NCCL_PROTO=Simple`，分别以 `NCCL_RUNTIME_CONNECT=0`、`1` 启动两次全新运行，各 rank 配置一致。
   用断点核对第 3、5 步状态，结合上面的日志分别记录初始化、首轮和稳态；若 cuMem 条件不满足，应记录“未进入 runtime 分支”，不编造对照结果。
   用已有应用的 stream/event 完成检查与正确性校验补上首轮完成证据；只看 host 返回时间不足以证明数据可消费。

最终给出一条带源码位置的因果链：哪里发布 comm ready，哪里才满足首轮连接 ready，哪项观察证明本次 GPU 通信正常完成。

## 11. 自测与简答

1. 八个进程都打印 `cudaDev=0`，是否一定选卡错误？  
   **答：**不一定；ordinal 属于各进程可见集合，应对照主机、bus ID/UUID 和 rank 映射。
2. `Init COMPLETE` 后第一轮仍出现连接工作，是否违反初始化语义？  
   **答：**不违反；runtime connect 可延迟算法连接，但提交依赖它的工作前必须补齐。
3. 非阻塞 API 的异步状态变成成功，能否立即在 CPU 上读取梯度？  
   **答：**不能据此判断；还要确认对应 GPU stream/event 完成，并遵守内存访问条件。

下一站：[06｜拓扑](06-topology.md)，继续解释硬件可达关系如何约束团队的通信骨架。
