# 03｜Device execution：让一份 work 在 GPU 上前进

[上一章](../02-host-execution.md) 已把请求变成 CUDA launch 参数和 GPU 可见的工作描述。
现在沿同一份 `ncclDevWorkColl`，追到一片数据的发送与接收。
源码基线为 NCCL **2.32.3**，`12df1a11`。

案例是两主机各四 GPU、八进程各一 rank，每 rank 64 MiB FP32 Sum AllReduce，
`count=16*1024*1024`。布局为教学假设，本章继续普通 host collective 的 **Ring/Simple 分支**。

## 1. Launch 参数怎样找到 channel

主机传入的参数缓冲包含 `ncclDevKernelArgs`、work batches，以及可能内嵌的 work。
`ncclDevKernelArgs` 保存设备 communicator 地址、`channelMask`、work 存储类型和地址。
生成的 kernel 包装函数接收 `ncclDevKernelArgs4K`，调用公共执行框架 `ncclKernelMain`。

```text
CUDA launch 参数
      |
ncclKernelMain
      | 按 channelMask 选择逻辑分区
      | 装载 communicator、channel、work batch
      v
RunWorkBatch -> RunWorkColl -> runRing -> Primitives
```

Channel 把集合工作分成可并行处理的数据区间，同时保存这份工作的通信邻居。
沿用上一章的四 channel 均分假设，每个 channel 负责 16 MiB，仍与全部八个 ranks 协作。
它把数据分工与 peer 连接放在一起，设备代码就能独立推进这一分区。

主机用 mask 的置位数作为 block 数，设备选第 `blockIdx.x` 个置位作为 channel ID。
例如活动 channel 为 `{1,4,7}`，三个 block 的映射是：

| `blockIdx.x` | 0 | 1 | 2 |
| --- | --- | --- | --- |
| channel ID | 1 | 4 | 7 |

同一次通用 kernel launch 中，一个 CTA 负责一个活动 channel。
CTA 在哪个 SM 上执行仍由 CUDA 调度；channel 记录的是工作分区和连接关系。

## 2. 装载执行上下文，再装载 work

本路径的设备 communicator 经 `comm->devComm` 传入，实际类型是 `ncclKernelComm`。
它保存 rank 数、协议缓冲大小、channel 地址和 abort 状态等 kernel 所需信息。
本版另有供 Device API 使用的 `ncclDevComm`，保存窗口、team/barrier 和 GIN 资源；
该接口见 [现代特性附录](modern-features.md)，本章沿前一种执行上下文追踪。

沿 Ring 的发送方向，取连接的路径是：

```text
ncclKernelComm
   -> ncclDevChannel[channelId]
        -> ring.next / ring.prev              算法邻居
        -> peers[peerRank]                    ncclDevChannelPeer
             -> send[connIndex] / recv[...]  ncclConnInfo
                  -> buffs[protocol]         payload 缓冲
                  -> head / tail / step      连接进度
```

Transport 在主机准备阶段建立这些地址，GPU primitive 按连接信息访问。
这样 `runRing` 可以只指定前驱、后继和数组偏移，具体地址由连接层提供。

`ncclKernelMain` 先把参数头复制到共享内存，再分工装载：
第一个 warp 搬 communicator，第二个 warp 搬当前 channel，其余线程搬首个 work batch。
CTA 同步后，所有执行线程看到同一份上下文。

Batch 中的 `workType` 决定描述尺寸，`funcId` 指定工作体，
`offsetBase + offsetBitset` 找到本 channel 应读取的 work。
描述来自参数空间，或 `workBuf` 指向的 FIFO/persistent storage，最后进入共享内存。
当前 batch 完成后，框架沿 `nextJump` 加载后续 batch。

这里出现的两种 FIFO 分工不同：

| 存储 | 谁生产、谁消费 | 存的是什么 |
| --- | --- | --- |
| work FIFO | 主机 planner → GPU kernel | 指针、分区、执行参数等描述 |
| 连接 payload FIFO | 当前连接生产者 → 消费者 | 通信中的数据片，由协议管理复用 |

装载 work 使 CTA 知道“接下来做什么”。
接收 payload 则要等该连接的数据就绪状态，二者有各自的容量和进度。

## 3. 从 work batch 到 Ring primitive

分派点在 `common.h:483–487`，有两种执行方式：

```cpp
if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId)
  SpecializedRunWorkBatch().run();
else
  ncclDevFuncTable[ncclShmem.funcId]();
```

命中 kernel 的专门化 ID 时直接执行，否则经设备函数表调用。
选中的 kernel 可以承载其他工作体，因此核对实际算法与协议要看 work 的 `funcId`。

`RunWorkBatch` 遍历共享内存里的 work，按每项的 `nWarps` 选择参与线程。
本例进入 `RunWorkColl<AllReduce, ..., Ring, Simple>`，再调用 `runRing`。
`ncclCollCbdPart` 从 `channelLo/Hi`、`countLo/Mid/Hi` 和 `chunkGrains*` 计算：
当前 channel 的起点、元素数量和每个 chunk 的元素数。

在四 channel 均分例子中，channel 2 取得从 32 MiB 开始的 16 MiB。
`runRing` 在这个区间内循环，每波安排 `nranks * chunkCount` 个元素，尾波调整块长。
因此设备可以用有限连接缓冲处理远大于缓冲容量的用户数组。

`Primitives` 接收前驱、后继、用户输入输出地址与规约参数。
`directSend`、`directRecvReduceDirectSend` 等方法说明本步需要收、规约、写出中的哪些动作。
Simple 根据连接 flags 和注册状态选择直接访问用户 buffer，或使用连接 FIFO。
接口名中的 Direct 表达能力，实际地址选择发生在 primitive 内部。

CTA 内还会分配等待、搬运和发布进度的角色。
等待线程维护上游就绪与下游信用，worker 线程执行数据读取、规约和写入，发布线程更新连接状态。
这些角色通过线程组同步交接。因此读设备代码时，要同时看角色标志和参与线程范围，
才能把单个线程里的等待循环还原为整组线程共同完成的一次操作。

## 4. Chunk、slice、step：把一块数据变成流水

三个粒度分别解决算法分工、搬运流水和缓冲复用：

| 粒度 | 在 Ring/Simple 中的作用 |
| --- | --- |
| chunk | `runRing` 一次交给 primitive 的逻辑数据块 |
| slice | primitive 内一次等待、处理、发布的数据片 |
| step | 连接进度与缓冲容量的计数单位 |

本版 `NCCL_STEPS=8`，普通连接 FIFO 按八个 step 槽循环使用。
逻辑 step 持续递增，物理起始槽位取 `step % 8`。
Ring AllReduce 的常量及实例化得到：

```text
ALLREDUCE_CHUNKSTEPS = 4
ALLREDUCE_SLICESTEPS = 2
ProtoSimple<SlicePerChunk=2, StepPerSlice=2>

一 chunk -> 两个 slice -> 共推进四个 steps
```

例如取 4 MiB 的普通 Simple FIFO，每个 step 的基础空间为 512 KiB，
一个 slice 使用两 steps 配额，八槽窗口可容纳四个这样的 slice。
实际有效数据量还受 chunk 大小和尾片影响，配额与有效字节数分别计算。
`genericOp` 对剩余空 slice 也推进状态，让两端继续使用相同的 step 序列。

发送者可以在接收者处理前一片时准备后一片。
每条连接独立检查空间和数据是否到齐，流水无需每传一片就在全环做 barrier。
某一跳变慢时，已占槽位逐渐填满，上游便在 credit 检查处等待。

## 5. 一次发送—接收：等空间、等数据，再读写

先取 Simple 的普通 FIFO 路径，讨论一条生产者—消费者连接。
`loadSendConn` 把 WaitSend 绑定到 `head`，PostSend 绑定到 `tail`；
`loadRecvConn` 把 WaitRecv 绑定到 `tail`，PostRecv 绑定到 `head`。
于是 head 表示已消费、可复用的进度，tail 表示已生产、可读取的进度。

`waitPeer` 中的条件可以写成：

```text
发送方取得空间：head + 8 >= sendStep + StepPerSlice
接收方取得数据：tail     >= recvStep + StepPerSlice
```

从双方 step、head、tail 都为 0 开始，追一个非空 slice：

1. 发送方检查 `0+8>=0+2`，取得槽位 0 起的两 steps 配额。
2. Worker 写入 payload；线程组 barrier 汇合本片的数据写入。
3. PostSend 按条件执行系统范围 fence，再发布 `tail=2`。
4. 接收方观察到 `tail>=2`，读取本片，完成复制或规约。
5. 接收线程组确认消费结束，PostRecv 发布 `head=2`，返还配额。

Head 保护旧数据免于过早覆盖；tail 保护新数据免于过早读取。
例如发送方已经推进到 step 8，而 head 仍为 0，下一片要求 `head>=2`。
它必须等消费者释放第一片，才能重新使用槽位 0 起的空间。

可见性由这一整条顺序保证。
`genericOp` 先完成数据处理和 barrier，`postPeer` 再做所需的 `fence_acq_rel_sys()`，
最后用 `st_relaxed_sys_global` 发布进度。
普通进度轮询使用 volatile PTX load，数据读取也在 `reduceCopyPacks` 中配套使用 volatile load，
避免观察到新进度后仍从陈旧缓存取 payload。
本地 barrier 汇合 CTA 线程，跨端依赖则由进度发布与轮询完成。

Ring 中间节点还要同时满足两个条件：上游已经提供输入，下游仍有输出空间。
`genericOp` 等待相应连接后建立源、目的指针，再调用规约复制代码。
例如收到 `[10,20]`、本地输入为 `[1,2]`，worker 在寄存器中得到 `[11,22]`，写往下游；
需要保留最终结果的步骤还会写本地输出。两条连接随后分别发布消费与生产进度。
这使背压自然穿过中间节点：下游空间不足，当前片就暂缓处理，上游的槽位也延后释放。
排查等待时，从当前线程所读的进度指针找到发布者，比只观察某次访存耗时更能定位停点。

连接 step 在构造时按 chunk 配额对齐，结束时写回 `conn->step`，供下一项工作续用。
对普通 NET，这种交接分别发生在 GPU 与发送 proxy、接收 proxy 与 GPU 之间；
两端的四个具体进度字段与网络完成条件留到 [08｜Proxy 与网络](../08-proxy-and-network.md)。

同样的算法还可采用其他协议，主要区别在数据布局与就绪通知：

| 协议 | payload 与就绪信息 | 主要取舍 |
| --- | --- | --- |
| Simple | 连续 payload，head/tail 等控制信息分离 | 大片搬运易摊薄控制开销 |
| LL | 每 16 B line 含 8 B 数据及两个 flag | 细粒度就绪，布局数据占比 50% |
| LL128 | 每 128 B line 含 120 B 数据及一个 flag | 降低标志占比，增加 warp 协作要求 |

表中的占比是存储布局；实际性能还取决于等待、规约和资源占用。
协议候选由平台支持与调优共同筛选，Ring/Simple 只是本章选择跟踪的一条执行路径。

## 6. 源码入口与离线推演

- [common.h:420](../../nccl/src/device/common.h#L420)：`ncclKernelMain`，channel 映射、装载和两路分派；142 行为 `loadWorkBatchToShmem`。
- [device.h:430、476](../../nccl/src/include/device.h#L430)：`ncclDevChannel`、`ncclKernelComm`；136 行为连接信息，514 行为 launch 参数。
- [device.h:287、337](../../nccl/src/include/device.h#L287)：`ncclDevWorkColl`、`ncclCollCbdPart`，恢复数据分区。
- [all_reduce.h:14、229](../../nccl/src/device/all_reduce.h#L14)：`runRing` 与 Ring/Simple 实例化。
- [prims_simple.h:103、167](../../nccl/src/device/prims_simple.h#L103)：`waitPeer`、`postPeer`；475、529 行绑定接收与发送角色。
- [common_kernel.h:40](../../nccl/src/device/common_kernel.h#L40)：`reduceCopyPacks`；97–119 行核对配套的 payload 读取。

**离线任务：给一个 CTA 恢复工作，再推进一个满窗口。**
令 `channelMask=0b10010010`，求 block 1 的 channel。
再取同一 FIFO 的快照：`head=0, tail=8, sendStep=8, recvStep=0`，每 slice 两 steps。
按第 5 节条件，推演接收一片后发送方能否复用空间。

可核对结论：block 1 选择 **channel 4**；发送方最初等待，接收方可消费第一片。
消费后 `head=2`，发送方得以使用槽位 0 起的配额，写完并发布 `tail=10`。
接收方的下一片仍从逻辑 step 2 开始，按序读取；八槽复用保留了数据的代际关系。

下一章 [04｜集合算法](../04-algorithms.md) 将这些 primitive 连成 Ring 的完整规约与结果传播。
