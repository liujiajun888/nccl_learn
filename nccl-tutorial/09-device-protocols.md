# 第 09 章：设备执行与协议——数据如何安全地流水前进

> 基线：NCCL `12df1a11`，版本 `2.32.3`。
> 本章以传统通用集合 kernel 为主，不把 symmetric kernel、Device API 或 CE 混成默认实现。
> 编写环境为无 GPU 的 macOS；下文源码推演不是 CUDA 实测。
> 前置：[主机提交](05-host-execution.md)、[传输](07-transports.md)、[集合算法](08-algorithms.md)。

## 9.1 学习目标

1. 从 `runRing` 跟到协议原语，解释 CTA 如何找到 channel、work 和数组范围。
2. 解释 FIFO 为何不被快发送者覆盖，接收者凭什么认为数据已可读。
3. 理解 Simple、LL、LL128 对启动成本、有效载荷比例和 GPU 资源的取舍。

**通信协议既安排搬数据，也管理“现在谁有权读写哪里”。**
只有地址，没有所有权和完成通知，两个 GPU 的并发 memcpy 仍然可能读到旧值。
CUDA/ROCm 的线程组、内存序思路相通，但本章的 PTX、warp 宽度和 multicast 条件不自动适用于 RCCL。

## 9.2 从一次 AllReduce 进入设备代码

源码阅读建议按下面的方向，而不是先钻入几千行模板：

```text
主机准备 work + channelMask + funcId
                 |
                 v
common.h::ncclKernelMain
   选 channel -> 搬 work 到 shared memory -> RunWorkBatch
                 |
                 v
all_reduce.h::RunWorkColl -> runRing / runTreeSplit
                 |
                 v
primitives.h::Primitives<T, RedOp, Fan, Direct, Proto, ...>
          /               |                \
 prims_simple.h       prims_ll.h       prims_ll128.h
          \               |                /
           数据读取 -> 规约/复制 -> 发布 -> 等待对端
```

从 [src/device/common.h:420，`ncclKernelMain`](../nccl/src/device/common.h#L420) 跟到 [all_reduce.h:14，`runRing`](../nccl/src/device/all_reduce.h#L14)，再看 [primitives.h:18–75、115](../nccl/src/device/primitives.h#L18)。
`common.h` 包含执行框架，`common_kernel.h` 包含规约/复制辅助实现，不要因文件名相近而看反。
`ncclKernelMain` 将参数复制到共享内存，避免某些取址方式使其溢出到线程局部栈。
它按 `channelMask` 第 `blockIdx.x` 个置位位置选择 channel；mask 启用 1、4、7 时，三个 block 对应 1、4、7。
第一个 warp 搬 communicator 信息，第二个搬 channel，其他线程加载 work batch，再同步发布。
随后运行专门化工作体或通过 `ncclDevFuncTable[funcId]` 分派，还可连续处理 batch。
故 kernel 名不一定完整反映每项工作的实际算法与协议。

## 9.3 block、warp、channel、chunk、slice、step 不在同一层

| 名称 | 本章中的含义 | 容易混淆的东西 |
|---|---|---|
| block / CTA | CUDA 调度和共享内存协作单元 | 不是一个固定 SM，也不是一条永久物理链路 |
| warp | 本 CUDA 实现中 32 线程的执行/协作组 | 不必对应一个 peer 或一份完整 chunk |
| channel | NCCL 的逻辑通信通道，含拓扑和 peer 连接 | 不是网卡端口，也不是 CUDA stream |
| chunk | 算法本轮交给 primitive 的逻辑数据块 | 不一定等于整次用户数组的 S/P |
| slice | primitive 内可独立流水推进的数据片 | 大小会受协议、尾块和参数影响 |
| step | 连接进度计数及 FIFO 配额单位 | 不是上一章全环的一轮，也不是 kernel 次数 |

本通用入口中，一个 CTA 在一次 launch 内映射到一个活动 channel；其驻留 SM 与调度时间仍由 CUDA 硬件决定。
一个 channel 可含树的多个 peer，多个 channel 也可争用同一物理 NVLink/NIC；CTA 内又可拆分角色，故没有“一 warp 一链路”规则。
[src/include/device.h:337，`ncclCollCbdPart`](../nccl/src/include/device.h#L337) 计算 channel 的范围与 chunk 数，供 `runRing` 确定实际数组偏移。
大数组会循环处理多波 chunk，尾波还会重新对齐 chunkCount；因此 S/P 只是前章的等分单波模型。

### 一个单位换算例子

[src/include/device.h:26](../nccl/src/include/device.h#L26) 定义 `NCCL_STEPS=8`。
[src/include/collectives.h:19–20](../nccl/src/include/collectives.h#L19) 给 Ring AllReduce 的 Simple 路径设定：

```text
ALLREDUCE_SLICESTEPS = 8/4 = 2
ALLREDUCE_CHUNKSTEPS = 8/2 = 4
ProtoSimple<SlicePerChunk=2, StepPerSlice=2>
```

对应实例化见 [src/device/all_reduce.h:229–232，`RunWorkColl`](../nccl/src/device/all_reduce.h#L229)。
假设 Simple FIFO 总空间为 4 MiB，则每 step 的基础空间为 512 KiB。
一个 slice 占用的进度配额为 2 steps，一个 chunk 为 4 steps，即两个 slice。
这不承诺每个 slice 都装满 1 MiB：小消息、尾部和 `genericOp` 的 sliceSize 计算会减小实际数据量。
空 slice 也可能需要推进协议状态，否则两端对后续 step 的解释会错位。

## 9.4 两种 FIFO：任务描述与连接数据不能混为一谈

主机提交的 work FIFO/参数空间保存“做什么”的描述，例如缓冲区、范围和 funcId。
连接数据 FIFO 保存传输中的数据或相关进度，服务于“对端已经写到了哪里”。
前者的回收不等于用户数据完成；后者的一次确认也不等于整个集合已经完成。

连接结构见 [src/include/device.h:136，`ncclConnInfo`](../nccl/src/include/device.h#L136)：
`buffs[proto]` 是协议缓冲，`head/tail` 是进度位置，`step` 保存本地连接进度。
此外还有 direct 指针交换、网络句柄及 `connFifo` 元数据，不能把它们都理解为 payload。
注释中的 local/remote 是相对于该连接端点而言；NET 的另一端可能是代理，而非另一 GPU 的直接内存。

```text
逻辑 step:  ... 6  7  8  9 10 11 ...      （持续增长）
物理槽位:      6  7  0  1  2  3 ...      （step % 8）

生产者 ---- payload + ready ----> 消费者
生产者 <--- consumed / credits -- 消费者
```

“槽位 0 可寻址”不等于“槽位 0 可重用”。
必须区分第 0 代和第 8 代的数据，step/flag 就是解决这个问题的代际信息。

## 9.5 Simple 的 head/tail 背压状态机

从常规 FIFO 路径看，发送者等待消费确认 head，接收者等待生产进度 tail。
源码 [src/device/prims_simple.h:475、529，`loadRecvConn/loadSendConn`](../nccl/src/device/prims_simple.h#L475) 明确绑定：

```text
WaitRecv -> conn->tail       PostRecv -> conn->head
WaitSend -> conn->head       PostSend -> conn->tail
```

不要只靠 head/tail 的英文直觉猜方向；应看当前角色具体把 `connStepPtr` 指向哪里。
[src/device/prims_simple.h:103，`waitPeer`](../nccl/src/device/prims_simple.h#L103) 中核心条件可译成：

```text
发送下一片：head + NCCL_STEPS >= step + StepPerSlice
接收下一片：tail              >= step + StepPerSlice
```

第一条保证发送者不超过 FIFO 容量；第二条保证接收者不读尚未发布的数据。
例如 `head=0, step=6, StepPerSlice=2`，发送可推进到 8，使用槽位 6 起的配额。
接着 `step=8` 想写下一片，需要 `head>=2`；否则它会覆盖接收者还未消费的槽位 0、1。
这就是背压：慢接收者反向限制快发送者，而不是假定“大家大约同时到达”。

同样，接收者 `step=4` 要消费两 steps，必须看到 `tail>=6`。
处理完后发布新的 head，允许发送者重用已消费的空间。
进度会在构造时按 chunk 配额对齐，在析构时写回 `conn->step`，不是每次调用都从零开始。
见 [src/device/prims_simple.h:485、787，连接加载与 `~Primitives`](../nccl/src/device/prims_simple.h#L787)。

## 9.6 有通知还不够：数据必须先可见

错误顺序非常直观：发送者先更新 tail，后写数据；接收者见 tail 已到便读出了旧值。
正确协议要保证**数据/元数据先发布，完成信息后发布**，并使用匹配的读取方式。

```text
发送侧                         接收侧
wait credits                   wait ready
写 payload                     确认可读后取 payload
线程组同步                     线程组确认本片已消费
系统范围 fence + 发布 tail      发布 head，返还 credits
```

[src/device/prims_simple.h:167，`postPeer`](../nccl/src/device/prims_simple.h#L167) 在需要时调用 `fence_acq_rel_sys()`，
随后用 `st_relaxed_sys_global` 写进度；第 278 行先做线程组 barrier，确保 worker 数据写入已完成。
这是一套配合使用的 fence、访存与同步协议，不能把最后那条 relaxed store 单独拿出来判断。

接收侧也不能抽象成“一个普通 C++ volatile 就解决了跨 GPU 内存序”。
[src/device/prims_simple.h:86，`loadStepValue`](../nccl/src/device/prims_simple.h#L86) 的普通路径用 volatile PTX load；
注释特别要求数据读取同样避开陈旧 L1 数据，NVLS min polling 则使用 acquire.sys 的 multimem 指令。
[src/device/common_kernel.h:94–99、118，`reduceCopyPacks`](../nccl/src/device/common_kernel.h#L94) 正是配套的 volatile 数据读取。
这些是针对特定 CUDA/PTX/传输假设的实现，不是可随意移植的通用 C++ 无锁队列证明。

CTA 内 `__syncthreads`、命名 barrier 或 `__syncwarp` 只协调参与的本地线程。
它们本身不等于跨 GPU barrier；跨 GPU 的等待依赖连接进度、flags、网络完成等机制。
不同原语组使用不同 barrier 编号和参与线程数，是为了允许规约与转发分组并行。
参见 [src/device/common.h:86，`barrier_sync`](../nccl/src/device/common.h#L86)。

## 9.7 Simple、LL、LL128 的数据布局

```text
Simple:
  FIFO payload: [连续用户数据........................]
  side metadata: head / tail / size / offset ...

LL，一条 16 B line:
  [data 4 B][flag 4 B][data 4 B][flag 4 B]
   ---------------- 8 B payload ----------------

LL128，一条 128 B line:
  [15 个 8 B 数据字 = 120 B][1 个 8 B flag]
```

Simple 的 ready/credits 与 payload 分离，能把大片连续数据有效搬走。
“Simple”不等于 CUDA memcpy API，也不等于它在所有大小上都最简单或最快。

LL 将标志嵌在细粒度数据旁，接收线程可以边等对应 line、边尽早规约。
[src/include/device.h:77，`ncclLLFifoLine`](../nccl/src/include/device.h#L77) 定义了两个 data/flag 对。
每条 16 B 中只有 8 B 是用户数据，因此布局有效载荷率为 50%。
源码注释强调 flag 放在数据之后，并依赖 socket 连续接收或 IB/RDMA 的相应 8 B 写入原子性假设。
不能把这段话改写成“任意设备上的 16 B 写总是原子的”。

LL128 减少标志占比，由 warp 协作搬运更大的批次。
[src/include/device.h:113–115](../nccl/src/include/device.h#L113) 定义 line 为 128 B、16 个 uint64、15 个数据字。
所以典型布局效率严格是 `120/128=93.75%`；不是 128 个线程，也不是承诺链路一定达到 93.75% 峰值。
实际还有等待、对齐、尾片和规约成本，调优模型可能使用不同经验折扣。

## 9.8 LL 的“数据到齐”与“可以重用”分别怎么判断

[src/device/prims_ll.h:108，`readLL`](../nccl/src/device/prims_ll.h#L108) 反复加载 data/flag，
只有两份 flag 都等于 `NCCL_LL_FLAG(recvStep+1)`，才把该 line 当作当前数据。
这里的检查防止半条数据到达或残留上一代内容被误读。
发送侧 `storeLL` 用配套的向量 volatile store，见同文件第 154 行。

这不意味着 LL 不需要流控：第 73 行 `waitSend` 仍检查 head 与 `NCCL_STEPS`。
行内 flag 解决“这一代数据准备好了没有”，head 解决“旧数据消费完、能覆盖了吗”。
二者是不同的安全条件，不能相互替代。

由于 LL flag 有有限位宽，数值最终会循环。
[src/device/prims_ll.h:99，`incSend`](../nccl/src/device/prims_ll.h#L99) 包含 cleanup，
在指定周期写完整片的 flags，避免空闲区域遗留的标志在绕回后被误认成新数据。
这也是为什么“每次消息很小，不用碰其他位置”不总成立。

## 9.9 LL128 的 warp 协作与默认适用条件

[src/device/prims_ll128.h:184，`recvReduceSendCopy`](../nccl/src/device/prims_ll128.h#L184) 由 flag thread 检查标志，
通过 `__any_sync` 让 warp 判断是否仍需重载，然后继续读取和规约。
发送端将 flag 放进指定字位置，见同文件第 270–287 行。
不要把 128 B line 当成“跨任意链路保证原子的 128 B 事务”；它是一套有平台前提的协议布局。

对齐数据可直接装入寄存器；非 16 B 对齐情况可能借助 shared-memory scratch 调整布局。
证据：[src/device/prims_ll128.h:99、154，`loadRegsBegin/storeRegs`](../nccl/src/device/prims_ll128.h#L99)。
第 297 行 `GenericOp` 负责循环、等待发送空间、最后推进 step 及发布进度。
LL128 对代理需要的 tail 通知还包含架构相关 fence，见同文件第 87 行 `postSend`。

默认启用判断在 [src/tuning/cost_model.cc:119，`isLL128Enabled`](../nccl/src/tuning/cost_model.cc#L119)：
它分别检查 interType、intraType、计算能力是否兼容，并含驱动/CUDA 特定例外。
部分条件允许 interType 到 PXB；Hopper 及更新架构在相应条件下允许到 PXN。
intraType 另有 `<= PATH_NVB` 限制，所以不能把前一句误解成“任意纯 PCIe GPU 拓扑都会默认启用”。
同样，也不能说“只要经过 PCIe，LL128 一律禁用”。
强制协议不提供新的硬件保证；默认门控在同文件第 364 行接入，实验前应核对完整条件。

## 9.10 规约、forward、最终写为什么融合

考虑 Ring 中一片数据：收到 `[10,20]`，本地同片为 `[1,2]`，下一跳需要 `[11,22]`。
朴素安排是接收 kernel、规约 kernel、发送 kernel，伴随额外启动与中间全局内存写回。
NCCL 原语可以一次读入，在寄存器中规约，再写下一跳和必要的本地输出。

```text
上游数据 ----\
             +--> 寄存器 applyReduce --> 下游连接/直接目的缓冲
本地输入 ----/                      \--> 本地输出（若这一阶段需要）
```

[src/device/common_kernel.h:40，`reduceCopyPacks`](../nccl/src/device/common_kernel.h#L40) 展示多源读取、
`applyReduce`、可选 `applyPostOp` 和多目的写入；这不是一次只允许“copy 或 reduce”二选一。
[src/device/all_reduce.h:55](../nccl/src/device/all_reduce.h#L55) 的中间步用 `directRecvReduceDirectSend`；
第 64 行用 `directRecvReduceCopyDirectSend` 完成最后规约、保存完整块并开始转发。
AG 阶段则 `directRecvCopyDirectSend`，不再把本地输入重复加一遍。
后处理只应出现在语义合适的位置，例如最后结果上的除法，而不是每跳都做一次。

方法名带 `Direct` 也不保证每条连接都零拷贝。
[src/device/primitives.h:119，`PrimitivesWithoutDirect`](../nccl/src/device/primitives.h#L119) 把 LL/LL128 的 direct 名称映射到普通原语。
Simple 才会结合连接 flags、注册状态与指针交换决定直接读写或退回 FIFO。
因此从算法调用名推断物理路径，比从源码实际分支推断弱得多。

## 9.11 pipeline：不是一轮结束后全体 barrier

算法示意图画整轮，是为便于证明；实现可以让 slice A 已向下游转发时，slice B 还在上游等待。

```text
时间 ---->
GPU0:  send A     send B     send C
GPU1:     reduce A  reduce B  reduce C
GPU2:        reduce A  reduce B  reduce C
                     ^ 每条边由自己的 credits 约束
```

每级收到足够数据便前进，不必等待整条用户数组完成。
慢的一跳会耗尽上游 credits，背压逐步传播，直到所有缓冲使用仍在安全容量内。
增加缓冲或 chunk 大小只能改变可在途数据量，不能消灭真正的慢链路。

NET 路径上，GPU、CPU proxy 和 NIC 可以共同推进不同阶段。
[src/transport/net.cc:1437](../nccl/src/transport/net.cc#L1437) 是发送代理调用网络 `isend` 的位置；
接收侧 [第 1707–1734 行](../nccl/src/transport/net.cc#L1707) 展示条件式 flush 与发布接收进度。
GPU 等待时间变长可能是网络/代理进展受阻，不一定是 GPU 规约指令太慢。

## 9.12 选择协议也是选择资源占用

Simple 适合摊薄大片数据的控制成本；LL 常以标志流量换小消息更早启动；LL128 改善数据比例，但有 warp 协作、寄存器与 scratch 成本。
这是因果方向，不是三个固定消息阈值，也不是“LL128 永远介于另外两者之间”。
增加 CTA/channel 可提高访存和链路利用率，却也占用 SM 调度槽、寄存器和带宽；挤慢 GEMM 可能使整步时间反增。
线程阈值见 [src/tuning/tuning_general.cc:101–109](../nccl/src/tuning/tuning_general.cc#L101)，Tree 拆分见 [all_reduce.h:155–164，`runTreeSplit`](../nccl/src/device/all_reduce.h#L155)。
调优应比较通信与计算重叠后的关键路径，不只比较独立通信 kernel 的吞吐。

## 9.13 可观察验证问题

以下是有 GPU 环境的验证方案，本机没有执行；步骤衔接[测试](10-nccl-tests.md)与[性能分析](11-performance.md)。

- 固定 Ring 和输入规模，比较合法协议：小消息延迟与大消息有效带宽是否符合控制/标志成本的解释？
- 记录实际 tuning、channel 和线程数；是否误把代表 kernel 名中的 LL 当成实际协议？
- 在受控实验中让一个 rank 晚提交，其他 rank 的等待是否延长？这是进度依赖，不是“自动丢包”。
- 改变 CTA 上限并同时运行计算 kernel：通信单测提升是否转换成训练迭代时间提升？
- 源码断点/插桩时分别看 `step`、`head`、`tail`，能否验证 `head+8` 限制了生产者前进？

在设备自旋循环加 printf 会严重扰动时序，不能把它测出的时间当作真实性能。
有限时间等待和异步错误检查也应保留；背压保障无覆盖，并不保障失效 peer 最终一定前进。

## 9.14 自测与答案

**题 1：NCCL_STEPS=8，发送 step=8、StepPerSlice=2、head=1，可否继续？**
答：不可以，`1+8<8+2`；至少要 head 达到 2，才能安全重用对应空间。

**题 2：LL128 的 120/128 是什么比例？能否据此承诺带宽达到硬件峰值的 93.75%？**
答：它是 line 布局的数据占比；不能承诺实际带宽，等待、对齐、规约与资源争用仍有成本。

**题 3：为何不能只用 __syncthreads，或者只看 directSend 的名字来判断传输正确性？**
答：CTA barrier 不完成跨 GPU 发布；还需协议的可见性和流控。Direct 是原语接口能力，是否走直接缓冲取决于协议和连接条件。
