# 第 09 章：设备执行与协议——数据如何安全地流水前进

> 基线：NCCL `12df1a11`，版本 `2.32.3`。
> 本章以传统通用集合 kernel 为主，不把 symmetric kernel、Device API 或 CE 混成默认实现。
> 编写环境为无 GPU 的 macOS；下文源码推演不是 CUDA 实测。
> 本章承接 [08 主机提交](08-host-execution.md)，结合已读的 [07 传输](07-transports.md)与 [04 集合算法核心](04-algorithms.md)，进入设备执行。
> **推荐学习方式：机制 → 源码 → 实验。** 用熟悉的 producer/consumer 队列与 fence 理解连接 FIFO，再沿角色追源码。
> 基础数组说明可回查 [01](01-mental-model.md) / [02](02-collectives.md)，无需从 malloc/for 重学 GPU。

## 9.1 学习目标

1. 从 `runRing` 跟到协议原语，解释 CTA 如何找到 channel、work 和数组范围。
2. 解释 FIFO 为何不被快发送者覆盖，接收者凭什么认为数据已可读。
3. 理解 Simple、LL、LL128 对启动成本、有效载荷比例和 GPU 资源的取舍。

**通信协议既安排搬数据，也管理“现在谁有权读写哪里”。** 只有地址、没有所有权和完成通知，两个 GPU 并发 memcpy 仍可能读到旧值。CUDA/ROCm 的线程组和内存序思路可以借用，但本章的 PTX、warp 宽度和 multicast 条件不自动适用于 RCCL。

## 9.2 从一次 AllReduce 进入设备代码

本节回答：CTA 启动后，怎样找到自己要用的 channel、work 和数组范围？读源码按下面的方向走，不要先钻进几千行模板：

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

主线只需记住三步。第一，`ncclKernelMain` 按 `channelMask` 中第 `blockIdx.x` 个置位选择 channel：mask 启用 1、4、7 时，三个 block 分别对应 channel 1、4、7。第二，CTA 把 work batch 搬到共享内存，同步后发布。第三，按 `funcId` 选择执行路径：命中本 kernel 的专门化 id 时直接运行专门化工作体（如 `runRing`），否则经 `ncclDevFuncTable` 分派；一个 CTA 还可连续处理整个 batch。

入口锚点：[src/device/common.h:420，`ncclKernelMain`](../nccl/src/device/common.h#L420) → [all_reduce.h:14，`runRing`](../nccl/src/device/all_reduce.h#L14) → [primitives.h:18–75、115](../nccl/src/device/primitives.h#L18)。

<details><summary>深入：ncclKernelMain 的装载分工与命名陷阱</summary>

- `ncclKernelMain` 先把参数复制到共享内存，避免某些取址方式使其溢出到线程局部栈。
- 装载分三路：第一个 warp 搬 communicator 信息，第二个搬 channel，其余线程加载 work batch，再同步发布。
- `common.h` 是执行框架，`common_kernel.h` 是规约/复制辅助实现；文件名相近，不要看反。
- kernel 名不一定完整反映每项工作的实际算法与协议。

</details>

## 9.3 block、warp、channel、chunk、slice、step 不在同一层

本节给六个名词分层，后面所有推导都靠这张表消歧：

| 名称 | 本章中的含义 | 容易混淆的东西 |
|---|---|---|
| block / CTA | CUDA 调度和共享内存协作单元 | 不是一个固定 SM，也不是一条永久物理链路 |
| warp | 本 CUDA 实现中 32 线程的执行/协作组 | 不必对应一个 peer 或一份完整 chunk |
| channel | NCCL 的逻辑通信通道，含拓扑和 peer 连接 | 不是网卡端口，也不是 CUDA stream |
| chunk | 算法本轮交给 primitive 的逻辑数据块 | 不一定等于整次用户数组的 S/P |
| slice | primitive 内可独立流水推进的数据片 | 大小会受协议、尾块和参数影响 |
| step | 连接进度计数及 FIFO 配额单位 | 不是第 04 章全环的一轮，也不是 kernel 次数 |

在这个通用入口里，一个 CTA 在一次 launch 内映射到一个活动 channel；它驻留哪个 SM、何时被调度，仍由 CUDA 硬件决定。一个 channel 可以带树的多个 peer，多个 channel 也可以争用同一条物理 NVLink/NIC；CTA 内部还会拆分角色，所以没有“一 warp 一链路”的规则。[src/include/device.h:337，`ncclCollCbdPart`](../nccl/src/include/device.h#L337) 计算每个 channel 的范围与 chunk 数，`runRing` 据此确定实际数组偏移。大数组会循环处理多波 chunk，尾波还会重新对齐 chunkCount，因此 S/P 只是第 04 章的等分单波模型。

### 从数组元素找到 channel：两通道教学例子

假设四个 ranks 各有 16 个元素，一次 Ring AllReduce 恰好由两个 channel 均分，每个 channel 在一波中把自己负责的 8 个元素分成四块、每块 2 个元素。这里只固定一个便于手算的调度，不声称 16 元素的真实调用会使用两个 channel；区间均为左闭右开，索引是元素而非字节。

| channel | 负责的用户数组区间 | 局部 C0 | 局部 C1 | 局部 C2 | 局部 C3 |
|---|---|---|---|---|---|
| 0 | `[0,8)` | `[0,2)` | `[2,4)` | `[4,6)` | `[6,8)` |
| 1 | `[8,16)` | `[8,10)` | `[10,12)` | `[12,14)` | `[14,16)` |

以元素 11 为例：它落在 channel 1 的区间里，减去区间起点 8 得局部偏移 3，即局部 C1 的第 2 个元素。四个 ranks 对元素 11 的贡献在 channel 1 的 Ring 中规约、传播，最终各 rank 的输出仍写回用户数组索引 11——不是只有 rank 1 拿到它。

两个 channel 分管不同元素，但每个 channel 都包含参与通信的 ranks，环次序也可以不同。真实分区可能不均匀，还有多波和尾部，要以 work 与 `ncclCollCbdPart` 的输出为准。

**注意：**不能把“每 channel 都处理整个用户数组的 S/P”当成通式。

### 一个单位换算例子

[src/include/device.h:26](../nccl/src/include/device.h#L26) 定义 `NCCL_STEPS=8`。
[src/include/collectives.h:19–20](../nccl/src/include/collectives.h#L19) 给 Ring AllReduce 的 Simple 路径设定：

```text
ALLREDUCE_SLICESTEPS = 8/4 = 2
ALLREDUCE_CHUNKSTEPS = 8/2 = 4
ProtoSimple<SlicePerChunk=2, StepPerSlice=2>
```

对应实例化见 [src/device/all_reduce.h:229–232，`RunWorkColl`](../nccl/src/device/all_reduce.h#L229)。拿它做一次换算：假设 Simple FIFO 总空间为 4 MiB，每 step 的基础空间就是 512 KiB；一个 slice 占 2 steps 进度配额，一个 chunk 占 4 steps，正好两个 slice。

**注意：**
- 这不承诺每个 slice 都装满 1 MiB：小消息、尾部和 `genericOp` 的 sliceSize 计算会减小实际数据量。
- 空 slice 也可能需要推进协议状态，否则两端对后续 step 的解释会错位。

## 9.4 两种 FIFO：任务描述与连接数据不能混为一谈

本节先把两类队列分开，再给出全章最重要的一条分工。

主机提交的 work FIFO/参数空间回答“做什么”：缓冲区、范围和 funcId。连接数据 FIFO 回答“对端已经写到了哪里”：传输中的数据和相关进度。连接结构见 [src/include/device.h:136，`ncclConnInfo`](../nccl/src/include/device.h#L136)：`buffs[proto]` 是协议缓冲，`head/tail` 是进度位置，`step` 保存本地连接进度。

借熟悉的 UMD 队列问两件事：producer 能否占用空间？consumer 能否读取数据？这只是协议类比，不是同一个实现对象。逻辑 step 像持续推进的序号，物理 slot 是循环使用的存储位置；地址相同不代表属于同一代。两问各由一类许可回答：

**credit/head 保护尚未消费的数据不被覆盖；tail/flag 保护尚未就绪的数据不被读取。** 两种许可不能互相替代。

```text
逻辑 step:  ... 6  7  8  9 10 11 ...      （持续增长）
物理槽位:      6  7  0  1  2  3 ...      （step % 8）

生产者 ---- payload + ready ----> 消费者
生产者 <--- consumed / credits -- 消费者
```

“槽位 0 可寻址”不等于“槽位 0 可重用”：必须区分第 0 代和第 8 代的数据，step/flag 就是解决这个问题的代际信息。

**注意：**
- 主机 task/plan 的回收不等于通信完成；本版普通非捕获路径的 work FIFO，在主机确认对应 launch stream 完成事件之后才返还容量。两者不能混为一谈，详见 [08 第 6 节](08-host-execution.md#6-cpu-与-gpu-交接的是-work-descriptor不是-c-planner)。
- 连接数据 FIFO 的单片确认不等于整个集合完成；本地完成也不是所有 ranks 的全局完成通知。
- NET 中要把“消费”限定到当前 FIFO：发送侧的 consumer 是网络发送流程，接收侧的 consumer 才是接收 GPU。
- 连接里还有 direct 指针交换、网络句柄及 `connFifo` 元数据，不能都理解为 payload。
- 注释中的 local/remote 相对于该连接端点而言；NET 的另一端可能是代理，而非另一 GPU 的直接内存。

## 9.5 Simple 的 head/tail 背压状态机

本节回答：发送者凭什么不重写未消费的数据，接收者凭什么不读未就绪的数据？从常规 FIFO 路径看，发送者等待消费确认 head，接收者等待生产进度 tail。源码 [src/device/prims_simple.h:475、529，`loadRecvConn/loadSendConn`](../nccl/src/device/prims_simple.h#L475) 把四个角色明确绑定到两个字段：

```text
WaitRecv -> conn->tail       PostRecv -> conn->head
WaitSend -> conn->head       PostSend -> conn->tail
```

方向不要靠 head/tail 的英文直觉猜，要看当前角色具体把 `connStepPtr` 指向哪里。[src/device/prims_simple.h:103，`waitPeer`](../nccl/src/device/prims_simple.h#L103) 中的核心条件可译成：

```text
发送下一片：head + NCCL_STEPS >= step + StepPerSlice
接收下一片：tail              >= step + StepPerSlice
```

第一条保证发送者不超过 FIFO 容量，第二条保证接收者不读尚未发布的数据。举例：`head=0, step=6, StepPerSlice=2`，发送可推进到 8，使用槽位 6 起的配额。接着 `step=8` 想写下一片，需要 `head>=2`；否则它会覆盖接收者还未消费的槽位 0、1。这就是背压：慢接收者反向限制快发送者，而不是假定“大家大约同时到达”。

接收侧同理：`step=4` 要消费两 steps，必须看到 `tail>=6`。处理完后发布新的 head，允许发送者重用已消费的空间。进度在构造时按 chunk 配额对齐，在析构时写回 `conn->step`，不是每次调用都从零开始，见 [src/device/prims_simple.h:485、787，连接加载与 `~Primitives`](../nccl/src/device/prims_simple.h#L787)。

承接 9.3 的 `NCCL_STEPS=8 / StepPerSlice=2`：上例一个 slice 消耗两 steps 配额，不能把“8 槽”直接读成“8 个这样的 slice”。后面的纸上追踪沿用这个单位；所有 head/tail 比较先对齐逻辑 step，再用 `%8` 找物理位置，不再另画一套流水。

## 9.6 有通知还不够：数据必须先可见

本节回答：接收者凭什么认为数据已可读？先看错误顺序有多直观：发送者先更新 tail、后写数据，接收者一见 tail 到达，便读出了旧值。正确协议要保证**数据/元数据先发布，完成信息后发布**，并使用匹配的读取方式：

```text
发送侧                         接收侧
wait credits                   wait ready
写 payload                     确认可读后取 payload
线程组同步                     线程组确认本片已消费
系统范围 fence + 发布 tail      发布 head，返还 credits
```

这套顺序在源码里是一组配合使用的 fence、访存与同步：[src/device/prims_simple.h:167，`postPeer`](../nccl/src/device/prims_simple.h#L167) 在需要时调用 `fence_acq_rel_sys()`，随后用 `st_relaxed_sys_global` 写进度；第 278 行先做线程组 barrier，确保 worker 数据写入已完成。不能把最后那条 relaxed store 单独拿出来判断对错。

跨 GPU 的等待靠连接进度、flags、网络完成等机制；CTA 内的 barrier 只协调本地线程，不等于跨 GPU barrier。

<details><summary>深入：PTX 级的进度读取、数据读取与本地 barrier</summary>

- 接收侧的进度读取不是“一个普通 C++ volatile 就解决了跨 GPU 内存序”。[src/device/prims_simple.h:86，`loadStepValue`](../nccl/src/device/prims_simple.h#L86) 的普通路径用 volatile PTX load；注释特别要求数据读取同样避开陈旧 L1 数据，NVLS min polling 则使用 acquire.sys 的 multimem 指令。
- [src/device/common_kernel.h:94–99、118，`reduceCopyPacks`](../nccl/src/device/common_kernel.h#L94) 正是配套的 volatile 数据读取。
- CTA 内 `__syncthreads`、命名 barrier 或 `__syncwarp` 只协调参与的本地线程。不同原语组使用不同 barrier 编号和参与线程数，是为了允许规约与转发分组并行，参见 [src/device/common.h:86，`barrier_sync`](../nccl/src/device/common.h#L86)。
- 这些是针对特定 CUDA/PTX/传输假设的实现，不是可随意移植的通用 C++ 无锁队列证明。

</details>

**注意：**
- 这里的 fence 指访存 ordering（先后约束），不是 UMD 中某个“作业已完成”的 fence 对象；名字相同，语义不同。
- fence 不会凭空让 peer 到达、让 FIFO 出现空间、让整个输出完成；这些分别依赖对端推进、消费反馈和完整计算依赖链。
- `waitPeer` 的条件轮询不能由一次 fence 替换，CTA 内 `__syncthreads` 也不能替换跨 GPU 的 ready/credit 协议。

## 9.7 Simple、LL、LL128 的数据布局

本节回答：三种协议把数据和“就绪标志”放在哪里，各付出什么代价？

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

LL 将标志嵌在细粒度数据旁，接收线程可以边等对应 line、边尽早规约。[src/include/device.h:77，`ncclLLFifoLine`](../nccl/src/include/device.h#L77) 定义了两个 data/flag 对；每条 16 B 中只有 8 B 是用户数据，因此布局有效载荷率为 50%。

LL128 减少标志占比，由 warp 协作搬运更大的批次。[src/include/device.h:113–115](../nccl/src/include/device.h#L113) 定义 line 为 128 B、16 个 uint64、15 个数据字，所以典型布局效率严格是 `120/128=93.75%`。

**注意：**
- “Simple”不等于 CUDA memcpy API，也不等于它在所有大小上都最简单或最快。
- LL 源码注释强调 flag 放在数据之后，并依赖 socket 连续接收或 IB/RDMA 的相应 8 B 写入原子性假设；不能把这段话改写成“任意设备上的 16 B 写总是原子的”。
- 93.75% 只是布局比例：不是 128 个线程，也不是承诺链路一定达到 93.75% 峰值。实际还有等待、对齐、尾片和规约成本，调优模型可能使用不同经验折扣。

## 9.8 LL 的“数据到齐”与“可以重用”分别怎么判断

本节回答：LL 用哪两个独立条件判断“能读了”和“能覆盖了”？

[src/device/prims_ll.h:108，`readLL`](../nccl/src/device/prims_ll.h#L108) 反复加载 data/flag，只有两份 flag 都等于 `NCCL_LL_FLAG(recvStep+1)`，才把该 line 当作当前数据。这个检查防止半条数据到达，或残留上一代内容被误读。发送侧 `storeLL` 用配套的向量 volatile store，见同文件第 154 行。

这并不意味着 LL 不需要流控：第 73 行 `waitSend` 仍检查 head 与 `NCCL_STEPS`。**行内 flag 回答“这一代数据准备好了没有”，head 回答“旧数据消费完、能覆盖了吗”；二者是不同的安全条件，不能相互替代。**

由于 LL flag 位宽有限，数值最终会循环。[src/device/prims_ll.h:99，`incSend`](../nccl/src/device/prims_ll.h#L99) 包含 cleanup，在指定周期写完整片的 flags，避免空闲区域遗留的标志在绕回后被误认成新数据。这也是为什么“每次消息很小，不用碰其他位置”不总成立。

## 9.9 LL128 的 warp 协作与默认适用条件

本节回答：LL128 的 warp 如何分工，默认在哪些平台上启用？

[src/device/prims_ll128.h:184，`recvReduceSendCopy`](../nccl/src/device/prims_ll128.h#L184) 由 flag thread 检查标志，通过 `__any_sync` 让 warp 判断是否仍需重载，然后继续读取和规约；发送端将 flag 放进指定字位置，见同文件第 270–287 行。不要把 128 B line 当成“跨任意链路保证原子的 128 B 事务”；它是一套有平台前提的协议布局。

对齐数据可直接装入寄存器；非 16 B 对齐情况可能借助 shared-memory scratch 调整布局，证据：[src/device/prims_ll128.h:99、154，`loadRegsBegin/storeRegs`](../nccl/src/device/prims_ll128.h#L99)。第 297 行 `GenericOp` 负责循环、等待发送空间、最后推进 step 及发布进度；LL128 对代理需要的 tail 通知还包含架构相关 fence，见同文件第 87 行 `postSend`。

默认启用判断在 [src/tuning/cost_model.cc:119，`isLL128Enabled`](../nccl/src/tuning/cost_model.cc#L119)：它分别检查 interType、intraType、计算能力是否兼容，并含驱动/CUDA 特定例外。

<details><summary>深入：LL128 默认启用的拓扑条件</summary>

部分条件允许 interType 到 PXB；Hopper 及更新架构在相应条件下允许到 PXN。intraType 另有 `<= PATH_NVB` 限制，所以不能把前一句误解成“任意纯 PCIe GPU 拓扑都会默认启用”；同样，也不能说“只要经过 PCIe，LL128 一律禁用”。强制协议不提供新的硬件保证；默认门控在同文件第 364 行接入，实验前应核对完整条件。

</details>

## 9.10 规约、forward、最终写为什么融合

用 Ring 中一片数据说明：收到 `[10,20]`，本地同片为 `[1,2]`，下一跳需要 `[11,22]`。朴素安排是接收 kernel、规约 kernel、发送 kernel，伴随额外启动与中间全局内存写回。NCCL 原语可以一次读入，在寄存器中规约，再写下一跳和必要的本地输出：

```text
上游数据 ----\
             +--> 寄存器 applyReduce --> 下游连接/直接目的缓冲
本地输入 ----/                      \--> 本地输出（若这一阶段需要）
```

[src/device/common_kernel.h:40，`reduceCopyPacks`](../nccl/src/device/common_kernel.h#L40) 展示多源读取、`applyReduce`、可选 `applyPostOp` 和多目的写入；这不是一次只允许“copy 或 reduce”二选一。[src/device/all_reduce.h:55](../nccl/src/device/all_reduce.h#L55) 的中间步用 `directRecvReduceDirectSend`；第 64 行用 `directRecvReduceCopyDirectSend` 完成最后规约、保存完整块并开始转发。AG 阶段则 `directRecvCopyDirectSend`，不再把本地输入重复加一遍。后处理只应出现在语义合适的位置，例如最后结果上的除法，而不是每跳都做一次。

**注意：**
- 方法名带 `Direct` 不保证每条连接都零拷贝。[src/device/primitives.h:119，`PrimitivesWithoutDirect`](../nccl/src/device/primitives.h#L119) 把 LL/LL128 的 direct 名称映射到普通原语；Simple 才会结合连接 flags、注册状态与指针交换，决定直接读写或退回 FIFO。
- 从算法调用名推断物理路径，比从源码实际分支推断弱得多。

## 9.11 pipeline：不是一轮结束后全体 barrier

算法示意图画整轮，是为便于证明；实现可以让 slice A 已向下游转发时，slice B 还在上游等待：

```text
时间 ---->
GPU0:  send A     send B     send C
GPU1:     reduce A  reduce B  reduce C
GPU2:        reduce A  reduce B  reduce C
                     ^ 每条边由自己的 credits 约束
```

每级收到足够数据便前进，不必等待整条用户数组完成。慢的一跳会耗尽上游 credits，背压逐步传播，直到所有缓冲使用仍在安全容量内。增加缓冲或 chunk 大小只能改变可在途数据量，不能消灭真正的慢链路。

NET 路径上，GPU、CPU proxy 和 NIC 可以共同推进不同阶段。[src/transport/net.cc:1437](../nccl/src/transport/net.cc#L1437) 是发送代理调用网络 `isend` 的位置；接收侧 [第 1707–1734 行](../nccl/src/transport/net.cc#L1707) 展示条件式 flush 与发布接收进度。GPU 等待时间变长可能是网络/代理进展受阻，不一定是 GPU 规约指令太慢。

**普通 NET 要分开两条本地交接：发送 GPU ↔ send proxy、recv proxy ↔ 接收 GPU，并非两端共用一个 head/tail。** 发送侧 `test` 完成可返还发送缓冲信用；接收侧网络完成加必要可见性处理才发布 tail，GPU 消费后才发布 head。硬件/网络完成、协议消费、应用使用输出是不同边界；接收 head 前进也不证明整个 AllReduce 输出已经完成。这条交接链就是 9.14 状态表的主线。

## 9.12 选择协议也是选择资源占用

Simple 适合摊薄大片数据的控制成本；LL 常以标志流量换小消息更早启动；LL128 改善数据比例，但有 warp 协作、寄存器与 scratch 成本。这是因果方向，不是三个固定消息阈值，也不是“LL128 永远介于另外两者之间”。

增加 CTA/channel 可提高访存和链路利用率，却也占用 SM 调度槽、寄存器和带宽；挤慢 GEMM 可能使整步时间反增。线程阈值见 [src/tuning/tuning_general.cc:101–109](../nccl/src/tuning/tuning_general.cc#L101)，Tree 拆分见 [all_reduce.h:155–164，`runTreeSplit`](../nccl/src/device/all_reduce.h#L155)。调优应比较通信与计算重叠后的关键路径，不只比较独立通信 kernel 的吞吐。

## 9.13 可观察验证问题

以下是有 GPU 环境的验证方案，本机没有执行；步骤衔接[测试](10-nccl-tests.md)与[性能分析](11-performance.md)。

- 固定 Ring 和输入规模，比较合法协议：小消息延迟与大消息有效带宽是否符合控制/标志成本的解释？
- 记录实际 tuning、channel 和线程数；是否误把代表 kernel 名中的 LL 当成实际协议？
- 在受控实验中让一个 rank 晚提交，其他 rank 的等待是否延长？这是进度依赖，不是“自动丢包”。
- 改变 CTA 上限并同时运行计算 kernel：通信单测提升是否转换成训练迭代时间提升？
- 源码断点/插桩时分别看 `step`、`head`、`tail`，能否验证 `head+8` 限制了生产者前进？

在设备自旋循环加 printf 会严重扰动时序，不能把它测出的时间当作真实性能。有限时间等待和异步错误检查也应保留；背压保障无覆盖，并不保障失效 peer 最终一定前进。

## 9.14 无 GPU 状态追踪：一片数据的四个交接字段

本节在纸上推演一次完整交接：普通 NET/Simple 路径，非 shared、非用户 buffer 注册、非 GDC（不取 `gdcSync/gdcFlush` 分支）；不改源码、不做故障注入。沿用 9.5 的 `NCCL_STEPS=8、StepPerSlice=2`，令两端 proxy 的 `sliceSteps=2、base=0`，初始两端 head/tail 都为 0。

**下面这张表追一片数据经过的两套本地缓冲：发送侧一套、接收侧一套。** 每套各有一对 head/tail，共四个交接字段；它们分属两端，不能合并成一条全局进度。先记住每个字段“谁写 → 谁读 → 发布什么含义”：

| 字段 | 谁写 → 谁读 | 发布的含义 |
| --- | --- | --- |
| `S_tail` | 发送 GPU `PostSend` → send proxy | GPU 已准备好发送数据 |
| `S_head` | send proxy → 发送 GPU `WaitSend` | 网络发送完成后的本地信用 |
| `R_tail` | recv proxy → 接收 GPU `WaitRecv` | 网络完成且必要可见性处理结束，数据可供 GPU 读取 |
| `R_head` | 接收 GPU `PostRecv` → recv proxy | GPU 已消费的接收信用；proxy 观察后推进接收侧 `done` |

再给图例。把两端各自的 `sub` 记为 `S`、`R`。**proxy 进度是相对本次操作的 step 计数，四个 head/tail 则是连接上的绝对 step；换算为 `绝对 step = 本端 base + 本端相对进度`。** 本例 base 都为 0，数值碰巧相同，不代表它们是同一个变量。

- 发送列依次为 `(S.posted, S.transmitted, S.done)`：已纳入发送窗口、`isend` 已受理、发送 `test` 已完成的进度。非 shared 路径的 `S.posted` 只是 proxy 记账，不发布 `S_head`，也不表示 GPU 已写好。
- 接收列依次为 `(R.posted, R.received, R.transmitted, R.done)`：`irecv` 已受理、接收 `test` 已完成、必要可见性处理后交给 GPU、proxy 已确认 GPU 消费的进度。这里的 `R.transmitted` **不是网络发送**，`R.done` 也不是接收 `test` 的完成标志。

再看追踪对象与窗口。只追第一片非空 slice：它从绝对 step 0 推进到 2，payload 起点是各自 FIFO 的 slot `0%8=0`，占两 steps 配额。取两端各 `nsubs=1、nsteps=16`；窗口上限按 `maxDepth=min(NCCL_STEPS, NCCL_SHARED_STEPS/nsubs)` 计算。本版 `NCCL_STEPS=8`、`NCCL_SHARED_STEPS=16`，所以 `maxDepth=min(8,16/1)=8`。**这里的常量 16 与本例操作总步数 `nsteps=16` 只是数值相同，窗口上限不是由 `nsteps` 算出的**；定义和计算见 [net.cc:665](../nccl/src/transport/net.cc#L665)、[1343](../nccl/src/transport/net.cc#L1343)。

先把 8 steps 的窗口投满，再追这片 payload。当前窗口的其他三片只准备了缓冲，GPU 暂不发布其数据，后半操作仍等待空位；**一个 slice 是两 steps，不是整个操作或整个 chunk。**

读表约定：每行是所述动作后的状态；窗口准备行合并了 `posted` 按 `0→2→4→6→8` 推进的多次调用。发送和接收可交叠，**这只是一个可行时序，不是两端全局串行的保证**。接收可见性部分选取需要异步 `iflush` 的情况；非 GDC 不等于非 GDR，无需 flush 时跳过等待即可。

| 动作与交接条件 | 发送相对进度 `(posted, transmitted, done)` | 接收相对进度 `(posted, received, transmitted, done)` | `S_head` | `S_tail` | `R_head` | `R_tail` |
| --- | --- | --- | --- | --- | --- | --- |
| 初始化；发送 FIFO 的 size 尚为 `-1` | `(0,0,0)` | `(0,0,0,0)` | 0 | 0 | 0 | 0 |
| 接收窗口已投递：各次 `irecv` 返回非空 request；send proxy 也完成窗口记账。接收已投递不等于收到数据 | `(8,0,0)` | `(8,0,0,0)` | 0 | 0 | 0 | 0 |
| 发送 GPU 通过 `S_head+8>=0+2` 的信用检查，在 `waitPeer` 先写 `connFifo[0].size`；payload 尚未发布。**size 先写不等于 ready**：size 非 `-1` 仍需 tail，此时不能发送 | `(8,0,0)` | `(8,0,0,0)` | 0 | 0 | 0 | 0 |
| GPU 写完 payload，经线程组 barrier、系统 fence，再由 `PostSend` 发布 `S_tail=0+2` | `(8,0,0)` | `(8,0,0,0)` | 0 | 2 | 0 | 0 |
| send proxy 见 size 有效且 `S_tail>S.base+S.transmitted`，尝试 `isend`，但 request 为空：**未受理**（不是完成），进度不动，稍后重试 | `(8,0,0)` | `(8,0,0,0)` | 0 | 2 | 0 | 0 |
| 再次 `isend` 返回非空 request，才将 `S.transmitted` 加 2；受理尚不等于发送完成 | `(8,2,0)` | `(8,0,0,0)` | 0 | 2 | 0 | 0 |
| 发送 `test` 完成：先将 size 重置为 `-1`，经 CPU `seq_cst` fence，再推进 `S.done` 并发布 `S_head=S.base+S.done=2` | `(8,2,2)` | `(8,0,0,0)` | 2 | 2 | 0 | 0 |
| 接收 `test` 完成，`R.received` 加 2；本例非空且 `useGdr && needFlush`，调用 `iflush` 并取得待完成 request。GPU 尚不可读 | `(8,2,2)` | `(8,2,0,0)` | 2 | 2 | 0 | 0 |
| flush request 的 `test` 完成（未完成就保持上一行）；`R.transmitted` 加 2，经 CPU `seq_cst` fence 发布 `R_tail=R.base+R.transmitted=2` | `(8,2,2)` | `(8,2,2,0)` | 2 | 2 | 0 | 2 |
| 接收 GPU 见 `R_tail>=0+2`，读取、处理本片，线程组确认消费结束后由 `PostRecv` 发布 `R_head=2` | `(8,2,2)` | `(8,2,2,0)` | 2 | 2 | 2 | 2 |
| recv proxy 观察到 `R_head>R.base+R.done`，且 `R.transmitted>R.done`，才推进 `R.done`（如插件提供，还调用 `irecvConsumed`） | `(8,2,2)` | `(8,2,2,2)` | 2 | 2 | 2 | 2 |

**`S_head=2` 不能推出 `R_head=2`。** 前者归还发送缓冲的信用；后者要等接收 GPU 实际消费，recv proxy 看到它才回收接收窗口。表中发送 `test` 先被观察到，不要求接收侧也按此顺序推进；即使 `R_tail=2`，只要 `R_head=0`，也只能说数据可读，不能说已消费。最后一行也仅完成了第一片的交接，不表示整个 AllReduce 输出可供应用使用。

窗口背压同样分两端计算：允许继续投递须满足本端 `posted<nsteps` 且 **`posted<done+maxDepth`**。本例还有工作（`8<16`），但接收窗口已满：`R.done=0` 时 `8<0+8` 不成立；仅提高 `R.received` 或 `R.transmitted` 不会腾出窗口。GPU 消费且 proxy 将 `R.done` 推进到 2 后，下一次投递才获得空间（`8<2+8`），可把 `R.posted` 推进到 10。发送窗口则由自己的 `S.done` 归还配额。慢接收 GPU 因而能堵住后续接收投递，再沿网络向发送端传递背压；发送侧先释放一片不等于消除了这条依赖。

<details><summary>深入：表后源码核查入口（PTX、fence、GDC 分支）</summary>

- [net.cc：`sendProxyProgress`](../nccl/src/transport/net.cc#L1324)：核查窗口优先投递、size 与 tail 的双条件、非空 `isend` request，以及发送完成时“清 size → fence → 发布 head”的顺序。
- [net.cc：`recvProxyProgress`](../nccl/src/transport/net.cc#L1493)：核查 `irecv → test → 必要 iflush → 发布 tail → 观察 head`。需要 flush 时，接收完成还不够；没有 flush request 则无需再 `test`，有 request 必须等完成。GDC 的同步映射写及 WC fence、flush 的 x86/非 x86 实现是另行核查的分支，不套进本表。
- [prims_simple.h：`loadStepValue` 起](../nccl/src/device/prims_simple.h#L86)：同文件 `loadRecvConn/loadSendConn` 绑定四个角色；`waitPeer` 先写 size，`genericOp` 的 barrier 后才 `postPeer`。普通进度读取用 volatile PTX；NVLS min polling 在 `__CUDA_ARCH__>=900 && CUDART_VERSION>=12010` 下另走 acquire.sys multimem 分支，不是本表的 NET 读取。
- [common_kernel.h：`reduceCopyPacks` 的配套读取](../nccl/src/device/common_kernel.h#L94)与 [op128.h：PTX 访存/屏障封装](../nccl/src/device/op128.h#L342)：volatile 进度轮询要配合 volatile payload 读取，避免陈旧 L1。`st_relaxed_sys_global` 在 `__CUDA_ARCH__>=700` 用 `st.relaxed.sys.global`，旧架构用 `st.volatile.global`；`fence_acq_rel_sys` 分别用 `fence.acq_rel.sys` / `membar.sys`。不能孤立看 relaxed store，也不能将整套协议改写成通用 C++ acquire/release 证明。

</details>

有 GPU 后，只在合法配置下核对日志中的 transport/协议与 timeline 的 kernel、CPU proxy 活动；日志没有显示某个 step，不等于它不存在。日志、timeline 和性能对照能提供进度线索，不能单独证明所有内存序正确；本节没有执行 GPU 实验或故障注入。

## 9.15 自测与答案

**题 1：NCCL_STEPS=8，发送 step=8、StepPerSlice=2、head=1，可否继续？**
答：不可以，`1+8<8+2`；至少要 head 达到 2，才能安全重用对应空间。

**题 2：LL128 的 120/128 是什么比例？能否据此承诺带宽达到硬件峰值的 93.75%？**
答：它是 line 布局的数据占比；不能承诺实际带宽，等待、对齐、规约与资源争用仍有成本。

**题 3：为何不能只用 __syncthreads，或者只看 directSend 的名字来判断传输正确性？**
答：CTA barrier 不完成跨 GPU 发布；还需协议的可见性和流控。Direct 是原语接口能力，是否走直接缓冲取决于协议和连接条件。

**下一站：[10 nccl-tests](10-nccl-tests.md)。** 将本章的就绪、完成与背压机制转成受控验证，再进入 11 的性能分析。
