# 02｜Host execution：沿一次 AllReduce 看执行主线

第一次学习建议先读 [00｜初学者主线](00-mainline.md)。本章不追求覆盖所有字段，而是用一个小例子看清：**一次 `ncclAllReduce` 怎样经过 task、调优、work 和 plan，最后变成 CUDA launch。**

进阶的队列、channel 调度和 FIFO 细节移到 [Host 执行进阶附录](appendix/deep-dive/host-execution.md)。

## 1. 先把例子和主线固定下来

假设单进程管理两张 GPU，每张 GPU 有 4 个 FP32：

```text
r0：[1,1,1,1]
r1：[2,2,2,2]
结果：[3,3,3,3]
```

一个 rank 的调用是：

```cpp
ncclAllReduce(grad, grad, 4, ncclFloat, ncclSum, comm, stream);
```

为了只观察主线，暂固定 Ring/Simple、1 个 channel、普通 host collective，不展开 Graph、CE、NVLS 和复杂注册。

整个 host 流程可以先记成：

```text
ncclInfo → ncclTaskColl → 调优结果 → ncclDevWorkColl
         → ncclKernelPlan → CUDA launch
```

用户数组始终在 GPU buffer 中，流转的是**地址、数量、偏移和执行参数**，不是 payload 副本。

## 2. API 入口：把调用变成 `ncclInfo`

公开 [`ncclAllReduce`](../nccl/src/collectives.cc#L204) 调用 `ncclAllReduceConfigImpl`，组装一份临时 `ncclInfo`：

| 信息 | 本例中的值 |
|---|---|
| collective | AllReduce |
| send/recv | 都是 `grad`，即原地操作 |
| count | 4 个元素 |
| datatype | FP32 |
| op | Sum |
| comm / stream | 当前 communicator 和 CUDA stream |

`ncclInfo` 只是“用户想做什么”的请求单。此时没有选择算法、分配 channel 或启动 kernel。

## 3. 形成任务：从 `ncclInfo` 到 `ncclTaskColl`

[`ncclEnqueueCheck`](../nccl/src/enqueue/enqueue.cc#L3478) 检查 communicator 和参数，然后进入 `taskAppend`。普通多 rank collective 沿传统 host 路径进入 [`collTaskAppend`](../nccl/src/enqueue/enqueue.cc#L2757)。

它分配 `ncclTaskColl` 并复制：

- send/recv 指针；
- count、datatype 和规约信息；
- 后续配置所需的任务元数据。

随后任务进入 `planner->collSorter`，等待最外层 GroupEnd。

两种对象可以这样区分：

| 对象 | 职责 |
|---|---|
| `ncclInfo` | 当前 API 调用的临时请求 |
| `ncclTaskColl` | 可以等待调优、连接和计划生成的内部任务 |

`ncclTaskColl` 保存的是 buffer 地址，不是那 4 个 float 的副本。

## 4. GroupEnd：决定“准备怎么做”

一个线程管理多 GPU 时，Group 先把各 rank 的任务收齐，避免第一个任务开始时，后续任务尚未提交。

最外层 GroupEnd 后，默认主线进入 `groupLaunchLegacy`，由 [`ncclPrepareTasks`](../nccl/src/enqueue/enqueue.cc#L423) 做三件事：

```text
整理任务 → 调优 → 把结果写回原始任务
```

### 调优输入与输出

[`ncclGetAlgoInfo`](../nccl/src/enqueue/enqueue.cc#L2159) 根据任务和 comm 上下文选择执行方案。初学时只看这些主要结果：

| 结果 | 含义 |
|---|---|
| `algorithm` | Ring、Tree 等，决定数据依赖关系 |
| `protocol` | Simple、LL、LL128，决定分片与进度组织 |
| `nMaxChannels` | 允许使用的 channel 上限，不是最终分配数 |
| `nWarps` | 执行工作的线程组规模 |
| `devFuncId` | device 后续分派到具体实现的编号 |

本例固定为：

```text
algorithm = Ring
protocol  = Simple
nMaxChannels = 1
```

`devFuncId` 对应 AllReduce/FP32/Sum/Ring/Simple。此时还没有启动 kernel。

### 必要时先补齐连接

如果 `runtimeConn` 已开启，而所选算法的实际 transport 连接尚未建立，准备阶段会记录 `algoNeedConnect`。Group 调度会先完成 preconnect，再进入注册与发射阶段。

因此，**拓扑图和逻辑邻居可以提前准备，实际连接可以等第一次选中算法时再补齐**。已有连接则直接复用。

## 5. 生成 work 和 plan：把任务变成可发射材料

连接准备完成后，[`ncclTasksRegAndEnqueue`](../nccl/src/enqueue/enqueue.cc#L351) 先按路径需要准备 buffer 访问关系，再生成 `ncclDevWorkColl` 并放入 `collWorkQueue`。

`ncclDevWorkColl` 主要给 GPU 使用，包含：

- send/recv 地址；
- 规约参数；
- warp 数；
- 后续的 channel 分工和 chunk 粒度。

### channel 与 plan 分别负责什么？

[`ncclLaunchPrepare`](../nccl/src/enqueue/enqueue.cc#L1695) 创建 `ncclKernelPlan`，再由 [`scheduleCollTasksToPlan`](../nccl/src/enqueue/enqueue.cc#L644) 计算实际 channel 数和数据分工。

可以先这样区分：

| 概念 | 解决的问题 |
|---|---|
| channel | 数据怎样并行分工，以及每个分区使用哪些邻居和连接 |
| plan | 一次 kernel launch 需要哪些参数、work batches 和 proxy 工作 |

调度时会更新 `channelLo/channelHi`、`countLo/Mid/Hi` 和 `chunkGrains*`。本例只有 1 个 channel，最终效果就是：

```text
channel 0 负责全部 4 个 float
task->nChannels = 1
```

`nMaxChannels` 是上限，`task->nChannels` 才是实际调度结果。复杂的负载均衡和首/中/尾计算暂不影响理解这条主线。

### work batch 连接 work 与执行入口

调度器还会为 channel 添加 work batch，把以下信息关联起来：

```text
哪个 channel 执行
执行哪个 devFuncId
读取哪份 ncclDevWorkColl
```

同一份 work 可以被多个 channel 的 batch 使用，各 channel 再根据自己的 ID 恢复负责的数据区间。

## 6. `finishPlan/uploadWork`：准备 launch 参数

[`finishPlan`](../nccl/src/enqueue/enqueue.cc#L236) 准备 `kernelArgs`：

- `comm = comm->devComm`，提供 device 端 communicator；
- `channelMask`，标记本次参与哪些 channel；
- work batches 和 work 存储方式。

小描述可以直接放进 kernel 参数区；较大描述可使用 work FIFO，Graph 场景还可使用 persistent buffer。

[`uploadWork`](../nccl/src/enqueue/enqueue.cc#L1365) 随后写入执行描述并修正 batch 偏移。再次强调：

> **这里上传的是“做什么”的描述，不是用户那 4 个 float。**

## 7. `ncclLaunchKernel`：真正启动 kernel

[`ncclLaunchKernel`](../nccl/src/enqueue/enqueue.cc#L1886) 用 `channelMask` 的置位数构造 grid，用 `threadPerBlock` 构造 block，再通过 CUDA Driver 启动 kernel。

例如：

| `channelMask` | 活动 channel | `grid.x` |
|---|---|---:|
| `0b0001` | channel 0 | 1 |
| `0b0110` | channel 1、2 | 2 |
| `0b1111` | channel 0、1、2、3 | 4 |

本例：

```text
channelMask = 0b0001
grid = {1,1,1}
```

在这条通用 kernel 路径中，一个活动 channel 对应一个 CTA。`blockIdx.x` 是连续编号，device 再根据 mask 将其映射回真实 channel ID。

plan 中的 `kernelFn` 决定 host 启动哪个 kernel 入口；batch 中的 `devFuncId` 决定 device 内部执行哪种 collective/algorithm/protocol 工作。

如果某些连接需要 CPU progress，plan 还会携带 proxy 工作，由 `hostStreamPlanTask` 上传并启动。普通本机 P2P 路径不必都经过 NET proxy。

## 8. 最后分清提交与完成

`ncclLaunchKernel` 返回，只表示 kernel 已按当前路径提交，**不代表 AllReduce 已完成**。

```text
launch 已提交
    ≠
GPU 结果已经可读
    ≠
plan、FIFO、注册和映射都能立即回收
```

输出是否可消费看相应 CUDA stream/event；host plan 和 work FIFO 的回收各有自己的完成条件。

## 9. 用一张表复习本章

| 阶段 | 主要产物 | 还没完成什么 |
|---|---|---|
| API 入口 | `ncclInfo` | 没有选择算法或发射 |
| 任务登记 | `ncclTaskColl` | 没有确定执行方案 |
| GroupEnd + 调优 | algorithm、protocol、channel 上限、devFuncId | 没有最终 channel 分工 |
| work 生成 | `ncclDevWorkColl` | 没有形成 launch |
| plan 调度 | `ncclKernelPlan`、`channelMask` | 没有上传和启动 |
| launch | CUDA kernel + 可选 proxy 工作 | 不代表 GPU 已完成 |

下一步读 [03｜Device execution](03-device-execution.md)：看 CTA 怎样从 `channelMask` 找到 channel，再装载 `devComm`、channel 和 work，最终进入 `runRing`。

需要研究 planner 队列、channel 调度、FIFO 容量和回收细节时，再读 [Host 执行进阶附录](appendix/deep-dive/host-execution.md)。
