# 06｜Host execution：把一次调用编成执行计划

承接 [05｜传输](05-transports.md)，连接已经说明了数据能怎样到达 peer。
本章沿一次请求看主机如何选择执行方式，并把材料交给 GPU 和 CPU proxy。
源码基线为 NCCL **2.32.3**，`12df1a11`。

案例仍是两台主机各四张 GPU，八个进程各管理一个 rank。
每个 rank 对 64 MiB FP32 数据做 Sum AllReduce，`count=16*1024*1024`。
主机与 rank 布局是教学假设；这里跟踪普通 host collective 的 **Ring/Simple 分支**，具体选择由调优决定。

## 1. 请求离开 API，进入 communicator 的 planner

各 rank 按相同顺序提交相容的操作，输入由 stream `s` 上的前序工作产生。
先看 blocking communicator 上的一次调用示意，初始化与错误检查宏沿用第 02 章：

```cpp
size_t count = 16 * 1024 * 1024;
NCCL_CHECK(ncclAllReduce(grad, grad, count, ncclFloat, ncclSum, comm, s));
```

这里的 count 是元素数，乘四字节才是 64 MiB。
`ncclAllReduce` 经 `ncclAllReduceConfigImpl` 构造栈上的 `ncclInfo`，
其中装着用户指针、count、类型、规约操作、comm、stream 和分块提示。
随后调用 `ncclEnqueueCheck`，检查 communicator 与参数，再进入 `taskAppend`。

主机的数据流可以压成四个对象：

```text
ncclInfo           ncclTaskColl          ncclKernelPlan        ncclDevWorkColl
临时 API 参数  ->  可调优的集合任务  ->  一次发射的计划  ->  GPU 消费的工作描述
```

`taskAppend` 先把 host reduction op 转成设备表示，普通集合再进入 `collTaskAppend`。
后者分配 `ncclTaskColl`，复制指针、count、datatype、op 等元数据。
64 MiB payload 留在用户 buffer 中，随任务流转的是地址和执行信息。
因此栈上的 `ncclInfo` 可以退出，用户 buffer 则要覆盖通信执行期。

`comm.planner` 是每个 communicator 的主机规划状态，类型为 `ncclKernelPlanner`。
它持有任务收集器、待调度队列、stream 列表，以及正在构造和等待发射的 plans。
它本身不执行算法，而是让同一批任务在各加工阶段有明确的存放位置。
任务保存单次操作的内容，planner 保存整组操作的组织关系。
例如 stream 被收集在 planner 的列表里，供后续统一建立发射依赖；
任务进入 plan 后，planner 的待调度计数减少，plan 则接管本次发射需要的任务与描述。

```text
comm.planner
  collSorter                    收集本组 ncclTaskColl
       | prepare
  collTaskQueue + collWorkQueue  已调优任务 + 对应设备描述
       | schedule
  wipPlan -> planQueue           正在组装 -> 等待发射
```

图中的设备描述先在注册阶段建立骨架，调度进 plan 时再补 channel 分区和 chunk 信息。
所以这条对象链表达的是信息逐渐具体化，不要求四种对象依次独占存在。

## 2. 最外层 group 结束，才开始处理收集到的任务

`ncclEnqueueCheck` 内部配对调用 `ncclGroupStartInternal/EndInternal`。
没有显式 group 时，这一对调用就是最外层边界，一次 API 收集完便开始提交。
有显式 group 时，内部结束只减少嵌套深度，任务一直留到外层 `ncclGroupEnd`。

```text
ncclGroupStart()          depth: 0 -> 1
  AllReduce A 内部 group         1 -> 2 -> 1，收集 A
  AllReduce B 内部 group         1 -> 2 -> 1，收集 B
ncclGroupEnd()                    1 -> 0，开始准备和提交
```

这样，NCCL 可以统一处理连接，并把多项工作打包。
Group 收集当前线程在本组内提交的 communicator 请求；本例每个进程只提交自己的 rank。
其他七个进程各自提交匹配操作，跨 rank 的数据依赖随后由协议满足。

谁执行 group 结束后的工作，由 communicator 的 blocking 配置决定：

| 配置 | 主机准备与发射由谁推进 | 应用如何继续 |
| --- | --- | --- |
| blocking | 调用 `GroupEnd` 的线程 | 主机提交结束后返回 |
| nonblocking | `groupLaunchNonBlocking` 后台线程 | 返回 `ncclInProgress` 时查询异步状态，提交完成后再使用相关 stream |

连接 job 可另外并行执行，并在需要的阶段汇合。
这里的 blocking 描述主机调用行为；GPU 通信完成仍由 stream 顺序或 event/synchronize 确认。
同一 stream 上排在 AllReduce 后面的消费 kernel，会等待通信完成。

本版 `enqueue.cc:33` 的 `EnqueueRearchEnable` 默认是 0，主线进入 `groupLaunchLegacy`。
开启新分支后先收集 raw tasks，并重组任务准备；
`group.cc:922–995` 最后仍回到 `doLaunches`，发射线程仍服从上表。

## 3. 调优、连接、注册：先决定怎么做，再备齐材料

外层 group 结束后，普通集合沿下面的顺序前进：

```text
groupLaunchLegacy
  ncclPrepareTasksAndCollPreconnect  group.cc:824
  asyncJobLaunch                    group.cc:828，完成连接 jobs
  ncclTasksRegAndEnqueue             group.cc:840
  doLaunches                        group.cc:873
```

准备阶段的 `ncclPrepareTasks` 从 sorter 取出任务，按集合、规约和类型整理。
相近大小的兼容任务可共享一次调优决策，各自的输入输出和结果仍然独立。
`ncclGetAlgoInfo` 调用 tuning，根据消息大小、拓扑、候选支持与资源限制选择方案。

本例沿选中的 Ring/Simple 继续：
`algorithm` 决定通信依赖，`protocol` 决定搬运与同步方式，
`nMaxChannels` 和 `nWarps` 决定可用并行度与线程规模。
`devFuncId` 将集合、类型、规约、算法和协议编码成设备执行入口的索引。

调优给出 channel 使用上限；实际使用哪些 channel、各处理多少数据，在下一步调度时确定。
这样能把单个任务的选择与整组任务的负载分配结合起来。

若 Ring 的运行期连接尚未建立，准备阶段会标记需求并创建连接 job。
`asyncJobLaunch` 在这些 job 完成后返回，注册阶段才使用已备齐的连接资源。
随后 `ncclTasksRegAndEnqueue` 按路径需要注册 buffer，建立 `ncclDevWorkColl`，
把 send/recv 指针、规约参数和 warps 等写入 `collWorkQueue`。

到这里，任务的执行方式已确定，连接与所需注册已就绪。
GPU 尚未消费用户数据；主机接下来要解决的是如何装进一次或多次发射。

## 4. Channel 分数据，plan 分发射

Channel 是集合操作的逻辑工作分区，带有自己的 ring/tree 邻居和 peer 连接。
一个任务可以使用多个 channel，让不同数据区间并行通信。
每个 channel 的环都覆盖参与 ranks，但它只处理本任务分给它的区间。

假设本例使用四个 channel，并恰好均分输入，分工如下：

| channel | 元素区间，左闭右开 | 数据量 |
| --- | --- | --- |
| 0 | `[0, 4*1024*1024)` | 16 MiB |
| 1 | `[4*1024*1024, 8*1024*1024)` | 16 MiB |
| 2 | `[8*1024*1024, 12*1024*1024)` | 16 MiB |
| 3 | `[12*1024*1024, 16*1024*1024)` | 16 MiB |

四份加起来仍是每 rank 的 64 MiB；每份都要汇合八个 rank 对相同元素的贡献。
真实分区由 `scheduleCollTasksToPlan` 计算，首尾 channel 可以与中间大小不同。
它用 `channelLo/Hi`、`countLo/Mid/Hi` 和 `chunkGrains*` 把分工编码进 work。
增加 channel 可以增加并行搬运，也会增加 CTA 和线程资源占用。

调度器先估计每个 channel 的负载，再按调度用的对齐单元（cell）铺开数据。
当前 channel 剩余的负载空间可以继续容纳后续任务，最后一个 cell 按真实 count 裁掉多余部分。
这就是描述要分别记录首、中、尾数量的原因：既能接续分配，又能保留精确的用户数组边界。
修改均衡策略时，检查每段起点、长度和总和，可以直接判断是否遗漏或重复处理了某段数据。

`doLaunches` 先调用 `ncclLaunchPrepare`，后者循环创建 `ncclKernelPlan`。
每轮用 `scheduleCollTasksToPlan` 消费待调度队列，直至当前 plan 的预算用完。
预算主要约束 kernel 参数空间和设备工作描述存储；普通非捕获 plan 的外部描述预算为 work FIFO 的一半。

装不下的任务留给下一个 plan，因此一个 group 可产生多个 kernel。
反过来，多项兼容任务也可进入同一个 plan。
64 MiB 是用户数据量，描述只需记录它的地址与划分；数据很大并不直接要求多个 plan。

Plan 最后包含 kernel 入口、参数、`channelMask`、work batches 和按需生成的 proxy ops。
一个 batch 标明 work 类型、`funcId` 及描述偏移，让同一 channel 连续执行兼容工作。
同一份 collective work 可以被多个 channel 的 batch 引用，各自取自己的数据区间。

## 5. 上传描述，然后分成 GPU 与 proxy 两路

`finishPlan` 把每个 channel 的首个 batch 排在参数区前部。
描述足够小时一并放进 kernel 参数；较多描述使用 work FIFO，图捕获则可使用 persistent storage。
发射前，`uploadWork` 写入对应存储，并修正 batch 的描述偏移。
这里上传的是执行描述，payload 仍由后面的通信执行路径搬运。

```text
                         ncclKernelPlan
                         /            \
          work + kernel 参数          ncclProxyOp（连接需要时）
                  |                           |
            CUDA driver launch          uploadProxyOps / ProxyStart
                  |                           |
            GPU 算法与协议              CPU proxy 网络 progress
```

`ncclLaunchKernel` 从 `channelMask` 的置位数构造 grid，
从 `threadPerBlock` 构造 block，再通过 `cudaGetFuncBySymbol` 取得 `CUfunction`。
参数缓冲交给 `cuLaunchKernelEx`，旧接口分支使用 `cuLaunchKernel`。
在本章通用 kernel 中，一个活动 channel 对应一个 CTA。

同一 GPU、同一 communicator 的多 stream group 会在发射前汇合依赖，结束后把完成依赖分发回各 stream。
因此主机打包优化仍保留用户输入已产生、输出随后才被消费的执行顺序。

另一侧，调度时的 `ncclAddProxyOpIfNeeded` 询问连接是否需要 CPU progress。
需要时记录 channel、protocol、步数和 buffer 等信息，交给 proxy 推进。
普通非捕获路径通常在 launch 后调用 `hostStreamPlanTask` 完成交接；
图捕获等场景可借 host callback 安排这一步。
GPU 与 proxy 随后通过协议状态协作，网络传输并不要求主机调用线程逐片等待。

回收按对象的最后使用者安排：非 persistent plan 交接后可进入主机回收队列，
work FIFO 容量则由 launch stream 完成事件确认后返还，用户 buffer 保持到通信完成。
图持有的描述与注册引用会更长寿，统一见 [10｜内存与生命周期](10-memory-and-lifetime.md)。

## 6. 源码入口与离线追踪

下面六处足以把本章的数据流接起来，行号对应上述基线：

- [collectives.cc:192、206](../nccl/src/collectives.cc#L192)：`ncclAllReduceConfigImpl`、`ncclAllReduce`，构造请求。
- [enqueue.cc:3478](../nccl/src/enqueue/enqueue.cc#L3478)：`ncclEnqueueCheck`；向上追 3337 的 `taskAppend`、2757 的 `collTaskAppend`。
- [comm.h:485](../nccl/src/include/comm.h#L485)：`ncclKernelPlanner`；同文件 212、357 为 task 与 plan。
- [group.cc:748](../nccl/src/group.cc#L748)：`groupLaunchLegacy`；427 的 `doLaunches` 串联准备与逐 plan 发射。
- [enqueue.cc:644](../nccl/src/enqueue/enqueue.cc#L644)：`scheduleCollTasksToPlan`；1365 的 `uploadWork` 完成描述交接。
- [enqueue.cc:1695](../nccl/src/enqueue/enqueue.cc#L1695)：`ncclLaunchPrepare`；1886 的 `ncclLaunchKernel` 与 140 的 `ncclAddProxyOpIfNeeded` 对应两路出口。

**离线任务：追踪同一个 `grad` 指针。**
从 `ncclInfo.sendbuff` 出发，依次找 task 的赋值、设备 work 的赋值、work 上传和 driver 参数。
再按第 4 节的四 channel 假设，计算 channel 2 的起始字节偏移。

可核对结论：指针在 `enqueue.cc:2802` 进入 task，369 行进入设备 work；
1425 行复制 work 元数据，1902 行把参数缓冲交给 launch。
Channel 2 从 `8*1024*1024` 个 FP32 元素处开始，即 **32 MiB** 偏移；payload 没被复制进 task。

下一章 [07｜Device execution](07-device-execution.md) 从这份 launch 参数出发，看 CTA 如何真正执行它。
