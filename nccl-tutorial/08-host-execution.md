# 08｜Host execution：一次 AllReduce 怎样变成 GPU 工作

> 基线：`12df1a11`，NCCL **2.32.3**。本章主线是普通 host API 的多 rank、非空 AllReduce，而非全部 device/CE 路径。

## 学习目标

- 沿真实调用链追踪一次 API 请求如何变成 task、plan 和 work descriptor。
- 理解 group 为什么既是提交边界，又影响连接、调优与批处理。
- 分清 channel、CUDA block、proxy op 的职责和对应关系。
- 能辨认本版默认 legacy 路径、rearch 准备框架与独立 scheduler 目录。

本章假定已掌握 [05 的 communicator 生命周期](05-communicator.md)、[06 的拓扑与 channel 资源](06-topology.md)、[07 的 transport 与 proxy 分工](07-transports.md)；stream 的基本规则按需速查 [03](03-cuda-semantics.md)。
本章只解释主机如何把通信意图翻译成执行材料，不重讲 [04 的算法核心（4.2～4.7）](04-algorithms.md)；下一站是 [09｜设备协议](09-device-protocols.md)。
[14｜源码地图](14-source-map.md)与 [15｜练习](15-exercises.md)贯穿阅读；[00｜环境](00-environment.md)与 [examples](examples/README.md)并行速查。

## 从 UMD 经验切入：提交结束为何仍不能消费梯度

先看一个场景：八 rank、64 MiB 的 AllReduce 已成功预热，走普通、非捕获的 Ring/Simple 分支，连接已就绪。下一轮，rank 7 的应用线程迟迟不调用 AllReduce。其他 ranks 的本地提交还会发生什么？

UMD 的 command recording → submission → completion 经验可以帮你提问，但 NCCL 的软件结构要按它自己的职责理解。本章的最小因果链是：**记录通信意图 → 准备资源并提交 → 执行中等待对端推进 → 通信完成后消费结果。**

先给结论：若没有额外的主机侧协商等待，其他 ranks 可能已返回 API、甚至已启动通信 kernel，却因 rank 7 缺席而无法完成。**本地提交不是全局匹配确认。** 软件交接见第 2～7 节，协议等待与完成证据见第 7 节和章末任务。

## 1. 贯穿案例：训练线程提交 64 MiB 梯度

本节回答：本章要追踪的那次调用长什么样？继续使用 A、B 两台主机、每台四 GPU、每 GPU 一进程的八 rank 案例。每个 rank 在 stream `s` 上先产生梯度，再发出相同的 AllReduce：

```cpp
// 各 rank 以兼容顺序调用；grad 的分配和设备选择已完成。
produce_gradient<<<grid, block, 0, s>>>(grad);
ncclAllReduce(grad, grad, 16 * 1024 * 1024,
              ncclFloat, ncclSum, comm, s);
consume_gradient<<<grid, block, 0, s>>>(grad);
```

以上是提交关系示意，不是完整可编译程序；实际程序必须检查 CUDA/NCCL 返回值。
同一 stream 保证消费 kernel 等待通信完成，但 CPU 通常不会等这 64 MiB 全部传完才返回。沿 `grad` 指针追踪四个问题：

| 层次 | 回答的问题 | 典型产物 |
| --- | --- | --- |
| API 请求 | 做什么、操作多少元素、在哪个 stream？ | `ncclInfo` |
| task preparation | 用哪个算法/协议，需要什么连接和注册？ | 已调优的 task |
| plan scheduling | 哪些 task 放一起，如何分配 channels？ | `ncclKernelPlan` 与 work batches |
| launch/progress | GPU 怎样得到描述，网络如何持续推进？ | kernel 参数、work storage、proxy ops |

## 2. API 入口很薄，因为它还不知道执行方案

本节回答：API 入口做了哪几件事，又没做什么？

`ncclAllReduce` 调用 `ncclAllReduceConfigImpl`，后者在栈上构造 `ncclInfo`——一次请求的临时描述，字段携带函数种类、buffer、count、datatype、reduction op、comm、stream 和分块提示。普通 API 传入空配置，仍会经过统一的 collective config 解析。这里没有遍历所有 GPU，更没有直接执行“把八张卡的数据加起来”。

源码：[collectives.cc](../nccl/src/collectives.cc)，`nccl/src/collectives.cc:192`，`ncclAllReduceConfigImpl`；`nccl/src/collectives.cc:206`，`ncclAllReduce`。
请求结构：[info.h](../nccl/src/include/info.h)，`nccl/src/include/info.h:17`，`ncclInfo`。

进入 `ncclEnqueueCheck` 后，主要检查 comm 是否可用、参数是否合法，并进入内部 group。随后 `taskAppend` 更新操作计数并退出内部 group。即使用户没写 `ncclGroupStart/End`，单次 API 也有隐式 group 边界。

`ncclInfo` 是栈上临时对象，API 返回后即失效。所以 `taskAppend` 系列会把所需字段拷入生命周期更长的 task；reduction op 也会转换为设备侧表示，避免内部执行依赖即将销毁的主机 op handle。

源码：[enqueue.cc](../nccl/src/enqueue/enqueue.cc)，`nccl/src/enqueue/enqueue.cc:3478`，`ncclEnqueueCheck`；
`nccl/src/enqueue/enqueue.cc:3337`，`taskAppend`；`nccl/src/enqueue/enqueue.cc:2757`，`collTaskAppend`。

**注意：** 复制进 task 的是元数据，不是把 64 MiB 梯度复制进 task；用户 buffer 仍必须保持有效。

## 3. group 是收集与提交边界，不是全局 barrier

本节回答：group 关闭的那一刻，主机才开始做什么？

最外层 `ncclGroupEnd` 才触发这一组工作的处理。在它之前，NCCL 可以收集多个 collective、多个本进程 GPU 的请求，统一连接、调优和打包。单线程管理多 GPU 时，这个收集作用尤其关键：先把各成员的调用收齐再一起提交，线程就不会卡在第一个成员的提交里、让其余成员迟迟得不到提交机会。

```text
主机 API 意图             group 关闭后的主机处理               执行侧
AllReduce -> info --+
AllReduce -> info --+--> tasks -> 调优/连接/注册 -> plans --+--> GPU work
Send/Recv -> info --+                                     +--> proxy ops（若需要）
```

这个图是结构示意，不表示本例调用了额外的 Send/Recv。

若 group 汇集了多个 stream，launch 准备会建立这些 stream 与发射 stream 的依赖：内部可能先汇合，再分发完成依赖。

**注意：**
- Group 不自动修复不同 rank 的调用顺序、count 或 datatype 不匹配。
- Group 不等于 MPI barrier，也不保证所有 GPU 在同一个物理时刻开始执行。
- “在同一个 group 中”不等于“各 stream 从此完全独立”。
- 非阻塞 communicator 的 `GroupEnd` 还可能返回 `ncclInProgress`，须按 [05](05-communicator.md) 查询。

## 4. 默认主线：task 的加工过程

本节回答：一个操作意图怎样被加工成“待调度 task”？

默认路径下，本例进入 `collTaskAppend`，生成 `ncclTaskColl` 并放入 planner 的 sorter。此时 task 主要表达操作意图；算法、协议、channel 使用上限等，会在后续准备中确定。

`groupLaunchLegacy` 先处理异步任务及 preconnect，再调用 `ncclPrepareTasksAndCollPreconnect`。其中的 `ncclPrepareTasks` 按函数、op、datatype、大小与调度约束整理 tasks：相近大小的兼容操作可以形成调优聚合，某些 per-call 配置又要求隔离，不能任意混在一起。

假设本例最终选择 Ring/Simple——这只是讲解分支，不承诺 64 MiB 必然采用该组合。准备阶段把算法、协议、warps、channel 上限等写回 task。若该算法的 runtime connections 尚未建立，先标记连接需求，由 group 的连接 job 补齐。随后 `ncclTasksRegAndEnqueue` 完成相应的 buffer 注册和设备工作描述准备。

交给后续调度/发射的前提，是所需连接、注册和描述已就绪，而不是只完成了调优或标记；对端的连接协作仍可能让主机准备等待。

```text
task：{grad, count, Sum, Float}
   |
   +--调优--> {Ring, Simple, nWarps, nMaxChannels, devFuncId}
   |
   +--需要时--> algorithm preconnect / buffer registration
   |
   v
待调度 task + ncclDevWorkColl + cleanup ownership
```

源码：[group.cc](../nccl/src/group.cc)，`nccl/src/group.cc:748`，`groupLaunchLegacy`；
[enqueue.cc](../nccl/src/enqueue/enqueue.cc)，`nccl/src/enqueue/enqueue.cc:423`，`ncclPrepareTasks`；
`nccl/src/enqueue/enqueue.cc:351`，`ncclTasksRegAndEnqueue`。

**注意：**
- `trafficBytes` 是调度使用的流量估计，不是“API count 的另一种单位”，也不是实测链路字节数。
- 调优聚合是共享调优/调度决策，不意味着数学上把两次用户 AllReduce 合成一项结果。
- 注册是“让已分配的内存满足某条访问路径的要求”，不是再分配一份用户梯度。
- 注册失败、连接未就绪与算法不支持是不同原因，应沿状态产生的位置分别诊断。

## 5. task、plan、channel：三个粒度各管一件事

本节回答：task、plan、channel 各负责哪个粒度？

**Task** 对应可调度的操作意图；**plan** 汇集一次发射所需材料；**channel** 提供通信并行分工。三者不是同义词，也不是固定的一一对应。

`ncclLaunchPrepare` 反复从队列取任务，受 kernel 参数空间、work FIFO（存放设备工作描述的队列）等预算约束，生成若干 plans。一个 group 可以产生多个 plans；多个兼容 task 也可以装进同一 plan。所以 kernel 发射次数由 plan 划分决定，不能用“API 次数”直接推算。

对普通 collective 分支，`scheduleCollTasksToPlan` 把工作铺到 channels，计算各部分数据范围和 chunking。Channel 带有拓扑阶段确定的 ring/tree 邻居，但这一次任务可能只使用其中一部分 channels。多 channel 能并行利用链路和 GPU，也会消耗更多 CTA/SM 资源，不是越多越好。

`ncclKernelPlan` 保存 `channelMask`、`kernelFn`、参数、work queue、proxy queue 以及清理责任。在普通 `ncclLaunchKernel` 分支，grid 大小由 `channelMask` 中的置位数决定，体现一 channel 一 CTA 的映射。

源码：[comm.h](../nccl/src/include/comm.h)，`nccl/src/include/comm.h:212`，`ncclTaskColl`；
`nccl/src/include/comm.h:357`，`ncclKernelPlan`；
[enqueue.cc](../nccl/src/enqueue/enqueue.cc)，`nccl/src/enqueue/enqueue.cc:644`，`scheduleCollTasksToPlan`；
`nccl/src/enqueue/enqueue.cc:1695`，`ncclLaunchPrepare`。

**注意：**
- Plan 是 NCCL 的软件发射计划，不是硬件 command buffer；选好的 kernel 仍须通过 CUDA 提交。
- Channel 不是 stream：前者承载算法邻接与数据分工，后者管理执行依赖。
- 不要把“一 channel 一 CTA”这个局部实现关系，扩展为“一 channel 永远等于一个 SM”或“所有特殊算法都是这个布局”；CUDA 仍负责 block 调度，SM 不是 communicator 私有资源。

## 6. CPU 与 GPU 交接的是 work descriptor，不是 C++ planner

本节回答：GPU 拿到的执行输入是什么？用完后各方怎么回收？

`ncclDevWorkColl` 是 GPU 能消费的紧凑工作描述：包含 buffer、channel 范围、warps、分块计数及 op 参数。它与描述函数/类型的 work batch、kernel 参数及设备侧 comm 一起，构成 kernel 的执行输入。因此不能只找到一个 `sendbuff`，就认为已经找齐了所有算法信息。

工作描述可放在 kernel 参数中，也可引用 FIFO 或 persistent storage。`finishPlan` 会在工作足够小时选择参数内承载；`uploadWork` 为相应布局准备 GPU 可见数据。这里上传的仍然主要是**执行描述**，不是通过 CPU 上传整个梯度。

数据与状态的实际交接，可按下面这张表阅读：

| 交接 | 携带什么 | 何时不能过早回收 |
| --- | --- | --- |
| API → task | 用户指针、count、op、stream 相关状态 | 内部仍需生成计划时 |
| task → plan | 调优结果、channel 分工、清理回调 | plan 尚未消费完时 |
| plan → GPU | kernel args、work batches、work descriptors | 描述存储仍被 GPU 使用时；work FIFO 按下述完成事件返还容量 |
| plan → CPU proxy | protocol、channel、nsteps、buffer/registration 信息 | proxy 尚未取得独立描述及所需资源引用时；运行期资源仍须覆盖网络操作 |
| GPU/proxy → 资源池 | 消费/完成状态、回调 | 不可仅因 host API 已返回就复用 |

回收要分两条路径，**主机 task/plan 与 GPU 可见 work FIFO 走不同的回收路径**：

- **task/plan 的主机回收**：非 persistent plan 在 proxy ops 交接后即可进入主机回收队列，不必等待整个通信完成。见 [enqueue.cc:1558–1574](../nccl/src/enqueue/enqueue.cc#L1558)。
- **work FIFO 的容量返还**：本版普通非捕获路径以 launch stream 上的完成事件为门槛——主机确认事件完成后，才推进 `workFifoConsumed`、返还对应容量，而不是 GPU 刚读完描述就立即返还。见 [enqueue.cc:2028–2056](../nccl/src/enqueue/enqueue.cc#L2028) 与 [comm.h:916–935](../nccl/src/include/comm.h#L916)。

无论哪条路径，这仍只是本地执行进度，不是所有 ranks 或远端应用都已完成的全局通知。

源码：[device.h](../nccl/src/include/device.h)，`nccl/src/include/device.h:287`，`ncclDevWorkColl`；
[enqueue.cc](../nccl/src/enqueue/enqueue.cc)，`nccl/src/enqueue/enqueue.cc:236`，`finishPlan`；
`nccl/src/enqueue/enqueue.cc:1365`，`uploadWork`。

**注意：**
- Work FIFO 是供 NCCL kernel 读取的描述存储，不是硬件提交队列；描述就绪后仍须经 CUDA 发射并满足 stream 依赖，GPU 才能消费它。
- 本例 `grad` 的所有权仍在应用；NCCL 管理内部描述、缓冲和注册引用，不替应用延长任意内存的生命周期。
- 若被 CUDA Graph 捕获，persistent plan 的寿命还会延伸到图引用释放，而不止一次 replay。

## 7. 发射不是结束：GPU work 与 proxy op 两路推进

本节回答：launch 之后，哪两路执行者在推进本次通信？

`doLaunches` 协调本进程相关 communicators 的准备和逐 plan 发射。普通 kernel 路径进入 `ncclLaunchKernel`，选择 CUDA driver 的 kernel launch 接口并传入参数。随后 GPU 才真正等待 stream 依赖、读取 work，开始执行算法与协议。

对需要 CPU progress 的网络连接，plan 还会形成 `ncclProxyOp`：描述 channel、协议、步数、chunk/slice、buffer 与注册 handle，供 proxy 线程推进网络操作。`ncclAddProxyOpIfNeeded` 会先询问该连接是否需要。

```text
                   plan
             +-------+-------+
             |               |
       GPU work descriptor  proxy op（按连接需要）
             |               |
       GPU 算法/规约       CPU 网络请求/完成推进
             |               |
             +--共享的 step / FIFO / head-tail 状态--+
                                                    |
                                             分片完成、资源复用
```

两路的分工随 transport 而不同。对于常规 NET，GPU 可能已启动，却在等待网络数据或 credit；此时 CPU proxy 的进度会影响 GPU 的进度。对于普通 P2P/SHM，数据协议不必由 host proxy 逐步推进。具体谁移动 payload、谁发布完成，可回查 [07](07-transports.md)，再在 [09](09-device-protocols.md) 追设备协议。

现在可以回答开头的 rank 7 问题了。本地梯度由前序 stream 依赖保证可读；远端分片则依赖协议的数据可见性与状态发布，二者都不能由 API 返回代替。rank 7 缺席时，其他 ranks 可能停在 GPU 协议等待：本地 launch 不确认所有 peer 已匹配，协议等不到对端数据或 credit，分片便无法继续。若存在全局参数检查或未完成的连接协商，等待也可能提前落在主机侧——成功预热的例子只用于隔离这些前置等待，不能保证所有实际运行都停在 GPU。

<details><summary>深入：协议等待与发布的源码对照</summary>

对照 [prims_simple.h](../nccl/src/device/prims_simple.h) `nccl/src/device/prims_simple.h:103` 的 `waitPeer` 与 167 行的 `postPeer`：前者等待 step/credit 等协议状态，后者按条件 fence 后发布 step。两者都是 GPU 侧协议动作，不是 CPU API 完成通知。

</details>

要分别取证的是 **CPU 提交结束、跨 rank 调用匹配、GPU 所需数据可见、本次通信完成**——它们不是一个 ready 位。匹配与数据到达可以同本地执行交叠，“可发射”不等于所有输入已到齐。通信未完成时，同 stream 后面的消费 kernel 只能等待，应用也不能因为提交成功就读取或复用 `grad`。恢复 rank 7 的匹配提交，只是给协议继续推进的机会；正常结束后，同 stream 消费 kernel 才能使用结果，CPU 则须同步相应 stream/event，并用结果校验确认正确性。

源码：[group.cc](../nccl/src/group.cc)，`nccl/src/group.cc:427`，`doLaunches`；
[enqueue.cc](../nccl/src/enqueue/enqueue.cc)，`nccl/src/enqueue/enqueue.cc:1886`，`ncclLaunchKernel`；
`nccl/src/enqueue/enqueue.cc:140`，`ncclAddProxyOpIfNeeded`；
[proxy.h](../nccl/src/include/proxy.h)，`nccl/src/include/proxy.h:72`，`ncclProxyOp`。

**注意：** Proxy op 不是 CUDA kernel，也不是整个 AllReduce 数据的主机副本；setup 阶段出现了 proxy，不代表数据协议每步都由 proxy 推进，也不代表每个 task 都有一个必需的网络线程。

## 8. 版本与路径分歧：不要被目录名误导

本节回答：哪些是默认路径，哪些只是准备框架或分支？先给四条结论，证据在折叠块里。

1. **默认仍是 legacy enqueue。** [enqueue.cc](../nccl/src/enqueue/enqueue.cc) 的 `nccl/src/enqueue/enqueue.cc:33` 定义 `EnqueueRearchEnable`，默认值 **0**；[group.cc](../nccl/src/group.cc) 的 `nccl/src/group.cc:1031`，`groupLaunch`，才是 legacy/rearch 的明确选择点。新目录不等于默认实现。
2. **Rearch 已重组任务准备，但不能写成完整的新发射后端**：实际 group 发射仍回落到 legacy `doLaunches`。
3. **`src/scheduler/` 不是上述占位框架的别名**，它包含实际被调用的 symmetric、AllGatherV 等调度逻辑。
4. **本章案例不代表所有 host API 分支。**

<details><summary>深入：rearch 的准备链路与执行线程分派</summary>

开启 `NCCL_ENQUEUE_REARCH_ENABLE=1` 后，`taskAppend` 使用 `rawTaskAppend`，保留 raw task。
[group.cc](../nccl/src/group.cc) 的 `nccl/src/group.cc:922`，`groupLaunchEnqueueRearch`，分别处理管理任务与准备任务。
[task_prep.cc](../nccl/src/enqueue/task_prep/task_prep.cc) 的 `nccl/src/enqueue/task_prep/task_prep.cc:11`，`ncclTaskPrepare`，顺序是：

```text
rawTaskQueue -> pre-tuning -> ncclTuningCompute
             -> classification -> post-tuning -> 可供既有 launcher 消费的队列
```

[task_posttuning.cc](../nccl/src/enqueue/task_prep/task_posttuning.cc) 的 `nccl/src/enqueue/task_prep/task_posttuning.cc:1012`，`ncclTaskPostTuning`，处理后续准备。
实际 group 发射在 `nccl/src/group.cc:991–995` 仍回落到 legacy `doLaunches`，但该处注释中的“用户线程”不能概括所有配置。
执行线程由 group 的 blocking 模式决定：blocking 在调用线程上执行；nonblocking 则由 group 后台线程经 `groupLaunchNonBlocking` 进入同一个 `groupLaunch`。分支见 `nccl/src/group.cc:1031–1036` 与 `nccl/src/group.cc:1131–1139`。
线程归属的变化，不改变 rearch 回落到 legacy `doLaunches` 这一事实。
[task_sched.cc](../nccl/src/enqueue/task_sched/task_sched.cc) 的 `nccl/src/enqueue/task_sched/task_sched.cc:12`，`ncclTaskSchedule`，也保留未来 scheduler 的注释框架，当前回落到 `doLaunches`。因此不能声称真实主链依次调用了所有 `task_sched/*` 专用调度器。

</details>

<details><summary>深入：src/scheduler/ 中真实被调用的调度逻辑</summary>

[symmetric_sched.cc](../nccl/src/scheduler/symmetric_sched.cc) 的 `nccl/src/scheduler/symmetric_sched.cc:79`，`ncclMakeSymmetricTaskList`，可在 legacy preparation 中抽取适用任务；
`nccl/src/scheduler/symmetric_sched.cc:268`，`ncclSymmetricTaskScheduler`，由 launch preparation 的对应分支调用。
“legacy”不意味着只有旧式 Ring/Tree，“rearch”也不意味着所有算法都换了实现。

</details>

**注意：**
- 空 collective、单 rank、对称注册窗口、CTA policy、CE availability 都可能改变路线。本版 `taskAppend` 有显式 CE 分流；`ncclLaunchPrepare` 也有 CE、RMA、symmetric 队列分支。
- Device API 在用户 kernel 内发起的通信，不能套用本章这条完整 host 调用链；扩展见 [13](13-advanced.md)。
- CUDA Graph 捕获会改变描述和回调寿命；ROCm/RCCL 则必须重新对照其自身源码。

## 9. 实操观察：把四种时间分开

本节回答：怎样用日志与 timeline 区分“CPU 还在准备”和“GPU 已在等待”？

以下命令是 Linux/CUDA 读者指引，本机未执行 CUDA profiling，也没有实测 timeline。用同一八 rank 应用，分别记录四种时间：API 时长、group 时长、首轮 stream 完成、预热后 stream 完成。多机启动器须一致传递环境；日志文件名应含主机与 PID，避免互相覆盖。

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,INIT,TUNING,NET,PROXY \
NCCL_DEBUG_FILE=/tmp/nccl-host.%h.%p.log ./your_nccl_app
```

有 Nsight Systems 的节点，可在每个 rank 的独立 shell 包装脚本中采集，使用不同输出路径：

```bash
nsys profile --trace=cuda,nvtx,osrt \
  -o "/tmp/nccl-${HOSTNAME:-node}-$$" ./your_nccl_app
```

判读时先问“CPU 在准备/连接，还是 GPU 已开始但在等待”，再比较 kernel 数量和 API 数量。

**注意：**
- 无 GPU kernel 不能立即判定漏发射：先排除单 rank、零 count、CE 与捕获尚未 replay 等情况。
- 日志中的 API 事件也不证明网络 payload 已完成；完成仍需 stream/event 与正确性校验。
- 可比较一个 group 中提交两次 collective 与分开提交，但不要预设它必然合为一个 kernel。
- 若研究 rearch，应做独立受控实验，不把内部开关当作稳定性能建议；测量方法见 [10](10-nccl-tests.md)。

### 源码追踪/验证任务：把已提交与可完成分开取证

复用本章同一操作，按 **生产者/执行线程 → 持有状态 → 消费者 → ready 条件 → 完成证据 → 外部依赖** 记录每一跳，不以函数名代替证据。

1. 从 `nccl/src/collectives.cc:192` 的 `ncclAllReduceConfigImpl` 到 `nccl/src/enqueue/enqueue.cc:3478` 的 `ncclEnqueueCheck`，再到 2757 行 `collTaskAppend`；对照 2802–2804 行，记录指针与 count 如何离开栈上 info 进入 task。
2. 在 `nccl/src/group.cc:1031` 的 `groupLaunch` 确认分支；默认值依据是 `nccl/src/enqueue/enqueue.cc:33` 的 0。若另追 rearch，必须记下 `nccl/src/group.cc:991–995` 仍回落到 `doLaunches`。
3. 沿 `nccl/src/group.cc:427` 的 `doLaunches` 到 `nccl/src/enqueue/enqueue.cc:1695` 的 `ncclLaunchPrepare`；记录 task 消费者、plan 的 `channelMask`、`workStorageType` 和连接准备已完成的前提。
4. 对照 `nccl/src/enqueue/enqueue.cc:1365` 的 `uploadWork` 与 1886 行 `ncclLaunchKernel`，记录描述落在 Args/FIFO/Persistent 哪一支，以及 kernel 参数交给哪个 CUDA launch 调用。
   再看 [common.h](../nccl/src/device/common.h) `nccl/src/device/common.h:420` 的 `ncclKernelMain`：475 行加载 work，484–486 行执行；描述已可读不等于整个通信已结束。
5. 从 `nccl/src/device/prims_simple.h:103` 的 `waitPeer` 与 167 行 `postPeer`，记录等待的 step/credit 条件和发布者；注明本地提交结束没有确认远端本轮调用匹配。
6. 可选 GPU 验证：仅在自己拥有或获明确授权的隔离实验作业中进行，不用于生产或共享作业；使用已有应用、默认检查、普通非捕获 Ring/Simple 路径，先让所有 ranks 成功预热，无需修改示例代码。
   用支持 non-stop 的调试器，仅将 rank 7 的应用线程停在下一轮 `ncclAllReduce` 入口，保持 proxy 等其他线程运行；预先设置短 deadline 并准备恢复该线程。
   结合各 rank 日志与 timeline，记录其他 ranks 的 CPU API 是否已返回、CUDA launch 是否已提交、GPU kernel 是否已开始，以及后续消费工作是否仍在等待。
   恢复 rank 7 的同序号匹配调用，让采集正常结束；以应用现有的 stream 同步或通信之后记录的 event，加结果校验，作为正常完成证据。
   如果实际卡在主机准备而未发射，就记录实际停点；不能把“缺席”预设成已经观察到 GPU 等待。

最终用实际证据标出四个边界：CPU 提交、peer 匹配、GPU 数据可见、通信完成；仅有日志/timeline 不能直接观测的协议可见性，须标为源码推断。

## 10. 自测与简答

1. `ncclInfo` 在栈上，API 返回后 GPU 为什么仍能工作？  
   **答：**所需元数据已复制到 task/plan/设备工作存储；用户 buffer 仍须保持有效。
2. 一个 group 中有十次 AllReduce，是否一定启动十个 kernel？  
   **答：**不一定；任务受兼容性与预算影响，可以合并成 plan，也可能拆成多个 plans。
3. 看到 `enqueue/task_sched/` 就说本版默认使用全新 scheduler，错在哪里？  
   **答：**默认开关是 0；rearch group 仍回落到 legacy `doLaunches`，不能以目录名代替调用证据。

下一站：[09｜设备协议](09-device-protocols.md)，沿 GPU 对 work 的消费继续追踪数据可见性与协议进度。
