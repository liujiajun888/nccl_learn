# 14 带着问题读源码：模块地图与六次阅读任务

## 学习目标

- 用文件、符号和状态流定位问题，而不是从头逐行读完整仓库。
- 分清公共契约、当前实现和教学模型的证据强度。
- 建立一个能随版本变化而更新的阅读方法。

## 随主线使用：从 UMD 的状态追踪方法入手

本章可随 01～12 主线查阅，不必等读完 13：读 05 配初始化任务，读 06 配拓扑任务，读 08 配 API/提交任务，读 09 配设备原语任务，读 10 配测试任务。算法手算与 transport 追踪分别随第 04、07 章进行，不要求先补完基础带练。

你可以沿用驱动调试中追“资源—命令—完成”的方法，但把跨 rank 的依赖一起记下。每到一个关键状态，填四项：

| 状态或对象 | 谁生产/更新 | 谁消费/等待 | 哪个条件允许继续或复用 |
|---|---|---|---|
| 用户 collective 意图 | 应用线程、API 封装 | enqueue/task preparation | comm 可用、参数满足契约；操作完成还需各 rank 匹配，但本地入队不等于已确认匹配 |
| kernel plan / device work | NCCL host 准备与调度 | launcher / device kernel | 工作描述已就绪且被正确提交，存储寿命覆盖设备使用 |
| 一条 transport 连接 | 两端 setup/connect 及服务逻辑 | kernel 或 proxy | 所需 handle、映射、注册与连接状态已就绪 |
| 一片数据的 ready/credit | GPU、proxy 或网络完成路径，视连接而定 | 下游消费者或上游生产者 | ready 防止读未发布数据，credit 防止覆盖未消费数据 |

表是定位问题的模板，不是按行顺序执行的调用栈；尤其最后一行必须进入具体协议分支，不能用一个抽象的“完成”包办。

**一次最小追踪的交付：**选普通 host API AllReduce，记录 `comm、输入输出范围、rank/channel、所走分支`；从一个工作描述追到一个 primitive，再说明它等待哪个 peer 状态。数据量与控制描述分别画，不把 plan 画成用户 payload，也不把 work FIFO 当硬件提交队列。

用证据约束结论：源码告诉你当前分支如何组织，日志帮助确认实际选路，CUDA 时间线显示执行/等待关系，正确性检查验证当前用例输出。若没有对应运行环境，就交付源码与纸面推导，并把尚待实验确认的选择、延迟和吞吐单独列出。

## 1. 先确认你正在读哪一版

```bash
git -C nccl rev-parse HEAD
git -C nccl-tests rev-parse HEAD
```

本教材基线见[首页](README.md)。文件链接是相对于教材目录的；行号写成 `nccl/src/collectives.cc:192`，指原始仓库文件，不是 Markdown 页内行号。

更新版本后，应搜索函数名重新定位，不能根据行号硬套。2.32.3 已经把很多 enqueue/tuning 实现拆到子目录，旧教程中的 `src/enqueue.cc` 或 `src/graph/tuning.cc` 不一定是当前路径。

## 2. 按问题找目录

| 你要找的问题 | 本地入口 | 先关注什么 |
|---|---|---|
| 参数与完成语义 | [src/nccl.h.in](../nccl/src/nccl.h.in) | 注释、类型、API 契约 |
| collective 入口封装 | [src/collectives.cc](../nccl/src/collectives.cc) | `ncclInfo` 如何填充 |
| communicator 初始化 | [src/init.cc](../nccl/src/init.cc) | init job、设备、bootstrap、topology |
| rank 怎样认识彼此 | [src/bootstrap.cc](../nccl/src/bootstrap.cc) | 引导地址、信息交换、连接 |
| group 何时真正执行 | [src/group.cc](../nccl/src/group.cc) | 外层边界、blocking、legacy/rearch |
| task/plan/launch | [src/enqueue/](../nccl/src/enqueue/) | 队列、计划、工作描述、kernel launch |
| 调度与资源安排 | [src/scheduler/](../nccl/src/scheduler/) | 分配何种任务和资源 |
| 硬件图与路径 | [src/graph/](../nccl/src/graph/) | topo、paths、search、connect |
| 成本与算法选择 | [src/tuning/](../nccl/src/tuning/) | 候选过滤、成本模型、参数 |
| transport 选择 | [src/transport.cc](../nccl/src/transport.cc) | 可连接性与 setup/connect |
| P2P/SHM/NET 实现 | [src/transport/](../nccl/src/transport/) | 缓冲、handle、注册与进度 |
| host 网络推进 | [src/proxy.cc](../nccl/src/proxy.cc) | 操作入队、请求、完成检测 |
| device collective | [src/device/](../nccl/src/device/) | 算法循环、原语调用、同步 |
| 注册与对称内存 | [src/register/](../nccl/src/register/) | 注册缓存、生命周期与路径条件 |
| device API | [src/nccl_device/](../nccl/src/nccl_device/) | 从应用 kernel 使用的接口 |
| GPU 发起网络 | [src/gin/](../nccl/src/gin/) | GIN 上下文和后端 |
| RMA | [src/rma/](../nccl/src/rma/) | window、远程操作与信号 |
| 插件 | [src/plugin/](../nccl/src/plugin/) | net、tuner、profiler 等集成 |
| 运行状态诊断 | [src/ras/](../nccl/src/ras/) | 诊断服务与状态 |
| 参数系统 | [src/param/](../nccl/src/param/) | 默认值、配置读取与查询 |
| 测试主体 | [nccl-tests/src/common.cu](../nccl-tests/src/common.cu) | rank映射、预热、计时、校验 |
| 某操作测试定义 | [nccl-tests/src/all_reduce.cu](../nccl-tests/src/all_reduce.cu) | count、执行、带宽归一化 |

这是导航图，不是说“所有路径都会经过表里每一行”。例如 device API、CE 和普通 host collective 的关键工作路径不同。

## 3. 第一次阅读：只追一个 API 的参数

起点：`nccl/src/nccl.h.in:602` 的 `ncclAllReduce`。

接着看 `nccl/src/collectives.cc:192` 的 `ncclAllReduceConfigImpl`，再到 `nccl/src/enqueue/enqueue.cc:3478` 的 `ncclEnqueueCheck`。

只回答：

- count、datatype、op、stream、comm 被放进什么结构？
- 参数检查是在 host 层完成，还是留给后续工作？
- 为什么公共函数本身没有循环读每一个数组元素？

完成标准：你能画出 `API参数 -> ncclInfo -> 后续任务`，知道规约不发生在这个薄封装里。

不要第一次就展开每个 `NCCLCHECK`、NVTX 宏、模板和 allocator。它们很重要，但不是这次问题的主因果链。

## 4. 第二次阅读：找到“从收集到执行”的边界

看 `nccl/src/group.cc:1031` 的 `groupLaunch`，注意它在当前版本存在不同路径。再看：

- `nccl/src/enqueue/enqueue.cc:1695`，`ncclLaunchPrepare`。
- `nccl/src/enqueue/enqueue.cc:1886`，`ncclLaunchKernel`。

回答：

1. group 内调用为什么不能直接等同于已启动 kernel？
2. 谁持有 task，谁生成 plan，谁消费 plan？
3. proxy 操作与 GPU work 描述如何关联？
4. 当前走哪条分支，判断条件来自哪里？

完成标准：用[第 08 章](08-host-execution.md)的模型解释一个具体调用，不把所有函数名拼成一条没有分支的假调用栈。

## 5. 第三次阅读：只理解一次初始化

看 `nccl/src/init.cc:2105` 的 `ncclCommInitRankFunc`，把本 rank 建立起来之前缺失的信息列出来：

```text
我是谁 -> 谁与我同机 -> 哪些GPU/NIC可达 -> 用什么连接 -> 如何让双方使用连接
```

沿 `bootstrapInit`（`nccl/src/bootstrap.cc:754`）和 topology 相关调用读，不深入所有异常回收分支。

关键区分：bootstrap 交换元数据和引导信息，后续 transport 才决定大块 GPU payload 如何前进。初始化并不是 CPU 把未来的所有 tensor 都分发一遍。

完成标准：解释 uniqueId 为什么需要分发、comm 指针为什么不能跨进程直接使用、为什么首次 collective 仍可能有额外连接开销。

## 6. 第四次阅读：硬件图为什么变成那条 ring

顺序：

1. `nccl/src/graph/topo.cc:1989`，`ncclTopoGetSystem`。
2. `nccl/src/graph/paths.cc:754`，`ncclTopoComputePaths`。
3. `nccl/src/graph/search.cc:1151`，`ncclTopoCompute`。

先在纸上画出两 CPU socket、四 GPU、两 NIC 的简图，再找代码如何表示节点/链路/路径。

完成标准：解释为什么 r0→r1→r2→r3 只是一个教学顺序，实际 ring 要考虑连接与资源；解释为什么图搜索与每次调用的算法选择是有关联但不同的问题。

## 7. 第五次阅读：一个块穿过 device 原语

打开 [all_reduce.h](../nccl/src/device/all_reduce.h)，从 Ring 的一个收发/规约步骤出发；再去 [primitives.h](../nccl/src/device/primitives.h) 和具体 `prims_*.h` 查对应实现。

记录四个信息：

| 问题 | 要找的内容 |
|---|---|
| 读哪里 | 用户输入、接收缓冲或已规约块 |
| 算什么 | 是否规约本地输入、是否做后处理 |
| 写哪里 | 下游缓冲、用户输出或两者 |
| 何时可前进 | 数据就绪标志、step、空间和线程同步 |

完成标准：能解释“接收并规约再发送”为什么可以融合，为什么必须防止接收者读到尚未完成的数据，以及生产者覆盖尚未消费的槽位。

## 8. 第六次阅读：亲自证明一个测试列的含义

从 `nccl-tests/src/all_reduce.cu:54` 附近的 `AllReduceGetBw` 入手，结合 [PERFORMANCE.md](../nccl-tests/doc/PERFORMANCE.md) 检查带宽定义；再读 `common.cu` 的运行循环。

不要止步于“打印了 algbw”：继续追 `count`、类型字节数、计时区间、迭代归一化和 rank 聚合。

完成标准：能够拿 `P=4, S=64MiB, T=2ms` 手算两种带宽，并说明它们为什么不是直接读取 NIC 计数器的结果。

## 9. 如何做最小源码实验

建议先保留上游基线，单独建立自己的实验分支或副本；教材不会自动改动上游仓库。

一次只做一个观察：

- 在已确认的 host 决策点记录选择，检查日志与参数是否一致。
- 改一个输入规模，观察候选或通道安排是否变化。
- 用同一 workload 对比 release 与带观测的构建，确认日志没有显著扰动结论。

不要在高频 device 循环里大量打印后，再把慢下来的结果当成原始性能。也不要删同步、强制关闭错误检查来“让测试不挂”，那可能只是隐藏了前进性或可见性错误。

## 10. 术语速查

| 术语 | 本教材中的含义 |
|---|---|
| rank | communicator 内的逻辑成员编号 |
| clique | 参与同一个 communicator 通信关系的一组 rank |
| local rank | 本节点内的进程/rank 编号，具体来源要看启动方式 |
| CUDA ordinal | 当前进程可见的设备编号 |
| bootstrap | 建立共同通信关系所需的引导与元数据交换 |
| topology | GPU/CPU/NIC 等资源及连接关系 |
| graph search | 在拓扑约束下构造通信连接安排 |
| algorithm | rank 与数据块的交互组织方式 |
| protocol | 数据传输粒度、有效标记与推进机制 |
| transport | 一对参与者间的连接及数据通道实现 |
| channel | 调度/并行通信通道，不等于一根物理线 |
| CTA / block | CUDA thread block，消耗 GPU 执行资源 |
| chunk / slice / step | 不同层级的数据划分与协议推进单位，不能一概等同 |
| proxy | 为部分路径提供 host 侧通信服务/进度的线程机制 |
| GDR | GPUDirect RDMA，设备内存与网络 DMA 相关能力 |
| registration | 建立缓冲可被目标传输使用的状态，不等于分配内存 |
| zero-copy | 省去某一段中间拷贝，需说明具体路径，非“零成本” |
| in-place | 满足该 API 契约的输入/输出复用布局 |
| algbw | 按该操作逻辑数据量除以时间 |
| busbw | 基于操作/rank 因子折算的带宽指标 |
| warmup | 正式计时前预热，减少首次执行扰动 |
| latency-bound | 固定启动/同步成本主导 |
| bandwidth-bound | 持续数据传输成本主导 |
| overlap | 有依赖约束下的计算/通信时间重叠 |
| critical path | 决定整体完成时刻的依赖链 |

## 自测与答案

1. 为什么既要看头文件，也要看实现？**头文件定义公开契约，实现解释当前如何达成；只看一边会混淆保证与细节。**
2. 找不到旧教程的 enqueue.cc 怎么办？**先查当前目录结构和符号，而不是猜代码被删了。**
3. 已读懂 Ring/Simple 就读懂所有 NCCL 路径了吗？**没有，但你已经具备比较 Tree、网络、NVLS 和新路径的共同坐标系。**

完成当前任务后，做[第 15 章的对应练习](15-exercises.md)，再回到[首页主线](README.md)的下一阶段，无需一次读完本章六个任务。
