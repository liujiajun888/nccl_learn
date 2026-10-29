# 附录：现代特性改变软件栈的哪一层

[返回教程主页](../../README.md)

基线为本地 NCCL `12df1a11` / `2.32.3`，版本定义见
[makefiles/version.mk:9–11](../../../nccl/makefiles/version.mk#L9)。
主线的两机八 rank、每 rank 64 MiB FP32 Sum AllReduce 已串起普通 host API 路径。
这里按开发者的接入层介绍扩展机制：改变提交方式、访问关系、执行者或资源选择。

> 阅读定位：专题索引，不建议从 Graph 到 RMA 全部顺序通读。先按当前特性选择小节；具体平台与阅读顺序见 [附录阅读指南](../README.md)。

## 1. 提交层：CUDA Graph 保留可重放计划

**问题：** 每轮“计算 → 通信 → 计算”重复提交，如何减少主机调度开销？

capture 保存依赖，instantiate 生成 graphExec，replay 重放相同地址与工作关系。
NCCL 将捕获的 plan 标为 persistent，保留设备 work 和 proxy 描述，
通过 CUDA user object 绑定清理；需要 proxy 的重放仍有 host callback 参与。

```text
固定地址与依赖 → capture → instantiate → 多次 replay
最后一次完成 → 释放 graph/exec 引用 → persistent plan 清理
```

适合结构稳定的重复工作。各 rank 保持匹配的通信顺序与重放次数，
数据内容按 stream 依赖更新，缓冲与 comm 活过全部引用。
本基线捕获支持要求构建 CUDA runtime 至少 11.3，并具备相应驱动能力。
生命周期细节接 [09](../../09-memory-and-lifetime.md)。

**入口：** [src/enqueue/enqueue.cc:427](../../../nccl/src/enqueue/enqueue.cc#L427) `ncclPrepareTasks` 设置 persistent；
[1858–1862](../../../nccl/src/enqueue/enqueue.cc#L1858) 注册析构；
[src/misc/strongstream.cc:92、132](../../../nccl/src/misc/strongstream.cc#L92)
`ncclCudaGetCapturingGraph/ncclCudaGraphAddDestructor` 处理捕获能力与引用。

## 2. 内存访问层：用户注册与对称窗口

**问题：** 如何复用内存访问准备，并直接定位 peer 的对应数据？

普通 `ncclCommRegister` 在 comm 中登记用户地址范围，
transport 按所选路径补充 MR、IPC 等状态，使长期复用缓冲可减少注册和中转成本。
应用保留分配，最后一次使用结束后注销；local/graph 引用由注册缓存分别管理。

对称内存进一步建立跨 rank 的 **window + offset + peer** 关系。
每个进程仍有自己的 VA，设备 runtime 用窗口元数据解析 peer 地址。
`ncclCommWindowRegister` 按匹配顺序由全体 rank 参与；单线程多 GPU 用 group。

窗口需要可保留的 cuMem allocation handle，子区间起点相对底层分配按 4096 B 对齐。
以 `NCCL_WIN_COLL_SYMMETRIC` 标记集合使用的对称布局，
再由具体集合核对窗口和偏移关系。

能力查询用 `NCCL_COMM_PROPERTIES_INITIALIZER` 初始化 `ncclCommProperties_t`，
调用 `ncclCommQueryProperties` 后查看 `deviceApiSupport`、`multimemSupport`、
`hostRmaSupport`；GIN 的能力字段是按类型索引的 `ginSupport[64]`。

**入口：** [src/register/register.cc:26](../../../nccl/src/register/register.cc#L26) `ncclRegister`；
[src/dev_runtime.cc:1149](../../../nccl/src/dev_runtime.cc#L1149) `ncclDevrWindowRegisterInGroup`；
[src/include/nccl_device/host.h:113–153](../../../nccl/src/include/nccl_device/host.h#L113)
`ncclCommProperties_t` 与查询接口。

## 3. 互连与执行层：NVLS、CollNet、MNNVL

**问题：** 数据路径还能利用哪些交换、规约或跨节点映射能力？

| 机制 | 改变的层与基本条件 |
|---|---|
| [NVLS](nvls.md) | GPU 使用 multicast/multimem 访问与规约，transport 管理 multicast 映射；需要支持该能力的 NVLink 域。基线设备实现从 CUDA 12.1、SM90 起 |
| [CollNet](collnet.md) | 通过集合网络插件汇聚、规约、分发；网络实现提供对应 collective 回调及类型/规约支持，GPU 仍承担域内工作 |
| [MNNVL](mnnvl.md) | 通过 NVLink fabric 扩大跨物理节点的 load/store 访问范围；需要 cuMem、FABRIC handle、已就绪的 fabric 和 IMEX 内存导入导出服务 |

三者可以组合：域内由 NVLS 处理，域间再连接其他路径。
MNNVL 首先改变拓扑与映射范围，算法仍由调优器从可用候选中选择。
开发时分别定位能力发现、transport 资源和设备实现，职责会比特性名称更清楚。
设备标识、P2P/NVLink 状态和 fabric 信息的查询方式，见 [NVML 专题](nvml.md)。

**PAT** 为 AllGather 和 ReduceScatter 提供分阶段聚合、传播的数据交换方案。本版多 rank 节点的 PAT 还可借用 NVLS 做节点内传输；从 [generate.py:104](../../../nccl/src/device/generate.py#L104) 的集合候选表和 [init.cc:1833](../../../nccl/src/init.cc#L1833) 的连接顺序进入实现。

复制类集合还有 CE 路径，由 copy engine 承担批量复制以降低 CTA 占用。
本基线 `ncclCeImplemented` 覆盖 AllGather、AlltoAll、Scatter、Gather，
其可用性还依赖驱动、对称窗口与访问域；主线 Sum AllReduce 使用规约 kernel。

**入口：** [src/transport/nvls.cc:166](../../../nccl/src/transport/nvls.cc#L166) `ncclNvlsInit`，
[src/device/generate.py:144–152](../../../nccl/src/device/generate.py#L144) `required_cuda`；
[src/include/plugin/net/net_v12.h:147](../../../nccl/src/include/plugin/net/net_v12.h#L147) `ncclCollNet_v12_t`；
[src/mnnvl.cc:14](../../../nccl/src/mnnvl.cc#L14) `ncclMnnvlCheck`；
[src/ce_coll.cc:170](../../../nccl/src/ce_coll.cc#L170) `ncclCeImplemented`。

## 4. 发起层：Device API 把通信放进用户 kernel

**问题：** 一个 tile 刚计算完成，怎样在同一 kernel 内把它交给其他 GPU？

主机先创建 comm、查询能力、注册窗口，再通过 `ncclDevCommRequirements_t`
声明 barrier、GIN signal/counter/context 等需求，创建 `ncclDevComm_t`。
用户 kernel 用 device communicator、team、窗口和同步设施组织通信。

```text
主机：comm → windows → DevCommCreate
设备：produce tile → 发布 → 通信 / 等待 → consume tile
主机：相关 kernel/graph 完成 → DevCommDestroy → 注销 windows
```

`ncclDevCommCreate` 是全 rank 匹配调用，创建结束包含全体 barrier。
单线程管理多 GPU 时，将各 rank 的创建放进同一 group。
应用 kernel 负责设计参与线程、槽位复用和同步依赖；主机继续负责资源及错误管理。

**入口：** [src/include/nccl_device/host.h:30、151](../../../nccl/src/include/nccl_device/host.h#L30)
`ncclDevCommRequirements_t/ncclDevCommCreate`；
[src/dev_runtime.cc:1979](../../../nccl/src/dev_runtime.cc#L1979) 创建入口，
[1712](../../../nccl/src/dev_runtime.cc#L1712) 是创建工作的全 rank barrier。

## 5. 网络发起层：GIN 与完成语义

**问题：** 用户 kernel 如何发起跨 NIC 的 put/get/signal？

GIN 将设备侧请求接到网络 backend。host-proxy backend 由 CPU 推进请求队列，
device-direct backend 则由设备代码直接对接网络机制；
两者都由主机准备连接、注册和 context。
使用 `ginSupport[type]` 选择可用类型，创建时声明所需的信号能力与连接范围。

通信协议可按“put tile → 通知可读 → consume → 返回 credit”组织。
其中通知和本地完成分别有明确的范围：

- **strong signal**：本次 put 及同一 context、发往同一 peer 的前序 puts 已完成落地。
- **weak signal**：只覆盖与该信号绑定的 put。
- **weak counter**：绑定 put 的源数据已被本地消费，可据此管理源槽复用。

strong/weak 描述网络操作的顺序范围；`givenRelease/requiredRelease`
描述源数据发布所需的 CUDA thread scope。
远端应用消费完成仍用自己的 credit/回执表达，便于下一轮复用目标槽。

**入口：** [src/include/nccl_device/gin.h:340–360](../../../nccl/src/include/nccl_device/gin.h#L340)
`ncclGin` 的 `put` 契约；
[src/gin/gin_host.cc:412](../../../nccl/src/gin/gin_host.cc#L412) `ncclGinDevCommSetup` 选择 backend；
[src/gin/gin_host_proxy.cc:461](../../../nccl/src/gin/gin_host_proxy.cc#L461) `ncclGinProxyCreateContext`。

## 6. Host 单边接口：RMA 的窗口、信号与 stream

**问题：** 主机如何把数据写到远端窗口，再让目标按 stream 顺序消费？

`ncclPutSignal` 写入目标窗口并通知，`ncclSignal` 单独通知，
`ncclWaitSignal` 将等待放进目标 stream。目标端用窗口和信号组织数据交接，
接收匹配由这些对象关系承担。LSA 可访问 peer 走 CE，跨域工作由网络 proxy 推进；
LSA 指能通过 load/store 互访的一组 rank。

本基线接入 Host RMA 的关键条件是：
`hostRmaSupport` 为真，CUDA driver API 版本至少 12.5，预先完成 RMA 初始化。
注册对称窗口可完成初始化；`sigIdx/ctx` 从 comm 配置范围选择，当前 `flags=0`。

对于 PutSignal，源 buffer 位于带 `NCCL_WIN_COLL_SYMMETRIC` 的窗口，
源、目标窗口都支持 RMA，且每个窗口由单个 device-backed 物理 cuMem segment 支撑。
偏移 `peerWinOffset` 按字节计，源长度由 count 与 datatype 给出。
这里保留接口契约入口，应用按实际窗口布局接入。

`ncclWaitSignalDesc_t::opCnt` 是**本次等待的正增量**。
同一 `(peer, sigIdx, ctx)` 每轮发一个 signal，两轮都传 1：

| 轮次 | 新增 signal | opCnt | 非 Graph CE 内部等待目标 |
|---|---:|---:|---:|
| 1 | 1 | 1 | 0 + 1 = 1 |
| 2 | 1 | 1 | 1 + 1 = 2 |

CE 的 Graph 与非 Graph 使用独立信号区域，配对的发送与等待采用同一模式。
非 Graph 分支累加等待目标，Graph 分支使用独立的 wait/reset/ack 循环。
目标消费排在 WaitSignal 之后，目标槽复用再接上应用的消费完成依赖。

**入口：** [src/nccl.h.in:763–842](../../../nccl/src/nccl.h.in#L763) Host RMA 公共契约；
[src/enqueue/enqueue.cc:2952–3073](../../../nccl/src/enqueue/enqueue.cc#L2952) `rmaTaskAppend` 核对能力和窗口；
[src/rma/rma_ce.cc:56–62](../../../nccl/src/rma/rma_ce.cc#L56) 信号区域，
[439–525](../../../nccl/src/rma/rma_ce.cc#L439) `ncclRmaCeWaitLaunch` 实现等待。

## 7. 扩展与调度层：插件、per-call config、enqueue rearch

**插件解决“把平台能力接在哪里”。** NET 管发现、连接、MR、request；
Tuner 修改合法候选的成本与资源选择；Profiler 接收各执行阶段的事件。
三者分别接入传输层、选择层和观测层。
入口是 [net_v12.h:76](../../../nccl/src/include/plugin/net/net_v12.h#L76) `ncclNet_v12_t`、
[tuner_v6.h:34–80](../../../nccl/src/include/plugin/tuner/tuner_v6.h#L34) `ncclTuner_v6_t`、
[nccl_profiler.h:11–30](../../../nccl/src/include/plugin/nccl_profiler.h#L11) 事件类型；开发流程见 [10](../../10-development-workflow.md)。

**Per-call config 解决“只调整这一次集合”。** `ncclCollConfig_t` 随 `ncclAllReduceConfig`
等接口传入，以 initializer 初始化，各 rank 使用相同配置。
CTA 上下界调资源，`algSelection` 筛候选，`userProfilerTag` 关联应用任务。
如用 `RING_SIMPLE` 做可控对照，默认 `forceAlgSelection=1` 会在候选无法满足时报错。
入口是 [src/nccl.h.in:193](../../../nccl/src/nccl.h.in#L193) `ncclCollConfig_t`，
[src/config/algorithm_registry.cc:30–37](../../../nccl/src/config/algorithm_registry.cc#L30) `algRegistry`。

**Enqueue rearch 解决“主机规划怎样拆模块”。** raw task 经 pre-tuning、成本计算、
classification、post-tuning 形成后续工作，适合从任务表示与调度边界切入开发。
本基线 `NCCL_ENQUEUE_REARCH_ENABLE` 默认 0；启用后的发射仍接 legacy `doLaunches`。
入口是 [src/enqueue/enqueue.cc:3337](../../../nccl/src/enqueue/enqueue.cc#L3337) `taskAppend`，
[src/enqueue/task_prep/task_prep.cc:11](../../../nccl/src/enqueue/task_prep/task_prep.cc#L11) `ncclTaskPrepare`，
[src/group.cc:991–995](../../../nccl/src/group.cc#L991) 发射交接。

## 8. Communicator 成员变化：按需查两项

| 接口 | 改变的对象与接入入口 |
|---|---|
| Split | 按 color 分组、key 排序生成新 comm；[src/nccl.h.in:330–338](../../../nccl/src/nccl.h.in#L330) `ncclCommSplit` |
| Shrink | 排除指定 rank，为保留成员生成重新编号的 comm；[src/nccl.h.in:340–346](../../../nccl/src/nccl.h.in#L340) `ncclCommShrink` |

它们改变成员关系与资源归属，正常收尾继续沿 [09](../../09-memory-and-lifetime.md) 的生命周期理解。

## 9. 离线任务：给 tile 交换选接入层

从案例中选一个跨机 tile，分别画 host PutSignal 与 kernel GIN put 的提交路线。
在图上标出窗口、请求推进者、源完成、远端可读、远端消费完成，再手算上述两轮信号。

核对点：Host RMA 从 host 入队，GIN 从设备发起但 backend 可含 host proxy；
GIN strong signal 的范围是同 context、同 peer；Host RMA 每轮 opCnt 都是 1；
CE 发送与等待同为 Graph 或同为非 Graph；窗口保留到最后一次使用与相关引用结束。

本附录只做源码核对和纸面推演，未执行 GPU 实验。

[返回教程主页](../../README.md)
