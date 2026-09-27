# 第 13 章：高级机制——改变谁发起、谁搬运、谁持有状态

> 基线：NCCL `12df1a11`，版本 `2.32.3`；版本定义见 [makefiles/version.mk:9–11](../nccl/makefiles/version.mk#L9)。
> 本章只据该提交说明能力，不把“本基线存在”误写成“全部首次引入于 2.32”。
> `ncclCollConfig_t` 明确属于 2.32 的逐集合配置接口；其他机制应各自核对编译、运行时与硬件条件。
> 编写环境为无 GPU 的 macOS，未执行 CUDA Graph、通信或性能实验。

**整章专题选读，不是 NCCL 入门的前置。** 普通两卡 AllReduce 不需要你先掌握这里的每种机制；[单进程程序](examples/README.md)和[基础测量](10-nccl-tests.md)可按 GPU 条件与主线并行实践，尝试高级机制前保留普通集合的正确性基线。遇到具体问题再选读：提交开销明显看 13.2，缓冲反复复用看 13.3，想在用户 kernel 内融合通信看 13.4～13.6，逐调用调资源看 13.10；硬件能力与新执行路径留作专题。

## 13.1 学习目标与阅读边界

学完应能判断优化减少的是提交、搬运、SM 占用还是网络工作量，并说明状态由谁创建/推进、何时释放、失败先查哪里。
机制背景按需回查：[拓扑](06-topology.md)、[传输](07-transports.md)、[主机执行](08-host-execution.md)、[性能](11-performance.md)。

| 机制 | 主要想解决的问题 | 主要执行者/状态 |
|---|---|---|
| CUDA Graph | 每轮重复主机提交开销 | CUDA graph/exec、NCCL persistent plan |
| Buffer registration | 重复注册与中间缓冲搬运 | 主机注册缓存、IPC/NIC/multicast 句柄 |
| Symmetric memory | 跨 rank 地址关联与专用数据通路 | window、映射表、设备 runtime |
| Device API / GIN | 计算 kernel 直接发起通信 | 用户 kernel、device communicator、网络 backend |
| Host RMA | 有注册窗口的单边写与通知 | 主机入队、CE/网络 proxy、signal/context |
| NVLS / CollNet / MNNVL | 使用不同范围的互连或卸载能力 | 交换域、网络集合插件、fabric 映射 |
| 逐集合配置 / tuner | 在合法候选中定制资源和选择 | 主机配置与调优状态 |

普通 AllReduce 不必然经过每种机制；“高级”不表示总比 Ring/Tree 快，也不表示必须改写应用。

## 13.2 CUDA Graph：把重复调度变成可重放计划

重复“计算 A→通信→计算 B”时，短 kernel 的 CPU 提交间隙可能占比很大；capture 记录依赖图，instantiate 形成可执行对象，replay 再提交。
减少的是重复提交/调度成本，不是删掉通信本身，更不是免除跨 rank 参与。

```text
准备期：comm + 固定缓冲 + 必要初始化
                |
capture: [produce] -> [NCCL collective] -> [consume]
                |
instantiate -> graphExec
                |
replay 0 -> replay 1 -> replay 2 ...
                |
最后一次完成 -> 销毁 graphExec/graph -> 注销/释放相关资源
```

NCCL 主机侧辨认捕获图，保留 replay 所需的 plan、工作描述及回调资源。
[src/enqueue/enqueue.cc:427](../nccl/src/enqueue/enqueue.cc#L427) 根据捕获状态设置 `planner->persistent`；
[第 1858–1862 行](../nccl/src/enqueue/enqueue.cc#L1858) 增加 persistent 引用并注册 `persistentDestructor`。
[src/misc/strongstream.cc:132，`ncclCudaGraphAddDestructor`](../nccl/src/misc/strongstream.cc#L132) 把清理挂到 CUDA user object。
有些路径 replay 仍需要 host callback/proxy 进展；Graph 不意味着 CPU 从通信栈彻底消失。

### 示例：数据内容可变，地址关系要稳定

下面是每 rank 一线程的 API 顺序片段，省略已有的错误检查、分配和计算 kernel 定义；未在本机编译运行。

```cpp
// 先初始化 comm、stream、dIn、dOut，并完成必要预热。
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
produce<<<grid, block, 0, stream>>>(dIn);
ncclAllReduce(dIn, dOut, count, ncclFloat, ncclSum, comm, stream);
consume<<<grid, block, 0, stream>>>(dOut);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
// 每次 launch 都执行同一组已捕获的地址与依赖。
cudaGraphLaunch(exec, stream);
```

可以在正确 stream 依赖下改变 dIn 指向内存的**内容**，让 replay 处理下一批数据。
但把主机变量 `dIn` 赋成新地址，不会自动修改图中的 kernel 参数或 NCCL 内部工作描述。
也不能释放旧分配后指望 allocator 返回同一地址就安全：注册句柄和映射可能已失效。
需要改地址/布局时，保守路线是重新捕获；如采用 graph update，必须验证整个 NCCL 计划的适用性，不能只改一个表面参数。

### 一致性、限制与生命周期

各 rank 可有不同本地计算节点，但匹配集合须保持相同顺序、参数语义与 replay 次数。
不能一部分 rank 重放，另一部分跳过或走不匹配的 eager 集合；单线程多 GPU 仍须遵守 group 语义。
同一 group 给一个 communicator 的 streams 须全未捕获或属于同一捕获图，检查见 [src/enqueue/enqueue.cc:2646](../nccl/src/enqueue/enqueue.cc#L2646)。
[src/misc/strongstream.cc:92，`ncclCudaGetCapturingGraph`](../nccl/src/misc/strongstream.cc#L92) 要求构建 CUDA runtime 至少 11.3，并检查相应驱动能力。
先完成初始化、分配和必要注册，再捕获稳态路径；不能只根据应用 CUDA 头文件判断捕获支持。
缓冲、comm、device 规约参数和事件须活过全部在途 replay；等最后一次完成后，再销毁图及相关资源。
内部引用计数不允许用户提前释放内存；多图并发和 eager/graph 混用也须核对 strong stream 排序。

**可观察问题：** 分别测 capture、instantiate、首轮与稳态 replay，收益来自哪部分？
只改 dIn 变量为什么不改变图的输入地址？可安全地改变已分配内存的内容来验证，不要制造释放后使用。

## 13.3 Buffer registration：提前建立可直接访问的资格

普通通信可能先把用户数据放进 NCCL 内部缓冲，再由网络或对端取走。
注册让 NCCL 记录用户内存范围，并按传输建立可访问句柄，从而有机会绕过中转、摊薄注册成本。
“有机会”很重要：注册不是承诺所有集合、协议和拓扑都将零拷贝。

公开接口见 [src/nccl.h.in:408、412，`ncclCommRegister/ncclCommDeregister`](../nccl/src/nccl.h.in#L408)。
主机注册缓存持有页面区间、本地/graph 引用以及 NET、IPC、NVLS、CollNet 等传输的注册状态。
[src/register/register.cc:95，`regCleanup`](../nccl/src/register/register.cc#L95) 按状态释放对应资源；
第 172 行 `commDeregister` 在 `localRefs` 与 `graphRefs` 都归零后才真正清理记录。

```text
稳定用户分配
   -> ncclCommRegister -> 本地注册记录
   -> 合法算法/连接使用 -> IPC 映射或网络 MR 等实际状态
   -> 最后一次通信完成
   -> ncclCommDeregister -> 释放用户分配
```

所有参与 rank 应采用一致的注册使用方式；通信访问须在完整注册范围内，生命周期涵盖全部收发。
可从 [src/nccl.h.in:259，`ncclMemAlloc/ncclMemFree`](../nccl/src/nccl.h.in#L259) 开始实验，它面向 NCCL 优化的分配要求，但分配成功仍不等于优化已选中。
[src/register/register.cc:149–158](../nccl/src/register/register.cc#L149) 显示 LocalRegister 关闭或 P2P memcpy 模式可成功返回空 handle。
因此须区分“API 未报错”“有注册记录”“实际使用注册路径”，后者继续读 [src/register/coll_reg.cc:70–112](../nccl/src/register/coll_reg.cc#L70)。

Graph persistent 计划也可触发注册，并把清理绑定计划生命周期。
[src/register/register.cc:161，`ncclCommGraphRegister`](../nccl/src/register/register.cc#L161) 是**内部函数**，不是公开应用 API。
普通注册与 graph 注册可引用同一记录；销毁图和注销用户 handle 不能相互替代。

**可观察问题：** 用长期复用缓冲比较首轮/稳态、注册/未注册；日志是否显示实际路径，而不仅是总时间下降？

## 13.4 Symmetric memory：一致的是对象关系，不一定是裸地址

Device API、专用 symmetric kernels 和部分 CE/RMA 路径需要快速定位其他 rank 的对应内存。
窗口把“哪块逻辑内存、哪个偏移、哪个 peer”与各进程自己的虚拟地址分离。
`ncclSymPtr<T>` 保存的是 window 和 offset，不是一个可跨进程直接照抄的 CUDA 指针。
证据：[src/include/nccl_device/ptr.h:15，`ncclSymPtr`](../nccl/src/include/nccl_device/ptr.h#L15)。

```text
逻辑：window W + offset 4096 + peer=1
                         |
映射：R0 进程的本地/peer 映射     R1 进程的本地/peer 映射
      地址可不同                 地址可不同
```

[src/nccl.h.in:457，`ncclCommWindowRegister`](../nccl/src/nccl.h.in#L457) 不只是普通注册的别名，还要建立跨 rank 内存关系和设备元数据。
所有 rank 须按匹配顺序注册；单线程多 GPU 按 group 语义组织，不能只让目标 rank 私自注册。
[src/dev_runtime.cc:699，`symMemoryObtain`](../nccl/src/dev_runtime.cc#L699) 含跨 rank 交换，不是纯本地字典插入。

[src/init.cc:1934](../nccl/src/init.cc#L1934) 的 `symmetricSupport` 依赖 `isAllCudaP2p`、window 开关、cuMem，以及 GIN 连接能力或全 comm 属于一个 LSA team。
LSA 是可通过 load/store 访问的一组 rank，不必永久等同一台主机；先查询实际能力：

```cpp
#include <nccl_device.h>
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
ncclCommQueryProperties(comm, &props); // 实际程序检查返回值
// 检查 deviceApiSupport、multimemSupport、ginSupport、hostRmaSupport。
```

属性定义在 [src/include/nccl_device/host.h:113–153](../nccl/src/include/nccl_device/host.h#L113)；不支持时注册可能成功返回空 window，见 [dev_runtime.cc:1800–1815](../nccl/src/dev_runtime.cc#L1800)。
窗口要求可保留的 cuMem 句柄，子区间起点相对底层分配须满足 4096 B 对齐，见 [src/dev_runtime.cc:1180–1207，`ncclDevrWindowRegisterInGroup`](../nccl/src/dev_runtime.cc#L1180)。
所以任意 cudaMalloc 指针或 16 B 对齐地址都不是充分条件。

`NCCL_WIN_COLL_SYMMETRIC` 声明集合专用路径的对称使用，不是所有窗口必选项。
例如 AllReduce 的收发各自须有跨 rank 一致的逻辑 window/用户 offset；AG/RS 按布局检查对应一侧，不能无视合法 in-place 偏移。
见 [src/misc/argcheck.cc:95–180，对称缓冲检查](../nccl/src/misc/argcheck.cc#L95)；也不要泛化为“所有 window 大小必须永远相等”。
分段、非对称大小和 CPU-backed 扩展各有门控，后续读 [src/dev_runtime_segments.cc:44、91](../nccl/src/dev_runtime_segments.cc#L44)，不要当作默认 GPU 窗口示例。

**可观察问题：** 注册后 window 是否非空、属性是否允许所需访问？普通注册成功为何不足以回答这两问？

## 13.5 Device API：把通信嵌入用户 kernel

算子产生小片数据后立即跨卡消费，独立主机集合可能增加 kernel 边界；Device API 让用户 kernel 利用 communicator、team、窗口和同步设施自行组织流程。
它不是在 `__global__` 内直接调用主机版 `ncclAllReduce`。
主机通过 [host.h:30、151，`ncclDevCommRequirements_t/ncclDevCommCreate`](../nccl/src/include/nccl_device/host.h#L30) 声明 barrier、GIN signal/counter/context 等需求并创建 `ncclDevComm_t`。
[src/dev_runtime.cc:1979，`ncclDevCommCreate`](../nccl/src/dev_runtime.cc#L1979) 校验版本/initializer、检查对称支持并复制需求；设备对象不替代主机初始化与错误管理。
创建本身也是集合操作，communicator 的全体 rank 须按匹配顺序参与。一个 CPU 线程管理多 GPU 时，应把各 rank 的创建调用放进同一 group；逐卡 blocking 创建可能在第一张卡等待尚未发起的其他 rank。创建结束的全体 rank barrier 见 [dev_runtime.cc:1710–1712](../nccl/src/dev_runtime.cc#L1710)，创建任务的 group 入队与结束见 [2062–2070](../nccl/src/dev_runtime.cc#L2062)。

```text
主机：comm -> 查询能力 -> 注册 windows -> 创建 devComm/同步资源
设备：produce tile -> 正确发布 -> peer 通信/等待 -> consume tile
主机：全部相关 kernel/graph 完成 -> destroy devComm -> 注销 windows
```

用户须设计 barrier 参与者、context/slot 并发规则和 epoch；若驻留 CTA 等待尚未调度的 CTA，可能无法前进，不能假定全 grid 同时驻留。
先用双 rank tile 交换证明可见性和无覆盖，再融合复杂算子。
从 [src/include/nccl_device.h:11–25](../nccl/src/include/nccl_device.h#L11) 找设备接口实现；不要将普通集合内部 `Primitives` 当稳定公共 API。

**可观察问题：** 比较独立通信与设备内融合，端到端收益是否超过新增的寄存器/同步成本？

## 13.6 GIN：GPU 发起网络操作，不等于永远没有 CPU

GIN 面向 GPU-initiated networking：kernel 可以提交网络 put/get/signal，而不必每片都回到应用主机线程。
设备侧持有 context、连接、信号影子状态与 backend handle；主机仍负责建连、注册和资源回收。
本基线存在代理型与设备直接网络型 backend，不能把 GIN 统一称为“完全绕过 CPU 的 RDMA”。
[src/gin/gin_host_proxy.cc:461，`ncclGinProxyCreateContext`](../nccl/src/gin/gin_host_proxy.cc#L461) 就是代理 backend 的证据。

[src/gin/gin_host.cc:412，`ncclGinDevCommSetup`](../nccl/src/gin/gin_host.cc#L412) 依据请求类型、活动 backend、信号能力逐个尝试配置。
所以 GPU、NIC、驱动、插件、连接范围和所申请的强信号能力都可能限制可用性。
`ginContextCount` 是需求提示，不应假定运行时永远原样满足；用实际创建出的对象和能力结果编程。

最重要的是区分三种完成：

- 本地数据已可复用：不必等价于远端消费者已读完。
- 远端本次 put 已可见：可让远端消费这一片，但不自动涵盖其他 context 的写。
- 远端应用已消费完：通常还需要自己的回执/credits，才能安全覆盖目标槽位。

[src/include/nccl_device/gin.h:23–103](../nccl/src/include/nccl_device/gin.h#L23) 区分 strong/weak signal 与 weak counter。
strong signal 的语义覆盖其规定顺序域内此前 puts，weak signal 只保证绑定的 put；weak counter 只保证绑定 put 本地完成。
不能拿一次 weak signal 推断无关前序写都已完成，也不能拿本地 counter 推断远端已经消费。
进一步读该头中的 `ncclGinPut_v3`、flush 与等待接口及对应 backend 实现，明确 scope 和排序范围。

**例子：** MoE 把 token tile 写进远端槽位，附 signal 通知；远端等信号后消费，再回传 credit。
这既解释了为何需要 GIN，也解释了为什么“能 put”还不足以构成安全的循环缓冲协议。

## 13.7 Host RMA：单边语义仍然需要窗口与生命周期

本基线主机 RMA 接口 `ncclPutSignal/ncclSignal/ncclWaitSignal` 按 CUDA stream 排序，见 [src/nccl.h.in:763–842](../nccl/src/nccl.h.in#L763)。
单边写无需目标匹配 `ncclRecv`，但仍需注册窗口、维护信号及内存生命周期；`peerWinOffset` 以字节计，窗口句柄不能用远端裸指针代替。

**PutSignal 不只是注册目标窗口。** 除 `hostRmaSupport` 外，还要求 CUDA driver API 版本至少 12.5；源 buffer 须位于带 `NCCL_WIN_COLL_SYMMETRIC` 的窗口，源/目标窗口都须支持 RMA，且任一窗口都不能由多个物理 cuMem segments 支撑或含 host-backed segment。这是本版 PutSignal 的必要条件，不与 13.4 中“该 flag 并非所有窗口必选”矛盾。
证据集中在 [src/enqueue/enqueue.cc:2957–2967（能力与版本）](../nccl/src/enqueue/enqueue.cc#L2957)、[2994–3034（PutSignal 窗口检查）](../nccl/src/enqueue/enqueue.cc#L2994)。版本检查是三种 RMA 操作共有的门槛，**上述源/目标窗口要求只限 PutSignal**；Signal/WaitSignal 不携带数据窗口，但仍要求 RMA 已初始化，见 [3067–3073](../nccl/src/enqueue/enqueue.cc#L3067)。

```text
R0: produce -> PutSignal(peer=1, W, offset, sigIdx, ctx)
R1:            WaitSignal(peer=0, sigIdx, ctx) -> consume
复用同一槽位前：还需确认上一轮 consume 已结束
```

应用须规划 signal index、context 和每轮计数；`ncclWaitSignalDesc_t::opCnt` 是该描述符**本次等待的信号数（增量，须大于 0）**，不是应用传入的累计目标。
最小两轮例子：同一 `(peer, sigIdx, ctx)` 每轮发 1 个 signal，每次等待都传 `opCnt=1`。在初始计数为 0 的**非 Graph CE 路径**中，内部等待目标依次为 1、2；若误传 1、2，内部目标就变成 1、3，两轮只发两个信号会少一个。
入队时 [enqueue.cc:3116–3120](../nccl/src/enqueue/enqueue.cc#L3116) 将 `opCnt` 写入 `nsignals`；[rma_ce.cc:471–488](../nccl/src/rma/rma_ce.cc#L471) 的非捕获分支累加 `signalsHost`。同文件 [492–525](../nccl/src/rma/rma_ce.cc#L492) 的 Graph 分支使用独立信号状态的 wait/reset/ack 循环，不能把 `signalsHost` 的实现泛化到 Graph 或所有后端；公共契约仍是本次信号数。
**本基线 CE 路径中，配对的 PutSignal/Signal 与 WaitSignal 必须使用一致的 Graph/非 Graph 模式，不能跨模式消费信号。** 两种模式使用不同信号区域，见 [rma_ce.cc:56–62](../nccl/src/rma/rma_ce.cc#L56)；捕获的 PutSignal 写入 `graphSignalsDev`（[222–231](../nccl/src/rma/rma_ce.cc#L222)），非捕获 WaitSignal 却等待 `signalsDev`（[481–488](../nccl/src/rma/rma_ce.cc#L481)），即使 peer、index、context 和计数匹配，也不能由前者满足后者。
将目标等待放进消费 stream 的依赖链；成功入队不表示 CPU 已观察到完成。
[src/rma/rma.cc:19，`ncclRmaProxyEnabled`](../nccl/src/rma/rma.cc#L19) 检查跨 LSA、context、全局 proxy 支持及开关；`ncclRmaInitialized` 还查 CE 初始化与代理连接。
同文件第 43、76 行可见 CE/proxy 分支及 stream 汇合：Host RMA 既非普通 Ring 原语，也非 GIN 设备 API。
先查询 `hostRmaSupport`，但它不是调用成功的充分条件；再核对上述门控，并跟 `scheduleRmaTasksToPlan` 到 `rma_ce.cc/rma_proxy*.cc`，确认数据/信号由谁推进。

**可观察问题：** put 入队后，另一个 stream 立即读目标为何仍会错？缺的是目标等待与依赖，而非更多带宽。

## 13.8 NVLS、CollNet、MNNVL：能力可组合，范围不同

**NVLS：** 为支持 multicast/multimem 的 NVLink 域减少软件规约与复制工作。
GPU 执行相应访存/同步指令，交换互连承担相关数据路径能力；状态包括 multicast 映射、heads 与连接。
生成约束见 [src/device/generate.py:144–152，`required_cuda`](../nccl/src/device/generate.py#L144)：CUDA 12.1、SM90 起，并有限定的类型/规约组合。
运行时 [src/transport/nvls.cc:166，`ncclNvlsInit`](../nccl/src/transport/nvls.cc#L166) 还检查设备属性/API、GPU 数及重复 NVML 设备等情形。
所以“所有 Ampere+NVLink 都支持”和“所有 Hopper 机器都能无条件跑 NVLS”都不成立。

**CollNet：** 为网络侧能做集合的实现提供接口，GPU 侧还需汇聚和分发。
普通网络插件支持 send/recv，并不自动支持网络 AllReduce，更不自动支持 AG/RS。
[src/include/plugin/net/net_v12.h:147–188，`ncclCollNet_v12_t`](../nccl/src/include/plugin/net/net_v12.h#L147) 有 `reduceSupport/iallreduce/iallgather/ireducescatter`。
运行时类型与 head/本地 rank 限制见 [src/tuning/collnet.cc:11、121](../nccl/src/tuning/collnet.cc#L11)，本基线还限制最大本地参与数。
优化原因是减少域间重复搬运或在网络侧规约，而不是因为“名字里带 collective 就零开销”。

**MNNVL：** 把 NVLink fabric 的可访问范围扩展到多节点；它首先是互连/映射能力，不是一个与 Ring 同层的算法枚举。
需要 cuMem、所有 rank 的 FABRIC handle 支持、fabric 初始化完成，以及有效 UUID/clique 与 IMEX 导入导出。
这些实检集中在 [src/mnnvl.cc:14，`ncclMnnvlCheck`](../nccl/src/mnnvl.cc#L14)。
只有普通以太网连接的两台 NVLink 主机，不会因设置一个变量就变成 MNNVL fabric。
拓扑与分配状态可因此改变，`nNodes`、LSA 域和物理机数量的解释应结合实际映射，不死套“跨主机必经 NIC”。

```text
普通层级：节点内 GPU 互连 -> 节点间 NET / CollNet
NVLS_TREE：域内 NVLS     -> 域间树连接
MNNVL：跨物理节点也可能属于可直接访问的 NVLink fabric 域
```

本基线 NVLS 可用于 AR/AG/RS，NVLS_TREE 的通用设备实现用于 AR；单节点禁用后者。
多节点 NVLS 又检查 CollNet 支持，AG/RS 还查插件对应回调和 heads；详见 [src/tuning/nvls.cc:19、127](../nccl/src/tuning/nvls.cc#L19)。
不能由“支持 NVLS”推出支持任意规约，也不能由“支持 MNNVL”推出所有用户缓冲注册优化都已支持。
例如 [src/tuning/tuning.cc:236–245](../nccl/src/tuning/tuning.cc#L236) 的 EFFICIENCY 注册优化分支明确排除 MNNVL。
此外 [src/include/comm.h:879–884](../nccl/src/include/comm.h#L879) 分别判断 NVLS transport 与 symmetric multimem，关闭一个分支不应被写成关闭一切高级能力。

**可观察问题：** 日志显示 NVLS 可用，最终却选 Ring：是类型不支持、跨域条件不足、注册分支受限，还是成本估计更好？

## 13.9 插件：扩展点不是同一层

**网络插件** 扩展设备发现、建连、注册和网络提交；插件与 transport/proxy 持有连接/MR/request，NIC 搬数据。
ABI 见 [src/include/plugin/nccl_net.h:44–63](../nccl/src/include/plugin/nccl_net.h#L44)，此基线为 v12，加载器还适配旧版本。
`isend/irecv` 返回空 request 可表示暂未提交；`iflush` 关乎 GPU 可见性，提交成功不等于可读，见 [net_v12.h:89–120](../nccl/src/include/plugin/net/net_v12.h#L89)。
通过 [src/plugin/net.cc:103、357](../nccl/src/plugin/net.cc#L103) 核对 `NCCL_NET_PLUGIN` 实际加载的库/ABI，而非只看文件是否存在。

**Tuner 插件** 将特定机器/工作负载经验反馈给选择器；它持有每 communicator 的主机 context，不亲自搬数据。
本基线 [tuner_v6.h:34–80](../nccl/src/include/plugin/tuner/tuner_v6.h#L34) 有 `init/getCollInfo/getChunkSize/finalize`；chunk override 受缓冲上限约束，也不能创造缺失的设备实现。
实际候选与调用见 [src/tuning/tuning.cc:202–231](../nccl/src/tuning/tuning.cc#L202)，不要仅凭接口注释假定任何插件报错都会静默回退。
选择变量 `NCCL_TUNER_PLUGIN` 的入口见 [src/plugin/tuner.cc:39–61](../nccl/src/plugin/tuner.cc#L39)。

**Profiler 插件** 区分 API、代理、GPU 等待；持有事件句柄，由 NCCL 在相应边界通知 start/state/stop，不修改算法。
事件见 [src/include/plugin/nccl_profiler.h:11–30](../nccl/src/include/plugin/nccl_profiler.h#L11)，本基线为 v7；KernelPhase 仅覆盖 symmetric kernels，不是普通 Ring 的逐 slice 跟踪。
加载与初始化见 [src/plugin/profiler.cc:96、336](../nccl/src/plugin/profiler.cc#L96)，选择变量为 `NCCL_PROFILER_PLUGIN`。
回调须控制开销并处理并发/生命周期；API stop 不等于设备完成，跨进程时间线也要校准时钟。

**可观察问题：** 换插件后，能否确认网络 backend、tuner 选择、profiler 开销没有一起变化？一次只变一个因素。

## 13.10 2.32 逐集合配置：不要为一个 bucket 改全局环境

一个训练步骤可能既有延迟敏感的小集合，也有应该让出 SM 的大集合。
仅有 communicator 级配置或进程环境变量，难以对某个 bucket 定制资源而不影响其他操作。
[src/nccl.h.in:193，`ncclCollConfig_v23200/ncclCollConfig_t`](../nccl/src/nccl.h.in#L193) 为此提供逐调用配置。
`ncclAllReduceConfig` 等入口在 [同文件第 683–725 行](../nccl/src/nccl.h.in#L683)，传 NULL 等价于普通 API。

```cpp
ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;
cfg.algSelection = "RING_SIMPLE"; // 仅作可控对照，不是通用推荐
cfg.forceAlgSelection = 1;
cfg.userProfilerTag = 42;
ncclAllReduceConfig(send, recv, count, ncclFloat, ncclSum,
                    comm, stream, &cfg); // 每个 rank 配置一致并检查返回值
```

`minCTAs/maxCTAs/nvlsCTAs` 管资源，`CTAPolicy` 管策略；它们是调优输入，不保证指定物理 SM。
`maxCTAs` 不能超过 communicator 上限；`cgaClusterSize` 在同一 group 内必须一致，否则行为未定义。
配置必须用 initializer，应用持有配置及 ext 存储并保证在使用它的调用期间有效。
所有 rank 要设置一致配置；头文件明确 NCCL **只做本地配置校验**，不能期待它自动发现跨 rank 不一致。

`algSelection` 是合法候选过滤器；`RING_SIMPLE` 的拼写来自 [src/config/algorithm_registry.cc:30–52](../nccl/src/config/algorithm_registry.cc#L30)。
选不到时 `forceAlgSelection=1` 报错，设为 0 才允许相应自动回退；解析见 [src/config/collconfig.cc:83](../nccl/src/config/collconfig.cc#L83)。
全局 `NCCL_ALGO/PROTO/SYM_KERNEL` 对其强制的集合仍优先于逐调用筛选，见 [src/enqueue/enqueue.cc:2169](../nccl/src/enqueue/enqueue.cc#L2169)。
配置还可能使调用与其他工作隔离聚合；[src/config/collconfig.cc:25](../nccl/src/config/collconfig.cc#L25) 解释了为何定制不总是免费。

`userProfilerTag` 是关联应用任务的 opaque 标签，不改变执行；应用使用最高位为零的值。
`launchCompletionEvent` 是调用者拥有的 rank-local 事件，**不要当作集合结果已经可读的通用凭证**。
本基线要求 DisableTiming，不支持 interprocess/interop；全体 rank 要么都提供要么都不提供，一个 group 每 communicator 最多一个非空事件。
CUDA <12.3 时它在 launch 前记录；事件需活过排队等待和图执行。精确契约见 [src/nccl.h.in:211–220](../nccl/src/nccl.h.in#L211)。

**可观察问题：** 清除全局强制选项后，用 tag 区分两个 bucket，核对逐集合资源与聚合行为是否真的改变，再测整步时间。

## 13.11 CE 与新 enqueue：只先找边界，不急着重写应用

CE 路径希望让 copy engine 承担复制类通信，减少通信占用计算 CTA；它不会让 copy engine 自动执行任意求和。
[src/ce_coll.cc:170，`ncclCeImplemented`](../nccl/src/ce_coll.cc#L170) 在驱动能力至少 12.5 的分支列出 AG、AlltoAll、Scatter、Gather，不含 AllReduce。
`ncclCeAvailable` 还检查 symmetric 支持、LSA 覆盖、注册类型，并排除 CPU-backed 段；不是仅凭版本号启用。
多节点 `ncclHierCeAvailable` 在 [第 878 行](../nccl/src/ce_coll.cc#L878) 只允许 AG/AlltoAll，另要求本地 LSA 覆盖、RMA 能力与双方注册。
后续路线：先读这些 gate，再读 `ncclCeLaunchBatchOps`、[src/tuning/ce_model.cc:106](../nccl/src/tuning/ce_model.cc#L106) 和 CE scheduler，最后检查实际时间线。
不要把“ZERO CTA 策略”理解为任意算术集合都可无 SM 执行，或所有数据搬运都不再争用内存带宽。

新 enqueue 重构把原始任务、分类、调优和具体调度进一步拆开；它改变主机组织方式，不改变集合数学语义。
本提交 [src/enqueue/enqueue.cc:33](../nccl/src/enqueue/enqueue.cc#L33) 的 `NCCL_ENQUEUE_REARCH_ENABLE` 默认是 0。
入口 [第 3337–3347 行，`taskAppend`](../nccl/src/enqueue/enqueue.cc#L3337) 区分 raw-task 路径与原路径。
继续读 [src/enqueue/task_prep/task_prep.cc:11，`ncclTaskPrepare`](../nccl/src/enqueue/task_prep/task_prep.cc#L11)，
它串联 pre-tuning、成本计算、classification、post-tuning；随后 [group.cc:991 的实际发射路径](../nccl/src/group.cc#L991) 回退到 legacy `doLaunches`。
[src/enqueue/task_sched/task_sched.cc:12，`ncclTaskSchedule`](../nccl/src/enqueue/task_sched/task_sched.cc#L12) 可作为未来调度边界的阅读入口，不能把目录中的框架误画成当前 group 的实际必经调用链；细节见 [第 08 章](08-host-execution.md)。
这是本基线的可选实现路线，不要为普通集合画一张“必经新 enqueue→GIN→CE”的错误调用图。

## 13.12 接入顺序、验证与自测

建议先保留普通集合的正确性基线，再依次尝试 Graph、稳定缓冲注册、逐集合资源控制。
只有明确需要设备内融合或单边通信时，再研究 Device API/GIN/RMA；硬件卸载则先验证能力与日志。
验证方法承接 [nccl-tests](10-nccl-tests.md)；应用关键路径与清理顺序结合[训练集成](16-training-integration.md)。
始终分别记录初始化、稳态、回退路径和清理结果，不把“成功返回”当“优化已生效”。

**题 1：捕获后把主机指针换成同大小新分配，为什么不能直接重放并释放旧分配？**
答：图及 NCCL 工作描述、映射/注册仍可能引用旧对象；主机变量赋值不更新它们，旧内存必须保持有效，或走经过验证的更新/重捕获流程。

**题 2：注册了 window、put 已发出，目标能否无需任何等待直接消费并立即允许覆盖？**
答：不能。先证明远端数据可见，再消费；覆盖前还要证明上一轮消费者已结束。窗口、提交、远端可见、消费完成是四种不同状态。

**题 3：2.32 配置强制了算法却未按预期生效，首先检查什么？**
答：初始化与跨 rank 一致性、全局环境覆盖、合法候选与硬件/类型门控、force 回退规则及日志；不要先假定 tuner 或 CE 会强行补齐不存在的实现。
