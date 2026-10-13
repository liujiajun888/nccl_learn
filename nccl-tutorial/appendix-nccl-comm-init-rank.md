# ncclCommInitRank 源码精读：从 API 入口到拓扑与路径

这篇文档汇总已经讲解过的 communicator 初始化链路，按“做了什么、为什么需要、给下一步留下什么”组织，便于反复查阅。它是 [03｜Communicator](03-communicator.md) 的逐段阅读补充，不是全部 NCCL API 的清单。

- 源码基线：NCCL **2.32.3**，提交 `12df1a11afad322be5a204a2db890161cbf8131d`。
- 最近整理：2026-10-13。源码链接指向本地仓库；换版本后请按符号重新定位。
- 主线假设：普通新建 communicator、单 unique ID、未设置 `NCCL_COMM_ID`、默认 Socket bootstrap；OS 相关机制主要按 Linux 路径解释。
- 调度主线：`NCCL_ENQUEUE_REARCH_ENABLE=0`；同时说明阻塞/非阻塞和新入队路径的差异。
- **当前精读进度：已覆盖 `initTransportsRank` 的 peerInfo 交换、能力与成员关系处理、硬件拓扑、路径裁剪重算和 CPU affinity，即 `init.cc:1412`。下一步从第 1414 行进入 CollNet/NVLS 准备与算法图规划。**
- 后续算法图、channels、实际传输资源和初始化收尾仍保留概览；特殊分支、完整异常清理及部分深层辅助函数未逐行展开。

文中的代码片段是教学节选或简化逻辑，省略了部分日志、错误处理和分支，不是可以直接编译运行的完整程序。本文基于源码阅读整理，未进行 GPU 运行验证。

## 目录

1. [先建立对象与调用链地图](#map)
2. [公开入口、配置宏与进程级初始化](#entry)
3. [ncclCommInitRankDev：创建对象并封装任务](#init-dev)
4. [Group 调度：任务放在哪里，由谁执行](#group)
5. [ncclCommInitRankFunc：参数与 GPU 环境准备](#gpu)
6. [commHash：为什么需要通信组的紧凑标识](#hash)
7. [commAlloc：为什么分配对象后还要初始化资源](#alloc)
8. [bootstrap 前提：unique ID 与 root 服务](#bootstrap-root)
9. [bootstrapInit 前半段：报到、领邻居、连接成环](#bootstrap-ring)
10. [bootstrapInit 后半段：服务入口、全组信息交换与 proxy 状态](#bootstrap-info)
11. [initTransportsRank 首段：收集身份与能力，发布 peerInfo](#peer-info)
12. [消费 peerInfo：兼容性、共同能力与同进程成员](#peer-checks)
13. [硬件拓扑与路径：建图、裁剪、重算和 CPU affinity](#topology)
14. [特殊分支与后续流程概览](#remaining)
15. [状态、错误与任务生命周期](#state)
16. [阶段小结、易混点与下一步阅读](#recap)

<a id="map"></a>

## 1. 先建立对象与调用链地图

### 1.1 一次调用初始化谁？

`ncclCommInitRank` 为当前 CUDA 设备创建一个 rank 的本地 communicator。一个逻辑通信组包含多个 ranks，每个 rank 持有自己的本地 `ncclComm_t`，不是所有进程共享同一个 C++ 对象。

公开类型是一个不透明指针，见 [nccl.h.in:35–41](../nccl/src/nccl.h.in#L35-L41)：

```cpp
typedef struct ncclComm* ncclComm_t;
```

应用使用句柄，不依赖内部 `struct ncclComm` 的布局。这样 NCCL 可以调整内部实现，而不要求应用直接访问这些内部字段。

| 对象/字段 | 回答的问题 |
|---|---|
| `comm` 指针 | 当前进程中的哪个 communicator 对象？ |
| `comm->rank` | 当前成员在通信组中是什么编号？ |
| `comm->nRanks` | 通信组共有多少个成员？ |
| `comm->cudaDev` | 当前 rank 使用本进程可见的哪个 CUDA 设备？ |
| `comm->commHash` | 这个本地对象属于哪个逻辑通信组？ |
| `job` | 这次初始化任务带着哪些参数，执行哪个回调？ |

`job` 是任务说明书，`comm` 是需要填充完善、随后长期使用的对象。任务完成后可以释放 `job`，不能因此释放成功创建的 `comm`。

### 1.2 rank 不等于 CUDA device 序号

`cudaGetDevice(&cudaDev)` 查询当前 CPU 线程的 current device，返回整数序号，不是返回 GPU 集合或已初始化设备的对象句柄。

- `cudaSetDevice(n)` 选择本线程后续 CUDA 调用所使用的设备。
- 设备序号受 `CUDA_VISIBLE_DEVICES` 过滤和重排影响。
- 这个返回值本身不能用来判断 CUDA context 是否已经完成初始化。

例如，一个跨机器通信组有 8 个 ranks，而每个进程只看见一张卡：rank 5 的 `cudaDev` 仍然可能是 0。多卡遍历由应用或 `ncclCommInitAll` 组织，不是一次 `cudaGetDevice` 完成的。

### 1.3 完整主线

```text
应用选择设备、获取并分发 unique ID
  ↓
ncclCommInitRank                         公开 API
  ├─ 环境准备、读取 current device
  ├─ ncclGroupStartInternal
  ├─ ncclCommInitRankDev                 创建本地对象、封装任务、入队
  └─ ncclGroupEndInternal                最外层 GroupEnd 触发调度
       ↓
     groupLaunch → groupLaunchLegacy → asyncJobLaunch
       ↓
     ncclAsyncJobMain → ncclCommInitRankFunc
       ├─ 选择 GPU、检查并配置 kernel 资源
       ├─ 计算 commHash
       ├─ commAlloc                     完善本地资源
       ├─ bootstrapInit                 建立初始联络通道
       ├─ initTransportsRank             拓扑、通道、连接及设备状态准备
       └─ 发布初始化结果、日志和收尾
```

`ncclCommInitRank`、`ncclCommInitRankConfig`、`ncclCommInitRankScalable`、`ncclCommInitAll` 都复用 `ncclCommInitRankDev`。`ncclCommInitRankDev` 和 `ncclCommInitRankFunc` 是内部函数，不是应用应直接调用的 API。

**本节小结：先区分“公开入口、任务描述、实际回调、长期 comm 对象”，再看各层如何交接。**

<a id="entry"></a>

## 2. 公开入口、配置宏与进程级初始化

### 2.1 ncclCommInitRank 的薄包装

源码：[init.cc:2945–2969](../nccl/src/init.cc#L2945-L2969)。省略日志和错误处理后：

```cpp
ncclInitEnv();
(void)ncclCudaLibraryInit();
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
cudaGetDevice(&cudaDev);

ncclGroupStartInternal();
ncclCommInitRankDev(newcomm, nranks, 1, &commId,
                    myrank, cudaDev, &config, __func__);
ncclGroupEndInternal();
```

这里的 `1` 是 unique ID 数量，不是 GPU 数量。`ncclCudaLibraryInit` 准备 CUDA driver 接口及相关 hooks；实际代码没有在这一行直接传播其返回值，后续 CUDA 操作仍会检查错误。

内部包一层 Group 的目的，是让单次初始化与用户显式 Group 中的多 rank 初始化复用同一套调度机制。公开入口没有简单地直接调用耗时回调。

### 2.2 NCCL_API 宏不是函数调用

源码中的 `NCCL_API(...)` 用于统一声明导出的函数，见 [core.h:16–36](../nccl/src/include/core.h#L16-L36)。它处理 C 链接和平台导出规则；Linux 的 `PROFAPI` 分支还生成弱符号与 `pnccl*` 别名，用于插桩包装。

它不会执行初始化，也不是把任务加入队列的宏。

公共声明来自构建生成的 `nccl.h`，其源模板是 `src/nccl.h.in`。普通初始化应用包含 `nccl.h` 即可；整个 NCCL 公共接口还包括 `nccl_device.h` 及配套头，不能把内部 `src/include` 目录整体当作应用 API。

### 2.3 NCCL_CONFIG_INITIALIZER 是花括号初始化宏

源码：[nccl.h.in:102–164](../nccl/src/nccl.h.in#L102-L164)。

```cpp
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
```

宏展开后是一组按结构体字段顺序排列的初始值。行末的 `\` 是预处理续行符，不是运行时逻辑。

| 配置内容 | 目的 |
|---|---|
| `size` | 告知调用者提供了多大的配置结构体 |
| `magic` | 检查是否按规定初始化 |
| `version` | 根据调用者头文件版本解释字段 |
| `NCCL_CONFIG_UNDEF_INT`、`NCCL_CONFIG_UNDEF_PTR` | 表达“用户尚未指定”，由库决定默认值 |

为什么不简单全部置零？因为 0 可能本来就是某字段的有效值，例如 `blocking=0` 表示非阻塞，而不是“没有设置”。

[parseCommConfig](../nccl/src/init.cc#L2558-L2624) 先初始化内部配置，再按 `size` 限制复制范围、检查 magic，并处理版本差异，随后校验字段、补默认值和应用环境覆盖。这是支持结构体演进与 ABI 兼容的设计。

这里不必死记全部字段。先记住：`ncclConfig_t` 管 communicator 级别配置，不是每次 collective 使用的 `ncclCollConfig_t`。

### 2.4 ncclInit 是进程级公共准备，不是一个 rank 的完整初始化

源码：[init.cc:159–198](../nccl/src/init.cc#L159-L198)。

```cpp
static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}
```

`initOnceFunc` 包含四项工作：

1. `ncclOsInitialize`：OS 层准备；Linux 路径按参数调整 CPU 栈限制。
2. `initGdrCopy`：按需初始化 GDRCopy，默认关闭。
3. `bootstrapNetInit`：选择 bootstrap 使用的本地网络接口和地址。
4. `initNvtxRegisteredEnums`：注册性能标记使用的枚举信息。

`ncclInitEnv` 则是另一套 once 包装，负责环境插件初始化。不要把它与 `ncclInit` 混成一个函数。

为什么只做一次？这些是多个 communicator 可复用的进程级准备，重复执行没有必要。`bootstrapNetInit` 只是准备网络接口信息，尚不是各 rank 之间的完整握手。

**本节小结：公开入口处理环境、设备、配置和 Group 包装；真正的逐 rank 初始化仍在后面。**

<a id="init-dev"></a>

## 3. ncclCommInitRankDev：创建对象并封装任务

源码：[init.cc:2851–2943](../nccl/src/init.cc#L2851-L2943)。

### 3.1 前置检查与运行时准备

主要包括 ID 数量检查、`ncclInit`、按日志级别打印版本、`cudaFree(NULL)`、指针检查及 rank 范围检查。

这里 `cudaFree(NULL)` 是 NCCL 用来确保 CUDA runtime 初始化的惯用调用，不是在释放应用的某个数据缓冲区。提前触发运行时准备，可以尽早发现设备/运行时问题。

### 3.2 ncclCalloc：分配并清零对象本身

```cpp
ncclCalloc(&comm, 1);
```

这不是直接调用 libc 的 `calloc`。宏会转到 `ncclCallocDebug`，见 [alloc.h:172–210](../nccl/src/include/alloc.h#L172-L210)：

```text
ncclCalloc(&comm, 1)
  → 带上源文件、行号、调用函数等调试信息
  → malloc(元素数量 × 元素大小)
  → memset 清零
  → 返回 ncclResult_t，并按日志设置记录分配信息
```

模板自动推导对象类型，所以传入 `1` 表示一个 comm 对象，不是一个字节。

为什么封装？统一处理类型大小、失败返回值和内存日志。但它不调用任意 C++ 对象的构造函数，也不会替指针字段创建子资源。

### 3.3 构造可交给任务的骨架

随后准备 `abortFlag`、设备可访问的 `abortFlagDev`、引用计数、完整性检查用的 `startMagic/endMagic`，并给 Group 链表字段设置初始哨兵。

```cpp
parseCommConfig(comm, config);
comm->initState = ncclInProgress;
*newcomm = comm;
```

为什么先写入 `*newcomm`？后续 Group 调度与非阻塞状态查询需要有一个可识别的 comm 对象。但“句柄已写出”不等于“公开 API 已返回”，更不等于“comm 可以立即参与通信”。

这些基础字段使任务具备身份、配置、状态和终止控制；此时拓扑、通道与通信连接还没有完成。

### 3.4 封装任务，并复制 unique ID

`ncclCommInitRankAsyncJob` 保存：

| 字段 | 用途 |
|---|---|
| `comm` | 待初始化对象 |
| `nranks`、`myrank` | 通信组规模和本端 rank |
| `cudaDev` | 回调执行时应选择的设备 |
| `nId`、`commId` | ID 数量及任务持有的 ID 副本 |
| `funcName` | 日志中保留原始公开 API 名称 |
| `parent`、`color/key`、`isGrow` 等 | 派生 communicator 路径的附加信息 |

复制 ID 的原因在 [2903–2910 行](../nccl/src/init.cc#L2903-L2910) 的注释里很明确：

- 任务可能延迟执行，不能依赖调用者临时参数一直有效。
- `ncclUniqueId` 是字节包装，和内部 bootstrap handle 的对齐要求不同；复制到分配的内存中解决对齐问题。

### 3.5 交给调度器

默认路径：

```cpp
ncclAsyncLaunch((struct ncclAsyncJob*)job,
                ncclCommInitRankFunc,
                NULL,
                ncclCommInitJobFree,
                comm);
```

执行函数是 `ncclCommInitRankFunc`，清理函数是 `ncclCommInitJobFree`。后者释放任务的 ID 副本并删除任务对象，见 [init.cc:2845–2849](../nccl/src/init.cc#L2845-L2849)。

若启用入队重构，则改走 `ncclMgmtTaskEnqueue`。提交前失败与提交后任务清理的责任不同，`launchedJob` 用于区分这些路径，避免把已经交给调度器的任务再由提交方直接清理。

**本节小结：这里创建的是“comm 骨架 + 初始化任务”，不是已经完成通信准备的 comm。**

<a id="group"></a>

## 4. Group 调度：任务放在哪里，由谁执行

### 4.1 默认队列属于调用线程

源码：[group.cc:28–74](../nccl/src/group.cc#L28-L74)。

```cpp
thread_local struct ncclIntruQueue<
    struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
```

`ncclAsyncLaunch` 在 Group 内保存 `func/destructor/comm` 等信息，然后把任务加入这个侵入式队列：任务自身的 `next` 字段用于串联节点。

它不是全局线程池的公共任务队列，也没有一个常驻初始化线程不停消费它。

补充边界：通用的 `ncclAsyncLaunch` 在 `ncclGroupDepth==0` 时会直接执行回调；但本章的公开初始化入口已经调用 `ncclGroupStartInternal`，正常经过入队分支。

### 4.2 最外层 GroupEnd 才触发执行

Group 支持嵌套。`ncclGroupStartInternal` 增加线程局部深度，`ncclGroupEndInternal` 减少深度；仍在嵌套层内就不提交。

最外层结束时，见 [group.cc:1061–1093](../nccl/src/group.cc#L1061-L1093)：

```cpp
ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs);
```

把线程局部队列转交给本次 Group 的任务对象，然后调度：

```text
ncclGroupEndInternal
  → groupLaunch
  → groupLaunchLegacy
  → asyncJobLaunch
  → ncclAsyncJobMain
  → job->func(job)
  → ncclCommInitRankFunc
```

为什么等到 GroupEnd？单线程管理多个 ranks 时，需要先把各 rank 的初始化任务收集齐，再一起推进。否则第一个 rank 等待其他 ranks，调用线程却尚未有机会创建其他任务。

### 4.3 是否另起线程，由两层决策共同决定

第一层看有效的 `blocking` 配置，见 [group.cc:1102–1142](../nccl/src/group.cc#L1102-L1142)：

- `blocking=1`：调用线程直接调度 Group，并等待完成。
- `blocking=0`：另起 Group 后台线程，允许调用线程返回 `ncclInProgress`。

第二层看本次任务队列有几个任务，见 [asyncJobLaunch](../nccl/src/group.cc#L609-L669)：

- 一个任务：直接在当前调度线程上执行 `ncclAsyncJobMain(job)`。
- 多个任务：为每个任务创建 `std::thread`，调度线程检查状态并 join。

| 有效配置 | 本组初始化任务数 | 回调在哪执行 |
|---|---:|---|
| `blocking=1` | 1 | 原 API 调用线程 |
| `blocking=1` | 多个 | 各任务线程；调用线程等待 |
| `blocking=0` | 1 | Group 后台线程直接执行 |
| `blocking=0` | 多个 | Group 后台线程再创建各任务线程并等待 |

`blocking` 未指定时默认取 1，但配置解析也支持环境覆盖。这里的任务数是本线程本次 Group 的数量，不是全局 `nranks`。

例如：8 个进程各初始化一个 rank，每个进程通常只有一个任务；一个线程用 `ncclCommInitAll` 初始化 8 张卡，则会累计多个任务并行执行。

这张表只描述初始化回调的执行者。bootstrap root、proxy 等内部服务线程是另一回事。

### 4.4 新入队路径的差异

启用 `NCCL_ENQUEUE_REARCH_ENABLE=1` 时：

```text
ncclMgmtTaskEnqueue → comm->mgmtTaskQueue
  → 最外层 GroupEnd → groupLaunchEnqueueRearch
  → ncclGroupJobLaunch → ncclMgmtTaskJobFunc
  → task->func(task)
```

任务放到每个 comm 的管理任务队列；外层管理任务调度仍保留单任务直接执行的优化。参考 [mgmt_task_enq.cc](../nccl/src/enqueue/mgmt_task_enq.cc#L11-L43) 和 [group.cc:304–328](../nccl/src/group.cc#L304-L328)。

**本节小结：“延迟到 GroupEnd 执行”“另起线程”“API 提前返回”是三个不同概念，不能仅凭 Async 命名判断行为。**

<a id="gpu"></a>

## 5. ncclCommInitRankFunc：参数与 GPU 环境准备

源码：[init.cc:2105–2134](../nccl/src/init.cc#L2105-L2134)。

### 5.1 从通用任务恢复初始化参数

```cpp
struct ncclCommInitRankAsyncJob* job =
    (struct ncclCommInitRankAsyncJob*)job_;
ncclComm_t comm = job->comm;
int cudaDev = job->cudaDev;
```

调度器统一接收 `ncclAsyncJob*`。具体初始化任务把 `base` 放在首成员，回调恢复具体类型以读取自己的参数，见 [任务结构](../nccl/src/init.cc#L2008-L2027)。

这个转换不会复制任务，也不是重新创建 comm。通用调度框架无需理解每种任务的全部字段。

### 5.2 选设备、查询架构能力

```cpp
timers[TIMER_INIT_TOTAL] = clockNano();
cudaSetDevice(cudaDev);
cudaDeviceGetAttribute(&maxSharedMem,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev);
cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev);
cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev);
cudaArch = 100 * archMajor + 10 * archMinor;
```

为什么再选一次设备？任务可能换线程执行。设备序号保存在 `job` 中，不代表当前执行线程已选择该设备。

`maxSharedMem` 是每个 block 可 opt-in 使用的 shared memory 上限，不是总显存，也不是多 GPU 共享的内存。这里只查询，没有为某次通信分配这块 shared memory。

`cudaArch` 是架构能力编码，例如计算能力 8.0 对应 800、9.0 对应 900，既不是 GPU 序号，也不是 CUDA Toolkit 版本。

### 5.3 准备 kernel 属性，而非执行集合通信

```cpp
ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes);
```

实现见 [enqueue.cc:55–118](../nccl/src/enqueue/enqueue.cc#L55-L118)，主要工作是：

1. 遍历常规、对称内存 kernel 及相应 profiling 变体。
2. 检查驱动版本要求，跳过不支持的 kernel。
3. 查询 `cudaFuncAttributes`，汇总 `localSizeBytes` 最大值。
4. 设置 shared memory carveout 等属性。
5. 按 `maxSharedMem - attr.sharedSizeBytes` 计算允许的动态 shared memory 上限，并检查 NCCL 的资源需求。

设置允许使用多少 shared memory，不等于某次 launch 实际就使用多少。这一步也没有发起一次 AllReduce。

### 5.4 尝试预设 GPU 线程栈限制

```cpp
if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
  CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
}
```

NCCL 使用汇总的 `localSizeBytes` 作为设置依据，尽量避免后续 kernel 加载引发 CUDA 内存重配置。这个属性不宜简单理解为精确的函数调用栈深度。

它不是 CPU 线程栈，也不同于 block shared memory。该步骤用 `CUDACHECKIGNORE`，失败不直接终止初始化；前面关键 CUDA 操作用 `CUDACHECKGOTO`，错误会跳到失败路径。

**本节小结：正确的执行设备与 kernel 资源条件已准备好；跨 rank 握手和完整传输连接尚未建立。**

<a id="hash"></a>

## 6. commHash：为什么需要通信组的紧凑标识

普通路径跳过 Split/Shrink/Grow 后，来到 [init.cc:2178–2183](../nccl/src/init.cc#L2178-L2183)：

```cpp
comm->commHash = commIdHash =
    getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
```

### 6.1 从 128 字节得到 64 位标识

[getHash](../nccl/src/include/bitops.h#L500-L543) 对输入字节做混合运算，得到 64 位值：

- `comm->commHash` 长期保存在对象里，供其他模块使用。
- `commIdHash` 是当前函数的局部变量，主要用于日志。

### 6.2 为什么不能用 comm 指针？

```text
进程 A：rank 0 的 comm 地址 = 0xAAA...
进程 B：rank 1 的 comm 地址 = 0xBBB...
```

指针只能表示进程内对象，无法直接表达两个对象属于同一个逻辑通信组。同组 ranks 拿到相同的 ID 内容后，可独立算出相同 hash，无需为这个标识额外通信。

较短的 hash 也比完整的 128 字节 ID 更方便携带、记录和比较。源码中的实际消费者包括：

- 初始化日志：关联不同进程中属于同一组的记录。
- [网络插件初始化](../nccl/src/plugin/net.cc#L191-L195)：传入通信组标识和配置。
- [Profiler 初始化](../nccl/src/plugin/profiler.cc#L338-L342)：关联通信组的分析数据。
- GIN、RMA 等插件上下文初始化。

这也解释了先 hash、后 `commAlloc` 的顺序：后者初始化插件时已经需要这个值。

### 6.3 标识的边界

| 信息 | 意义 |
|---|---|
| `comm` 指针 | 进程内对象身份 |
| `commHash` | 逻辑通信组的紧凑标识 |
| `rank` | 组内成员编号 |
| 原始 unique ID | 包含实际 bootstrap 会合所需的信息 |

hash 不是网络地址，也不能代替 unique ID；后面的 `bootstrapInit` 仍需要原始 ID。它不是加密凭据，也不保证数学上绝不碰撞。复用相同 ID 还会产生相同 hash。

**本节小结：commHash 让本地对象具有可跨进程关联的组标识，但不负责建立连接。**

<a id="alloc"></a>

## 7. commAlloc：为什么分配对象后还要初始化资源

调用处：[init.cc:2182](../nccl/src/init.cc#L2182)；实现：[init.cc:512–657](../nccl/src/init.cc#L512-L657)。

```cpp
commAlloc(comm, NULL, job->nranks, job->myrank);
```

### 7.1 对象内存与对象依赖资源是两回事

```text
ncclCalloc(&comm, 1)：分配并清零对象本身
commAlloc(comm, ...)：初始化字段、子对象和运行资源
```

清零只会让 `comm->sharedRes` 等指针为空，不会自动创建它们指向的资源。因此 `commAlloc` 更像资源初始化函数，而不是再次分配 comm。

`NULL` 表示这里不从父 comm 共享资源。形参名虽然叫 `ndev`，本路径传入的却是总 `nranks`，不能理解成本机 GPU 数量。

### 7.2 身份与两套内部内存分配器

设置 `rank/nRanks/cudaDev/compCap`，并初始化：

- `memPermanent`：管理随 comm 长期存在的内部数据。
- `memScoped`：管理可按作用域集中回收的临时数据。

它们是 NCCL 自有分配结构，非 CPU 调用栈、非 GPU 线程栈。集中分配、按生命周期回收能减少零散分配和逐个清理的负担。初始构造主要设置管理状态，见 [utils.h:235–277](../nccl/src/include/utils.h#L235-L277)。

### 7.3 sharedRes：内部 streams、events 与共享机制

普通新建路径创建 `ncclSharedResources`，包含内部 `deviceStream/hostStream`、`launchEvent/scratchEvent` 等。

[ncclStrongStreamConstruct](../nccl/src/misc/strongstream.cc#L146-L154) 确实会创建 CUDA stream。`hostStream` 也不是 CPU 线程，只是 NCCL 内部 stream 的命名。

为什么需要内部 stream/event？资源准备和执行依赖管理不能只依赖用户后续传入的通信 stream；event 可以表达先后关系。

父子 comm 在允许共享时可共同引用 `sharedRes`，以引用计数管理生命周期，避免重复创建资源。但不是所有同卡 comm 自动共享。

### 7.4 初始化通信后端与内存管理器

```cpp
ncclNetInit(comm);
ncclRmaInit(comm);
ncclGinInit(comm);
```

它们准备后端及上下文。以 [ncclNetInit](../nccl/src/plugin/net.cc#L357-L389) 为例，会检查插件状态和 `netName`，为 comm 选择网络实现。

“选好后端、初始化上下文”不等于“已经与所有 peers 建好数据连接”。后续知道对端连接信息后，才调用相应后端的连接能力。

此外，`ncclMemManagerInit` 创建资源记账与管理状态；允许共享时可复用父 comm 的管理器。它与下面的 CUDA memory pool 不是同一个对象。

### 7.5 CUDA、PCI、NVML 三个设备视角

```text
cudaDev → PCI busId → NVML 设备句柄 → nvmlDev
```

CUDA 编号是进程可见下标，PCI bus ID 对应系统硬件位置，NVML 编号用于设备管理。由于可见设备过滤和重排，不能假设 CUDA 编号等于 NVML 编号，需要通过硬件身份对应起来。

`ncclCudaContextTrack` 跟踪已有的当前 context，并维护相关 launch ordering 状态，不是每次创建 comm 都新建一个 CUDA context。

### 7.6 为后续连接和任务准备容器

| 准备内容 | 为什么需要 |
|---|---|
| Kernel plan、proxy op、raw task 对象池 | 复用高频内部对象，降低分配开销 |
| `connectSend` / `connectRecv` 数组 | 按 peer 记录哪些 channels 有连接需求 |
| `channels[c].id = -1` | 显式表示未初始化，0 本身是有效编号 |
| `topParentRanks` | 保存 rank 映射；普通新建先取恒等映射 |
| 回调、任务、清理队列 | 后续调度与延迟资源回收 |
| CUDA memory pool | 在设备支持时准备可复用的内存池 |

`connectSend` / `connectRecv` 每个元素是 channel 位图，不是一个已建立的网络连接。创建数组也不等于与所有 peers 建立连接。

CUDA memory pool 的释放阈值设得很高，是为了保留池中的内存供复用，代价是可能保留更多显存。创建 pool 不代表预先分配了所有未来通信所需显存。

**本节小结：本地资源和记录结构已准备，但完整 peer 视图、拓扑规划与实际传输连接仍在后面。**

<a id="bootstrap-root"></a>

## 8. bootstrap 前提：unique ID 与 root 服务

### 8.1 为什么需要 bootstrap？

调用处：[init.cc:2190](../nccl/src/init.cc#L2190)。

```cpp
bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId,
              comm, job->parent);
```

建立高性能连接需要先交换地址、内存句柄、连接 handle 等；可连接尚未建立，不能依赖已经完成的 NCCL collective 来交换这些信息。

bootstrap 利用基础网络能力建立最初的控制通道，再帮助完整通信系统启动：

```text
基础网络联络 → 交换初始化信息 → 建立数据传输所需资源与连接
```

默认 bootstrap 环用 TCP Socket；设置 `NCCL_OOB_NET_ENABLE` 后也可使用 Net 后端。因此“bootstrap 永远是纯 TCP”不成立；此处研究的是默认 Socket 分支。

### 8.2 unique ID 不是纯随机编号

内部 handle 见 [bootstrap.h:14–20](../nccl/src/include/bootstrap.h#L14-L20)：

```cpp
struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
  int nRanks;
};
```

| 字段 | 用途 |
|---|---|
| `addr` | bootstrap root 服务的 IP、端口等地址信息 |
| `magic` | Socket 握手时检查是否匹配预期实例 |
| `nRanks` | Grow 使用的原有 rank 数；普通初始化时为 0 |

[ncclGetUniqueId](../nccl/src/init.cc#L209-L222) 将内部 handle 复制进清零的公开 128 字节包装中。应用只应原样分发 ID，不依赖这个内部布局。

区别三个 magic/hash：配置结构体的 `NCCL_API_MAGIC` 检查初始化格式；handle 的 `magic` 参与连接匹配；`commHash` 是完整 ID 的紧凑摘要。不要因名字接近就视为同一字段。

### 8.3 root 在生成 ID 时已经启动

未设置 `NCCL_COMM_ID` 的普通路径：

```text
ncclGetUniqueId
  → bootstrapGetUniqueId
      → 生成 magic、选择本地网络地址
      → bootstrapCreateRoot
          → 创建监听 socket
          → 将实际 IP、端口写回 handle
          → 启动 bootstrapRoot 后台线程
  → 把 handle 包装为 unique ID 返回给应用
```

实现见 [bootstrap.cc:408–463](../nccl/src/bootstrap.cc#L408-L463)。线程名是 `NCCL BootstrapR`，创建后 detach。

为什么另起线程？生成 ID 的调用必须返回，应用才能继续分发 ID、创建 ranks；root 同时需要等待各 rank 报到。

root 是服务角色，不等同于 rank 0。常见程序由 rank 0 所在进程调用 `ncclGetUniqueId`，但默认源码不强制这一对应。它是该进程中的线程，不是另外启动的进程。

例外：设置 `NCCL_COMM_ID` 后走指定地址路径，本章主线假设不适用；`ncclCommInitRankDev` 会在 `myrank==0` 时创建对应 root，见 [init.cc:2912–2920](../nccl/src/init.cc#L2912-L2920)。

**本节小结：unique ID 携带“到哪里报到”的信息；root 服务通常在生成 ID 的时候就已开始监听。**

<a id="bootstrap-ring"></a>

## 9. bootstrapInit 前半段：报到、领邻居、连接成环

源码：[bootstrap.cc:754–894](../nccl/src/bootstrap.cc#L754-L894)。以下继续假设普通初始化、单 ID、默认 Socket 后端。

### 9.1 创建 bootstrapState 并挂到 comm

```cpp
state->rank = rank;
state->nranks = nranks;
state->cudaDev = comm->cudaDev;
state->abortFlag = comm->abortFlag;
state->net = comm->ncclNet;
comm->bootstrap = state;
comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic;
```

为什么独立保存？后续控制信息交换和部分管理操作仍然要使用 bootstrap，不是这一次函数返回后就全部丢弃。

### 9.2 每个 rank 准备两个监听入口

创建位置：[799–830 行](../nccl/src/bootstrap.cc#L799-L830)。

| 入口 | 谁来连接 | 用途与生命周期 |
|---|---|---|
| `state` 的环监听 socket | 前一个 rank | 建立后续 bootstrap 环连接 |
| `listenSockRoot` | root 服务 | 接收一次邻居信息，随后关闭 |

为什么分开？两种连接的协议、对端角色和生命周期不同。拆开后无需在一个入口中混合处理 root 回传和邻居建连。

这里描述的是前半段的两个入口，不代表整个初始化期间每个 rank 只创建两个 sockets；后半段还有其他服务入口。

### 9.3 向 root 报到

每个 rank 的 `extInfo` 携带 rank、总 rank 数、环监听地址和 root 回传地址等，结构见 [bootstrap.cc:267–275](../nccl/src/bootstrap.cc#L267-L275)。

[sendToRoot](../nccl/src/bootstrap.cc#L737-L748) 使用 ID 中的 root 地址：

```text
连接 root → 发送 extInfo → 关闭本次报到连接
```

单 ID 普通路径下所有 ranks 联系同一个 root。人数较多时会按 rank 错开报到时间，避免瞬间大量连接压到 root；这是 [833–842 行](../nccl/src/bootstrap.cc#L833-L842) 的 stagger 延迟。

### 9.4 root 只分发后继信息，不发送完整地址表

以 4 个 ranks 为例：

```text
root 给 rank 0：rank 1 的环地址
root 给 rank 1：rank 2 的环地址
root 给 rank 2：rank 3 的环地址
root 给 rank 3：rank 0 的环地址
```

root 通过各 rank 提供的回传监听地址发送这些信息，而不是复用已经关闭的报到连接。本地接收到 `nextPeer` 后关闭临时回传 socket 和 `listenSockRoot`，见 [869–876 行](../nccl/src/bootstrap.cc#L869-L876)。

为什么只给邻居？每个 rank 得到一份地址就足以形成连通环；不必由 root 给 N 个 ranks 分别复制完整 N 份地址表，后续信息可沿环分布式传播。

root 也不是等所有 ranks 报到后才开始回复。相邻信息齐备就可提前发送，最后补发剩余信息，见 [bootstrapRoot:354–388](../nccl/src/bootstrap.cc#L354-L388)。

### 9.5 主动连接后继，接收前驱连接

调用位置：[885–894 行](../nccl/src/bootstrap.cc#L885-L894)；默认分支调用 [socketRingConnect](../nccl/src/bootstrap.cc#L687-L700)。

```text
主动连接下一个 rank → send 连接
接受上一个 rank 的连接 → recv 连接

rank 0 → rank 1 → rank 2 → rank 3
   ↑                         │
   └─────────────────────────┘
```

每个 rank 只需维护相邻连接，就能通过多轮转发让信息传播到全组。图中箭头用于说明建连方向；TCP Socket 是全双工的，当前 Socket AllGather 会利用两个方向传播，详见第 10.6 节。

root 完成分发后清理资源并退出，后续环交换不需要 root 持续当中转站。

### 9.6 bootstrap 环不是 AllReduce 的 Ring

| bootstrap 环 | 集合算法 Ring |
|---|---|
| 交换控制与初始化信息 | 传输用户集合通信数据 |
| 普通路径按 rank 次序构造 | 后续结合拓扑等因素规划 |
| 帮助通信系统启动 | 通信系统实际执行算法的一种布局 |

即便全部 GPUs 在一台机器且通过 NVLink 互联，控制通道与数据通道仍是不同职责；不能从 bootstrap 走 Socket 推断后续 AllReduce 数据也必然走 TCP。

**本节小结：到第 894 行，root 已帮助 ranks 建立 bootstrap 环；完整 peer 服务地址表及数据传输通道仍未因此全部就绪。下一节利用这个环交换全组联系信息。**

<a id="bootstrap-info"></a>

## 10. bootstrapInit 后半段：服务入口、全组信息交换与 proxy 状态

源码：[bootstrap.cc:896–964](../nccl/src/bootstrap.cc#L896-L964)。本节继续普通新建、单 ID、默认 Socket bootstrap 主线，UDS 的具体机制以 Linux 实现为例。

```text
创建本 rank 的 proxy、bootstrap P2P 入口与 UDS 标识
  → 准备可选 RAS 记录
  → ringAllInfo 打包
  → bootstrapAllGather 沿已有环交换
  → 拆回全组联系信息表
  → ncclProxyInit 保存资源并创建本地 IPC socket
  → RAS 信息交接、计时与返回
```

### 10.1 已经有环，为什么还需要新的入口？

| 通道 | 主要用途 |
|---|---|
| 已建立的 bootstrap 环 | 全体 ranks 集体交换初始化信息 |
| bootstrap P2P 入口 | 指定的两个 ranks 直接交换控制信息 |
| proxy 服务入口 | 请求某个 rank 的代理服务处理资源、连接等操作 |

环适合全员参与的信息交换，但后续可能只需要 rank 0 和 rank 3 交换一份连接信息。为这种操作提供定向入口，可以避免每次都沿环中转并协调其他 ranks。

这里的 bootstrap P2P 不是公开 API `ncclSend/ncclRecv`，也不是 GPU P2P 数据通道。proxy 服务 socket 也不等同于承载用户张量的网络数据连接。

### 10.2 每个 rank 先把自己的地址填入自己的槽位

先看 [898–902 行](../nccl/src/bootstrap.cc#L898-L902)：

```cpp
ncclCalloc(&state->peerProxyAddresses, nranks);
ncclCalloc(&proxySocket, 1);
createListenSocket(comm, comm->magic, proxySocket,
                   state->peerProxyAddresses + rank,
                   ncclSocketTypeProxy);
```

代码分配长度为 `nranks` 的地址数组，但只创建本 rank 的监听 socket。`state->peerProxyAddresses + rank` 等价于 `&state->peerProxyAddresses[rank]`，是本地地址的输出位置。

例如，4 个 ranks 中的 rank 2 此时看到：

```text
peerProxyAddresses = [ 未知, 未知, rank 2 的 proxy 地址, 未知 ]
```

为什么现在就分配完整数组？下一步需要收集其余 ranks 的地址，最终每个 rank 都会保留一份完整联系表。分配数组不意味着在本进程创建其他 ranks 的监听服务。

[907–913 行](../nccl/src/bootstrap.cc#L907-L913) 对 bootstrap P2P 入口做同样的事：

```cpp
createListenSocket(comm, comm->magic,
                   &STATE_LISTEN(state, peerSocket),
                   &peerSocketAddress,
                   ncclSocketTypeBootstrap);
ncclCalloc(&state->peerP2pAddresses, nranks);
memcpy(state->peerP2pAddresses + rank, &peerSocketAddress,
       sizeof(union ncclSocketAddress));
```

后续 [socketConnect](../nccl/src/bootstrap.cc#L1067-L1077) 从 `peerP2pAddresses[peer]` 查到目标地址，再连接指定 peer。知道对端地址不代表现在就建立连接；本段没有直接创建一张全连接网络。

### 10.3 UDS：先生成标识，稍后才创建 IPC socket

[904–905 行](../nccl/src/bootstrap.cc#L904-L905)：

```cpp
ncclCalloc(&state->peerProxyAddressesUDS, nranks);
getUDS(state->peerProxyAddressesUDS + rank);
```

不要根据函数名把 `getUDS` 理解为“取得一个已创建的 Socket”。[其实现](../nccl/src/bootstrap.cc#L558-L563)只是：

```cpp
uint64_t randId;
getRandomData(&randId, sizeof(randId));
*peerUDS = getPidHash() + randId;
```

这里生成一个 64 位标识，之后结合 rank 构造 IPC endpoint 名称。它不是文件描述符、显存地址或 `commHash`，也不是 TCP 的 IP/端口。

UDS 即 Unix Domain Socket，用于同机进程间通信。为什么 proxy 还需要它？资源共享有时需要跨进程传递文件描述符，Linux 实现使用 UDS 的 `SCM_RIGHTS` 机制完成这件事，见 [linux_ipcsocket.cc:205–214](../nccl/src/os/linux_ipcsocket.cc#L205-L214)。

把“我的 fd 是 5”通过 TCP 发送给另一个进程没有这种效果：对方进程中的 fd 5 可能指向完全不同的对象。UDS 传递的是对相应内核资源的访问引用，接收端得到的 fd 数值不必相同。交换到远端机器的 UDS 标识也不意味着可以跨机器使用 UDS。

真正创建、绑定本 rank IPC socket 的动作在后面的 `ncclProxyInit` 中，调用的 Linux 实现见 [ncclIpcSocketInit](../nccl/src/os/linux_ipcsocket.cc#L28-L84)。

### 10.4 RAS：复用同一轮信息交换，并保留集体参与

[915–932 行](../nccl/src/bootstrap.cc#L915-L932)在 RAS 启用时准备本 rank 的记录，包括地址、PID、CUDA/NVML 设备编号、主机 hash 和进程 hash。

RAS 用于可靠性、可用性和可维护性相关的监控诊断。既然即将收集全组信息，就可以把 RAS 记录一起带上，复用这一轮交换。

这里处理 `ncclRasCommInit` 失败的方式值得注意：

```text
记录错误
  → 清零本 rank 的 RAS 记录，使地址明确无效
  → performRasAddRanks = false
  → 仍然参加下面的 ringAllInfo
```

为什么不能直接跳过交换？其他 ranks 仍会等本 rank 参与。某项可容忍的本地监控失败，不应让它离开集体步骤，导致其他参与者一直等待。

这不代表所有初始化错误都可忽略；它是此处可选 RAS 初始化的明确处理策略。

### 10.5 ringAllInfo：打包、AllGather、拆包

调用见 [935–937 行](../nccl/src/bootstrap.cc#L935-L937)，实现见 [701–735 行](../nccl/src/bootstrap.cc#L701-L735)。以下按成功路径看数据如何流动。

**第一步：打包自己的记录。** 临时记录类型为：

```cpp
struct bootstrapRingData {
  union ncclSocketAddress peerAddress;
  union ncclSocketAddress peerProxy;
  uint64_t peerUDS;
  struct rasRankInit rasRank;
};
```

分配 `nRanks` 个记录，只把各输入数组的 `[rank]` 项复制进 `ringData[rank]`。把 rank i 的完整记录记作 `Ai`：

```text
rank 0：[ A0,  —,  —,  — ]
rank 1：[  —, A1,  —,  — ]
rank 2：[  —,  —, A2,  — ]
rank 3：[  —,  —,  —, A3 ]
```

**第二步：内部 AllGather。**

```cpp
bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData));
```

`size` 是一个 rank 的记录大小，不是整个数组大小。`ringData` 同时承载输入和输出，本地输入必须已经位于自己的 `[rank]` 槽位。成功后每个 rank 都得到：

```text
[ A0, A1, A2, A3 ]
```

它不是调用公开的 `ncclAllGather`，也不发射 NCCL 通信 kernel；它在主机侧使用已有 bootstrap 连接交换控制数据。

**第三步：拆回各数组。** 遍历所有 rank，把临时记录拆回 `peerP2pAddresses[i]`、`peerProxyAddresses[i]`、`peerProxyAddressesUDS[i]` 和可选的 `rasRanks[i]`，最后释放临时 `ringData`。

为什么先打包？这些信息具有相同的参与者和交换时机，合并为一份记录可以减少分别组织多轮交换的开销。函数也允许部分数组参数为空，未提供的字段不进行相应打包/拆包，以复用到其他初始化分支。

### 10.6 socketRingAllGather：当前实现沿环双向传播

[bootstrapAllGather](../nccl/src/bootstrap.cc#L1413-L1438) 根据 `NCCL_OOB_NET_ENABLE` 选择 Net 或 Socket 后端。本节继续默认的 [socketRingAllGather](../nccl/src/bootstrap.cc#L1363-L1412)，不要把它的具体轮数直接套到其他后端或版本。

第 9 节的箭头表示邻居连接关系。TCP Socket 是全双工的，名为 `send` 的连接也能接收，名为 `recv` 的连接也能发送；本实现沿已有相邻连接的两个方向传播记录，并未因此再创建一套环连接。

关键索引如下，`step` 从 0 开始：

```cpp
int totalSteps = nranks / 2;

int sendSliceRing0 = (rank - step + nranks) % nranks;
int recvSliceRing0 = (rank - step - 1 + nranks) % nranks;
int sendSliceRing1 = (rank + step) % nranks;
int recvSliceRing1 = (rank + step + 1) % nranks;
```

这些 slice 索引表示“本次发送/接收哪一个 rank 的记录”，不是切换当前相邻连接的目标：

- Ring0：向后继发送，从前驱接收。
- Ring1：向前驱发送，从后继接收。
- 第 0 轮发送自己的记录，后续轮次转发上一轮从对应方向收到的记录。
- 缓冲位置是 `data + slice * size`；`size` 仍为单个 rank 的记录大小。

普通双向轮把四项收发放入 `ncclSocketOp ops[4]`，交给 `socketDoubleSendRecv` 推进。这里只说明其收发关系，不把四个操作理解为四个新线程。

以 4 个 ranks 中的 rank 0 为例，前驱为 rank 3，后继为 rank 1：

| 时刻 | rank 0 的动作 | rank 0 已知记录 |
|---|---|---|
| 开始 | 只有自己的记录 | `A0` |
| 第 1 轮，`step=0` | 向两侧发送 `A0`；从 rank 3 收 `A3`，从 rank 1 收 `A1` | `A0、A1、A3` |
| 第 2 轮，`step=1` | 向 rank 1 转发 `A3`；从 rank 3 收到其转发的 `A2` | `A0、A1、A2、A3` |

为什么最后一轮只有一个方向？偶数个 ranks 时，最远的对面 rank 只有一个；若两个方向都继续传，会收到两份相同记录。因此源码判断：

```cpp
bool isFinalUnidirectional =
    (step == totalSteps - 1) && (nranks % 2 == 0);
```

最后这个单向轮只用 Ring0，通过 `socketSendRecv` 完成。

- 奇数 N：每轮从两侧各得到一份新记录，共 `(N-1)/2` 轮。
- 偶数 N：最后一轮只收一份新记录，共 `N/2` 轮。
- N=1：不需要传播，循环不执行。

因此总轮数是 `floor(N/2)`，4 ranks 用 2 轮，而非普通单向环的 3 轮。传播轮数减少，不等于实际耗时严格减半，也不表示需要收齐的信息量减半。

### 10.7 ncclProxyInit 不等于启动 proxy 服务线程

[941–942 行](../nccl/src/bootstrap.cc#L940-L942)把准备好的资源交给：

```cpp
ncclProxyInit(comm, proxySocket,
              state->peerProxyAddresses,
              state->peerProxyAddressesUDS);
```

其[实际实现](../nccl/src/proxy.cc#L2111-L2128)主要是：

1. 创建 `comm->sharedRes->proxyState`，让 `comm->proxyState` 指向它，并设置引用计数。
2. 保存 `listenSock`、`peerAddresses`、`peerAddressesUDS`。这里保存原有指针，不是重新复制整张表。
3. 使用本 rank 的 UDS 标识调用 `ncclIpcSocketInit`，创建本地 IPC socket。

**这一阶段没有启动 proxy 服务线程。** 真正创建线程的是后面的 [ncclProxyCreate](../nccl/src/proxy.cc#L2130-L2165)，普通路径从 [initTransportsRank](../nccl/src/init.cc#L1775-L1786) 调用，启动 `ncclProxyService` 和 `ncclProxyServiceUDS`。

为什么分两步？服务线程工作需要本地 rank 数量、channel 数量、缓冲参数和后端上下文等信息，其中一部分在后面的初始化阶段才确定。

```text
bootstrapInit：入口、地址表、proxyState、IPC socket
  → initTransportsRank：补齐拓扑与运行参数
  → ncclProxyCreate：启动服务线程
```

因此不能只根据附近的服务创建注释，把 `ncclProxyInit` 当成“服务线程已经开始运行”。资源生命周期也有分工：[bootstrapClose](../nccl/src/bootstrap.cc#L1587-L1592) 关闭 bootstrap P2P 入口并释放其地址表，proxy 地址表由 [proxy 清理路径](../nccl/src/proxy.cc#L2215-L2225) 释放。

### 10.8 返回前的 RAS 交接与阶段边界

[944–964 行](../nccl/src/bootstrap.cc#L944-L964)完成 RAS 信息交接、计时、日志和返回：

- RAS 已启用且本地初始化允许继续时，尝试 `ncclRasAddRanks(rasRanks, nranks)`。
- 成功后将局部指针 `rasRanks` 置空，由 RAS 接管；失败则记录日志，局部数组仍在退出路径释放。
- 输出 bootstrap 分阶段耗时，返回初始化结果。这里不深入展开所有异常清理路径。

成功完成这段主线后：

```text
已有：bootstrap 环、全组服务联系表、proxyState、本地 IPC 入口
未完：完整 peer 能力与成员关系分析、GPU/NIC 拓扑、算法图、传输资源、设备状态
```

服务地址表回答“到哪里联系某个 rank 的服务”，不是完整的硬件拓扑；RAS 记录携带部分设备信息，也不能替代后续 `comm->peerInfo` 和拓扑分析。

**本节小结：已从“拥有相邻控制连接”推进到“拥有全组服务联系信息”。bootstrap 主线讲解到函数返回；proxy 服务线程尚待后续启动，下一步回到 `initTransportsRank`。**

<a id="peer-info"></a>

## 11. initTransportsRank 首段：收集身份与能力，发布 peerInfo

调用处：[init.cc:2196](../nccl/src/init.cc#L2193-L2196)。本节深入 [1234–1239 行](../nccl/src/init.cc#L1234-L1239) 的 `fillInfo → bootstrapAllGather → peerInfoValid`，尚未进入兼容性检查与拓扑搜索。

### 11.1 为什么服务地址交换后，还要再次 AllGather？

`bootstrapInit` 的服务联系表回答“怎样联系其他 ranks”；`peerInfo` 回答“其他 ranks 在哪里、用什么 GPU、具备什么能力”。地址可达，不代表可以使用共享内存、GPU 直接访问或某种网络路径。

```text
收齐身份与能力记录
  → 检查兼容性、识别成员关系
  → 发现 GPU/NIC 拓扑与路径
  → 规划通信图和 channels
  → 启动服务、准备传输资源与设备状态
```

这次复用已有 bootstrap 通道，只是交换的数据类型变了。不能把初始化理解成“调用一次 AllGather 后就拥有所有信息”。

### 11.2 四步主线与 nranks + 1 的含义

省略错误宏后：

```cpp
ncclCalloc(&comm->peerInfo, nranks + 1);
fillInfo(comm, comm->peerInfo + rank, comm->commHash);
bootstrapAllGather(comm->bootstrap, comm->peerInfo,
                   sizeof(struct ncclPeerInfo));
COMPILER_ATOMIC_STORE(&comm->peerInfoValid, true,
                      std::memory_order_release);
```

四步分别是：分配全组记录数组、填本 rank 的记录、收齐其他记录、向本进程内的读取者发布表已就绪。

多出的一个槽位用于表示 CollNet root：

```text
peerInfo[0 ... nranks-1]：真实 ranks
peerInfo[nranks]：CollNet root 的内部预留槽位
```

它不增加用户通信组的参与者数量。`bootstrapAllGather` 仍按照 bootstrap state 中的真实 rank 数交换。

### 11.3 fillInfo：设备身份不能只用 cudaDev

实现：[init.cc:888–982](../nccl/src/init.cc#L888-L982)；记录类型：[transport.h:46–74](../nccl/src/include/transport.h#L46-L74)。

`comm->peerInfo + rank` 等价于 `&comm->peerInfo[rank]`，所以 `fillInfo` 只填自己的一个槽位，不负责获取所有人的信息。

| 字段 | 身份视角 |
|---|---|
| `rank` | 通信组成员编号 |
| `cudaDev` | 该进程可见的 CUDA 设备下标 |
| `nvmlDev` | NVML 管理视角的设备编号 |
| `busId` | PCI 硬件位置 |
| `gpuUuid` | GPU/分区身份 |

教学例子：进程 A 和进程 B 都只看见一张 GPU，各自的 `cudaDev` 都可以为 0，却分别对应物理 GPU A 和 GPU B。因此后续重复设备检查不能单独比较 `cudaDev`，需要结合主机身份与 GPU UUID。

### 11.4 hostHash、pidHash 为什么再加 commHash？

```cpp
info->hostHash = getHostHash() + commHash;
info->pidHash = getPidHash() + commHash;
```

基础标识来源见 [utils.cc:85–187](../nccl/src/misc/utils.cc#L85-L187)：

- Linux 的 `getHostHash()` 默认对 hostname 与 boot ID 的组合求 hash，支持 `NCCL_HOSTID` 覆盖；读取 boot ID 失败时保留 hostname 回退。
- Linux 的 `getPidHash()` 组合 PID 与 PID namespace 信息。不同容器都可能有 PID 1，不能只凭这个整数认定同进程。
- 判断同进程时还要同时比较主机身份，避免把不同主机上的进程混为一谈。

设主机基础标识为 H、通信组标识为 G：

```text
同组同机：H + G 与 H + G 仍然相等
同机另一组：保存值变成 H + G2
```

这使 `peerInfo` 内保存的身份带上逻辑通信组的作用域，不需要额外通信。它不是重新探测主机，也不是认证凭据；不同组的 hash 可能碰撞，容器环境和配置也会影响基础标识，不能视为绝对的物理身份保证。

### 11.5 shmDev：同机不代表共享同一个 /dev/shm

Linux 分支：

```cpp
struct stat statbuf;
stat("/dev/shm", &statbuf);
info->shmDev = statbuf.st_dev;
```

`shmDev` 是 `/dev/shm` 所在文件系统的设备标识，不是 CUDA 设备编号，也不是共享内存指针。

为什么收集它？同机的两个容器可能具有不同的 `/dev/shm` 挂载。后续 [shmCanConnect](../nccl/src/transport/shm.cc#L61-L82)先比较 `hostHash`，再比较 `shmDev`；不匹配时提前返回，可连接输出仍为 0。

这里先记录共享内存环境的线索，后面 transport 再判断是否适用。通过这些检查仍不等于共享内存资源已经创建成功。

### 11.6 版本与能力是后续选择的输入

| 字段示例 | 后续用途 |
|---|---|
| `version`、`gitVersionHash` | NCCL 版本与源码构建指纹检查 |
| `cudaDriverVersion`、`cudaCompCap` | 驱动条件和 GPU 计算能力 |
| `cuMemSupport` | 本地 cuMem 路径的启用状态 |
| `gdrSupport`、`cuMemGdrSupport` | GPUDirect RDMA 及相关内存路径能力 |
| `fabricInfo`、`fabricHandleSupport` | MNNVL/Fabric 判断 |
| `gpuCftSupport` 及相关布尔字段 | CFT 支持等级与细分能力 |
| `supportedGinTypeBitMask` | 本地 GIN 后端类型集合 |
| `crossNicSupport`、`rmaPluginAvailable` | 后端能力和可用性 |

本地支持不代表全组可以统一使用，后续还要求共同能力并检查配置。`gdrSupport` 为真，也不保证某条实际 GPU↔NIC 路径最终采用 GDR，仍要结合拓扑、后端和配置。

`fillInfo` 并非全是字段复制：它会调用 CUDA、NVML 和能力探测函数。主要产物是身份与能力记录，不是完整的 peer 数据连接；MNNVL、GIN、CFT 的全部内部实现暂不展开。

### 11.7 为什么连 comm 指针也放进交换记录？

```cpp
info->comm = comm;
```

多个 ranks 可以位于同一个进程。教学例子：

```text
进程 A：rank 0 → comm0，rank 1 → comm1
进程 B：rank 2 → comm2
```

rank 0 和 rank 1 共享地址空间，可以使用彼此的 comm 指针。rank 2 收到这些指针值，不意味着获得远程对象访问能力，不能解引用它们。

后面的 [1330–1338 行](../nccl/src/init.cc#L1330-L1338)正是先确认 `hostHash` 和 `pidHash` 都匹配，再使用 `peerInfo[i].comm` 串联同进程对象。统一交换结构不代表其中每个字段对所有接收者都能直接使用。

### 11.8 AllGather 与 peerInfoValid 的阶段边界

将 rank i 的记录记作 Pi：

```text
交换前：
rank 0：[ P0,  —,  —,  — ]
rank 1：[  —, P1,  —,  — ]
rank 2：[  —,  —, P2,  — ]
rank 3：[  —,  —,  —, P3 ]

交换后，每个 rank：[ P0, P1, P2, P3 ]
```

本地输入在 `[rank]`，`size` 是单条记录字节数；这是内部主机侧通信，不重新建环，也不是公开 `ncclAllGather`。每个 rank 都需要完整副本，以执行后面的本地决策。

随后以 release 写入 `peerInfoValid=true`。例如 [RAS 诊断](../nccl/src/ras/diagnostics.cc#L88-L94)会用 acquire 读取这个标志，未就绪时不使用表。配对同步保证读者看到发布前完成的记录写入。

它只表示“本进程里的这张表已可读”，不是跨机器内存屏障，也不是整个 communicator 初始化成功。版本与能力检查就在它后面，因此表可读甚至不等于兼容性检查已通过。

**本节小结：到 `init.cc:1239`，各 rank 从“知道服务地址”推进到“拥有全组身份与能力记录”。验证这些记录、汇总能力、构造成员关系仍在下一节。**

<a id="peer-checks"></a>

## 12. 消费 peerInfo：兼容性、共同能力与同进程成员

源码：[init.cc:1241–1379](../nccl/src/init.cc#L1241-L1379)。本节从“收齐记录”推进到“据此作决策”，仍未开始硬件拓扑发现。

### 12.1 contiguousRanksPerHost 统计的是连续区间

[1247–1270 行](../nccl/src/init.cc#L1247-L1270)按 rank 顺序扫描 `hostHash`，用 `prevHostHash` 记录当前主机区间，用 `currentHostSize` 记录连续区间长度。

```text
rank： 0  1  2  3  4  5
host： A  A  B  B  C  C
       └─┘  └─┘  └─┘
长度：  2    2    2

contiguousRanksPerHost = 2
```

第一次区间结束时建立长度基准，后面区间与它比较；长度不一致时以 `INT_MAX` 表示不规则排列。

为什么需要这个值？规则排列下可以按步长选取对应成员，例如 `0、2、4` 或 `1、3、5`。后续 [GIN rail 连接模式](../nccl/src/gin/gin_host.cc#L138-L144)会使用这个步长。

它不是任意主机的总 rank 数统计：`A B A B` 会形成四个长度为 1 的连续区间，而不是直接统计成两个各有 2 个 ranks 的主机集合。

**源码边界：**[1258 行](../nccl/src/init.cc#L1253-L1259)在发现中间区间长度不一致时执行 `break`，退出的是整个外层 `for`；版本检查、重复 GPU 检查和能力归并也在这个循环中，因此该分支会跳过当前及后续成员的这些处理。下面按循环能够完整扫描的正常路径解释，不将其描述为所有布局下一定完成全组检查。这里记录的是静态源码行为，尚未进行该分支的运行验证。

### 12.2 两种版本检查，处理强度不同

| 检查 | 当前处理 |
|---|---|
| `peerInfo[i].version` 与本 rank 的版本不同 | 设为 `ncclInvalidUsage`，转到失败路径 |
| 版本号相同但 `gitVersionHash` 不同 | rank 0 输出提醒，本项检查不直接终止初始化 |

版本检查见 [1273–1277 行](../nccl/src/init.cc#L1273-L1277)，源码指纹检查见 [1304–1311 行](../nccl/src/init.cc#L1304-L1311)。

为什么检查？后续 ranks 需要按共同的内部数据格式和协议协作，不能简单假定不同版本可以混用。相同发布版本也可能来自不同源码构建，额外指纹有助于暴露差异；由 rank 0 提醒可以减少重复日志。

### 12.3 重复设备：NVML 编号与 GPU UUID 的区别

[1287–1296 行](../nccl/src/init.cc#L1287-L1296)对每个 i 比较此前所有 `j < i`，不是只比较“别人是否与我重复”。

简化条件：

```text
未启用 MULTI_RANK_GPU_ENABLE
  且同主机
  且 GPU UUID 相同
    → ncclInvalidUsage
```

`NCCL_MULTI_RANK_GPU_ENABLE` 当前默认是 0，见 [init.cc:73](../nccl/src/init.cc#L73)。这里表达的是默认禁止、可显式启用的配置约束，不是任何条件下都不允许多 rank 共用 GPU。

还有两个不能混为一谈的判断：

| 条件 | 含义 |
|---|---|
| 同主机、相同 `nvmlDev` | 多个 ranks 对应同一 NVML 设备，记录 `hasMultiRankNvml` |
| 同主机、相同 `gpuUuid` | 多个 ranks 使用同一 GPU/分区身份，参与重复设备检查 |

MLOPart 的不同分区可以属于同一 NVML 设备，但具有不同 UUID。因此 `hasMultiRankNvml=true` 本身不等于重复 GPU 检查失败。是否出现 MLOPart 则另用 `hasMloPart` 记录。

### 12.4 共同能力：AND、交集与最小值

[1280–1302 行](../nccl/src/init.cc#L1280-L1302)把各 rank 的记录归并成后续选择条件：

| 信息 | 归并方式 | 设计目的 |
|---|---|---|
| `cuMemSupport`、`cuMemGdrSupport` | 初始为真，任何不支持者使其为假 | 判断全员支持 |
| `globalCrossNicSupport`、`globalRmaPluginSupport` | 布尔 AND | 收缩为共同能力 |
| `globalGinTypeBitMask` | 按位 AND | 保留所有 ranks 都支持的后端类型 |
| `gpuCftSupport` | 最小支持等级 | 取共同可依赖的等级 |
| `gpuCftMulticastSupport`、`gpuCftCountedSupport` | 分别做 AND | 检查 CFT 细分能力是否全员具备 |
| `minDriverVersion` | 最小版本 | 保留组内驱动的保守边界 |

例如，三个位表示三种 GIN 类型：

```text
rank 0：011
rank 1：110
交集：  010
```

`globalGinTypeBitMask` 初始为 `UINT64_MAX`，因为全 1 与第一个集合相与，结果仍是第一个集合；从 0 开始则永远得不到任何类型。

为什么不是并集？某些路径要求参与者共同支持，不能因为自己支持就让其他成员走同一机制。这也不是把所有 GPU 的性能强行降到相同水平，而是为选择提供兼容性边界。

这些结果是后续决策的输入，不是路径已经启用。例如 [1923–1932 行](../nccl/src/init.cc#L1923-L1932)还会结合 cuMem GDR、MLOPart、跨 NIC 支持和 rank 排列决定 GIN/RMA 相关能力。

### 12.5 MNNVL 检查与局部 nNodes 的命名陷阱

[1316–1320 行](../nccl/src/init.cc#L1316-L1320)在发现异机成员、配置允许且相关 P2P 条件满足时尝试 `ncclMnnvlCheck`；显式启用 MNNVL 的设置也可触发检查。

前面局部 `nNodes` 初始为 1，但更新方式是：

```cpp
if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash)
  nNodes++;
```

这里没有按主机去重。两台机器各 4 ranks，完整扫描后本机视角会得到 5，而非 2。此处主要用 `nNodes > 1` 判断是否出现异机成员，不能当作最终精确节点数 `comm->nNodes`。

[ncclMnnvlCheck](../nccl/src/mnnvl.cc#L14-L94)检查 cuMem、Fabric handle 支持、Fabric 初始化状态，确定 NVLink domain/clique 成员，并试验 Fabric handle 的导出与导入。上一节采集的 `fabricInfo` 在这里被消费。

为什么此时检查？有些跨节点系统存在 NVLink Fabric，不能只凭主机不同就认定只能走传统网络。

返回 `ncclSuccess` 不一定表示 MNNVL 已启用：条件不满足时也可正常返回，而不设置 `comm->MNNVL`。相反，确认存在相关能力但 Fabric 资源实际不可用时，还可能报错。检查过程成功与能力启用是不同概念。

### 12.6 同进程成员的三个编号

[1322–1379 行](../nccl/src/init.cc#L1322-L1379)用 `do { ... } while (0)` 包住一组局部逻辑；它只执行一次，不是重试循环。

按全局 rank 递增扫描，确认 `hostHash` 与 `pidHash` 都匹配后，统计：

```cpp
if (intraProcRanks == 0) intraProcRank0 = i;
if (i == rank) intraProcRank = intraProcRanks;
intraProcRanks++;
```

教学布局：

```text
主机 A：进程 P 有 rank 0、rank 2；进程 Q 有 rank 1
主机 B：进程 R 有 rank 3
```

rank 2 看到的同进程成员是 `0、2`：

| 变量 | 值 | 含义 |
|---|---:|---|
| `intraProcRank0` | 0 | 本进程成员中首个成员的全局 rank |
| `intraProcRank` | 1 | 自己在本进程成员中的局部编号 |
| `intraProcRanks` | 2 | 这个通信组在本进程的成员数 |

通过检查后写入 `comm->intraComm0`、`comm->intraRank`、`comm->intraRanks`。进程 R 只有 rank 3 时，其 `intraComm0` 指向 rank 3 自己，不指向全局 rank 0。

`intraRanks` 不等于同机 rank 数，也不是所有逻辑通信组在本进程中的 comm 总数。同一轮循环还求全组 `minCompCap/maxCompCap`，为异构 GPU 决策保留能力范围。

### 12.7 intraNext 链表与 barrier 状态分别做什么？

只有本进程首 rank 负责串联其他本进程 comm：

```cpp
if (intraProcRank0 == rank && rank != i) {
  comm->peerInfo[i].comm->intraNext = comm->intraNext;
  comm->intraNext = comm->peerInfo[i].comm;
}
```

单个构造者避免多个初始化任务同时修改同一条链。上述例子形成 `comm0 → comm2 → nullptr`。这是进程内对象链表，不是 bootstrap 环或算法 Ring；后续 [finalize/清理逻辑](../nccl/src/init.cc#L3260-L3280)会沿它遍历。

随后清零 `intraBarrierPhase`、`intraBarrierCounter`、`intraBarrierGate`。后续 [ncclCommIntraBarrierIn/Out](../nccl/src/include/comm.h#L943-L976)通过共享的 `intraComm0` 计数与放行，供特定 Group launch 等主机侧协调使用。

这里仅准备状态，没有执行一次 barrier，更不代表 GPU 工作完成。

### 12.8 NVLS 注册路径还依赖进程组织方式

[1342–1355 行](../nccl/src/init.cc#L1342-L1355)更新 `nvlsRegSupport`：

1. 初始为 1。
2. 通信组中存在两个 ranks 属于同一进程时，先设为 0。
3. MNNVL 已启用时，设为 0。
4. 否则，显式开启 `NCCL_SINGLE_PROC_MEM_REG_ENABLE` 可设回 1；该参数默认关闭，见 [group.cc:680](../nccl/src/group.cc#L680)。

`nvlsRegSupport` 是 NVLS 相关注册路径的支持条件，不是 `nvlsSupport` 本身。设为 0 不代表整个 NCCL 无法通信，也不能直接推断 NVLS 算法完全不可用。

**本节小结：正常推进到 `init.cc:1379` 后，已完成本段兼容性与能力处理，并建立同进程编号、对象链表和协调状态。硬件之间怎样连接，仍要下一节建图和算路。**

<a id="topology"></a>

## 13. 硬件拓扑与路径：建图、裁剪、重算和 CPU affinity

主线：[init.cc:1381–1412](../nccl/src/init.cc#L1381-L1412)。此前已知道成员身份和能力，现在要知道 GPU、CPU、NIC 怎样连接、哪些路径可用，为后续 Ring/Tree 规划提供输入。

### 13.1 硬件图与算法通信图不是同一个对象

| 对象 | 回答的问题 |
|---|---|
| `ncclTopoSystem`，即 `comm->topo` | 硬件资源怎样连接，端点之间有哪些可用路径？ |
| 后续的 `ncclTopoGraph` | 为某种算法，怎样安排 GPU 顺序、通道和网络出口？ |

先认识路网，再规划通信顺序。当前这段还没有确定算法 Ring/Tree：

```cpp
ncclTopoGetSystem(comm, &comm->topo);
ncclTopoComputePaths(comm->topo, comm);
ncclTopoTrimSystem(comm->topo, comm);
ncclTopoComputePaths(comm->topo, comm);
ncclTopoSearchInit(comm->topo);
ncclTopoComputeCommCPU(comm);
ncclTopoPrint(comm->topo);
```

三层职责是“建图 → 算路、裁剪、重算 → 准备搜索参数”。

### 13.2 ncclTopoGetSystem：融合多个来源的硬件描述

实现：[topo.cc:1989–2178](../nccl/src/graph/topo.cc#L1989-L2178)。

```text
读取可选 XML 描述
  → 探测本 rank 的 GPU 及相关硬件
  → 导入网络插件的设备信息
  → 交换、融合各 rank 的局部描述
  → 转成 ncclTopoSystem
```

开头尝试读取 `NCCL_TOPO_FILE`，未指定时尝试默认的 topology 服务文件；没有现成描述时创建 XML 根节点，继续自动发现。

XML 是统一的中间表示，用于融合外部描述、OS 探测、NVML 和网络插件属性。它大部分时间是内存结构，并非每一步都读写磁盘，也不要求用户必须提供 XML 文件。

`NCCL_TOPO_DUMP_FILE` 则用于导出发现、合并后的 XML。`init.cc` 中的 dump 调用与实际设置 `comm->topo` 的调用分开；不能把 dump 文件当作随后裁剪完成、包含所有路径缓存的完整对象快照。

[2026–2035 行](../nccl/src/graph/topo.cc#L2026-L2035)通过本 rank 的 `busId` 调用 `ncclTopoFillGpu`，给节点填入 `keep`、`rank`、`gdr` 等属性。[ncclTopoFillGpu](../nccl/src/graph/xml.cc#L1077-L1085)结合 OS 层 PCI 信息和 NVML 查询补充 GPU 描述。

为什么各 rank 主要探测自己负责的 GPU？各进程的设备可见性可能不同，局部探测后再合并，才能拼接分散的视图。网络设备则由 Net、CollNet、GIN、RMA 等相应插件提供列表和属性，使图反映后端实际可用的资源。

### 13.3 XML 融合：再次交换，但参与范围不同

[2118–2156 行](../nccl/src/graph/topo.cc#L2118-L2156)调用 `bootstrapIntraNodeAllGather`。普通路径用 `hostHash` 选出同机 ranks；MNNVL 路径按 clique 组织融合，范围不必局限于一台机器。

```text
rank 0 提供 GPU 0 的局部描述
rank 1 提供 GPU 1 的局部描述
rank 2 提供 GPU 2 的局部描述
            ↓
参与融合的 ranks 各自得到合并后的描述
```

这不是由某个 rank 探测整个集群所有交换机布线后广播。普通路径主要形成同机相关硬件视图，跨节点算法关系还要后续协调。

XML 内部也有父子指针。[ncclTopoConvertXml](../nccl/src/graph/xml.cc#L256-L272)在发送前把它们转换成相对偏移，接收后再恢复成本地指针；不能直接把其他进程的 XML 指针拿来遍历。

融合后由 [ncclTopoGetSystemFromXml](../nccl/src/graph/topo.cc#L1057-L1088)创建节点、补充 NVLink/C2C/PCI 连接，处理部分硬件特例并整理成内部图。

### 13.4 节点、边、路径三层表示

定义见 [topo.h:46–183](../nccl/src/graph/topo.h#L46-L183)：

| 层次 | 关键内容 |
|---|---|
| `ncclTopoNode` | 设备或资源类型、身份、直接连接与路径缓存 |
| `ncclTopoLink` | 直接连接的类型、带宽模型值、对端节点 |
| `ncclTopoLinkList` | 到某个目标的连接序列、连接数量、路径带宽与类别 |

常见节点包括 `GPU/DEV`、`PCI`、`NVS`、`CPU`、`NIC/NET` 等。`GPU/DEV` 区分 rank/分区与底层设备表示；`CPU` 在这里实际按 NUMA domain 建模，不是一个 CPU core。

省略部分内部节点层次的教学图：

```text
GPU 0 ── PCIe switch ── CPU/NUMA 0
             │              │
            NIC 0        系统互联
                            │
GPU 1 ── PCIe switch ── CPU/NUMA 1
             │
            NIC 1
```

它表达了 GPU 0 到 NIC 0 与 NIC 1 的路径差异，也表明同机 GPU 之间可能要跨 NUMA。只有“同机”标签不足以指导性能选择。

节点和路径是规划数据，不是已创建的 CUDA、Socket 或 RDMA 连接。

### 13.5 ncclTopoComputePaths：先算硬件路径，再结合策略修正

实现：[paths.cc:754–898](../nccl/src/graph/paths.cc#L754-L898)。先清除旧路径，围绕 CPU、GPU、DEV、网络端点、NVSwitch 等节点计算路径，再应用能力与配置约束。

底层 [ncclTopoSetPaths](../nccl/src/graph/paths.cc#L52-L147)采用广度优先式扩展，但并非仅按跳数寻找普通无权最短路。当前更新优先级为：

1. 路径类别更优。
2. 类别相同，带宽模型值更大。
3. 类别与带宽相同，跳数更少。

扩展一条连接时：

```cpp
float bw = std::min(path->bw, link->bw);
```

串联路径受瓶颈链路约束。假设两段的模型值为 50 和 32，则路径值为 32。这些是由属性和规则得到的建模值，不是本阶段运行带宽测试得到的实测吞吐。

常见路径类型见 [graph.h:120–158](../nccl/src/include/graph.h#L120-L158)：

| 类型 | 含义 |
|---|---|
| `PATH_LOC` | 本地自身 |
| `PATH_NVL` | 经过 NVLink |
| `PATH_NVB` | 经中间 GPU 的 NVLink 路径，不是泛指经过 NVSwitch |
| `PATH_PIX` | 至多经过一个 PCIe bridge |
| `PATH_PXB` | 经过多个 PCIe bridges，但不经过 Host Bridge |
| `PATH_PHB` | 经过 PCIe Host Bridge |
| `PATH_SYS` | 经过 NUMA 节点间系统互联 |
| `PATH_NET` | 网络路径 |
| `PATH_DIS` | 不连通 |

这些类别帮助后续代码比较路径，不是所有机器上的绝对性能排名；本节也不穷举所有特殊类型。

### 13.6 物理连通不等于直接路径可用

[795–826 行](../nccl/src/graph/paths.cc#L795-L826)检查 P2P 和 SHM：

- 不允许或不能使用 GPU Direct P2P 时，将相关路径模型调整为经 CPU 中转。
- 再调用 P2P/SHM transport 的 `canConnect`，结合实际能力与策略判断可用性。
- 两者都不可用时，将相应 GPU 路径标为 `PATH_NET`，供后面裁剪本地可达域。

“经 CPU”在这里表达主机侧中转，不表示 CPU 必须执行用户归约。`PATH_NET` 也不是把 rank 从 communicator 删除，而是不再将它当作可用本地 P2P/SHM 到达的同域 peer。

[845–889 行](../nccl/src/graph/paths.cc#L845-L889)还会根据 GDR 条件调整 GPU↔NIC 路径；在满足特定条件时考虑 PXN：

```text
本 GPU → NVLink 相邻 GPU → 更合适的 NIC
```

因此最终路径模型不是纯硬件连线，而是“硬件连接 + 能力条件 + 配置策略”。完整 P2P/GDR/PXN 条件仍留待专项深入。

### 13.7 ncclTopoTrimSystem：裁剪的是局部图，不是通信组

实现：[paths.cc:900–945](../nccl/src/graph/paths.cc#L900-L945)。根据 GPU 之间的路径类型划分 domain，保留当前 rank 所属 domain 的 GPU 节点；判断包含 `path.type < PATH_NET`。

目的在于让本地规划只保留同域可达 GPU，需要经网络处理的其他成员不继续充当这个局部图中的同域 GPU。

它不会卸载硬件、减少 `comm->nRanks`、删除对应 `peerInfo` 或禁止后续网络通信。

附近注释提到“移除不可达 GPU 和不用的 NIC”，但当前函数主体直接删除的是域外 GPU 节点，并更新：

```cpp
system->inter = system->nodes[GPU].count == comm->nRanks ? 0 : 1;
```

不能根据注释补出该实现中并不存在的 NIC 删除流程，也不能把 `system->inter` 简单等同于物理机器数是否大于 1。

为什么随后必须再算一次路径？路径保存了连接指针并依赖节点索引；删节点会改变图和存储位置，旧路径不能继续使用。裁剪先清除旧路径，再删除节点，调用者随后重算。

```text
第一次 ComputePaths：为保留哪些节点提供依据
TrimSystem：修改图并清除旧路径
第二次 ComputePaths：为后续搜索提供有效路径
```

### 13.8 SearchInit 与 CPU 类型：准备参数，不是执行算法搜索

[ncclTopoSearchInit](../nccl/src/graph/search.cc#L18-L53)计算 `system->maxBw` 和 `system->totalBw` 等带宽模型参数。`totalBw` 不是全机所有链路带宽之和；当前实现先计算各 DEV 的相关带宽值，再取最大值。

真正的算法图搜索是后面的 `ncclTopoCompute`，到这里还没执行。

[ncclTopoComputeCommCPU](../nccl/src/graph/search.cc#L56-L63)从当前拓扑 CPU 节点取架构与厂商，写入 `comm->cpuArch/cpuVendor`。它不计算 CPU 运算能力，也不设置线程绑定。随后 `ncclTopoPrint` 打印模型，结束本段拓扑计时。

### 13.9 CPU affinity：让初始化线程尽量靠近 GPU

[init.cc:1406–1412](../nccl/src/init.cc#L1406-L1412)：

```cpp
ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity);
if (ncclOsCpuCount(comm->cpuAffinity)) {
  ncclOsGetAffinity(&affinitySave);
  ncclOsSetAffinity(comm->cpuAffinity);
}
```

默认选择 [topo.cc:2447–2472](../nccl/src/graph/topo.cc#L2447-L2472)中的交集：

```text
线程当前允许的 CPU 集合 ∩ 靠近 GPU 的 CPU 集合
```

交集为空则不应用新绑定；相关配置也允许忽略原 affinity 限制，采用 GPU 邻近集合。

为什么通信初始化还关心 CPU？NUMA 系统中的主机内存也有远近关系。让初始化线程靠近 GPU，有利于后续主机资源分配与访问的局部性。

Linux 的 [sched_setaffinity(0, ...)](../nccl/src/os/linux.cc#L606-L622)作用于当前初始化线程，不是重绑整个应用所有线程。[函数退出时](../nccl/src/init.cc#L1977-L1978)恢复保存的 affinity。此操作不强制迁移已有内存，也不能保证所有分配绝对落在指定 NUMA 节点。

**本节小结：到 `init.cc:1412`，已得到结合能力和策略修正、裁剪并重算后的硬件拓扑与路径，以及搜索参数和 CPU affinity。算法 Ring/Tree、channels、实际数据连接仍未完成；下一次从 `init.cc:1414` 继续。**

<a id="remaining"></a>

## 14. 特殊分支与后续流程概览

本节保留已经介绍过的整体轮廓，不代表这些内部实现已经逐行学完。

### 14.1 Split、Shrink、Grow 的入口差异

源码：[init.cc:2136–2191](../nccl/src/init.cc#L2136-L2191)。

- **Split/Shrink：**`job->parent && !job->isGrow`。先计算新成员和新 rank，派生子组 hash，然后 `commAlloc(comm, parent, ...)`，再通过 `bootstrapSplit` 借助父组完成协调。
- **Split 不参与新组：**`NCCL_SPLIT_NOCOLOR` 的 rank 仍参加所需的信息交换，随后提前退出，不走普通成功创建子组的完整路径。
- **Grow：**与普通初始化同属 `else` 大分支，但 hash、旧成员信息和 bootstrap 会合方式有特殊处理，不能直接套普通单 ID 图。
- **普通 INIT：**使用第一个 ID 内容的 hash，`commAlloc` 不复用父组资源，然后 `bootstrapInit`。

把共同的资源初始化与不同的成员协调分开，可以复用主要逻辑，又不强行把所有派生场景写成同一种握手。

### 14.2 initTransportsRank 剩余部分：从硬件图到可执行通信

第 11–13 节已覆盖信息交换、成员关系与硬件拓扑。下一步从 [init.cc:1414](../nccl/src/init.cc#L1414-L1437)继续 CollNet/NVLS 准备和 `ncclTopoCompute` 算法图搜索，再分析图信息协调、channels、proxy 服务启动、transport 资源及 GPU 使用的设备状态。

不能简单概括为“初始化时一次性建好所有算法的全部连接”。源码会根据 cuMem 支持和运行时连接配置选择分支，见 [init.cc:1811](../nccl/src/init.cc#L1811)；某些连接可延迟到首次使用相关算法时建立。

后续精读按“算法图 → channel/transport → 设备状态”逐段展开。宏观框架可对照 [03｜Communicator](03-communicator.md)。

<a id="state"></a>

## 15. 状态、错误与任务生命周期

### 15.1 完成与失败通过 initState 发布

回调执行完传输初始化后尝试启动进度计数监控；该监控失败会记日志，但不阻止初始化主线继续。随后发布成功，见 [init.cc:2198–2208](../nccl/src/init.cc#L2198-L2208)：

```cpp
COMPILER_ATOMIC_STORE(&comm->initState, ncclSuccess,
                      std::memory_order_release);
```

失败路径则将错误码写入 `initState`，见 [2241–2250 行](../nccl/src/init.cc#L2241-L2250)。

为什么用 release？任务可能由另一个线程执行。正确配对的 acquire 读取观察到发布状态后，应同时观察到此前完成的初始化写入；不是只让一个状态整数发生变化。

普通路径早已在 `ncclCommInitRankDev` 写入 `*newcomm`。回调末尾的 `if (job->newcomm)` 是可选交接，不能理解为所有普通初始化句柄都只在这个位置返回。

### 15.2 阶段就绪、初始化结果与 GPU 完成不要混淆

| 状态或事件 | 含义 |
|---|---|
| `comm->peerInfoValid` | 身份与能力表已可读，不代表后续检查或初始化完成 |
| `comm->initState` | communicator 初始化自身的结果 |
| `job->state` | 调度任务处于 Running、Done、Joined 等状态 |
| `comm->asyncResult` | 对外查询的异步主机操作状态/错误 |
| CUDA stream/event 完成 | 相应 GPU 工作是否已完成 |

非阻塞 Group 会设置 `asyncResult=ncclInProgress`，结束时更新结果。[ncclCommGetAsyncError](../nccl/src/init.cc#L3934-L3967) 读取该字段，并在成功且存在关联 Group job 时完成线程回收；它不只是读取 `initState`。

因此：

- `*newcomm` 非空不代表初始化完成。
- 阻塞 API 或最外层 GroupEnd 等待的是对应主机初始化工作。
- 非阻塞使用者要同时检查查询函数返回值与输出的异步状态。
- 后续 collective 返回、GroupEnd 返回或异步错误查询成功，都不能单独代替 GPU stream/event 完成判断。

### 15.3 错误宏表达不同的失败策略

见 [checks.h:24–43](../nccl/src/include/checks.h#L24-L43) 和 [158–166 行](../nccl/src/include/checks.h#L158-L166)。

| 宏 | 行为 |
|---|---|
| `CUDACHECKGOTO` | CUDA 失败时记录、清理 CUDA 错误状态、设置 NCCL 错误码并跳转 |
| `CUDACHECKIGNORE` | 记录并清理 CUDA 错误状态，但继续执行 |
| `NCCLCHECKGOTO` | 保存 NCCL 返回值；除 Success/InProgress 外，错误转到指定标签 |

失败不意味着在该标签里直接销毁所有对象；任务、comm 子资源与 Group 有各自的清理责任。不能把 `goto fail` 简化为“立即 free 整个 communicator”。

### 15.4 为什么保留分阶段计时？

`timers[]` 用 `clockNano()` 记录 kernel、alloc、bootstrap、拓扑、图、连接等阶段耗时，最后输出 `Init timings`。这样初始化慢时可以先判断时间花在哪一层，而不是只得到一个总时长。

回调的总计时从回调开始算起，不包含此前等待 GroupEnd 的全部时间，也不能直接等同于从 `ncclGetUniqueId` 开始的应用端总耗时。

**本节小结：初始化结果、调度线程结束和 GPU 工作结束是不同边界；任务清理也不等于销毁成功创建的 comm。**

<a id="recap"></a>

## 16. 阶段小结、易混点与下一步阅读

### 16.1 一张表复习已深入学习的过程

| 阶段 | 已得到什么 | 仍未得到什么 |
|---|---|---|
| `ncclGetUniqueId` | root 联络信息、默认路径下的后台 root 服务 | 所有 peers 的完整信息 |
| `ncclCommInitRankDev` | comm 骨架、配置、终止标志、任务参数副本 | 完整运行资源与通信关系 |
| 最外层 GroupEnd | 明确的任务执行者与生命周期交接 | 调度本身不替代初始化 |
| 回调的 GPU 准备 | 正确设备、架构能力、kernel 属性配置 | 跨 rank 握手与拓扑 |
| `commHash` | 紧凑的逻辑组标识 | 连接地址和实际连接 |
| `commAlloc` | 本地资源、后端上下文、管理容器 | 完整 peer 视图和传输连接 |
| bootstrap 环建立 | 相邻 ranks 的控制连接 | 全组服务地址表与算法数据通道 |
| `ringAllInfo → bootstrapAllGather` | 每个 rank 都有全组服务联系信息 | 完整 peer 能力视图、拓扑与数据连接 |
| `ncclProxyInit` | proxyState、监听 socket 与地址表交接、本地 IPC socket | 后续启动的 proxy 服务线程及完整传输准备 |
| `fillInfo → bootstrapAllGather` | 全组身份与能力表，`peerInfoValid` 发布 | 后续兼容性检查和拓扑 |
| peerInfo 检查与成员分析 | 本段能力处理、同进程编号、comm 链表与协调状态 | 硬件路径与算法图 |
| 拓扑发现、裁剪与路径重算 | 硬件图、策略修正后的路径、搜索模型参数 | Ring/Tree 排列、channels 与实际数据连接 |
| CPU affinity 设置 | 初始化线程的局部性选择 | 不强制迁移已有内存，不保证全部 NUMA 分配位置 |

peerInfo 检查行按正常完整扫描路径理解；当前源码的非均匀区间 `break` 边界见第 12.1 节。

### 16.2 常见误解速查

| 容易误解的说法 | 准确理解 |
|---|---|
| `ncclCommInitRank` 一次初始化全部 GPUs | 一次创建本 rank 的本地 comm；多卡需多次调用或 InitAll |
| `cudaGetDevice` 返回一组 GPUs 或已初始化设备句柄 | 返回当前线程的 current device 整数序号 |
| `ncclCalloc` 就是 libc calloc | 此实现是带类型和日志包装的 malloc + memset |
| comm 内存分配完成就是初始化完成 | 还需要子资源、成员协调、拓扑、连接及设备状态 |
| Async 命名意味着一定新建线程 | 看 Group 的 blocking 配置和任务数量；单任务可能直接执行 |
| `hostStream` 是 CPU 工作线程 | 它是 NCCL 管理的 CUDA stream 对象 |
| `commHash` 是地址或安全令牌 | 它是紧凑组标识，非地址、非认证凭据且可能碰撞 |
| unique ID 只有随机字节，没有地址信息 | 内部封装 bootstrap root 地址及匹配信息 |
| root 总是 rank 0 | 默认路径由生成 ID 的进程承载；指定地址路径另有规则 |
| root 必须等齐所有人后才回复 | 相邻信息齐备即可提前回复，最后补发 |
| bootstrap 永远是纯 TCP | 默认 Socket，环还可选择 Net 后端 |
| bootstrap 环就是 AllReduce Ring | 一个交换控制信息，一个传输用户数据，构造目的不同 |
| bootstrap P2P 入口就是公开 Send/Recv 或 GPU P2P | 它用于指定 ranks 间的控制信息交换 |
| 拿到全组地址就已建立全连接 | 联系信息与连接对象不同，后续仍需按需建连 |
| `getUDS` 已经创建 UDS socket | 它只生成 64 位命名标识，`ncclProxyInit` 才创建本地 IPC socket |
| bootstrap AllGather 总是单向 N−1 轮 | 当前默认 Socket 实现双向传播，共 `floor(N/2)` 轮 |
| `bootstrapAllGather` 的 size 是整个数组大小 | 它是单个 rank 记录的字节数，数组需容纳全组记录 |
| `ncclProxyInit` 已启动服务线程 | 它准备状态与入口；后续 `ncclProxyCreate` 才启动服务线程 |
| `peerInfo` 分配 `nranks + 1` 表示多一个参与者 | 额外项是 CollNet root 内部槽位，真实 rank 数不变 |
| 收到别的 rank 的 comm 指针就能解引用 | 只能在确认同进程后使用，跨进程指针不是远程句柄 |
| `peerInfoValid` 表示全部检查已通过 | 只发布表可读，后续检查与完整初始化仍未完成 |
| 连续主机区间长度就是每台机器的总 rank 数 | 它按 rank 顺序统计连续区间，不做任意主机分组计数 |
| 局部 `nNodes` 已是精确节点数 | 本段按异机成员递增，用于异机判定，不等于最终 `comm->nNodes` |
| `intraComm0` 总指向全局 rank 0 | 它指向本通信组在本进程的首个成员 |
| `nvlsRegSupport=0` 表示不能通信或 NVLS 完全不可用 | 它只描述相关注册路径条件，不等于 `nvlsSupport` |
| `comm->topo` 已经包含选好的算法 Ring | 它是硬件与路径模型，算法图由后续搜索产生 |
| 路径计算只找最少跳数 | 当前实现先比较类型，再比较带宽，最后比较跳数 |
| 拓扑路径带宽是本阶段实测结果 | 它是属性和规则形成的模型值，不是运行带宽测试 |
| 裁剪拓扑删除了 communicator 成员 | 只修改局部图，不减少 `nRanks` 或删除 `peerInfo` |
| SearchInit 已完成 Ring/Tree 搜索 | 它准备带宽模型参数，实际搜索在后面 |
| CPU affinity 重绑了应用全部线程并迁移内存 | Linux 此处调整当前初始化线程，退出恢复，不强制迁移已有内存 |
| 初始化成功表示所有算法连接都已提前建立 | 允许按需连接，首次使用时可能继续准备 |
| 异步错误查询成功等于 GPU 数据已完成 | 还需依据 CUDA stream/event 的完成与依赖语义 |

### 16.3 源码定位表

| 入口 | 对应内容 |
|---|---|
| [init.cc:209](../nccl/src/init.cc#L209) | unique ID 的公开生成入口 |
| [init.cc:2946](../nccl/src/init.cc#L2946) | `ncclCommInitRank` 公开包装 |
| [init.cc:183](../nccl/src/init.cc#L183) | 进程级 `ncclInit` |
| [init.cc:2851](../nccl/src/init.cc#L2851) | `ncclCommInitRankDev` |
| [group.cc:36](../nccl/src/group.cc#L36) | `ncclAsyncLaunch` 入队 |
| [group.cc:1039](../nccl/src/group.cc#L1039) | 最外层 GroupEnd 调度 |
| [group.cc:609](../nccl/src/group.cc#L609) | 单/多任务执行分支 |
| [init.cc:2105](../nccl/src/init.cc#L2105) | 初始化回调 |
| [enqueue.cc:55](../nccl/src/enqueue/enqueue.cc#L55) | kernel 资源准备 |
| [init.cc:2179](../nccl/src/init.cc#L2179) | 普通路径 hash |
| [init.cc:512](../nccl/src/init.cc#L512) | `commAlloc` 本地资源初始化 |
| [bootstrap.cc:408](../nccl/src/bootstrap.cc#L408) | root 服务创建 |
| [bootstrap.cc:291](../nccl/src/bootstrap.cc#L291) | root 报到处理与邻居信息分发 |
| [bootstrap.cc:754](../nccl/src/bootstrap.cc#L754) | 各 rank 的 bootstrap 初始化 |
| [bootstrap.cc:687](../nccl/src/bootstrap.cc#L687) | Socket 环连接 |
| [bootstrap.cc:896](../nccl/src/bootstrap.cc#L896-L964) | 服务入口准备、全组信息交换与返回 |
| [bootstrap.cc:558](../nccl/src/bootstrap.cc#L558-L563) | `getUDS` 生成 IPC 命名标识 |
| [bootstrap.cc:701](../nccl/src/bootstrap.cc#L701-L735) | `ringAllInfo` 打包、交换与拆包 |
| [bootstrap.cc:1413](../nccl/src/bootstrap.cc#L1413-L1438) | `bootstrapAllGather` 后端选择 |
| [bootstrap.cc:1363](../nccl/src/bootstrap.cc#L1363-L1412) | 当前 Socket 双向环 AllGather |
| [proxy.cc:2111](../nccl/src/proxy.cc#L2111-L2128) | `ncclProxyInit` 准备状态与 IPC socket |
| [proxy.cc:2130](../nccl/src/proxy.cc#L2130-L2165) | 后续 `ncclProxyCreate` 启动服务线程 |
| [init.cc:1234](../nccl/src/init.cc#L1234-L1239) | `peerInfo` 分配、交换与发布 |
| [init.cc:888](../nccl/src/init.cc#L888-L982) | `fillInfo` 的身份与能力采集 |
| [transport.h:46](../nccl/src/include/transport.h#L46-L74) | `ncclPeerInfo` 记录类型 |
| [init.cc:1241](../nccl/src/init.cc#L1241-L1314) | 连续区间、兼容性检查与能力汇总 |
| [mnnvl.cc:14](../nccl/src/mnnvl.cc#L14-L94) | MNNVL 条件与 Fabric handle 检查 |
| [init.cc:1322](../nccl/src/init.cc#L1322-L1379) | 同进程编号、对象链表与协调状态 |
| [init.cc:1381](../nccl/src/init.cc#L1381-L1412) | 拓扑发现、路径处理与 affinity 主线 |
| [topo.cc:1989](../nccl/src/graph/topo.cc#L1989-L2178) | `ncclTopoGetSystem` 与 XML 融合 |
| [topo.h:46](../nccl/src/graph/topo.h#L46-L183) | 拓扑节点、连接、路径与系统结构 |
| [paths.cc:52](../nccl/src/graph/paths.cc#L52-L147) | 路径扩展与候选比较规则 |
| [paths.cc:754](../nccl/src/graph/paths.cc#L754-L898) | `ncclTopoComputePaths` 与策略修正 |
| [paths.cc:900](../nccl/src/graph/paths.cc#L900-L945) | `ncclTopoTrimSystem` 裁剪局部图 |
| [search.cc:18](../nccl/src/graph/search.cc#L18-L63) | 搜索参数与 CPU 类型准备 |
| [topo.cc:2447](../nccl/src/graph/topo.cc#L2447-L2490) | CPU affinity 选择 |
| [init.cc:1414](../nccl/src/init.cc#L1414-L1437) | 下一步：CollNet/NVLS 与算法图规划 |

### 16.4 当前断点与接下来的顺序

**已讲解至 `initTransportsRank` 的 CPU affinity 设置完成（`init.cc:1412`），下一次从 [init.cc:1414](../nccl/src/init.cc#L1414-L1437)继续。特殊初始化分支、完整异常清理和部分深层辅助函数仍未逐行展开。**

1. 简要看 CollNet 支持检查与 `ncclNvlsInit`，再进入 `ringGraph/treeGraph → ncclTopoCompute`，理解硬件图如何变成算法通信图。
2. 分析图信息交换、参数协调、节点/rank 映射与 channels。
3. 继续 proxy 服务启动、transport 资源、连接策略与设备状态准备。
4. 最后回看成功发布、异常处理、按需连接与首次 collective 的衔接。

复习时可先不看源码，尝试回答：unique ID、commHash、rank、comm 指针各解决什么问题？入队后是谁执行？服务地址表、peerInfo、硬件图、算法图各自提供什么？为什么路径需要计算两次？每个阶段的产物由下一层的哪个函数消费？
