# ncclCommInitRank 源码精读：从 API 入口到 bootstrap 初始化

这篇文档汇总已经讲解过的 communicator 初始化链路，按“做了什么、为什么需要、给下一步留下什么”组织，便于反复查阅。它是 [03｜Communicator](03-communicator.md) 的逐段阅读补充，不是全部 NCCL API 的清单。

- 源码基线：NCCL **2.32.3**，提交 `12df1a11afad322be5a204a2db890161cbf8131d`。
- 整理日期：2026-10-12。源码链接指向本地仓库；换版本后请按符号重新定位。
- 主线假设：普通新建 communicator、单 unique ID、未设置 `NCCL_COMM_ID`、默认 Socket bootstrap。
- 调度主线：`NCCL_ENQUEUE_REARCH_ENABLE=0`；同时说明阻塞/非阻塞和新入队路径的差异。
- **当前精读进度：已覆盖 `bootstrapInit` 后半段的服务入口、全组信息交换、proxy 状态准备及返回，即 `bootstrap.cc:896–964`。下一步进入 `initTransportsRank` 的首轮信息交换。**
- 后续传输初始化和收尾只保留已讲过的概览，未展开部分明确标注。

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
11. [特殊分支与后续流程概览](#remaining)
12. [状态、错误与任务生命周期](#state)
13. [阶段小结、易混点与下一步阅读](#recap)

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

<a id="remaining"></a>

## 11. 特殊分支与后续流程概览

本节保留已经介绍过的整体轮廓，不代表这些内部实现已经逐行学完。

### 11.1 Split、Shrink、Grow 的入口差异

源码：[init.cc:2136–2191](../nccl/src/init.cc#L2136-L2191)。

- **Split/Shrink：**`job->parent && !job->isGrow`。先计算新成员和新 rank，派生子组 hash，然后 `commAlloc(comm, parent, ...)`，再通过 `bootstrapSplit` 借助父组完成协调。
- **Split 不参与新组：**`NCCL_SPLIT_NOCOLOR` 的 rank 仍参加所需的信息交换，随后提前退出，不走普通成功创建子组的完整路径。
- **Grow：**与普通初始化同属 `else` 大分支，但 hash、旧成员信息和 bootstrap 会合方式有特殊处理，不能直接套普通单 ID 图。
- **普通 INIT：**使用第一个 ID 内容的 hash，`commAlloc` 不复用父组资源，然后 `bootstrapInit`。

把共同的资源初始化与不同的成员协调分开，可以复用主要逻辑，又不强行把所有派生场景写成同一种握手。

### 11.2 initTransportsRank：完整通信准备的大头

回到 [init.cc:2196](../nccl/src/init.cc#L2196)。整体职责是交换 peer 信息、分析成员关系和拓扑、规划通信图与 channels、按策略准备传输资源，并建立 GPU 使用的设备状态。

这里不能简单概括为“初始化时一次性建好所有算法的全部连接”。源码会根据 cuMem 支持和运行时连接配置选择分支，见 [init.cc:1811](../nccl/src/init.cc#L1811)；某些连接可延迟到首次使用相关算法时建立。

后续精读会继续拆成“信息交换 → 拓扑与图 → channel/transport → 设备状态”几个阶段。宏观框架可对照 [03｜Communicator](03-communicator.md)。

<a id="state"></a>

## 12. 状态、错误与任务生命周期

### 12.1 完成与失败通过 initState 发布

回调执行完传输初始化后尝试启动进度计数监控；该监控失败会记日志，但不阻止初始化主线继续。随后发布成功，见 [init.cc:2198–2208](../nccl/src/init.cc#L2198-L2208)：

```cpp
COMPILER_ATOMIC_STORE(&comm->initState, ncclSuccess,
                      std::memory_order_release);
```

失败路径则将错误码写入 `initState`，见 [2241–2250 行](../nccl/src/init.cc#L2241-L2250)。

为什么用 release？任务可能由另一个线程执行。正确配对的 acquire 读取观察到发布状态后，应同时观察到此前完成的初始化写入；不是只让一个状态整数发生变化。

普通路径早已在 `ncclCommInitRankDev` 写入 `*newcomm`。回调末尾的 `if (job->newcomm)` 是可选交接，不能理解为所有普通初始化句柄都只在这个位置返回。

### 12.2 四种“完成”不要混淆

| 状态或事件 | 含义 |
|---|---|
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

### 12.3 错误宏表达不同的失败策略

见 [checks.h:24–43](../nccl/src/include/checks.h#L24-L43) 和 [158–166 行](../nccl/src/include/checks.h#L158-L166)。

| 宏 | 行为 |
|---|---|
| `CUDACHECKGOTO` | CUDA 失败时记录、清理 CUDA 错误状态、设置 NCCL 错误码并跳转 |
| `CUDACHECKIGNORE` | 记录并清理 CUDA 错误状态，但继续执行 |
| `NCCLCHECKGOTO` | 保存 NCCL 返回值；除 Success/InProgress 外，错误转到指定标签 |

失败不意味着在该标签里直接销毁所有对象；任务、comm 子资源与 Group 有各自的清理责任。不能把 `goto fail` 简化为“立即 free 整个 communicator”。

### 12.4 为什么保留分阶段计时？

`timers[]` 用 `clockNano()` 记录 kernel、alloc、bootstrap、拓扑、图、连接等阶段耗时，最后输出 `Init timings`。这样初始化慢时可以先判断时间花在哪一层，而不是只得到一个总时长。

回调的总计时从回调开始算起，不包含此前等待 GroupEnd 的全部时间，也不能直接等同于从 `ncclGetUniqueId` 开始的应用端总耗时。

**本节小结：初始化结果、调度线程结束和 GPU 工作结束是不同边界；任务清理也不等于销毁成功创建的 comm。**

<a id="recap"></a>

## 13. 阶段小结、易混点与下一步阅读

### 13.1 一张表复习已深入学习的过程

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

### 13.2 常见误解速查

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
| 初始化成功表示所有算法连接都已提前建立 | 允许按需连接，首次使用时可能继续准备 |
| 异步错误查询成功等于 GPU 数据已完成 | 还需依据 CUDA stream/event 的完成与依赖语义 |

### 13.3 源码定位表

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
| [init.cc:1234](../nccl/src/init.cc#L1234-L1239) | 下一步：`fillInfo → bootstrapAllGather` |

### 13.4 当前断点与接下来的顺序

**已讲解至 `bootstrapInit` 返回（`bootstrap.cc:964`），下一次回到 [init.cc:2196](../nccl/src/init.cc#L2193-L2196)，进入 `initTransportsRank`。特殊初始化分支和完整异常清理仍未逐行展开。**

1. 先看 [init.cc:1234–1239](../nccl/src/init.cc#L1234-L1239) 的 `fillInfo → bootstrapAllGather`：从服务联系表进一步得到各 rank 的设备、进程、主机与能力信息。
2. 分析成员关系与能力约束，再进入 GPU/NIC 拓扑发现和路径计算。
3. 继续算法图与 channels、proxy 服务启动、transport 资源及设备状态准备。
4. 最后回看成功发布、异常处理、按需连接与首次 collective 的衔接。

复习时可以先不看源码，尝试回答：unique ID、commHash、rank、comm 指针各解决什么问题？入队后是谁执行？每个阶段的产物由下一层的哪个函数消费？这几个问题能把整条初始化链路串起来。
