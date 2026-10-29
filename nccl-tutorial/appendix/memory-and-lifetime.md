# 09 内存与生命周期：谁持有对象，何时交还资源

[上一章：Proxy 与网络](../08-proxy-and-network.md) · [教程主页](../README.md)

两台机器各 4 GPU，8 rank 各一进程，每 rank 做 64 MiB FP32 Sum AllReduce。
继续追普通 host API 规划出的 Ring/Simple 路径。
前面已经说明 buffer 如何被访问，本章沿对象寿命把分配、映射、注册和回收连起来。

## 1. 先给三类对象命名

每个 rank 用一块 64 MiB 的用户缓冲 `U` 做原地 AllReduce，
调用时收发指针都指向 `U`，`count=16,777,216`。
再选一条 NET 连接的内部 Simple 缓冲 `B`，以及本次工作的描述集合 `D`。

- **U：用户分配。** 应用拥有地址与内容，负责最后释放；NCCL 保存其地址供异步执行。
- **B：内部缓冲。** transport/proxy 维护 payload 槽、head/tail、FIFO 元数据及 MR。
- **D：执行描述。** planner 创建 plan、work、proxyOp；GPU 与 proxy 各自接收所需部分。

```text
U：分配 ─── 注册 / 映射关系 ─── 多次通信 ─── 解除关系 ─── 释放
B：连接建立 ─────────────── 多次通信复用 ─────────── 连接回收
D：每次提交 → work / proxy 工作移交 → 各对象按自己的完成条件回收
```

同一段数据会出现在多个对象里：D 记录 U 的地址，MR 描述某段内存的 NIC 访问资格，
B 暂存切片。所有权、可访问性和执行进度分别由不同层维护。
一次集合结束后，应用常把用户缓冲留给下一轮，连接也继续复用内部缓冲；
本次执行描述则陆续退出。看内存占用时，把长期复用的资源与仍在途的资源分开，
再追各自引用的增加、移交与减少，就能解释为何不同对象在不同时间释放。

## 2. Allocation：先有物理存储和本地 VA

应用可以用 CUDA 分配 U，也可以用 `ncclMemAlloc` 获得适合 NCCL 优化的分配。
后者在 cuMem 路径上创建物理 allocation handle，保留 VA、map 后设置设备访问权限，
并按粒度扩大实际分配尺寸；与之配对的释放接口是 `ncclMemFree`。

U 始终由应用持有。即便调用了 NCCL allocator，用户内存也由应用决定寿命，
与 NCCL 内部 memory manager 跟踪的通信资源分开。

B 则在 transport 连接建立期间创建。
NET 为协议缓冲建立 `connectMap`，记录 host/device memory 区域，
并分别保存 CPU 使用的地址和 GPU 使用的地址。
GDR/Simple 的 payload 可放在 GPU memory，控制字仍按访问双方的需要布置。

开发时先确认某个指针属于 U、B 还是 D，再找它的释放函数。
相同的 `void*` 类型可能代表 GPU VA、主机结构或插件句柄，字段归属比类型更有信息量。

## 3. IPC mapping/access：让另一进程的 GPU 能访问

跨进程共享 GPU memory 时，拥有者导出可共享句柄，接收者导入物理分配，
在自己的 VA 空间 reserve/map，再为本 GPU 设置 access。
cuMem 分支里，这几步分别对应 import、address reserve、map、set access。

```text
物理 allocation
  ├─ 拥有者 VA → 本 GPU
  └─ 导入者 VA → 对端进程的 GPU
```

两个 VA 可以不同；共同指向的 allocation 以及访问权限把它们联系起来。
legacy CUDA IPC 走 `cudaIpcOpenMemHandle`；cuMem 导入对象由 memory manager 记录，
其中包含 peer、allocation handle、映射范围等信息。

这一步服务于同机 P2P，或需要跨进程访问缓冲的代理连接。
第五章的 connector 保存最终可用的 GPU 地址；注册缓存还能保存用户缓冲的 peer 映射。
关闭导入映射回收的是本地视图，拥有者分配还有自己的寿命。

## 4. MR registration：让 NIC 能访问指定范围

NET 连接持有插件的 sendComm/recvComm。
service 侧用 `regMr` 或 `regMrDmaBuf` 注册 B，得到 `mhandles[protocol]`，
随后 progress 将对应 handle 连同地址和长度传给 `isend/irecv`。
插件维护具体 MR 对象，transport 保存不透明 handle；GDR 路径由驱动建立 NIC 的访问关系。
这也解释了为何注销从 transport 发起，实际资源释放落在插件与驱动中。

把链条按职责排开，就是：

```text
allocation → IPC mapping/access（需要跨进程访问时）
           → MR registration（需要 NIC 访问时）
           → 异步使用 → 完成 → 注销 / 解除映射 → 释放 allocation
```

映射与 MR 是两种访问关系，具体连接按需要建立它们。
内部 B 的 MR 常与连接同寿命，持续供多次 AllReduce 使用；
插件 request 则跟随一片或一组网络传输完成。

用户注册走另一层：`ncclCommRegister(comm, U, bytes, &handle)`
先在 communicator 的 `regCache` 中登记页面区间。
后续算法/transport 使用该记录时，再建立需要的 NET、IPC 等具体状态。
具备相应能力的路径可以直接访问 U，减少经 B 中转的复制。

### 注册缓存的两个引用来源

`ncclReg` 保存页面对齐的 `begAddr/endAddr`、传输状态和具体句柄。
再次注册被已有记录覆盖的范围，会复用记录并增加引用：

- `localRefs`：应用显式注册获得的本地引用。
- `graphRefs`：内部 Graph 注册获得的引用。

例如同一记录先有 `(localRefs, graphRefs)=(1,0)`，
被一个 Graph 注册引用后成为 `(1,1)`；应用注销本地 handle 后是 `(0,1)`。
最后一个 Graph 引用解除，才进入 `regCleanup`，释放对应 MR 与 IPC 资源。

缓存复用依赖 allocation 持续有效。应用把 U 留到最后一次异步使用结束，
并完成相关注册与映射的解除，再交还 allocator。
这样，下次即使重用相同 VA，也会重新建立属于新分配的访问关系。

## 5. D 的回收：plan 与 work FIFO 分走两条路

提交时，planner 把任务编成 `ncclKernelPlan`。
`uploadWork` 将设备 work 放进 kernel 参数、comm work FIFO 或 persistent work buffer；
`uploadProxyOps` 将 proxy 工作交给共享 op pool。

**第一条路回收 host plan。**
`hostStreamPlanTask` 完成 `uploadProxyOps` 和 `ncclProxyStart` 后，
把非 persistent plan 的 reclaimer 放入 `comm->callbackQueue`。
主机轮询该队列就可以执行 `reclaimPlan`，归还 plan、任务和原始 proxyOp。
这个回收条件是 host 侧移交结束，GPU 此时可以仍在运行。

移交能够成立，是因为后续执行者已有自己的状态。
`ncclLocalOpAppend` 在 `proxy.cc:515–516` 复制整个 op，
并为其中的 `ringAlgo` 增加引用；转换后的 sub 接着持有它。
plan 回收和 proxy 完成分别减引用，共享算法对象活到最后一方释放。

**第二条路归还 comm work FIFO 容量。**
FIFO 中的设备 work 仍供 kernel 读取，容量由 `workFifoProduced/workFifoConsumed` 维护。
这里计数的是 **work 字节**，与上一章网络协议的 step 分开。

`ncclLaunchFinish` 在 launch stream 上记录完成事件，
并在达到记录阈值时，把当前 produced 快照封装为 `KernelFinishCallback`，
放进 `eventCallbackQueue`。

`ncclCommPollEventCallbacks` 先 query 或 synchronize 这个 CUDA event，
事件成功后才调用 `KernelFinishCallback_fn` 更新 `workFifoConsumed`。
`waitWorkFifoAvailable` 据此判断下一批 work 能否覆盖旧区域。

```text
proxy 工作已复制并发布 → callbackQueue      → 回收普通 host plan
launch stream 完成事件 → eventCallbackQueue → 推进 workFifoConsumed
```

这两个队列可以在不同时间被处理。
修改描述布局时，分别检查“描述是否已移交”和“存储是否仍被设备读取”，就能找到正确回收点。

## 6. CUDA Graph：完成一轮后，计划仍供下一轮使用

捕获时，NCCL 将相关 plan 标为 persistent，
为重放保留 work、proxy 描述及清理回调，并增加 communicator 的 persistent 引用。
CUDA user object 把这些资源的析构绑定到图的引用寿命。

每次 replay 继续使用捕获的地址与依赖，必要的 host callback 重新上传 proxy 工作。
本轮 GPU 完成以后，U 的内容可以按依赖更新，plan 仍为下一轮重放保留。
当相关 graph/exec 的引用最终释放，`persistentDestructor` 才把 plan 送入 callbackQueue。

`reclaimPlan` 处理 persistent work buffer、注册清理回调和 persistent 引用。
Graph 注册通常随计划清理；legacy IPC 注册的清理另挂在 comm 的
`legacyRegCleanupQueue`，收尾时排空。

因此要同时标出两个终点：最后一次设备使用完成，以及最后一个 Graph 引用解除。
comm 收尾会等待其 `localPersistentRefs` 归零；应用先结束重放、释放图引用，
就给 NCCL 留出了完整的资源回收通路。

## 7. 正常收尾：Finalize 开始排空，Destroy 回收 comm

一个清楚的应用收尾顺序是：停止新增通信，等待相关 stream 工作完成；
结束 Graph 重放并释放 graph/exec 引用，注销应用的注册 handle；
各 rank 执行 Finalize，完成后 Destroy，最后释放保留的用户分配 U。

`ncclCommFinalize` 启动收尾任务 `commDestroySync`：
同步内部 host/device streams，处理事件回调和主机回调，等待图引用结束，再停止 proxy。
非阻塞 communicator 可用 `ncclCommGetAsyncError` 观察收尾进展。
`ncclCommDestroy` 回收剩余 communicator 资源；故障退出用 `ncclCommAbort` 放弃在途操作。

收尾中有一个 **host-local barrier**：
`init.cc:3155–3172` 让同一物理主机上的 rank 一起到达，再停止 proxy，
使其他 rank 尚在完成的 PXN 建连可以继续使用本机代理。
案例中每台机器的四个独立进程需要共同进入收尾。

若改成单线程管理多 GPU，显式 blocking Finalize 应把各 GPU 的调用放进同一 group；
使用 nonblocking comm 时，可以先发起全部 Finalize，再轮询各自状态。
这样每个本机 rank 的收尾任务都已启动，能够共同到达 barrier。

直接逐个 Destroy 有专门的协调路径：`commReclaim` 先累计本进程的到达数，
最后一个到达者为尚未 Finalize 的 comm 组织并发 `commDestroySync`，再统一清理。
这里应沿 Destroy 自己的协调逻辑理解，而非套用逐个 blocking Finalize 的调用顺序。

## 8. 关键源码入口

以下位置按本地 NCCL `12df1a11` / `2.32.3` 核对：

1. **分配与导入**：[src/allocator.cc:17](../../nccl/src/allocator.cc#L17) `ncclMemAlloc`；
   [src/transport/p2p.cc:267](../../nccl/src/transport/p2p.cc#L267) `ncclP2pImportShareableBuffer`。
2. **内部缓冲的 MR**：[src/transport/net.cc:1004](../../nccl/src/transport/net.cc#L1004) `sendProxyConnect` 注册循环；
   [1216](../../nccl/src/transport/net.cc#L1216) `sendProxyFree` 先注销 MR，再释放缓冲和连接。
3. **用户注册缓存**：[src/register/register.cc:26](../../nccl/src/register/register.cc#L26) `ncclRegister`；
   [172](../../nccl/src/register/register.cc#L172) `commDeregister` 检查 local/graph refs。
4. **plan 移交与清理**：[src/enqueue/enqueue.cc:1558](../../nccl/src/enqueue/enqueue.cc#L1558) `hostStreamPlanTask`、
   [1589](../../nccl/src/enqueue/enqueue.cc#L1589) `reclaimPlan`、[1655](../../nccl/src/enqueue/enqueue.cc#L1655) `persistentDestructor`；
   [src/proxy.cc:515–516](../../nccl/src/proxy.cc#L515) `ncclLocalOpAppend` 的复制与引用交接。
5. **FIFO 完成边界**：[src/enqueue/enqueue.cc:2028](../../nccl/src/enqueue/enqueue.cc#L2028) `KernelFinishCallback_fn`、
   [2037](../../nccl/src/enqueue/enqueue.cc#L2037) `ncclLaunchFinish`；
   [src/include/comm.h:916–935](../../nccl/src/include/comm.h#L916) `ncclCommPollEventCallbacks` 确认事件后执行回调。
6. **收尾协调**：[src/init.cc:3122](../../nccl/src/init.cc#L3122) `commDestroySync`；
   [3255–3307](../../nccl/src/init.cc#L3255) `commReclaim` 的最后到达者与并发收尾。

## 9. 离线任务：为一次 AllReduce 画三条寿命线

在纸上画 U、B、D，给 D 再分出 host plan 与 work FIFO 两条线。
从 `hostStreamPlanTask` 追到 `reclaimPlan`，再从 `ncclLaunchFinish` 追到事件回调。
最后加入“同一 U 被 Graph 注册引用”的分支，标出 `(1,1) → (0,1) → (0,0)`。

核对点：普通 plan 的回收证据是工作移交，FIFO 的复用证据是完成事件；
proxy 的副本和 `ringAlgo` 引用能跨过 plan 回收点；
GPU 一轮完成后，Graph 仍可持有 U 的注册关系；B 的 MR 由连接清理负责。

这项任务只需本地源码；本章未执行 GPU 实验。

下一章：[开发工作流](../10-development-workflow.md)。
