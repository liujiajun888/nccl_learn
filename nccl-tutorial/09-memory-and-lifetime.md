# 09｜内存与生命周期：谁拥有资源，何时可以回收

前面已经看到多种地址：用户 buffer、协议缓冲、共享映射、MR、work 和 proxy 请求。它们并不是同一种对象，也不能在同一时刻统一释放。

本章只用三类对象理解生命周期：**U（用户内存）、B（内部缓冲）、D（执行描述）。** 具体字段和回收队列见 [内存与生命周期进阶附录](appendix/memory-and-lifetime.md)。

## 1. 先分清三类对象

| 对象 | 谁拥有 | 通常何时结束 |
|---|---|---|
| U：用户 buffer | 应用分配并管理 | 最后一次通信和注册/映射解除后，由应用释放 |
| B：内部协议缓冲 | transport/proxy | 连接或 communicator 资源回收时释放 |
| D：执行描述 | planner、work FIFO、proxy args 等分别管理 | 各自的使用者完成读取后回收 |

同一份 AllReduce 中：

```text
D 保存 U 的地址和执行参数
B 暂存通信分片
MR 描述 NIC 对某段内存的访问资格
```

**地址、访问权和执行进度是三件不同的事。**

## 2. 分配、映射、注册是三个不同阶段

```text
分配：先有一段内存和本地地址
映射：让另一个进程或 GPU 能访问它
注册：让 NIC 能按网络权限访问它
```

### 分配

应用可以使用 CUDA allocator，也可以使用 `ncclMemAlloc` 获得更适合 NCCL 的分配。无论哪种，U 都由应用决定最终释放时机。

### IPC 映射

跨进程访问 GPU 内存时，拥有者导出 handle，接收者导入并建立自己的地址映射和访问权限。

两边虚拟地址可以不同；关闭导入映射只结束本地视图，不会自动释放拥有者的 allocation。

### MR 注册

NET 通过 `regMr` 或 `regMrDmaBuf` 让 NIC 获得访问资格。transport 保存不透明 handle，具体 MR 对象由插件和驱动管理。

因此：

> **映射解决“另一个进程/GPU 能否访问”，MR 解决“NIC 能否访问”。两者不能互相替代。**

## 3. 用户注册为什么可以复用？

`ncclCommRegister` 会先在 communicator 的注册缓存中记录页面区间，后续算法和 transport 再按需要建立 MR、IPC 等具体状态。

缓存中常见的引用来源有：

- `localRefs`：应用显式注册产生的引用；
- `graphRefs`：CUDA Graph 捕获产生的引用。

例如：

```text
(1,0) → Graph 注册后 (1,1)
     → 应用注销本地 handle 后 (0,1)
     → 最后一个 Graph 引用解除后 (0,0)，进入清理
```

所以，**应用注销本地 handle 不一定意味着注册资源立即释放**，还要看 Graph 是否仍持有引用。

用户分配必须保留到最后一次异步使用和相关访问关系都结束以后。

## 4. D 的回收要分成两条路

### 第一条：host plan 的回收

普通非 persistent plan 在 proxy 工作已经复制并发布后，可以进入 host 回收队列。

此时 GPU 可能仍在运行，但后续执行者已经拿到需要的描述，所以 host plan 可以回收。

```text
proxy 工作已复制并发布 → callbackQueue → reclaimPlan
```

这并不意味着用户输出已经可读。

### 第二条：work FIFO 容量的回收

FIFO 中的 work 仍可能被 kernel 读取，不能只凭 host plan 已移交就覆盖旧区域。

NCCL 在 launch stream 上记录完成事件，事件确认后再推进 `workFifoConsumed`，允许后续 work 复用空间。

```text
launch stream 完成事件 → eventCallbackQueue → 更新 workFifoConsumed
```

这两条回收线要分开：

| 回收对象 | 允许回收的证据 |
|---|---|
| host plan | host 侧工作描述已移交，后续执行者已有自己的状态 |
| work FIFO 空间 | 对应 device 工作已完成，旧描述不再被读取 |

## 5. CUDA Graph 为什么延长描述寿命？

Graph 捕获的 plan 需要为重放保留 work、proxy 描述和清理回调，因此变成 persistent。

一轮 replay 结束后：

- U 的内容可以按 stream 依赖更新；
- plan 仍要保留给下一次 replay；
- 只有 Graph/Exec 的引用最终释放，persistent plan 才进入回收。

所以 Graph 场景有两个终点：

```text
最后一次设备使用完成
+
最后一个 Graph 引用解除
```

## 6. 正常收尾：先停止新工作，再回收 communicator

一个稳妥的顺序是：

```text
停止新增通信
  → 等待相关 stream 工作完成
  → 结束 Graph 重放并释放 graph/exec 引用
  → 注销应用注册 handle
  → 各 rank Finalize
  → Destroy
  → 最后释放用户保留的 U
```

`ncclCommFinalize` 会协调内部 stream、事件回调、Graph 引用和 proxy 收尾；非阻塞 comm 可继续用 `ncclCommGetAsyncError` 观察状态。

本机 ranks 还需要协调收尾：同一主机上的其他 rank 可能仍在通过 PXN 使用本机 proxy，不能在它仍被需要时提前停掉。

故障退出则使用 `ncclCommAbort` 放弃在途操作，不要把它当成正常完成路径。

## 7. 本章自测

1. 用户 U、内部 B、描述 D 分别由谁释放？
2. IPC 映射和 MR 注册分别解决什么问题？
3. 为什么注销本地注册 handle 后资源可能仍未释放？
4. host plan 回收和 work FIFO 容量回收为什么不是一回事？
5. 为什么 Graph 一轮完成后 plan 还不能立即销毁？

下一步读 [10｜开发工作流](10-development-workflow.md)，把这条软件栈地图用于第一次小改动。

需要注册缓存字段、回收队列、persistentDestructor 和 Finalize/Destroy 的具体协调时，再读 [内存与生命周期进阶附录](appendix/memory-and-lifetime.md)。
