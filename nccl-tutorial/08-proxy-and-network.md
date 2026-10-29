# 08｜Proxy 与网络：GPU、CPU 和 NIC 怎样推进一片数据

上一章建立了 NET 连接。本章只追一条跨机边：**GPU 产生一片数据后，proxy 怎样提交网络请求，接收 GPU 又何时能读、何时归还槽位？**

proxy op 的聚合、args/sub 字段和插件状态机留到 [Proxy 与网络进阶附录](appendix/deep-dive/proxy-and-network.md)。

## 1. 先分清 proxy 的两类工作

| 角色 | 负责什么 |
|---|---|
| Proxy service | 处理控制请求：setup、connect、注册、映射和释放 |
| Proxy progress | 推进在途通信：提交网络请求、查询完成、发布 GPU 可见状态和回收信用 |

可以把它理解成：

```text
service：把连接和资源准备好
progress：让已经规划好的网络工作持续前进
```

应用线程不需要逐片等待网络完成，GPU、proxy 和 NIC 可以并行推进不同阶段。

## 2. 一次 proxy 工作怎样变成网络请求？

host plan 中的 `ncclProxyOp` 描述某个 channel 上要推进的通信工作，包括连接、协议、步数和缓冲信息。

```text
plan.proxyOpQueue
  → 复制到共享 op pool 并发布
  → proxy 转成 active args / sub
  → 调用插件 isend / irecv
  → 得到网络 request
  → test 查询完成并更新进度
```

三种粒度不要混淆：

```text
一次 collective：整个 AllReduce
一个 proxy op：某个连接上的一批通信工作
一个网络 request：由插件提交的一次或一组传输
```

连接和 MR 通常长寿命复用；args/sub 描述本次工作；request 则随网络操作完成而结束。

## 3. 先记住四个控制状态

为了只理解职责，给它们起教学名字：

| 状态 | 谁推进 | 表示什么 |
|---|---|---|
| `S_tail` | 发送 GPU | 已经生产到哪个 step，send proxy 可以读到哪里 |
| `S_head` | send proxy | 发送槽已网络消费到哪里，GPU 可以复用到哪里 |
| `R_tail` | recv proxy | 接收数据已经可见到哪里，GPU 可以读到哪里 |
| `R_head` | 接收 GPU | 已经消费到哪里，recv proxy 可以复用接收槽到哪里 |

可以先只记：

```text
tail：生产进度，告诉对方“可以读”
head：消费进度，告诉对方“可以复用”
```

这些字段在发送和接收连接上的具体地址由 transport 准备，职责比字段名更重要。

## 4. 发送方向：GPU 发布 → proxy 发送 → proxy 归还

```text
① 发送 GPU 检查 S_head，确认还有可写空间
② GPU 写入 payload，并推进 S_tail
③ send proxy 观察到 S_tail，调用 isend
④ NIC 发送数据；send proxy 用 test 确认完成
⑤ send proxy 推进 S_head，发送 GPU 获得新的可写空间
```

这里有两个不同的“完成”：

- GPU 已发布数据：`S_tail` 已推进。
- NIC 已读完源槽：`S_head` 才推进，源槽才能复用。

如果插件返回成功但 request 为空，表示这次尚未真正接受请求，progress 下一轮再试。

## 5. 接收方向：网络完成 → GPU 可读 → GPU 已消费

```text
① recv proxy 预先提交 irecv
② NIC 将数据写入接收槽，recv proxy 用 test 确认网络完成
③ 必要时执行 flush，建立 GPU 可见性
④ recv proxy 推进 R_tail，通知接收 GPU 可以读取
⑤ GPU 完成复制或规约，推进 R_head
⑥ recv proxy 看到 R_head，允许接收槽再次用于网络写入
```

所以，接收侧有三个边界：

| 边界 | 含义 |
|---|---|
| 网络完成 | 插件 `test` 报告 request 完成 |
| GPU 可读 | 必要的可见性处理完成，`R_tail` 已发布 |
| GPU 已消费 | GPU 完成使用并推进 `R_head`，接收槽才能复用 |

**网络完成不等于 GPU 已经可读，更不等于接收槽已经可复用。**

## 6. step、base 和槽位怎样对应？

head/tail 是持续增长的绝对 step；proxy 内部的 posted/received/transmitted/done 通常是相对本次 `base` 的计数。

例如：

```text
base=16，sliceSteps=2，NCCL_STEPS=8
本片覆盖 [16,18)，起始槽号 = 16 % 8 = 0
```

可以简化成：

```text
GPU 发布：S_tail = 18
发送完成：S_head = 18
接收可见：R_tail = 18
接收消费：R_head = 18
```

绝对 step 用来区分同一物理槽的不同代际；`step % NCCL_STEPS` 只决定当前落在哪个槽。

## 7. 背压是怎样传回上游的？

如果接收 GPU 一直不消费：

```text
R_head 不前进
  → recv proxy 不能复用接收槽
  → 对端发送请求和本端生产逐渐占满窗口
  → 上游发送者也要等待 credit
```

因此，长 kernel 不一定表示网络慢，也可能是：

- 对端 GPU 尚未消费；
- proxy 尚未发布接收可读状态；
- 某个 send/recv/flush request 尚未完成；
- 应用 stream 的前序依赖尚未结束。

调试时按 `(rank, channel, direction, step)` 对齐 GPU 与 proxy 状态，比只看“进入过 proxy 函数”更有效。

## 8. 本章主线复习

```text
发送 GPU 生产并发布 S_tail
  → send proxy 提交并完成网络请求
  → S_head 归还源槽
  → recv proxy 确认网络完成和可见性
  → R_tail 通知接收 GPU
  → 接收 GPU 消费并发布 R_head
  → 接收槽重新可用
```

下一步读 [09｜内存与生命周期](09-memory-and-lifetime.md)，看用户 buffer、内部缓冲、连接和执行描述分别什么时候回收。

需要 proxy op 聚合、request 状态机和插件接口细节时，再读 [Proxy 与网络进阶附录](appendix/deep-dive/proxy-and-network.md)。
