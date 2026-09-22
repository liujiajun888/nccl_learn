# 02 集合通信：先算对结果，再谈性能

## 学习目标

- 给出每种操作的输入/输出长度、rank 顺序和合法的原地布局。
- 分清 `count` 是元素数还是字节数，是每个 peer 的块还是完整数组。
- 从四个 rank 的数据亲手推导结果。

本章 `P` 表示 communicator 的 rank 数，`r` 是当前 rank，`C` 是元素数，`sizeof(T)` 是一个元素的字节数。所有数组按元素展示，不按字节展示。

## 1. 一张表掌握接口契约

| 操作 | API 中 count 的含义 | 每 rank 发送缓冲容量 | 每 rank 接收缓冲容量 | 结果位置 |
|---|---|---:|---:|---|
| AllReduce | 完整输入 `C` | C | C | 每个 rank 都有完整规约结果 |
| Reduce | 完整输入 `C` | C | root 为 C | 仅 root |
| Broadcast | 广播长度 `C` | root 为 C | C | 每个 rank |
| AllGather | 每 rank 的 `sendcount=C` | C | P×C | 按源 rank 拼接 |
| ReduceScatter | 每 rank 的 `recvcount=C` | P×C | C | rank r 获得第 r 块规约结果 |
| AlltoAll | 每个 peer 的 `count=C` | P×C | P×C | 接收缓冲按源 rank 分块 |
| Gather | 每 rank 的 `count=C` | C | root 为 P×C | root 按源 rank 拼接 |
| Scatter | 每 rank 的 `count=C` | root 为 P×C | C | rank r 收到 root 的第 r 块 |

非 root 的非相关指针按各 API 契约处理，不能用一条“所有指针都可以为 NULL”的规则概括。

特别容易错的是 ReduceScatter：传入的不是完整发送长度，而是**每个 rank 最终接收的长度**。如果 `P=4, recvcount=1024, type=float`，发送缓冲至少是 `4×1024×4=16384` 字节。

## 2. AllReduce 与 Reduce：求和并决定结果归属

输入：

```text
r0: [1, 2]   r1: [3, 4]   r2: [5, 6]   r3: [7, 8]
```

`AllReduce(Sum, count=2)`：

```text
r0: [16, 20]  r1: [16, 20]  r2: [16, 20]  r3: [16, 20]
```

`Reduce(Sum, root=2, count=2)`：只有 r2 的接收缓冲得到 `[16, 20]`。其他 rank 的接收内容不是可用的全局结果。

规约操作包括 `ncclSum`、`ncclProd`、`ncclMax`、`ncclMin`、`ncclAvg`；类型/路径支持仍需匹配。NCCL 的自定义规约能力不是传入任意 CPU/CUDA 函数指针：例如 `ncclRedOpCreatePreMulSum` 表达先乘标量再求和，见[进阶章](13-advanced.md)。

### 浮点数为何可能每次不完全相同

实数加法满足结合律，有限精度浮点加法一般不满足。例如按 FP32 近似计算：

```text
(1e20 + -1e20) + 3.14 ≈ 3.14
1e20 + (-1e20 + 3.14) ≈ 0
```

算法、拓扑、分块和规约顺序变化可能改变舍入结果。校验浮点结果通常用合理的绝对/相对误差，不默认要求所有配置逐位一致。也不能把任意巨大误差都归咎于舍入；先排查计数、输入初始化和同步。

## 3. Broadcast：一份输入复制给所有人

如果 root=1，r1 持有 `[9, 8]`：

```text
r1 send [9, 8] -> r0/r1/r2/r3 recv 都是 [9, 8]
```

使用 `ncclBroadcast` 的 send/recv 形式。旧 `ncclBcast` 是已弃用的原地接口，不建议新示例使用它。

它与 AllGather 的区别是：Broadcast 的来源只有一个 root；AllGather 中每个 rank 都贡献一个块。

## 4. AllGather：按 rank 顺序拼起来

`P=4, sendcount=2`：

```text
r0 send [10,11]   r1 send [20,21]
r2 send [30,31]   r3 send [40,41]

每个 rank 的 recv:
[10,11 | 20,21 | 30,31 | 40,41]
   r0      r1      r2      r3
```

结果布局由 user rank 决定，而不是由内部 ring 的物理顺序决定。即使实际通信先经过 r3 再到 r1，最终 API 的排列也不能改变。

发送缓冲只需两个元素；接收缓冲需要八个元素。把 `sendcount=8` 传进去不会“告诉 NCCL 总长度”，而是让它以为每个 rank 要贡献八个元素，从而要求总输出 32 个元素。

## 5. ReduceScatter：先对应位置规约，再按 rank 分片

每个 rank 发送八个元素，`recvcount=2`：

```text
r0: [ 0, 1 |  2, 3 |  4, 5 |  6, 7]
r1: [10,11 | 12,13 | 14,15 | 16,17]
r2: [20,21 | 22,23 | 24,25 | 26,27]
r3: [30,31 | 32,33 | 34,35 | 36,37]

概念上的完整 Sum:
    [60,64 | 68,72 | 76,80 | 84,88]

实际输出:
r0 [60,64]  r1 [68,72]  r2 [76,80]  r3 [84,88]
```

“先求完整和，再切片”用于定义语义，**不要求实现先在某张 GPU 上生成完整结果**。Ring 等算法会在传输中逐步规约，直接形成分片。

### 与 AllReduce 的关系

在长度可按 P 等分、相同规约等条件下：

```text
AllReduce 的结果 = ReduceScatter 的结果再做 AllGather
```

这是理解 Ring AllReduce 与梯度分片的关键。并不意味着一次 `ncclAllReduce` 必须在 host 上调用两次 API，也不意味着任意短数组都要人为补齐再调用。

## 6. AlltoAll：发送按目的 rank，接收按来源 rank

`P=4, count=1`，令 `x_ij` 表示 r_i 要发给 r_j 的数据：

```text
发送矩阵（每行在一张卡）       接收矩阵
r0 [x00 x01 x02 x03]          r0 [x00 x10 x20 x30]
r1 [x10 x11 x12 x13]          r1 [x01 x11 x21 x31]
r2 [x20 x21 x22 x23]          r2 [x02 x12 x22 x32]
r3 [x30 x31 x32 x33]          r3 [x03 x13 x23 x33]
```

可以把它理解成“以 rank 为维度的矩阵转置”。每个块可以含 C 个元素，块内部顺序不变。

当前版本提供原生 `ncclAlltoAll`。对于不等长的 all-to-all，需设计匹配的 Send/Recv 或使用上层封装；**nccl-tests 有 `alltoallv_perf` 不代表当前公共头文件有同名 `ncclAlltoAllv` API**。

不要把原地 AllReduce 的经验直接搬到 AlltoAll。本版本头文件没有为该操作承诺上述通用原地形式，测试实现也明确提示不支持 in-place；学习时使用独立 send/recv 缓冲。

## 7. Gather 与 Scatter

Gather 把 AllGather 的结果只放在 root：各 rank 发送 C 个元素，root 接收 P×C。

Scatter 则将 root 的 P×C 个元素按目的 rank 分发，每个 rank 接收 C 个。它不做规约，不能与 ReduceScatter 混淆。

这两个 API 在当前头文件中存在。读旧教程时应先核对版本，不能把历史局限当成当前能力。

## 8. In-place 不是“所有指针都一样”

以下以 `T*` 指针运算表示元素偏移。若用 `char*`，需乘 `sizeof(T)`：

| 操作 | 合法原地关系 |
|---|---|
| AllReduce | `sendbuff == recvbuff` |
| Reduce | root 上 `sendbuff == recvbuff` |
| Broadcast | `sendbuff == recvbuff` 的原地形式 |
| AllGather | `sendbuff == recvbuff + r*C` |
| ReduceScatter | `recvbuff == sendbuff + r*C` |
| Gather | root 上 `sendbuff == recvbuff + root*C` |
| Scatter | root 上 `recvbuff == sendbuff + root*C` |

AllGather 的直观解释：每个 rank 先把自己的局部数据放进最终大数组里属于自己的槽，再填满其他槽。

ReduceScatter 的直观解释：最终只保留完整输入里属于自己的那一段，但这段的值变成全局规约结果。

**部分重叠但不符合约定**的缓冲不是合法原地优化。比如 AllReduce 中令 recv 指向 send 的中间位置，可能覆盖还没发送的数据。

## 9. Send/Recv：配对合同与前进条件

每个 Send 要有匹配的 Recv，peer、类型、count 必须一致。NCCL P2P API 没有 MPI 那样的任意 tag 供你区分消息，因此同一对 rank 上的调用顺序也重要。

循环交换示意：

```cpp
ncclGroupStart();
ncclSend(sendbuf, count, ncclFloat, (rank + 1) % nranks, comm, stream);
ncclRecv(recvbuf, count, ncclFloat, (rank + nranks - 1) % nranks, comm, stream);
ncclGroupEnd();
```

这是解释结构的片段，实际程序必须检查返回值。对于需要同时前进才可完成的多个 Send/Recv，用 group 让 NCCL 一起规划，不能先发送后在 GPU 上等待完成、再提交本应解除等待的接收。

不是所有 communicator rank 都必须参与每个 P2P 交换，但参与配对的 rank 必须匹配。集合操作则要求整个相关 clique 的参与和语义一致。

## 10. 检查清单

提交前逐项问：

1. 每个 rank 的 collective 顺序是否一致？
2. root/peer 是该 communicator 的 rank 吗？
3. count 是元素数，且该 API 的局部/整体含义正确吗？
4. send/recv 容量是否足够，指针是否在正确设备并满足布局？
5. 输入已就绪，输出在完成前不会被覆盖或释放吗？
6. dtype 和规约 op 是否符合真实内存内容？

## 源码锚点

统一参看[公共头文件](../nccl/src/nccl.h.in)：

- `nccl/src/nccl.h.in:550`，Reduce；`:580`，Broadcast；`:594`，AllReduce。
- `nccl/src/nccl.h.in:607`，ReduceScatter；`:625`，AllGather，包含原地偏移规则。
- `nccl/src/nccl.h.in:640`，AlltoAll；`:653`，Gather；`:668`，Scatter。
- `nccl/src/nccl.h.in:727`，Send/Recv 的配对与 group 前进要求。
- [AlltoAll 测试](../nccl-tests/src/alltoall.cu)：`nccl-tests/src/alltoall.cu:43`，in-place 限制。

## 自测与答案

1. 8 卡 AllGather，每卡 256 个 FP32，输出要多大？**每 rank 2048 个元素，即 8192 字节；API 的 sendcount 仍为 256。**
2. Scatter 后每 rank 的数据已是全局求和结果吗？**不是；Scatter 只分发，不做 reduction。**
3. AllGather 的内部 Ring 顺序变了，输出是否也跟着重排？**不会，输出仍按 user rank 排列。**

下一章：[CUDA 与 API 正确性](03-cuda-semantics.md)。
