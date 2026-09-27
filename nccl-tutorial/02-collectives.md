# 02 集合通信：先算对结果，再谈性能

## 学习目标

- 用同一组两卡输入，区分求和复制、拼接、求和分片。
- 从输入输出长度推导 `count`，不把元素数当字节数。
- 画对原地（in-place）指针位置，再用四 rank 算例加深理解。

**CUDA/UMD 读者主线：**两卡数据流用于校准集合语义，不是复习加法；重点是第 1 节的 count/结果归属、第 5 节的 RS+AG 分解、第 8 节的原地契约，以及第 9 节 Send/Recv 的匹配与共同前进。指针运算与逐句调用解释可以快读，但不要因为熟悉显存分配就跳过 NCCL 特有的布局约定；其余有根操作按需查表。

## 起步：同样的两份输入，三种不同需求

沿用上一章：两位 rank 各持有四个元素。下面三个操作是**各自从原始输入开始的独立例子**，不是把前一个操作的结果作为后一个的输入：

```text
r0 输入 [1,2,3,4]       r1 输入 [10,20,30,40]
          │                       │
          └───── 同一通信小组 ─────┘

AllReduce(Sum)          AllGather                     ReduceScatter(Sum)
对应位置求和，每卡一份    不求和，按 rank 拼接           对应位置求和，每卡一片
r0 [11,22,33,44]        r0 [1,2,3,4 | 10,20,30,40]    r0 [11,22]
r1 [11,22,33,44]        r1 [1,2,3,4 | 10,20,30,40]    r1 [33,44]
每卡输入4 → 输出4        每卡输入4 → 输出8              每卡输入4 → 输出2
```

- **AllReduce：**两张卡都要完整和，所以各得到 `[11,22,33,44]`。
- **AllGather：**两张卡都要保留双方的原始数据，所以各得到八个元素，rank 0 的块在前、rank 1 的块在后；`1` 和 `10` 不相加。
- **ReduceScatter：**两张卡各只需要一半的和，所以 rank 0 得到前两项，rank 1 得到后两项。这是分片，不是漏掉一半结果；每个输出仍包含双方的贡献，如 `3+30=33`。

这里“先求完整和再切片”只是定义结果，不要求某张 GPU 真的先存下完整和。**是否规约、结果是否完整留给每个 rank**，决定了三者的区别。

## 从数组长度推到 count 和 API

现在才引入符号：`P` 是 rank 数，`r` 是当前 rank，`C` 是该接口的 count 值，`sizeof(T)` 是一个元素的字节数。数组均按元素展示。由上图逐项数出：

| 操作 | count 应填什么 | 每 rank 输入 → 输出 | float 缓冲字节数：输入 → 输出 |
|---|---|---|---|
| AllReduce | `count=4`，完整输入长度 | `C → C`，即 `4 → 4` | `16 → 16` |
| AllGather | `sendcount=4`，本 rank 贡献长度 | `C → P×C`，即 `4 → 8` | `16 → 32` |
| ReduceScatter | `recvcount=2`，本 rank 接收长度 | `P×C → C`，即 `4 → 2` | `16 → 8` |

**count 的单位是元素，不是字节。**`ncclFloat` 表示每元素四字节，所以容量计算才乘 `sizeof(float)=4`。例如 AllGather 填 4 已经表示发送 16 字节；误填 16 会要求每卡发送 16 个 float、接收 32 个 float，超出上例容量。ReduceScatter 则由 `2 ranks × recvcount 2 = 4` 推回完整输入长度；误填 4 会要求每卡有八元素输入。

**本章所有 C++ 片段均为调用示意，不是可独立运行的程序。**假定每个 rank 由独立进程或线程提交，已选好本 GPU、成功初始化普通阻塞模式的 comm 并创建 stream；GPU buffer 已按标注容量分配，输入初始化在同 stream 上排在调用之前。`NCCL_CHECK` / `CUDA_CHECK` 是应用已定义的错误检查宏，分别检查 NCCL / CUDA 返回值，失败就报错并停止正常执行，不继续使用结果；多进程故障还需应用协调退出。宏的写法可对照[完整示例](examples/single_process_allreduce.cu)。单线程管理多卡还需 group，细节留到下一章。

对上述两卡例子，每个 rank 都执行以下相同顺序的调用；`input` 是它自己的四元素原始输入，三个输出 buffer 相互独立、也不与输入重叠：

```cpp
NCCL_CHECK(ncclAllReduce(input, sumOut, 4, ncclFloat, ncclSum,
                        comm, stream));  // sumOut: 4 个 float
NCCL_CHECK(ncclAllGather(input, gatheredOut, 4, ncclFloat,
                       comm, stream));  // gatheredOut: 8 个 float
NCCL_CHECK(ncclReduceScatter(input, shardOut, 2, ncclFloat, ncclSum,
                           comm, stream));  // shardOut: 2 个 float
CUDA_CHECK(cudaStreamSynchronize(stream));  // 等待完成后才检查/回收这些结果
```

最后的同步只用于这里明确完成边界；若 CPU 要查看数值，还需将 GPU 数据复制到主机，不能直接解引用 GPU 指针。

## 1. 一张表掌握接口契约

**后查表：不必现在背八类 API。**当前公共头文件原生提供下面八类接口，包括 `ncclAlltoAll`、`ncclGather`、`ncclScatter`。root 是指定的数据源或结果接收 rank；peer 是一次发送/接收的对端 rank。例如 `root=1` 指这个小组的 rank 1，不是设备编号。

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

**第二遍四 rank 手算，首遍可跳。**输入：

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

**选读：校验结果时再回来看。**实数加法满足结合律，有限精度浮点加法一般不满足。例如按 FP32 近似计算：

```text
(1e20 + -1e20) + 3.14 ≈ 3.14
1e20 + (-1e20 + 3.14) ≈ 0
```

算法、拓扑、分块和规约顺序变化可能改变舍入结果。校验浮点结果通常用合理的绝对/相对误差，不默认要求所有配置逐位一致。也不能把任意巨大误差都归咎于舍入；先排查计数、输入初始化和同步。

## 3. Broadcast：一份输入复制给所有人

**首遍可跳。**如果 root=1，r1 持有 `[9, 8]`：

```text
r1 send [9, 8] -> r0/r1/r2/r3 recv 都是 [9, 8]
```

使用 `ncclBroadcast` 的 send/recv 形式。旧 `ncclBcast` 是已弃用的原地接口，不建议新示例使用它。

它与 AllGather 的区别是：Broadcast 的来源只有一个 root；AllGather 中每个 rank 都贡献一个块。

## 4. AllGather：按 rank 顺序拼起来

**第二遍四 rank 手算，首遍可跳。**`P=4, sendcount=2`：

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

**第二遍四 rank 手算，首遍可跳。**每个 rank 发送八个元素，`recvcount=2`：

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

例如将开头 ReduceScatter 得到的 r0 `[11,22]`、r1 `[33,44]` 再做一次 `AllGather(sendcount=2)`，每卡就有 `[11,22,33,44]`。

在长度可按 P 等分、相同规约等条件下，数学语义上：

```text
AllReduce 的结果 = ReduceScatter 的结果再做 AllGather
```

这是理解 Ring AllReduce 与梯度分片的关键。并不意味着一次 `ncclAllReduce` 必须在 host 上调用两次 API，也不意味着任意短数组都要人为补齐再调用；若规约顺序不同，浮点结果仍可能有舍入差别。

## 6. AlltoAll：发送按目的 rank，接收按来源 rank

**首遍可跳。**`P=4, count=1`，令 `x_ij` 表示 r_i 要发给 r_j 的数据：

```text
发送矩阵（每行在一张卡）       接收矩阵
r0 [x00 x01 x02 x03]          r0 [x00 x10 x20 x30]
r1 [x10 x11 x12 x13]          r1 [x01 x11 x21 x31]
r2 [x20 x21 x22 x23]          r2 [x02 x12 x22 x32]
r3 [x30 x31 x32 x33]          r3 [x03 x13 x23 x33]
```

可以把它理解成“以 rank 为维度的矩阵转置”。每个块可以含 C 个元素，块内部顺序不变。

对于不等长的 all-to-all，需设计匹配的 Send/Recv 或使用上层封装；**nccl-tests 有 `alltoallv_perf` 不代表当前公共头文件有同名 `ncclAlltoAllv` API**。

不要把原地 AllReduce 的经验直接搬到 AlltoAll。当前头文件没有为它承诺通用原地形式，测试实现也明确提示不支持 in-place；学习时使用独立 send/recv 缓冲。

## 7. Gather 与 Scatter

**首遍可跳。**Gather 把 AllGather 的结果只放在 root：各 rank 发送 C 个元素，root 接收 P×C。Scatter 则将 root 的 P×C 个元素按目的 rank 分发，每个 rank 接收 C 个。例如 `P=2, C=2, root=1`：

```text
Gather:  r0 [1,2]、r1 [10,20] -> 仅 r1 recv [1,2 | 10,20]
Scatter: r1 send [1,2 | 10,20] -> r0 recv [1,2]，r1 recv [10,20]
```

Scatter 只分发、不做规约，输出中不会出现 `1+10=11`，不能与 ReduceScatter 混淆。

## 8. In-place 不是“所有指针都一样”

原地（in-place）指按 API 约定复用输入输出空间，不另分配完整输出。例如 AllReduce 可以把四元素输入原地改写成四元素的和，使用 `send == recv`。但这条规则不能照搬给所有操作。

### 先画地址：P=2、C=2 的两个独立例子

这里 `P=2`、`C=2`。为画清地址，**AllGather 改为每卡输入 2、输出 4**，不同于开头的输入 4、输出 8；ReduceScatter 仍是每卡输入 4、输出 2。代码中的 `rank` 是当前 comm 的 rank，沿用前面的初始化和错误检查前提。

**AllGather：输入预先放在输出数组中属于自己的槽。**每卡分配四个 float，调用前只需填好自己的两个：

```text
元素下标       0   1 |  2   3
r0 调用前:  [ 1,  2 |  ?,  ?]    recv 指向 0，send 指向 0
r1 调用前:  [ ?,  ? | 10, 20]    recv 指向 0，send 指向 2
两卡调用后: [ 1,  2 | 10, 20]    各自 recv 都包含完整结果
```

```cpp
constexpr size_t P = 2, C = 2;
float* recv = gatherBuffer;       // 已分配 P*C 个 float，按上图初始化自己的槽
float* send = recv + rank * C;
NCCL_CHECK(ncclAllGather(send, recv, C, ncclFloat, comm, stream));
```

rank 1 必须是 **`send = recv + 2`**。若简单写 `send == recv`，会指向上图未初始化的第 0、1 项，而不是 `[10,20]`；即使把输入挪到开头，也不符合 rank 1 的原地契约。rank 0 能让两指针相等，只因为它的偏移恰好是零。

**ReduceScatter：完整输入仍要保留足够容量，输出落在本 rank 的分片位置。**每卡分配并初始化四个 float：

```text
元素下标       0   1 |  2   3
r0 调用前:  [ 1,  2 |  3,  4]    send 指向 0，recv 指向 0
r1 调用前:  [10, 20 | 30, 40]    send 指向 0，recv 指向 2
完成后有效输出: r0 的下标 0、1 为 [11,22]；r1 的下标 2、3 为 [33,44]
```

```cpp
constexpr size_t P = 2, C = 2;
float* send = reduceBuffer;       // 已分配并初始化 P*C 个 float
float* recv = send + rank * C;
NCCL_CHECK(ncclReduceScatter(send, recv, C, ncclFloat, ncclSum, comm, stream));
```

rank 1 必须是 **`recv = send + 2`**。简单的 `send == recv` 会把输出地址放在第 0 块，而契约要求它落在第 1 块；这不是受支持的原地布局，不能据此假定会得到正确结果。完成后只把 `recv` 指向的两个元素当作结果，其余输入位置的内容不作保证。

两段代码各自独立；完成边界与前面的调用相同，不能在操作完成前读取或释放缓冲。`float*` 的 `+2` 移动两个元素，即八字节，不是两字节。若换成 `char*` 计算同一偏移，就要写 `+2*sizeof(float)`。

### 后查表：各接口的原地关系

以下指针关系都按 `T*` 的元素偏移解释，`C` 取对应 API 的 count：

| 操作 | 合法原地关系 |
|---|---|
| AllReduce | `sendbuff == recvbuff` |
| Reduce | root 上 `sendbuff == recvbuff` |
| Broadcast | `sendbuff == recvbuff` 的原地形式 |
| AllGather | `sendbuff == recvbuff + r*C` |
| ReduceScatter | `recvbuff == sendbuff + r*C` |
| Gather | root 上 `sendbuff == recvbuff + root*C` |
| Scatter | root 上 `recvbuff == sendbuff + root*C` |

**部分重叠但不符合约定**的缓冲不是合法原地优化。比如 AllReduce 中令 recv 指向 send 的中间位置，可能覆盖还没发送的数据。

## 9. Send/Recv：配对合同与前进条件

**首遍可跳。**Send/Recv 是点对点（P2P）操作：例如 r0 向 r1 发送两个 float，r1 就要从 r0 接收两个 float。每个 Send 要有匹配的 Recv，peer 要相互对应，类型、count 必须一致。NCCL P2P API 没有 MPI 那样的任意 tag（消息标签）供你区分消息，因此同一对 rank 上的调用顺序也重要。

循环交换示意：两 rank 时取 `nranks=2, count=2`，r0 发送 `[1,2]`，r1 发送 `[10,20]`，最终各接收对方的两个数。各 rank 的 sendbuf/recvbuf 是独立的两元素 GPU buffer，沿用前面已初始化、已定义错误检查宏的前提：

```cpp
NCCL_CHECK(ncclGroupStart());
NCCL_CHECK(ncclSend(sendbuf, count, ncclFloat, (rank + 1) % nranks, comm, stream));
NCCL_CHECK(ncclRecv(recvbuf, count, ncclFloat, (rank + nranks - 1) % nranks, comm, stream));
NCCL_CHECK(ncclGroupEnd());
```

对于需要同时前进才可完成的多个 Send/Recv，用 group 让 NCCL 一起规划，不能先发送后在 GPU 上等待完成、再提交本应解除等待的接收。GroupEnd 也不是 GPU 完成通知，结果仍遵循 stream 的完成边界。

不是所有 communicator rank 都必须参与每个 P2P 交换，但参与配对的 rank 必须匹配。集合操作则要求整个相关通信小组的参与和语义一致。

## 10. 检查清单

提交前逐项问：

1. 每个 rank 的 collective 顺序是否一致？
2. root/peer 是该 communicator 的 rank 吗？
3. count 是元素数，且该 API 的局部/整体含义正确吗？
4. send/recv 容量是否足够，指针是否在正确设备并满足布局？
5. 输入已就绪，输出在完成前不会被覆盖或释放吗？
6. dtype 和规约 op 是否符合真实内存内容？

## 源码锚点

**主线核查：画对数组以后，对照 count、原地布局与配对要求。**统一参看[公共头文件](../nccl/src/nccl.h.in)，有根操作按需查：

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
