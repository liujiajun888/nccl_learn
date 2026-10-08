# 附录：集合接口与内存布局

用到某个 API 时，先确认 count 的含义、输入输出容量和结果归属。以下 `P` 表示 rank 数，`r` 表示当前 rank，`C` 表示该接口的 count；容量均以元素计，换成字节再乘类型大小。

## 1. 集合接口速查

| 操作 | count 含义 | 每 rank 输入容量 | 每 rank 输出容量 | 结果 |
|---|---|---:|---:|---|
| AllReduce | 完整长度 | C | C | 每个 rank 一份完整规约结果 |
| Reduce | 完整长度 | C | root 为 C | 仅 root 得到规约结果 |
| Broadcast | 广播长度 | root 为 C | C | 每个 rank 得到 root 的输入 |
| AllGather | 本 rank 发送长度 | C | P*C | 按来源 rank 拼接 |
| ReduceScatter | 本 rank 接收长度 | P*C | C | rank r 得到第 r 块规约结果 |
| AlltoAll | 发往每个 peer 的长度 | P*C | P*C | 接收块按来源 rank 排列 |
| Gather | 每 rank 贡献长度 | C | root 为 P*C | root 按来源 rank 拼接 |
| Scatter | 每 rank 接收长度 | root 为 P*C | C | rank r 得到 root 的第 r 块 |

`root` 和 `peer` 都是当前 communicator 内的 rank。Reduce 的非 root 接收指针可为 NULL；Gather 的非 root 接收指针、Scatter 的非 root 发送指针不参与数据访问。具体指针约定查对应头文件注释。

### 两个例子

四卡 ReduceScatter，`recvcount=1024`、FP32：每卡输入容量 `4*1024*4=16384` 字节，输出 `1024*4=4096` 字节。

四卡 AlltoAll，`count=1`，`xij` 表示 rank i 发给 rank j 的值：

```text
发送                         接收
r0 [x00 x01 x02 x03]          r0 [x00 x10 x20 x30]
r1 [x10 x11 x12 x13]          r1 [x01 x11 x21 x31]
r2 [x20 x21 x22 x23]          r2 [x02 x12 x22 x32]
r3 [x30 x31 x32 x33]          r3 [x03 x13 x23 x33]
```

每个块可以包含多个元素，块内次序保持不变。不等长交换可由匹配的 Send/Recv 组织；本基线公共头文件提供 `ncclAlltoAll`，tests 的 `alltoallv_perf` 则是测试程序名称。

## 2. 原地复用按 rank 计算位置

原地（in-place）复用同一分配中的输入和输出。以下偏移按 `T*` 的元素运算解释：

| 操作 | 原地关系 |
|---|---|
| AllReduce | `send == recv` |
| Reduce | root 上 `send == recv` |
| Broadcast | `send == recv` |
| AllGather | `send == recv + r*C` |
| ReduceScatter | `recv == send + r*C` |
| Gather | root 上 `send == recv + root*C` |
| Scatter | root 上 `recv == send + root*C` |

以两卡、`C=2` 的 AllGather 为例，每卡分配四个 float，输入预先位于自己的接收槽：

```text
下标        0  1 |  2  3
r0 调用前  [1, 2 |  ?, ?]    send=recv
r1 调用前  [?, ? | 10,20]    send=recv+2
调用后均为 [1, 2 | 10,20]
```

ReduceScatter 方向相反：完整输入在 `send`，rank 1 的输出写入 `send+2`，完成后读取这一片。

这些关系定义了支持的重叠方式。其他部分重叠使用独立缓冲处理；AlltoAll 在本基线测试中使用独立 send/recv，不提供通用原地形式。

## 3. 规约与浮点校验

常用操作包括 Sum、Prod、Min、Max、Avg。类型与操作的组合通过对应 API 和实现路径支持。

规约顺序影响浮点舍入。例如 FP32 中，先计算 `1e20 + (-1e20)` 再加 `3.14`，与先计算 `(-1e20) + 3.14` 再加 `1e20`，可能得到不同结果。一般浮点测试采用合适的绝对/相对误差，并验证所有输出元素。

`ncclRedOpCreatePreMulSum` 提供“先乘标量再求和”的规约形式，其接口按标量、类型和驻留位置描述操作。扩展入口见[现代特性附录](appendix-modern-features.md)。

## 4. Send/Recv：匹配一对参与者

发送端指向接收 rank，接收端指定发送 rank，两端类型、count 和对应消息次序一致。API 通过 peer 与调用顺序配对消息。

下面是循环交换的调用示意：每 rank 使用独立 send/recv buffer，容量均为 count，输入已就绪，comm/stream 已初始化；错误检查宏来自[完整示例](examples/single_process_allreduce.cu)。

```cpp
NCCL_CHECK(ncclGroupStart());
NCCL_CHECK(ncclSend(send, count, ncclFloat, (rank + 1) % nranks,
                   comm, stream));
NCCL_CHECK(ncclRecv(recv, count, ncclFloat, (rank + nranks - 1) % nranks,
                   comm, stream));
NCCL_CHECK(ncclGroupEnd());
```

Group 把需要共同推进的收发交给 NCCL 一起组织。结果消费继续遵循[第 02 章](02-api-and-execution.md)的 stream 完成顺序。

## 源码入口

- [nccl/src/nccl.h.in:550](../nccl/src/nccl.h.in#L550)：Reduce、Broadcast、AllReduce。
- [nccl/src/nccl.h.in:607](../nccl/src/nccl.h.in#L607)：ReduceScatter、AllGather 的容量与偏移。
- [nccl/src/nccl.h.in:640](../nccl/src/nccl.h.in#L640)：AlltoAll、Gather、Scatter。
- [nccl/src/nccl.h.in:727](../nccl/src/nccl.h.in#L727)：Send/Recv 匹配。
- [nccl-tests/src/alltoall.cu:43](../nccl-tests/src/alltoall.cu#L43)：测试中的 in-place 限制。

回到[教程首页](README.md)。
