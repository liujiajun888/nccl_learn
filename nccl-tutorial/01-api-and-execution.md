# 01 API 与执行约定：应用怎样把工作交给 NCCL

一次 NCCL 调用同时描述两件事：参与者共同计算什么结果，以及本 rank 的 GPU 在什么依赖之后执行。前者由 collective 参数决定，后者接入 CUDA stream。

## 1. 先读懂一次调用

贯穿案例中，每个 rank 持有 64 MiB FP32 输入。元素数为 `64 * 1024 * 1024 / sizeof(float)`，即 `16 * 1024 * 1024`。

下面是调用示意。已选好设备，普通 blocking communicator 已初始化，buffer 已分配，输入准备排在同一 stream 上。`NCCL_CHECK`、`CUDA_CHECK` 沿用[完整示例](examples/single_process_allreduce.cu)中的错误检查宏。

```cpp
constexpr size_t count = 16 * 1024 * 1024;
NCCL_CHECK(ncclAllReduce(send, recv, count, ncclFloat, ncclSum,
                        comm, stream));
```

| 参数 | 本次工作中的含义 |
|---|---|
| `send` / `recv` | 本 GPU 的输入和输出地址，各容纳 count 个元素 |
| `count` / `ncclFloat` | 元素数量和类型，共同决定访问字节数 |
| `ncclSum` | 对所有 rank 的对应元素求和 |
| `comm` | 确定参与者以及本 rank 的通信资源 |
| `stream` | 确定本地执行依赖和后续完成顺序 |

八个 rank 以相同的 collective 顺序提交，数量、类型和规约操作匹配。调用后，每个 rank 的输出都是八份输入的逐元素和。

NCCL 的 `count` 以元素为单位。地址范围则由应用根据类型和 count 分配；这也是以后新增 API 或调整任务描述时，首先要保持一致的一组字段。

## 2. 三个操作把核心语义讲清

用两 rank 各四个数，比较最常见的三种操作：

```text
r0 输入 [1,2,3,4]       r1 输入 [10,20,30,40]

AllReduce(Sum, count=4)
  r0、r1 各得到 [11,22,33,44]

AllGather(sendcount=4)
  r0、r1 各得到 [1,2,3,4 | 10,20,30,40]

ReduceScatter(Sum, recvcount=2)
  r0 得到 [11,22]       r1 得到 [33,44]
```

AllReduce 与 ReduceScatter 都做规约，区别是结果完整复制还是按 rank 分片。AllGather 收集各 rank 原有的数据，按来源 rank 排列。

由此得到三种容量关系。设 rank 数为 `P`，接口 count 为 `C`：

| 操作 | 每 rank 输入 → 输出 |
|---|---|
| AllReduce | `C → C` |
| AllGather | `C → P*C` |
| ReduceScatter | `P*C → C` |

可等分的数据先做 ReduceScatter，再做 AllGather，可以得到 AllReduce 的数学结果。第 04 章会说明 Ring 怎样把这两个阶段组织在同一个设备工作体中。

其他集合接口与原地布局放在[接口附录](appendix/collectives.md)，读具体 API 时查即可。

## 3. 本地 stream 顺序与跨 rank 顺序

同一 stream 内，生产输入、通信、使用输出按顺序执行：

```text
本地 stream：输入就绪 → NCCL 工作 → 使用输出
                              ↕
                   其他 rank 的匹配通信工作
```

应用把后续 kernel 排在通信之后，它就能使用本地结果。CPU 要检查结果时，先等对应 stream/event 完成，再把 GPU 数据读回。

跨 stream 时，用 event 把两段队列连起来。以下仍是调用示意，`produce`、`consume` 表示应用 kernel，events 和两条 stream 已在本 GPU 创建：

```cpp
produce<<<grid, block, 0, computeStream>>>(send);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaEventRecord(ready, computeStream));
CUDA_CHECK(cudaStreamWaitEvent(commStream, ready, 0));
NCCL_CHECK(ncclAllReduce(send, recv, count, ncclFloat, ncclSum,
                        comm, commStream));
CUDA_CHECK(cudaEventRecord(done, commStream));
CUDA_CHECK(cudaStreamWaitEvent(computeStream, done, 0));
consume<<<grid, block, 0, computeStream>>>(recv);
CUDA_CHECK(cudaGetLastError());
```

`ready` 把输入生产接到通信之前，`done` 把结果消费接到通信之后。这里没有外层 group；使用显式 group 时，把 `done` 的记录放在成功结束 group 之后。

跨 rank 的调用顺序同样是依赖的一部分。例如各 rank 都先提交 A，再提交 B，NCCL 才能把两轮工作正确对应。CPU 抵达时间可以不同，程序定义的通信次序保持一致。

## 4. Group：先收齐调用，再一起组织提交

一个 CPU 线程管理多 GPU 时，它需要为每个 rank 都发起调用。`ncclGroupStart/End` 让 NCCL 先收集这些请求，再统一准备和提交。

[单进程示例](examples/single_process_allreduce.cu)的核心代码是：

```cpp
NCCL_CHECK(ncclGroupStart());
for (int r = 0; r < ndev; ++r) {
  CUDA_CHECK(cudaSetDevice(devices[r]));
  NCCL_CHECK(ncclAllReduce(send[r], recv[r], count, ncclFloat, ncclSum,
                          comms[r], streams[r]));
}
NCCL_CHECK(ncclGroupEnd());
```

这解决了提交上的循环依赖：第一个 rank 即使需要其他 rank 配合，CPU 仍能把其他请求交进来。外层 GroupEnd 再组织连接、计划和发射。

Group 也用于需要共同推进的 Send/Recv。它是主机侧组织边界；GPU 上最后何时完成，继续由 stream/event 表达。

第 02 章会看到，一个 group 可以形成多个 kernel plan；是否合并取决于任务兼容性与计划容量。

## 5. 提交状态与数据完成各看一个接口

普通 blocking communicator 的 collective 成功返回，表示调用已按该模式完成提交；显式 group 内的操作在外层 GroupEnd 处理。GPU 随后沿 stream 执行。

运行时有两类状态值得关注：

| 想知道什么 | 查看什么 |
|---|---|
| NCCL 是否报告异步错误或主机工作仍在进行 | `ncclCommGetAsyncError` 的返回值及输出状态 |
| 本 GPU 的已提交工作是否完成 | `cudaStreamQuery` 或相应 event |

完整示例把这两项放在一个等待循环里。只有各 GPU 完成且没有错误，才读回并校验输出。

配置 `blocking=0` 时，部分 NCCL 主机工作会以 `ncclInProgress` 返回。应用先查询相关 communicator 的主机状态均为 `ncclSuccess`，再记录完成 event、提交依赖通信结果的 CUDA 工作或继续使用 comm。这时提交顺序已经建立，数据消费继续遵循 CUDA 依赖。

用户输入在通信完成前保持可读，输出在通信完成后交给消费者；输出的下一次复用还需排在消费者之后。内部内存和 communicator 的回收在第 09 章统一展开。

## 6. 从公开约定进入实现

- [nccl/src/nccl.h.in:540](../nccl/src/nccl.h.in#L540)：collective 的参与、提交和 group 约定。
- [nccl/src/nccl.h.in:594](../nccl/src/nccl.h.in#L594)：AllReduce 的结果与原地关系。
- [nccl/src/nccl.h.in:607](../nccl/src/nccl.h.in#L607)：ReduceScatter 与 AllGather 的 count 和布局。
- [nccl/src/nccl.h.in:727](../nccl/src/nccl.h.in#L727)：Send/Recv 的配对与共同推进。
- [examples/single_process_allreduce.cu:75](examples/single_process_allreduce.cu#L75)：group 提交；第 83 行开始查询状态，第 107 行开始结果校验。

### 阅读任务：画出示例中的三个边界

在单进程示例里标出 GroupEnd、等待循环退出、D2H 完成这三个位置，给每个位置写一句说明。

核对结果应是：**各 rank 的请求已提交；本地 GPU 通信工作完成；CPU 已获得可校验的结果。** 再看输入 vector 的作用域，确认它覆盖了 H2D 使用期。

有 GPU 时运行两卡示例，检查全部元素的 `wrong=0`。想看提交与执行时间的分离，用第 11 章的 CUDA timeline 观察，保留现有依赖关系。

下一章：[05 Communicator](05-communicator.md)。
