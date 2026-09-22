# 03 CUDA 与 API 正确性：三种“完成”不能混为一谈

## 学习目标

- 分清 CPU 调用返回、NCCL 主机侧提交完成、GPU 工作完成。
- 正确组织多 GPU group 和跨 stream 依赖。
- 理解非阻塞 communicator 的状态检查及缓冲生命周期。

## 1. CPU 返回成功，不代表输出已经可读

普通默认 blocking communicator 下，collective 的成功返回通常表示工作已经排入 stream；这里 blocking 不表示“CPU 等到 GPU 通信结束”。

```text
CPU: 提交输入生成 -> ncclAllReduce返回 -> 提交消费kernel -> 做其他事
GPU:        生成输入 ---------> 通信 ---------> 消费结果
```

同一 stream 按序执行，所以 GPU 上的消费 kernel 可以直接排在 NCCL 后面，不需要 CPU 先 `cudaDeviceSynchronize()`。

但 CPU 若立刻读回或释放缓冲，则必须有正确的完成保证。最简单的学习方式是在该 stream 上同步后，再读回结果。

### 三个状态

| 状态 | 能说明什么 | 不能说明什么 |
|---|---|---|
| API 参数被接受 / group 内调用返回 | 当前调用未立即失败 | group 的操作已启动 |
| 默认 communicator 的外层 GroupEnd 成功，或非阻塞状态回到 success | 相关主机提交工作完成 | GPU 输出已经完成 |
| CUDA stream/event 显示完成，且无相关 NCCL 异步错误 | 已提交的相关 GPU 工作完成 | 未来网络/后续 collective 永远不会失败 |

## 2. 单进程多 GPU 为什么需要 group

假设一个 CPU 线程依次控制两张 GPU。第一张卡的调用可能需要第二个 rank 的参与，如果等第一张完整推进后才提交第二张，就有前进性问题。

正确结构：

```cpp
ncclGroupStart();
for (int r = 0; r < ndev; ++r) {
  cudaSetDevice(devices[r]);
  ncclAllReduce(send[r], recv[r], count, ncclFloat, ncclSum,
                comms[r], streams[r]);
}
ncclGroupEnd();
```

这是省略错误检查的结构示意；[完整示例](examples/single_process_allreduce.cu)检查每个调用。

GroupEnd 是一个“现在这些工作已经齐了，可以一起规划/提交”的边界，**不是 GPU barrier，也不保证只产生一个 kernel**。一次 group 可能拆出多个 plan 或 kernel。

### 错误结构

```text
GroupStart
  提交 rank0
  cudaStreamSynchronize(rank0_stream)  <- 此时group内操作还可能未提交
  提交 rank1
GroupEnd
```

这种同步不仅不能正确等待刚才的 NCCL，若和其他依赖形成循环还会挂起。依赖 group 内通信的 CUDA 操作应放在有效的 GroupEnd 完成之后。

初始化的 `ncclCommInitRank` 可按 group 组织，但不要把 communicator 初始化和 collective 混在同一个 group 中。

## 3. 跨 stream 必须把依赖画出来

想让计算与通信重叠，通常需要 compute stream 和 comm stream。正确依赖是：

```text
compute stream: produce ---- record(ready) -------- unrelated compute ---- wait(done) -> consume
                                  |                                  ^
comm stream:                  wait(ready) -> AllReduce -> record(done)
```

CUDA 片段：

```cpp
produce<<<grid, block, 0, compute_stream>>>(input);
cudaEventRecord(ready, compute_stream);
cudaStreamWaitEvent(comm_stream, ready, 0);
ncclAllReduce(input, output, count, ncclFloat, ncclSum, comm, comm_stream);
cudaEventRecord(done, comm_stream);
cudaStreamWaitEvent(compute_stream, done, 0);
consume<<<grid, block, 0, compute_stream>>>(output);
```

这是依赖示意，不是独立示例：`produce/consume` 未定义，实际使用需要检查错误。上述片段假设普通 blocking communicator 且不在未结束的 group 内；非阻塞模式先确认主机提交完成，再添加依赖通信完成的 CUDA 操作。

事件创建、记录、等待要在正确设备上下文中，且存活至所有使用结束。不要让默认 stream 的隐式同步“碰巧掩盖”缺失依赖，换成 nonblocking stream 后问题就暴露。

### 能并发不等于真的重叠

即使事件图正确，通信与计算可能都争用 SM、HBM 或互连，导致：

```text
计算独跑 10 ms，通信独跑 4 ms，并发后总耗时仍可能 > 10 ms
```

是否有利要看应用关键路径和时间线，不能只看用了两个 stream。见[第 16 章](16-training-integration.md)。

### 一个 group 中混合多个 stream

NCCL 需要协调该 group 涉及的 stream 依赖，可能带来跨 stream 的同步约束。不要把“group 减少提交开销”理解成“group 内每条 stream 都保持完全独立”。先保持每个设备的通信在一条 stream 上，再增加有证据支持的并发。

## 4. 非阻塞 communicator：另外一层异步

配置结构必须由宏初始化：

```cpp
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;
```

`ncclCommInitRankConfig`、非阻塞 group 提交等可能返回 `ncclInProgress`。这不意味着失败，也不意味着已经可随意使用该 communicator 的下一步操作。

概念流程：

```text
发起操作
  -> success: 主机侧该操作完成
  -> inProgress: 轮询 ncclCommGetAsyncError
       -> inProgress: 尚在进行
       -> success: 该主机侧工作完成
       -> 其他错误: 停止正常路径并协调异常处理
  -> 其他错误: 处理立即失败
```

必须检查两个返回值：

```cpp
ncclResult_t state;
ncclResult_t query_status = ncclCommGetAsyncError(comm, &state);
```

`query_status` 是“查询是否成功”；`state` 是“communicator 的异步状态”。只判断其中一个会漏错。

对于涉及多个非阻塞 communicator 的 group，要检查相关 communicator，不能只看其中一个成功就宣布整个 group 已可继续。非阻塞 host 状态的成功仍不是 GPU 完成通知：需要通信结果时另查 CUDA stream/event。

## 5. 等待时也要允许错误被发现

学习示例往往用 `cudaStreamSynchronize()`，容易理解。但如果某个 peer 永久不参与，单纯等待 GPU 完成可能一直等下去。

更可诊断的模式：

```text
循环直到所有相关 stream 完成:
  查询 NCCL 异步状态并检查查询返回值
  查询 cudaStreamQuery:
    cudaSuccess -> 该stream完成
    cudaErrorNotReady -> 继续
    其他值 -> CUDA错误
  若超过应用设置的deadline -> 报告并进入协调退出/恢复
```

[示例代码](examples/README.md)在已提交后的等待中采用这个模式。**它不为 blocking 初始化或 GroupEnd 内部自动增加可中断超时**；这些阶段仍需外部作业超时或更完整的非阻塞控制。

超时本身不证明“网络坏了”，也可能是 rank 调用顺序不同、某卡之前的 kernel 卡住、进程退出或死锁。处理策略见[第 12 章](12-debugging.md)。

## 6. 缓冲区何时可以修改、复用或释放

`ncclAllReduce` 返回后，send 和 recv 仍可能被设备访问。原则是：

- 输入在通信实际读取前准备好，读取完成前不被其他工作覆盖。
- 输出在通信完成前不被其他工作消费或覆盖。
- buffer、stream、communicator、注册 handle 和 Graph 相关对象的生命周期覆盖其所有未完成使用。
- 只在明确支持的布局使用 in-place；它不是任意指针别名的许可证。

将两个 bucket 放在不重叠的缓冲区，是实现安全流水的重要条件。若复用一个缓冲区，必须让上一轮最后一个使用者完成后才写下一轮数据。

## 7. Collective 顺序是分布式的合同

正确：

```text
rank0: AllReduce(A) -> AllGather(B)
rank1: AllReduce(A) -> AllGather(B)
```

错误：

```text
rank0: AllReduce(A) -> AllGather(B)
rank1: AllGather(B) -> AllReduce(A)
```

每个 rank 的本地 stream 都“有序”，并不能修复全局合同不一致。相同 API 但 count/type/root/op 不匹配也不合法；可能报错，也可能表现为挂起或错误数据，不承诺所有错误都立即被诊断。

多 communicator 场景更复杂：各 GPU、线程与通信器间的 launch 顺序可能产生循环依赖。当前版本有相关顺序控制能力，但入门应采用所有 rank 一致的显式提交顺序，不依赖隐式行为解决任意死锁。

对同一个 communicator 不要从多个线程无协调地并发提交。group 的线程局部状态也不是让多个线程共同组成一对 Start/End 的锁。

## 8. 正常结束与异常结束

正常路径：完成所有使用 → 按 API 生命周期 finalize/destroy → 释放其余资源。完成语义及局部/全局资源回收见[第 04 章](04-communicator.md)。

异常路径：停止继续提交 → 记录各 rank 状态 → 协调 abort/退出或重建。`ncclCommAbort` 不能补齐已经丢失的梯度，也不会自动重启远程进程；应用或作业系统要承担恢复策略。

本教程最小示例采用 fail-fast，而不是生产级容错状态机。不要把错误退出代码当成可直接复制到在线服务的完整恢复方案。

## 源码锚点

- [公共 API](../nccl/src/nccl.h.in)：`nccl/src/nccl.h.in:135`，`NCCL_CONFIG_INITIALIZER`；`:281`，带配置初始化。
- 同文件 `:391`，`ncclCommGetAsyncError`；`:537`，collective 的排入 stream 语义。
- 同文件 `:844`，group 语义；`:877`，依赖操作应位于 GroupEnd 之后。
- [非阻塞初始化返回处理](../nccl/src/init.cc)：`nccl/src/init.cc:3049`，`ncclCommInitRankConfig`。

## 自测与答案

1. `blocking=1` 是否表示 AllReduce 返回时 GPU 完成？**不是，它与非阻塞主机提交模式对应，不替代 CUDA 完成同步。**
2. `ncclCommGetAsyncError` 返回 success，是否说明输出可读？**查询成功不等于 state 成功；即使 state 成功，也仍需适当 CUDA 完成保证。**
3. 先把 AllReduce 放进 group，再立即 record event，最后 GroupEnd，event 能代表该 AllReduce 完成吗？**不能，这个依赖操作顺序不正确。应在正确的 GroupEnd/非阻塞提交完成边界后记录。**

下一章：[Communicator 生命周期](04-communicator.md)。
