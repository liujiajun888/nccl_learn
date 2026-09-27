# 03 CUDA 与 API 正确性：让数据按正确顺序被使用

## 学习目标与阅读顺序

**CUDA/UMD 读者主线：**快速检查 1.3 的完成语义，重点读第 2 节 group、第 4 节跨 rank 匹配、第 5 节异步错误及第 7 节多 stream 依赖。第 1 节的复制/分配基础和第 3 节的通用生命周期可作速查；第 8 节非阻塞 communicator 按管理需求深入，但要先分清它与 CUDA stream 属性不是同一个开关。

## 从本地依赖图到分布式前进条件

熟悉 UMD 提交的人通常会先问“依赖是否满足、命令是否已提交、完成通知在哪里”。NCCL 还多一个问题：**对端是否进入同一轮操作，并持续提供数据或返还可复用空间？**

```text
本 rank: produce -> ready -> NCCL work -----------------> done -> consume
                              ^                 |
                              |                 v
对端:                  匹配操作、发布数据 <-> 消费数据、返还credit
```

这里 `ready/done` 表示正确安排的本地 CUDA 依赖；数据就绪与 credit 是通信协议状态，不是通过相同 event 名自动关联。credit 表示当前协议缓冲的消费者返还的可用槽位额度；NET 发送侧可能由 send proxy 返还，并非都直接来自远端 GPU，细节在第 09 章。

- 本地 `ready` 满足，只说明输入先于本地通信可用，不证明远端已发起匹配 collective。
- 本地 kernel 已 launch，仍可能在等远端数据、CPU proxy 或缓冲 credit。launch 成功不能替代前进性分析。
- 在正确提交边界之后记录的完成 event 才能代表对应 stream 位置的完成。普通 AllReduce 无相关错误且本地通信正常完成后，即可按 CUDA 依赖消费本地输出，不必另等“远端应用消费确认”；复用本地输出仍须等自己的后续消费者结束。

因此源码与调试时要分别画**应用/CUDA 依赖图**和**NCCL 参与者之间的等待图**。第 2、4、5 节是这两张图的交界；第 07/09 章进一步解释是谁发布和等待每一个协议状态。

本章假设你已读过第 01/02 章，知道 rank、communicator，以及 AllReduce 对应位置求和的含义。下面统一从两卡输入 `[1,1,1,1]`、`[2,2,2,2]`，输出均为 `[3,3,3,3]` 出发。

## 1. 先看一张卡上的顺序：输入、通信、输出

<details>
<summary>基础速查：H2D/D2H 与同 stream 顺序；熟悉 CUDA 可直接读 1.3</summary>

### 1.1 CPU 数组与 GPU 数组是不同的存放位置

最小示例使用普通 CPU 数组和 `cudaMalloc` 分配的 GPU 数组：

```text
rank 0:
CPU input [1,1,1,1] --H2D--> GPU send [1,1,1,1]
                                      |
                               AllReduce（与rank 1一起）
                                      |
                           GPU recv 最终应为 [3,3,3,3]
                                      |
                      完成后 D2H -> CPU output [3,3,3,3]
```

在 GPU 写完前，不能把接收缓冲中的内容当作这一轮的结果。正确做法是等相关 GPU 工作结束，再完成读回：

```text
CPU input --H2D--> GPU send --AllReduce--> GPU recv --D2H--> CPU output
                 准备输入          写输出          读回并校验
```

H2D 是 Host to Device（CPU 内存到 GPU 内存），D2H 是反方向。对这个普通 `cudaMalloc` 示例，CPU 不能像读 CPU 数组那样直接解引用设备指针 `recv[0]`；必须通过正确的数据传输得到 CPU 可读结果。

### 1.2 Stream 保证这张卡上已提交工作的先后关系

想象 CPU 提交一张任务单，GPU 按依赖执行。`cudaStream_t` 是 CUDA stream 的句柄，表达设备任务顺序；它不是网络连接。

下面是**每 rank 一个进程/线程**的顺序片段，所有参与 rank 都要执行对应的通信。设备、buffer、stream、comm 和错误检查宏已经准备好；这不是独立可编译程序。

```cpp
CUDA_CHECK(cudaMemcpyAsync(send, input.data(), bytes,
                           cudaMemcpyHostToDevice, stream));
NCCL_CHECK(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
CUDA_CHECK(cudaStreamSynchronize(stream));
CUDA_CHECK(cudaMemcpy(output.data(), recv, bytes, cudaMemcpyDeviceToHost));
```

逐句理解：

1. `cudaMemcpyAsync` 把输入复制安排在 `stream` 上；`bytes` 是字节数。
2. `ncclAllReduce` 使用同一 stream，因此通信读取输入前，前面的复制必须完成；`count` 是元素数。
3. `cudaStreamSynchronize` 让 CPU 等待这条 stream 已提交的工作完成。
4. 同步 D2H copy 完成后，CPU 才遍历 `output` 检查是否全为 3。

这里先用 synchronize 讲清顺序；完整示例为了能检测异步错误和等待超时，采用第 5 节的轮询。普通 host 内存的 `cudaMemcpyAsync` 也不保证与 CPU/计算完全重叠，所以不要用本片段推导传输并发性能。

</details>

### 1.3 API 返回与 GPU 完成是两个时刻

```text
CPU:  提交复制 -> AllReduce返回 -> 等待stream完成 ----------> 读回/校验
GPU:       复制输入 --------> 通信/等待peer --------> 写完输出
```

默认 communicator 下，一次不在外层 group 中的 collective 成功返回，表示相关提交已经完成，并不是 GPU 已经算出结果。CPU 可以做不依赖输出的其他事，却不能立即复用仍在使用的缓冲。

如果下一步也是 GPU 计算，把消费输出的 kernel 放在**同一 stream**的 NCCL 后面即可，通常不用 CPU 先等；kernel 是在 GPU 上运行的函数。

**停一下：** AllReduce 返回了，CPU 能否马上认为 `recv` 全是 3？**不能；还缺相关 GPU 工作完成及正确读回的保证。**

## 2. 两张卡一起提交：为什么需要 group

上一节每个 rank 有自己的线程/进程。现在只有**一个 CPU 线程依次管理两张 GPU**：第一张卡的提交可能需要其他 rank 参与，如果卡在第一张，就没机会提交第二张。

Group 让这个线程先收集两张卡的调用，再一起处理：

```text
CPU线程: GroupStart -> 收集rank0 -> 收集rank1 -> GroupEnd -> 等待两卡完成
GPU 0:                                             AllReduce -------|
GPU 1:                                             AllReduce -------|
```

以下片段摘自完整示例的提交阶段，`ndev=2`，数组保存每张卡自己的对象：

```cpp
NCCL_CHECK(ncclGroupStart());
for (int r = 0; r < ndev; ++r) {
  CUDA_CHECK(cudaSetDevice(devices[r]));
  NCCL_CHECK(ncclAllReduce(send[r], recv[r], count, ncclFloat, ncclSum,
                           comms[r], streams[r]));
}
NCCL_CHECK(ncclGroupEnd());
```

- `cudaSetDevice` 选当前线程接下来操作的 GPU，不会把所有 GPU 变成一张设备。
- `send[r]`、`recv[r]`、`comms[r]`、`streams[r]` 属于同一个 rank 的设备上下文。
- 循环是在 **CPU 上提交调用**，没有把 GPU 数组复制到 CPU 求和。
- 外层 GroupEnd 是提交边界，不是 GPU 完成屏障，也不保证只产生一个 kernel。

### 不要把等待放进尚未结束的 group

```text
错误的组织方式（不要运行）：
GroupStart
  收集 rank0 的 AllReduce
  等待 rank0 stream
  收集 rank1 的 AllReduce
GroupEnd
```

group 内的通信可能尚未排入 stream，此时等待不能证明它已完成；与其他依赖结合还可能形成死锁。依赖这组通信的 CUDA 工作和完成等待，应放在有效的 GroupEnd 提交完成之后。

初始化用的 `ncclCommInitRank` 也能按 group 组织，但不要把 communicator 初始化和 collective 混进同一 group。最小程序用 `ncclCommInitAll` 完成单进程初始化，详细实现留到第 05 章。

**停一下：** 两卡通信提交齐了，是否只等 GPU 0 就能释放 GPU 1 的缓冲？**不能；要等待每个相关设备的工作完成。**

## 3. 缓冲生命周期：什么时候可以覆盖或释放

`cudaMalloc` 分配的是空间，输入复制才放入有意义的测试数据；AllReduce 的结果在执行结束后才有效。

| 阶段 | send | recv | CPU 此时可以做什么 |
|---|---|---|---|
| 分配后、初始化前 | 未准备好输入 | 未产生结果 | 安排输入复制 |
| 复制已排队、通信已提交 | GPU 可能仍要读 | GPU 可能仍要写 | 做不依赖这两个缓冲的工作 |
| 相关通信已完成 | 本轮不再读取 | 有本轮结果 | 在正确设备上安排读回或后续使用 |
| D2H 及全部其他使用结束 | 可复用/释放 | 可复用/释放 | 校验 CPU 数组、清理资源 |

例如通信尚未读取完 `send`，另一个 stream 就写下一批输入，会造成“同一个地址上两轮数据相互覆盖”。不是再加一次 NCCL 调用就能修复，必须为两次使用建立依赖，或采用不重叠的双缓冲。

原地 AllReduce 的 send 和 recv 是同一块内存，所以通信会覆盖原始输入；需要保留输入作对照时，要提前保留一份。其他 collective 的合法原地关系见第 02 章，不能都套 `send==recv`。

buffer、stream 和 communicator 都要活到相关使用结束；进阶的注册 handle、事件和 Graph 对象同样遵守这一原则。

## 4. 本地顺序正确，还需要所有 rank 的调用匹配

同一 stream 排序只管本地依赖，不能替你纠正不同 rank 提交了不同操作。

```text
正确：
rank0: AllReduce(A) -> AllGather(B)
rank1: AllReduce(A) -> AllGather(B)

错误：
rank0: AllReduce(A) -> AllGather(B)
rank1: AllGather(B) -> AllReduce(A)
```

两个 rank “最终都调用了这两个函数”还不够，它们必须按同一逻辑顺序匹配。相同 API 也要有一致的 count、datatype、root、op 等相应契约；否则可能报错、挂起或出现错误数据。

**怎么检查？** 先为每轮写一张表：`轮号、comm逻辑标识、rank、操作、count、type、op/root`。比较第一个分歧，而不是等挂起后先改网卡配置。跨进程的 communicator 指针不相等是正常的，不能拿指针值作为全局标识。

第一轮实验只使用一个 communicator，每个 rank 按相同顺序提交。多个 communicator 的交错发射、从多个线程无协调地访问同一 comm，会引入额外依赖；它们留到基础稳定后再研究。group 也不是多个线程可以共同使用的锁，Start/End 要按其线程语义配对。

## 5. 看懂示例里的等待代码：查询不等于报错

完整示例没有只做一个可能永久等待的 synchronize，而是重复查询状态。初读时把它理解为三个问题：

```text
循环检查所有相关GPU：
  NCCL是否报告了异步错误？
  这条CUDA stream是否已经完成？
  如果还没完成，等待时间是否已超过期限？
```

### CUDA query 的三个结果

| `cudaStreamQuery` 结果 | 含义 | 动作 |
|---|---|---|
| `cudaSuccess` | 查询到该 stream 的相关工作已完成 | 标记这一张卡完成 |
| `cudaErrorNotReady` | 工作尚未完成 | 继续等待，不当作通信失败 |
| 其他错误 | CUDA 执行/查询出现问题 | 退出正常流程并报告 |

### NCCL query 有两个值，不能只看函数返回

```cpp
ncclResult_t state;
ncclResult_t query_status = ncclCommGetAsyncError(comm, &state);
```

- `query_status`：查询 API 本身是否成功。
- `state`：查到的 communicator 异步状态。

默认示例正常提交后期望两者都成功，再结合 CUDA query 判断完成。NCCL 未报告错误，不意味着 CUDA stream 已经完成；反过来，也不应忽略已发现的 NCCL 异步错误。

**查询错误 ≠ 选择非阻塞 communicator。** 默认 communicator 也有异步执行错误，所以示例会调用 `ncclCommGetAsyncError`，但没有因此启用第 8 节的非阻塞配置。

示例的 60 秒 deadline 只覆盖**成功提交后**的 GPU 等待，不覆盖可能阻塞的初始化、GroupEnd 或清理本身。超时表示没有按期完成，并不直接证明网络损坏；排查见[第 12 章](12-debugging.md)。

## 6. 跑完与失败是两种退出路径

正常路径是：确认相关 GPU 工作结束 → 完成读回并校验 → 销毁 communicator、释放用户缓冲与 stream。完整示例中每张卡都在对应设备上下文下清理。

出现错误则停止继续提交，不把半完成输出当作有效数据；需要协调 abort/退出或恢复。`ncclCommAbort` 不会补齐丢失的梯度，也不会自动重启远程进程。最小示例采用 fail-fast（报错后停止），不是生产级容错框架。

**机制检查：** 若本地 `ready` 已满足、NCCL kernel 已 launch，但 stream 迟迟不完成，请至少区分 peer 未匹配、网络/proxy 尚未推进、缓冲 credit 耗尽这几类假设。它们并不都能靠增加一个本地 CUDA 同步解决；把等待对象列清楚，再到第 07/09 章找实际状态。需要复习程序生命周期时，再查[示例导读](examples/README.md)。

## 7. 主线：NCCL 如何接入多个 stream 的依赖图

当输入生成、通信、结果消费不在同一 stream 上，原有的自动顺序就不够了，需要显式连接：

```text
compute stream: produce -> record(ready) -> unrelated compute -> wait(done) -> consume
                              |                                  ^
comm stream:              wait(ready) -> AllReduce -> record(done)
```

这里 event 是记录某个设备执行位置的事件。`cudaStreamWaitEvent` 为目标 stream 安排等待，不是让 CPU 在调用处等到事件完成；因此 CPU 可以继续提交其他工作。

以下仅是依赖示意，`produce/consume`、事件、stream 和错误宏须在完整程序中定义。假设默认 communicator、每 rank 一线程且没有未结束的外层 group：

```cpp
produce<<<grid, block, 0, compute_stream>>>(input);
cudaEventRecord(ready, compute_stream);
cudaStreamWaitEvent(comm_stream, ready, 0);
ncclAllReduce(input, output, count, ncclFloat, ncclSum, comm, comm_stream);
cudaEventRecord(done, comm_stream);
cudaStreamWaitEvent(compute_stream, done, 0);
consume<<<grid, block, 0, compute_stream>>>(output);
```

逐个箭头看：ready 保证输入生成后通信才读它；done 保证通信写完后消费 kernel 才读输出。完整程序还要检查 CUDA/NCCL 返回值与 kernel launch 错误。

事件应在正确设备上下文中创建/记录/等待，并活到所有相关使用结束。不要依赖默认 stream 的隐式同步碰巧掩盖缺失依赖；换成独立 stream 后，错误就可能显现。

两个 stream 能并发，也不意味着端到端一定更快：通信与计算会争用 GPU 的 SM（执行线程块的计算单元）、HBM（显存）带宽等。若计算独跑 10 ms、通信独跑 4 ms，并发后的总时间仍可能超过 10 ms；这些数字只是示意。是否有收益要看[第 16 章](16-training-integration.md)的应用关键路径。

### 同一 group 中使用多条 stream：先汇合，再分发完成依赖

与上面的单个 collective 不同，设同一 GPU、同一 communicator 的两次 collective 在一个 group 内分别使用 stream A、B；其他 ranks 也提交匹配的调用。下面只画本 GPU，在本版普通、非捕获的默认 host 路径中，假设 A 被选为承载发射的 stream：

```text
A 前序工作 ──┐                        ┌──→ A 后续工作
             ├─→ A 上的本组 NCCL 工作 ─┤
B 前序工作 ──┘                        └──→ B 后续工作
              发射前汇合                 完成后分发依赖
```

发射前，A 等待 B 已有的前序工作；发射后，B 的后续工作又要等待本组通信完成。NCCL 因此不只是把两个 CPU 调用装进同一批，它也为参与 stream 建立了设备依赖；图中的 NCCL 工作可能包含多个 kernel，不承诺合成一次发射。

例如 B 前面有一项很长的计算，即使它不生产 A 的通信输入，也可能通过这次汇合推迟 A 的通信。只有跨过有效 GroupEnd/主机提交完成边界后，再向 A、B 添加依赖本组结果的后续工作，才能使用上述完成关系。

源码对应 [nccl/src/enqueue/enqueue.cc:1797–1801](../nccl/src/enqueue/enqueue.cc#L1797) 的发射前等待，以及 [2079–2081](../nccl/src/enqueue/enqueue.cc#L2079) 的完成后等待；这里不把同 GPU 的安排泛化成不同 GPU 共用一条 CUDA stream。

## 8. 选读：非阻塞 communicator 是另一层异步

### 两个相似名字，控制的不是同一件事

| 配置 | 控制什么 |
|---|---|
| `cudaStreamNonBlocking` | CUDA stream 与 legacy default stream 的隐式同步关系 |
| `ncclConfig_t::blocking=0` | NCCL 某些主机操作是否允许尚在进行时返回 |

`cudaStreamNonBlocking` 不是“让 NCCL 永远立即返回”，默认 blocking communicator 也不是“等待所有 GPU 通信完成”。最小示例采用前者创建 stream，但没有设置后者。

如果确实需要非阻塞 communicator，先用宏初始化配置：

```cpp
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;
```

配合 `ncclCommInitRankConfig` 或非阻塞 group 等操作时，可能得到 `ncclInProgress`。这表示相关主机内部工作尚未结束，此时不能直接提交依赖它的下一步操作。

```text
发起主机操作
  -> success: 主机侧该操作完成
  -> inProgress: 查询 ncclCommGetAsyncError
       -> inProgress: 继续等待/检查deadline
       -> success: 主机侧该操作完成
       -> 其他错误: 停止正常路径并协调异常处理
  -> 其他错误: 处理立即失败
```

每次查询仍要检查第 5 节的两个结果。多个非阻塞 communicator 组成 group 时，要检查相关 comm，不是其中一个成功就代表全组成功。

主机提交结束后，才可正确添加依赖这些提交的 CUDA 操作或完成事件；此时也仍要另外等待 GPU 数据完成。把三种时刻放在一起看：

| 时刻 | 可以推断什么 |
|---|---|
| group 内一次调用返回成功 | 当前调用未立即失败，工作可能还未提交 |
| 默认外层 GroupEnd 成功，或相关非阻塞状态变为 success | 相关主机提交工作已完成 |
| 无相关 NCCL 错误且对应 CUDA stream/event 确认完成 | 已提交的相关 GPU 工作完成，后续按正确依赖使用结果 |

完整非阻塞生命周期和 finalize/destroy/abort 的边界见[第 05 章](05-communicator.md)，不是在示例上加一句 `blocking=0` 就完成了转换。

## 源码锚点：理解顺序后再查契约

- [公共 API](../nccl/src/nccl.h.in)：`nccl/src/nccl.h.in:135`，`NCCL_CONFIG_INITIALIZER`；`:281`，带配置初始化。
- 同文件 `:391`，`ncclCommGetAsyncError`；`:537`，collective 的排入 stream 语义。
- 同文件 `:844`，group 语义；`:877`，依赖操作应位于 GroupEnd 之后。
- [非阻塞初始化返回处理](../nccl/src/init.cc)：`nccl/src/init.cc:3049`，`ncclCommInitRankConfig`。

## 自测与答案

1. 输入复制与通信在同一 stream 上，需要每个调用后都做 CPU 同步吗？**不需要，同 stream 顺序可以表达设备依赖；CPU 要使用结果时再建立正确完成保证。**
2. 两卡 AllReduce 为什么不是 CPU 循环把两个数组相加？**循环只提交每个 rank 的调用，数据规约由 NCCL 选择的数据通路执行。**
3. `cudaErrorNotReady` 是本轮通信失败吗？**不是，它表示查询时尚未完成。**
4. group 内 AllReduce 后马上记录 event，再 GroupEnd，这个 event 能代表通信完成吗？**不能，应放在有效的 GroupEnd/主机提交完成边界后。**
5. 选读：stream 设置 `cudaStreamNonBlocking`，是否自动启用 NCCL 非阻塞 communicator？**没有，它们属于不同 API 层和同步机制。**

下一章：[04 集合算法](04-algorithms.md)，先用 4.2～4.7 手算 Ring/Tree 的数据流，再进入初始化与连接。需要动手时并行查阅[环境准备](00-environment.md)与[单进程示例](examples/README.md)，基础逐行带练不是源码主线的前置关卡。
