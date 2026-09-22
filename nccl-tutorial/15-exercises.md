# 15 分级练习：从会复述到能验证

建议每个阶段先独立做题，再看本页后半部分的答案。涉及 GPU 的题在自己的 Linux 实验环境执行；不要用未获授权的集群资源做故障注入。

## A. 基础：不用 GPU 也能完成

### A1. 识别三个编号

两节点各四张 GPU，每节点四个进程、每进程一卡。第一节点承载 global ranks `[0,1,2,3]`，第二节点承载 `[4,5,6,7]`；节点内按 global rank 升序确定 local rank，所有本地进程看到相同顺序的完整 GPU 列表，并按 local rank 选卡。global rank 6 的 local rank 是多少？应选哪个 CUDA device？如果子 communicator 只含 global ranks `[2,6]` 并按此顺序编号，它在其中 rank 是多少？

### A2. 精确计算缓冲长度

`P=8`、FP32、每 rank 的局部片段 256 个元素：

1. AllGather 的 sendcount、send 字节数、recv 字节数分别是多少？
2. ReduceScatter 的 recvcount、send 字节数、recv 字节数分别是多少？
3. AlltoAll 的 per-peer count=256 时，每 rank send/recv 各多大？

### A3. 手算数据布局

三个 rank 的输入分别为 `[0,1,2]`、`[10,11,12]`、`[20,21,22]`：

- AllReduce Sum 的各 rank 输出？
- ReduceScatter Sum，recvcount=1 的各 rank 输出？
- 若改为每 rank 发送 `[rank]` 做 AllGather，输出？

### A4. 判断原地关系

`P=4, rank=2, C=128`，用 `float* buffer` 分配了 512 个元素：

1. AllGather 应把本地输入放到 `buffer` 的哪个偏移？
2. ReduceScatter 原地输出应指向哪里？
3. `send=buffer, recv=buffer+1` 是合法 AllReduce in-place 吗？

### A5. 区分三种完成

分别判断能否立即让 CPU 读取 recv：

- group 内 `ncclAllReduce` 返回 success。
- 默认 blocking communicator 的 GroupEnd 返回 success。
- NCCL 状态无错误，相关 CUDA stream 已确认完成，随后正确执行 D2H copy。

## B. 算法与测量：纸笔 + Python

### B1. 运行并解释模拟器

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

解释：为什么 ReduceScatter 内部阶段结束后 r0 拥有 c1，而不是 c0？这违反公共 `ncclReduceScatter` 语义吗？为什么每轮必须先收集所有发送快照？

### B2. 修改推导，不修改程序

若第 0 轮令 rank r 发送块 `(r-1) mod P` 而不是 r，推导 P−1 轮后 rank r 完成的块编号。它与原模拟器是不同算法，还是不同编号起点？

### B3. 计算带宽

4 rank 的 AllReduce，每 rank 数组 64 MiB，耗时 2 ms：

- algbw 是多少十进制 GB/s？
- busbw 是多少？
- Ring 模型下每 rank 发送的总 payload 是多少 MiB？
- 若改成 8 rank、S/T 不变，归一化 busbw 是多少？

### B4. 识别无效结论

下面哪些结论不成立，并说明原因：

1. busbw=100 GB/s，所以某一根 NVLink 的实测流量就是 100 GB/s。
2. 开启四倍 channel，一定得到四倍性能。
3. 大包 Ring 快于 Tree，所以所有小包也应强制 Ring。
4. `#wrong=N/A`，所以正确性已通过。
5. 同一 count 的 FP16 和 FP32 测试传输字节量相同。

### B5. 用简单模型找拐点

假设某路径 `T ≈ alpha + S/B`，alpha=10 μs，B=50 GB/s。当数据传输项等于固定项时，S 大约多少字节？这是不是 NCCL 在所有机器上切算法的精确阈值？

## C. 编程：两张 Linux GPU

### C1. 复现基线与合法 in-place

先运行[单进程示例](examples/README.md)，确认两卡所有元素为 3。另建自己的实验副本，把 AllReduce 改为合法 in-place，继续检查所有元素。

验收：两种形式结果相同；能够指出修改了哪些缓冲/指针与生命周期，不能只看到程序没有崩溃。

### C2. 从 AllReduce 改成 AllGather

令 rank r 的局部数组全为 r+1。修改接收容量并使用正确 sendcount，检查每个接收块。

验收：P=2 时前半全为 1、后半全为 2；能区分“传输顺序”和“API 输出顺序”。

### C3. 添加一个有意义的错误检测

只在你自己的 CPU 校验逻辑中故意修改读回数组的中间一个值，再运行检查，不修改 NCCL 或网络配置。

验收：程序必须报告 FAIL。此题验证的是**你的校验器**，不是 NCCL 的容错机制。完成后恢复自己的实验副本。

### C4. 用事件表达双 stream 依赖

为输入生成和输出消费增加 CUDA kernel，通信使用独立 stream，使用 ready/done events 连接依赖。先画依赖图，再编码。

验收：结果正确；时间线没有读未就绪输入或消费未完成输出；不要仅凭用了两个 stream 宣称性能提升。

## D. 系统与源码：从现象到证据

### D1. 做一份三组对照

同一两卡环境、同一尺寸扫描，对比自动配置、适用的 Ring/Simple、适用的 Tree/Simple。保持其他条件一致；不支持的组合记录为不支持，不要篡改实现去绕过。

验收记录至少包括：版本、拓扑、完整命令、校验状态、重复运行分布、小包延迟与大包带宽。结论只适用于实际测量条件。

### D2. 追踪一次 AllReduce

沿 `ncclAllReduce → ncclAllReduceConfigImpl → ncclEnqueueCheck`，再根据当前路径到 group/plan/launch。写出至少三个实际结构或对象的“生产者、消费者、生命周期”。

验收：不能只交函数名列表，要说明用户参数何时变成 task/plan，GPU 看到的描述与 host 结构有什么区别。

### D3. 解释一份网络日志

假设日志同时出现 bootstrap Socket 与 NET/IB。能否判断矛盾？还需要什么证据确认目标 collective 的 payload 路径？

验收：区分引导与数据通道，并说明为什么还要核对 rank/GPU/NIC 对应关系、实际连接及必要的时间线/网络计数器。

### D4. 设计挂起排查顺序

现象：两机各四卡，初始化正常，运行第 10 次 collective 时挂起。你不能修改系统安全配置。列出优先检查项，以及为什么应先缩小复现。

验收：包括 rank 调用顺序和 count/type/op、此前 CUDA 错误、进程存活、异步状态、网络与日志；不能直接以“关掉 ACS/IOMMU”作答案。

### D5. 解释训练反常结果

某修改使 nccl-tests 峰值提高 15%，训练 step 却慢了 3%。提出至少三个可验证假设，并说出各自需要的证据。

## 参考答案

### A 组

**A1：** local rank=2，普通完整可见列表下 device=2；子 communicator 按 `[2,6]` 的顺序建组时，global rank 6 的子 rank=1。若建组 key 或列表顺序改变，子 rank 也可能改变。

**A2：** AllGather：sendcount=256，send=1024 B，recv=8192 B。ReduceScatter：recvcount=256，send=8192 B，recv=1024 B。AlltoAll：每 rank send/recv 都是 `8×256×4=8192 B`。

**A3：** AllReduce 所有 rank 都是 `[30,33,36]`。ReduceScatter 分别得到 `[30]`、`[33]`、`[36]`。AllGather 所有 rank 得到 `[0,1,2]`。

**A4：** 两种偏移都是 `rank*C=256` 个 float，即 1024 字节；AllGather 的输入在大接收数组的该位置，ReduceScatter 的输出在大发送数组的该位置。第三种不是合法原地关系。

**A5：** 前两种不能。第三种在 D2H 完成后可以读取 CPU 数组；不能把 GPU 完成与尚未完成的异步 D2H copy 又混为一谈。

### B 组

**B1：** 模拟器采用起始发送块为 r 的内部安排，P−1 轮后完成块为 r+1。它模拟 AllReduce 内部阶段，不直接实现公共 ReduceScatter 的用户布局，不构成 API 输出承诺。先做发送快照是为了保持同步轮次，避免 Python for 循环让后处理 rank 读到本轮刚更新的数据，多走一跳。

**B2：** 完成块为 r。改变的是块编号起点，通信方向、每轮大小和轮数不变；两种安排可通过映射对齐。

**B3：** `S=64×2^20=67,108,864 B`，`T=0.002s`：

```text
algbw = S/T/10^9 = 33.554432 GB/s
P=4: busbw = algbw × 2×3/4 = 50.331648 GB/s
每rank发送量 = 2×3/4×64 MiB = 96 MiB
P=8: busbw = algbw × 2×7/8 = 58.720256 GB/s
```

每 rank 还会接收相应数据，但不能在此公式上再随意乘二后仍称为同一 busbw 定义。

**B4：** 全部不成立。依次因为：busbw 是归一化指标；资源有瓶颈与争用；小包受固定延迟影响；N/A 表示未报告有效校验结果；count 是元素数，类型字节数不同。

**B5：** `S=alpha×B=10×10^-6×50×10^9=500,000 B`，约 488.3 KiB。只是简化模型的成本相等点，不是 NCCL 的精确算法阈值。

### C 组

**C1：** AllReduce 的输入输出用同一个设备缓冲，初始化、提交、完成等待和校验仍须正确；不要把已被覆盖的 send 当成原始数据继续使用。

**C2：** 发送长度仍为 count，接收长度改为 P×count。`recv[source_rank*count+i]` 应为 source_rank+1。每个 rank 的最终布局相同。

**C3：** 校验应遍历所有元素，汇总非零错误并返回失败退出码。只检查第一个元素会漏掉中间错误。

**C4：** 至少有 `produce → ready → comm wait → AllReduce → done → compute wait → consume`。如果使用 group，事件记录要放到有效提交边界后；非阻塞 communicator 还需确认 host 提交完成。

### D 组

**D1：** 没有统一跑分答案。高质量结论可能是“该机器上大包配置A略优，但小包或并发计算场景不优”。记录失败/不支持和波动，比只留最高一次更可信。

**D2：** 可选 `ncclInfo`、任务队列、kernel plan、device work 描述、proxy operation 等，按当前实际分支追踪。参考第 05/14 章，不要求把全部新旧调度路径一次读完。

**D3：** 不矛盾，bootstrap 可用 socket，bulk 可用 IB。需要确认日志对应的阶段与连接，以及是否有 fallback、插件、不同 rank 路径混合等情况。

**D4：** 先排应用合同和前置 GPU 工作，再看异步错误/peer退出、各 rank 日志及网络状态；固定第10次输入和调用序列，逐级缩到单机、两机各一卡、最小 collective。保留能触发问题的条件，不靠删同步/关闭安全机制规避。

**D5：** 示例假设：更多 CTA 抢占计算资源；提升的是训练很少用的大包；通信原本已被覆盖；慢 rank 到达更晚；bucket/注册导致显存或 host 开销增长。分别用 CUDA 时间线、实际消息分布、关键路径、跨rank ready时刻、内存与CPU记录验证。

## 毕业任务

用一次你能访问的真实工作负载，交付一份可重复实验记录：

1. 画出 rank/GPU/NIC 与 communicator 映射。
2. 列出主要 collective 的大小与依赖。
3. 通过源码指出一个关键决策点和一个数据推进点。
4. 对照测量提出改动，并验证结果正确性与端到端收益。
5. 明确哪些结论是实测、哪些是源码事实、哪些仍是待验证假设。

能做到这五点，就已经从“会调用 NCCL”迈入“能解释和优化 NCCL”的阶段。
