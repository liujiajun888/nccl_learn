# 15 分级练习：从会复述到能验证

建议每个阶段先独立做题，再看本页后半部分的答案。涉及 GPU 的题在自己的 Linux 实验环境执行；不要用未获授权的集群资源做故障注入。

## 机制检查站：CUDA/UMD 读者的推荐入口

这一组检查的是机制理解，不用先重做数组和指针练习，也不用按题号一次做完：读 04 核心配 B1/B2；读 05/06/07 配 D3；读 08 配 M1/M2，再用 D2 验证批处理；读 09 配 M3/M4；读 10/11 配 M5。答题先写清状态、执行者、前进条件，再给源码或观测依据。所有题目均可先做纸面分析，有 GPU 时再做自己的受控验证。

### M1. 本地提交完成了，为什么通信仍可能不动？

假设本 rank 输入已就绪、NCCL kernel 已 launch，但 stream 迟迟不完成。列出三类仍未满足的跨参与者条件，并说明每类分别查哪份证据。再回答：为什么向同一 stream 多排一个同步/等待，不能补齐缺席 rank 的操作？

入口：[03 的跨 rank 等待图](03-cuda-semantics.md)、[08 的 work/proxy 两路推进](08-host-execution.md)、[12 的分层诊断](12-debugging.md)。

### M2. 将一次 API 请求追到设备，但不把对象画错

从 `ncclAllReduceConfigImpl` 开始，标出四类对象各由谁持有、谁消费：临时 `ncclInfo`、较长寿命的 task、kernel plan、GPU work descriptor。再解释两点：它们为什么不是同一个硬件 command buffer；host API 返回后，用户缓冲为何仍必须有效。

入口：[08](08-host-execution.md)、[14 的状态追踪模板](14-source-map.md)。验收不是函数名列表，而是一张生产者/消费者/寿命表。

### M3. 地址可访问、MR 已注册、请求完成分别证明什么？

对普通 NET/GDR 路径，依次讨论四种状态：CUDA 允许 GPU 访问某映射；网络已为该范围建立 MR（memory region，注册内存区域）；某个网络 request 被确认完成；整个 collective 输出已可消费。为什么前三项不能任意替代最后一项？在非用户 buffer 直接注册路径中，协议 FIFO 的消费反馈与应用输出缓冲的复用依赖，分别由谁管理？

入口：[07 的连接与注册](07-transports.md)、[09 的数据发布与背压](09-device-protocols.md)。先明确“哪一端、哪一片、哪种完成”，不预设 request 对应整个用户数组。

### M4. 手算 FIFO 安全条件，并追到代码

在普通 Simple FIFO 模型下，`NCCL_STEPS=8`、`StepPerSlice=2`。发送者 `step=8, head=1` 时能否写下一片？head 变成 2 后呢？若接收者 `step=8, tail=9`，能否读取下一片？一个 system-scope fence 能否把这些不满足的条件变成满足？

入口：[09 的 waitPeer/postPeer](09-device-protocols.md)。同时回答 ready 与 credit 各保护哪种错误，并找出 GPU 或 proxy 在当前连接上更新相应状态的位置。

### M5. 让一个性能解释可以被验证或推翻

假设增加 channel/CTA 后，通信独跑更快，但与计算重叠的端到端耗时更长。列出要固定的变量，以及至少两类需要收集的时间线/计数证据。为什么只比较 `Avg bus bandwidth` 不足以决定保留改动？

入口：[10 的计时与数据量](10-nccl-tests.md)、[11 的资源竞争](11-performance.md)、[16 的关键路径](16-training-integration.md)。无需预设你懂训练框架，先以一个 producer→通信→consumer 依赖图分析。

### 答案要点：用于核查，不替代源码证据

| 题目 | 答案要点 |
|---|---|
| M1 | peer 没有匹配/尚未产生数据、proxy/网络尚未完成请求、消费者未返还 credit 都可能阻塞；分别核对各 rank 调用序列、请求/进度与 FIFO 状态。额外本地等待只等既有依赖，不会替对端创造输入或调用 |
| M2 | info 传递意图，task/plan 保留调度元数据，device work 供 kernel 消费；它们不等于 UMD 的硬件命令存储。元数据已复制不代表 payload 已读完，用户 buffer 寿命另行保证 |
| M3 | 映射/权限、网络注册、某请求完成各属不同层；collective 还须完成所选协议的可见性、剩余分片和集合步骤。NCCL 管理协议 FIFO 的信用，应用用 CUDA 依赖管理用户输出的寿命；本地 AllReduce 正常完成后可消费本地输出，复用须等自己的后续消费者结束，不另等远端应用确认 |
| M4 | head=1 时 `1+8<8+2`，不能发送；head=2 时空间条件满足；tail=9 时 `9<8+2`，不能接收。fence 约束访存顺序，不增加空间、不生成远端数据，不能代替这两个谓词 |
| M5 | 固定操作、尺寸、rank/设备映射、算法协议与测量口径，比较通信/计算独跑和并发时间；检查 CTA/带宽争用、rank 到达与尾部等待。聚合 busbw 不等于应用关键路径耗时 |

接着按需要完成后面的 B/D 组源码与测量题；N 组和 C1/C2 保留作基础速查或快速正确性练习。

## 入门检查站：基础带练速查

需要数组与调用顺序复习时再做这一组，不作为机制主线的前置。没有 GPU 也能完成前三项；第四项要在 Linux NVIDIA GPU 上实际验证，暂时没有设备就标为“待实测”。

### N1. 不看函数名，先画答案

rank 0 的输入为 `[1,2,3,4]`，rank 1 为 `[10,20,30,40]`，类型为 FP32。请分别填写三个操作的每 rank 输出、API count 和缓冲字节数：

1. AllReduce Sum，所有元素参与。
2. AllGather，每个 rank 贡献完整四个元素。
3. ReduceScatter Sum，每 rank 接收两个元素。

**提示：** AllGather 拼接，不相加；ReduceScatter 先按对应位置定义规约结果，再按 rank 分块。先写元素数，最后乘 4 得到字节数。

### N2. 四张纸条怎样排序

把下面动作排成安全顺序，并指出哪些可以通过同一 stream 依赖保证，而不需要每一步都让 CPU 同步：

- CPU 校验读回的数组。
- 把 CPU 输入复制到 GPU。
- 提交两张卡的 AllReduce。
- 等待相关 GPU 通信完成并完成 D2H 读回。

**追加一问：** 一个 CPU 线程管理两张卡时，GroupStart/End 应包住什么？

### N3. 找出“没有检查”的地方

某次 tests 的两组结果分别写着 `#wrong=0` 和 `#wrong=N/A`。能否说两种布局都已校验正确？两组 time 是 GPU 0 与 GPU 1 的耗时吗？

**提示：** 回第 10 章找 out-of-place/in-place 表头，不凭数字所在位置猜含义。

### N4. 程序跑通以后，逐段解释一次

按[示例导读](examples/README.md)运行两卡程序。指着代码回答：每个 rank 的输入在哪里填充？每份显存多大？哪个调用建立通信关系？哪个循环只是提交而不是 CPU 求和？哪里确认 GPU 完成？哪里检查了全部 1024 个元素？

**验收：** 每 rank `first=3 expected=3 wrong=0`，最后 `PASS`，且能回答上述六问。只看到 `PASS` 但不知道程序检查了什么，还不算完成这道题。

### 参考答案与回读入口

| 检查项 | 答案要点 | 不确定时回哪里 |
|---|---|---|
| N1 AllReduce | 两 rank 都是 `[11,22,33,44]`；count=4；每 rank send/recv 各 16 B | 第 02 章的两卡例子 |
| N1 AllGather | 两 rank 都是 `[1,2,3,4,10,20,30,40]`；sendcount=4；send 16 B、recv 32 B | 第 02 章的拼接与容量 |
| N1 ReduceScatter | rank 0 为 `[11,22]`，rank 1 为 `[33,44]`；recvcount=2；send 16 B、recv 8 B | 第 02 章的分片与 count |
| N2 | 复制→通信→完成等待及读回→CPU校验；同 stream 保证复制在通信前；group 包住两卡的通信提交，不把等待夹在 group 内 | 第 03 章第 1～2 节 |
| N3 | N/A 不代表已校验；两组为 oop/ip 布局结果，不是两张卡 | 第 10 章入门输出解读 |
| N4 | 输入全为 rank+1；send/recv 各 1024×4=4096 B；InitAll 建组；GroupEnd 后轮询；遍历全部输出再报 PASS | examples 中的逐段程序说明 |

通过后先尝试 C1 的合法原地改造；A1 的多节点编号、B 组算法、D 组性能与源码题是后续阶段，不要求第一天全部做完。

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

### D2. 验证两次调用是否会合成一次发射

在 M2 已追清单个请求生命周期的基础上，比较“一个 group 内提交两次独立 AllReduce”和“分别提交”。先固定前提：每个 rank 的两次操作顺序一致、尺寸/类型/规约相同、使用同一 stream，各次输入输出缓冲互不重叠；选普通非捕获路径，连接已通过预热建立。

先沿 task 聚合与 plan 打包源码，找出兼容性及参数/work 存储预算怎样允许合并或迫使拆分。若有 GPU，再用自己的受控实验记录两种组织方式的实际 plan/kernel 发射数量及输出正确性；没有 GPU 就给出分支条件，不能把推断写成观测值。

验收：能解释为何“同一个 group”不保证“一次 kernel”，并分别给出源码判断与实际观测的证据；不再重复 M2 的对象定义表。

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

**C4：** 至少有 `produce → ready → comm wait → AllReduce → done → compute wait → consume`。输入事件 `ready` 在 produce 后记录，通信 stream 再等待它；若使用 group，代表通信输出完成的事件 `done` 必须在有效 GroupEnd 提交边界后记录。非阻塞 communicator 还需先确认相关 host 提交完成，才能安排这个 `done` 事件。

### D 组

**D1：** 没有统一跑分答案。高质量结论可能是“该机器上大包配置A略优，但小包或并发计算场景不优”。记录失败/不支持和波动，比只留最高一次更可信。

**D2：** 兼容 task 可以共同调优、装入 plan，但参数空间和 work 存储预算等仍可能令同一 group 拆成多个 plans；API 次数不能直接当作 kernel 次数。参考第 08 章的 task 加工、plan 打包与实际 launch 路径，分别记录分支条件和观测到的数量。正确性校验须覆盖两次操作；没有 GPU 时只给源码推断，不填写假想发射次数。

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
