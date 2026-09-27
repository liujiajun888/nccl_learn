# 01 建立全局模型：NCCL 究竟在解决什么问题

## 学习目标

- 校准 rank/communicator 与 CUDA device/context 的区别，理解集合通信的结果契约。
- 建立语义、算法、协议、transport 和硬件的分层模型，不把本地提交成功当作分布式进度保证。
- 用既有 CUDA/UMD 经验理解执行者分工，定位哪些机制需要继续追源码。

**推荐路线：**快速核对第 1 节的 AllReduce 语义和 2.1/2.2/2.5 的对象定义，重点读 2.6 的 rank/进程映射、第 3～5 节的分层与执行链。2.3/2.4 的 buffer/stream 基础及两卡自答题保留作速查；无需先学深度学习，但多 rank 的契约要重新建立。

## 1. 从两张卡的数组说起

两张 GPU 各有四个数，接下来两张卡都要用到“对应位置相加”的完整结果。先把两位参与者叫作 rank 0、rank 1：

```text
局部输入（各卡目前只有自己的数据）
rank 0: [ 1,  2,  3,  4] ─┐
                          ├─ 对应位置求和: [11, 22, 33, 44]
rank 1: [10, 20, 30, 40] ─┘             │
                            ┌─────────┴─────────┐
全局结果                 rank 0              rank 1
                    [11,22,33,44]       [11,22,33,44]
```

“局部”指只来自一位参与者；“全局”指合并了这个通信小组的全部贡献，不是机器里所有 GPU。例如第一个结果是 `1+10=11`。这里的合并叫 **规约（reduction）**，规则是 Sum。它既不是把 `[1,2,3,4]` 加成一个数，也不是把两个数组拼接起来。

为什么结果每卡一份？因为题目要求**两张卡都能继续用完整结果计算**，不能只让一张卡拿到答案。这个约定叫 `AllReduce(Sum)`：Reduce 表示规约，All 表示每个参与者都有结果。每张卡的输出是自己的数组副本，不是两张卡共用一个指针。图中“求和”只是结果定义，不指定在哪张卡上算。

> **停一下自己答 1：**rank 0 的输出是 `[10]`、八个数，还是 `[11,22,33,44]`？
>
> **答案：**最后一种。跨 rank 对应位置相加，仍有四个元素；另外两种分别混淆了卡内求和与拼接。

现在再看四卡训练。可先把**梯度**理解为“每张卡根据自己的样本算出的参数调整建议数组”，不必先学它如何求导。同样的两项参数，在四张卡上分别得到：

```text
rank 0: [1, 2]      rank 1: [3, 4]
rank 2: [5, 6]      rank 3: [7, 8]
                 ↓ AllReduce(Sum)
每张卡: [16, 20]    （1+3+5+7=16，2+4+6+8=20）
```

若各卡原有参数及更新规则相同，使用同一个总梯度才能保持更新一致；若训练需要平均值，再按约定缩放。**规约规则**与**结果放到哪些 rank**是两个独立维度，下一章会对照展示。

可以先让 CPU 收齐数据、求和后再发回，但这会引入 GPU↔CPU 搬运、中心节点瓶颈和额外等待。NCCL 利用 GPU 与互连，把搬运、规约、转发组织成高效的并行通信。

## 2. 接口对象与提交者：哪些不是驱动对象的别名

### 2.1 Rank：通信小组里的逻辑编号

上例两位参与者分别是 rank 0、rank 1。若小组有 `P` 位参与者，编号就是 `0...P-1`。rank **相对于某个 communicator**，不是 GPU 的永久身份证：同一 GPU 在小组 A 中可以是 rank 0，在小组 B 中可以是 rank 1。

### 2.2 Communicator（comm）：规定谁和谁通信

comm 是通信小组及相关资源的上下文，应用通过 `ncclComm_t` 句柄使用它。两卡例子中，两位 rank 各持有自己的本地 comm 句柄，它们属于同一个小组。只加入这个小组的两张卡贡献数据，旁边未加入的第三张卡不参与。

### 2.3 GPU buffer：显存里的一段数组空间

buffer 不是什么特殊通信容器，就是**一段数组空间**。例如 GPU 上四个 `float` 占 `4×4=16` 字节；`float* send` 指向输入 `[1,2,3,4]`，`float* recv` 指向用来接收 `[11,22,33,44]` 的空间。输入输出可以分开；满足操作的原地规则时也可以复用空间，下一章再画布局。

### 2.4 Stream：本 GPU 上任务的执行顺序

stream 是本设备的有序任务队列，不是网络连接，也不是 rank。**kernel 就是在 GPU 上执行的函数**，例如逐元素生成输入或使用结果的函数。把相关工作放在同一 stream：

```text
rank 0 的 GPU: 生成 [1,2,3,4] 的 kernel -> NCCL 通信 -> 使用结果的 kernel
rank 1 的 GPU: 生成 [10,20,30,40]      -> NCCL 通信 -> 使用结果的 kernel
```

每条箭头表达本 GPU 的先后依赖，不是两张卡共用一条队列。CPU 提交了通信，不等于 GPU 已完成通信；跨 stream 须用事件等建立正确依赖，详见[第 03 章](03-cuda-semantics.md)。

> **停一下自己答 2：**CPU 的 NCCL 调用返回后，能立刻把输出当成已完成的数据读吗？
>
> **答案：**不能。提交不等于完成；同 stream 的后续 GPU 工作会按序使用结果，CPU 读取则需要明确等待完成。

### 2.5 Collective：小组共同遵守的一次操作

collective（集合通信）规定“小组成员一起做什么”。上例必须由两个 rank 都提交 AllReduce，不能只让 rank 0 调用、期望 NCCL 替 rank 1 猜测输入。各 rank 应以一致顺序提交，元素数量、数据类型和规约规则等也要匹配。

> **停一下自己答 3：**rank 0 做 AllReduce，rank 1 做数组拼接，能因为输入都是四个数就配合成功吗？
>
> **答案：**不能。长度相同不代表操作契约相同；双方必须约定同一种 collective。

### 2.6 主线：rank、设备与进程怎样对应

#### CUDA device：当前进程看到的设备编号

`cudaSetDevice(0)` 的 0 是进程可见的 CUDA ordinal，不保证是机器物理标签上的“GPU 0”。例如：

```bash
CUDA_VISIBLE_DEVICES=3,1 ./nccl-tutorial/examples/build/single_process_allreduce 2
```

进程里的 device 0 对应原列表的 GPU 3，device 1 对应 GPU 1。物理身份可用 PCI bus ID/UUID 核对。

#### Process 与 thread：谁提交调用

process 是运行中的程序实例，thread 是进程内执行调用的线程。常见两种组织：

```text
单进程多卡                         多进程，每进程一卡
process A                           process A -> GPU A -> rank 0
  comm[0] -> GPU A -> rank 0         process B -> GPU B -> rank 1
  comm[1] -> GPU B -> rank 1         process C -> GPU C -> rank 2
  comm[2] -> GPU C -> rank 2         process D -> GPU D -> rank 3
  comm[3] -> GPU D -> rank 3
```

rank 数不天然等于进程数。nccl-tests 甚至可以让每个进程有多个线程，每个线程再控制多个 GPU。入门实验采用每个 rank 绑定一个独占 GPU 的普通布局；不要把同一通信小组的多个 rank 随意放到同一 GPU 上，假设它们一定能正常前进。

#### Communicator 内部与一个具体映射

同一通信小组的本地 communicator 合在一起称为 clique。每个句柄内部关联本 rank、总 rank 数、设备、拓扑、连接、调度和错误状态等。跨进程交换的是 bootstrap（启动引导）信息、连接句柄等可传递信息，**不是把一个 C++ 指针广播后共同解引用**。

例如用 MPI（一种进程通信工具，此处只借它编号）启动两台机器、每台两张 GPU，安排每个进程控制一卡：

| 机器 | MPI global rank | local rank | CUDA device | NCCL rank |
|---|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 0 |
| A | 1 | 1 | 1 | 1 |
| B | 2 | 0 | 0 | 2 |
| B | 3 | 1 | 1 | 3 |

这里 global rank 是跨机器的进程编号，local rank 是本机内的进程编号。`cudaSetDevice(global_rank)` 在机器 B 会错误地选择 device 2/3。这个布局下设备按 local rank 映射，NCCL rank 采用 global rank；这不是两类编号必须相等的普遍规则。

rank 还决定 AllGather 的排列、ReduceScatter 的分片归属，以及 root（指定的结果接收者或数据源）和 peer（通信对端）的含义。例如 `root=2` 指小组的 rank 2，不是本机 device 2。

## 3. NCCL 的五层问题

**先区分“语义 vs 实现”。**两卡都得到 `[11,22,33,44]` 是语义；按什么次序传数据、经过哪条互连是实现。换一条合法路径，不能把求和结果改成拼接结果。

**下面五层是后续源码阅读的主坐标。** 与追驱动调用链类似，先分清每层作出的决策，不要把同名概念当作同一对象。例如 Ring（环形交换）和 Tree（树形交换）可以用不同组织方式实现同一个 AllReduce：

| 层 | 回答什么 | 例子 |
|---|---|---|
| API / 语义 | 最终每个 rank 得到什么 | AllReduce、AllGather、Send/Recv |
| Algorithm | 哪些 rank 以什么顺序交换哪些块 | Ring、Tree、PAT、NVLS 相关算法 |
| Protocol | 一次传输如何分块、标记有效与推进 | Simple、LL、LL128 |
| Transport | 两个参与者之间如何建立和使用通路 | P2P、SHM、NET |
| Hardware | 字节实际经过什么资源 | HBM、SM、PCIe、NVLink、NIC、网络 |

这些不是同义词：

- Ring 可以跨 NVLink，也可以经过网络，不能说“Ring 就是网卡通信”。
- Simple 不是 TCP 的别名，也不是“只使用最简单算法”。
- P2P transport 不等于 API `ncclSend`；AllReduce 内部同样可以走 GPU P2P。
- `NCCL_ALGO` 与 `NCCL_PROTO` 调的是不同层。某些组合不支持，不能任意组合。
- 拓扑决定可选路线与成本，算法决定怎么组织通信，协议决定如何稳定高效地推进数据。

## 4. 由谁完成：CPU、GPU、驱动、网卡的接力

**主线：把熟悉的驱动职责放回整个通信栈。**例如两张 GPU 分处两台机器时，NIC（网卡）可参与跨机传输。下面是普通 host API（由 CPU 调用的接口）网络路径的示意，不表示单机通信也必须经过网卡；重点关注 CUDA 提交之后还由谁持续推动进度：

```text
应用 CPU: ncclAllReduce(send, recv, ..., stream)
  |
NCCL host: 检查与排队 -> 算法/协议/资源选择 -> 生成执行计划
  |                                      |
CUDA runtime / user-mode driver           CPU proxy / 网络插件
  |                                      |
kernel-mode driver + GPU 调度             NIC 队列与完成事件
  |                                      |
GPU kernel: 读、规约、转发 <---互连---> GPU memory / NIC DMA
```

### 分工而不是“全由某一层完成”

| 执行者 | 典型职责 | 不应归给它的职责 |
|---|---|---|
| NCCL host 用户态代码 | rank 协调、拓扑/连接、算法选择、plan、proxy 进度 | 不是 OS 页表和驱动安全边界的最终管理者 |
| CUDA runtime / UMD（用户态驱动，user-mode driver） | 设备上下文、stream、内存与 kernel launch API | 不替应用决定 AllReduce 的数学含义 |
| KMD（内核态驱动，kernel-mode driver） | 配合管理设备资源、映射、权限及提交路径 | 不逐元素执行 NCCL 的求和 |
| GPU kernel / SM（GPU 计算单元） | 常规路径中的数据搬运、规约、协议同步和转发 | 不代表所有高级路径都必须占用同样的 SM |
| GPU/NIC FW（固件，firmware）与硬件 | 在已建立资源和命令下执行调度、事务、DMA/互连传输等 | 不自行推断应用下一次 collective 应是什么 |
| CPU proxy / net plugin | 部分 transport 的提交、完成检测、缓冲推进 | GDR 下通常不必经 CPU 复制每个 payload 字节 |

例如走 GPUDirect RDMA（GDR）时，网卡可直接搬运 GPU 内存中的 payload（实际数组数据），但 CPU proxy 仍可能负责提交和检查进度。“数据不经 CPU 复制”不等于“CPU 不参与”。

这些边界会随 transport 和新路径变化。例如支持的 NVSwitch 归约能力、CE 路径或 device-initiated networking 会改变数据和控制如何推进；基础模型不能因此变成“所有事情都是 DMA”或“CPU 完全不参与”。

## 5. 一次 AllReduce 的完整故事

把两卡例子串起来，先标出“本地准备”和“跨 rank 前进”的边界；第 4、5 步将在后续主线展开：

1. **准备内存。** 在各自 GPU 上分配四元素的输入输出 buffer；把 `[1,2,3,4]` 和 `[10,20,30,40]` 准备好，生成输入的工作排在通信之前。
2. **形成小组。** 创建包含这两个 rank 的 communicator，就设备与连接信息建立必要共识；部分连接可延迟到首次使用。
3. **提交一致的操作。** 两边都提交 AllReduce：`count=4`（元素数量）、`type=ncclFloat`（float 类型）、`op=ncclSum`（求和），各自传入 comm 和 stream。具体调用见下一章。
4. **形成执行计划。** NCCL 根据消息、拓扑和能力选择路径，安排 channel（并行通信通道）、工作描述和必要的 proxy 操作。
5. **推进数据。** GPU、CPU proxy、网络或交换硬件按选定路径协同，完成分块规约/转发和结果写入；不必由 CPU 收齐数组再求和。
6. **确认可消费。** 每卡的输出成为 `[11,22,33,44]`。同 stream 的后续计算按序使用它；CPU 要读结果须等待明确的完成条件。
7. **回收资源。** 所有使用结束后释放用户缓冲、stream 和 communicator，故障则走有协调的异常路径。

学习源码时反复问：**这一步是准备关系，还是传输 payload？它返回时代表哪种“完成”？** 很多误解就在把两个阶段混为一谈。

## 6. NCCL 与几个邻居

**第二遍选读。**例如训练框架发起通信、NCCL 执行通信、CUDA 提供 GPU 执行环境，各有分工：

- **CUDA**：GPU 编程与执行平台；NCCL 在其上提供 GPU 通信机制，不是 CUDA 的替代品。
- **MPI**：有丰富的进程/通信模型，也可用于 NCCL 的启动和引导。使用 MPI 启动并不意味着 NCCL payload 走 MPI。
- **PyTorch 等框架**：决定哪些张量何时通信、bucket 如何划分、如何重叠；NCCL 实现被调用的底层通信。
- **RCCL**：AMD 平台上类似的集合通信库，许多算法直觉相通，但 CUDA、驱动、互连和具体版本行为不可直接照搬。
- **网络插件**：NCCL 与具体网络数据通路的适配层，不是训练框架，也不等同于 Linux 网卡驱动。

## 源码锚点

**主线核查：用公共契约校准模型，再定位实现入口。** 此处只看下面三组符号，完整提交路径留到第 08 章。

- [公共 API](../nccl/src/nccl.h.in)：`nccl/src/nccl.h.in:286`，rank 与 device；`:537`，集合操作提交语义。
- [AllReduce 入口](../nccl/src/collectives.cc)：`nccl/src/collectives.cc:192`，`ncclAllReduceConfigImpl`。
- [算法与协议名称](../nccl/src/init.cc)：`nccl/src/init.cc:57`，`ncclAlgoStr`；`:59`，`ncclProtoStr`。

## 自测与答案

进入下一章前，用以下三题核对 rank 身份和控制/数据通道的边界。

1. 四个进程就一定是四个 NCCL rank 吗？**不一定，取决于每个进程管理多少参与设备/rank。**
2. root=0 指机器上的第 0 张 GPU 吗？**指该 communicator 的 rank 0。**
3. 看到了 bootstrap Socket 日志，就能说 AllReduce 用 TCP 吗？**不能，引导通道和大块数据通道要分开确认。**

下一章：[集合通信的数学与内存布局](02-collectives.md)。
