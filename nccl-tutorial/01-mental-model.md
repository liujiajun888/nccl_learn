# 01 建立全局模型：NCCL 究竟在解决什么问题

## 学习目标

- 用一个四卡例子解释集合通信，而不是先背函数名。
- 分清 rank、CUDA device、进程、communicator、stream。
- 把 API 语义、算法、协议、transport 和硬件放到正确层级。

## 1. 从四张卡共同训练说起

假设四张 GPU 各自处理一批样本，计算出同一个参数的局部梯度：

```text
rank 0: [1, 2]      rank 1: [3, 4]
rank 2: [5, 6]      rank 3: [7, 8]
```

下一步所有 GPU 都需要总梯度 `[16, 20]`，才能采用一致的参数更新。这就是一次 `AllReduce(Sum)`：

```text
四份同样布局的输入 -> 对应位置求和 -> 四张卡都有完整结果
```

注意它不是把同一张 GPU 上 `[1, 2]` 求和成一个数，也不是把四个数组拼接。**对应元素之间的规约**和**结果放到哪些 rank**是两个独立维度。

可以先让 CPU 收齐所有数据、求和后再发回，但这会引入 GPU↔CPU 搬运、中心节点瓶颈和额外等待。NCCL 的价值是利用 GPU 与互连的能力，把搬运、规约、转发组织成高效的并行通信。

## 2. 五个对象，先不要混淆

### 2.1 Rank：通信小组里的逻辑编号

rank 范围是 `0...P-1`。它决定 AllGather 的排列、ReduceScatter 的归属、root 和 peer 的含义。rank 是**相对于某个 communicator**的，同一 GPU 在两个 communicator 中可以有不同 rank。

### 2.2 CUDA device：当前进程看到的设备编号

`cudaSetDevice(0)` 的 0 是进程可见的 CUDA ordinal，不保证是机器物理标签上的“GPU 0”。例如：

```bash
CUDA_VISIBLE_DEVICES=3,1 ./nccl-tutorial/examples/build/single_process_allreduce 2
```

进程里的 device 0 对应原列表的 GPU 3，device 1 对应 GPU 1。物理身份可用 PCI bus ID/UUID 核对。

### 2.3 Process 与 thread：谁提交调用

常见两种组织：

```text
单进程多卡                         多进程，每进程一卡
process A                           process A -> GPU A -> rank 0
  comm[0] -> GPU A -> rank 0         process B -> GPU B -> rank 1
  comm[1] -> GPU B -> rank 1         process C -> GPU C -> rank 2
  comm[2] -> GPU C -> rank 2         process D -> GPU D -> rank 3
  comm[3] -> GPU D -> rank 3
```

rank 数不天然等于进程数。nccl-tests 甚至可以让每个进程有多个线程，每个线程再控制多个 GPU。

入门实验采用每个 communicator rank 绑定一个独占 GPU 的普通布局；不要在同一 clique 内把多个 rank 随意放到同一 GPU 上，假设它们一定能正常前进。特殊支持不应成为默认编程模型。

### 2.4 Communicator：通信关系及资源的上下文

应用看到 `ncclComm_t` 句柄；内部关联本 rank、总 rank 数、设备、拓扑、连接、调度和错误状态等。

一个 collective 的参与者是同一 clique 中的各 rank，每个 rank 持有自己的本地句柄。**不是把一个 C++ 指针广播给所有进程后共同解引用。** 跨进程交换的是 bootstrap 信息、连接句柄等可传递信息。

### 2.4.1 一个具体映射

两台机器，每台两张 GPU：

| 机器 | MPI global rank | local rank | CUDA device | NCCL rank |
|---|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 0 |
| A | 1 | 1 | 1 | 1 |
| B | 2 | 0 | 0 | 2 |
| B | 3 | 1 | 1 | 3 |

`cudaSetDevice(global_rank)` 在机器 B 会错误地选择 device 2/3。设备通常按 local rank 映射，communicator 则使用 global rank。

### 2.5 Stream：GPU 工作的有序提交队列

stream 表达本设备上的执行依赖，不是网络连接，也不是 NCCL rank。正常使用时：

```text
同一 CUDA stream:
生成输入 -> NCCL collective -> 使用输出
```

CPU 提交到中间一步后通常不等于 GPU 已完成它。跨 stream 必须通过事件或其他正确依赖连接，详见[第 03 章](03-cuda-semantics.md)。

## 3. NCCL 的五层问题

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

看普通 host API 网络路径的示意：

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
| CUDA runtime / UMD | 设备上下文、stream、内存与 kernel launch API | 不替应用决定 AllReduce 的数学含义 |
| KMD（内核态驱动） | 配合管理设备资源、映射、权限及提交路径 | 不逐元素执行 NCCL 的求和 |
| GPU kernel / SM | 常规路径中的数据搬运、规约、协议同步和转发 | 不代表所有高级路径都必须占用同样的 SM |
| GPU/NIC 固件与硬件 | 在已建立资源和命令下执行调度、事务、DMA/互连传输等 | 不自行推断应用下一次 collective 应是什么 |
| CPU proxy / net plugin | 部分 transport 的提交、完成检测、缓冲推进 | GDR 下通常不必经 CPU 复制每个 payload 字节 |

这些边界会随 transport 和新路径变化。例如支持的 NVSwitch 归约能力、CE 路径或 device-initiated networking 会改变数据和控制如何推进；基础模型不能因此变成“所有事情都是 DMA”或“CPU 完全不参与”。

## 5. 一次 AllReduce 的完整故事

1. **准备内存。** 应用在正确 GPU 上分配输入输出；此前生成输入的工作必须在 stream 上排在通信之前。
2. **形成小组。** 创建 communicator，各 rank 就设备与连接信息建立必要共识；部分连接可延迟到首次使用。
3. **提交一致的操作。** 所有 rank 按一致顺序提交同一 collective，count/type/op 等语义匹配。
4. **形成执行计划。** NCCL 根据消息、拓扑和能力选择路径，安排 channel、工作描述和必要的 proxy 操作。
5. **推进数据。** GPU、CPU proxy、网络或交换硬件按选定路径协同，完成分块规约/转发和结果写入。
6. **确认可消费。** 下游同 stream 的计算自然在通信后执行；CPU 要读结果须等待明确的完成条件。
7. **回收资源。** 所有使用结束后释放用户缓冲、stream 和 communicator，故障则走有协调的异常路径。

学习源码时反复问：**这一步是准备关系，还是传输 payload？它返回时代表哪种“完成”？** 很多误解就在把两个阶段混为一谈。

## 6. NCCL 与几个邻居

- **CUDA**：GPU 编程与执行平台；NCCL 在其上提供 GPU 通信机制，不是 CUDA 的替代品。
- **MPI**：有丰富的进程/通信模型，也可用于 NCCL 的启动和引导。使用 MPI 启动并不意味着 NCCL payload 走 MPI。
- **PyTorch 等框架**：决定哪些张量何时通信、bucket 如何划分、如何重叠；NCCL 实现被调用的底层通信。
- **RCCL**：AMD 平台上类似的集合通信库，许多算法直觉相通，但 CUDA、驱动、互连和具体版本行为不可直接照搬。
- **网络插件**：NCCL 与具体网络数据通路的适配层，不是训练框架，也不等同于 Linux 网卡驱动。

## 源码锚点

- [公共 API](../nccl/src/nccl.h.in)：`nccl/src/nccl.h.in:286`，rank 与 device；`:537`，集合操作提交语义。
- [AllReduce 入口](../nccl/src/collectives.cc)：`nccl/src/collectives.cc:192`，`ncclAllReduceConfigImpl`。
- [算法与协议名称](../nccl/src/init.cc)：`nccl/src/init.cc:56`，`ncclAlgoStr` / `ncclProtoStr`。

## 自测与答案

1. 四个进程就一定是四个 NCCL rank 吗？**不一定，取决于每个进程管理多少参与设备/rank。**
2. root=0 指机器上的第 0 张 GPU 吗？**指该 communicator 的 rank 0。**
3. 看到了 bootstrap Socket 日志，就能说 AllReduce 用 TCP 吗？**不能，引导通道和大块数据通道要分开确认。**

下一章：[集合通信的数学与内存布局](02-collectives.md)。
