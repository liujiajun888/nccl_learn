# 04｜集合算法：一块数据怎样汇合所有贡献

[上一章](../03-device-execution.md) 解释了 primitive 如何搬运、等待和发布一片数据。
本章把这些动作连起来，回答 Ring 为什么能得到 AllReduce，以及 Tree 改变了什么。
源码基线为 NCCL **2.32.3**，`12df1a11`。

贯穿案例仍是两主机各四 GPU、八进程各一 rank，每 rank 64 MiB FP32 Sum AllReduce，
`count=16*1024*1024`。我们跟踪普通 host collective 的 **Ring/Simple 分支**，调优也可以选择其他方案。

## 1. 缩小数据，保留完整的依赖关系

为了手算，暂把八 rank 缩成四 rank，把每 rank 的 64 MiB 缩成八个元素。
数组分成 C0、C1、C2、C3，每块两个元素；只算一个 channel 的一波四块。
这是数学模型，实际设备分块还包含对齐、多个 channel 和多波处理。

```text
        C0        C1        C2        C3
R0:   [0,1]     [2,3]     [4,5]     [6,7]
R1: [10,11]   [12,13]   [14,15]   [16,17]
R2: [20,21]   [22,23]   [24,25]   [26,27]
R3: [30,31]   [32,33]   [34,35]   [36,37]
```

Chunk 编号表示数组位置：四个 rank 都有自己的 C0，规约的是同一位置的四份贡献。
本例小整数可由 FP32 精确表示，便于直接校验每次加法。

教学环固定为下面的次序：

```text
R0 -> R1 -> R2 -> R3 -> R0
   每个 rank 从前驱收，向后继发
```

这里用户 rank 与环位置恰好相同。
真实 `runRing` 使用 `ring.index`，前驱和后继来自当前 channel 的拓扑结果。
换一条环时，应先按环位置推导，再映射回用户 rank；数组元素的位置保持不变。

## 2. 跟住 C0：先规约分散，再传播完整结果

Ring AllReduce 分成 reduce-scatter（RS）与 allgather（AG）两个数学阶段。
RS 让每个 rank 先得到一个完整规约块，AG 再把这些块送到所有 rank。
本节采用与 `runRing` 对应的起点：环位置 r 首先发送 `C[(r-1) mod P]`。
其中 mod 取非负余数，R0 因而先发 C3，环首尾使用同一套编号规则。

四 rank 的三轮 RS 发送安排如下；每行表示所有边同时推进的一轮：

| 轮次 | R0 → R1 | R1 → R2 | R2 → R3 | R3 → R0 |
| --- | --- | --- | --- | --- |
| RS 0 | C3 | C0 | C1 | C2 |
| RS 1 | C2 | C3 | C0 | C1 |
| RS 2 | C1 | C2 | C3 | C0 |

只追 C0，就能看到贡献如何逐个加入：

| 接收位置 | 收到的值 | 加入本地 C0 后 | 已含贡献 |
| --- | --- | --- | --- |
| RS 0，R2 | R1 的 `[10,11]` | `[30,32]` | `{1,2}` |
| RS 1，R3 | `[30,32]` | `[60,63]` | `{1,2,3}` |
| RS 2，R0 | `[60,63]` | `[60,64]` | `{0,1,2,3}` |

第一跳发送原始输入，后面的跳发送刚形成的部分和。
每个接收者都加入自己同一数组区间的输入，贡献集合每跳增加一个 rank。
经过 `P-1=3` 条边，C0 恰好包含四份贡献，完整结果落在 R0。

另外三块沿同样的模式推进，RS 结束时：

```text
R0 持有 Y0 = [60,64]    R1 持有 Y1 = [68,72]
R2 持有 Y2 = [76,80]    R3 持有 Y3 = [84,88]
```

这就是正确性不变量：部分和中的 rank 集合持续扩大，新增贡献与已有集合不相交。
四块交错占用链路，让每条边都能推进自己的数据，而非等 C0 结束后再处理 C1。
RS 0 中，R2 接收 C0，也向 R3 发送 C1；收发可以重叠，是因为它们处理不同的块。实现用数组偏移把两项工作分开，协议则分别确认各连接上的读写条件。

AG 继续沿原方向传递完整结果，接收后只复制，不再加入本地输入。
C0 的完整结果依次传播：

```text
AG 0：R0 -> R1，[60,64]
AG 1：R1 -> R2，[60,64]
AG 2：R2 -> R3，[60,64]
```

每轮每个 rank 都新增一个完整块。再经过 `P-1=3` 轮，所有 rank 都收齐四块。
写入位置由 chunk 对应的数组偏移决定，最终输出统一为：

```text
[60,64,68,72,76,80,84,88]
```

恢复到八 rank 时，每个阶段各有七轮逻辑传递。
恢复到 64 MiB 时，设备在各 channel 上分波重复这些依赖，沿途再以 slice 流水推进。
逻辑轮数用于证明数据关系，运行中各连接按自己的就绪状态前进。

## 3. 两个数学阶段，融合在同一个设备工作体里

对可等分的数据，按公开 API 的 rank 块序组织中间结果，有：

```text
AllReduce = AllGather ∘ ReduceScatter
```

独立调用时，ReduceScatter 的每 rank 接收数量为原 count 的 `1/P`，
AllGather 再把各 rank 的结果按块序拼回完整数组。
Ring AllReduce 的设备实现直接安排这两阶段，不需要在中间退出 kernel 或再次进入 host API。

`all_reduce.h` 中 `runRing` 每波的执行顺序是：

```text
directSend                         先把一块原始输入推出去
  -> directRecvReduceDirectSend    中间跳：收 + 加入本地贡献 + 转发
  -> directRecvReduceCopyDirectSend
                                   最后规约 + 保存完整块 + 开始传播
  -> directRecvCopyDirectSend      传播完整块并保存
  -> directRecv                    接收最后一个完整块
```

关键衔接在第 64 行的 `directRecvReduceCopyDirectSend(..., postOp=true)`。
例如 R0 收到 C0 的 `[60,63]`，与本地 `[0,1]` 规约，
把 `[60,64]` 写入自己的输出，同时送往 R1。
这一个 primitive 连接了 RS 的最后接收与 AG 的首次发送。

融合让收到的数据在一次处理里完成规约和多目的写入，减少中间搬运与控制开销。
表里的两个数学阶段仍然成立，只是实现把边界处的动作重叠起来。
一般 FP32 输入会受加法结合顺序影响，不同算法的最低几位可能不同。

仓库的 CPU 模拟器选择了另一种等价起点：Rr 首先发送 Cr，
所以它的 RS 结束后，Rr 持有 `C[(r+1) mod P]`。
源码这段 `runRing` 则让环位置 r 持有 Cr；两者的 AG 都按原数组位置写回相同答案。
这些是 AllReduce 内部的块归属约定；公开 `ncclReduceScatter` 仍要求用户 rank r 接收第 r 块。

## 4. 为什么 Ring 的带宽系数是 2(P−1)/P

令每 rank 的完整输入为 S 字节，数学上等分为 P 块，每块 `S/P`。
RS 和 AG 各发送 `P-1` 次，所以每 rank 的总发送量为：

```text
RS：   (P-1) * S/P
AG：   (P-1) * S/P
合计： 2(P-1)/P * S
```

接收量与发送量相同。理想全双工、收发可重叠时，用单向有效带宽 B 估计带宽时间，
得到 `2(P-1)S/(P*B)`；若每轮还有不可隐藏的开销 α，则：

```text
T_ring ≈ 2(P-1)α + 2(P-1)S/(P*B)
```

四 rank 的系数为 1.5；八 rank 的系数为 1.75。
贯穿案例中，每 rank 理想发送 **112 MiB**，接收也为 **112 MiB**。
这是整个环算法的逻辑通信量，跨机网卡实际承担多少，还要看环的节点映射和共享链路。

P 增大时，带宽系数趋近 2，串行依赖的轮数却持续增长。
这解释了 Ring 搬大数组时容易摊薄开销，小消息、大规模时则更需要缩短依赖。
多 channel 可以并行利用资源，有效 B 仍受共同瓶颈限制；这个公式是解释模型，不是完整调优器。

## 5. Tree 缩短依赖，协议和传输保持独立

树形规约让子树的部分结果在父节点汇合，根得到最终结果后向下广播。
同一 chunk 可以在不同子树同时规约；平衡结构把逐 rank 的长链改成按树层推进。
节点间的层数随节点数近似对数增长，减少小消息传播中的串行等待。

NCCL 使用双树分担数据和内部节点负载。
`ncclGetDtree` 先构造一棵树，再用偶数参与者镜像、奇数参与者移位生成另一棵。
`connectTrees` 以节点为参与者，把两棵树接到两组 channel，并衔接节点内 GPU 路径。
不同数据分区走不同树，使规约与转发压力分散到不同位置。
每部分数组都在所属树上完成规约和广播，因此各 rank 最后仍收齐整份结果。双树分的是数据和转发负担，算法语义仍是所有参与者对同一位置共同求和。
本例只有两个节点，实际依赖深度还包括各节点内四 GPU 的连接。

Tree 同样按 chunk 流水：较早的块可以向下广播，较晚的块仍在向上规约。
`runTreeSplit` 把非根 CTA 的线程分成上行规约组与下行广播组；
根节点则用融合 primitive 完成最后规约、写输出和向下发送。
缩短依赖与提高带宽可以同时进行，效果取决于拓扑映射、消息大小和线程分配。

把执行方案分成三个轴，修改代码时就更容易定位职责：

- **Algorithm**：Ring/Tree 决定一块数据经过谁、何时规约或复制。
- **Protocol**：Simple/LL/LL128 决定怎样切片、表示就绪状态、管理 credit。
- **Transport**：P2P/SHM/NET 为每条 peer 连接提供具体通路。

例如同一个 Ring/Simple 中，节点内边可走 P2P，跨节点边走 NET。
三轴可以组合，合法组合由实现与平台条件共同约束。
普通设备代码的生成范围可先记住这两行：

| 算法 | 集合操作 | 协议范围 |
| --- | --- | --- |
| Ring | Broadcast、Reduce、AllGather、ReduceScatter、AllReduce | Simple、LL、LL128 |
| Tree | AllReduce | Simple、LL、LL128 |

其他入口中，NVLS、CollNet 引入不同层级或集合能力，PAT 面向 AG/RS；
它们以及 symmetric/CE 路径统一放在 [现代特性附录](modern-features.md)。
调优先筛选有效候选，再比较估计成本，并可接受 tuner 插件的选择。
开发时用 GRAPH 记录核对连线，用任务的 algorithm/protocol 和 channel 分区核对实际方案；
本基线普通调度点 `enqueue.cc:912` 有 INFO/TUNING 记录，连线日志本身只描述拓扑。

## 6. 源码入口与 CPU 模拟

- [all_reduce.h:14](../../nccl/src/device/all_reduce.h#L14)：`runRing`；64 行看阶段融合，146 行看 `runTreeSplit`。
- [device.h:178](../../nccl/src/include/device.h#L178)：`ncclRing`，区分用户 rank、环位置与邻居。
- [trees.cc:90](../../nccl/src/graph/trees.cc#L90)：`ncclGetDtree`；节点到 channel 的连接在 [connect.cc:138](../../nccl/src/graph/connect.cc#L138) 的 `connectTrees`。
- [generate.py:104](../../nccl/src/device/generate.py#L104)：`algos_of_coll`；130 行的 `required_cuda` 继续筛选组合。
- [tuning.cc:180](../../nccl/src/tuning/tuning.cc#L180)：`ncclTuningCompute`，候选估计、插件与选择。
- [ring_simulator.py:6](../examples/ring_simulator.py#L6)：`ring_allreduce`，用贡献集合检查每跳是否重复计入。

**离线任务：运行原有模拟器，并解释它与正文的块归属差异。**
无需 GPU，在项目根目录运行：

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

可核对结论：四 rank 输出都是 `[60,64,68,72,76,80,84,88]`，
共六轮，每 rank 发送十二个元素；自测输出 **PASS: 85 cases**。
模拟器的 C0 从 R0 经 R1、R2 到 R3，RS 后归 R3；正文的 C0 从 R1 起发，最后归 R0。
两种时序都使每份贡献恰好出现一次，再经 AG 得到同一完整结果。

下一章 [08｜Proxy 与网络](../08-proxy-and-network.md) 沿跨机那条边，继续追 GPU、proxy 与网络请求的交接。
