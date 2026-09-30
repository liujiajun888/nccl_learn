# 第 04 章：集合算法——从四张卡手算到真实调优

> 基线：NCCL `12df1a11`，版本 `2.32.3`。
> 本章是源码阅读与纸面推演；编写环境为无 GPU 的 macOS，未运行 CUDA/NCCL 性能实验。
> 前置：[集合语义](02-collectives.md)。推荐在 [03 的提交与完成](03-cuda-semantics.md)之后，先读本章的算法核心，再进入初始化与资源章节。

**本章核心是 4.2～4.7：**先逐轮推演 Ring，再比较 Tree，建立“每个参与者处理哪片数据”的直觉。源码映射用来核对这张图，不要求先掌握拓扑搜索或调优实现。4.8～4.10 的支持矩阵与特殊算法放在折叠区，按需展开；完成自测后直接进入第 05 章。

## 4.1 学习目标：分清三个问题

学完本章，应能解释：

1. AllReduce 的答案是什么，Ring 为何能在每张卡上得到同一答案。
2. 同一数学操作为何有 Ring、Tree、NVLS 等不同通信安排。
3. 为什么“存在设备实现”“本机可用”“调优器选中”是三件事。

开始前，先把算法和相邻两层分开：

- **算法**回答“哪些数据以什么顺序经过哪些参与者”。
- **协议**回答“如何表示数据、通知对端、等待可用空间”，在第 09 章展开。
- **传输**回答“这条逻辑连接由 P2P、SHM、NET 等什么机制承载”。

把三层合为一个词，就容易误认为 Ring 必须用某种物理链路。

## 4.2 先固定 rank、环位置、chunk 的含义

本节固定后面所有推演的记号。设有 `P=4` 个 rank，输入数组均含 8 个元素，操作为逐元素求和。`Rr` 表示用户 rank r；`Cc` 表示数组中的第 c 块，每块 2 个元素。**chunk（分块）**的编号表示**数组位置**，绝不表示“来自哪张卡”。每个 rank 一开始都拥有 C0、C1、C2、C3 四块自己的输入。

```text
          C0       C1       C2       C3
R0:     [1,2]    [3,4]    [5,6]    [7,8]
R1:   [10,20]  [30,40]  [50,60]  [70,80]
R2: [100,200][300,400][500,600][700,800]
R3:[1000,2000][3000,4000][5000,6000][7000,8000]
```

本例**特意选择**以下逻辑环，使环位置与用户 rank 暂时相等：

```text
R0 ----> R1 ----> R2 ----> R3
 ^                         |
 +-------------------------+
每个 rank 只从前驱接收，向后继发送。
```

**注意：**真实 NCCL 环不是天然 `0→1→2→3`。拓扑搜索可能给某个 channel 排成 `0→2→3→1`，另一个 channel 又不同。所以要先用环位置推导传输，再用映射还原用户 rank，不能直接拿 rank 做模运算。

<details><summary>深入：环位置映射的源码证据</summary>

- [src/include/device.h:178，`ncclRing`](../nccl/src/include/device.h#L178) 中有 `prev/next/userRanks/rankToIndex/index`。
- 设备入口 [src/device/all_reduce.h:14，`runRing`](../nccl/src/device/all_reduce.h#L14) 使用的是 `ring->index`。

</details>

## 4.3 Ring 的第一半：reduce-scatter

**reduce-scatter（规约分散）**的目标不是立即让所有卡拿到完整答案，而是让每卡先拥有一个**完整规约块**。本例的最终所有权约定为：R0 拿 C0 的和，R1 拿 C1 的和，以此类推。为了与源码的初始发送对应，第一轮 Rr 发送 `C[(r-1) mod 4]`。所有轮次都同时发送和接收；表中一行表示全环的一轮，不是串行执行四次。

用 `X[r,c]` 表示 Rr 的原始 Cc，`A[c;{...}]` 表示已累加指定 rank 的 Cc。下表**只列发送的 chunk 编号**；每列的接收块就是左侧前驱所发送的块。

| 阶段、轮次 | R0→R1 | R1→R2 | R2→R3 | R3→R0 |
|---|---|---|---|---|
| RS 第 0 轮 | C3 | C0 | C1 | C2 |
| RS 第 1 轮 | C2 | C3 | C0 | C1 |
| RS 第 2 轮 | C1 | C2 | C3 | C0 |

统一公式：RS 第 t 轮，发送块 `c=(r-1-t) mod P`，接收块 `c=(r-2-t) mod P`。这里 `t=0,1,2`；数学的 mod 总取非负余数，不照搬 C/C++ 的负数 `%`。接收者将收到的部分和与**自己同一块的原始输入**相加。

### 第 0 轮：每个到达的块含两份贡献

```text
R0 收 C2：X[3,2] + X[0,2] = [5000,6000]+[5,6]   = [5005,6006]
R1 收 C3：X[0,3] + X[1,3] = [7,8]+[70,80]       = [77,88]
R2 收 C0：X[1,0] + X[2,0] = [10,20]+[100,200]   = [110,220]
R3 收 C1：X[2,1] + X[3,1] = [300,400]+[3000,4000]= [3300,4400]
```

注意，R0 自己的 C0 并未在这一轮获得任何新贡献：“rank 0 收到了数据”不等于“chunk 0 收到了数据”。这一区分是检查所有 Ring 手算表的第一道关。

### 第 1 轮：转发刚形成的部分和

R0 发 C2，R1 发 C3，R2 发 C0，R3 发 C1。发送内容不再只是本地原始数据，而是上一轮形成的两份贡献之和。

```text
R0 收 C1：A[1;{2,3}] + X[0,1] = [3300,4400]+[3,4]   = [3303,4404]
R1 收 C2：A[2;{3,0}] + X[1,2] = [5005,6006]+[50,60] = [5055,6066]
R2 收 C3：A[3;{0,1}] + X[2,3] = [77,88]+[700,800]   = [777,888]
R3 收 C0：A[0;{1,2}] + X[3,0] = [110,220]+[1000,2000]= [1110,2220]
```

为什么不能把 R0 当前所有块都发出去？因为环算法正在用不同块交错占用链路，每轮每卡只需发一个块。把所有块都发出去虽然可能另有正确算法，却已经不是这里的通信量与时间模型。

### 第 2 轮：最后一个贡献补齐

```text
R0 收 C0：[1110,2220]+[1,2]     = [1111,2222]
R1 收 C1：[3303,4404]+[30,40]   = [3333,4444]
R2 收 C2：[5055,6066]+[500,600] = [5555,6666]
R3 收 C3：[777,888]+[7000,8000] = [7777,8888]
```

现在定义完整规约块：`Y0=[1111,2222]`，`Y1=[3333,4444]`，`Y2=[5555,6666]`，`Y3=[7777,8888]`。Rr 已经拥有 Yr，但还没有其他三个完整块，故此时不是 AllReduce 的最终状态。

### 一个足以证明正确性的不变量

RS 第 t 轮**发送前**，Rr 发的 Cc 已包含 `r,r-1,...,r-t` 共 `t+1` 个 rank 的贡献。接收者 R(r+1) 加入自己的贡献，既不遗漏，也不重复。换到接收视角：第 t 轮结束，Rr 的 `C[(r-2-t) mod P]` 含 `t+2` 个连续环位置的贡献。到 `t=P-2`，该块编号为 r，贡献数为 P，所以 Rr 得到 Yr。本例采用整数避免舍入干扰；浮点加法改变结合顺序后，低位可能与其他算法不同。

## 4.4 Ring 的第二半：allgather

**allgather（全收集）**只做复制：把各卡已有的完整块转发给所有人，**不再求和**。第 t 轮 Rr 发送 `Y[(r-t) mod P]`，接收 `Y[(r-1-t) mod P]`。继续沿同一个方向转发，不反转环，也不重新编号数组。

| 阶段、轮次 | R0→R1 | R1→R2 | R2→R3 | R3→R0 |
|---|---|---|---|---|
| AG 第 0 轮 | Y0 | Y1 | Y2 | Y3 |
| AG 第 1 轮 | Y3 | Y0 | Y1 | Y2 |
| AG 第 2 轮 | Y2 | Y3 | Y0 | Y1 |

逐轮记录各卡**新增的**完整块，可同时校验数值与索引：

```text
AG0: R0 收 Y3=[7777,8888]；R1 收 Y0=[1111,2222]
     R2 收 Y1=[3333,4444]；R3 收 Y2=[5555,6666]
AG1: R0 收 Y2=[5555,6666]；R1 收 Y3=[7777,8888]
     R2 收 Y0=[1111,2222]；R3 收 Y1=[3333,4444]
AG2: R0 收 Y1=[3333,4444]；R1 收 Y2=[5555,6666]
     R2 收 Y3=[7777,8888]；R3 收 Y0=[1111,2222]
```

AG 第 t 轮后，Rr 拥有 `Yr,Y(r-1),...,Y(r-t-1)`，共有 `t+2` 个完整块。三轮后每卡都有四块，最终数组必须按 **C0、C1、C2、C3 的数组顺序**摆放：

```text
[1111,2222,3333,4444,5555,6666,7777,8888]
```

**注意：**“到达顺序”是环上的时间顺序，“输出顺序”是 API 的数据布局，二者不能混淆。独立 ReduceScatter/AllGather API 还必须遵守用户 rank 的分块语义；内部环映射负责衔接。

## 4.5 两个数学阶段，不一定是两次 API

对等长分块和相容规约语义，有数学关系：

```text
AllReduce(X) = AllGather(ReduceScatter(X))
```

这不是在说 NCCL 必须先调用一次 `ncclReduceScatter`，再调用一次 `ncclAllGather`。源码 [src/device/all_reduce.h:42–81，`runRing`](../nccl/src/device/all_reduce.h#L42) 直接在一个设备工作体中安排：

```text
send -> recv+reduce+send -> 最后规约+写输出+send
                          -> recv+copy+send -> 最后 recv
```

第 64 行 `directRecvReduceCopyDirectSend(..., postOp=true)` 把 RS 的末次接收与 AG 的首次发送衔接起来。

**注意：**表中的“逻辑通信轮”不等于一条 primitive 调用，更不等于一次 kernel launch。若框架只需要分片梯度，可在数学上停在 ReduceScatter，不必为了套 AllReduce 而做无用 AllGather；这与[训练集成](16-training-integration.md)中的参数/梯度分片直接相关。

## 4.6 通信量和时间近似从哪里来

设每卡完整输入大小为 S 字节，各块等长 `S/P`。RS 有 `P-1` 轮，AG 有 `P-1` 轮；每轮每卡发送 `S/P`，同时接收同样大小。所以每卡总发送量为 `2(P-1)S/P`，总接收量也相同。在收发可重叠的理想全双工模型中，不应再把二者相加后套同一个 B。

定义 alpha 为一轮不可隐藏的启动/传播开销，B 为该简化模型中的有效单向带宽：

```text
T_ring ≈ 2(P-1) alpha + [2(P-1)/P] S/B
```

这里忽略规约计算、协议标志、对齐与尾块、内存访问、拥塞和多 channel 启动等成本。它也不是 NCCL 调优器逐字采用的完整公式。例如 `P=4, S=4 MB, B=40 GB/s, alpha=2 us`，使用十进制字节单位：启动项 `6×2=12 us`，带宽项 `1.5×4 MB / 40 GB/s=150 us`，合计约 `162 us`。这只是代数例子，不是任何 GPU 的实测数值。

当 P 增大，带宽项系数趋近 2，但启动项线性增长。这解释了 Ring 对大消息很有吸引力，却可能对“小消息、大规模”不够友好。实际分成多个 channel 后，B 不是把物理峰值机械乘以 channel 数；共享瓶颈仍然存在。源文件 [src/tuning/ring.cc:21，`ncclTuningRingModelInit`](../nccl/src/tuning/ring.cc#L21) 应与本模型对照读，而非相互替代。

## 4.7 Tree：缩短依赖深度，而不是放弃带宽

树把叶子的部分结果向上规约，再把结果向下传播。平衡树的依赖层数约为 `log2(P)`，不像 Ring 必须沿 P 个位置逐步传播。多节点实现还包含节点内结构，所以不能把所有运行时间都写成纯 `2 log2(P) alpha`。

```text
一棵四参与者树          互补的另一棵树
      0                       3
      |                       |
      2                       1
     / \                     / \
    1   3                   2   0
向上：reduce；向下：broadcast；两树承担不同的数据分区。
```

这与 [src/graph/trees.cc:90，`ncclGetDtree`](../nccl/src/graph/trees.cc#L90) 的四参与者构造对应：偶数参与者用镜像，奇数参与者用位移构造第二棵树，避免总让同一些位置承担内部节点负担。实际多节点映射入口见 [src/graph/connect.cc:148，`ncclGetDtree` 调用](../nccl/src/graph/connect.cc#L148)。图中的数字是树构造的参与者编号，不能直接当作任意机器上的 GPU 布线图。

树也可以分 chunk、管线化，并用不同线程组同时处理上行规约与下行广播；[src/device/all_reduce.h:146，`runTreeSplit`](../nccl/src/device/all_reduce.h#L146) 正是在分配两类线程。因而“树延迟低，但带宽必然差于 Ring”不是可靠结论：最终还取决于树的映射、内部节点链路、双树分工、channel 数和消息大小。[src/tuning/tree.cc:28–57，`ncclTuningTreeModelInit`](../nccl/src/tuning/tree.cc#L28) 分别建模带宽与层级延迟。

以上是本章核心。需要检查块的归属时可做[第 15 章 B1/B2](15-exercises.md)；初读可跳过下面的专题折叠区，做完章末自测后继续第 05 章。

**动手验证（需 GPU）：**跑一次小尺寸 AllReduce 并设 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,TUNING`，对照日志中的环/树映射与 4.2 的手算环序；再扫几个尺寸，看选择在哪里变化。手算模型与真实映射对上号，本章才算用起来了。

<details>
<summary>选读：4.8～4.10 支持矩阵、特殊算法与验证方向</summary>

## 4.8 本基线的通用设备算法支持矩阵

下表是**生成集合与算法组合的范围**，不是硬件无条件支持表。依据 [src/device/generate.py:104，`algos_of_coll`](../nccl/src/device/generate.py#L104)、
[src/device/generate.py:130，`required_cuda`](../nccl/src/device/generate.py#L130) 与
[src/tuning/cost_model.cc:230，`modelMap`](../nccl/src/tuning/cost_model.cc#L230) 交叉核对。

| 通用算法 | 集合操作 | 生成的协议范围 |
|---|---|---|
| Ring | Broadcast、Reduce、AllGather、ReduceScatter、AllReduce | Simple、LL、LL128 |
| Tree | AllReduce | Simple、LL、LL128 |
| CollNet Direct | AllGather、ReduceScatter、AllReduce | Simple |
| CollNet Chain | AllReduce | Simple |
| NVLS | AllGather、ReduceScatter、AllReduce | Simple |
| NVLS_TREE | AllReduce | Simple |
| PAT | AllGather、ReduceScatter | Simple |

生成器还列有内部 `AllGatherV/RING` 和 `SendRecv`；不能据此反推完整公共 API 清单。
本基线**有**原生 [ncclAlltoAll:648](../nccl/src/nccl.h.in#L648)、[ncclGather:663](../nccl/src/nccl.h.in#L663)、[ncclScatter:678](../nccl/src/nccl.h.in#L678)。
它们的提交/分解路径与上述五类通用集合算法矩阵不是一一对应；CE 和 symmetric kernels 也在矩阵之外。

## 4.9 NVLS、NVLS_TREE、CollNet、PAT 分别改变了什么

**NVLS** 借助受支持 NVLink/NVSwitch 平台的 multicast、multimem 规约能力，减少 GPU 逐跳软件搬运。
它不是“所有 Ampere 加 NVLink 都支持”：生成规约代码至少要求 CUDA 12.1、SM90，并限制类型/运算。
例如 FP32/FP64 这里只有 Sum，部分整数及 FP16/BF16 有 Sum/MinMax；应看转换后的设备规约操作。
运行时还检查 multicast API、设备属性及拓扑；见 [src/transport/nvls.cc:166，`ncclNvlsInit`](../nccl/src/transport/nvls.cc#L166)。

**NVLS_TREE** 用本地域的 NVLS 与域间树结合，不能简化为“NVLS 换个名字”。
本基线单节点会禁用 NVLS_TREE；多节点 NVLS 本身则检查 CollNet 配置/支持。
AG/RS 还涉及插件 `iallgather/ireducescatter`、head 数和 arity；不要将 AR 的条件复制给它们。
完整筛选见 [src/tuning/nvls.cc:19、127，`ncclTuningNvlsModelInit/Sim`](../nccl/src/tuning/nvls.cc#L19)。

**CollNet** 把网络侧集合能力纳入层级算法。以 AllReduce 为例，head 是本节点代表一部分数据接入集合网络的 GPU；Direct 与 Chain 主要改变节点内怎样把数据送到 head、怎样分回结果。

- **Direct：按 head 分区、直接汇聚。** 可有多个 head；每张 GPU 将不同数据片送给对应 head，head 规约本地各 GPU 的同片 → 各节点对应 head 参与网络 AllReduce → 结果回到 head，再分发给本节点 GPU；每张 GPU 收齐各片。head 自身的输入也参与规约，并把其他分区送到对应 head。
- **Chain：沿节点内链逐跳规约，再反向传播。** 假设某 channel 的本地顺序为 G0（head）、G1、G2：`G2 → G1 → G0` 逐跳合并输入 → 网络 AllReduce → `G0 → G1 → G2` 保存并转发结果。不同 channel 可以选择不同 head，这不是把所有节点的 GPU 串成一条跨机链。

两条路线省略用户 buffer 直接注册等优化；网络 AllReduce 由集合网络插件承接，不是 GPU 把数据交给某个“下一个远端 rank”。对照 [all_reduce.h:248–365 的 Direct](../nccl/src/device/all_reduce.h#L248)、[638–754 的 Chain](../nccl/src/device/all_reduce.h#L638)，以及 [coll_net.cc:815–826 的插件调用](../nccl/src/transport/coll_net.cc#L815)。

CollNet 不是普通 NET 点对点通路的同义词，也不是只要有 IB 网卡就能工作。
本基线先检查集合网络插件和操作支持，并有本地 rank 数等约束。
当前 `maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1` 会使候选无效，须把这当实现条件而非数学定理。
见 [src/tuning/collnet.cc:11、121，`ncclTuningCollnetModelInit/Sim`](../nccl/src/tuning/collnet.cc#L11)。

**PAT** 用并行树式调度与管线减少 AG/RS 的串行传播等待，不是“另一种普遍支持的 AllReduce”。
它不只沿固定 next 转发，而是用 sendDim/recvDim 描述不同通信维度，把若干小步聚合并交给并行 worker。
聚合程度还受数据大小与 FIFO 深度制约，见 [src/include/collectives.h:810–874，`PatAGAlgorithm/getNextOp`](../nccl/src/include/collectives.h#L810)。
[src/device/all_gather.h:143–186，`PatAGAlgorithm/getNextOp/patCopy`](../nccl/src/device/all_gather.h#L143) 把计算调度描述与 worker 执行分开。
[src/tuning/pat.cc:15，`ncclPatEnable`](../nccl/src/tuning/pat.cc#L15) 的自动条件包含 SM60+、网络设备类型等。
多 rank/节点路径已存在：自动门控要求跨节点且使用 NVLS 做节点内阶段，模型还核对每节点 rank 数与 NVLS heads。
多 rank/节点 RS 另受 NVLS 类型/规约支持限制；模拟成本设为 `FLT_MAX/2`，注释明确它目前为 opt-in 路径。
见同文件 [第 80–99 行，`ncclTuningPatModelSim`](../nccl/src/tuning/pat.cc#L80)。
这些是该提交的条件，不应写成“PAT 永远只允许一节点一卡”或“PAT 的总耗时严格对数”。

## 4.10 可观察验证：不要背诵“某算法最快”

调优先排除无效候选，再比较成本；[src/tuning/tuning.cc:128、155、180](../nccl/src/tuning/tuning.cc#L128) 对应枚举、选择与插件介入。
选不中可能因为类型、拓扑、插件、协议或用户筛选，不一定因为估计更慢。
在有 GPU 的 Linux 环境，可结合[测试](10-nccl-tests.md)与[性能分析](11-performance.md)做以下观察：

- 固定消息大小，增加 rank：小消息延迟是否更接近轮次成本？记录实际算法而不只记录时长。
- 固定 rank 扫描大小：默认选路的转折点在哪里？改变算法后正确性误差是否仍在允许范围？
- 检查 `NCCL_DEBUG=INFO` 与 `NCCL_DEBUG_SUBSYS=GRAPH,TUNING,COLL`，能否看到环映射和候选信息？
- 必要时在支持 TRACE 的构建中读最佳 tuning 记录；kernel 名可能是代表入口，不能单靠名字断言协议。
- 想试 PAT，先选 AG/RS 并核对其有效条件；不要对 AllReduce 强制不存在的 PAT 组合。

</details>

## 4.11 自测与答案

**题 1：RS 第 1 轮后，R0 更新的是哪块？包含哪些 rank？**
答：C1，包含 R2、R3、R0，值为 `[3303,4404]`；R0 的编号不决定此轮 chunk 编号。

**题 2：P=4 时为什么带宽项是 `1.5S/B`，不是 `3S/B`？**
答：每卡发六次 S/4，总发送 1.5S；理想全双工模型重叠收发，不把接收量再次串行相加。

**题 3：换成树后带宽更高，是否说明测试错了？**
答：不一定，双树与管线可能更适配拓扑；算法表现还取决于实际映射、资源和消息大小，不能预设 Ring 永远最快。

下一章：[05 Communicator 生命周期](05-communicator.md)，看参与者怎样建立团队与必要的通信资源。
