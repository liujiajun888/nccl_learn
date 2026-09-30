# 第 11 章：从曲线、时间线到可解释的性能优化

> 基线：NCCL `12df1a11`（2.32.3），nccl-tests `b4d5bee`（2.20.0）。
> 所有 shell 示例从仓库根目录 `nccl_learn`、在 Linux NVIDIA GPU 节点执行。
> `ROOT` 为用户设置的该节点仓库根目录，未设置时取 `$PWD`；多机要求各节点路径一致。
> 当前编写环境无可用 NVIDIA GPU；下文是实验设计与预期趋势，不是已验证的性能成绩。

## 学习目标

- 用 `T(S) ≈ α + S/B` 判断启动延迟受限还是数据搬运带宽受限。
- 定义可复现的 benchmark，使用 Nsight Systems 区分 CPU、GPU 和网络等待。
- 以单变量矩阵研究算法、协议与资源占用，而不是堆叠环境变量碰运气。
- 正确启动两机四卡测试，并把 GPU、CPU、NIC 拓扑与实际通信路径联系起来。

**推荐路线：**先读第 1、2 节建立测量合同与小模型；第 3～6 节是单机实验主线，按顺序做；第 7、8 节留到跨机时再读；第 9 节变量分层表与第 10 节源码锚点作速查。

主线承接：[nccl-tests](10-nccl-tests.md)；机制回查：[传输](07-transports.md)、[算法](04-algorithms.md)、[协议](09-device-protocols.md)。
单机实验默认已按第 10 章构建 `nccl-tests/build` 并核对动态库版本；第 8 节的多机实验另需 `build-mpi`。

## 1. Benchmark 首先是一份“测量合同”

本节回答：一个性能数字在什么条件下才可比较？**benchmark（基准测试）**不只是跑分，更是一份合同。同一个 all_reduce_perf，换计时边界、rank 映射或统计方式，就可能是在回答另一道题。先固定并记录以下信息，再写“提升了多少”。

| 维度 | 必须固定／记录的内容 |
| --- | --- |
| 软件 | 两个提交、CUDA runtime、驱动、MPI、网络插件、实际加载的 libnccl |
| 硬件 | GPU 型号与数量、NVLink／PCIe 拓扑、NIC／端口速率、NUMA |
| 工作量 | collective、类型、归约方式、实际 size 与扫描范围、rank 数与每机 GPU 数 |
| 布局 | 每进程线程数、每线程 GPU 数、设备可见性、CPU 绑定、oop/ip |
| 时间 | 默认提交到 stream 完成，还是 event／Graph／逐轮阻塞口径 |
| 样本 | 预热、计时迭代、校验、独立重跑次数、rank 平均或最大口径 |
| 外部状态 | 是否独占、功耗／温度／频率状态、其他作业与 fabric 负载 |
| 目标 | 最小消息延迟、大消息吞吐，还是训练 step 的关键路径 |

读结果按下面的顺序：

- 先看正确性，再看同尺寸的时间分布。
- 主表 `time` 是微秒，`GB/s` 是十进制；跨操作 S 与 busbw 系数见第 10 章，不与双向链路数字直接混比。
- `-a 1` 是进程／线程耗时平均，`-a 3` 关注最慢参与者；二者都不自动等于训练的全局 step wall time。

**注意：**
- 不同扫描的 `Avg bus bandwidth` 不宜排名。
- 先跑无 profiler 的基线，再短时观测；INFO／TRACE 和 profiler 都会带来潜在开销。
- 多次独立重跑，观察中位数和波动，不只报最优值；改善小于自然波动时，应判断“证据不足”。

## 2. 用一个小模型拆开延迟和带宽

本节回答：一条 size-时间曲线说明什么？固定操作、rank 数、拓扑、算法与协议后，在一段尺寸区间内可以近似：

```text
T(S) ≈ α + S/B
α：与消息大小弱相关的启动、调度、同步和协议开销
B：采用同一 S 定义的有效算法带宽，不是 NIC 物理额定带宽
```

两个极端情形：

- 当 `S/B << α`：加大消息，时间变化不大，带宽却随 S 上升——这是 **latency-bound（延迟受限）**。
- 当 `S/B >> α`：时间近似随 S 线性增长，带宽趋于平台——这是 **bandwidth-bound（带宽受限）**。
- 粗略的转折规模为 `S* ≈ αB`。

估计 α 和 B 用两点拟合，再加第三个尺寸验证：

1. 从同一稳定大消息区间选 `(S1,T1)`、`(S2,T2)`，用 `B ≈ (S2−S1)/(T2−T1)`、`α ≈ T1−S1/B` 估斜率与截距。代入前先把时间从微秒（us）换算成秒，估出的 B 单位才是字节/秒。
2. 再取同区间内**未用于拟合**的第三个尺寸 `S3`，预测 `T̂3 = α + S3/B`，用测得的 `T3` 计算残差 `T3−T̂3`——两个拟合点本身不能验证预测能力。
3. 保持版本、映射、命令与扫描范围等条件一致，独立重跑这三个尺寸，对照残差与重跑波动。

**注意：**
- α 包含 rank 数、算法步数和网络延迟影响，换 P 后不能直接沿用。
- 残差超出波动仍持续存在、截距明显为负或曲线有跳变时，先检查区间选择和模型适用性；残差落在波动内也不能证明模型处处成立。
- 不要跨越算法切换点硬拟合直线：减小启动开销更可能改善小消息，提高链路吞吐更可能改善大消息。

**动手验证（需 GPU）：**实测三个尺寸，完成上面的两点拟合与第三点残差检验；残差持续超过重跑波动时，按"注意"里的清单检查区间选择与算法切换点。
- 拥塞、频率变化和 rank 迟到都可能破坏模型；源码也会叠加架构及协议修正。模型只是提出假设的工具。

## 3. 建立不带强制算法的基线

本节回答：第一条基线怎么跑？先读拓扑，再让 NCCL 自动选择算法与协议，把命令和结果都记下来。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；当前作业至少获准使用两张卡。
ROOT="${ROOT:-$PWD}"
nvidia-smi topo -m
lscpu --extended=CPU,NODE,SOCKET,CORE
env -u NCCL_ALGO -u NCCL_PROTO \
  LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 8 -e 256M -f 2 -g 2 -t 1 -d float -o sum -w 5 -n 50 -c 1 -a 3
```

拓扑先行：`nvidia-smi topo -m` 能读出两张 GPU 之间是 NVLink、同 PCIe switch，还是跨 CPU socket。若同一机器不同 GPU 对差异很大，优先研究路径、NUMA 和共享 uplink，而非直接换算法。

`env -u` 只清当前子进程环境；这里的“自动”指未施加 ALGO/PROTO 强制值。记录管理员提供的必要设置，对未知的遗留变量先解释来源。

**注意：**
- “自动”不等于“环境完全干净”：站点配置文件、tuner 插件和框架 config 仍可能影响选择。
- 不要擅自清空站点配置或改 `/etc/nccl.conf`，也不要复制一长串所谓最佳配置。
- `topo -m` 中 GPU/NIC 的亲和关系是解释线索，不是已经完成的数据通路测量。

## 4. 单变量调优矩阵：Ring、Tree、Simple 与自动

本节回答：怎样判断某个算法或协议“更快”？一次只改一个变量。先只研究 AllReduce，保持第 1 节所有其他维度不变。

| 实验 | ALGO | PROTO | 与谁比较、回答什么 |
| --- | --- | --- | --- |
| A | 未设 | 未设 | 默认决策的基准 |
| B | Ring | 未设 | B 对 A：限制算法候选后的效果 |
| C | Tree | 未设 | C 对 A：另一种算法候选限制 |
| D | 未设 | Simple | D 对 A：限制协议候选后的效果 |
| E | Ring | Simple | E 对 B：固定 Ring 后只改变协议限制 |
| F | Tree | Simple | F 对 C：固定 Tree 后只改变协议限制 |

E 对 F 也可研究“Simple 固定时改变算法”；但 E 对 A 同时改了算法和协议两层限制，不能说成单变量因果。即使只改一个环境变量，内部 channel 或其他决策也可能随之变化，需记录实际选择。以下是**有条件的 AllReduce 实验模板**，不是承诺所有机器都支持每一项。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；只对本版支持该组合的 AllReduce 使用。
ROOT="${ROOT:-$PWD}"
clean=(env -u NCCL_ALGO -u NCCL_PROTO
  "LD_LIBRARY_PATH=$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
bench=("$ROOT/nccl-tests/build/all_reduce_perf"
  -b 8 -e 256M -f 2 -g 2 -t 1 -d float -o sum -w 5 -n 50 -c 1 -a 3)
"${clean[@]}" "${bench[@]}"                                      # A
"${clean[@]}" NCCL_ALGO=Ring "${bench[@]}"                       # B
"${clean[@]}" NCCL_ALGO=Tree "${bench[@]}"                       # C
"${clean[@]}" NCCL_PROTO=Simple "${bench[@]}"                    # D
"${clean[@]}" NCCL_ALGO=Ring NCCL_PROTO=Simple "${bench[@]}"     # E
"${clean[@]}" NCCL_ALGO=Tree NCCL_PROTO=Simple "${bench[@]}"     # F
```

逐项运行、保存命令与退出状态；某项不支持就记为“不适用”，不要给失败结果填零带宽。

预期假设：Tree 的阶段组织可能有利于部分小消息／大 rank 场景；Ring 常适合流水化大消息。但硬件、拓扑、协议及自动选择的其他算法都可能改变结果，不能提前给赢家。

观测实际选择可短跑 `NCCL_DEBUG=INFO` 配合 `INIT,GRAPH,TUNING,ENV` 子系统。nccl-tests 2.20.0 还可用 `-U 1` 请求 tuning 列；它使用 profiler 事件，缺失字段会显示 N/A。观测成功后去掉额外日志／profiler 再测性能，避免把观测开销算作算法差异。

**注意：**
- 自动选择可能采用 Ring/Tree 以外的实现，不能假定“自动就是 Ring”。
- 遇到无可用算法／协议，撤销强制值回到 A；不能假设 NCCL 总会偷偷回退到 Ring。
- 本版 Tree 成本模型只为 AllReduce 开放，不要把 F 命令机械改成 all_gather_perf。
- 不应强制 LL128 到不支持的平台；协议可用性与正确性不是性能实验能绕过的约束。

## 5. Nsight Systems：慢在提交、等待还是执行

本节回答：慢在 CPU 提交、依赖等待，还是 kernel 执行？用 Nsight Systems（**profiler**，性能分析器）录一段固定尺寸的短测试就能分辨；不要一开始就抓整个集群训练。

方法的关键是**先跑同参数、无 profiler 的固定尺寸对照**，再加 nsys；不要直接拿它与第 3 节扫描中的 8 MiB 行相减当作 profiler 开销。先确认本站安装了 Nsight Systems 且允许 profiling；以下不要求修改系统采样权限。两次测试之间只允许一个变量——有无 profiler：NCCL/tests 版本、实际加载库、GPU/rank 映射、环境变量、测试命令参数（含尺寸范围）都要一致。最后对照重跑波动，再判断 profiler 开销。

<details><summary>深入：为什么不能拿扫描里的 8 MiB 行当对照</summary>

按[第 10 章的地址轮换规则](10-nccl-tests.md#5-默认计时主机时钟覆盖提交到-stream-完成)，实际 maxbytes 未缩小时，AllReduce 固定 `8M..8M` 复用一个槽位，`8..256M` 扫描中的 8 MiB 行则轮换 32 个槽位，缓存工作集可能不同。扫描与本节还改变了迭代数，差异不能全归于 profiler。

</details>

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；已构建 tests 并安装 nsys。
ROOT="${ROOT:-$PWD}"
# 无 profiler 对照：测试参数与下面的 nsys 命令完全一致。
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 8M -e 8M -g 2 -t 1 -w 5 -n 20 -c 1 -a 3

TRACE_DIR="$(mktemp -d "$ROOT/nccl-profile.XXXXXX")"
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output="$TRACE_DIR/allreduce" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 8M -e 8M -g 2 -t 1 -w 5 -n 20 -c 1 -a 3
```

这些命令只在读者有授权的 Linux GPU 作业中执行；报告写到本次独立目录。打开生成的报告，按 CPU 线程、CUDA API、stream、GPU kernel 和 NVTX 范围对齐阅读：

1. 先区分初始化、数据准备、预热、计时、校验，不要把 cudaMemset 算作 collective。
2. 看 CPU 提交是否存在长空洞；GPU 无任务时，增加通信 CTA 通常治不了 CPU 迟到。
3. 看 NCCL kernel 启动之前是否被同 stream 的计算或跨 stream event 依赖阻挡。
4. 看 NCCL kernel 是否持续很长；它可能正在等待 peer／网络，不代表一直有效搬运数据。
5. 关联 proxy 线程调度、CPU 竞争和 NET 日志；时间线本身不能证明 RDMA 链路已饱和。
6. 多 rank 抓图时使用每进程唯一输出，并结合各节点时钟信息，不能直接拼接未对齐的时间戳。

**注意：**
- 不要把多个并行 stream 的 kernel 时长直接相加当作 wall time。
- Nsight Systems 擅长显示时间关系；链路利用率还需 NIC 计数器或专门带宽测试佐证。
- `-I 1` 的 event 统计是另一种诊断手段；`-G` 则研究 Graph 重放，两者本版不能同时启用。

## 6. Channel／CTA：多用资源不一定让应用更快

本节回答：给通信更多 GPU 资源，应用一定更快吗？不一定。**Channel** 是逻辑并行通信通道，不是物理 NVLink 或独占 NIC；**CTA** 是 CUDA thread block。

常规内核中，channel 并行度影响 CTA/warp 资源使用，但映射随算法与调度变化，不等于占用的 SM 数。更多块可能提高链路填充率，也消耗 SM 调度槽、寄存器、内存带宽和缓存；小消息拆太碎还会放大同步成本。

本版优先研究 `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS`，不堆叠旧 NCHANNELS 别名；设上限不保证实际使用该数量。先固定已验证的算法／协议，再比较最大 CTA 上限 8 与 16——它们只是两个对照值。做这组比较前，先确认没有更高的 `MIN_CTAS` 或其他配置冲突干扰结果。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；本例仅比较已支持的 Ring/Simple AllReduce。
ROOT="${ROOT:-$PWD}"
for cap in 8 16; do
  env -u NCCL_MIN_CTAS -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS \
    LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MAX_CTAS="$cap" \
    "$ROOT/nccl-tests/build/all_reduce_perf" \
    -b 32M -e 32M -g 2 -t 1 -w 5 -n 50 -c 1 -a 3
done
```

这里只观察纯通信；还须把相同两种配置放回真实负载，记录 `T_compute_alone`、`T_comm_alone`、串行与重叠总时间。理想重叠接近两者最大值，现实却可能因竞争使两者都变慢。

**注意：**
- stream 时间重叠或 overlap 百分比不代表没有干扰。
- 若 CTA 增加使 busbw 提高却令训练 step 变慢，应选择应用关键路径更短的方案。
- `-m` 聚合或 `-G` 重放不能替代真实 compute overlap 验证；bucket、依赖与通信时机见 [训练集成](16-training-integration.md)。

## 7. NET／GDR：把拓扑上的瓶颈画出来

本节回答：跨机性能差，先查哪一段？先把路径画出来。普通 RDMA 跨机路径可粗看成：GPU 内存 → GPU/NIC 间互连 → NIC → fabric → 对端 GPU。无 GDR 时可能经主机内存中转；GDR 减少中转，但不会增加网卡的线速。

| 现象 | 优先检查 | 因果解释 |
| --- | --- | --- |
| 单卡对外好，多卡同时对外差 | 共享 NIC、PCIe uplink、rail 分配 | 多个 rank 争同一个瓶颈出口 |
| 同机快、跨机明显退化 | NET 插件、端口速率、GDR、网络拥塞 | 跨机新增了链路与转发阶段 |
| 换 GPU/NIC 配对影响明显 | NUMA、PCIe switch、跨 socket 路径 | 绕行增加跳数和共享带宽压力 |
| 平均正常但尾部很长 | rank 迟到、proxy 调度、重试计数 | collective 被慢参与者拖住 |

先读 `nvidia-smi topo -m`、端口状态和 NET/GRAPH 日志；GDR 还要求 GPU/NIC、驱动及注册机制兼容，并通过路径距离判断。请管理员提供 GPU 对及 host/GPU 内存 RDMA 测试。

**注意：**
- 具备多机 NVLink 的系统还可能使用 MNNVL，不能把所有跨机通信都当成 NET。
- 缺少 nvidia-peermem 不等于 GDR 不可用，受支持系统可能走 DMA-BUF。
- host-memory RDMA 正常不证明 GPU-memory 路径正常。
- ACS、IOMMU、容器等仅做只读核查，不修改安全开关。

## 8. 两台机器、每机两卡：完整 MPI 映射

本节回答：两机四卡的最小完整启动怎么写？假设已获分配 `gpu-a`、`gpu-b`，每台允许当前作业使用两张卡，且两卡对该机两个进程共同可见。以下采用 Open MPI 语法：每机 2 个进程，每进程 1 个线程管理 1 张 GPU，共 P=4。

两台机器必须有相同绝对 ROOT、相同构建及兼容的运行库；替换实际 hostname。先按 [环境准备](00-environment.md) 核对 MPI 启动与初始化，不在此修改 SSH／防火墙。

```bash
# Linux NVIDIA GPU；从启动节点的 nccl_learn 执行；远端也用相同 ROOT。
(
  ROOT="${ROOT:-$PWD}"
  : "${MPI_HOME:?请先设置实际 MPI 安装前缀}"
  export ROOT
  export LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$MPI_HOME/lib:$MPI_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$MPI_HOME/bin/mpirun" -np 4 --host gpu-a:2,gpu-b:2 \
    --map-by ppr:2:node --bind-to none --report-bindings --wdir "$ROOT" \
    -x ROOT -x LD_LIBRARY_PATH \
    "$ROOT/nccl-tests/build-mpi/all_reduce_perf" \
    -b 8 -e 256M -f 2 -t 1 -g 1 -d float -o sum -w 5 -n 50 -c 1 -a 3
)
```

默认每机 localRank 0/1 分别使用可见 device 0/1；`--map-by` 只决定 MPI 分布，不替应用选 GPU。

**注意：**
- 本例不覆盖 CUDA_VISIBLE_DEVICES 或 NCCL_TESTS_DEVICE；若调度器逐进程暴露独占单卡，改用该模式并转发 `NCCL_TESTS_DEVICE=0`。
- “两卡共同可见”时不能这样统一覆盖，否则都挤到 device 0；启动后必须核对设备报告。
- `--bind-to none` 只用于初次映射验证；正式实验按站点亲和规则绑定，不与无绑定 smoke test 直接混比。

### Bootstrap 的 IP 网卡和 bulk 的 RDMA 网卡不是一回事

`NCCL_SOCKET_IFNAME` 选择 NCCL 使用的 IP socket 接口，通常影响 bootstrap／带外交换。`NCCL_IB_HCA` 选择内置 IB/RoCE 后端的 **HCA**（IB/RoCE 网卡设备）／端口，不是 Linux IP 接口名。

例如获准接口为 `ens5f0`、RDMA 设备为 `mlx5_0:1` 时，可在上面的子 shell 中设置：`NCCL_SOCKET_IFNAME='=ens5f0'` 与 `NCCL_IB_HCA='=mlx5_0:1'`，export 后分别加 `mpirun -x` 转发。前导 `=` 表示精确匹配，避免 `mlx5_1` 意外选中 `mlx5_10`。

**注意：**
- 如果 bulk 数据也采用 Socket 网络后端，`NCCL_SOCKET_IFNAME` 也会影响 bulk；不要把它永远称作“只管初始化”。
- 这些名字只是例子；多 NIC 节点强制只留一个端口可能主动制造瓶颈，不应作为默认基线。
- MPI 自己的控制连接／数据传输由 MPI 配置决定，NCCL 的两个变量不替 MPI 选网卡。
- 外部网络插件也可能有独立的选择规则，要结合实际加载插件的日志和文档。

## 9. 环境变量按责任分层，不建“万能全局配置”

本节回答：这么多变量，谁该碰哪些？按使用责任分层：

| 层级 | 变量 | 建议用途与边界 |
| --- | --- | --- |
| 日常观察 | `NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS` | 需要时用 INFO 加子系统筛选；TRACE 仅短时复现 |
| 日常观察 | `NCCL_DEBUG_FILE` | 每次作业、每进程唯一文件，详见第 12 章 |
| 实验强制 | `NCCL_ALGO`、`NCCL_PROTO` | 限定候选，可能失败；默认优先不设置 |
| 实验强制 | `NCCL_MIN_CTAS`、`NCCL_MAX_CTAS` | 研究资源／吞吐权衡，最终以应用时间验收 |
| 实验隔离 | `NCCL_P2P_DISABLE`、`NCCL_IB_DISABLE`、`NCCL_NET_GDR_LEVEL` | 经授权的单路径对照；不是普遍加速参数 |
| 管理员／站点约定 | `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`NCCL_NET` | 保证选对可达接口和已部署后端，不凭机器名猜值 |
| 管理员／fabric 专项 | `NCCL_IB_QPS_PER_CONNECTION`、`NCCL_CROSS_NIC` | 结合交换网络、rail 与拥塞证据调整 |
| 管理员／故障策略 | `NCCL_IB_TIMEOUT`、`NCCL_IB_RETRY_CNT`、`NCCL_SOCKET_RETRY_CNT` | 重试策略不是吞吐调优，也不是 collective 总超时 |
| 管理员／诊断部署 | `NCCL_RAS_ADDR` | 管理监听地址及访问边界，不对外随意暴露 |

分层表示使用责任，不表示普通用户技术上无法设置这些变量。默认优先自动选择；必要站点配置沿用管理员要求；所有临时强制值仅限本次子进程。若要保留某项优化，必须覆盖真实消息分布、规模、正确性和 compute overlap，再记录适用范围。更多非常规通信、注册及资源策略见 [高级主题](13-advanced.md)。

## 10. 源码锚点

路径相对仓库根目录，行号固定在本章基线；不要拿旧版本目录结构套用。

| 路径与行号 | 符号／可核验结论 |
| --- | --- |
| `nccl/src/tuning/tuning_general.cc:52-65` | `ncclTuningGetTime`：延迟项与数据／带宽项 |
| `nccl/src/tuning/tuning_general.cc:141-198` | `ncclTuningGetChannels`：按尺寸缩减 channel／线程 |
| `nccl/src/tuning/tree.cc:11-26,63-80` | `ncclTuningTreeModelInit/Sim`：Tree 的操作适用范围 |
| `nccl/src/tuning/tuning.cc:308-329` | `ncclTuningCompute`：无可用候选时报告错误 |
| `nccl/src/init.cc:2000-2001,2298-2318` | `MaxCTAs/MinCTAs`、`envConfigOverride` |
| `nccl/src/graph/paths.cc:494-564` | `ncclTopoCheckGdr`：能力、读方向、距离限制 |
| `nccl/src/misc/socket.cc:211-252` | `ncclFindInterfaces`：IP 接口选择 |
| `nccl/src/transport/net_ib/init.cc:454-477` | `ncclIbInitDevices` 内 HCA 筛选及设备打开 |
| `nccl-tests/src/common.cu:1576-1584,1657-1661` | `run`：localRank 与默认 GPU 映射 |
| `nccl/docs/userguide/source/env.rst:1477-1565,1866-1925` | `NCCL_ALGO/PROTO/MIN_CTAS/MAX_CTAS` 定义与限制 |
| `nccl/docs/userguide/source/troubleshooting/performance_and_tuning.rst:94-171` | Tuning、CPU/memory affinity 的使用边界 |

## 11. 自测题与答案

1. 消息翻倍、耗时几乎不变，但 algbw 翻倍，是否证明网络线速翻倍？
   **答：**不是；很可能仍由 α 主导，带宽增长来自分子变大，不是链路容量改变。
2. Ring+Simple 比自动快，能直接说“Simple 比默认协议快”吗？
   **答：**不能；同时改变了算法和协议限制。先比较 Ring+自动协议与 Ring+Simple，并记录实际选择。
3. 两机四 rank 初始化成功，是否证明 GPU 经目标 RDMA NIC 直接通信且已达峰值？
   **答：**不能；初始化只证明部分控制路径可用，还要核对 NET/GDR、GPU/NIC 映射及数据通路表现。

下一章：[12 调试与故障定位](12-debugging.md)，把调用、设备和网络问题分层排查；实际遇错时也可提前查阅。
