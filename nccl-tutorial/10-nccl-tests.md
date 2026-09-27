# 第 10 章：用 nccl-tests 建立可信的通信基线

> 基线：NCCL `12df1a11`（2.32.3），nccl-tests `b4d5bee`（2.20.0）。
> 所有 shell 示例都从仓库根目录 `nccl_learn`、在 Linux NVIDIA GPU 节点的 Bash 中执行。
> `ROOT` 可由用户预设为该节点上的仓库根目录；否则取 `$PWD`。不要照搬其他机器的路径。
> 当前编写环境是无 CUDA 的 macOS，未构建或实测；下文的算术示例均明确标为教学构造，不是性能成绩。

## 学习目标

- 能分别构建单进程版和 MPI 版，并确认运行时加载的是目标 NCCL。
- 能解释每一列，区分测试 size、输出 count 和 NCCL API 的 count。
- 能说明预热、计时、校验、跨 rank 统计分别发生在哪里。
- 能手算算法带宽与总线带宽，并设计不同通信操作的对照实验。

**本章在机制主线中的位置：验证解释，而不是只追峰值。** 先核对第 5 节的计时边界、第 6 节的进程/GPU 映射、第 7 节各操作的数据量定义，再把观察与[传输](07-transports.md)、[算法](04-algorithms.md)、[协议](09-device-protocols.md)的实际分支对应。GPU 实验仍先完成[环境准备](00-environment.md)并跑通小规模正确性。

即使熟悉 GPU profiling，仍需核实这里默认使用主机批次计时、校验在计时外、busbw 为归一化量；这些不是从 CUDA event 或硬件计数器语义直接继承的。第 11 章进一步安排日志与时间线的单变量实验。

### 基线带练速查：第一次使用 nccl-tests 时按需阅读

若需要基础带练，可先回答三个问题：**两卡算对了吗？一行输出是什么意思？尺寸变大后为什么带宽变化？**

1. 第 1 节只做单进程构建与库检查；已完成第 00 章的可直接核对 `ldd` 后跳过构建。
2. 第 2 节固定 **64 KiB、FP32、两卡（P=2）**；第 3 节读懂一行并手算，再扫 `8..16M`。
3. 第 5 节只读默认计时流程，分清预热、计时、校验；到这里就能建立首个基线。

上面三步是按需使用的基础带练，不是机制主线的必经起点，也不替代统计与数据量分析；其中标为“选读”的内容可按实验目标随时展开。上述两卡带练的首次正确性命令不加 MPI、CUDA Graph、profiler 或复杂选项，之后一次只增加一个变量；源码索引在章末，用来核对实际计时、调度和校验边界。

## 1. 先把“测到了哪个库”说清楚

nccl-tests 是调用 NCCL 的测试程序，不是 NCCL 库本身，也不是训练框架。
单进程可以管理多张 GPU；跨进程、跨机需要 MPI 构建及正确的 launcher。
先准备 Linux 驱动、CUDA toolkit、C++ 编译器、make 和 Python 3（NCCL 设备代码生成需要 `python3`）；MPI 仅多进程实验需要。
本地源码布局、被 `.gitignore` 忽略的源码如何准备，以及 `export`、`$PWD`、`git -C ... rev-parse HEAD`、`make -C/-j/src.build` 的拆读见第 00 章。下面每一步成功后再执行下一步。

```bash
# Linux/bash + NVIDIA GPU；当前目录必须是 nccl_learn。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
nvidia-smi
"$CUDA_HOME/bin/nvcc" --version
make -C "$ROOT/nccl" -j 8 src.build CUDA_HOME="$CUDA_HOME"
make -C "$ROOT/nccl-tests" -j 8 MPI=0 \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$ROOT/nccl/build" \
  BUILDDIR="$ROOT/nccl-tests/build"
LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$ROOT/nccl-tests/build/all_reduce_perf"
```

`${ROOT:-$PWD}` 表示 ROOT 未设置或为空时才取当前目录；CUDA 路径的写法同理。这里固定 8 个并行构建任务，内存紧张时减少 `-j`，不必一开始就开满所有 CPU。`BUILDDIR` 指定测试产物目录。
**检查点：** GPU 和 `nvcc` 可用，两个 make 都成功，`ldd` 无 `not found` 且 NCCL 指向本地 build。哪一步失败就停在哪一步，先按第 00 章排查，不进入性能测试。

NCCL 默认产物在 `nccl/build`，不是源码目录下的 `lib`；检查 `ldd` 中 `libnccl.so` 的实际解析路径。
`NCCL_HOME` 提供头文件和链接搜索路径，**不会替你设置 NCCL 的运行时 rpath**；漏设可能变成“新头文件、旧运行库”。
后续命令用进程局部的 `LD_LIBRARY_PATH`，不修改系统库或 shell 启动文件。`env 变量=值 命令` 只给该命令设置环境；行末 `\` 表示命令续行。

<details>
<summary>选读：MPI 版构建；单进程带练可直接进入第 2 节</summary>

```bash
# Linux/bash + NVIDIA GPU；从 nccl_learn 执行；MPI_HOME 先设为本站真实安装前缀。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${MPI_HOME:?请先设置本站 MPI 的安装前缀}"
make -C "$ROOT/nccl-tests" -j 8 MPI=1 \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$ROOT/nccl/build" MPI_HOME="$MPI_HOME" \
  BUILDDIR="$ROOT/nccl-tests/build-mpi"
LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:$MPI_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$ROOT/nccl-tests/build-mpi/all_reduce_perf"
```

默认 `MPI=0`；MPI 版放进 `build-mpi`，避免复用未启用 MPI 的旧对象。
`MPI_INCLUDE` 可单独覆盖头文件位置；非标准安装还须核对库目录，以及 MPI 编译器、头文件、libmpi、mpirun 的兼容性。
运行时每节点都要能找到 NCCL、CUDA、MPI 动态库；错误应从参数、`ldd`、版本定位，不删除系统库。

</details>

## 2. 第一次实验：AllReduce 从小消息扫到大消息

本节先固定尺寸，读懂第 3 节的一行输出后再扫描。至少分配两张可用 GPU；`0,1` 指当前作业可见范围内的设备，须与调度器授权一致。
不要覆盖调度器的独占卡映射去占用别人的 GPU。

**先画数据流：** FP32 每个元素占 4 字节，64 KiB 就是每卡 16,384 个元素。rank 是 NCCL 通信参与者的编号；本例一个进程、一个工作线程管理两张卡，所以 P=2。

```text
GPU 0 / rank 0：A[0..16383] ─┐
                            ├─ AllReduce(sum) ─┬─> GPU 0：A[i]+B[i]，共 16384 个
GPU 1 / rank 1：B[0..16383] ─┘                  └─> GPU 1：A[i]+B[i]，共 16384 个

一次固定尺寸测试：准备输入 → 预热 → 计时批次 → 计时外重新准备并校验 → 打印
                         先做 out-of-place，再做 in-place，合成一行
```

A、B 只表示两卡输入，实际校验数据由 tests 生成。下面不加算法、Graph 或 profiler 开关，走默认 host API 调用 NCCL。

```bash
# Linux/bash + NVIDIA GPU；从 nccl_learn 执行；当前作业已获准使用两张卡。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
env NCCL_DEBUG=INFO \
  LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 64K -e 64K -g 2 -t 1 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

逐组读命令即可：

- `-b 64K -e 64K`：起止尺寸相等，只生成一个尺寸的结果；`-d float -o sum` 固定 FP32 求和。
- `-g 2 -t 1`：每线程两张 GPU、每进程一个工作线程；不需要 MPI。
- `-w 5 -n 20`：每种缓冲模式先预热 5 次，再计时 20 次，最后除以次数。
- `-c 1`：计时后重新准备数据，另做 1 轮正确性校验，不把校验时间混入 `time`。
- `-a 3`：对各进程／线程的**平均耗时**取最大值，再算带宽；不是选最慢的单次迭代。本例只有一个计时线程，它等待两张卡的 stream 都完成。

**检查点：** 先核对设备报告中的两张卡，再找 `size=65536` 的结果行；两组 `#wrong` 都为 0、摘要无失败且程序正常结束，才进入扫描。若卡在初始化、报错或校验非零，停在固定用例排查，暂不讨论性能。逐列读法就在下一节，基础带练无需展开参数表。

<details>
<summary>选读：完整的常用参数速查与统计口径；按实验需要查阅</summary>

### 常用选项及本版默认值

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `-b` / `-e` | 都是 32 MiB | 请求的最小／最大测试字节规模 |
| `-i` | 1 MiB | 线性步长；`-f > 1` 时改用乘法步进 |
| `-f` | 1 | 大于 1 时按倍率扫描，常用 2 |
| `-n` | 20 | 性能计时中的外层迭代次数 |
| `-w` | 1 | 正式计时前的预热次数 |
| `-c` | 1 | 计时之后的校验轮数，0 表示不校验 |
| `-t` / `-g` | 1 / 1 | 每进程 CPU 工作线程数／每线程 GPU 数 |
| `-m` | 1 | 每个计时迭代聚合的 collective 次数 |
| `-a` | 1 | 0：rank 0 口径；1：平均；2：最小；3：最大耗时 |
| `-d` / `-o` | float / sum | 数据类型／归约操作，按操作是否适用解释 |
| `-T` | 0 | tests 的 stream 等待超时秒数；0 不启用此超时 |
| `-G` | 0 | CUDA Graph 重放次数；非默认实验模式 |

`K/M/G` 大小后缀使用二进制倍率；输出带宽 `GB/s` 则使用十进制 `10^9 B/s`。
`-a` 聚合的是各进程／线程测得的耗时，而非对多个带宽值取平均。
一线程管理多卡时，该线程等待自己的所有 stream，不是分别输出每张卡的独立延迟。
`-a 0` 不做 MPI 全局耗时归约；本版实现只是进程内 thread 0 广播，通常最终显示主进程值。
因此它不适合证明“全部 rank 都快”；本章用 `-a 3` 观察最慢参与者口径。

</details>

## 3. 输出每一列究竟是什么

### 先读一行：64 KiB、FP32、P=2

**以下整行数据为“人工构造、仅教学、非实测”**，用于练习标准输出的列与单位，不代表任何 GPU 的速度。假定上面命令的 oop 计时批次耗时 640 us、ip 耗时 1280 us，分别做 20 次通信；再假定各自的计时外校验错误数为 0。按默认文本格式可写成：

```text
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
       65536         16384     float     sum      -1    32.00    2.05    2.05       0    64.00    1.02    1.02       0
```

**左右两组不是两张 GPU。** 左组是输入输出分开（out-of-place，oop），右组是按 API 约定复用缓冲区（in-place，ip）。程序先测左组、再测右组；每组都由两张卡共同完成 AllReduce，不是两次同时进行的通信。

先按这个顺序读：`size/count/type` 确认测了什么 → 两个 `#wrong` 确认是否通过 → `time` 看耗时 → `algbw/busbw` 看换算后的带宽。AllReduce 没有根 rank，所以这里 `root=-1` 不是错误。

| 列 | 如何阅读 |
| --- | --- |
| `size` | 实际每 rank 发送／接收缓冲有效字节数中的较大者，不是全作业总字节数 |
| `count` | 传给该测试操作的元素参数 `paramCount`，不是总线传输次数 |
| `type` | 元素类型，决定每元素字节数 |
| `redop` | sum/prod 等归约方式；不归约的操作通常显示 none |
| `root` | 有根操作的根 rank；无根操作通常显示 -1 |
| `time` | 单次操作平均耗时，单位 **微秒 us** |
| `algbw` | 该操作规定的数据规模除以完成时间，十进制 GB/s |
| `busbw` | `algbw` 乘以该操作的流量归一化系数 |
| `#wrong` | 校验发现的错误元素数；`N/A` 表示未报告校验，不等于通过 |

### 手算这一行，不先背通用公式

仍只使用上面的人工构造数据：

1. **尺寸换元素数：** `S = 64 × 1024 = 65536 B`，FP32 每元素 4 B，所以 `count = 65536 / 4 = 16384`。这是**每卡**参与求和的元素数，不是两卡合计，也不是 65,536 个元素。
2. **批次换单次时间：** oop 的 `time = 640 / 20 = 32 us = 0.000032 s`；ip 为 `1280 / 20 = 64 us = 0.000064 s`。每个计时批次在末尾等两卡工作完成，并非每次迭代都同步；不能再除以 GPU 数 2。
3. **算法带宽：** oop 的 `algbw = S / t / 10^9 = 65536 / 0.000032 / 10^9 = 2.048 GB/s`，显示为 **2.05**；ip 同理为 `1.024 GB/s`，显示为 **1.02**。K/M 尺寸用 1024 倍，输出 GB/s 用十进制 `10^9`，不要混用。
4. **总线带宽：** tests 对 AllReduce 固定使用 `busbw = algbw × 2(P−1)/P`。本例 P=2，系数 `2×(2−1)/2=1`，所以两组各自的 busbw 与 algbw 相等。系数来源见第 7 节；它是归一化流量模型，不是物理链路计数器。

真实输出也会四舍五入，手算值应与显示精度相符；不要把 2.05 当成没有舍入的原始值。这里故意让 ip 更慢以便分清两组，缓冲区复用本身不决定快慢。

### 通过这一行，才扫尺寸

**本次 AllReduce 两组 `#wrong` 都为 0 才通过校验。** `-c 1` 在计时后另外生成输入并对照期望结果；若改成 `-c 0`，就没有这一证据，`N/A` 不能算正确。页尾 `Out of bounds values` 是错误检查摘要，也要无失败且程序正常退出。校验覆盖的是本次生成的数据和用例，并非对所有输入的证明。

确认固定用例通过后，只改变尺寸范围和步进：

```bash
# Linux/bash + NVIDIA GPU；从 nccl_learn 执行；固定 64 KiB 两卡校验已通过。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
env NCCL_DEBUG=INFO \
  LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 8 -e 16M -f 2 -g 2 -t 1 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

`-b 8 -e 16M -f 2` 是从 8 字节到 16 MiB，每次翻倍；每一行仍按刚才的方法读。这里写出的预热、迭代数用于复现，不代表上游默认值。日志级别也先保持不变；正式稳态测量若降低日志级别，对照组应一起调整。

用一个直观模型理解趋势：`时间 ≈ 固定提交／启动成本 + 随尺寸增长的数据处理与传输时间`。

- **小消息：** 固定成本占主要部分，size 翻倍时 time 可能变化不大；因为 `algbw=S/t`，带宽会随尺寸上升。
- **较大消息：** 搬数据及归约的成本逐渐占主导，time 随 size 增长，带宽开始接近平台；16 MiB 是否已进入平台要看设备与拓扑。
- **台阶或回落：** 算法、协议、channel 或分块策略切换，以及设备竞争，都可能改变成本；先看重复运行和校验，再回查相应机制章节，不要求曲线单调。

以上是解释模型，不是本教材的实测趋势。先确认全部尺寸输出、每行两组校验为 0；若发生失败，停在出问题的尺寸排查。
显存不足时减小 `-e`；测试还需 send/recv/expected 和内部缓冲，`-e` 不是显存峰值。程序也可能依据显存减少 maxBytes，记录开头提示，确认实际扫到了哪里，不把 OOM 当作算法失败。稳定后可逐步扩到 256 MiB 等更大范围。

`Avg bus bandwidth` 是所测项目的汇总平均，强烈依赖扫描范围和项目构成，不是峰值。一个覆盖大量小消息的扫描，其平均值自然可能低于仅测大消息的扫描；比较时先对齐范围，再对比相同 size 的行。

**基础带练到此已会读输出，接着读第 5 节的默认计时即可；机制主线仍需核对映射与流量口径。**

<details>
<summary>选读：扩展输出列、逐迭代统计与 profiler</summary>

进阶选项会扩展表头；不打开它们，默认没有以下附加列：
- `-C 1`：把 `time` 显示为 `cputime`；带宽仍按完成时间计算，不能拿 cputime 反推它。
- `-S 1`：增加 `timestamp`，供日志关联，不是通信持续时间。
- `-I 1`：增加 `i_min/i_max/i_p99/i_cv%`，分别为逐迭代最小、最大、99 分位和变异系数。
- `-U 1`：增加 `impl/algo/proto/kernelVariant/sync/#channels/#warps/netChunkSize`。
- 这些 tuning 列依次描述实现、算法、协议、内核变体、同步方式、channel 数、warp 数和网络块大小。
- tuning 信息依赖 profiler 事件；`N/A` 不表示该资源为零，观察模式也不应混入最终性能基线。

</details>

<details>
<summary>选读：把 count 的读法推广到其他通信操作</summary>

### 三个容易混淆的 count

令 `P` 为 NCCL communicator 的 rank 数，`q` 为表中 count，`d` 为元素字节数。
测试先把请求 size 换成候选元素数，再由操作实现生成 sendCount、recvCount、paramCount。
输出 count 是最后一个；输出 size 是 `max(sendCount, recvCount) × d`。

| 操作 | NCCL API 中 q 的含义 | 输出 size，也就是本章带宽的 S |
| --- | --- | --- |
| AllReduce | 每 rank 输入／输出元素数 | `q × d`，每 rank 完整数组 |
| AllGather | 每 rank 的 sendcount | `P × q × d`，每 rank 完整接收数组 |
| ReduceScatter | 每 rank 的 recvcount | `P × q × d`，每 rank 完整输入数组 |
| SendRecv | 每 rank 给下一个 peer 的元素数 | `q × d`，单方向消息长度 |
| Broadcast | 根发送、各 rank 接收的元素数 | `q × d`，被广播的数组 |
| Reduce | 每 rank 输入、根输出的元素数 | `q × d`，被归约的数组 |
| AlltoAll | 每个 peer 的元素数 | `P × q × d`，每 rank 全部 peer 的数组 |

AllGather、ReduceScatter、AlltoAll 还会按 rank 数拆分并做每分片 16 字节对齐。
所以请求 `-b` 不一定等于最终 size；小到无法形成分片时尤其不能机械相除。
例：P=4、float、实际 size=64 MiB，AllReduce 的 count 是 16,777,216。
相同 size 下，AllGather 和 ReduceScatter 的 count 都是 4,194,304。
前者每 rank 输入 16 MiB、输出 64 MiB；后者每 rank 输入 64 MiB、输出 16 MiB。
注意它们的 **count 相同，数据流方向却相反**。

</details>

## 4. in-place 不是“这次测试只分配一份显存”

**选读：已经分清左右两组后，再看缓冲区细节。**

Out-of-place（oop）使用分离的输入与输出；in-place（ip）按 API 约定复用缓冲区。
AllReduce 可以令输入输出为同一地址；AllGather 的输入位于输出中本 rank 的分片位置。
ReduceScatter 的输出则位于输入中本 rank 对应的分片位置，不可随便重叠。
nccl-tests 先测 oop，再测 ip；`AllocateBuffs` 仍然分配 sendbuff、recvbuff，校验还要 expected。
因此测试进程并没有因为 ip 列而少分配这些缓冲；地址复用也未消除归约和通信成本，速度仍需在相同条件下比较。

本版 **SendRecv 和 AlltoAll 不支持 in-place 正确性测试**，该列校验不报告。
即使右侧仍打印了时间，也不应将其当作受支持的原地语义或正确性结果。
比较这两项时只采用 oop 的有效数据，检查该列的 `#wrong`。

## 5. 默认计时：主机时钟覆盖提交到 stream 完成

现在再问：刚才的 `32 us` 到底包含了什么？CUDA 工作是异步提交的，API 返回时 GPU 可能还在忙；只量返回时间会漏掉等待结果的时间。

```text
准备数据 → 预热并等待 → 对齐 │ 主机 timer 开始 → 连续提交 n 次 → 等所有 stream 完成 │
                            └──────────── 主计时区间，最后除以 n ────────────────┘
                            → 按 -a 汇总耗时并算带宽 → 重新准备数据、校验 -c 轮
```

把 `BenchTime` 按程序执行顺序分成五段读，不必先追 GPU 内核（默认 `agg_iters=1`，本例 `iters=20`）：
1. **准备段：** 准备数据，执行预热，等待预热 stream 完成，再做进程／线程对齐。
2. **提交段：** 创建主机 `timer`，执行 `iters × agg_iters` 次通信提交；本例两卡的调用由同一线程组织。
3. **完成段：** `completeColl` 等待相关 CUDA stream 完成；然后读取主机时间。这里才把 GPU 真正完成纳入测量。
4. **统计段：** 除以通信次数，对各进程／线程耗时按 `-a` 归约，再计算带宽；默认 `-a 1` 求平均，本章命令显式选 `-a 3` 求最大，均不是先算带宽再平均。
5. **校验段：** 重新初始化数据，另外运行 `-c` 指定的校验轮数，检查输出，再打印该模式的结果。

默认不是“只量 NCCL API 返回时间”，也不是“用 CUDA event 量内核”。
主机 timer 基于 `std::chrono::steady_clock`，等待通过 `cudaStreamQuery` 等完成检查实现。
默认批量连续提交并在末尾等待，所以它是批次平均，不等于每轮都同步的孤立延迟。
初始化、预热和计时后的校验不计入默认主计时区间。`-c` 增多会延长整次程序运行，但不会把那些校验轮次加进主表的 `time`。

**批次内还会轮换缓冲地址，而非总在同一地址上重复。** 对非零 `totalnbytes = max(sendBytes, expectedBytes)`，`startColl` 计算 `steps = maxbytes / totalnbytes`、`shift = totalnbytes * (iter % steps)`，将 send/recv 指针加上 shift 后传给通信调用。
以 AllReduce 为例，实际最大尺寸未被显存限制缩小时，固定 `-b 8M -e 8M` 的 `steps=1`；扫描 `-b 8 -e 256M` 中的 8 MiB 行则为 `steps=32`，迭代在这些槽位间轮换。因此即使比较同一 size，地址复用与缓存工作集也可能不同，不能把差异全归于 profiler；做 profiling 对照须固定同一尺寸范围与其余测试参数。
源码依据集中见 `nccl-tests/src/common.cu:557–570,623–626`：计算偏移、移动指针、默认 host API 调用；这是测试程序的地址策略，不是 NCCL API 规定的行为。

<details>
<summary>选读：提交时间、group 聚合、CUDA Graph 与逐迭代统计</summary>

`-C 1` 才把等待之前的主机提交口径显示出来；它也不一定是纯 API 开销。
`-m 8` 用 group 聚合更多操作，可能摊薄提交成本，也可能改变调度；不能和默认混比。
`-G` 先捕获并实例化 graph，然后重新开始计时重放，结果不含捕获／实例化成本。
它适合研究重放吞吐，不等于普通训练程序的逐次提交性能。
`-I 1` 才额外记录 CUDA events；`-K` 只跳过逐迭代统计的前 K 项，不改主表总平均。
本版 `-G` 会禁用 `-I`，也会禁用 `-z 3`；看到提示要记录实际生效模式。

</details>

## 6. GPU、线程、MPI rank 怎么对应

**机制主线：核对进程/GPU 映射；多线程、MPI 多进程／多机细节按需展开。** 单进程 `t=1,g=2` 的基础带练用例只需记住 P=2。

未使用 tests 的特殊 split 分组时：`P = np × t × g`。
NCCL rank 为 `(MPI rank × t + thread) × g + gpu`。
默认 CUDA device 为 `localRank × t × g + thread × g + gpu`，编号相对可见设备列表。
例如一机 `np=2,t=1,g=2`：两个 MPI 进程分别管理 device 0/1 与 2/3，共四个 NCCL rank。
另一种 `np=4,t=1,g=1` 也有四个 NCCL rank，但 CPU 线程和进程组织不同。
两者性能可能不同，不能把总 GPU 数相等误认为实验条件相等。

若调度器让每个 MPI 进程只看见各自独占的一张卡，设备编号都成为 0。
此时在 `t=g=1` 下可用 `NCCL_TESTS_DEVICE=0` 覆盖默认 localRank 偏移。
若所有 GPU 共同可见，不能给所有进程统一设 0，否则会重复占用同一 GPU。
启动后核对设备报告中的 hostname、rank、device、PCI bus ID；不要只数输出行。
MPI 两机启动的完整映射例子放在 [第 11 章](11-performance.md)。

## 7. 带宽推导：先定义 S，再谈数值

**机制主线：核对流量模型与测量口径。** 仅做首次两卡正确性带练时可暂缓，但解释性能前必须区分各操作的 S 与归一化系数。

以下 `S` 都以**字节**计，`t` 以秒计，`algbw = S / t / 10^9`。
S 的逐操作定义在第 3 节表格；尤其 AG/RS 的 S 不是 API 的 `q × d`。
以 Ring 作直观推导：把 S 分为 P 份，每步处理 `S/P`。
AllReduce 的 ReduceScatter 阶段 P−1 步，AllGather 阶段再 P−1 步。
所以每 rank 归一化发送量为 `2(P−1)S/P`，得到 `busbw = algbw × 2(P−1)/P`。
AG 或 RS 只有一个阶段，归一化发送量为 `(P−1)S/P`，系数是 `(P−1)/P`。
AlltoAll 含本地分片的总 S 中，远端部分占 `(P−1)/P`；SendRecv 系数为 1。
Broadcast、Reduce 的测试归一化系数也为 1，不乘总 rank 数。

### P=4 和 P=8 手算（纯假设，不是实测）

统一假设某操作 `S=64,000,000 B`、完成 `t=0.010 s`，故 algbw=6.4 GB/s。
这个 S 对 AG 是完整输出，对 RS 是完整输入；不能把假设偷换成每 rank 分片。

| 操作 | P=4：系数及 busbw | P=8：系数及 busbw |
| --- | --- | --- |
| AllReduce | `2×3/4=1.5`；`6.4×1.5=9.6` | `2×7/8=1.75`；`6.4×1.75=11.2` |
| AllGather / ReduceScatter | `3/4=0.75`；`6.4×0.75=4.8` | `7/8=0.875`；`6.4×0.875=5.6` |
| AlltoAll | `3/4`；4.8 | `7/8`；5.6 |
| Broadcast / Reduce / SendRecv | 1；6.4 | 1；6.4 |

另一种核算：AR 在 P=4 每 rank 归一化发送 96 MB，在 P=8 为 112 MB。
AG/RS 则分别为 48 MB、56 MB；除以相同 0.010 s，与表格吻合。
这些系数由 tests 固定应用，选 Tree 也不会改用另一张系数表。
**busbw 不是物理链路计数器，不是所有链路收发字节之和，也不是硬件实测峰值。**
多 NIC、全双工口径、NVSwitch／卸载算法都会影响它与硬件标称数字的可比性。

## 8. 用其他操作交叉检查解释

**选读：AllReduce 基线稳定后再做跨操作实验。** 此处刻意扩展到 256 MiB、50 次计时；用于交叉对照时，各操作（包括 AllReduce）要统一这些条件、日志级别与设备分配。

```bash
# Linux/bash + NVIDIA GPU；从 nccl_learn 执行；已分配两张卡，使用默认 host API 实现。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
for op in all_gather reduce_scatter sendrecv; do
  env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ROOT/nccl-tests/build/${op}_perf" \
    -b 1K -e 256M -f 2 -g 2 -t 1 -d float -w 5 -n 50 -c 1 -a 3
done
```

SendRecv 是每 rank 向下一个发送、从上一个接收的并发环，不是串行 ping-pong；对比 AR 可区分单跳与多阶段成本。
AG/RS 大消息时间可能接近，但归约计算、内存方向与算法会使两者不同；AR 语义上等于 RS+AG，但融合后的调度、数据处理不同，独立测试时间并不必然精确相加。
记录完整命令、版本、拓扑、映射、实际 size、oop/ip、计时模式和正确性结果。
保存多次独立结果后再去 [性能分析](11-performance.md) 建模，不要只摘最大 busbw。

## 9. 自测题与答案

先自测，再回查源码。基础带练可先做第 2、4 题；第 1、3 题用于核对跨操作口径与多进程映射。

1. P=8、float、AllGather 的 count=1024，S 是多少？busbw/algbw 是多少？
   **答：**S=`8×1024×4=32768 B`，是每 rank 完整输出；比例为 `7/8`。
2. 默认 `time` 是 CUDA event 结果吗？`-c 3` 是否把三轮校验加入它？
   **答：**不是；默认是提交到 stream 完成的主机批次平均，校验轮次在计时之后。
3. 两个 MPI 进程都能看见 GPU 0/1，能否统一设 `NCCL_TESTS_DEVICE=0`？
   **答：**不能，这会使它们指向同一张卡；只有分别隔离为独占单卡且 `t=g=1` 才可这样使用。
4. 64 KiB FP32 的 count 是多少？左右两组是两张卡吗？`N/A` 能算校验通过吗？
   **答：**16,384 个元素／rank；两组是 oop/ip，每组都包含两张卡；`N/A` 是未报告校验，不能代替 0 错误。

## 10. 源码锚点：沿这些符号复核

**随机制问题核查，不必等全部实验做完。** 先沿 `TimeTest → BenchTime → startColl → AllReduceRunColl → ncclAllReduce` 找到默认执行路径，再分别回查计时、校验和带宽换算；暂时跳过 `deviceImpl > 0` 的教学内核分支。

以下路径均相对仓库根目录，行号对应本章固定提交。

| 路径与行号 | 符号／阅读目的 |
| --- | --- |
| `nccl/README.md:15-33` | Build：`src.build`、CUDA_HOME、默认 build 目录 |
| `nccl-tests/src/Makefile:12-39,127-130` | `MPI`、`NCCL_HOME`、链接规则，无 NCCL rpath |
| `nccl-tests/src/common.cu:89-122,1198-1238,1417-1466` | 选项默认值、`longopts` 与 `-h` 的帮助文本 |
| `nccl-tests/src/common.cu:928-959,1134-1145` | `setupArgs`、`TimeTest`、`AllocateBuffs` |
| `nccl-tests/src/common.cu:720-887` | `BenchTime`：预热、主机计时、校验、输出 |
| `nccl-tests/src/common.cu:554-645`；`nccl-tests/src/all_reduce.cu:500-508` | `startColl` 组织两卡调用；默认 `AllReduceRunColl` 调用 `ncclAllReduce` |
| `nccl-tests/src/timer.cc:8-27` | `now`、`timer::elapsed`：steady_clock |
| `nccl-tests/src/common.cu:324-375,496-548` | `Allreduce` 耗时归约、`testStreamSynchronize` |
| `nccl-tests/src/util.cu:501-504,594-607,837-880` | `writeBenchmarkLinePreamble`、`writeBenchmarkLineBody`、`writeResultHeader`，单位 us |
| `nccl-tests/src/all_reduce.cu:34-64` | `AllReduceGetCollByteCount`、`AllReduceGetBw` |
| `nccl-tests/src/all_gather.cu:10-43`；`nccl-tests/src/reduce_scatter.cu:10-42` | `AllGatherGetBw`、`ReduceScatterGetBw` 及各自 `GetCollByteCount`，分片与 S |
| `nccl-tests/src/sendrecv.cu:33-43,94-108`；`nccl-tests/src/alltoall.cu:43-53` | ip 限制、并发邻居、带宽系数 |
| `nccl-tests/src/common.cu:1576-1584,1657-1661,1836-1842` | `run`：localRank、GPU 与线程映射 |
| `nccl-tests/src/common.cu:1496-1514` | `main`：Graph 与逐迭代计时的互斥处理 |

上游 `nccl-tests/doc/PERFORMANCE.md:18-145` 可辅助理解公式。
但该文开头的 ms 描述与当前输出不符；以 `timeUsec` 和表头 `(us)` 的实现为准。

下一章：[11 性能分析与调优](11-performance.md)，用受控对照检验性能解释。
