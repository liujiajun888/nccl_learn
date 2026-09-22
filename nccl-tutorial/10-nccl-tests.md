# 第 10 章：用 nccl-tests 建立可信的通信基线

> 基线：NCCL `12df1a11`（2.32.3），nccl-tests `b4d5bee`（2.20.0）。
> 所有 shell 示例都从仓库根目录 `nccl_learn`、在 Linux NVIDIA GPU 节点执行。
> `ROOT` 可由用户预设为该节点上的仓库根目录；否则取 `$PWD`。不要照搬其他机器的路径。
> 本章只提供教学步骤；当前编写环境没有可用 NVIDIA GPU，未构建或实测，不提供虚构成绩。

## 学习目标

- 能分别构建单进程版和 MPI 版，并确认运行时加载的是目标 NCCL。
- 能解释每一列，区分测试 size、输出 count 和 NCCL API 的 count。
- 能说明预热、计时、校验、跨 rank 统计分别发生在哪里。
- 能手算算法带宽与总线带宽，并设计不同通信操作的对照实验。

先修：[环境准备](00-environment.md)、[传输路径](07-transports.md)。
算法与协议背景见 [第 8 章](08-algorithms.md)、[第 9 章](09-device-protocols.md)。

## 1. 先把“测到了哪个库”说清楚

nccl-tests 是调用 NCCL 的测试程序，不是 NCCL 库本身，也不是训练框架。
单进程可以管理多张 GPU；跨进程、跨机需要 MPI 构建及正确的 launcher。
先准备 Linux 驱动、CUDA toolkit、C++ 编译器、make 和 Python 3（NCCL 设备代码生成需要 `python3`）；MPI 仅多进程实验需要。

```bash
# Linux NVIDIA GPU；当前目录必须是 nccl_learn。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
nvidia-smi
"$CUDA_HOME/bin/nvcc" --version
make -C "$ROOT/nccl" -j"$(nproc)" src.build CUDA_HOME="$CUDA_HOME"
make -C "$ROOT/nccl-tests" -j"$(nproc)" MPI=0 \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$ROOT/nccl/build" \
  BUILDDIR="$ROOT/nccl-tests/build"
LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$ROOT/nccl-tests/build/all_reduce_perf"
```

NCCL 默认产物在 `nccl/build`，不是源码目录下的 `lib`；检查 `ldd` 中 `libnccl.so` 的实际解析路径。
`NCCL_HOME` 提供头文件和链接搜索路径，**不会替你设置 NCCL 的运行时 rpath**；漏设可能变成“新头文件、旧运行库”。
后续命令用进程局部的 `LD_LIBRARY_PATH`，不修改系统库或 shell 启动文件。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；MPI_HOME 先由用户设为真实安装前缀。
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${MPI_HOME:?请先设置本站 MPI 的安装前缀}"
make -C "$ROOT/nccl-tests" -j"$(nproc)" MPI=1 \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$ROOT/nccl/build" MPI_HOME="$MPI_HOME" \
  BUILDDIR="$ROOT/nccl-tests/build-mpi"
```

默认 `MPI=0`；MPI 版放进 `build-mpi`，避免复用未启用 MPI 的旧对象。
`MPI_INCLUDE` 可单独覆盖头文件位置；非标准安装还须核对库目录，以及 MPI 编译器、头文件、libmpi、mpirun 的兼容性。
运行时每节点都要能找到 NCCL、CUDA、MPI 动态库；错误应从参数、`ldd`、版本定位，不删除系统库。

## 2. 第一次实验：AllReduce 从小消息扫到大消息

至少分配两张可用 GPU；`0,1` 指当前作业可见范围内的设备，须与调度器授权一致。
不要覆盖调度器的独占卡映射去占用别人的 GPU。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；本例要求当前作业已获准使用两张卡。
ROOT="${ROOT:-$PWD}"
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 8 -e 256M -f 2 -g 2 -t 1 -d float -o sum -w 5 -n 50 -c 1 -a 3
```

这里刻意写出预热和迭代数便于复现，不代表上游默认值；首次保留校验，显存不足则减小 `-e`。
程序也可能依据显存减少 maxBytes；记录开头提示，确认实际扫到了哪里，不把 OOM 当作算法失败。
预期小消息带宽低、大消息逐渐趋向平台；算法、协议、channel 或分块策略切换也可能造成台阶或回落。
这些趋势取决于设备和拓扑，不承诺单调，更不是本教材的实测结果。

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

## 3. 输出每一列究竟是什么

默认文本表头如下；这里没有填入任何伪造的测量行。

```text
                                      out-of-place                    in-place
size  count  type  redop  root         time algbw busbw #wrong          time algbw busbw #wrong
(B)   (elements)                       (us) (GB/s)(GB/s)                (us) (GB/s)(GB/s)
```

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

左右各有一套 `time/algbw/busbw/#wrong`，不是两次同时进行的通信。
页尾 `Out of bounds values` 是错误检查摘要；通过不能证明所有输入都正确。
`Avg bus bandwidth` 是所测项目的汇总平均，强烈依赖扫描范围和项目构成，不是峰值。
一个覆盖大量小消息的扫描，其平均值自然可能低于仅测大消息的扫描。

进阶选项会扩展表头；不打开它们，默认没有以下附加列：
- `-C 1`：把 `time` 显示为 `cputime`；带宽仍按完成时间计算，不能拿 cputime 反推它。
- `-S 1`：增加 `timestamp`，供日志关联，不是通信持续时间。
- `-I 1`：增加 `i_min/i_max/i_p99/i_cv%`，分别为逐迭代最小、最大、99 分位和变异系数。
- `-U 1`：增加 `impl/algo/proto/kernelVariant/sync/#channels/#warps/netChunkSize`。
- 这些 tuning 列依次描述实现、算法、协议、内核变体、同步方式、channel 数、warp 数和网络块大小。
- tuning 信息依赖 profiler 事件；`N/A` 不表示该资源为零，观察模式也不应混入最终性能基线。

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

## 4. in-place 不是“这次测试只分配一份显存”

Out-of-place（oop）使用分离的输入与输出；in-place（ip）按 API 约定复用缓冲区。
AllReduce 可以令输入输出为同一地址；AllGather 的输入位于输出中本 rank 的分片位置。
ReduceScatter 的输出则位于输入中本 rank 对应的分片位置，不可随便重叠。
nccl-tests 先测 oop，再测 ip；`AllocateBuffs` 仍然分配 sendbuff、recvbuff，校验还要 expected。
因此不能从 ip 一列推断测试进程显存需求减半，也不能保证 ip 一定更快。

本版 **SendRecv 和 AlltoAll 不支持 in-place 正确性测试**，该列校验不报告。
即使右侧仍打印了时间，也不应将其当作受支持的原地语义或正确性结果。
比较这两项时只采用 oop 的有效数据，检查该列的 `#wrong`。

## 5. 默认计时：主机时钟覆盖提交到 stream 完成

按 `BenchTime` 阅读一次 oop 或 ip 测试的生命周期：
1. 准备数据，执行预热，等待预热 stream 完成，再做进程／线程对齐。
2. 创建主机 `timer`，执行 `iters × agg_iters` 次通信提交。
3. `completeColl` 等待相关 CUDA stream 完成；然后读取主机时间。
4. 除以通信次数，对各进程／线程耗时按 `-a` 归约，计算带宽。
5. 重新初始化数据，另外运行 `-c` 指定的校验轮数，检查输出。

默认不是“只量 NCCL API 返回时间”，也不是“用 CUDA event 量内核”。
主机 timer 基于 `std::chrono::steady_clock`，等待通过 `cudaStreamQuery` 等完成检查实现。
默认批量连续提交并在末尾等待，所以它是批次平均，不等于每轮都同步的孤立延迟。
`-C 1` 才把等待之前的主机提交口径显示出来；它也不一定是纯 API 开销。
初始化、预热和计时后的校验不计入默认主计时区间。

`-m 8` 用 group 聚合更多操作，可能摊薄提交成本，也可能改变调度；不能和默认混比。
`-G` 先捕获并实例化 graph，然后重新开始计时重放，结果不含捕获／实例化成本。
它适合研究重放吞吐，不等于普通训练程序的逐次提交性能。
`-I 1` 才额外记录 CUDA events；`-K` 只跳过逐迭代统计的前 K 项，不改主表总平均。
本版 `-G` 会禁用 `-I`，也会禁用 `-z 3`；看到提示要记录实际生效模式。

## 6. GPU、线程、MPI rank 怎么对应

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

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；已分配两张卡，使用默认 host API 实现。
ROOT="${ROOT:-$PWD}"
for op in all_gather reduce_scatter sendrecv; do
  env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ROOT/nccl-tests/build/${op}_perf" \
    -b 1K -e 256M -f 2 -g 2 -t 1 -d float -w 5 -n 50 -c 1 -a 3
done
```

SendRecv 是每 rank 向下一个发送、从上一个接收的并发环，不是串行 ping-pong；对比 AR 可区分单跳与多阶段成本。
AG/RS 大消息时间可能接近，但归约计算、内存方向与算法会使两者不同；AR 语义上等于 RS+AG，不保证独立测试时间能精确相加。
记录完整命令、版本、拓扑、映射、实际 size、oop/ip、计时模式和正确性结果。
保存多次独立结果后再去 [性能分析](11-performance.md) 建模，不要只摘最大 busbw。

## 9. 源码锚点：沿这些符号复核

以下路径均相对仓库根目录，行号对应本章固定提交。

| 路径与行号 | 符号／阅读目的 |
| --- | --- |
| `nccl/README.md:15-33` | Build：`src.build`、CUDA_HOME、默认 build 目录 |
| `nccl-tests/src/Makefile:12-39,127-130` | `MPI`、`NCCL_HOME`、链接规则，无 NCCL rpath |
| `nccl-tests/src/common.cu:89-122,1198-1238` | 选项默认值与 `longopts` |
| `nccl-tests/src/common.cu:928-959,1134-1145` | `setupArgs`、`TimeTest`、`AllocateBuffs` |
| `nccl-tests/src/common.cu:720-887` | `BenchTime`：预热、主机计时、校验、输出 |
| `nccl-tests/src/timer.cc:8-27` | `now`、`timer::elapsed`：steady_clock |
| `nccl-tests/src/common.cu:324-375,496-548` | `Allreduce` 耗时归约、`testStreamSynchronize` |
| `nccl-tests/src/util.cu:594-607,837-880` | `writeBenchmarkLineBody`、`writeResultHeader`，单位 us |
| `nccl-tests/src/all_reduce.cu:34-64` | `AllReduceGetCollByteCount`、`AllReduceGetBw` |
| `nccl-tests/src/all_gather.cu:10-43`；`nccl-tests/src/reduce_scatter.cu:10-42` | `AllGatherGetBw`、`ReduceScatterGetBw` 及各自 `GetCollByteCount`，分片与 S |
| `nccl-tests/src/sendrecv.cu:33-43,94-108`；`nccl-tests/src/alltoall.cu:43-53` | ip 限制、并发邻居、带宽系数 |
| `nccl-tests/src/common.cu:1576-1584,1657-1661,1836-1842` | `run`：localRank、GPU 与线程映射 |
| `nccl-tests/src/common.cu:1496-1514` | `main`：Graph 与逐迭代计时的互斥处理 |

上游 `nccl-tests/doc/PERFORMANCE.md:18-145` 可辅助理解公式。
但该文开头的 ms 描述与当前输出不符；以 `timeUsec` 和表头 `(us)` 的实现为准。

## 10. 自测题与答案

1. P=8、float、AllGather 的 count=1024，S 是多少？busbw/algbw 是多少？
   **答：**S=`8×1024×4=32768 B`，是每 rank 完整输出；比例为 `7/8`。
2. 默认 `time` 是 CUDA event 结果吗？`-c 3` 是否把三轮校验加入它？
   **答：**不是；默认是提交到 stream 完成的主机批次平均，校验轮次在计时之后。
3. 两个 MPI 进程都能看见 GPU 0/1，能否统一设 `NCCL_TESTS_DEVICE=0`？
   **答：**不能，这会使它们指向同一张卡；只有分别隔离为独占单卡且 `t=g=1` 才可这样使用。
