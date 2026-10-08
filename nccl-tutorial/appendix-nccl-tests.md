# 附录：读懂 nccl-tests 的输入、计时和输出

nccl-tests 是 NCCL 的调用方：它分配 buffer、创建 communicator、提交操作并校验结果。
它测量选定用例，不模拟整个训练 step。
本文核对 nccl-tests `b4d5bee`（2.20.0）及 NCCL `12df1a11`（2.32.3）。
主场景是两机各四 GPU、八 rank，每 rank 64 MiB FP32 Sum AllReduce；布局为教学假设。
本机为无 CUDA 的 macOS，所有构建和 GPU 命令均为待读者实验。

## 1. 构建调用方，确认加载的是目标库

命令在 Linux GPU 节点的 Bash 中，从读者自己的 `nccl_learn` 根目录执行。
先按[环境参考](00-environment.md)准备驱动、toolkit、编译器与 Python 3。
下面生成未修改的基线库；第 11 章的观察版本使用另一构建目录。

```bash
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB="${CUDA_LIB:-$CUDA_HOME/lib64}"
make -C "$ROOT/nccl" -j 8 src.build CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_LIB"
make -C "$ROOT/nccl-tests" -j 8 MPI=0 \
  CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_LIB" NCCL_HOME="$ROOT/nccl/build" \
  BUILDDIR="$ROOT/nccl-tests/build"
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$ROOT/nccl-tests/build/all_reduce_perf"
```

`NCCL_HOME` 给编译器头文件和链接搜索路径，`LD_LIBRARY_PATH` 决定运行时库搜索。
核对 `ldd` 中的 NCCL 路径；启动报告的 `nccl-headers` 和 `nccl-library` 在本版应为 23203。
同一版本号仍可有不同补丁，构建/加载身份的完整核验见[第 11 章](11-development-workflow.md)。

## 2. 一个固定 64 KiB 的合法用例

先用单机两张已分配的 GPU 缩小测试；设备对当前进程共同可见。
此时 P=2，64 KiB FP32 对应每 rank 16384 个元素。
以下用默认普通 host API，不启用 device API、Graph 或逐迭代阻塞模式。

```bash
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 64K -e 64K -t 1 -g 2 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

| 参数 | 含义 |
| --- | --- |
| `-b/-e` | 最小/最大请求字节数；相等时只测一个尺寸 |
| `-t/-g` | 每进程 CPU 工作线程数/每线程管理的 GPU 数 |
| `-w/-n` | 每种缓冲模式的预热次数/计时批次中的迭代次数 |
| `-c` | 计时后的校验轮数，0 表示关闭校验 |
| `-a` | 耗时聚合：0 为输出主进程的 thread 0 口径，1 平均，2 最小，3 最大 |
| `-d/-o` | 数据类型与规约操作，本例固定 float/sum |

`-a 3` 先取各进程/线程的批次平均耗时最大值，再换算带宽。
它不是取最慢的一次迭代；一个线程管理多卡时，会等待它负责的全部 stream。
K/M/G 后缀使用 1024 的幂，输出 GB/s 使用十进制 `10^9 B/s`。

## 3. 主表一行在说什么

```text
                               out-of-place                 in-place
size count type redop root     time algbw busbw #wrong       time algbw busbw #wrong
(B)  (elements)                (us) (GB/s)(GB/s)              (us) (GB/s)(GB/s)
```

`size` 是有效发送/接收字节数中的较大者，`count` 是该操作的元素参数，`type` 决定元素大小。
AllReduce 的 size=count×类型字节数；主场景为 67108864 B、16777216 个 float。
`redop=sum` 表示求和；AllReduce 没有根，`root=-1` 是正常输出。
其他 collective 的 count 含义要按操作查[操作附录](appendix-collectives.md)。

oop 即 out-of-place，输入输出分离；ip 即 in-place，按 API 约定复用存储。
AllReduce 的 ip 令输入输出同址，两组都是完整 communicator 的结果，并非分别代表两张卡。
程序先测 oop 再测 ip；测试分配的辅助 buffer 不会因为 ip 列而自动消失。
`#wrong` 是校验错误元素数，检查两组为 0、错误摘要正常且进程成功退出；N/A 表示未报告校验。

`time` 是单次操作的批次平均时间，单位微秒；`algbw = S / T / 10^9`。
对 AllReduce，S 是每 rank 的完整数组字节数，`busbw = algbw × 2(P−1)/P`。
按 Ring 理解：两个阶段各 P−1 步，每步约 S/P，所以每 rank 归一化流量是 `2(P−1)S/P`。
tests 固定使用此系数，实际选 Tree 时也不会换公式；busbw 是归一化指标而非 NIC 计数器。

教学算例：P=8、S=64 MiB、T=2 ms，则 algbw≈33.55 GB/s，busbw≈58.72 GB/s。
这里的时间为假设值，不是实测；系数为 1.75，不再乘八个 rank。
比较结果时对齐尺寸与缓冲模式，`Avg bus bandwidth` 是受扫描范围影响的汇总值。

## 4. 默认计时边界：批量提交，加最后一次等待

从 `common.cu:720` 的 `BenchTime` 按执行顺序读，默认 `agg_iters=1`：

1. 722–725 行按校验需求准备数据，727–731 行执行 `-w` 次预热并等待完成。
2. 733 行做进程/线程对齐；756 行构造主机 `timer`，从这里开始主计时。
3. 757–771 行批量提交 `iters × agg_iters` 次操作，默认就是 `-n` 次。
4. 797 行 `completeColl` 等待关联 stream，800 行读取 elapsed 并除以操作数。
5. 811 行按 `-a` 聚合耗时，再计算带宽；828–876 行重新准备输入、通信并校验。

`timer.cc` 使用 `std::chrono::steady_clock`，默认 time 覆盖主机提交和完成等待。
初始化、预热、计时前 barrier 及计时后校验都在这个主计时区间之外。
连续提交后统一等待，使它成为稳态批次平均，而非每次提交后同步的孤立延迟。
计时会包含该批次实际发生的准备开销；预热充分有助于减少首次路径对稳态结果的影响。

`-C` 可改变显示的时间口径，`-G` 引入 Graph 重放，`-I` 增加逐迭代 event 统计。
采用这些模式时需重新核对测量区间，本附录命令保持默认模式。
地址轮换也影响缓存工作集，固定尺寸与扫描的比较方法见[性能附录](appendix-performance.md)。

## 5. MPI 进程数怎样变成八个 NCCL rank

未启用 tests 特殊 split 分组时，总 rank 数 `P=np×t×g`。
逻辑 rank 为 `(MPI rank×t+thread)×g+gpu`，默认设备号为 `localRank×t×g+thread×g+gpu`。
localRank 是本节点内的进程序号，设备号相对于该进程可见的 CUDA 设备列表。

以下假设两机分别为 `gpu-a/gpu-b`，每机四卡对本机四个进程共同可见。
两机使用相同绝对 ROOT 和构建，MPI_HOME 为本站 Open MPI 安装前缀；命令沿用前面的 Bash 变量。

```bash
: "${MPI_HOME:?请设置本站 Open MPI 安装前缀}"
make -C "$ROOT/nccl-tests" -j 8 MPI=1 MPI_HOME="$MPI_HOME" \
  CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_LIB" NCCL_HOME="$ROOT/nccl/build" \
  BUILDDIR="$ROOT/nccl-tests/build-mpi"
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_LIB:$MPI_HOME/lib:$MPI_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_ENQUEUE_REARCH_ENABLE=0 \
  "$MPI_HOME/bin/mpirun" -np 8 --host gpu-a:4,gpu-b:4 \
  --map-by ppr:4:node --bind-to none --wdir "$ROOT" \
  -x LD_LIBRARY_PATH -x NCCL_ALGO -x NCCL_PROTO -x NCCL_ENQUEUE_REARCH_ENABLE \
  "$ROOT/nccl-tests/build-mpi/all_reduce_perf" \
  -b 64M -e 64M -t 1 -g 1 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

每节点 localRank 0–3 分别使用可见设备 0–3；核对启动报告中的 rank、hostname 和 PCI bus ID。
若调度器逐进程只暴露独占单卡，在 `t=g=1` 下转发 `NCCL_TESTS_DEVICE=0`，匹配这种映射。
共同可见四卡时则沿用默认偏移；两种模式选择其一。CPU 绑定按站点要求固定并记录。
这里限制 Ring/Simple 只为与正文追踪对齐；实际性能基线也应保留自动选择结果。

## 6. 本附录任务：用单变量检验计时解释

先保存第 2 节固定 64K 用例，保持设备、库、oop/ip 和其余参数一致。
第一组只把 `-w 5` 改为 `-w 1`，两次都保留 `-n 20`；解释首次路径残留和稳态波动。
第二组回到 `-w 5`，只把 `-n 20` 改为 `-n 100`；解释批量平均的摊销与波动变化。
每组独立重跑数次，核对校验和退出状态，再按输出 time 手算 algbw/busbw。
预热与校验不在主计时内，因此不能把完整进程运行时间直接除以 `-n` 来对拍 time。
交付两组记录及各自能解释的范围；这不是用增加迭代数证明算法变快。

## 7. 六个源码锚点

| 路径:行 | 符号与用途 |
| --- | --- |
| [nccl-tests/src/Makefile:24](../nccl-tests/src/Makefile#L24) | `NCCL_HOME`、`MPI`：头文件与链接路径 |
| [nccl-tests/src/common.cu:720](../nccl-tests/src/common.cu#L720) | `BenchTime`：预热、计时、校验边界 |
| [nccl-tests/src/timer.cc:8](../nccl-tests/src/timer.cc#L8) | `now`、`timer::elapsed`：steady_clock |
| [nccl-tests/src/common.cu:324](../nccl-tests/src/common.cu#L324) | `Allreduce`：先聚合耗时再计算带宽 |
| [nccl-tests/src/all_reduce.cu:34](../nccl-tests/src/all_reduce.cu#L34) | `AllReduceGetCollByteCount`；59 行 `AllReduceGetBw` |
| [nccl-tests/src/util.cu:719](../nccl-tests/src/util.cu#L719) | `writeDeviceReport`：版本、rank、可见设备映射 |

执行失败时回到[12 验证与调试](12-testing-and-debugging.md)，分析性能时转到[性能附录](appendix-performance.md)。
