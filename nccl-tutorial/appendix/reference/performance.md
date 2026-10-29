# 附录：从性能数字回到执行原因

性能分析先固定工作量，再问时间花在哪一段。
主场景为两机各四 GPU、八 rank，每 rank 64 MiB FP32 Sum AllReduce；节点与 rank 布局是教学假设。
本文使用 NCCL `12df1a11`（2.32.3）和 nccl-tests `b4d5bee`（2.20.0）。
本机 macOS 无 CUDA，数值算例为教学构造，GPU 和 profiler 命令均为待读者实验。
计时和带宽列的精确定义见[tests 附录](nccl-tests.md)。

> 阅读定位：进阶性能分析。先理解 nccl-tests 的输出和计时边界，再做同参数 baseline/profile，不建议只拟合公式而没有对应执行证据。

## 1. 小包看固定成本，大包看有效吞吐

固定 rank 数、拓扑、算法、协议和测量方式后，可在一段尺寸区间内近似：

```text
T(S) ≈ α + S/B
S：每 rank 的 AllReduce 数组字节数
α：提交、启动、同步等与尺寸弱相关的成本
B：使用同一 S 定义的有效算法带宽
```

小包的 `S/B` 很小，时间主要由 α 决定，增加消息量会提高 algbw 而未必显著增加 time。
大包的 `S/B` 占主导，时间近似随字节数增长，algbw 逐渐接近 B。
转折规模约为 `αB`；改变 rank 数或算法后，α 和 B 都可能变化。
B 已经综合了内存访问、规约、通信步数和共享链路的影响，不等于 NIC 标称线速。

### 两点拟合，再让第三点检验解释

假设同一八 rank Ring/Simple 路径得到以下教学数据：

| 每 rank 尺寸 | 时间 | 用途 |
| --- | --- | --- |
| 8 MiB | 430 us | 拟合点 1 |
| 32 MiB | 1690 us | 拟合点 2 |
| 64 MiB | 3390 us | 独立核对点 |

先换单位：`B=(33554432−8388608)/(0.001690−0.000430)≈19.97×10^9 B/s`。
再算 `α=430 us−8388608/B=10 us`。
于是 64 MiB 的预测时间为 `10+3360=3370 us`，第三点残差为 `3390−3370=20 us`。
两个拟合点总能定出一条直线，第三点才开始检验该解释能否外推。

用重复运行的波动判断这 20 us 是否值得解释。
若残差持续超出波动，查看选择日志中的算法、协议、channel 预算，以及频率和共享资源变化。
选择切换会改变曲线斜率，跨切换点拟合得到的负截距通常没有启动延迟的物理意义。
NCCL 自身 tuning 也使用延迟项加数据量/带宽项，但还叠加拓扑、架构和候选约束。

## 2. 比较前先固定“测的是什么”

保存库身份、CUDA/驱动、MPI/插件、GPU/NIC 拓扑和 rank 到设备的映射。
工作量记录 size 范围、datatype/op、oop/ip、预热、迭代、校验和 `-a` 口径。
同时记录 CPU 绑定、其他作业负载以及实际算法/协议；相同环境变量不总等于相同选择。
默认 tests 的 time 是提交到完成的主机批次平均，用它回答稳态通信问题。

先比较相同 size、相同缓冲模式的 time，再解释 algbw/busbw。
独立重跑数次，保留中位数与波动；扫描范围不同的 `Avg bus bandwidth` 不适合直接排名。
观察构建的 INFO 日志和 profiler 都可能改变时序，性能基线使用相同的低日志级别。
第 10 章的 task 日志能证明选择；最终 channel 分配还需后续 work 或 profiler 信息。

### 固定尺寸与扫描：同一 8 MiB 行也可能不同

`startColl` 先计算 `totalnbytes=max(sendBytes, expectedBytes)`。
非零消息时，`steps=maxbytes/totalnbytes`，`shift=totalnbytes×(iter%steps)`。
send/recv 指针都加上 shift，这使批次中的调用轮换访问不同 buffer 槽位。

假设程序实际 maxbytes 未因显存限制缩小，AllReduce 固定 `-b 8M -e 8M` 时 steps=1。
扫描 `-b 8 -e 256M -f 2` 的 8 MiB 行则有 32 个可轮换槽位。
`-n 20` 的默认计时批次访问其中 20 个槽位，每个参与轮换的 buffer 覆盖 160 MiB。
所以固定测试的地址复用与扫描测试的工作集不同，即使显示的 size 都是 8 MiB。
用固定尺寸的 profile 减去扫描中同尺寸的 time，会把工作集差异也算成 profiler 开销。

## 3. 本附录任务：同参数 baseline/profile 对照

在主场景的八 rank 映射上，把消息暂缩到 8 MiB，便于阅读时间线。
先完成 tests 附录的 MPI 构建和正确性检查，保持每节点四卡共同可见。
以下为 Open MPI 示例：替换 `gpu-a/gpu-b`，两机使用相同绝对 ROOT、库和 nsys 安装。
在 Linux GPU 启动节点的 Bash 中，从读者自己的 `nccl_learn` 根目录执行。
每个远端进程在自己的 ROOT 下创建独立报告目录，ROOT 在各节点都需可写。

```bash
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB="${CUDA_LIB:-$CUDA_HOME/lib64}"
: "${MPI_HOME:?请设置本站 Open MPI 安装前缀}"
launch=(env "ROOT=$ROOT"
  "LD_LIBRARY_PATH=$ROOT/nccl/build/lib:$CUDA_LIB:$MPI_HOME/lib:$MPI_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  NCCL_DEBUG=WARN NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_ENQUEUE_REARCH_ENABLE=0
  "$MPI_HOME/bin/mpirun" -np 8 --host gpu-a:4,gpu-b:4
  --map-by ppr:4:node --bind-to none --wdir "$ROOT"
  -x ROOT -x LD_LIBRARY_PATH -x NCCL_DEBUG -x NCCL_ALGO -x NCCL_PROTO -x NCCL_ENQUEUE_REARCH_ENABLE)
bench=("$ROOT/nccl-tests/build-mpi/all_reduce_perf"
  -b 8M -e 8M -t 1 -g 1 -d float -o sum -w 5 -n 20 -c 1 -a 3)
# 无 profiler 基线。
"${launch[@]}" "${bench[@]}"
# 相同 workload，只加入 profiler；每个 rank 得到独立 mktemp 目录。
"${launch[@]}" bash -c '
  TRACE_DIR="$(mktemp -d "$ROOT/nccl-profile.XXXXXX")" || exit 1
  printf "rank=%s trace=%s\n" "$OMPI_COMM_WORLD_RANK" "$TRACE_DIR"
  exec nsys profile --trace=cuda,nvtx,osrt --sample=none \
    --output="$TRACE_DIR/allreduce" "$@"
' _ "${bench[@]}"
```

两次使用同一 launch/bench 数组，尺寸范围、库、映射和测试选项保持一致。
核对各自正确性，再打开各 rank 的报告；保留命令输出中报告目录与 rank 的对应关系。
把 timeline 分为初始化、数据准备、预热、计时和校验，主表 time 只对应计时批次。
先看 CUDA launch 是否迟到，再看 stream 依赖，最后看 NCCL kernel 与 proxy 活动。

跨节点时间戳需要时钟对齐信息；先在每个 rank 内分析因果，再按同一操作关联。
CUDA timeline 能证明时间关系，NET 日志和端口计数器补充网络请求与链路证据。
任务交付是同参数 time 对照、阶段划分，以及一段等待的上游原因；不把长 kernel 直接命名为网络瓶颈。
回到每 rank 64 MiB 时，两次都把 bench 的 `-b/-e` 改为 64M，继续保持成对比较。

## 4. 从强制对照回到自动选择

上面的 Ring/Simple 限制只用于关联普通 host API 的具体实现，不是生产配置推荐。
评估实际收益时另存自动选择基线，并按一次一个限制建立对照：
自动 → 仅 Ring → Ring/Simple；最后一对只改变协议限制。
若直接比较自动与 Ring/Simple，算法和协议同时变化，解释范围应是整个组合。
每次记录实际选择；候选不可用时记录该结果，不把失败当成零带宽。

性能差异随后按资源解释：GPU/NIC 绕行增加共享链路压力，proxy CPU 迟到会推迟 NET 进度。
更多 channel/CTA 可能提高纯通信吞吐，也会占用 SM、内存带宽和缓存。
只有在目标 workload 上减少完成时间，资源调整才算有效。

## 5. 训练 overlap 看关键路径，而非简单取 max

bucket 是把若干梯度合成一次通信的缓冲；关键路径是决定本轮最终完成时刻的依赖链。
继续以每 bucket 64 MiB、八 rank AllReduce 为例，假设每次通信耗时 3 ms。
计算总共 8 ms，两个 bucket 分别在 2 ms 和 6 ms 就绪，放在同一通信 stream。
第一次通信运行于 2–5 ms，第二次运行于 6–9 ms，优化器最早在 9 ms 使用全部结果。
而 `max(计算8 ms, 通信总计6 ms)=8 ms`，漏掉了第二个 bucket 的释放时刻。

这就是 overlap 要看依赖和就绪时间的原因；出现重叠也可能因资源竞争让计算与通信都变慢。
纯通信 busbw 提高，但最后一个 bucket 更晚完成，训练 step 仍可能退化。
最终验收记录 bucket 就绪、通信结束和结果消费者开始的时间，而不只看通信内核重叠比例。

## 6. 四个源码锚点

| 路径:行 | 符号与解释范围 |
| --- | --- |
| [nccl/src/tuning/tuning_general.cc:52](../../../nccl/src/tuning/tuning_general.cc#L52) | `ncclTuningGetTime`：延迟项加数据/带宽项 |
| [nccl-tests/src/common.cu:554](../../../nccl-tests/src/common.cu#L554) | `startColl`：totalnbytes、steps、shift 与指针轮换 |
| [nccl-tests/src/common.cu:720](../../../nccl-tests/src/common.cu#L720) | `BenchTime`：profile 阶段与主表计时边界 |
| [nccl-tests/src/all_reduce.cu:59](../../../nccl-tests/src/all_reduce.cu#L59) | `AllReduceGetBw`：algbw 与固定 busbw 系数 |

选择机制回查[04 算法](../../04-algorithms.md)，网络推进回查[08 proxy 与网络](../../08-proxy-and-network.md)，状态停滞回查[11 验证与调试](../../11-testing-and-debugging.md)。
