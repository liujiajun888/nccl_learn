# 12 验证与调试：沿执行阶段缩小问题

[上一章](11-development-workflow.md)增加了 task 选择日志。
本章验证三件事：加载了改动后的库，记录了正确状态，原来的通信仍然算对并完成。
主场景仍是两机各四 GPU、八 rank，每 rank 64 MiB FP32 Sum AllReduce，按 Ring/Simple 追踪。
机器布局是教学假设；当前源码基线为 NCCL `12df1a11`（2.32.3），tests 为 `b4d5bee`（2.20.0）。
本机 macOS 无 CUDA，以下 GPU 命令均为待读者实验。

## 1. 先找最后一个已经完成的阶段

一次 collective 横跨应用线程、NCCL 主机工作、GPU 和网络后端。
“API 返回了”“kernel 出现了”“网络请求完成了”分别证明不同阶段前进。
定位时先记录最后一个确定完成的状态，再检查下一个状态的生产者。

| 阶段 | 可以获得的证据 | 下一入口 |
| --- | --- | --- |
| init/bootstrap | 进程/GPU 映射、INIT/BOOTSTRAP 日志、初始化返回与异步状态 | `ncclCommInitRankFunc`、`bootstrapInit` |
| 连接准备 | channel/peer、实际 transport、setup/connect 与注册结果 | `ncclPrepareTasksAndCollPreconnectFunc`、`selectTransport` |
| host 提交 | task 选择、plan、CUDA launch API 及其返回 | `ncclLaunchPrepare`、`ncclLaunchKernel` |
| GPU 等待 | stream 依赖、kernel 时间段、primitive 等待条件 | Ring `runRing`、Simple `waitPeer` |
| NET 请求 | proxy 是否提交请求、请求句柄、完成状态与端口错误 | `sendProxyProgress`、`recvProxyProgress`、net plugin |
| 完成/回收 | CUDA completion、NCCL 异步状态、校验结果、回收回调 | `testStreamSynchronize`、`ncclCommGetAsyncError`、`reclaimPlan` |

这是依赖顺序，不是要求所有日志严格按表排序。
连接可能延迟到第一次使用，多个 channel 的 GPU 与 proxy 工作也会并行推进。
寻找最早的状态分歧，通常比寻找最后一条报错更接近原因。

## 2. init 和连接准备：先证明参与者与路径成立

bootstrap 是 rank 相互认识并交换引导信息的过程。
先核对八个 rank 是否使用同一逻辑 communicator、设备映射是否符合两机各四卡。
保存 hostname、进程号、rank、CUDA device 和 PCI bus ID，跨进程用逻辑编号关联记录。
每进程的 comm 指针只在本进程有意义。

初始化返回失败，就从该 rank 最早的 CUDA、系统或 bootstrap 错误向前追。
非阻塞初始化还需等待 communicator 的异步状态就绪。
初始化成功后，首次 collective 仍可能准备算法连接与内存注册。
因此“初始化通过、第一次调用未完成”要继续看 preconnect，而不是直接跳到 kernel。

transport 负责实际连接，不能从 `NCCL_ALGO=Ring` 推出它用了哪块 NIC。
Ring 描述通信关系，同一个 ring 的本机边可以走 P2P，跨机边可以走 NET。
从连接日志核对 rank/peer/channel，再看 P2P、SHM、NET 及具体网络后端的选择。
`NET/IB`、Socket 或插件名说明后端；GDR 注册与连接信息帮助确认 GPU 内存路径。
GDR 即 GPUDirect RDMA，让 NIC 能直接访问已注册的 GPU 内存。

bootstrap 使用的 IP 接口与传送 payload 的 RDMA 端口可以不同。
一条“网络插件已加载”只证明加载成功，实际连接日志才说明该连接用了谁。
按日志对照只读的 GPU/NIC 拓扑、端口状态、容器设备可见性与资源限额。
不通过修改 ACS/IOMMU、防火墙或隔离配置来掩盖问题。

## 3. host 提交和 GPU 等待：把选择与执行分开

`[tutorial-task]` 出现，证明 task 已得到算法、协议和资源预算。
它所在的位置早于最终 channel 划分、连接完成和 kernel 启动。
若日志存在而时间线没有对应 kernel，下一站是 plan 生成、连接准备和 CUDA launch。
先查这些入口的返回值，再检查应用是否结束了外层 group。

Nsight Systems 的 CUDA timeline 是按时间排列的 API、stream 和 GPU 活动。
它能区分“CPU 迟迟没提交”“已提交但受 stream 依赖阻挡”“kernel 已在 GPU 上运行”。
例如同一 stream 前面的计算尚未结束，NCCL kernel 晚开始就是明确的依赖结果。
若主机 launch 本身迟到，则看应用调度、group 边界或准备线程，而不是增加通信 CTA。
CTA 是 CUDA thread block，增加它无法补回尚未提交的工作。

一个长 NCCL kernel 可能正在等待 peer 或网络，持续时间不等于有效传输时间。
在 Simple 的 `waitPeer` 中，接收方向等数据发布，发送方向等可复用槽位。
credit 指消费者归还的可写空间；缺 credit 的发送者要等消费者前进。
找到等待的是哪条连接、哪个 step，再去看对端或对应 proxy 的推进状态。

断点适合确认调用分支、对象字段和当前线程。
停住参与者会改变其他 rank 的等待时间，因此断点用于结构判断，插桩用于连续进度观察。
设备模板常被内联，可先读 host work 的字段，再缩小到具体 primitive。
第 11 章的选择日志不进入 device 热循环，观测范围也更容易界定。

## 4. NET 请求：把未提交与未完成分开

proxy 是为部分传输提供主机侧推进的线程。
普通 NET 路径中，发送 proxy 等待 GPU 发布数据，然后调用网络接口的 `isend`。
接收侧发布 `irecv`，随后通过 `test` 检查请求是否完成，并推进相关可见性和 credit 状态。
这些接口通过函数表调用，下一层可能是内置 IB/Socket，也可能是外部 net plugin。

先看请求是否真正得到句柄：接口成功返回但句柄为空时，这次尝试尚未发布请求。
已有请求而 `test` 持续未完成，则检查对端接收、后端进度和端口错误。
请求完成后 GPU 仍等待，则继续追接收可见性、必要的 flush 和 step 发布。
记录 `channel/step/request/done` 的变化，比只记录“进入 proxy 函数”更能说明进度。

TRACE 可在短复现中提供更多请求事件；端口计数器应比较本次运行前后的增量。
CUDA 时间线通常不能展示完整 RDMA 请求链，NET 日志也不能独自证明链路线速。
把二者按同一操作和连接关联，才知道等待发生在提交前、网络中还是完成发布后。

## 5. 完成与回收：同时检查查询返回和状态

普通 host API 成功表示其主机侧处理成功；输出可用还取决于 CUDA 执行依赖。
`ncclCommGetAsyncError(comm, &state)` 有两个结果：函数返回值说明查询是否成功，`state` 说明 communicator 状态。
`state == ncclSuccess` 也要配合 stream/event completion 才能认定 GPU 输出可消费。

下面是单 communicator 的控制流示意；`fail` 终止正常路径并进入应用错误处理。
`deadline` 使用单调时钟，查询期间由同一责任方保证 comm 存活。

```cpp
ncclResult_t state = submit_or_group_end();
while (state == ncclInProgress) {
  ncclResult_t rc = ncclCommGetAsyncError(comm, &state);
  if (rc != ncclSuccess) fail("async query", rc);
  if (deadline_expired()) fail("host progress deadline");
  yield_cpu();
}
if (state != ncclSuccess) fail("submission", state);
// 主机提交就绪后，才进入依赖它的 CUDA 查询。
for (;;) {
  ncclResult_t rc = ncclCommGetAsyncError(comm, &state);
  if (rc != ncclSuccess) fail("async query", rc);
  if (state != ncclSuccess) fail("asynchronous state", state);
  cudaError_t cu = cudaStreamQuery(stream);
  if (cu == cudaSuccess) break;
  if (cu != cudaErrorNotReady) fail("CUDA execution", cu);
  if (deadline_expired()) fail("completion deadline");
  yield_cpu();
}
```

多 GPU 应检查所有相关 comm 和 stream，group 则检查涉及的全部 communicator。
正常完成后再校验输出、复用用户 buffer；主机 plan 回收与设备 FIFO 槽位复用分别按各自条件推进。
`reclaimPlan` 释放了主机 task，并不单独证明用户输出已经可读。

出现错误后，应用停止继续提交，通过带外控制面协调各 rank，统一管理 abort 与退出。
清理线程与状态查询线程要先交接 comm 的使用权，再执行 `ncclCommAbort`。
恢复成员和应用状态属于应用/作业管理器的职责。

## 6. 本章任务：验证选择日志没有改变通信行为

使用第 11 章保留的基线库和观察库，复用同一个 tests 二进制。
固定 64 KiB、两 rank、FP32 Sum、Ring/Simple、预热和计时次数，仅改变加载库。
命令在 Linux GPU 节点的 Bash 中，从读者自己的 `nccl_learn` 根目录执行。

```bash
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB="${CUDA_LIB:-$CUDA_HOME/lib64}"
LOG_DIR="$(mktemp -d "$ROOT/nccl-verify.XXXXXX")"
run_case() {
  local name="$1" libdir="$2"
  env LD_LIBRARY_PATH="$libdir:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    NCCL_ENQUEUE_REARCH_ENABLE=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,GRAPH,P2P,SHM,NET,TUNING,PROXY \
    NCCL_DEBUG_FILE="$LOG_DIR/$name.%h.%p.log" \
    "$ROOT/nccl-tests/build/all_reduce_perf" \
    -b 64K -e 64K -t 1 -g 2 -d float -o sum -w 5 -n 20 -c 1 -a 3 -T 30 \
    > "$LOG_DIR/$name.stdout" 2> "$LOG_DIR/$name.stderr"
  local status=$?
  printf '%s\n' "$status" > "$LOG_DIR/$name.status"
  return "$status"
}
run_case baseline "$ROOT/nccl/build/lib" && \
run_case observed "$ROOT/nccl/build-observe/lib"
```

NCCL 展开 `%h/%p` 为主机名和 PID；每次使用新目录，因为同名日志按写模式打开。
多机时目录应在各节点可写，逐节点收集日志；容器名称/PID 重复时再加入 launcher 的进程标识。
`-T 30` 只给 tests 的部分 stream 轮询等待限时，整体 deadline 交给作业管理器。
它不覆盖所有初始化、MPI 启动或 barrier，也不是本实验的故障注入手段。

验收先看退出状态、错误摘要与两组 `#wrong=0`，再看观察库的 task 字段。
两个版本的已有 tuning 日志应选择相同算法/协议，观察版另有 count=16384 的新日志。
若新日志缺失，按“库路径—DEBUG 子系统—运行时分支—观察点是否执行”的顺序核查。
随后保留观察构建、将日志级别降到 WARN，与相同日志级别的基线比较重复运行时间。
这样分开验证日志输出和低观测开销下的回归；完整 profile 方法见[性能附录](appendix-performance.md)。

### 按改动收敛回归矩阵

| 维度 | 取哪些合法用例 | 能发现什么 |
| --- | --- | --- |
| rank 数 | 两 rank 起步，再到两机八 rank | 跨 rank 分配、连接及规模分支 |
| count 边界 | 1、分块阈值前后一个元素、16384、16777216 | 小任务、尾块、整块假设；后两项 FP32 大小为 64 KiB、64 MiB |
| datatype/op | 固定 float/sum，再选改动覆盖的类型/操作 | 模板实例、规约语义与支持条件 |
| in-place | tests 的 oop 和 ip 两组 | 输入输出分离或复用时的地址处理 |
| 进程组织 | 相同 GPU 集合的单进程与多进程 | 地址空间、IPC、主机线程和所有权差异 |

仅增加日志时，先完成固定用例和八 rank 主场景；改选择阈值则增加阈值两侧及自动选择。
改 device 分块时扩大 count/datatype/op；改 transport 时扩大跨进程与目标连接组合。
所有 rank 使用匹配的操作、count、datatype 和顺序，通过合法输入变化检查边界。
八 rank 的启动模板见[测试附录](appendix-nccl-tests.md)，保持每 rank 64 MiB 时 count=16777216。
本章交付是基线/观察版的库身份、日志、正确性结果，以及本次实际覆盖的回归范围。

## 7. 六个源码锚点

| 路径:行 | 符号与证据 |
| --- | --- |
| [nccl/src/init.cc:2105](../nccl/src/init.cc#L2105) | `ncclCommInitRankFunc`；3934 行 `ncclCommGetAsyncError`：初始化与异步状态 |
| [nccl/src/group.cc:282](../nccl/src/group.cc#L282) | `ncclPrepareTasksAndCollPreconnectFunc`：选择到连接准备 |
| [nccl/src/enqueue/enqueue.cc:1886](../nccl/src/enqueue/enqueue.cc#L1886) | `ncclLaunchKernel`；1589 行 `reclaimPlan`：提交与主机回收 |
| [nccl/src/device/prims_simple.h:103](../nccl/src/device/prims_simple.h#L103) | `waitPeer`；167 行 `postPeer`：等待与发布条件 |
| [nccl/src/transport/net.cc:1324](../nccl/src/transport/net.cc#L1324) | `sendProxyProgress`；1493 行 `recvProxyProgress`：请求提交与完成 |
| [nccl-tests/src/common.cu:496](../nccl-tests/src/common.cu#L496) | `testStreamSynchronize`：CUDA query、异步检查、测试超时 |

至此，01 软件栈到 12 验证调试形成闭环：理解职责，找到对象，修改交接点，再用分层证据验收。
[00 环境](00-environment.md)与[examples](examples/README.md)并行使用；[操作](appendix-collectives.md)、[现代特性](appendix-modern-features.md)、[tests](appendix-nccl-tests.md)、[性能](appendix-performance.md)四个附录按需查阅。
