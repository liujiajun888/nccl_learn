# 第 12 章：把报错、超时与挂起拆成可验证的问题

> 基线：NCCL `12df1a11`（2.32.3），nccl-tests `b4d5bee`（2.20.0）。
> 所有 shell 示例从仓库根目录 `nccl_learn`、在 Linux NVIDIA GPU 节点执行。
> `ROOT` 可由用户设置，未设置时取 `$PWD`；所有路径都属于读者的运行节点。
> 当前编写环境无可用 NVIDIA GPU；仅核对源码，未运行故障注入、GPU 或集群任务。

## 学习目标

- 区分应用契约错误、CUDA 错误、初始化失败、数据通路错误和进度停滞。
- 理解 NCCL API 返回、异步状态轮询、CUDA stream 完成三个不同层次。
- 安全收集唯一日志，核查本版 RAS 和 DEBUG_GLOBAL 真正能做什么。
- 形成“停止提交—协调 abort—应用恢复”的流程，不误以为 NCCL 会重启 rank。

可独立按本章的清单排查；测试程序构建见 [第 10 章](10-nccl-tests.md)。
通信路径背景见 [第 7 章](07-transports.md)，性能而非正确性问题见 [第 11 章](11-performance.md)。

## 1. 先分层，不要看到 NCCL 就先改网卡

| 层次 | 常见表现 | 第一份证据 |
| --- | --- | --- |
| 应用契约 | count/root 错、collective 顺序不同、设备重复 | 所有 rank 的调用序列与映射 |
| CUDA／内存 | illegal address、invalid device、OOM | 最早的 CUDA 错误及 buffer 生命周期 |
| 启动／运行库 | 找不到 libnccl/libmpi、版本不一致 | 每节点构建版本、动态库解析、启动日志 |
| NCCL 初始化 | bootstrap 卡住、无法分配 SHM、连接失败 | INIT/BOOTSTRAP/NET/SHM 日志 |
| 数据通路 | 首次通信卡住、IB completion 错误 | NET/PROXY/REG 日志与端口计数器 |
| 应用进度 | 某 rank 未进入下一次 collective | 框架异常、数据加载、CPU/GPU 时间线 |
| 收尾／恢复 | 其他 rank 仍等待、销毁迟迟不返 | 故障传播、abort 调用、进程退出状态 |

`ncclInvalidArgument` / `ncclInvalidUsage` 通常指向参数或契约，也可能是强制配置无可用实现。
`ncclUnhandledCudaError` / `ncclSystemError` 表示底层调用失败；`ncclRemoteError` 提示远端／网络相关失败，都要看具体日志。
`ncclInternalError` 需保留版本和复现，不能直接归咎硬件；非阻塞模式的 `ncclInProgress` 是状态而非普通故障码。
最晚报错的 rank 可能只是受害者；先找最早退出或首次序列分歧，数据加载异常也可能让其他 rank 像网络挂起一样等待。
CUDA 异步错误可能到后续 NCCL 或同步调用才显现，报错位置不总是致错位置。

## 2. 先保存一次能对齐的最小复现

记录提交、CUDA／驱动、插件、MPI、容器镜像、主机、GPU bus ID、rank 数和完整启动参数。
保存实际环境和调度器资源分配；分享日志前脱敏主机名、地址和私有路径。
不要先删库、清理共享内存或修改共享系统：这会破坏现场并影响其他作业。
如果需要重跑，只在已获授权的资源内，从单机两卡、小规模、校验开启开始。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；仅在读者已分配的两张卡上复现。
ROOT="${ROOT:-$PWD}"
LOG_DIR="$(mktemp -d "$ROOT/nccl-debug.XXXXXX")"
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,GRAPH,ENV,SHM,REG,PROXY \
  NCCL_DEBUG_FILE="$LOG_DIR/nccl.%h.%p.log" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 1K -e 1M -f 2 -t 1 -g 2 -w 2 -n 10 -c 1 -a 3 -T 30 \
  > "$LOG_DIR/tests.stdout" 2> "$LOG_DIR/tests.stderr"
```

NCCL（不是 shell）展开 `%h/%p` 为主机名／PID；本次独立目录防止跨次覆盖，文件名区分本次进程。
`NCCL_DEBUG_FILE` 以写模式打开文件，同名可能丢日志；多机目录须各节点可写，非共享存储须分别收集。
容器 hostname/PID 若不唯一，再加作业／容器 ID 或 launcher rank 前缀；不要只取启动节点日志。
默认 INFO 未必含 NET/GRAPH；需要调用序列时，短复现再追加 `COLL,CALL,P2P`，必要时短时 TRACE。
TRACE 会扰动时序，不能混入性能基线；stdout/stderr 也须保留，CUDA、MPI、框架错误可能只在其中。
`-T 30` 不是整个进程的万能 30 秒看门狗，下节解释其边界。

## 3. 超时、挂起和慢：三个不同判断

- **慢：**进度持续前进，只是低于目标，需要时间线和重复样本判断。
- **超时：**某一层预设期限到了，说明没按期完成，不说明根因必然是网络。
- **挂起：**观察窗口里没有进展，可能是等待缺席 peer、应用死锁或底层故障。

先问“哪个时钟触发了超时”：

| 超时层 | 覆盖什么 | 不覆盖什么 |
| --- | --- | --- |
| tests `-T` | `testStreamSynchronize` 中的 stream 轮询等待 | MPI 启动、所有初始化调用、所有 MPI barrier |
| `NCCL_IB_TIMEOUT/RETRY_CNT` | IB QP 传输应答／重试机制 | 训练 step 总耗时、应用调用匹配 |
| `NCCL_SOCKET_RETRY_*` | 特定 socket 建连错误后的重试 | 所有 collective 的完成期限 |
| RAS 超时 | 健康探测、状态收集响应 | CUDA kernel 的完成期限 |
| 框架／调度器 watchdog | 应用或作业管理定义的范围 | 不自动定位底层根因 |

本版 `NCCL_IB_TIMEOUT` 默认 20、`NCCL_IB_RETRY_CNT` 默认 7；前者使用指数时间编码，20 不是“20 秒”。
反复加大 timeout 可能只延后发现缺席 rank；先查端口、拥塞、进度与契约，作业截止时间则用站点批准的 launcher／调度器策略。
确认策略能处理全部 rank；只终止启动端不保证远端清理，本章不提供绕过站点的强杀或系统改动。

## 4. 三个“完成”：API、异步状态、CUDA completion

普通 collective 往 stream 提交工作，API 成功不等于 GPU 输出可用；非阻塞 communicator 还可能返回 `ncclInProgress`。
这表示主机侧尚未完成提交／推进，须先 poll NCCL 状态再进行依赖它的 CUDA 操作；group 则检查 group 结束与所有相关 comm。
`ncclCommGetAsyncError(comm, &state)` 有两个结果：
1. 函数返回值：这次查询 API 自身是否成功。
2. `state`：communicator 的异步状态，包括 success、in-progress 或错误。

`state == ncclSuccess` 不等于 stream 完成；安全消费 buffer 还需正确的 CUDA stream/event 依赖或完成查询。
只做可能永久等待的 `cudaStreamSynchronize` 而没有独立错误监控，可能失去及时 abort 的机会。
下面是**单 communicator 控制流示意，不可独立编译**；辅助函数由应用实现，`fail` 必须终止正常路径并进入协调恢复。

```cpp
// 前提：当前 CUDA device 正确；禁止其他线程同时销毁 comm。
// deadline 是应用的单调时钟期限，yield_cpu 不执行 CUDA 工作。
state = submit_or_group_end();
while (state == ncclInProgress) {
  rc = ncclCommGetAsyncError(comm, &state);
  if (rc != ncclSuccess) fail("query API", rc);
  if (deadline_expired()) fail("host progress timeout");
  yield_cpu();
}
if (state != ncclSuccess) fail("NCCL submission", state);
// 至此才能依赖已完成的提交；CUDA event 也应在正确顺序位置记录。
for (;;) {
  rc = ncclCommGetAsyncError(comm, &state);
  if (rc != ncclSuccess) fail("query API", rc);
  if (state != ncclSuccess) fail("NCCL asynchronous state", state);
  cu = cudaStreamQuery(stream);
  if (cu == cudaSuccess) break;
  if (cu != cudaErrorNotReady) fail("CUDA execution", cu);
  if (deadline_expired()) fail("device completion timeout");
  yield_cpu();
}
```

多 GPU 应轮询全部相关 communicator 和 stream，不能第一张卡完成就释放所有 buffer。
失败期间的输出不能当作有效梯度；也不能在设备仍可能访问时提前释放或复用内存。
nccl-tests 的等待函数展示了 stream query、异步错误查询与 abort，但它不是完整容错训练框架。

## 5. Collective 不匹配与 P2P 死锁

同一 communicator 的 rank 必须按匹配的逻辑顺序调用 collective。
操作类型、count、datatype、归约方式、root，以及该操作要求的分片语义必须一致。
不能为了某 rank 没有数据就直接跳过一轮；应采用应用层一致的协议。
“每个 rank 最终都调用了一次 AllReduce”不够，它们必须匹配到同一轮。

在应用边界记录：逻辑 communicator ID、step、sequence、rank、op、count、type、redop、root。
再记录 buffer 生命周期、CUDA device 与 stream 依赖，按 sequence 找第一个分歧。
跨进程的 comm 指针数值本来就不同，不要用指针地址相等作为匹配条件。
先比较逻辑序列，再解释后面的网络等待；不要在已经坏掉的 communicator 上补一个 barrier 验证。

P2P 的 send/recv 需要匹配 peer、元素数量、类型与顺序，且使用符合语义的独立缓冲区。
典型死锁：双方先 send，然后等待 send 的 stream 完成，完成后才提交对应 recv。
若发送推进依赖接收先被发布，两边都不会走到 recv。
应将需要并发推进的 send 与 recv 放入同一个 `ncclGroupStart/End`，再等待 group 提交和设备完成。
只在每个 send 外面各包一个 group，不能解决整体依赖环。
`sendrecv_perf` 的邻居环在一个 group 中提交 send/recv，可用于阅读正确的基本组织方式。
不要在共享集群故意制造缺失 rank 或死锁；先纸面画依赖图，在隔离且获准环境再验证。

## 6. 初始化成功以后，数据路径仍可能失败

按层缩小范围，预期结果只用于判断下一步方向，不是性能保证。

1. **启动层：**所有进程是否真的启动？各节点库是否一致？MPI 初始化／小归约是否成功？
2. **设备层：**每 rank 是否选择独占的正确 GPU？单 GPU CUDA 分配和执行是否正常？
3. **bootstrap：**unique ID 是否一致分发？IP 接口可达吗？有没有缺席 rank？
4. **本机传输：**同机两卡能否完成并校验？跨进程时 SHM／IPC 是否可用？
5. **跨机传输：**再做两机小消息，检查 HCA、端口、GDR 注册和网络插件。
6. **规模层：**小规模正常后增大消息或 rank，观察连接数、内存、拥塞和资源限额。

初始化通过仅说明已走过的控制路径可用；部分连接会延迟到第一次通信或首次使用某算法建立。
因此“挂在第一次 AllReduce”既可能是应用不匹配，也可能是预连接／注册失败。
单机正常、跨机异常优先关注 NET，但若启动映射在跨机时改变，也必须重新核对 GPU 分配。
如果关闭某个传输后的单变量对照恢复，只能说明问题与该路径相关，不能直接断言硬件损坏。
相关隔离变量见第 11 章；每次只改一项，复现后撤销，不保留为全局“修复”。

## 7. 容器、IB、NUMA 与安全边界

以下命令只读检查；缺少工具或权限时保存错误，交给管理员补充，不自行提权或改权限。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；在发生问题的同一作业／容器中检查。
ROOT="${ROOT:-$PWD}"
nvidia-smi topo -m
ls -ld /dev/shm /sys/bus/pci/devices
ls -l /dev/infiniband
df -h /dev/shm
ulimit -l
ulimit -s
ulimit -n
ip -br addr
ibstat
rdma link show
lscpu --extended=CPU,NODE,SOCKET,CORE
lspci -tv
```

### 共享内存与锁页内存

`/dev/shm` 是共享内存空间，memlock 限额影响锁页／注册资源；容器能看见 GPU 不代表这两项或 RDMA／PCI 拓扑都正常。
对照分配错误、`df` 和进程限额，不删除 `/dev/shm/nccl-*`；容器 `--shm-size`、`--ulimit memlock=...` 应按需求获得批准。
本章不执行这些修改，固定容量或 unlimited 也非万能；不用 privileged、扩大设备权限或宿主机 IPC 共享来兜底。

本版满足条件时可能用 cuMem host allocation，依赖 NUMA、运行库和驱动，并有失败检测／回退机制。
所以 SHM 空间足够仍可能失败，没 SHM 文件也不代表没分配；规模扩大时还要读栈、FD、cgroup 限额，不改共享系统。

### RDMA 权限、端口与拓扑

- `/dev/infiniband` 存在不等于当前身份有权打开设备；核对 job/container 的设备授权。
- `ibstat` 中关注端口 Active、物理链路、link layer 与实际速率，而非只看 HCA 名称。
- IB 需要正常 fabric 管理；RoCE 还涉及 GID、地址和拥塞配置，由管理员核对。
- IP bootstrap 与 bulk RDMA 接口独立；能 ping 通不等于 RDMA／GDR 数据路径通过。
- TCP 端口、RDMA P_Key／GID、交换网络策略分别核查，不靠关闭防火墙验证。
- GDR 可通过兼容的 peer-memory 或 DMA-BUF 机制；不能看到模块缺失就自行加载驱动。
- GPU、NIC、CPU 与 host memory 跨 NUMA 会增加绕行；对照 launcher 绑定和 cpuset 限制。
- ACS 可能改变 PCIe P2P 路由，IOMMU 涉及 DMA 地址转换、隔离与虚拟化兼容性。
- 仅凭 P2P 查询为 OK 不能排除上述系统问题；整理 PCI 树和内核模式证据交管理员核验。
- **不建议关闭 ACS/IOMMU、改 BIOS、安全开关、设备权限或共享网络配置。**

如果管理员提供低层测试结果，区分 host-memory RDMA 和 GPU-memory RDMA；前者通过不保证后者。
对失败端口记录时间和计数器增量，避免把历史累计错误都算到本次实验头上。

## 8. 本版 RAS 能看什么，不能做什么

源码确认：`NCCL_RAS_ENABLE` 默认 1，RAS 在 NCCL 初始化过程中启动。
RAS 线程建立健康监测连接，客户端默认监听 `localhost:28028`，可查询进程与 communicator 状态。
它能提供缺失／无响应进程、异步错误和 collective 进度等线索，不是训练容错管理器。

```bash
# Linux NVIDIA GPU；从 nccl_learn 执行；在已有、获准查询的 NCCL 作业所在节点使用。
ROOT="${ROOT:-$PWD}"
"$ROOT/nccl/build/bin/ncclras" --help
"$ROOT/nccl/build/bin/ncclras" -h localhost -p 28028 -v -t 10
```

源码构建提供 ncclras，发行包路径可能不同；核对客户端版本，`-h` 是 host，帮助用 `--help`。
本版还有 JSON、monitor、诊断等能力，以帮助为准；这里仅查询，不执行控制命令或故障注入。
保持本地监听，不暴露到所有网卡；多作业节点须核对 communicator/job，地址隔离由站点管理。

短暂操作计数差异可能只是采样不同，持续不前进才值得追踪；查询无响应也可能是 RAS 路径或初始化问题，不独立证明 GPU 死锁。
`NCCL_RAS_TIMEOUT_FACTOR` 缩放 RAS 内部超时，不是 collective 总超时；RAS **不会自动重启进程、替换 rank 或恢复模型状态**。

## 9. DEBUG_GLOBAL：名字很大，承诺要按实现来

本版实际环境变量是 `NCCL_CHECK_MODE=DEBUG_GLOBAL`，不是 `NCCL_DEBUG=DEBUG_GLOBAL`。
`DEBUG_LOCAL` 增加本地 CUDA 指针等检查；GLOBAL 还把参数信息排入全局检查流程。
核对 `ncclArgsGlobalCheck`：当前额外工作主要是 `registrationCheck`，检查对称注册状态／窗口／偏移。
它排除 Send/Recv 和部分单边操作，不能被宣传成“检测所有 collective 次序、count、类型不匹配”。
该检查会做 bootstrap 数据交换，如果某个参与者根本没调用到这里，它本身也可能等待。
所以不应对所有挂起盲目打开 GLOBAL，更不能以“未报错”证明应用序列正确。
针对已怀疑的对称注册问题才短时启用；应用调用日志和 CUDA 完成检查仍不可省略。
注册、设备 API 与更高级检查场景见 [高级主题](13-advanced.md)。

## 10. 故障后：abort 是清理动作，恢复需要应用协议

1. 停止向受影响 communicator 提交新通信；保存第一个错误和相关序列。
2. 通过仍然可用的带外控制面／launcher 通知其他 rank，不能依赖已坏的 collective 做协调。
3. 管理每个本地 communicator 的唯一清理责任，避免 poll 线程与销毁线程同时访问已释放对象。
4. 对故障 communicator 调用 `ncclCommAbort`，终止可能仍在设备运行的通信并释放关联资源。
5. 不再使用已 abort 的 communicator；核实设备和进程状态，按框架策略退出或重新初始化。
6. 全体幸存／重启成员达成新的成员关系、rank 编号和状态恢复点，再创建新的 communicator。
7. 从一致 checkpoint 恢复模型、优化器与数据进度；重放不能把半完成的梯度当有效输入。

`ncclCommDestroy` 属于正常生命周期管理，不应用它替代所有故障情形下的 abort。
遇到严重 CUDA context 错误时，通常需要由应用管理器重启进程；不要承诺原进程能继续训练。
本版有 revoke/shrink/grow 等管理 API，但它们不是自动选主、自动重启或自动 checkpoint 恢复。
采用这些 API 仍要应用提供一致的成员变更协议和错误处理，详见第 13 章。
恢复后先运行最小正确性检查，再回到 [训练集成](16-training-integration.md) 验证完整 step。

## 11. 源码锚点与复核路径

路径相对仓库根目录；下列符号可用于重新定位，而不是只依赖会随版本漂移的行号。

| 路径与行号 | 符号／证据 |
| --- | --- |
| `nccl-tests/src/common.cu:496-548` | `testStreamSynchronize`：CUDA query、异步错误、tests 超时与 abort |
| `nccl/src/init.cc:3933-3966` | `ncclCommGetAsyncError`：comm／proxy／GIN 状态，不是 CUDA 完成 |
| `nccl/src/init.cc:3488-3530` | `ncclCommAbort`：设置 abort flags 并回收资源 |
| `nccl/src/nccl.h.in:304-345` | `ncclCommFinalize/Destroy/Abort/Revoke/Shrink` 生命周期声明 |
| `nccl/src/debug.cc:159,225-272` | `ncclDebugInit`：展开 `%h/%p`，以 `w` 打开日志 |
| `nccl/src/init.cc:2533-2547` | `envConfigOverride`：解析 `NCCL_CHECK_MODE` |
| `nccl/src/misc/argcheck.cc:47-108,190-198,227-250` | `registrationCheck`、`ncclArgsGlobalCheck`、`ArgsCheck` |
| `nccl/src/group.cc:844-869` | `groupLaunchLegacy` 内 debug check 调度 |
| `nccl/src/ras/ras.cc:89-101` | `RasEnable` 默认值、`ncclRasCommInit` |
| `nccl/src/ras/client_support.cc:166-190`；`nccl/src/ras/client.cc:50-78` | `rasClientInitSocket`、`printUsage` |
| `nccl/src/ras/ras_param.cc:26-61` | `rasLoadTimeoutFactor`、`rasTimeoutFactorSec` |
| `nccl/src/transport/net_ib/connect.cc:13-23` | `IbTimeout`、`IbRetryCnt` 等参数 |
| `nccl-tests/src/sendrecv.cu:94-108` | `SendRecvRunColl`：同 group 提交 send/recv |
| `nccl/docs/userguide/source/troubleshooting/runtime_and_mpi_issues.rst:26-96` | Shared memory、cuMem host allocation 条件 |

## 12. 自测题与答案

1. `ncclCommGetAsyncError` 返回 success，是否立即可以释放通信 buffer？
   **答：**不能；还要检查输出 state，并确认相关 CUDA 工作完成或正确建立依赖，失败路径也须防止仍在访问。
2. DEBUG_GLOBAL 没有报错，能否排除所有 rank 的 count／collective 顺序不一致？
   **答：**不能；本版全局附加检查重点是对称注册一致性，不是通用调用契约检测器。
3. RAS 发现 rank 无响应后，谁负责替换进程并恢复训练？
   **答：**应用／框架和作业管理器；NCCL 提供状态与通信管理能力，不自动重启 rank 或恢复 checkpoint。
