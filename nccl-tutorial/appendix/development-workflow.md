# 10 开发工作流：从一个需求走到可验证的改动

前九章已经把 API、连接、提交、GPU 执行和回收串起来。
现在把这条链反过来用：先写出要改变的行为，再找到负责它的对象和交接点。
本章做一个小改动：记录普通 host AllReduce 的 task 选择结果；[第 11 章](../11-testing-and-debugging.md)负责验证它。

源码基线为 NCCL `12df1a11`（2.32.3），测试为 nccl-tests `b4d5bee`（2.20.0）。
贯穿场景是两机各四 GPU、八 rank，每 rank 64 MiB FP32 Sum AllReduce。
rank 是 communicator 内的成员编号；机器布局和 rank 顺序是教学假设，实际连接由拓扑决定。
具体追踪固定 Ring/Simple：Ring 决定块怎样绕环，Simple 决定数据怎样发布和推进。
本机为无 CUDA 的 macOS；本章只提供修改指引，构建和 GPU 命令均为待读者实验。

## 1. 先把需求放进正确的层

修改层次由“谁决定这个行为”来选，而不是由报错出现在哪个文件来选。
例如，想减少小消息的启动成本，应先看任务组织与选择；只改 Ring 循环可能根本碰不到瓶颈。

| 需求 | 改哪层 | 关键对象 | 最小回归 |
| --- | --- | --- | --- |
| 扩展参数或调整 API 检查 | `nccl.h.in`、`collectives.cc`、参数检查 | `ncclInfo`、每次调用的 config | 合法边界输入、返回值、原有调用兼容性 |
| 改算法选择或 task 合并策略 | `enqueue/`、`tuning/` | `ncclTaskColl`、tuning 输入/结果、`ncclKernelPlan` | 选择阈值两侧、相同 group 的单/多 task、结果校验 |
| 改 GPU/NIC 路径或 ring 构造 | `graph/` | topology 节点/路径、`ncclTopoGraph`、channel | 同机与跨机、不同 GPU 对、连接图和正确性 |
| 改 device 算法或 protocol | `device/`、生成器、host dispatch | `ncclDevWorkColl`、`RunWorkColl`、primitive | 分块边界、尾块、datatype/op、oop/ip |
| 改 transport 或 net plugin | `transport/`、`plugin/`、插件接口 | connector、注册句柄、proxy 请求 | 建连、发送/接收、请求完成、释放及跨进程 |

planner 是把 task 编排成可启动计划的主机调度器。
dispatch 是依据运行时编号找到具体实现的分派过程。
transport 是一对端点之间的连接和传输实现；net plugin 则提供网络后端接口。
表中的最小回归用于快速发现改错层或破坏相邻职责，之后再按改动扩大覆盖。

## 2. 固定源码、构建和运行库三种身份

先记录源码提交、分支名、未提交改动以及编译参数。
分支名说明工作位置，提交说明基点；本地补丁也是实验输入的一部分。
版本号 2.32.3 只说明接口版本，同版本的两个 `.so` 可以包含不同代码。

修改源码前，先按[测试附录](nccl-tests.md)构建未修改库与 tests，将基线保留在 `nccl/build`。
随后在自己的开发副本中添加下文插桩，使用 `nccl/build-observe` 存放观察版本的产物。
插桩就是在程序某个状态点增加观测代码，本例只增加一条 host 日志。

本版 `enqueue/enqueue.cc:33` 的 `EnqueueRearchEnable` 默认值为 0。
因此本章追 `ncclPrepareTasks` 所在的默认 enqueue 路径，并在实验中显式固定这个开关。
源码分支名和运行时分支是两件事；同一构建仍可因环境、插件或调用配置进入不同实现。

下面及后续命令在 Linux GPU 节点的 Bash 中，从读者自己的 `nccl_learn` 根目录执行。
`CUDA_LIB` 指向实际 CUDA 动态库目录，默认采用 toolkit 的 `lib64`。

```bash
ROOT="${ROOT:-$PWD}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB="${CUDA_LIB:-$CUDA_HOME/lib64}"
env LD_LIBRARY_PATH="$ROOT/nccl/build/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$ROOT/nccl-tests/build/all_reduce_perf"
```

核对 `libnccl.so.2` 的解析路径，并读取 tests 开头的 `nccl-headers` 与 `nccl-library`。
本版两者应为 23203；前者来自编译头文件，后者来自 `ncclGetVersion`。
对运行中的进程，还可从 `/proc/<pid>/maps` 核对实际映射的库路径。
记录该库的 `sha256sum`，把“读的源码—构建产物—加载的库”关联起来；多机逐节点核对。

## 3. 读数据结构时，跟着生产者和消费者走

先看结构定义，再找第一次赋值、入队、出队和释放。
看到一个指针时，问它是拥有对象，还是只借用另一层的存储。
拥有意味着负责回收；借用意味着使用期限必须落在拥有者的生命周期内。

`ncclInfo` 是 API 意图的临时描述；`ncclTaskColl` 是待调度 collective 的主机记录。
task 保存 count、datatype、输入输出指针以及后续补齐的算法信息。
用户 buffer 由应用拥有，task 保存地址不会复制整份 64 MiB 数据。
应用按 stream/event 依赖保证输入输出可用，直到相关 GPU 工作完成才复用存储。

默认路径把 task 放进 `comm->planner`，按函数、规约操作和类型组织选择。
`ncclPrepareTasks` 用栈上的 `agg` 临时汇总相近任务，再把选择结果写回原 task。
`ncclGetAlgoInfo` 的参数名虽然叫 `info`，类型实际是 `ncclTaskColl*`。
这里没有可读取的 `ncclInfo::stream`；也不应把 `&agg` 留给后台线程稍后使用。

随后 `scheduleCollTasksToPlan` 把任务划到 channel，并生成 device work 和必要的 proxy 工作。
channel 是逻辑并行通信通道；它与物理网线、NIC 或 SM 都不是一一对应关系。
`ncclKernelPlan` 保存一次启动需要的主机元数据，GPU 消费的是提交后的工作描述。
`reclaimPlan` 回收 plan 持有的 task；这与 GPU work FIFO 槽位何时可复用是不同条件。

再问“这段函数由哪个线程执行”。group 的准备工作可能由异步 job 承担。
NET proxy 是推进网络请求的主机线程，CUDA kernel 又是独立执行者。
跨这些边界传递状态，要找队列、回调、计数或内存可见性规则，而不是假设函数返回代表全部完成。
对象寿命的具体追踪方法见[第 09 章](../09-memory-and-lifetime.md)。

## 4. 本章任务：给选择完成的 task 加一条日志

任务范围是“观察选择结果”，不改变选择、通道分配或同步行为。
先读 `ncclGetAlgoInfo`：它在 2218–2224 行把 tuning 结果写到传入对象。
再回到 `ncclPrepareTasks`：514 行完成聚合选择，536–543 行将结果写回每个原 task。
把日志放在这次回填之后、544 行入队之前，便能记录原始 task 的 count。
若直接在 `ncclGetAlgoInfo` 末尾打印，其 count 可能是多个 task 的聚合值。

在读者开发副本的 `nccl/src/enqueue/enqueue.cc` 中，把该位置扩展为：

```cpp
        aggBeg->isNvls = isNvls;
        if (aggBeg->func == ncclFuncAllReduce) {
          INFO(NCCL_TUNING,
               "[tutorial-task] rank=%d count=%zu algo=%s proto=%s maxChannels=%d warps=%d",
               comm->rank, aggBeg->count,
               ncclAlgoToString(aggBeg->algorithm), ncclProtoToString(aggBeg->protocol),
               int(aggBeg->nMaxChannels), int(aggBeg->nWarps));
        }
        ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
```

六个字段各有明确来源：rank 属于 `ncclComm`，其余五个属于 `ncclTaskColl`。
`count` 的类型是 `size_t`，所以用 `%zu`；算法和协议通过已有函数转成名字。
`nMaxChannels` 是选择阶段的通道预算，后面的资源上限和任务划分还会调整它。
`nWarps` 是选出的线程规模参数；这条日志没有声称 kernel 已经启动。
task 的 `nChannels`、`channelLo/Hi` 在后续调度才填入，当前观察点读取它们没有意义。

### 构建观察版本，再确认加载位置

复用修改前保存的基线库与 tests，只将当前源码构建到观察目录。
以下在同一 Bash 中执行；make 会生成观察版本的独立产物目录。

```bash
make -C "$ROOT/nccl" -j 8 src.build \
  CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_LIB" \
  BUILDDIR="$ROOT/nccl/build-observe"
env LD_LIBRARY_PATH="$ROOT/nccl/build-observe/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$ROOT/nccl-tests/build/all_reduce_perf"
```

这次只增加 host 日志且不改 ABI，可复用同一个 tests 可执行文件。
若修改公开头文件或 ABI，则重新构建调用方，让编译期与运行期契约一起更新。

### 用固定合法用例确认观察点

先在分配到的单机两卡上缩小问题，输入固定为每 rank 64 KiB FP32 Sum。
这里 `-t 1 -g 2` 表示一个 CPU 工作线程管理两个 GPU，共两个 rank。

```bash
env LD_LIBRARY_PATH="$ROOT/nccl/build-observe/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  NCCL_ENQUEUE_REARCH_ENABLE=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 64K -e 64K -t 1 -g 2 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

观察两个 rank 的 `[tutorial-task]`：count 应为 16384，选择应为 Ring/Simple。
预热、计时和校验都会调用通信，所以同一字段组合会重复出现。
检查 oop/ip 两组 `#wrong=0`、错误摘要和正常退出，再由第 11 章比较基线与观察版本。
本章交付是一条日志、字段归属解释和一次合法用例记录；macOS 上可完成源码核对部分。
两机八 rank 的 64 MiB 主场景留给回归，届时原始 task 的 count 应为 16777216。

## 5. 改 device 代码时，构建链也属于开发地图

设备模板只有被实例化，才会成为可调用的机器代码。
例如 `RunWorkColl` 的 AllReduce/float/Sum/Ring/Simple 特化表示一组确定的操作组合。
模板实现位于 `device/all_reduce.h`，Simple 收发原语位于 `device/prims_simple.h`。

Make 构建从仓库顶层的 `src.build` 进入 `src/Makefile`，再调用 device 子构建。
`device/Makefile` 运行 Python 生成器，在 `BUILDDIR/obj/device/gensrc` 生成源码。
其中包括按操作/类型组织的 `.cu`、`host_table.cc`、`device_table.cu` 和 `rules.mk`。
生成的 `DEFINE_ncclDevFunc` 把组合落实为实例；host/device 表把函数编号连接到入口。
nvcc 编译这些对象并执行 device link，manifest 再把它们交给主库链接。

因此改模板通常需要重编依赖它的多个实例；改组合列表还会改变生成表。
开发时可以用 `ONLY_FUNCS` 缩小生成集合，回归构建则应覆盖声明支持的全部组合。
改变生成选项时使用新的构建目录，使选项和生成产物保持一致。
时间线上的 kernel 名可能是代表入口，实际 work 仍按函数编号分派；名称不足以确认协议。

新增算法会同时触及四条链，适合先画地图，再拆成可审查的小改动：

1. **选择链：**定义可用条件、拓扑需求和成本模型，保证支持的调用能够选中它。
2. **分派链：**同步算法编号、`ncclDevFuncId` 行索引和生成器枚举顺序，保持两侧表一致。
3. **实现链：**连接准备、work 布局、chunk 划分、device 实现及必要的 proxy 推进共同满足协议。
4. **验证链：**覆盖合法 datatype/op、尾块、oop/ip、单/多进程与目标拓扑，并测自动选择回归。

这些是受影响面的地图，不是“添加一个枚举就完整支持新算法”的实现步骤。
本章的小日志只触及选择链，正适合先练熟构建、加载、观察和回归闭环。

## 6. 六个源码锚点

路径相对本教材目录；行号对应上述本地基线，改动后用符号重新定位。

| 路径:行 | 符号与用途 |
| --- | --- |
| [nccl/src/include/enqueue.h:69](../../nccl/src/include/enqueue.h#L69) | `ncclGetAlgoInfo`：核对参数实际类型 |
| [nccl/src/include/comm.h:212](../../nccl/src/include/comm.h#L212) | `ncclTaskColl`；357 行 `ncclKernelPlan`：字段与持有关系 |
| [nccl/src/enqueue/enqueue.cc:423](../../nccl/src/enqueue/enqueue.cc#L423) | `ncclPrepareTasks`；644 行调度、1589 行回收、2159 行选择 |
| [nccl/src/device/Makefile:80](../../nccl/src/device/Makefile#L80) | `$(OBJDIR)/gensrc`；133 行 device link：生成到链接 |
| [nccl/src/device/generate.py:186](../../nccl/src/device/generate.py#L186) | `enumerate_func_rows`；447 行实例生成：组合到代码 |
| [nccl/src/include/device.h:646](../../nccl/src/include/device.h#L646) | `ncclDevFuncId`：host 分派索引与生成表对应 |

继续到[11 验证与调试](../11-testing-and-debugging.md)，用证据区分“打印了选择”与“改动已通过回归”。
[环境准备](environment.md)和[示例说明](../examples/README.md)是并行参考；collective 语义查[操作附录](collectives.md)，新路径查[现代特性附录](modern-features.md)。
