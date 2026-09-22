# 动手示例：从 CPU 模拟到多机 GPU

所有命令从同时含 `nccl/`、`nccl-tests/`、`nccl-tutorial/` 的项目根目录执行。

| 文件 | 用途 | 硬件要求 |
|---|---|---|
| [ring_simulator.py](ring_simulator.py) | 逐轮展示并校验 Ring 两阶段 | Python 3，无 GPU 依赖 |
| [single_process_allreduce.cu](single_process_allreduce.cu) | 一个 CPU 线程管理多 GPU | Linux、CUDA、NCCL；默认两卡 |
| [mpi_allreduce.cu](mpi_allreduce.cu) | 每 MPI 进程一张 GPU | 上述环境 + MPI；可单机/多机 |
| [Makefile](Makefile) | 构建两个 GPU 示例 | `nvcc`；MPI 示例另需 MPI C++ wrapper |

GPU 示例固定每 rank 1024 个 FP32 元素，输入全为 `rank+1`，规约为 Sum；输出每元素应为 `P(P+1)/2`。它们是小规模正确性示例，**不是 benchmark，也不是完整生产级容错实现**。

## 1. 无 GPU：看清每轮发送什么

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --ranks 3 --chunk-size 1
python3 nccl-tutorial/examples/ring_simulator.py --ranks 1 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

默认四 rank 的输入与第 02 章 ReduceScatter 例子一致：rank r 的数组为 `[10r,10r+1,...,10r+7]`，最终每 rank 得到 `[60,64,68,72,76,80,84,88]`。

模拟器明确选择以下内部块编号约定：

- rank r 发给 `(r+1) mod P`，从 `(r-1) mod P` 接收。
- ReduceScatter 第 s 轮发送块 `(r-s) mod P`，结束后 rank r 持有全规约块 `(r+1) mod P`。
- AllGather 第 s 轮发送块 `(r+1-s) mod P`，将这些完整块传遍所有 rank。

这是**AllReduce 内部阶段**的偏移归属，不是公共 `ncclReduceScatter` API 的输出布局。公共 API 必须使 rank r 得到第 r 块；实现可通过块编号/顺序映射满足它。

第 08 章也可采用另一套等价的起始块编号来说明；比较时先对齐“第一轮发哪块”，不要只比较表里 c0/c1 的名字。

程序先收集本轮所有发送快照，再统一接收，避免 Python 顺序执行不小心让一个块在“一轮”里前进多跳。它还记录规约贡献来源，断言没有重复加入同一 rank 的数据。

`--self-test` 覆盖 P=1..9、多个块长、含负数的随机整数、空输入及非法形状。固定随机种子，结果可重复。不支持非等长或不能等分的输入，这是教学模型限制，不是 NCCL AllReduce 的限制。

## 2. Linux：编译单进程示例

先完成[环境章](../00-environment.md)中的 NCCL 构建：

```bash
export ROOT="$PWD"
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME="$ROOT/nccl/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

make -C nccl-tutorial/examples CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
ldd nccl-tutorial/examples/build/single_process_allreduce
./nccl-tutorial/examples/build/single_process_allreduce 2
```

预期两卡都输出 `first=3 expected=3 wrong=0`，最后 `PASS`。这是预期结果说明，不是本机 GPU 实测输出。

也可用 1 卡验证退化路径、4 卡验证结果为 10：

```bash
./nccl-tutorial/examples/build/single_process_allreduce 1
./nccl-tutorial/examples/build/single_process_allreduce 4
```

参数必须为正整数且不超过可见 GPU 数。程序使用可见列表的前 N 张卡，尊重 `CUDA_VISIBLE_DEVICES`；运行前确认不会占用其他人的任务资源。

### 按数据生命周期读代码

1. `cudaSetDevice`、各设备分配 send/recv 和 nonblocking stream。
2. CPU 输入保留到通信完成，H2D copy 与 collective 在同一 stream 上有序。
3. `ncclCommInitAll` 一次建立单进程 clique，devlist 顺序决定 rank。
4. 一对 GroupStart/End 包住所有 GPU 的 AllReduce，保证共同提交。
5. 轮询所有 communicator 异步状态和所有 stream，GPU 完成后读回并检查每个元素。
6. 正常销毁 communicator，再释放用户内存与 stream。

这里的 host vector 不是 pinned memory，所以不要用该 copy 片段推导“必然实现 H2D/compute overlap”。本示例关注正确性。

## 3. MPI：构建和单机验证

MPI 示例不含自定义 device kernel，全部是 host API 调用，因此 Makefile 用 `mpicxx -x c++` 编译并链接 CUDA runtime/NCCL。这样不需要猜 MPI 库的依赖列表。

```bash
make -C nccl-tutorial/examples mpi \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" MPICXX=mpicxx

mpirun -np 2 ./nccl-tutorial/examples/build/mpi_allreduce
```

`mpirun` 的选项会随 MPI 实现变化。下面多机命令使用 Open MPI 风格；其他 MPI/调度器请转换为等价配置，不要混用不同 MPI 实现的编译器与 launcher。

程序的引导链路：

```text
MPI 启动进程 -> global rank / shared-memory local rank
 -> 选择本地 GPU -> 检查同节点 PCI device 不重复
 -> rank0 ncclGetUniqueId -> MPI_Bcast(id)
 -> 各 rank ncclCommInitRank
 -> NCCL AllReduce GPU payload
 -> MPI_Allreduce 汇总少量CPU校验计数
```

两处 MPI 通信分别传引导 ID 和校验计数，**没有用 MPI 对 GPU 测试数组执行 AllReduce**。

### GPU 可见性规则

- 所有本地进程看到相同的完整 GPU 列表：`device=local_rank`。
- 调度器已为每个进程分配独占单卡、每个进程只见一张：`device=0`。
- 程序交换选中的 PCI bus ID，拒绝同节点多个 rank 选中同一个物理设备。

本示例面向独占整卡，不覆盖 MIG、多 rank 同卡等特殊布局。不同进程看到任意重排的多卡列表也不在简化映射假设内；若被检测为冲突，应修正任务分配，不要删除检查强行运行。

## 4. 两机各两卡

先在两台节点上确认：相同源码与库、相同绝对目录、相容的 CUDA/驱动、MPI 可正常启动进程、网络互通。`node-a/node-b` 是待替换的主机名。

```bash
export NCCL_DEBUG=INFO
mpirun --host node-a:2,node-b:2 -np 4 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x NCCL_DEBUG \
  "$ROOT/nccl-tutorial/examples/build/mpi_allreduce"
```

预期日志显示每台机器两个 local rank，global rank 共 0..3；结果为 10，`total_wrong=0 PASS`。不要先假定网络用了 IB，按第 07/12 章检查实际 transport 日志。

若节点目录不同，应部署到约定位置或按集群机制指定工作目录和环境。示例不会替你分配 GPU、配置 SSH、修改防火墙或修复 RDMA 权限。

## 5. 等待与错误处理的边界

- 这两个示例使用默认 blocking communicator，不演示完整非阻塞 init 状态机。
- 已成功提交后的轮询带 60 秒 deadline，同时查询 NCCL 异步错误和 CUDA stream 状态。
- deadline **不覆盖**可能阻塞的初始化、GroupEnd、MPI 调用或异常清理本身。需要全作业超时时，使用调度器 walltime/管理员允许的作业控制方式。
- 普通错误采用 fail-fast；MPI 示例调用 `MPI_Abort` 终止自身作业，不做训练状态恢复。只在自己获准使用的实验作业里运行。
- 单进程示例在等待超时时尝试 abort communicator；这不是对任意故障都保证有界返回的容错框架。
- 小规模整数值在 FP32 下可精确表示，因此这里使用精确比较；不要据此让任意浮点规约都要求逐位一致。

## 6. 推荐修改练习

先复制到你自己的实验文件再改，保留基线用于对照：

1. 两卡 AllReduce 改成合法 in-place，仍校验所有元素。
2. 改为 AllGather，并按 rank 验证不同块，注意接收容量扩大 P 倍。
3. 改为 ReduceScatter，扩大发送容量，并根据 rank 计算期望值。
4. 添加正确的 ready/done CUDA events，接入生产者与消费者 kernel。
5. 对比自己逐元素校验与只检查第一个元素：构造中间一个元素错误，验证前者能发现。

性能实验请使用 nccl-tests 并遵循[测量章节](../10-nccl-tests.md)，不要把本程序的 1 ms 轮询 sleep、初始化和 PCI 映射检查计入通信吞吐。

## 7. 本次交付验证范围

CPU 模拟器已在本机通过 85 项内置测试、21 项额外规模/数据测试和 3 项非法 CLI 参数检查；Makefile 的普通/MPI 两个目标均完成 dry-run。CUDA 示例已对照公共头文件和生命周期规则编写，但**未在本次 macOS 环境中编译或运行**。Makefile dry-run 只验证构建命令展开，不等于 GPU 编译通过。
