# 示例：建立正确性基线，再跟踪软件栈

这里保留三个小程序。CPU 模拟用于看清算法；GPU 示例用于验证 API、参与者映射和完成顺序；性能测量使用 nccl-tests。

| 文件 | 用途 | 环境 |
|---|---|---|
| [ring_simulator.py](ring_simulator.py) | 逐轮追踪 Ring 的分块与贡献来源 | Python 3 |
| [single_process_allreduce.cu](single_process_allreduce.cu) | 一个 CPU 线程管理多张 GPU | Linux、CUDA、NCCL |
| [mpi_allreduce.cu](mpi_allreduce.cu) | 每个 MPI 进程管理一张 GPU | 上述环境加 MPI |
| [Makefile](Makefile) | 构建两个 GPU 示例 | 单进程用 nvcc，MPI 版用 mpicxx |

命令从项目根目录执行。GPU 实验使用获分配的设备；先完成 [00 环境准备](../00-environment.md)。本次 macOS 环境未编译或运行 GPU/MPI 示例。

## 1. CPU：追踪一个块的完整旅程

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

第一条命令中，每 rank 有四块、每块两个元素。输入与最终结果为：

```text
r0 输入 [ 0, 1, 2, 3, 4, 5, 6, 7]
r1 输入 [10,11,12,13,14,15,16,17]
r2 输入 [20,21,22,23,24,25,26,27]
r3 输入 [30,31,32,33,34,35,36,37]

每 rank 输出 [60,64,68,72,76,80,84,88]
PASS: 6 rounds; sent per rank = 12 elements
```

程序会在两者之间打印每轮 `r0->r1: c0` 这样的记录。`c0` 是数组的第 0 块，沿环传播时逐渐累加其他 rank 的贡献。

阅读 `ring_allreduce` 时看两个设计：每轮先快照所有发送，再统一接收；`sources` 集合记录已经加入的 rank。前者让一轮只前进一跳，后者验证每份贡献恰好加入一次。

本模拟的 ReduceScatter 阶段结束后，rank r 持有块 `(r+1)%P`。这是所选 Ring 调度的中间归属；公开 ReduceScatter API 的结果按 rank r 的第 r 块布局，见[接口附录](../appendix-collectives.md)。

`--self-test` 覆盖 85 个用例，包括不同 rank 数、块长、带符号整数和非法形状。它验证数学与调度，第 07 章的 GPU 协议和第 09 章的网络推进另有实现。

## 2. 单进程：把两个 rank 一起提交

```bash
export ROOT="$PWD"
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME="$ROOT/nccl/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -C nccl-tutorial/examples CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
./nccl-tutorial/examples/build/single_process_allreduce 2
```

参数是 GPU 数。程序固定每 rank 1024 个 FP32 元素，rank r 的输入全部为 `r+1`，所以每个输出元素应为 `P(P+1)/2`。

基线版本下的两卡预期输出如下，属于源码推导：

```text
NCCL header=23203 runtime=23203 ranks=2 count=1024
rank=0 device=0 first=3 expected=3 wrong=0
rank=1 device=1 first=3 expected=3 wrong=0
PASS
```

按下面五处读源码即可抓住执行主线：

| 位置 | 看什么 |
|---|---|
| [single_process_allreduce.cu:63](single_process_allreduce.cu#L63) | 逐卡选设备，准备缓冲和同 stream 的 H2D |
| [single_process_allreduce.cu:73](single_process_allreduce.cu#L73) | InitAll 建立通信组，group 收集各 rank 的 AllReduce |
| [single_process_allreduce.cu:83](single_process_allreduce.cu#L83) | NCCL 异步状态与 CUDA stream 完成查询 |
| [single_process_allreduce.cu:107](single_process_allreduce.cu#L107) | 读回并检查全部元素 |
| [single_process_allreduce.cu:119](single_process_allreduce.cu#L119) | 通信对象、用户缓冲和 stream 分别释放 |

等待循环在 GroupEnd 返回后设置 60 秒期限，覆盖已提交 GPU 工作的等待。完整作业的运行时限交给启动器或调度器。示例采用出错退出，重点是小规模正常路径的正确性。

## 3. 两个合法修改练习

每项都从原始单进程示例开始，保留 group、完成等待和全元素校验。

### 练习 A：改为原地 AllReduce

1. 只分配 `send[r]`，去掉独立 recv 分配。
2. AllReduce 的输入输出都传 `send[r]`。
3. D2H 从 `send[r]` 读回，每张卡只释放这一块缓冲一次。

两卡结果仍应全部为 3。这个实验验证 API 的原地关系和应用内存管理。

### 练习 B：改为 AllGather

1. 保留发送长度 `count=1024`，新增 `recv_count=count*ndev`。
2. GPU recv 和 CPU output 都扩大到 `recv_count`，D2H 长度同步扩大。
3. 调用改为 `ncclAllGather(send[r], recv[r], count, ncclFloat, comms[r], streams[r])`。
4. 校验每个 `output[source_rank*count+i]` 是否等于 `source_rank+1`；替换原先求和用的 expected 和打印格式。

两卡每个输出应是“1024 个 1，接着 1024 个 2”，全部元素通过后才算完成。

想研究内部调度，用[第 11 章](../11-development-workflow.md)的插桩任务；本示例的 PASS 验证输出，选路信息另从对应观测点取得。

## 4. MPI：把引导和数据通信分开看

```bash
make -C nccl-tutorial/examples mpi \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" MPICXX=mpicxx
mpirun -np 2 ./nccl-tutorial/examples/build/mpi_allreduce
```

程序路径是：

```text
MPI 创建进程并取得 global/local rank
   → 选 GPU、交换 PCI bus ID 检查映射
   → rank 0 生成 NCCL ID，MPI_Bcast 分发
   → ncclCommInitRank 建立通信关系
   → NCCL AllReduce 处理 GPU 数组
   → MPI_Allreduce 汇总 CPU 校验错误数
```

源入口为 [mpi_allreduce.cu:48](mpi_allreduce.cu#L48) 的本地分组、[第 82 行](mpi_allreduce.cu#L82) 的 ID 分发、[第 97 行](mpi_allreduce.cu#L97) 的 NCCL 调用。

设备映射面向独占整卡：每进程只见一张卡时选 device 0；各进程看到同一完整列表时按 local rank 选卡。多机使用现有作业启动配置，将相同二进制、库路径和环境传到各节点。四个 rank 时，每个元素预期为 10。

MPI 版没有自定义 device kernel，因此 Makefile 用 `mpicxx -x c++` 编译 host 调用，并链接 CUDA runtime 和 NCCL。

## 5. 验证分工

- CPU 模拟与自测：验证块流转和求和结果。
- GPU 单进程示例：验证本地多设备参与、提交、完成和结果。
- MPI 示例：增加跨进程引导与设备映射。
- [nccl-tests](../appendix-nccl-tests.md)：扩展数据规模、操作、模式并测量性能。

回到[教程首页](../README.md)。
