# 动手示例：从 CPU 模拟到多机 GPU

这份示例集回答三件事：每个程序验证什么、怎么跑、跑完看什么。所有命令从同时含 `nccl/`、`nccl-tests/`、`nccl-tutorial/` 的项目根目录执行。

**CUDA/UMD 读者使用方式：** 把这里当作最小正确性基线和机制实验入口，不必重读每个 C++/CUDA 基础调用。完成[环境核对](../00-environment.md)后，编译运行单进程例子，重点核对下面的五个 NCCL 交接点。普通分配、vector 与拷贝的八段带练留在折叠区，需要时再展开。没有 GPU 时，先跑第 1 节的模拟，结合第 04 章分析贡献来源。需要研究跨进程时，再用第 3～4 节的 MPI 示例。

| 追踪点 | 现有单进程代码位置 | 要核对的机制 |
|---|---|---|
| 建立参与关系 | `single_process_allreduce.cu:73`，`ncclCommInitAll` | 每 GPU 的本地句柄如何属于同一 clique；不是新建两个互不相关的上下文 |
| 多设备共同提交 | 同文件 `:75–81`，GroupStart/AllReduce/GroupEnd | 为什么一个 host 线程不能先等第一张卡再提交下一张；group 不是 GPU 完成通知 |
| 本地数据完成与错误 | 同文件 `:83–105`，异步查询及 stream query | 查询返回值、comm 状态、CUDA 完成分别说明什么 |
| 用户结果校验 | 同文件 `:107–118` | 完成后的全部元素是否满足 collective 契约，而非只看 first |
| 正常资源退出 | 同文件 `:119–125` | work 不再引用资源后，comm 与用户分配各自如何释放 |

代码见 [single_process_allreduce.cu](single_process_allreduce.cu)。这些示例不插桩 NCCL 内部。要证明 task/plan 或 FIFO 的状态变化，继续用[源码任务](../14-source-map.md)和[机制检查站](../15-exercises.md)；不要从示例 `PASS` 反推出特定算法或 transport 已被使用。

| 文件 | 用途 | 硬件要求 |
|---|---|---|
| [ring_simulator.py](ring_simulator.py) | 逐轮展示并校验 Ring 两阶段 | Python 3，无 GPU 依赖 |
| [single_process_allreduce.cu](single_process_allreduce.cu) | 一个 CPU 线程管理多 GPU | Linux、CUDA、NCCL；默认两卡 |
| [mpi_allreduce.cu](mpi_allreduce.cu) | 每 MPI 进程一张 GPU | 上述环境 + MPI；可单机/多机 |
| [Makefile](Makefile) | 构建两个 GPU 示例 | `nvcc`；MPI 示例另需 MPI C++ wrapper |

GPU 示例固定每 rank 1024 个 FP32 元素，输入全为 `rank+1`，规约为 Sum；输出每元素应为 `P(P+1)/2`。它们是小规模正确性示例，**不是 benchmark，也不是完整生产级容错实现**。

## 1. 无 GPU：看清每轮发送什么

本节用 CPU 上的 Ring 模拟器回答：每一轮谁把哪一块发给谁。不需要 CUDA：

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --ranks 3 --chunk-size 1
python3 nccl-tutorial/examples/ring_simulator.py --ranks 1 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

先看第一条命令。以下是**本次 macOS 上实际执行 Python 得到的完整输出**，不需要 CUDA：

```text
P=4, elements per chunk=2
input r0: [0, 1, 2, 3, 4, 5, 6, 7]
input r1: [10, 11, 12, 13, 14, 15, 16, 17]
input r2: [20, 21, 22, 23, 24, 25, 26, 27]
input r3: [30, 31, 32, 33, 34, 35, 36, 37]

ReduceScatter (snapshot all sends before receiving):
  step 0: r0->r1: c0; r1->r2: c1; r2->r3: c2; r3->r0: c3
  step 1: r0->r1: c3; r1->r2: c0; r2->r3: c1; r3->r0: c2
  step 2: r0->r1: c2; r1->r2: c3; r2->r3: c0; r3->r0: c1
  r0 owns reduced c1: [68, 72]
  r1 owns reduced c2: [76, 80]
  r2 owns reduced c3: [84, 88]
  r3 owns reduced c0: [60, 64]

AllGather:
  step 0: r0->r1: c1; r1->r2: c2; r2->r3: c3; r3->r0: c0
  step 1: r0->r1: c0; r1->r2: c1; r2->r3: c2; r3->r0: c3
  step 2: r0->r1: c3; r1->r2: c0; r2->r3: c1; r3->r0: c2
output r0: [60, 64, 68, 72, 76, 80, 84, 88]
output r1: [60, 64, 68, 72, 76, 80, 84, 88]
output r2: [60, 64, 68, 72, 76, 80, 84, 88]
output r3: [60, 64, 68, 72, 76, 80, 84, 88]
PASS: 6 rounds; sent per rank = 12 elements
```

这样读一遍输出：

1. `--chunk-size 2` 表示**每块**两个元素。四个 rank 各有四块，所以每个输入有 8 个元素。rank r 的数组为 `[10r,10r+1,...,10r+7]`，不是后面 GPU 示例的常数输入。
2. `r0->r1: c0` 表示 r0 把数组的第 0 块发给 r1；c0 是位置编号，不是来源 rank。ReduceScatter 阶段先让每个 rank 拿到一个完整求和块，AllGather 再把这些块传遍各 rank。
3. 最终第 0 个位置为 `0+10+20+30=60`，第 1 个位置为 `1+11+21+31=64`。程序检查**所有 rank 的全部元素**，不只检查这两个数。
4. 两阶段各 `P-1=3` 轮，每 rank 每轮发 2 个元素，所以末行为 6 轮、每 rank 共发送 12 个元素。这不是耗时或带宽测量。

另两条规模命令也已实际运行：三 rank 输出均为 `[30, 33, 36]`，末行为 `PASS: 4 rounds; sent per rank = 4 elements`；一 rank 保留 `[0, 1]`，末行为 `PASS: 0 rounds; sent per rank = 0 elements`。`--self-test` 的本机实测输出为：

```text
PASS: 85 cases (P=1..9, multiple chunk sizes, signed data, invalid shapes)
```

`--self-test` 使用固定随机种子，覆盖 P=1..9、多个块长、含负数的随机整数、空输入及非法形状。模拟器不支持非等长或不能等分的输入，这是教学模型限制，不是 NCCL AllReduce 的限制。它是 CPU 上的整数教学模型，不调用 NCCL，也不证明真实 NCCL 本次一定选择 Ring。

<details><summary>深入：为什么 r0 中途拿到 c1，而不是 c0？</summary>

在 [ring_allreduce](ring_simulator.py) 中查找 `chunk` 和 `owned`：

- rank r 发给 `(r+1) mod P`，从 `(r-1) mod P` 接收。
- ReduceScatter 第 s 轮发送块 `(r-s) mod P`，结束后 rank r 持有全规约块 `(r+1) mod P`。
- AllGather 第 s 轮发送块 `(r+1-s) mod P`，将这些完整块传遍所有 rank。

这是 **AllReduce 内部阶段**的偏移归属，不是公共 `ncclReduceScatter` API 的输出布局。公共 API 必须使 rank r 得到第 r 块；实现可通过块编号/顺序映射满足它。对照[集合通信章](../02-collectives.md)与[算法章](../04-algorithms.md)时，先对齐“第一轮发哪块”，不要只比较 c0/c1 的名字。

程序先收集本轮所有发送快照，再统一接收，避免 Python 顺序执行让一个块在“一轮”里前进多跳；`sources` 集合检查没有重复加入同一 rank 的贡献。

</details>

## 2. Linux：编译单进程示例

本节把单进程示例编译出来并跑通两卡。先完成[环境章](../00-environment.md)中的 NCCL 构建：

```bash
export ROOT="$PWD"
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME="$ROOT/nccl/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

make -C nccl-tutorial/examples CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
ldd nccl-tutorial/examples/build/single_process_allreduce
./nccl-tutorial/examples/build/single_process_allreduce 2
```

默认 `make` 目标仅构建单进程程序，使用 `nvcc`，**不需要 MPI**；规则见 [Makefile](Makefile) 的 `all` 和 `$(BUILD)/single_process_allreduce`。参数是 GPU 数，不是元素数：必须为正整数，且不超过可见 GPU 数。程序使用可见列表的前 N 张卡，尊重 `CUDA_VISIBLE_DEVICES`；运行前确认这些 GPU 已获准使用。

<details>
<summary>基础速查：完整程序的八段带练（C++ 容器、分配、拷贝与 NCCL 调用）</summary>

### 跟着完整程序走八段

先画出 [single_process_allreduce.cu](single_process_allreduce.cu) 中 `main` 的生命周期：

```text
CPU main
  检查参数、可见 GPU 数 → 建立每 GPU 的资源表
  → 逐卡选设备、建 stream、分配 send/recv、填 CPU 输入并提交 H2D
  → ncclCommInitAll：建立所有 rank 的通信关系
  → 一对 GroupStart / GroupEnd：为所有 GPU 提交 AllReduce
  → 查询所有 NCCL 异步状态与 stream，直到 GPU 工作全部完成
  → 逐卡 D2H 读回、校验全部元素、destroy/free
  → 汇总 PASS / FAIL 与退出码
```

把“两卡求和”具体画出来（每个数组均为 1024 个 FP32 元素，`...` 表示重复值）：

```text
CPU 内存                         GPU 显存（各设备分别分配）
inputs[0] = [1,1,...,1] --H2D--> GPU 0 / rank 0: send[0] = [1,1,...,1]
                                stream[0]      recv[0] = [待写入]
inputs[1] = [2,2,...,2] --H2D--> GPU 1 / rank 1: send[1] = [2,2,...,2]
                                stream[1]      recv[1] = [待写入]

NCCL AllReduce(Sum)，对每个位置 i = 0..1023：
  send[0][i] + send[1][i] = 1 + 2 = 3
  → recv[0] = [3,3,...,3]，recv[1] = [3,3,...,3]

GPU 全部完成后：
  recv[0] --D2H--> CPU output → 校验 rank 0
  recv[1] --D2H--> CPU output → 复用同一数组，校验 rank 1
```

这里一个进程、一个 CPU 线程就能管理两个 rank；rank 是通信组内的编号，不必等于进程数。这是**逐元素求和并把结果交给每个 rank**，不是把整个数组加成一个标量，也不是拼接数组。图只描述数据契约，不规定 NCCL 选 Ring 还是其他算法。

下面八段均摘自上述完整源码，行号按当前文件给出，并附可搜索的符号；摘录用于对照阅读，**不是可以各自编译的独立程序**。

#### 1. 参数与资源表：为什么 vector 要按 GPU 分份？

定位：`main` 参数检查在第 28–48 行；以下为第 49–57 行，搜索 `ndev`、`inputs`。

```cpp
  const int ndev = static_cast<int>(requested);
  constexpr size_t count = 1024;
  const size_t bytes = count * sizeof(float);
  std::vector<int> devices(ndev);
  std::vector<ncclComm_t> comms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<float*> send(ndev), recv(ndev);
  std::vector<std::vector<float>> inputs(ndev, std::vector<float>(count));
  std::vector<float> output(count);
```

- 前面的 `strtol` 检查参数是否为完整的正整数，`cudaGetDeviceCount` 检查可见卡数，通过后才把 `requested` 转为 `int ndev`。无参数时默认 2。
- `count=1024` 是**每 rank 的元素数**；CUDA 分配/拷贝使用字节数，所以 `bytes=count*sizeof(float)=4096`。后面的 NCCL 调用传 `count`，不能误传 `bytes`。
- `vector<T>(ndev)` 是在 CPU 上建立 ndev 个槽位。`devices[r]` 记设备编号，`comms[r]` 记该 rank 的通信句柄，`streams[r]` 记该 GPU 的执行队列；不同 GPU 的资源不能混用。
- `send`、`recv` 两个 vector **只存指针**，尚未分配显存。稍后每卡各分配一份输入与一份输出，原始 send 不被本次 out-of-place AllReduce 覆盖。
- `inputs` 是每 rank 一行、每行 count 个 float 的 CPU 数组，保证两份输入同时存活；`output` 只有一行，因为读回和校验按 rank 顺序完成，可以复用。

#### 2. 选设备、建 stream、分配显存

定位：第 63 行的逐卡循环内，第 64–68 行；搜索 `cudaStreamCreateWithFlags`。

```cpp
    devices[r] = r;
    CUDA_CHECK(cudaSetDevice(devices[r]));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&send[r]), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&recv[r]), bytes));
```

- `devices[r]=r` 选可见列表中的第 r 张卡。若可见列表被重排，逻辑 device 0 不一定是机器上的物理 GPU 0。
- `cudaSetDevice` 设置当前 CPU 线程后续 CUDA 操作的目标设备，所以要在该卡的 stream 创建、分配等操作前调用。
- `&streams[r]` 是接收新 stream 句柄的位置；`cudaStreamNonBlocking` 避免与 legacy default stream 的隐式同步，**不代表本调用已完成后续 GPU 工作**。同一个 stream 内的任务仍按序执行。
- 两次 `cudaMalloc` 各分配 `bytes` 字节；`&send[r]` / `&recv[r]` 用来接收设备地址，`reinterpret_cast<void**>` 适配 API 参数类型。它们不是可在 CPU 上直接解引用的数组；recv 暂无有效结果，不需预先清零，因为 AllReduce 将写满它。
- `CUDA_CHECK` 是检查返回码的护栏，不参与求和；错误处理统一见第 5 节。

#### 3. 填入 1 和 2，提交 H2D，保留 CPU 输入

定位：同一循环内第 69–71 行；搜索 `cudaMemcpyHostToDevice`。

```cpp
    for (size_t i = 0; i < count; ++i) inputs[r][i] = static_cast<float>(r + 1);
    CUDA_CHECK(cudaMemcpyAsync(send[r], inputs[r].data(), bytes,
                               cudaMemcpyHostToDevice, streams[r]));
```

- 内层循环只在 CPU 上**准备输入**：rank 0 填 1，rank 1 填 2；它不是规约算法。
- 拷贝参数依次为目标 GPU 指针、源 CPU 指针、字节数、H2D 方向、该卡的 stream。`.data()` 取得 vector 连续存储的地址。
- 后面的 AllReduce 也提交到 `streams[r]`，因此该 GPU 必须先完成这次 H2D 才会读取 send；不必在两者之间让 CPU 逐卡同步。
- 不要在提交后立刻销毁或改写源数组。`inputs` 定义在外层 `main` 中，跨过所有循环一直保留到通信完成之后；这比在循环里创建临时输入更容易保证寿命。普通 host vector 的异步拷贝边界见第 5 节。

#### 4. InitAll：让两个 rank 加入同一个通信组

定位：第 73 行；搜索 `ncclCommInitAll`。

```cpp
  NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devices.data()));
```

- `comms.data()` 提供通信句柄数组的写入位置，`ndev` 是组内设备数，`devices.data()` 指定设备列表。
- 列表顺序决定 rank：本例第 r 项是逻辑设备 r，返回的 `comms[r]` 就代表这个 rank。一次调用建立本进程内的整个组，而不是每卡单独建互不相关的组。
- 单进程可直接用 InitAll，不需要 `ncclGetUniqueId`、MPI 或手工广播 ID；选读的 MPI 示例才用另一条初始化路径。
- 初始化通信组不替代 CUDA stream 的完成等待。

#### 5. 一对 group 包住所有 GPU 的 AllReduce

定位：第 75–81 行；搜索 `ncclGroupStart`、`ncclAllReduce`。

```cpp
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclAllReduce(send[r], recv[r], count, ncclFloat, ncclSum,
                             comms[r], streams[r]));
  }
  NCCL_CHECK(ncclGroupEnd());
```

- Start 在循环外、End 在循环外，**只有一对**，将这个 CPU 线程为所有 GPU 发出的调用组织成一组；避免先在一个 rank 的调用里等待尚未提交的其他 rank。不要改成每卡一对并在卡间等待。
- `send[r]` 是输入，`recv[r]` 是独立输出；`count` 是 1024 个元素，`ncclFloat` 对应 float，`ncclSum` 指定逐元素相加。
- `comms[r]` 确定参与者，`streams[r]` 指定执行队列；所有 rank 必须以匹配顺序调用同一 collective，且元素数、类型、规约操作一致。
- CPU 循环只负责**提交**调用，真正的数据通信和规约由 NCCL 驱动 GPU 执行。GroupEnd 返回不等于 GPU 已算完，不能此时就打印 recv 或释放显存。

#### 6. 等 GPU 完成：查询是护栏，不是核心算法

定位：第 83–105 行等待循环；以下摘自第 88–93 行，搜索 `ncclCommGetAsyncError`、`cudaStreamQuery`。

```cpp
      ncclResult_t state = ncclSuccess;
      NCCL_CHECK(ncclCommGetAsyncError(comms[r], &state));
      NCCL_CHECK(state);
      cudaError_t status = cudaStreamQuery(streams[r]);
      if (status == cudaErrorNotReady) all_done = false;
      else CUDA_CHECK(status);
```

- 外层每轮先令 `all_done=true`，内层逐卡 `cudaSetDevice` 后执行上述查询；任一卡未完成，就继续等。不是只等最后一张卡。
- 第一个 `NCCL_CHECK` 检查“查询调用本身是否成功”；第二个检查写入 `state` 的**通信异步错误**。查询成功并不等于通信没有错误。
- `cudaStreamQuery` 不阻塞 CPU：`cudaErrorNotReady` 是“还在执行”，不是故障；其他状态交给 `CUDA_CHECK`，其中 `cudaSuccess` 表示该 stream 的已提交工作完成。
- 所有卡都完成后 `break`，才进入下一段读回。未完成时每轮睡眠 1 ms，避免 CPU 空转；循环不搬运或累加测试数组。
- 第 83 行在 GroupEnd 返回后设置 60 秒期限。超时尝试 abort 并失败退出；**这不是整个程序的 60 秒总超时**，覆盖范围见第 5 节。

#### 7. D2H 与全元素校验：first 只是日志摘要

定位：第 107–118 行；搜索 `expected`、`rank_wrong`。

```cpp
  const double expected = static_cast<double>(ndev) * (ndev + 1.0) / 2.0;
  size_t wrong = 0;
  for (int r = 0; r < ndev; ++r) {
    CUDA_CHECK(cudaSetDevice(devices[r]));
    CUDA_CHECK(cudaMemcpy(output.data(), recv[r], bytes, cudaMemcpyDeviceToHost));
    size_t rank_wrong = 0;
    for (float value : output) {
      if (static_cast<double>(value) != expected) ++rank_wrong;
    }
    wrong += rank_wrong;
    std::printf("rank=%d device=%d first=%.0f expected=%.0f wrong=%zu\n",
                r, devices[r], output[0], expected, rank_wrong);
```

- 输入依次是 `1,2,...,ndev`，因此**每个位置**的期望值都是 `ndev*(ndev+1)/2`，两卡为 3。
- 先前已等 GPU 完成；这里再按卡选设备，用同步 `cudaMemcpy` 将 recv 拷到 CPU `output`。参数仍是目标、源、字节数、方向，返回后 CPU 才检查数组。
- `for (float value : output)` 遍历该 rank 的全部 1024 个元素；`rank_wrong` 记录该卡错误数，`wrong` 累计所有卡。不是只检查 `output[0]`。
- 日志中的 `first` 仅方便人眼查看；`%.0f` 打印不带小数的浮点数，`%zu` 打印 `size_t` 计数。是否通过看全部 rank 的 `wrong=0` 和最后的 `PASS`。

#### 8. 正常释放：通信句柄不替你释放用户显存

定位：上述逐卡校验循环末尾第 119–122 行；汇总输出和返回值在第 124–125 行。

```cpp
    NCCL_CHECK(ncclCommDestroy(comms[r]));
    CUDA_CHECK(cudaFree(send[r]));
    CUDA_CHECK(cudaFree(recv[r]));
    CUDA_CHECK(cudaStreamDestroy(streams[r]));
```

- 所有 GPU 工作已完成，且当前卡读回结束，才销毁 `comms[r]`；destroy 释放通信资源，不负责用户自己申请的 send/recv。
- 分别 free 两块独立显存，再销毁该卡 stream；不能提前释放仍在被 GPU 使用的资源，也不能漏掉第二块或重复释放。
- 循环之后打印 `wrong == 0 ? "PASS" : "FAIL"`，并据此返回成功或失败退出码。CPU 上的 vector 在离开 `main` 时自动释放；宏错误退出不等同于走完这条正常清理路径。

</details>

### 两卡完整 stdout 应该是什么？

对上面的两卡命令，以下是**理论预期，不是本机 CUDA 实测**。假设编译头文件与运行时库版本均为教程基线 `23203`，且未额外开启 NCCL 日志，程序自身的完整 stdout 为：

```text
NCCL header=23203 runtime=23203 ranks=2 count=1024
rank=0 device=0 first=3 expected=3 wrong=0
rank=1 device=1 first=3 expected=3 wrong=0
PASS
```

版本行来自 `NCCL_VERSION_CODE` 和 `ncclGetVersion`（第 59–62 行），你的安装可能打印不同数值。不必为了匹配文档强改版本号；若与预期安装不符，应检查头文件与动态库路径。一卡的退化验证每元素应为 1；有四张获准使用的可见卡时，四卡结果应为 10。

### 固定路线：command → 成功看到什么 → 失败先查什么

GPU 行仅在 Linux CUDA 环境执行，沿用本节前面的环境变量。先跑基线，最后两行仅在相应问题出现时参考；不要把排错过程变成首次运行的必做实验。

| command（项目根目录） | 成功看到什么 | 失败先检查什么 |
|---|---|---|
| `python3 nccl-tutorial/examples/ring_simulator.py --self-test` | `PASS: 85 cases ...` | Python 3 与工作目录；这一步不依赖 CUDA/MPI |
| `make -C nccl-tutorial/examples CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"` | 退出码 0，生成 `build/single_process_allreduce`（位于 examples 下） | `nvcc` 是否位于 `$CUDA_HOME/bin`，NCCL 是否已构建，include/lib 路径是否正确 |
| `ldd nccl-tutorial/examples/build/single_process_allreduce` | `libnccl`、`libcudart` 等依赖均解析到实际路径，无 `not found` | **库找不到**：区分编译期 `-I/-L` 与运行期 `LD_LIBRARY_PATH`；先修正本节 export 再运行 |
| `./nccl-tutorial/examples/build/single_process_allreduce 1` | `first=1 expected=1 wrong=0`，最后 `PASS` | CUDA 错误先看驱动是否可用、任务是否获配 GPU；一卡通过尚未验证卡间通信 |
| `./nccl-tutorial/examples/build/single_process_allreduce 2` | 上述完整两卡 stdout，所有 `wrong=0` | **可见卡不足**会报 `Requested 2 GPUs, but only N are visible`；检查分配与 `CUDA_VISIBLE_DEVICES`，仅一张时改跑 1，不强占其他卡 |
| `./nccl-tutorial/examples/build/single_process_allreduce bad`（可选参数检查） | **预期拒绝**：stderr 为 `gpu-count must be a positive integer`，非零退出码 | **bad arg**：参数只能是一个正整数，不能传 `2x`、0 或把 1024 当元素数参数；多余参数会显示 `Usage` |
| `NCCL_DEBUG=INFO ./nccl-tutorial/examples/build/single_process_allreduce 2`（挂起排查） | 额外初始化/通信日志，最终仍应 `PASS` | **NCCL 挂起**：看最后停在初始化、GroupEnd 还是已提交后的等待，核对 GPU 分配与日志中的 transport；60 秒只覆盖最后一种等待，继续按[排障章](../12-debugging.md)定位 |

## 3. 选读：MPI 构建和单机验证

本节回答：跨进程时谁负责启动进程和交换引导信息。先完成单进程两卡验证再来这里。MPI 的作用是启动多个进程并交换引导信息，不是第 2 节的依赖。

[mpi_allreduce.cu](mpi_allreduce.cu) 不含自定义 device kernel，全部是 host API 调用。因此 Makefile 的 `mpi` 目标用 `mpicxx -x c++` 编译并链接 CUDA runtime/NCCL，而不是沿用普通目标的 `nvcc`，这样不需要猜 MPI 库的依赖列表。

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

按稳定符号读引导代码。`MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` 建立节点内共享内存通信组，再从中取得 `local_rank`，不是用 global rank 猜本地设备。`MPI_Bcast` 广播 rank 0 生成的 NCCL ID。`MPI_Allreduce` 汇总 CPU 校验错误数，**没有用 MPI 对 GPU 测试数组执行 AllReduce**。此外 `MPI_Allgather` 交换节点内的 PCI bus ID，用于下述冲突检查。

### GPU 可见性规则

- 所有本地进程看到相同的完整 GPU 列表：`device=local_rank`。
- 调度器已为每个进程分配独占单卡、每个进程只见一张：`device=0`。
- 程序交换选中的 PCI bus ID，拒绝同节点多个 rank 选中同一个物理设备。

此简化映射的适用范围与故障处理统一见第 5 节；遇到冲突应修正任务分配，不要删除检查强行运行。

## 4. 选读：两机各两卡

本节回答：两台机器各两卡时，命令和预期输出长什么样。先在两台节点上确认：相同源码与库、相同绝对目录、相容的 CUDA/驱动、MPI 可正常启动进程、网络互通。`node-a/node-b` 是待替换的主机名。

```bash
export NCCL_DEBUG=INFO
mpirun --host node-a:2,node-b:2 -np 4 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x NCCL_DEBUG \
  "$ROOT/nccl-tutorial/examples/build/mpi_allreduce"
```

理论预期日志显示每台机器两个 local rank，global rank 共 0..3；结果为 10，`total_wrong=0 PASS`。不要先假定网络用了 IB，按[传输章](../07-transports.md)和[排障章](../12-debugging.md)检查实际 transport 日志。

若节点目录不同，应部署到约定位置或按集群机制指定工作目录和环境。示例不会替你分配 GPU、配置 SSH、修改防火墙或修复 RDMA 权限。

## 5. 集中看护栏与适用边界

前面各节的“但不是”集中在这里，逐条核对：

- **返回码护栏，不是算法。** 单进程的 `CUDA_CHECK` / `NCCL_CHECK`（第 12–26 行）执行调用、检查返回值，出错时向 stderr 打印文件、行号和错误描述并退出。`do { ... } while (0)` 是让宏表现为一条语句的写法，不是 GPU 求和循环。普通错误是 fail-fast，不保证走完正常资源释放；MPI 版还调用 `MPI_Abort` 终止自身作业，不做训练恢复。
- **提交成功不等于完成。** 两例使用默认 blocking communicator，不演示完整的非阻塞 init 状态机。“blocking communicator”和 `cudaStreamNonBlocking` 是不同概念。GPU 完成由 stream 状态确认，NCCL 异步错误另查。
- **60 秒仅是已提交工作等待的期限。** 单进程从 GroupEnd 返回后、MPI 版从 AllReduce 返回后开始计时。它不覆盖可能阻塞的初始化、GroupEnd、MPI 调用、读回或清理本身。单进程等待超时尝试 abort communicator，但 abort 也不是有界返回保证。需要全作业超时时，用调度器 walltime 或管理员允许的作业控制方式，不要把轮询当作完整生产容错。
- **普通 host vector 不是 pinned memory。** `cudaMemcpyAsync` 在这种内存上可能涉及暂存或 host 阻塞。同 stream 顺序保证这里的数据依赖，但不能从片段推导“必然实现 H2D/compute overlap”。输入内存仍保留至完成。
- **本例验证小规模整数的和。** 两卡等小规模结果在 FP32 下可精确表示，所以使用精确比较。任意浮点规约可能因求和顺序产生误差，不能一律要求逐位一致。
- **MPI 简化映射只面向独占整卡。** 不覆盖 MIG、多 rank 同卡；任意重排且各进程不同的多卡可见列表也不在假设内。PCI 冲突检查不能代替正确的调度器分配。

## 6. 跟着修改：C1 / C2，再选进阶练习

本节对应[练习章](../15-exercises.md)的 C1/C2。先跑通未修改的完整 `single_process_allreduce.cu`，保留基线以便对照。**一次只做一个变体，C2 从原始 out-of-place 基线开始，不接着 C1 改**。下面是对完整程序的修改提示，不是另一份可独立运行的残缺程序。每次改后重新执行第 2 节 `make` 和两卡命令，并继续检查所有元素及退出码。

### C1：让输入输出用同一块显存（合法 in-place）

选择保留 `send[r]`，沿用它已有的 H2D 初始化：

1. 在 `main` 的指针表（原第 55 行）只保留 `std::vector<float*> send(ndev);`，并删除 recv 的 `cudaMalloc`（原第 68 行）。不要把 `recv[r]` 简单赋为 `send[r]` 后还保留两次 free。
2. 在 `ncclAllReduce`（原第 78 行）把第二个参数从 `recv[r]` 改为 `send[r]`，形成 `ncclAllReduce(send[r], send[r], count, ncclFloat, ncclSum, comms[r], streams[r])`。外层 `NCCL_CHECK`、设备选择与整对 group 都保留。
3. D2H（原第 111 行）的源也从 `recv[r]` 改成 `send[r]`，否则会检查错误的缓冲。等待完成后，这块显存保存的是规约结果，不再是原始输入。
4. 删除 `cudaFree(recv[r])`（原第 121 行），保留一次 `cudaFree(send[r])`。CPU inputs、stream、communicator、等待、全元素校验都不变。改完后搜索 `recv`，应已没有该变量的残留引用。
5. 验收：两卡 stdout 的结果部分仍与基线相同，每卡 `first=3 expected=3 wrong=0`，最后 `PASS`。显存从每卡两块变成一块，不能只以“没崩溃”为通过。

### C2：从 AllReduce 改成 AllGather（不再求和）

目标是每 rank 保留自己的 1024 个输入，最终每卡收到按来源 rank 排列的完整拼接结果。两卡时输出长度为 2048：前 1024 个全是 1，后 1024 个全是 2。

1. **分清发送量与接收量。** 保留 `count=1024`、`bytes=count*sizeof(float)` 用于 send 和 H2D。在它们之后增加 `const size_t recv_count = count * static_cast<size_t>(ndev);` 与 `const size_t recv_bytes = recv_count * sizeof(float);`。CPU `inputs` 每行仍为 count；`output` 改为 `std::vector<float> output(recv_count);`。
2. **配套扩大 GPU 与 CPU 的接收容量。** recv 的 `cudaMalloc` 长度改成 `recv_bytes`，D2H 拷贝长度也改成 `recv_bytes`。不能只扩大其中一个：两卡时每卡 send 为 4096 字节，recv 与 CPU output 各为 8192 字节。
3. **只替换 collective 调用，保留提交框架。** 在 group 内把 AllReduce 调用替换为 `ncclAllGather(send[r], recv[r], count, ncclFloat, comms[r], streams[r])`，仍包在 `NCCL_CHECK` 中。第三参数是每 rank 的 **sendcount**，不是 recv_count；AllGather 没有 `ncclSum` 参数。
4. **替换验收条件，而不是继续期待 3。** 删除原来的统一 `expected` 公式。在每张接收卡 r 的校验位置，用外层 `source_rank=0..ndev-1`、内层 `i=0..count-1` 遍历全部输出，比较 `output[source_rank * count + i]` 与 `static_cast<float>(source_rank + 1)`，不等则增加 `rank_wrong`。保留每卡清零 `rank_wrong`、累计到 `wrong` 的逻辑。
5. **更新日志，避免留下失效变量。** 将原含 `first/expected` 的整条 printf 改为 `std::printf("rank=%d device=%d wrong=%zu\n", r, devices[r], rank_wrong);`。末尾的 `PASS/FAIL` 和退出码判断不变；等待、两块显存释放、stream/communicator 销毁也保持不变。
6. **验收。** 每张卡所有块都通过检查，两个 rank 均打印 `wrong=0`，最终 `PASS`。rank 0 和 rank 1 的接收布局都必须是 `[1,...,1,2,...,2]`，不能让“本 rank 的块”一律排第一。公共 API 按 source rank 排布，不能照搬 CPU 模拟器内部阶段的偏移归属。

### 选读：全元素校验、ReduceScatter 与事件

- **校验练习（C3）：** 在 D2H 后、校验循环前，临时给 CPU `output[count/2]` 加 1，保持 `output[0]` 不变。未改动的两卡基线应每 rank 检出 1 个错误并最终 `FAIL`；这能说明只看 first 会漏错。实验后去掉故障注入再确认 `PASS`。
- **ReduceScatter：** 先按[集合通信章](../02-collectives.md)核对参数：每 rank 的 send 容量为 `ndev * recvcount`，recv 容量为 recvcount；公共 API 给 rank r 第 r 块。再按块构造输入与期望值，不要沿用模拟器的 `owned=(r+1)%P` 作为公共 API 验收条件。
- **事件与 kernel（C4，进阶选读）：** 基线中 H2D 与通信在同一 stream，首次运行不必引入事件。需要独立生产/消费 stream 时，先在纸上画 `produce → ready event → comm stream wait → AllReduce → done event → compute stream wait → consume`，再按[CUDA 语义章](../03-cuda-semantics.md)接入 kernel。done 事件必须在有效的 group 提交边界之后记录；每卡使用属于该卡的 stream/events，并在完成后释放。不要仅把基线的等待循环删掉就宣称实现了重叠。

性能实验请使用 nccl-tests 并遵循[测量章节](../10-nccl-tests.md)，不要把本程序的 1 ms 轮询 sleep、初始化和 PCI 映射检查计入通信吞吐。

## 7. 本次修订验证范围

本次在 macOS 上实际运行了第 1 节三条规模命令及 `--self-test`（85 项通过），文中的 CPU 输出来自这些执行。Makefile 普通/MPI 两个目标仅做 dry-run，验证命令展开；这不等于编译通过。当前环境无 CUDA，**未编译或运行 CUDA/NCCL/MPI 示例，也未实测 C1/C2、事件或跨机练习**；GPU stdout 均为按源码推导的理论预期。
