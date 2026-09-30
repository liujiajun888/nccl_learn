# 从会用到看懂：NCCL 系统学习教程

这不是 API 速查表，而是一条完整的学习链路：**通信的数学含义 → 正确使用 CUDA/NCCL → 初始化与执行机制 → 数据如何跨 GPU/跨机器 → 如何测量、解释和优化**。

教材主线面向**有一定 CUDA 基础、做过 UMD（用户态驱动）开发，但第一次系统学习 NCCL**的读者。默认你理解设备内存、命令提交与基本同步；不预设你熟悉集合算法、多进程协同或 RDMA。中文讲机制，保留英文术语与源码符号。

## 推荐主线：从本地提交模型走向多 rank 协同

你不必重新学习 `cudaMalloc` 或指针运算。真正要补的是跨 rank 的进度观念：**本地动作成功，只是整个集合操作的第一步。** UMD 的经验能帮你提问，但 NCCL 的对象要按它自己的职责来理解。

```text
01 → 02 → 03             契约：结果、布局、提交与完成
           ↓
          04             算法：Ring/Tree 如何组织数据
           ↓
05 → 06 → 07             通信关系：团队、邻居、数据通路
           ↓
      08 → 09             执行机制：主机任务、设备协议
           ↓
10 → 11 → 12             测量、优化与调试
```

**文件名和章号就是主线阅读顺序：从 01 一直读到 12，无需跳章。** 00 是实验环境准备；13 为进阶专题；14/15 是随主线使用的源码索引与练习；16 为训练集成。章节内标为“选读”的特殊路径可以暂缓，不影响进入下一章。有 GPU 时并行运行 examples，遇错随时查 12。

**注意：**
- 本地 kernel 已提交，不等于对端已提供数据；内存可访问，不等于数据已就绪；网络请求完成，也不等于整个 collective 或应用消费已完成。
- communicator 不是 CUDA context，kernel plan 不是硬件命令缓冲，channel 也不是 stream 或某个固定 SM。后文按各自的执行者与状态来区分。

| 阶段 | 机制问题 | 阅读与源码入口 | 验证产物 |
|---|---|---|---|
| 1. 契约与进度 | 本地有序提交还缺哪些跨 rank 条件？ | [01](01-mental-model.md) 的分层模型、[02](02-collectives.md) 的 count/布局/SendRecv、[03](03-cuda-semantics.md) 的 group 与完成边界 | 为一次 collective 写清各 rank 的参数、提交顺序和完成依据 |
| 2. 算法直觉 | 为什么分块，如何保证每份贡献恰好规约一次？ | [04 的 4.2～4.7](04-algorithms.md)，先手算 Ring，再比较 Tree | 追踪一个 chunk 的贡献来源、去向和总传输量 |
| 3. 通信关系与资源 | 团队如何形成，邻居与访问条件由谁建立？ | [05](05-communicator.md) 初始化、[06](06-topology.md) 图与邻接、[07](07-transports.md) setup/connect 与 proxy | 区分身份已知、路径可行、连接就绪；画出 GPU、CPU proxy、NIC 的分工 |
| 4. 主机执行 | API 意图怎样变成 GPU 能消费的工作？ | [08](08-host-execution.md)，`ncclEnqueueCheck`、`ncclLaunchPrepare`、`ncclLaunchKernel` | 追踪一份 task/plan/work：谁创建、谁消费、何时可回收 |
| 5. 设备协议 | 已有连接，如何防止读旧值或覆盖未消费数据？ | [09](09-device-protocols.md) 的 `waitPeer/postPeer` 与四字段追踪 | 为一片数据列出 ready、credit 和可见性条件 |
| 6. 验证与诊断 | 慢在准备、等待 peer、传输，还是资源竞争？ | [10](10-nccl-tests.md) 测量、[11](11-performance.md) 优化、[12](12-debugging.md) 调试 | 单变量实验、完整配置、正确性及时间线/统计证据，形成分层排查顺序 |

**验证产物不要求另写报告文件。** 在笔记或纸上列出状态即可；没有 GPU 时做源码追踪与算法模拟，硬件路径和性能结论保留为待实测。

### 哪些要学，哪些可以直接跳过？

- **主线不能跳：** rank/comm 语义、集合布局、group 的前进条件、bootstrap 与 bulk 分离、连接/注册、host 与 device 工作交接、算法分块、协议可见性和背压。这些分布式约束不会因为熟悉 CUDA 就自动成立。
- **结合经验快速读：** 第 01 章的普通 buffer/stream 定义，第 00 章的 shell 参数解释，示例中的 C++ 容器、分配和拷贝语法。保留作速查即可。
- **按专题读：** 多机启动与 MPI、非阻塞管理、Graph、NVLS/CollNet、Device API/GIN/RMA、训练集成。先建立普通 host collective 的模型，再看改变执行者和资源归属的特殊路径。

有 Linux NVIDIA GPU 时，尽早用 [00](00-environment.md) 和[单进程示例](examples/README.md)建立正确性基线，与机制阅读并行；没有 GPU 则继续手算和源码追踪，不把实验环境当作阅读门槛。

<details>
<summary>补充路线：需要数组、CUDA 或命令行带练时展开；不作为上述主线的必修</summary>

## 基础带练：先完成一条小而完整的路线

**不用先学 MPI、RDMA 或深度学习，也不用先通读 17 章。** 第一轮只做一件事：让两张 GPU 的数组对应位置相加，并能解释结果为什么正确、何时可以使用。

```text
先算答案           再安排顺序             最后跑通和解释
两卡输入/输出 -> stream + group -> 构建 -> 完整示例 -> 看懂一行测试结果
       01、02           03          00       examples          10
```

基础带练只需完成下面的实践关卡；系统学习则按 01～12 顺读。两者不要求同一天完成：

| 关卡 | 去哪里读 | 本关只做什么 | 过关标志 |
|---|---|---|---|
| 1. 能算出结果 | [01 全局模型](01-mental-model.md)、[02 集合通信](02-collectives.md)的两卡例子 | 比较 AllReduce、AllGather、ReduceScatter | 能画出每张卡输入/输出，并算出元素数与字节数 |
| 2. 能安排顺序 | [03 CUDA 语义](03-cuda-semantics.md)第 1～6 节 | 理解同 stream、group、等待与缓冲寿命 | 能解释 CPU 返回后为什么不能直接读取结果 |
| 3. 能找到正确的库 | [00 环境](00-environment.md)的单机主线 | 检查环境、构建、确认运行库 | `ldd` 指向预期 NCCL，固定尺寸两卡测试校验通过 |
| 4. 能读懂程序 | [示例带练](examples/README.md)的单进程部分 | 按准备→初始化→提交→等待→校验→清理逐段阅读 | 两卡每个元素为 3，`wrong=0` 且最终 `PASS` |
| 5. 能读懂测量 | [10 nccl-tests](10-nccl-tests.md)的入门部分 | 先解释一行输出，再扫描尺寸 | 不把两列当两张卡，不把 busbw 当链路计数器 |
| 6. 能独立检查 | [15 分级练习](15-exercises.md)的入门检查站 | 不看答案重做一个两卡例子 | 能说明结果、容量、执行顺序与校验标准 |

没有 Linux NVIDIA GPU 时，先做前两关和 Python 模拟；第 3～5 关可以阅读，留到有设备时验证，不把模拟结果当成实机通过。

### 需要先会多少 C++ / CUDA？

- 能读 `for`、数组、函数参数、指针和 `std::vector`；知道 `float* p` 是地址，不是已经装好数据的数组。
- 能算 `元素数 × sizeof(类型) = 字节数`，理解 `float* p + 2` 前进两个 float，而非两个字节。
- CUDA 只需先认识“选设备、分配设备内存、复制、按 stream 排队、等待”这几步，示例会逐段解释；第一份程序不要求你先写复杂 GPU kernel。

前两点不熟时，先补 C/C++ 的数组与指针，再读第 02 章的原地布局。模板、PTX、warp、网卡术语都暂时不用背，它们不是第一轮的考点。

### 基础带练完成后

这些关卡只服务于补基础和首次正确运行，不另设一套“源码必修顺序”。完成后按编号从第 04 章 Ring/Tree 核心继续，顺读到初始化、连接与执行机制；第 14/15 章随对应阶段使用。

源码链接用于核对当前问题，不要求每遇到一个链接就跳出去通读整份源码。MPI、非阻塞管理、Graph 和特殊算法仍按需要选读。

</details>

## 版本与适用范围

| 项目 | 本教材使用的本地基线 |
|---|---|
| NCCL | **2.32.3**，commit `12df1a11afad322be5a204a2db890161cbf8131d` |
| nccl-tests | **2.20.0**，commit `b4d5beebca8a76cf01335f724d154b9b9d394d96` |
| 初稿 / 入门带练修订 | 2026-09-22 / 2026-09-26 |
| GPU 实验环境 | Linux + NVIDIA GPU + 匹配的 CUDA Toolkit/驱动；多机增加网络与 MPI |
| 本地编写环境 | macOS arm64，未安装 CUDA，不能运行 NCCL GPU 实验 |

源码行号针对上述 commit；更新仓库后以**文件路径与符号名**重新定位。先用 Ring/Simple 等常规 host API 路径建立模型，再学本版本的 device API、GIN、对称内存、CE 和新调度路径——任何一条路径都不能代表 NCCL 的全部。

特别注意：当前源码已经提供 `ncclAlltoAll`、`ncclGather`、`ncclScatter` 和 per-collective config。网上针对旧版本的“没有这些 API”不能直接套用。

## 目录

| 章 | 内容 | 学完应能回答的问题 |
|---|---|---|
| [00 环境与第一轮实验](00-environment.md) | 源码构建、动态库、第一轮正确性测试 | 我实际运行的是哪一个 NCCL？ |
| [01 建立全局模型](01-mental-model.md) | rank、comm、stream、软硬件分工 | 一次 AllReduce 到底由谁完成？ |
| [02 集合通信的数学与内存布局](02-collectives.md) | 八类 collective、Send/Recv、count 与 in-place | 每个 rank 输入输出多大，结果放哪里？ |
| [03 CUDA 与 API 正确性](03-cuda-semantics.md) | stream、group、事件、异步错误 | CPU 返回成功后能否马上读结果？ |
| [04 集合通信算法](04-algorithms.md) | Ring、Tree；选读 NVLS、CollNet、PAT | 为什么没有一个永远最快的算法？ |
| [05 Communicator 生命周期](05-communicator.md) | bootstrap、初始化、连接、销毁与恢复 | GPU 在开始通信前要达成什么共识？ |
| [06 拓扑与图搜索](06-topology.md) | PCIe、NVLink、NUMA、NIC、ring/tree | 为什么换卡位或进程绑定会改变速度？ |
| [07 Transport 与网络](07-transports.md) | P2P、SHM、NET、GDR、CPU/GPU 协作 | 数据经过哪些内存和硬件？ |
| [08 Host 执行主线](08-host-execution.md) | API → task → plan → launch → proxy | 一行 ncclAllReduce 如何成为执行任务？ |
| [09 Device 与协议](09-device-protocols.md) | channel、chunk、slice、step、LL/LL128/Simple | 数据切成小块后如何有序、无覆盖地流动？ |
| [10 读懂 nccl-tests](10-nccl-tests.md) | 参数、计时、正确性、带宽公式 | busbw 到底测到了什么？ |
| [11 性能分析与调优](11-performance.md) | 小包/大包、对照实验、overlap | 先优化哪个瓶颈，如何证明有效？ |
| [12 调试与故障定位](12-debugging.md) | 挂起、日志、异步错误、系统检查 | 是调用不匹配、GPU 错误还是网络故障？ |
| [13 现代 NCCL 进阶](13-advanced.md) | Graph、注册、device API、插件与新特性 | 哪些优化需要应用与硬件共同配合？ |
| [14 源码阅读方法与索引](14-source-map.md) | 六次阅读任务、模块索引、术语表 | 如何带着问题读源码而不迷路？ |
| [15 分级练习与参考答案](15-exercises.md) | 手算、模拟、编程、测量、源码追踪 | 我是否真正建立了可验证的理解？ |
| [16 分布式训练中的 NCCL](16-training-integration.md) | DDP、分片、张量/流水/专家并行 | 通信优化为什么必须看训练关键路径？ |
| [实验代码](examples/README.md) | 单进程多卡、MPI 多进程、CPU Ring 模拟 | 如何从最小程序开始亲手验证？ |

## 每轮学习：机制解释 → 源码追踪 → 验证实验

每次只研究一个问题，例如“有 GDR 为什么还需要 CPU proxy”，而不是从 `init.cc` 第一行读到最后一行：

1. **先给出机制解释。** 画出 payload 和控制状态两条线，明确本地 GPU、CPU 和远端各在等待什么。
2. **再找实现证据。** 用[第 14 章](14-source-map.md)的入口，在当前实际分支里标出状态的生产者、消费者和复用条件。不要用目录名或函数名替代调用证据。
3. **提出能推翻解释的观察。** 例如：日志是否选了预想的 transport？第一次与稳态的差异是否来自准备？时间线是否表明 host 尚未提交？不先预设最快算法。
4. **把结论限定在证据范围内。** 正确性检查、日志、源码和性能各自回答不同的问题：日志能帮助确定分支，但不能单独证明没有内存序错误；CPU 模拟也不能证明某台机器上协议可用。

完成一段主线就做[第 15 章的机制检查站](15-exercises.md)，不用等全部章节读完。第 16 章帮助把通信时间放回训练关键路径，第 13 章按实际需求逐项选读。

## 没有 NVIDIA GPU 也能开始

在项目根目录运行：

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
```

它会展示 Ring 的两阶段以及最终校验；**这是数学/调度教学模型，不模拟 CUDA 并发、协议开销或真实硬件性能**。

前 2 章、算法手算、源码阅读和大部分练习都可离线完成。GPU 实验留到 Linux 机器执行，不能用 macOS 的模拟结果代替 NCCL 实测。

## 如何使用教材

1. 所有 shell 实验默认从同时含 `nccl/`、`nccl-tests/`、`nccl-tutorial/` 的项目根目录开始；个别章节会再次标明。
2. 标注“示意”“伪代码”“手算”的内容用于解释；带辅助函数或省略初始化的片段不能独立编译，完整示例集中在 `examples/`。
3. 输出分三类：本机确实运行的 Python 结果、由代码推导的 CUDA 预期输出、明确标注的人工测试数据；后两者都不是 GPU 实测成绩。
4. 首次 GPU 验证只要求结果正确、路径可解释。先保留默认算法和协议，不急着设置几十个环境变量。
5. 每次性能实验记录版本、GPU/NIC/CPU 拓扑、进程映射、完整命令与原始日志，不要只保存一张峰值截图。
6. 自测题答案就在章节末；综合练习先自行完成，再看第 15 章答案。

## 验证边界

教材的 API、参数与源码锚点以本地源码核对；交付检查包含 Markdown 本地链接、代码结构和 CPU 模拟测试。**CUDA 示例及多机实验没有在本次 macOS 环境编译或执行**，文中也不提供冒充实测的跑分。

拿到 Linux GPU 后，建议按 `00 → 单进程示例 → nccl-tests → MPI 示例` 的顺序验证，不要第一步就上大规模集群。上游仓库保持原样；教材和示例全部位于本目录。
