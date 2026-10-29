# 附录阅读指南：按问题查阅，不要顺序通读

`appendix/` 是主线之后的参考资料库，不是第二套需要顺序学习的教材。先读 [教程首页](../README.md) 中的 `00–11` 主线，再按当前问题选择对应附录。

**不建议删除这些文档来“减负”。**更合理的使用方式是：通用参考随用随查，进阶原文按主题深入，现代特性按具体硬件和接口选择。

## 1. 先确定自己属于哪条阅读路径

| 当前目标 | 推荐路径 |
|---|---|
| 只想理解整体架构 | 读主线即可；附录只看 [软件栈全景](software-stack.md) 作目录地图 |
| 构建并运行 NCCL | [环境准备](environment.md) → [示例](../examples/README.md) → [nccl-tests](nccl-tests.md) |
| 深入 host/device 执行 | 读对应主线章，再读该章的进阶原文，不要把所有进阶原文连读 |
| 深入 communicator 初始化 | [05 主线](../05-communicator.md) → [06 拓扑](../06-topology.md) → [初始化精读](nccl-comm-init-rank.md) 的第 1、13、14、16 节 |
| 深入网络与协议 | [07 Transport](../07-transports.md) → [08 Proxy](../08-proxy-and-network.md) → 对应进阶原文 |
| 做性能或开发实验 | [nccl-tests](nccl-tests.md) → [性能分析](performance.md) → [开发工作流](development-workflow.md) → [测试与调试](testing-and-debugging.md) |
| 研究新硬件或新接口 | 按主题读 [现代特性](modern-features.md)，再查 NVLS、CollNet、MNNVL 或 NVML 专篇 |

## 2. 通用参考：有独立使用价值，按需阅读

| 文档 | 是否有必要保留 | 适合什么时候读、怎样读 |
|---|---|---|
| [环境准备](environment.md) | 构建和运行基线所必需 | 需要编译本地 NCCL、核对加载库和运行示例时使用。偏命令操作，不必作为理论章学习 |
| [软件栈全景](software-stack.md) | 分层职责与 `src/` 目录地图有用 | 初读主线后回查，不必逐节精读。适合回答“该去哪个目录找证据” |
| [集合接口与布局](collectives.md) | count、结果归属和原地关系易查错，值得保留 | 写调用或核对 buffer 大小时查阅，属于 API 速查，不要求连续学习 |
| [nccl-tests](nccl-tests.md) | 运行测试、理解输出和计时口径有用 | 跑性能或正确性实验前阅读。先固定 64 KiB 用例，再扩展到 MPI 和扫描 |
| [性能分析](performance.md) | 性能模型、对照实验和 overlap 分析有价值 | 先理解 nccl-tests，再做同参数 baseline/profile。公式和 profiler 命令都属于进阶内容 |

这些文档的共同点是：**可以独立解决问题，但不要求先读其他进阶原文。**

## 3. 主线进阶原文：有保留价值，但不适合初学者通读

这些文件保留了主线章的详细实现，适合在主线已经建立流程后查具体机制。它们不是“多余重复”，但也**不应该成为初学者的第二条长阅读队列**。

| 文档 | 保留价值 | 推荐阅读方式 |
|---|---|---|
| [Host 执行进阶](host-execution.md) | planner 队列、channel 调度、work FIFO、proxy 和回收细节 | 先读 [02 Host 主线](../02-host-execution.md)，再按 task→plan→launch 查具体字段 |
| [Device 执行进阶](device-execution.md) | CTA 角色、chunk/slice/step、LL/LL128、FIFO 可见性 | 先读 [03 Device 主线](../03-device-execution.md)，理解 runRing 后再看协议，不要从协议细节倒推整体架构 |
| [集合算法进阶](algorithms.md) | 完整 Ring 手算、带宽系数、双树与候选组合 | 先读 [04 算法主线](../04-algorithms.md)，用具体块验证后再看公式和 Tree 实现 |
| [Communicator 进阶](communicator.md) | 初始化 API、阶段产物和 runtimeConn 的完整概览 | 先读 [05 Communicator 主线](../05-communicator.md)，用于补细节，不建议与初始化精读同时通读 |
| [Topology 进阶](topology.md) | XML、路径类型、PXN、rail/crossNic 和图搜索 | 先读 [06 Topology 主线](../06-topology.md)，按路径模型或邻接问题查，不必研究所有搜索分支 |
| [Transport 进阶](transports.md) | connector、setup/connect、P2P/SHM/NET、GDR 与映射 | 先读 [07 Transport 主线](../07-transports.md)，按具体 transport 查建连过程 |
| [Proxy 与网络进阶](proxy-and-network.md) | proxy op、args/sub、request 状态机和网络完成语义 | 先读 [08 Proxy 主线](../08-proxy-and-network.md)，适合排查“网络完成但 GPU 仍等待”的问题 |
| [内存与生命周期进阶](memory-and-lifetime.md) | 注册缓存、FIFO、Graph 引用和 Finalize/Destroy 协调 | 先读 [09 生命周期主线](../09-memory-and-lifetime.md)，按对象类型和最后使用者查回收条件 |
| [开发工作流进阶](development-workflow.md) | 插桩位置、构建/加载核验和回归范围完整可操作 | 先完成 [10 开发主线](../10-development-workflow.md) 的小实验，再按命令复现，不必先背所有字段 |
| [测试与调试进阶](testing-and-debugging.md) | baseline/observed 脚本、日志管理和完整回归矩阵 | 先读 [11 调试主线](../11-testing-and-debugging.md)，用于实际复现和扩展验证，不适合只作理论阅读 |

## 4. 初始化长篇精读：必要，但必须分段使用

[ncclCommInitRank 源码精读](nccl-comm-init-rank.md) 是普通新建 communicator 的完整初始化参考，内容有价值，但篇幅很长，**不适合从头连续读到尾**。

| 阅读目标 | 建议章节 |
|---|---|
| 只想复习整体初始化流程 | 第 1、13、14、16 节 |
| 已读 Communicator 主线，只补图搜索到 device 状态 | 直接读第 14 节，再回看第 16 节 |
| 重点研究 bootstrap、peerInfo、拓扑与 paths | 第 8–13 节 |
| 需要完整理解普通初始化链路 | 按 1→16 分段阅读，每次只追一个阶段，并与 [05](../05-communicator.md)、[06](../06-topology.md)、[07](../07-transports.md) 对照 |

不要把它与 [Communicator 进阶](communicator.md) 同时顺序读完：前者用于完整深读，后者用于快速回顾。

## 5. 现代特性与专用硬件：按主题选读，不是通用必读

| 文档 | 保留价值与阅读边界 |
|---|---|
| [现代特性](modern-features.md) | 覆盖 Graph、对称窗口、NVLS、CollNet、MNNVL、Device API、GIN、RMA、插件和成员变化。主题多，**必须按节选读，不能当作单一特性连续学习** |
| [NVLS](nvls.md) | 需要理解 multicast/multimem、UC/MC 地址和硬件归约时阅读。适合有对应 NVLink/NVSwitch 平台的读者 |
| [CollNet](collnet.md) | 需要理解集合网络插件、Direct/Chain 和 head 数据流时阅读。不是普通 NET 的入门替代品 |
| [MNNVL](mnnvl.md) | 需要理解跨节点 NVLink、FABRIC handle 和远端显存映射时阅读。没有相关硬件时只作概念参考 |
| [NVML](nvml.md) | 需要设备身份、NVLink/P2P 状态和 fabric 信息时阅读。适合与拓扑和 MNNVL 排查配套使用 |

## 6. 最终使用建议

- **只学习架构：**主线足够，附录不需要全部阅读。
- **需要继续深入：**先读对应主线，再读同名进阶原文，避免同时打开多份长文。
- **做实验：**环境、示例和 nccl-tests 优先于性能公式；先得到正确性结果，再分析性能。
- **研究特殊硬件：**先确认平台确实支持，再读 NVLS、CollNet、MNNVL 和现代特性，避免把特性名称当作实际可用能力。
- **遇到长篇：**使用目录和分段入口，目标是回答一个具体问题，而不是完成“读完”本身。
