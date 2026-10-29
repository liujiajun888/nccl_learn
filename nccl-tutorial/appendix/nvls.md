# NVLS：利用 NVLink 的 multicast 与硬件归约

[返回教程主页](../README.md) · [相关：CollNet](collnet.md) · [相关：MNNVL](mnnvl.md)

**NVLS 全称 NVLink SHARP，是 NCCL 利用 NVLink/NVSwitch 的 multicast 与归约能力加速集合通信的机制。** GPU 线程通过特殊的内存访问指令，把一部分原本需要逐个接收、累加、发送的工作交给互连硬件完成。

本文以同一 NVLink 域内、四个 rank 的 FP32 Sum AllReduce 为例，源码基线为 NCCL 2.32.3（`12df1a11`）。先理解普通中间缓冲路径，再看用户缓冲注册和跨域扩展。

## 1. NVLS 加速的是哪部分工作？

软件归约中，负责某个分片的 GPU 需要接收其他 GPU 的贡献，在 GPU 上合并数据，再分发结果。NVLS 使用两类关键操作：

| 操作 | 语义 |
|---|---|
| `multimem.ld_reduce` | 从一个 multimem 地址所关联的多份显存读取数据，在硬件中归约，将结果返回发起线程 |
| `multimem.st` | 将一个值写入该 multimem 地址所关联的多份显存，实现 multicast 分发 |

例如，四张 GPU 在对应位置分别准备 `1、2、3、4`，负责该位置的线程执行 Sum 归约读取，可以得到 `10`；再将 `10` multicast 写回输出区域，各 GPU 就能取得同样的结果。

归约读取与结果广播是两种操作，普通 `multimem.st` 本身不做求和。GPU 仍负责发指令、组织分片、搬运本地数据和推进同步，所以 NVLS 加速通信，并不意味着完全不占用 SM。

## 2. UC、MC 与 multimem 地址是什么？

理解 NVLS 的关键是分清两种映射：

- **UC（unicast）地址**：普通单播映射，在本文路径中用于访问当前 GPU 的本地显存。
- **MC（multicast）地址**：映射到一个 multicast 对象，该对象绑定了参与 GPU 各自的显存区域。

同一 MC 偏移对应组内多张 GPU 的显存位置，GPU 用相应的 multimem 指令对这些位置执行归约读取或组播写入。

```text
GPU 0 本地显存 ─┐
GPU 1 本地显存 ─┼── 绑定到同一 multicast 对象
GPU 2 本地显存 ─┤               ↑
GPU 3 本地显存 ─┘          MC 地址映射
```

数据仍保存在各张 GPU 的显存里，MC 地址不是一块交换机显存，也不要求各进程使用相同的虚拟地址数值。

NCCL 的 host 侧负责创建 multicast 对象、让参与设备加入、绑定显存并设置访问权限；device 侧使用准备好的地址。中间缓冲分别安排归约输入区和广播输出区，数据方向为：

```text
输入：各 GPU 写自己的 UC 区域 → 负责线程通过 MC 归约读取
输出：负责线程向输出 MC 写入 → 各 GPU 从自己的 UC 区域读取
```

## 3. 四个 rank 的 AllReduce 怎样执行？

假设四张 GPU 都作为 head，每个 head 负责数组的四分之一。这里的 head 是负责发起某组分片操作的 GPU rank。

### 第一步：各 GPU 准备自己的分片

每个 rank 将输入按分片写入自己的 NVLS 本地缓冲，源码把这类工作称为 Scatter。

```text
GPU 0：x0[0]  x0[1]  x0[2]  x0[3]
GPU 1：x1[0]  x1[1]  x1[2]  x1[3]
GPU 2：x2[0]  x2[1]  x2[2]  x2[3]
GPU 3：x3[0]  x3[1]  x3[2]  x3[3]
```

`xg[j]` 表示 GPU g 的第 j 个分片。贡献先放进各自的 UC 区域，**不需要先通过普通 P2P 把它们都集中到 head 的显存中**。

### 第二步：各 head 发起硬件归约

head j 对相应 MC 输入区域发起归约读取，获得：

```text
y[j] = x0[j] + x1[j] + x2[j] + x3[j]
```

多个 head 同时负责不同分片，NVLink/NVSwitch 的归约硬件完成成员数据的合并，结果返回发起操作的 GPU 线程。

### 第三步：各 head 广播结果

head j 把 `y[j]` 写入对应的 MC 输出区域，硬件将它分发到每个成员 GPU 的输出缓冲。

### 第四步：各 GPU 收齐完整结果

各 GPU 从本地输出 UC 区域取出 `y[0]、y[1]、y[2]、y[3]`，写入自己的用户接收缓冲，源码把这类工作称为 Gather。

整体关系是：

```text
各 GPU 输入
    ↓ 本地 Scatter
各 GPU 的 UC 输入区
    ↓ MC 归约读取
各 head 得到负责的结果分片
    ↓ MC 组播写入
各 GPU 的 UC 输出区
    ↓ 本地 Gather
各 GPU 得到完整 AllReduce 结果
```

这四步是逻辑分工，不是四次独立 kernel 启动。单域设备实现把归约和广播合在同一类线程工作中，通过 `directRecvDirectSend` 衔接，Scatter、Gather 与各 chunk 的处理可以流水重叠。

## 4. 用户缓冲注册改变什么？

上面的模型使用 NCCL 中间缓冲，便于看清数据交接。若所选路径支持并成功完成 NVLS 用户缓冲注册，NCCL 可以直接建立相应的用户内存映射，减少输入 Scatter 和输出 Gather 的数据复制；协议同步仍然保留。

不同路径的注册能力需分别看待。在本基线中，MNNVL 场景关闭传统 host NVLS 用户缓冲注册路径，但对称窗口与 symmetric multimem 是另一套能力和实现。由此不能概括为“MNNVL 完全不支持任何形式的内存注册”。本篇主线采用中间缓冲，不依赖这项优化。

## 5. 单域 NVLS、NVLSTree 与 CollNet

NVLS 在 NCCL 中既用于描述底层能力和 transport，也出现在算法名称中。分析一次集合调用时，要继续看域间部分怎样衔接。

| 场景或算法 | 域内部分 | 域间部分 |
|---|---|---|
| 单 NVLink 域的 NVLS | multicast/multimem 归约和分发 | 不需要额外的域间通信 |
| NVLSTree | 使用 NVLS 汇聚、分发 | 各域 head 组成点对点树，继续归约与传播 |
| 本基线中结合 CollNet 的多域 NVLS 路径 | 使用 NVLS 汇聚、分发 | 通过集合网络逻辑端点调用 CollNet 后端 |

NVLSTree 的跨域树仍有 GPU 参与数据归约；结合 CollNet 的路径则把域间集合操作交给插件。不能把所有 NVLS 域外通信都叫作 CollNet。

这里的“域”是支持相应 NVLink 能力的一组 GPU，不必等于一台物理服务器。**MNNVL 扩展域的物理覆盖范围，NVLS 在具备相应硬件和映射支持的域内提供集合加速。** 因而多台物理主机也可能落在同一个 NVLS 处理域内。

## 6. 支持条件与实际收益

本基线相关设备代码以 CUDA 12.1、SM90 为基础门槛，还需要支持 multicast/归约的 GPU、NVLink/NVSwitch 系统以及配套驱动。自动检测会查询 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`，仅有一张 Hopper GPU 或仅升级 Toolkit 并不足够。

NCCL 还会根据 collective、数据类型、归约操作、拓扑和消息大小筛选算法。NVLS 可以减少 GPU 软件归约与重复点对点分发的工作量；其收益也受缓冲搬运、同步开销和可用带宽影响，并非所有消息都更快。

核对时可以先看 NVLS 能力与资源初始化日志，再看具体任务的算法选择。做性能对照时，应保持 collective、数据大小、GPU 数和缓冲使用方式一致，并同时记录耗时与 SM 占用，而不只看“NVLS 已启用”。

本文的数值与流程用于解释机制，未在支持 NVLS 的 GPU 系统上实测。

## 7. 源码入口

| 阅读目标 | 入口 |
|---|---|
| multicast 能力检测 | [src/transport/nvls.cc:166](../../nccl/src/transport/nvls.cc#L166)，`ncclNvlsInit` |
| 分配本地 UC 显存并绑定 MC 对象 | [src/transport/nvls.cc:224](../../nccl/src/transport/nvls.cc#L224)，`nvlsAllocBindUc` |
| UC/MC 输入输出连接 | [src/transport/nvls.cc:324](../../nccl/src/transport/nvls.cc#L324) |
| 单域与多域 NVLS AllReduce | [src/device/all_reduce.h:390](../../nccl/src/device/all_reduce.h#L390) |
| head、网络端点与树连接 | [src/graph/connect.cc:248](../../nccl/src/graph/connect.cc#L248)，`connectNvls` |
| Sum 的 multimem 归约读取 | [src/device/reduce_kernel.h:1126](../../nccl/src/device/reduce_kernel.h#L1126) |
| multimem 组播存储 | [src/device/op128.h:414](../../nccl/src/device/op128.h#L414) |
| MNNVL 与传统 NVLS 注册的边界 | [src/init.cc:1353](../../nccl/src/init.cc#L1353) |

继续阅读：[CollNet](collnet.md) 解释集合网络后端；[MNNVL](mnnvl.md) 解释跨物理节点的 fabric 和显存映射。
