# NVML：GPU 管理、状态观测与设备识别

[返回教程主页](../../README.md) · [相关：MNNVL](mnnvl.md) · [相关：NVLS](nvls.md)

**NVML 全称 NVIDIA Management Library，是 NVIDIA 提供的 GPU 管理与监控库。** 它回答“这张 GPU 是谁、运行状态怎样、与其他设备怎样连接”，CUDA 则负责计算任务和相应资源的创建、提交与执行。

NVML 是运行在主机侧的软件接口，不是某一代 GPU 新增的硬件指令。本文说明其常用能力，以及 NCCL 为什么会使用它；源码基线为 NCCL 2.32.3（`12df1a11`）。

## 1. NVML 在软件栈中的位置

```text
计算程序                         监控、管理或拓扑发现代码
   ↓                                      ↓
CUDA Runtime / Driver API                NVML
   ↓                                      ↓
   └──────────── NVIDIA 驱动 ───────────────┘
                       ↓
                      GPU
```

NVML 提供 C 接口，也可以通过 Python 等语言的绑定调用。`nvidia-smi` 就建立在 NVML 之上，因此命令行工具和程序可以取得相近类别的设备信息。

NVML 运行库通常随驱动提供，Linux 上的库名是 `libnvidia-ml.so.1`。查询 GPU 状态不需要为了 NVML 安装完整 CUDA Toolkit；若要编译调用 NVML 的程序，还需要相应头文件和链接库。

NVML 查询围绕设备句柄展开，不要求调用者先设置 CUDA 当前设备或创建 CUDA 计算 context。NVML 自身的初始化仍可能触发驱动侧的设备初始化。

## 2. 它能查询和管理什么？

| 类别 | 典型能力 | 常见用途 |
|---|---|---|
| 设备身份 | GPU 名称、UUID、PCI bus ID | 对齐应用日志、驱动信息与实际设备 |
| 运行状态 | GPU 利用率、显存占用、温度、功耗、时钟 | 监控负载、温度和功耗限制 |
| 进程信息 | 使用 GPU 的进程及相关显存占用 | 定位资源使用者 |
| 互连和拓扑 | NVLink 状态、远端设备信息、P2P 能力等 | 辅助判断 GPU 之间的连接与访问关系 |
| 健康状态 | ECC 错误、部分硬件故障与 fabric 健康信息 | 故障诊断与运维 |
| 设备管理 | 功耗上限、支持的时钟配置、MIG 配置等 | 按平台策略管理 GPU 资源 |

这些是接口覆盖的能力类别，具体支持范围由 GPU、驱动和权限决定。查询程序需要处理“不支持”和“无权限”等返回状态，管理操作则应遵循所在机器的权限与运维策略。

NVML 主要提供设备级状态。若需要分析某个 kernel 的指令、访存瓶颈或执行时间，应继续使用 Nsight、CUPTI 等分析工具。

## 3. CUDA 编号、NVML 编号和 PCI bus ID

读 NCCL 初始化代码时，最容易混淆的是几种设备标识：

| 标识 | 表达的含义 |
|---|---|
| CUDA device ordinal | 当前进程可见 CUDA 设备列表中的编号，如 `0`、`1` |
| NVML device index | NVML 枚举设备时使用的编号 |
| PCI bus ID | 设备在本机 PCI/PCIe 系统中的地址，如 `0000:65:00.0` |
| GPU UUID | GPU 的身份标识，适合跨工具追踪设备 |
| `nvmlDevice_t` | NVML 返回的设备句柄，后续查询通过它指定目标设备 |

`CUDA_VISIBLE_DEVICES` 可以筛选和重排 CUDA 设备，因此 **CUDA device 0 不能直接当作 NVML device 0**。

对完整物理 GPU，一种常见的对应方式是：

```text
CUDA device ordinal
        ↓ cudaDeviceGetPCIBusId
PCI bus ID
        ↓ nvmlDeviceGetHandleByPciBusId_v2
NVML device handle
        ↓ nvmlDeviceGetIndex
NVML device index
```

NCCL 就沿这条关系找到 NVML 设备，并分别保存 `comm->cudaDev`、`comm->busId` 和 `comm->nvmlDev`。

如果启用了 MIG，同一物理 GPU 上的多个实例共享 PCI bus ID。此时 bus ID 用于定位父 GPU，区分 CUDA 逻辑实例还需要 MIG UUID，或父 GPU 身份与 GPU Instance / Compute Instance 标识。PCI 地址适合本机定位，也不应代替跨主机的完整设备身份。

## 4. 基本使用方式

一个查询程序通常按以下顺序组织：

```text
nvmlInit_v2
    ↓
按 UUID、PCI bus ID 或枚举结果取得 nvmlDevice_t
    ↓
调用状态查询接口
    ↓
nvmlShutdown
```

常见查询接口包括：

| 接口 | 作用 |
|---|---|
| `nvmlDeviceGetUUID` | 获取 GPU UUID |
| `nvmlDeviceGetMemoryInfo` | 查询显存容量与使用情况 |
| `nvmlDeviceGetUtilizationRates` | 查询设备利用率指标 |
| `nvmlDeviceGetTemperature` | 查询温度 |
| `nvmlDeviceGetPowerUsage` | 查询功耗 |

函数返回 `nvmlReturn_t`，查询结果一般通过输出参数返回。应先检查返回值，再读取输出；`nvmlErrorString` 可将错误码转换为可读信息。

如果只是想查看机器状态，可以先在 Linux NVIDIA GPU 环境运行只读命令：

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid --format=csv
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv
nvidia-smi topo -m
```

第一条核对设备身份，第二条查看运行状态，第三条观察拓扑关系。信息是否可用仍取决于具体平台。

## 5. NCCL 为什么需要 NVML？

NCCL 需要先了解设备与互连，才能构造通信路径。NVML 在这一过程中主要交付三类信息。

### 对齐设备身份

应用使用 CUDA 编号选卡，NCCL 通过 PCI bus ID 找到对应 NVML 设备。这样，后续查询的拓扑信息才能与正在创建 communicator 的 GPU 对应起来。

### 补充 P2P 与 NVLink 信息

NCCL 查询 GPU 对的 P2P 读写能力、NVLink 状态和远端设备信息，用于构建或核查拓扑。最终路径还要结合 CUDA 能力、系统 PCIe 信息、网络设备和用户配置，而不是仅凭一个 NVML 查询结果决定。

### 提供 GPU fabric 信息

MNNVL 初始化需要 GPU 的 `clusterUuid`、`cliqueId`、fabric 状态等信息。NVML 提供这些观测结果；NCCL 再结合成员交换、CUDA FABRIC handle 支持和实际导入、导出测试，判断跨节点 NVLink 能否使用。

```text
NVML 提供身份与状态
          ↓
NCCL 构建拓扑、检查可用能力
          ↓
Transport 准备连接与内存映射
          ↓
GPU kernel、proxy 和网络设备执行通信
```

**NVML 参与的是管理与能力发现，不负责替 NCCL 搬运数据或执行 AllReduce。** CollNet、MNNVL、NVLS 的运行路径可以利用这些发现结果，但各自承担不同的通信职责。

## 6. 源码与进一步阅读

| 阅读目标 | 入口 |
|---|---|
| Linux 动态加载 NVML | [src/os/linux.cc:628](../../../nccl/src/os/linux.cc#L628)，`ncclOsNvmlOpen` |
| CUDA 到 NVML 的设备对应 | [src/init.cc:580](../../../nccl/src/init.cc#L580) |
| GPU 与 NVLink 拓扑信息 | [src/graph/xml.cc:908](../../../nccl/src/graph/xml.cc#L908)，`ncclTopoGetXmlFromGpu` |
| 获取 GPU fabric 信息 | [src/init.cc:931](../../../nccl/src/init.cc#L931) |
| 消费 fabric 信息、检测 MNNVL | [src/mnnvl.cc:14](../../../nccl/src/mnnvl.cc#L14)，`ncclMnnvlCheck` |

官方参考：[NVML 概览](https://developer.nvidia.com/management-library-nvml)、[NVML 设备查询接口](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)、[MIG 设备标识](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/mig-device-names.html)。

可以先用设备查询命令对齐一张 GPU 的 CUDA 编号、PCI bus ID 和 NVML 身份，再沿上面的初始化入口阅读。本文在 macOS 上完成源码核对，未执行 GPU 查询命令。
