# 00 环境准备：从源码到第一次正确的 AllReduce

## 学习目标

- 分清 NVIDIA 驱动、CUDA Toolkit、NCCL 库、nccl-tests 各自的作用。
- 从当前源码构建，不误用系统里另一份 NCCL。
- 建立一个小而可靠的正确性基线。

## 1. 需要什么机器

| 条件 | 用途 |
|---|---|
| Linux + NVIDIA GPU | 本教程 GPU 实验的运行平台 |
| 与 GPU、Toolkit 相容的 NVIDIA 驱动 | CUDA 执行、设备内存管理及跨设备能力 |
| CUDA Toolkit，含 `nvcc`、头文件、运行库 | 编译 NCCL 和示例 |
| C/C++ 编译器、GNU make | 构建 host 代码 |
| Python 3，命令名为 `python3` | NCCL 构建时生成 device 代码；也用于 CPU 教学模拟 |
| 至少两张可用 GPU | 观察真正跨 GPU 的集合通信；单卡只能验证退化路径 |
| MPI，按需 | 启动多进程并分发 uniqueId；不是单进程 NCCL 的必需品 |
| 配置正确的 NIC/网络，按需 | 多节点数据传输 |

NCCL 不是 GPU 驱动，nccl-tests 也不是 NCCL 库。MPI 可以负责启动进程和引导信息交换，之后大块 GPU 数据由 NCCL 选择的 transport 搬运，并非必然经过 MPI。

当前教材编写机器是 macOS arm64，没有 `nvcc`；下面命令需到 Linux GPU 机器运行。可以保持三个目录相邻，再按本章操作。

## 2. 先看环境，不先装一堆包

以下均从项目根目录执行：

```bash
export ROOT="$PWD"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

nvidia-smi
nvidia-smi topo -m
nvcc --version
c++ --version
make --version
python3 --version
git -C nccl rev-parse HEAD
git -C nccl-tests rev-parse HEAD
```

检查点：

1. `nvidia-smi` 能看到预计使用的 GPU，且无其他重要任务被本实验干扰。
2. `nvidia-smi` 显示的 CUDA Version 表示驱动支持能力，不等于实际安装的 Toolkit 版本；编译器看 `nvcc --version`。
3. 记录 GPU 型号、显存、GPU 间拓扑。两张 PCIe 卡与 NVLink 相连的两张卡，不能套用同一个预期带宽。
4. 当前 shell 的 `CUDA_VISIBLE_DEVICES` 是否限制或重排了设备？NCCL 使用的是进程可见的 CUDA ordinal。

不要把“源码某处分支能编译旧 CUDA”理解为所有新特性都支持旧工具链。先用目标 GPU 支持、且与驱动兼容的工具链建立基线；新能力的要求在第 13 章单独讨论。

## 3. 构建本地 NCCL

```bash
make -C nccl -j 8 src.build CUDA_HOME="$CUDA_HOME"
ls nccl/build/include/nccl.h nccl/build/lib/libnccl.so*
```

这里的 `-j 8` 是编译并发数，不是 GPU 数。编译可能占用较多内存，机器资源不足时降低并发。

产物的关系是：

```text
nccl/src/nccl.h.in + 版本信息 -> build/include/nccl.h
host/device 源码            -> build/lib/libnccl.so / 静态库
下游应用                    -> 使用生成的头文件并链接库
```

不要直接把 `src/nccl.h.in` 当作可用的安装头文件，它含待替换的版本变量。

### 可选：只编译目标架构

本地 README 给出了 Hopper `sm_90` 的例子：

```bash
make -C nccl -j 8 src.build CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
```

**仅在你的目标确为对应架构时使用。** A100、H100、更新 GPU 的 compute capability 不相同；不合适的目标会导致无法运行，不能为了编译快盲目照抄。默认架构集合由 `nccl/makefiles/common.mk:63` 起的 CUDA 版本判断产生。

调试构建可另用 `DEBUG=1` 和独立 `BUILDDIR`，但优化等级不同，不能拿它代表发布构建性能。先保持 release 默认。

## 4. 构建 nccl-tests，并固定运行时库

```bash
export NCCL_HOME="$ROOT/nccl/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

make -C nccl-tests -j 8 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
ldd nccl-tests/build/all_reduce_perf
```

`NCCL_HOME` 使编译器/链接器找到头文件和库；**不等于动态加载器运行时一定选择那份库**。在 `ldd` 输出中确认 `libnccl.so` 指向预期的 `nccl/build/lib`，同时检查没有 `not found`。

为什么会发生“我改了 NCCL 源码，测试表现却没变”？典型原因就是编译链接用了一份，运行时加载另一份。Python 框架还可能携带自己的 NCCL，修改 shell 的路径并不能未经验证就推断框架也已切换。

本教程不需要把库安装到 `/usr`，不需要 `sudo make install`，也不需要覆盖系统/框架自带的 NCCL。

## 5. 第一次只跑小规模

确保至少有两张闲置且可访问的 GPU：

```bash
NCCL_DEBUG=INFO \
  ./nccl-tests/build/all_reduce_perf -b 8 -e 16M -f 2 -g 2 -w 5 -n 20 -c 1
```

含义：单进程、默认一个线程、每线程两张 GPU；从 8 字节扫到 16 MiB，每次翻倍，预热 5 次、计时 20 次、额外正确性校验 1 轮。

先确认四件事：

- 程序实际使用了预期两张 GPU。
- 正常输出全部尺寸而不是卡在初始化。
- `#wrong` 为 0；关闭检查后的 `N/A` 不等于“已验证正确”。
- 进程正常结束。

首次不要直接扫到几十 GiB。测试还有接收、发送、校验及 NCCL 内部缓冲，`-e` 不是整进程显存峰值。先用小尺寸验证，再逐步增大。

`NCCL_DEBUG=INFO` 用于核对路径，正式稳态测量可以降低日志级别并保持所有对照实验一致。带宽解释见[第 10 章](10-nccl-tests.md)。

## 6. 编译并运行自己的最小程序

```bash
make -C nccl-tutorial/examples CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
./nccl-tutorial/examples/build/single_process_allreduce 2
```

程序令 rank `r` 的每个元素等于 `r+1`，两卡求和应得到 3，四卡应得到 10。它验证每个元素，而不是只打印第一个元素就宣布正确。

详见[示例说明](examples/README.md)。它没有计时优化，不应拿来替代 nccl-tests 做吞吐结论。

## 7. 多进程与多机稍后再加

MPI 构建和启动见[第 10 章](10-nccl-tests.md)与[示例说明](examples/README.md)。正确顺序是：

```text
单卡退化正确 -> 单进程两卡 -> 单机多进程 -> 两机各一卡 -> 两机多卡
```

每增加一层就多一个潜在问题：进程映射、引导网络、RDMA、GPU/NIC 亲和性。这样能明确“从哪一层开始坏”，而不是把所有问题揉成一次集群挂起。

## 8. 记录模板

每次实验至少保存以下内容；文本记录即可，不要求新工具：

```text
日期 / hostname / GPU 型号与数量 / CPU 与 NUMA
NCCL commit / nccl-tests commit / 驱动 / nvcc / MPI
CUDA_VISIBLE_DEVICES / 进程×线程×GPU 映射
完整命令 / NCCL_* 环境变量 / ldd 结果
topo -m / 日志 / size-time-algbw-busbw-#wrong
是否独占设备 / 预热 / 重复次数 / 异常现象
```

## 源码入口

- [NCCL 构建说明](../nccl/README.md)：`nccl/README.md:15`，`src.build`、`BUILDDIR`。
- [版本信息](../nccl/makefiles/version.mk)：`nccl/makefiles/version.mk:9`。
- [编译配置](../nccl/makefiles/common.mk)：`nccl/makefiles/common.mk:8`，CUDA 路径；`:63`，架构集合。
- [测试构建](../nccl-tests/src/Makefile)：`nccl-tests/src/Makefile:24`，NCCL 头文件/库；`:29`，MPI 支持。

## 自测

1. `nvidia-smi` 写 CUDA 13.x，`nvcc` 写 12.x，是一定装错了吗？
2. `NCCL_HOME` 正确，为什么还需要查 `ldd`？
3. 单卡测试通过，是否已证明 NVLink 或 RDMA 正常？

**答案：** 1. 不一定，两者含义不同，需检查具体兼容性。2. 构建时库搜索与运行时动态加载是两件事。3. 没有，单卡可能完全不经过跨设备链路。

下一章：[建立全局模型](01-mental-model.md)。
