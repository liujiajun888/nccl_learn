# 00 环境准备：从源码到第一次正确的 AllReduce

> 基线：NCCL 2.32.3（`12df1a11`），nccl-tests 2.20.0（`b4d5bee`）。

## 学习目标

- 分清 NVIDIA 驱动、CUDA Toolkit、NCCL 库、nccl-tests 各自的作用。
- 从当前源码构建，不误用系统里另一份 NCCL。
- 建立一个小而可靠的正确性基线。

本章是推荐机制路线的**环境与基线速查**：确认加载的 NCCL 正确、两卡结果正确，再带着这个基线研究内部实现。熟悉 CUDA/UMD 的读者可直接核对第 2 节的源码与工具版本、第 3～4 节的构建/加载路径和第 5 节的固定尺寸校验；shell 参数逐项解释保留作回查，不必逐句重学。

```text
本地 NCCL 源码 ──CUDA Toolkit 编译──> 头文件 + libnccl.so
                                              │ 链接并在运行时加载
nccl-tests 源码 ───────编译─────────> all_reduce_perf
                                              │ 驱动执行 GPU 工作
                    GPU 0 输入 A ──┐          │
                                   ├─ AllReduce(sum) ─> 两卡各得到 A+B
                    GPU 1 输入 B ──┘                    │
                                                   逐元素校验
```

这是数据流示意，不要求你先理解通信算法。先读第 1～5 节跑通固定尺寸；第 6 节再换成自己的最小程序，源码入口放在章末回查。

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

当前教材编写机器是 macOS arm64，没有 CUDA／`nvcc`，未进行 GPU 构建或实测。下面的 Bash 命令需到 **Linux NVIDIA GPU 机器**运行，不是在 Mac 上安装几个包就能完成。

## 2. 先看环境，不先装一堆包

### 第一步：确认根目录与本地源码

在你的工作目录中，保持三份素材相邻；根目录结构如下（只列本章相关项）：

```text
nccl_learn/
├── .gitignore
├── nccl/             # 本地 NCCL 源码，自带 Git 历史
├── nccl-tests/       # 本地测试源码，自带 Git 历史
└── nccl-tutorial/    # 教程及 examples/
```

外层 `.gitignore` 已忽略 `nccl/`、`nccl-tests/`，所以**拿到教程仓库不等于拿到了这两份源码**。已有本地源码可继续使用；迁移到 Linux 机器后，也须在根目录下准备好对应版本的源码及其 Git 元数据。本章只使用已经准备好的本地目录。

**本章命令约定：** 在 Linux 的 Bash 中，当前目录已经是该机器的 `nccl_learn` 根目录。后续小节沿用同一个 Bash 会话及这里设置的变量；新开终端时先回到根目录并重新设置。

```bash
# Linux/bash；当前目录为本机 nccl_learn 根目录。
export ROOT="$PWD"
ls
ls nccl/Makefile nccl-tests/Makefile nccl-tutorial/00-environment.md
git -C nccl rev-parse HEAD
git -C nccl-tests rev-parse HEAD
```

把刚才的命令拆开看：

- `$PWD` 是 Bash 保存的当前目录；`ROOT="$PWD"` 把它记下来。双引号让带空格的路径仍作为一个参数。
- `export` 让之后启动的子进程也能继承这个变量；它只影响当前会话及其子进程，不会改系统配置。
- **`git -C nccl rev-parse HEAD`**：`git` 是程序；`-C nccl` 让这一次 Git 操作在根目录下的 `nccl` 中执行，不改变你所在的 shell 目录；`rev-parse` 把版本名称解析成标识；`HEAD` 指当前检出的提交。整条命令只读取提交号，不下载、编译或切换版本。
- 两条 Git 命令通常各输出一行完整的 40 位提交哈希；本教材对应前缀分别是 `12df1a11`、`b4d5bee`。哈希用于定位源码快照，不是 GPU 型号，也不是 `2.32.3` 这样的发布版本号；未提交的本地修改也不会改变 `HEAD`。

**检查点 A：** `ls` 能找到三个目录及 Makefile，提交号与基线相符。若出现 `No such file or directory`，先检查位置和源码是否准备好；若出现 `not a git repository`，检查该源码的 Git 元数据。提交不符时先确认素材版本，不继续拿本章行号硬对，也不盲目切换或覆盖源码。

### 第二步：检查机器和编译工具

`CUDA_HOME` 是 Toolkit 安装前缀，下面假定为 `/usr/local/cuda`；若本站不同，只把它改为真实路径。`PATH` 是查找可执行程序的目录列表，把 CUDA 的 `bin` 放在前面，才能让 `nvcc` 优先指向这份 Toolkit；末尾保留原来的 `$PATH`，其他命令仍能找到。

```bash
# Linux/bash；沿用根目录会话。
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

nvidia-smi
nvidia-smi topo -m
nvcc --version
c++ --version
make --version
python3 --version
```

**检查点 B：**

1. `nvidia-smi` 能看到预计使用的 GPU，且无其他重要任务被本实验干扰。
2. `nvidia-smi` 显示的 CUDA Version 表示驱动支持能力，不等于实际安装的 Toolkit 版本；编译器看 `nvcc --version`。
3. 记录 GPU 型号、显存、GPU 间拓扑。两张 PCIe 卡与 NVLink 相连的两张卡，不能套用同一个预期带宽。
4. 当前 shell 的 `CUDA_VISIBLE_DEVICES` 是否限制或重排了设备？NCCL 使用的是进程可见的 CUDA ordinal（从 0 开始的设备编号），不要覆盖调度器授予的设备映射。

若 `nvidia-smi` 失败或没有获准使用的 GPU，停在机器／驱动检查；若 `nvcc: command not found`，停在 Toolkit 路径检查；若 `c++`、`make` 或 `python3` 缺失，先补齐本站构建环境再继续。后面的编译命令不能修复这些前置问题。

不要把“源码某处分支能编译旧 CUDA”理解为所有新特性都支持旧工具链。先用目标 GPU 支持、且与驱动兼容的工具链建立基线；新能力的要求在第 13 章单独讨论。

## 3. 构建本地 NCCL

现在只做一件事：把源码变成供测试程序使用的头文件和库。

```bash
# Linux/bash；从根目录执行，沿用第 2 节的 CUDA_HOME。
make -C nccl -j 8 src.build CUDA_HOME="$CUDA_HOME"
ls nccl/build/include/nccl.h nccl/build/lib/libnccl.so*
```

按参数拆读，不需要先背 Makefile：

| 片段 | 意义 |
|---|---|
| `make` | 按 Makefile 的规则构建 |
| `-C nccl` | 让 make 到 `nccl` 目录读取规则，作用类似刚才 Git 的 `-C` |
| `-j 8` | 最多并行运行 8 个构建任务，不是 GPU 数；内存不足时降低并发 |
| `src.build` | NCCL Makefile 中的目标名，不是文件或目录名；它转到 `src` 执行 `build`，构建库而非安装到系统 |
| `CUDA_HOME="$CUDA_HOME"` | 把当前 Bash 变量的值作为本次 make 的配置传入 |
| `ls ...` | 查看生成的头文件和共享库；`*` 匹配不同版本后缀 |

**检查点 C：** make 成功结束，且 `ls` 能列出 `nccl.h` 与 `libnccl.so*`。若编译报错或产物缺失，就停在这里看第一个错误；不要继续构建 tests，否则容易把“缺少产物”误当成测试程序的问题。

产物的关系是：

```text
nccl/src/nccl.h.in + 版本信息 -> build/include/nccl.h
host/device 源码            -> build/lib/libnccl.so / 静态库
下游应用                    -> 使用生成的头文件并链接库
```

不要直接把 `src/nccl.h.in` 当作可用的安装头文件，它含待替换的版本变量。

<details>
<summary>选读：目标架构与调试构建；首轮保持默认，直接进入第 4 节</summary>

### 可选：只编译目标架构

本地 README 给出了 Hopper `sm_90` 的例子：

```bash
# Linux/bash；从根目录执行，沿用第 2 节的 CUDA_HOME；仅适用于对应 GPU。
make -C nccl -j 8 src.build CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
```

**仅在你的目标确为对应架构时使用。** A100、H100、更新 GPU 的 compute capability 不相同；不合适的目标会导致无法运行，不能为了编译快盲目照抄。默认架构集合由 `nccl/makefiles/common.mk:63` 起的 CUDA 版本判断产生。

调试构建可另用 `DEBUG=1` 和独立 `BUILDDIR`，但优化等级不同，不能拿它代表发布构建性能。先保持 release 默认。

</details>

## 4. 构建 nccl-tests，并固定运行时库

```bash
# Linux/bash；从根目录执行，沿用 ROOT、CUDA_HOME。
export NCCL_HOME="$ROOT/nccl/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

make -C nccl-tests -j 8 MPI=0 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
ldd nccl-tests/build/all_reduce_perf
```

这组命令分三步读：先指定 NCCL 产物位置，再构建单进程版 tests（`MPI=0`），最后用 Linux 的 `ldd` 查看可执行文件会加载哪些共享库。

- `NCCL_HOME` 使编译器／链接器找到头文件和库；**不等于动态加载器运行时一定选择那份库**。
- `LD_LIBRARY_PATH` 是运行时共享库搜索路径，以冒号分隔；这里把 NCCL 的 `lib` 和 CUDA 的 `lib64` 放在前面。
- `${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}` 表示原变量非空时才追加“冒号 + 原值”；为空时不追加，避免产生空路径项。它与前面的 `PATH` 不同：一个找库，一个找命令。

**检查点 D：** make 成功，`ldd` 中 `libnccl.so` 指向预期的 `nccl/build/lib`，且没有 `not found`。若库路径指向别处或缺失，先修正本会话中的路径再重查，不运行测试。若改过 MPI 构建方式，按第 10 章用独立产物目录隔离，不混用旧对象。

为什么会发生“我改了 NCCL 源码，测试表现却没变”？典型原因就是编译链接用了一份，运行时加载另一份。Python 框架还可能携带自己的 NCCL，修改 shell 的路径并不能未经验证就推断框架也已切换。

本教程不需要把库安装到 `/usr`，不需要 `sudo make install`，也不需要覆盖系统／框架自带的 NCCL。

## 5. 第一次只跑小规模

确保当前作业已获准使用至少两张闲置 GPU。**先固定 64 KiB，不扫尺寸**，这样一旦失败，只需围绕同一个用例排查。

```bash
# Linux/bash；从根目录执行，沿用 ROOT、CUDA_HOME、NCCL_HOME。
NCCL_DEBUG=INFO \
LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 64K -e 64K -g 2 -t 1 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

把命令分成四组就够了：

| 参数组 | 本轮要做的事 |
|---|---|
| `-b 64K -e 64K` | 最小、最大尺寸相等，只测 65,536 字节；K 是 1024 倍 |
| `-g 2 -t 1` | 单进程、一个 CPU 工作线程管理两张 GPU，NCCL rank 数 P=2 |
| `-d float -o sum` | 每卡 16,384 个 FP32 元素，逐元素求和，两卡都得到完整结果 |
| `-w 5 -n 20 -c 1 -a 3` | 每种缓冲模式预热 5 次、计时 20 次，再在计时外校验 1 轮；按参与进程／线程的最大平均耗时报告 |

行首的 `变量=值` 只给这一次命令设置环境，行末 `\` 表示命令下一行继续。首轮用 `NCCL_DEBUG=INFO` 核对设备和通信路径，不要被日志长度吓到，先找结果行与校验摘要。

**检查点 E：**

- 设备报告确实使用了预期的两张 GPU；否则停下检查作业分配和可见设备，不改成未经授权的编号。
- 出现 `size=65536`、`count=16384` 的结果行；左右两组是 out-of-place／in-place，**不是两张卡各一组**。
- 两组 `#wrong` 都为 **0**，检查摘要无失败，进程正常结束，才算本用例通过；`N/A` 是未报告校验，不能当作 0。
- 若卡在初始化、报 CUDA／NCCL 错误或出现非零 `#wrong`，停在这个固定用例，保留完整命令与日志排查，不继续扩大尺寸或解读带宽。

首轮通过后先进入第 6 节，读懂自己的完整程序，再到[第 10 章](10-nccl-tests.md)解释测试输出。不用现在就扫遍所有尺寸。

<details>
<summary>选读：读懂一行测试输出后，再扫描尺寸</summary>

看完第 10 章第 3 节、学会读一行和手算带宽后，可执行同条件的扫描：

```bash
# Linux/bash；从根目录执行；仅在固定 64 KiB 两卡校验通过后运行。
NCCL_DEBUG=INFO \
LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/nccl-tests/build/all_reduce_perf" \
  -b 8 -e 16M -f 2 -g 2 -t 1 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

这里只把尺寸改为从 8 字节到 16 MiB、每次翻倍；其余条件不变。确认全部尺寸输出且两组校验均为 0。小消息通常受固定启动成本影响，大消息更容易摊薄成本；趋势的逐步解释在第 10 章。

首次不要直接扫到几十 GiB。测试还有接收、发送、校验及 NCCL 内部缓冲，`-e` 不是整进程显存峰值。若显存不足，先查占用并减小最大尺寸。

正式稳态测量可以降低日志级别，但要在所有对照实验中保持一致。

</details>

## 6. 编译并运行自己的最小程序

```bash
# Linux/bash；从根目录执行，沿用 ROOT、CUDA_HOME、NCCL_HOME。
make -C nccl-tutorial/examples CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./nccl-tutorial/examples/build/single_process_allreduce 2
```

程序令 rank `r` 的每个元素等于 `r+1`，两卡求和应得到 3，四卡应得到 10。它验证每个元素，而不是只打印第一个元素就宣布正确。

读程序时先分成“准备每卡输入 → 提交 AllReduce → 等待完成 → 逐元素对照期望值”四段；这比一次追完全部 API 更容易对应本章开头的数据流。构建失败时不运行旧二进制，运行报错时先停在两卡用例。

详见[示例说明](examples/README.md)。它没有计时优化，不应拿来替代 nccl-tests 做吞吐结论。

## 7. 多进程与多机稍后再加

**选读：单进程两卡通过后再引入 MPI。**

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

## 自测

1. `nvidia-smi` 写 CUDA 13.x，`nvcc` 写 12.x，是一定装错了吗？
2. `NCCL_HOME` 正确，为什么还需要查 `ldd`？
3. 单卡测试通过，是否已证明 NVLink 或 RDMA 正常？
4. `git -C nccl rev-parse HEAD` 会切换版本或编译代码吗？

**答案：** 1. 不一定，两者含义不同，需检查具体兼容性。2. 构建时库搜索与运行时动态加载是两件事。3. 没有，单卡可能完全不经过跨设备链路。4. 不会，它只在指定源码目录读取当前提交号。

## 源码入口

**选读：先跑通，再按需要对照规则；不是首轮前置阅读。**

- [NCCL 构建说明](../nccl/README.md)：`nccl/README.md:15`，`src.build`、`BUILDDIR`。
- [版本信息](../nccl/makefiles/version.mk)：`nccl/makefiles/version.mk:9`。
- [编译配置](../nccl/makefiles/common.mk)：`nccl/makefiles/common.mk:8`，CUDA 路径；`:63`，架构集合。
- [测试构建](../nccl-tests/src/Makefile)：`nccl-tests/src/Makefile:24`，NCCL 头文件/库；`:29`，MPI 支持。

首次 GPU 验证接着运行[单进程示例](examples/README.md)，再核对[第 10 章的测试输出](10-nccl-tests.md)；逐行带练按需展开。机制学习回到[首页推荐顺序](README.md)，没有 GPU 也可继续算法手算和源码追踪；尚未理解两卡求和时先补[第 01 章](01-mental-model.md)。
