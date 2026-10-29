# 附录：环境准备——构建并运行自己读的那份 NCCL

源码实验的第一步是建立版本、构建产物和运行程序之间的对应关系。后续修改才能确定进入了实际执行路径。

GPU 部分使用 Linux、NVIDIA GPU、可用驱动和 CUDA Toolkit。以下命令从同时包含 `nccl/`、`nccl-tests/`、`nccl-tutorial/` 的项目根目录执行，GPU 使用当前作业获分配的设备。

## 1. 记录源码与设备

```bash
git -C nccl rev-parse HEAD
git -C nccl-tests rev-parse HEAD
nvidia-smi
nvidia-smi topo -m
nvcc --version
```

前两项标识代码版本；`nvidia-smi` 查看驱动和设备；`topo -m` 提供 GPU、CPU、NIC 的距离信息；`nvcc` 来自用于构建的 Toolkit。

`nvidia-smi` 显示的 CUDA 版本表示驱动支持能力，Toolkit 版本看 `nvcc`。基线提交见[首页](../README.md)。

## 2. 构建本地库与测试

设置路径，`CUDA_HOME` 换成实际 Toolkit 位置。`JOBS` 控制编译并行度。

```bash
export ROOT="$PWD"
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME="$ROOT/nccl/build"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

make -C nccl -j"${JOBS:-4}" src.build CUDA_HOME="$CUDA_HOME"
make -C nccl-tests -j"${JOBS:-4}" \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
make -C nccl-tutorial/examples \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
```

产物分别位于：

- `nccl/build/lib/`：NCCL 动态库。
- `nccl-tests/build/`：集合通信测试程序。
- `nccl-tutorial/examples/build/`：教程的单进程示例。

构建 GPU kernel 可能较久。只面向一类 GPU 开发时，可以按 [nccl/README.md:30](../../nccl/README.md#L30) 的 `NVCC_GENCODE` 配置缩小目标架构集合，取值与测试 GPU 的 compute capability 对应。

## 3. 确认运行时加载的库

```bash
ldd nccl-tests/build/all_reduce_perf
ldd nccl-tutorial/examples/build/single_process_allreduce
```

找到 `libnccl.so`，路径应指向本项目 `nccl/build/lib/`；CUDA 依赖也应成功解析。头文件搜索路径解决编译时声明，`LD_LIBRARY_PATH` 解决运行时库查找，两项都要指向预期版本。

之后每次修改 NCCL，先重新构建库，再用同样的环境运行测试。若改了公共头文件，也重新构建调用程序。

## 4. 先正确性，再性能

有两张可见 GPU 时：

```bash
./nccl-tutorial/examples/build/single_process_allreduce 2
./nccl-tests/build/all_reduce_perf \
  -b 64K -e 64K -g 2 -t 1 -d float -o sum -w 5 -n 20 -c 1 -a 3
```

第一条只做一次小规模通信，输入分别全为 1、2，两卡结果应全为 3，所有 `wrong=0`，最终 `PASS`。

第二条固定每 rank 64 KiB，便于重复对照。检查 out-of-place、in-place 两组输出的错误数，再看耗时。参数与列含义见 [nccl-tests 附录](nccl-tests.md)。

只有一张卡时，可把两条命令的设备数改为 1，验证构建、加载和本地执行；卡间路径留待多卡测试。

## 5. 跨进程时加入 MPI

MPI 示例由 MPI 启动器创建进程，交换 NCCL 初始化 ID，每个进程选自己的 GPU。数据规约仍调用 NCCL。

```bash
make -C nccl-tutorial/examples mpi \
  CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" MPICXX=mpicxx
mpirun -np 2 ./nccl-tutorial/examples/build/mpi_allreduce
```

这份示例采用独占整卡映射：进程只看到一张卡时选 device 0；看到同一完整设备列表时按 local rank 选卡。程序会检查同主机 PCI device 重复选择。

多机启动沿用集群的 MPI/调度器配置，把相同程序、库路径和环境传到各节点。具体引导流程见[示例说明](../examples/README.md)。

## 6. 没有 GPU 时做什么

```bash
python3 nccl-tutorial/examples/ring_simulator.py --ranks 4 --chunk-size 2
python3 nccl-tutorial/examples/ring_simulator.py --self-test
make -n -C nccl-tutorial/examples
make -n -C nccl-tutorial/examples mpi
```

Python 程序验证 Ring 的块传递和求和结果；`make -n` 展示构建命令。GPU 编译、连接、协议和性能仍需在 Linux CUDA 环境验证。

本次教材整理环境为 macOS，未执行上面的 GPU/MPI 实验。

回到[教程首页](../README.md)，或从 [00｜初学者主线](../00-mainline.md) 开始阅读。
