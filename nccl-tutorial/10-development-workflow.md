# 10｜开发工作流：用一次小改动建立开发闭环

前九章已经建立软件栈地图。现在不再继续加新概念，而是完成一个最小开发任务：**给普通 AllReduce 的任务选择结果增加一条日志，并证明它没有改变通信行为。**

完整命令、字段类型和构建细节见 [开发工作流进阶附录](appendix/development-workflow.md)。

## 1. 先判断需求属于哪一层

不要从“报错出现在哪个文件”决定改哪里，而要从“谁决定这个行为”入手。

| 需求 | 优先查看的层 | 最小验证 |
|---|---|---|
| API 参数或检查 | `nccl.h.in`、`collectives.cc` | 合法/非法输入、返回值和兼容性 |
| 算法选择或任务合并 | `enqueue/`、`tuning/` | 阈值两侧、单/多 task、自动选择 |
| GPU/NIC 路径或 Ring 构造 | `graph/` | 拓扑、图和全局邻接是否正确 |
| device 算法或 protocol | `device/`、生成器 | 尾块、datatype/op、oop/ip、协议完成 |
| transport 或 net plugin | `transport/`、`plugin/` | 建连、请求完成、释放和跨进程行为 |

本章只观察选择结果，所以落在 `enqueue/` 的任务准备路径，不需要改 Ring kernel。

## 2. 固定三种身份：源码、构建产物、运行库

在修改前记录：

```text
源码提交与本地补丁
构建目录和编译参数
运行时实际加载的 libnccl.so 路径
```

保留未修改的基线库和 tests，观察版本使用另一个构建目录，例如：

```text
nccl/build          基线
nccl/build-observe  观察版本
```

运行时用 `ldd` 和测试启动报告核对：

- `libnccl.so` 是否来自预期目录；
- 头文件版本与运行时版本是否匹配；
- 多机时每个节点是否使用同一份预期库。

“源码已修改”并不等于“运行程序已经加载了新代码”。

## 3. 选择观察点：在结果写回原 task 之后

本章要记录的是**原始 task 的选择结果**，而不是聚合任务的 count。

默认路径中：

```text
ncclGetAlgoInfo 得到调优结果
  → ncclPrepareTasks 将结果写回每个原始 task
  → 原始 task 进入后续调度队列
```

适合在写回之后、入队之前插入一条日志，观察：

```text
rank、count、algorithm、protocol、nMaxChannels、nWarps
```

示意：

```cpp
INFO(NCCL_TUNING,
     "[tutorial-task] rank=%d count=%zu algo=%s proto=%s maxChannels=%d warps=%d",
     comm->rank, aggBeg->count,
     ncclAlgoToString(aggBeg->algorithm),
     ncclProtoToString(aggBeg->protocol),
     int(aggBeg->nMaxChannels), int(aggBeg->nWarps));
```

这条日志只证明：

> task 已经得到算法、协议和资源预算。

它不证明最终 channel 分工完成，也不证明 kernel 已启动或 GPU 已完成。

## 4. 建立“修改—构建—加载—观察”闭环

按顺序执行：

1. 在未修改源码上构建并保存基线。
2. 只在观察副本中增加日志，构建到 `build-observe`。
3. 用 `ldd` 确认测试加载观察库。
4. 使用固定的合法小用例，保持设备、消息大小、datatype/op、预热和迭代次数不变。
5. 检查日志是否出现，并核对 count 和选择结果。
6. 检查 oop/ip 两组 `#wrong=0`、错误摘要和正常退出。
7. 降低日志级别后，再与基线比较重复运行表现。

例如测试环境可以固定：

```bash
NCCL_ENQUEUE_REARCH_ENABLE=0
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,TUNING
```

64 KiB FP32 对应 `count=16384`。这只是用于确认观察点，不是生产环境的算法推荐。

## 5. 修改 device 代码时，还要看生成链

设备模板只有实例化后才成为可执行代码：

```text
device 模板/生成器
  → 生成 .cu、host_table、device_table 和 rules.mk
  → nvcc 编译与 device link
  → 主库链接
```

因此：

- 改模板可能触发多个实例重编；
- 改组合列表会改变生成表；
- 改生成选项最好使用新的构建目录；
- kernel 名称不足以确认实际 work 的算法，仍要看 `funcId`。

新增算法则要同时检查选择链、分派链、实现链和验证链，不能只增加一个枚举。

## 6. 本章的验收标准

完成以下证据，才算建立第一次开发闭环：

```text
基线库与观察库身份已记录
观察点确实位于原始 task 写回之后
日志中的 count 和算法/协议与用例一致
原有通信结果和完成行为未改变
基线与观察版本的回归范围已记录
```

下一步读 [11｜验证与调试](11-testing-and-debugging.md)，用分层证据判断问题发生在初始化、连接、提交、GPU、网络还是回收阶段。

需要完整构建命令、观察日志位置、nccl-tests 参数和回归矩阵时，再读 [开发工作流进阶附录](appendix/development-workflow.md)。
