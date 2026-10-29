# 11｜验证与调试：沿最后完成的阶段缩小问题

上一章只增加了一条选择日志。本章的任务是：**怎样证明改动正确，以及出现问题时先查哪一层，而不是凭感觉修改配置。**

完整回归命令、日志文件管理和性能对照见 [测试与调试进阶附录](appendix/deep-dive/testing-and-debugging.md)。

## 1. 先找最后一个确定完成的阶段

一次 collective 可能经过：

```text
初始化/bootstrap
  → 连接准备
  → host 提交
  → CUDA launch
  → GPU 等待
  → NET 请求
  → 完成与回收
```

排查时先写：

```text
最后确定完成的是哪一步？
下一步由谁生产什么结果？
```

| 最后已知的证据 | 下一步优先检查 |
|---|---|
| 初始化返回或异步状态已就绪 | 首次算法连接、注册和 preconnect |
| task 已选择算法/协议 | plan 生成、连接完成和 CUDA launch |
| 已发生 launch | stream 依赖、peer 进度和 proxy 工作 |
| 网络 request 已完成 | flush、GPU 可见性、R_tail/R_head 和信用回收 |
| GPU 已完成 | 输出校验、异步状态和对象回收 |

这些阶段并行推进，日志不一定按表格顺序出现，重点是找最早的状态分歧。

## 2. 初始化成功，但第一次 collective 不动

先不要直接归因于 Ring kernel。

可能是：

```text
初始化已完成
  → 首次选中某算法
  → 该算法连接尚未建立
  → 需要 preconnect 或注册
```

所以应检查：

- 各 rank、GPU、uniqueId 和版本是否匹配；
- 首次选择是否发生；
- 对应连接 job 是否完成；
- transport 和具体 NIC/后端是否符合预期。

**初始化通过不代表所有算法连接都已经提前建立。**

## 3. 有 task 日志，但没有 kernel

`[tutorial-task]` 只证明 task 已获得执行方案。

接下来检查：

1. 是否结束了最外层 Group。
2. preconnect 是否完成。
3. plan 是否成功生成。
4. CUDA launch API 是否被调用以及返回值。
5. 同一 stream 是否有前序工作尚未结束。

如果 kernel 已出现但持续时间长，继续看它在等待谁：peer、credit、网络请求还是 stream 依赖。增加 CTA 无法补回尚未提交的工作。

## 4. 网络请求完成，GPU 仍然等待

普通 NET 接收方向有三个边界：

```text
网络 test 完成
  → 可见性处理完成，R_tail 发布
  → GPU 消费，R_head 发布
  → 接收槽复用
```

所以网络完成以后仍要检查：

- 是否需要 flush 以及 flush request 是否完成；
- recv proxy 是否发布接收可读进度；
- GPU 是否完成消费并归还信用；
- 对端是否因窗口占满而停止继续发送。

不要只记录“进入了 proxy 函数”，要记录 `(channel, step, request, done)` 的变化。

## 5. 两个状态查询接口分别回答什么？

| 查询 | 回答的问题 |
|---|---|
| `ncclCommGetAsyncError` 的返回值及输出状态 | NCCL 主机工作是否成功、是否仍在进行或是否出错 |
| `cudaStreamQuery` 或 event | 本 GPU 上已提交的 CUDA 工作是否完成 |

即使 `state == ncclSuccess`，输出是否可消费仍要看 CUDA stream/event。

出现错误后，应停止新增通信，通过带外控制面协调各 rank 的 abort 和退出，先交接 comm 的使用权，避免查询线程与清理线程竞争。

## 6. 用固定用例验证“只加日志，没有改变行为”

保持以下条件一致，只改变加载库：

```text
设备与 rank 映射
消息大小与 datatype/op
Ring/Simple 或其他强制对照
预热和计时次数
oop/ip、校验和日志级别
```

建议按下面顺序验收：

1. 基线库运行并保存正确性结果。
2. 观察库运行，确认 `[tutorial-task]` 的 count 和选择符合用例。
3. 两组 `#wrong=0`，错误摘要和退出状态正常。
4. 对比两个版本的已有 tuning 选择是否一致。
5. 将日志级别降回 WARN，再比较重复运行表现。

如果新日志缺失，按这个顺序查：

```text
库路径 → DEBUG 子系统 → 运行时分支 → 观察点是否执行
```

## 7. 回归范围按改动类型扩大

| 改动 | 优先增加的用例 |
|---|---|
| 只加日志 | 固定小用例和主场景即可 |
| 改选择阈值 | 阈值前后各一个元素，并加入自动选择 |
| 改 device 分块 | 尾块、多个 count、datatype/op、oop/ip |
| 改 transport | 单/多进程、目标 peer、建连、完成和释放 |
| 改拓扑/图 | 不同 GPU 对、同机与跨机、节点内与全局邻接 |

每次所有 rank 使用匹配的操作、count、datatype 和调用顺序。只覆盖自己实际验证过的范围，不把“跑通一次”写成“全部路径正确”。

## 8. 本章自测

1. 为什么 task 日志存在还不能证明 kernel 已启动？
2. 初始化成功后为什么还要检查首次 preconnect？
3. 网络 request 完成后，接收槽为什么不能立即复用？
4. `ncclCommGetAsyncError` 与 `cudaStreamQuery` 分别证明什么？
5. 一次回归应该固定哪些变量，才能说明只改变了目标行为？

至此，`00–11` 的主线形成闭环：理解职责，找到对象，观察交接，再用分层证据验收。

需要完整 baseline/observed 脚本、日志文件管理、回归矩阵和性能对照时，再读 [测试与调试进阶附录](appendix/deep-dive/testing-and-debugging.md)。
